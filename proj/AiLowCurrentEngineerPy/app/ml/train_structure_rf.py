import argparse
import hashlib
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torchvision.models.segmentation import deeplabv3_resnet50

from app.ml.resplan_dataset import ResPlanRasterDataset


CLASS_NAMES = ["bg", "wall", "door", "window", "front_door"]


@dataclass
class TrainCfg:
    data: str
    out: str
    epochs: int
    batch: int
    lr: float
    val_split: float
    num_workers: int
    seed: int
    num_classes: int
    auto_class_weights: bool
    dice_weight: float
    drop_last: bool
    hist_cache: str
    hist_cache_force: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    # no pretrained weights, since your domain is CAD-like
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def _mask_files(masks_dir: str) -> List[str]:
    fs = []
    for fn in os.listdir(masks_dir):
        l = fn.lower()
        if not l.endswith(".png"):
            continue
        if "_overlay" in l:
            continue
        fs.append(fn)
    fs.sort()
    return fs


def _signature_for_masks(masks_dir: str, files: List[str]) -> str:
    """
    Fast-ish signature: file name + size + mtime.
    This avoids reading all pixels to validate cache.
    """
    h = hashlib.sha1()
    for fn in files:
        p = os.path.join(masks_dir, fn)
        st = os.stat(p)
        s = f"{fn}|{st.st_size}|{int(st.st_mtime)}\n".encode("utf-8", errors="ignore")
        h.update(s)
    return h.hexdigest()


def _default_hist_cache_path(dataset_root: str, num_classes: int) -> str:
    return os.path.join(dataset_root, f"pixel_hist_{num_classes}.npz")


def compute_pixel_hist_cached(dataset_root: str, num_classes: int, cache_path: str, force: bool) -> np.ndarray:
    masks_dir = os.path.join(dataset_root, "masks")
    if not os.path.isdir(masks_dir):
        raise RuntimeError(f"Missing masks dir: {masks_dir}")

    files = _mask_files(masks_dir)
    if not files:
        raise RuntimeError(f"No mask png files (excluding overlays) in: {masks_dir}")

    if not cache_path:
        cache_path = _default_hist_cache_path(dataset_root, num_classes)

    sig = _signature_for_masks(masks_dir, files)

    if (not force) and os.path.exists(cache_path):
        try:
            npz = np.load(cache_path, allow_pickle=True)
            cached_sig = str(npz["signature"].item())
            cached_k = int(npz["num_classes"].item())
            cached_hist = npz["hist"].astype(np.int64)
            npz.close()
            if cached_sig == sig and cached_k == num_classes and cached_hist.shape[0] == num_classes:
                print(f"[hist] cache hit: {cache_path}")
                return cached_hist
            else:
                print("[hist] cache mismatch -> recompute")
        except Exception as e:
            print(f"[hist] cache read failed ({e}) -> recompute")

    # Recompute with progress
    import cv2

    hs = np.zeros(num_classes, dtype=np.int64)
    pbar = tqdm(files, desc="[hist] scanning masks", dynamic_ncols=True)
    for fn in pbar:
        m = cv2.imread(os.path.join(masks_dir, fn), 0)
        if m is None:
            continue
        b = np.bincount(m.reshape(-1), minlength=num_classes)
        hs += b[:num_classes]
        # small live info
        pbar.set_postfix(bg=int(hs[0]), wall=int(hs[1]))
    pbar.close()

    # Save cache
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, hist=hs, num_classes=np.int64(num_classes), signature=np.array(sig))
        print(f"[hist] cache saved: {cache_path}")
    except Exception as e:
        print(f"[hist] cache save failed ({e})")

    return hs


def make_class_weights_from_hist(hist: np.ndarray) -> torch.Tensor:
    """
    Robust weights for heavy imbalance.
    IMPORTANT:
      - classes with hist==0 must NOT affect normalization (front_door may be absent).
    """
    hist = hist.astype(np.float64)
    w = np.zeros_like(hist, dtype=np.float64)

    nonzero = hist > 0
    if nonzero.sum() == 0:
        return torch.ones_like(torch.tensor(hist, dtype=torch.float32))

    total = float(hist[nonzero].sum() + 1e-9)
    freq = hist[nonzero] / total
    freq = np.clip(freq, 1e-9, 1.0)

    inv = 1.0 / np.sqrt(freq)      # milder than 1/freq
    inv = inv / float(inv.mean())  # normalize only over non-zero classes
    inv = np.clip(inv, 0.2, 10.0)  # cap

    w[nonzero] = inv
    w[~nonzero] = 0.0              # absent class => ignore
    return torch.tensor(w, dtype=torch.float32)


def dice_loss_multiclass(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Soft Dice over classes (excluding bg=0).
    logits: (N,C,H,W)
    target: (N,H,W) int64
    """
    probs = torch.softmax(logits, dim=1)

    t = torch.zeros_like(probs)
    t.scatter_(1, target.unsqueeze(1), 1.0)

    dices: List[torch.Tensor] = []
    for c in range(1, num_classes):  # exclude bg
        pc = probs[:, c, :, :]
        tc = t[:, c, :, :]
        inter = (pc * tc).sum(dim=(1, 2))
        denom = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2)) + eps
        d = 1.0 - (2.0 * inter + eps) / denom
        dices.append(d.mean())

    if not dices:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(dices).mean()


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, ce: nn.Module, cfg: TrainCfg, device: torch.device) -> float:
    model.eval()
    losses = []

    pbar = tqdm(loader, desc="eval", dynamic_ncols=True, leave=False, total=len(loader))
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        out = model(x)["out"]
        loss = ce(out, y)
        if cfg.dice_weight > 0:
            loss = loss + cfg.dice_weight * dice_loss_multiclass(out, y, cfg.num_classes)

        lv = float(loss.item())
        losses.append(lv)
        pbar.set_postfix(loss=f"{lv:.4f}")
    pbar.close()

    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, ce: nn.Module, cfg: TrainCfg, device: torch.device) -> float:
    model.train()
    losses = []

    pbar = tqdm(loader, desc="train", dynamic_ncols=True, leave=False, total=len(loader))
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = ce(out, y)
        if cfg.dice_weight > 0:
            loss = loss + cfg.dice_weight * dice_loss_multiclass(out, y, cfg.num_classes)
        loss.backward()
        opt.step()

        lv = float(loss.item())
        losses.append(lv)
        pbar.set_postfix(loss=f"{lv:.4f}")
    pbar.close()

    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=5)
    ap.add_argument("--auto-class-weights", action="store_true")
    ap.add_argument("--dice-weight", type=float, default=0.35)
    ap.add_argument("--drop-last", action="store_true", help="avoid BN crash when last batch has size 1")

    ap.add_argument("--hist-cache", default="", help="Path to pixel_hist cache .npz (default: <data>/pixel_hist_K.npz)")
    ap.add_argument("--hist-cache-force", action="store_true", help="Force recompute pixel_hist ignoring cache")
    args = ap.parse_args()

    cfg = TrainCfg(
        data=args.data,
        out=args.out,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        num_classes=args.num_classes,
        auto_class_weights=bool(args.auto_class_weights),
        dice_weight=float(args.dice_weight),
        drop_last=bool(args.drop_last),
        hist_cache=str(args.hist_cache or ""),
        hist_cache_force=bool(args.hist_cache_force),
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | cuda_available={torch.cuda.is_available()}")

    # Dataset
    print("[data] loading dataset...")
    ds = ResPlanRasterDataset(cfg.data, augment=True)
    n = len(ds)
    print(f"[data] samples={n}")

    n_val = max(1, int(n * cfg.val_split))
    n_tr = max(1, n - n_val)
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    print(f"[split] train={len(tr_ds)} val={len(va_ds)}")

    # DataLoader speed knobs (only helps when num_workers>0, and pin_memory helps with GPU transfer)
    pin_memory = bool(device.type == "cuda")
    persistent_workers = bool(cfg.num_workers > 0)
    prefetch_factor = 2 if cfg.num_workers > 0 else None

    dl_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.batch,
        shuffle=True,
        drop_last=cfg.drop_last,
        **dl_kwargs,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=max(1, cfg.batch),
        shuffle=False,
        drop_last=False,
        **dl_kwargs,
    )

    # Model
    print("[model] building...")
    model = build_model(cfg.num_classes).to(device)

    # Loss
    if cfg.auto_class_weights:
        print("[loss] computing pixel_hist / class_weights...")
        t0 = time.time()
        hist = compute_pixel_hist_cached(cfg.data, cfg.num_classes, cfg.hist_cache, cfg.hist_cache_force)
        w = make_class_weights_from_hist(hist).to(device)
        dt = time.time() - t0
        print("pixel_hist:", hist.tolist())
        print("class_weights:", w.detach().cpu().numpy().round(4).tolist())
        print(f"[loss] done in {dt:.1f}s")
        ce = nn.CrossEntropyLoss(weight=w)
    else:
        ce = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best = 1e18
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)

    for ep in range(1, cfg.epochs + 1):
        print(f"\n=== epoch {ep}/{cfg.epochs} ===")
        tr = train_one_epoch(model, tr_loader, opt, ce, cfg, device)
        va = eval_one_epoch(model, va_loader, ce, cfg, device)
        print(f"epoch={ep}/{cfg.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best:
            best = va
            ckpt = {
                "model": model.state_dict(),
                "num_classes": cfg.num_classes,
                "class_names": CLASS_NAMES[: cfg.num_classes],
            }
            torch.save(ckpt, cfg.out)
            print(f"saved best -> {cfg.out}")

    print("done.")


if __name__ == "__main__":
    main()
