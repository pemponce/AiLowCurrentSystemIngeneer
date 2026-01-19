import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

from app.ml.resplan_dataset import ResPlanRasterDataset


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def compute_pixel_hist(dataset_root: str, num_classes: int) -> np.ndarray:
    masks_dir = os.path.join(dataset_root, "masks")
    hs = np.zeros(num_classes, dtype=np.int64)
    for fn in os.listdir(masks_dir):
        if not fn.lower().endswith(".png"):
            continue
        if "_overlay" in fn.lower():
            continue
        import cv2
        m = cv2.imread(os.path.join(masks_dir, fn), 0)
        if m is None:
            continue
        b = np.bincount(m.reshape(-1), minlength=num_classes)
        hs += b[:num_classes]
    return hs


def make_class_weights_from_hist(hist: np.ndarray) -> torch.Tensor:
    """
    Robust weights for heavy imbalance.
    We use inverse sqrt frequency with smoothing to avoid insane weights.
    """
    hist = hist.astype(np.float64)
    total = float(hist.sum() + 1e-9)
    freq = hist / total

    # Avoid div by zero
    freq = np.clip(freq, 1e-9, 1.0)

    inv = 1.0 / np.sqrt(freq)  # milder than 1/freq
    inv = inv / inv.mean()     # normalize

    # Cap extreme weights
    inv = np.clip(inv, 0.2, 10.0)
    return torch.tensor(inv, dtype=torch.float32)


def dice_loss_multiclass(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice over classes (excluding bg=0).
    logits: (N,C,H,W)
    target: (N,H,W) int64
    """
    probs = torch.softmax(logits, dim=1)

    # one-hot target
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
    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        out = model(x)["out"]
        loss = ce(out, y)
        if cfg.dice_weight > 0:
            loss = loss + cfg.dice_weight * dice_loss_multiclass(out, y, cfg.num_classes)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, ce: nn.Module, cfg: TrainCfg, device: torch.device) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = ce(out, y)
        if cfg.dice_weight > 0:
            loss = loss + cfg.dice_weight * dice_loss_multiclass(out, y, cfg.num_classes)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
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
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ResPlanRasterDataset(cfg.data, augment=True)
    n = len(ds)
    n_val = max(1, int(n * cfg.val_split))
    n_tr = max(1, n - n_val)
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, drop_last=cfg.drop_last)
    va_loader = DataLoader(va_ds, batch_size=max(1, cfg.batch), shuffle=False, num_workers=cfg.num_workers, drop_last=False)

    model = build_model(cfg.num_classes).to(device)

    if cfg.auto_class_weights:
        hist = compute_pixel_hist(cfg.data, cfg.num_classes)
        w = make_class_weights_from_hist(hist).to(device)
        print("class_weights:", w.detach().cpu().numpy().round(4).tolist())
        ce = nn.CrossEntropyLoss(weight=w)
    else:
        ce = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best = 1e18
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)

    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, tr_loader, opt, ce, cfg, device)
        va = eval_one_epoch(model, va_loader, ce, cfg, device)
        print(f"epoch={ep}/{cfg.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), cfg.out)
            print(f"saved best -> {cfg.out}")

    print("done.")


if __name__ == "__main__":
    main()
