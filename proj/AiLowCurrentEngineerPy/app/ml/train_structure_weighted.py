# app/ml/train_structure_weighted.py
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from app.ml.resplan_dataset import ResPlanRasterDataset

try:
    from torchvision.models.segmentation import deeplabv3_resnet50
except Exception as e:
    raise RuntimeError("torchvision is required for training DeepLabV3") from e


CLASSES = ["bg", "wall", "door", "window", "front_door"]
NUM_CLASSES = 5


def count_pixels_in_masks(masks_dir: Path, num_classes: int) -> np.ndarray:
    hs = np.zeros(num_classes, dtype=np.int64)
    files = [p for p in masks_dir.iterdir() if p.suffix.lower() == ".png" and "_overlay" not in p.name.lower()]
    for p in files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        binc = np.bincount(m.reshape(-1), minlength=num_classes)
        hs += binc.astype(np.int64)
    return hs


def make_class_weights_from_hist(hist: np.ndarray, eps: float = 1.0) -> torch.Tensor:
    """
    Median frequency balancing (simple and robust).
    """
    h = hist.astype(np.float64) + eps
    freq = h / h.sum()
    med = np.median(freq)
    w = med / freq
    # Don't let background dominate
    w[0] = min(w[0], 1.0)
    # clip extreme
    w = np.clip(w, 0.2, 50.0)
    return torch.tensor(w, dtype=torch.float32)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_bg: bool = True) -> torch.Tensor:
    """
    Multi-class soft Dice. logits: [B,C,H,W], target: [B,H,W]
    """
    probs = torch.softmax(logits, dim=1)
    tgt = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    if ignore_bg:
        probs = probs[:, 1:, :, :]
        tgt = tgt[:, 1:, :, :]

    dims = (0, 2, 3)
    inter = torch.sum(probs * tgt, dims)
    denom = torch.sum(probs + tgt, dims)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def build_model(num_classes: int) -> nn.Module:
    # New torchvision API prefers weights=None
    try:
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
    except TypeError:
        # Older API
        model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    return model


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, ce: nn.Module, device: torch.device, dice_w: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)["out"]
        loss = ce(out, y)
        if dice_w > 0.0:
            loss = loss + dice_w * dice_loss(out, y, NUM_CLASSES, ignore_bg=True)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, ce: nn.Module,
                    device: torch.device, dice_w: float) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = ce(out, y)
        if dice_w > 0.0:
            loss = loss + dice_w * dice_loss(out, y, NUM_CLASSES, ignore_bg=True)
        loss.backward()
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use-auto-weights", action="store_true", help="Compute class weights from masks and use them in CE")
    ap.add_argument("--dice", type=float, default=0.5, help="Dice loss weight (0 disables). Recommended 0.3..1.0 for imbalance")
    ap.add_argument("--drop-last", action="store_true", help="Drop last incomplete batch (prevents BatchNorm batch=1 crashes)")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data)
    masks_dir = data_root / "masks"
    if not masks_dir.exists():
        raise RuntimeError(f"masks dir not found: {masks_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    full = ResPlanRasterDataset(str(data_root), augment=True)
    n_total = len(full)
    n_val = max(1, int(round(n_total * float(args.val_split))))
    n_train = max(1, n_total - n_val)

    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=bool(args.drop_last)
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch), shuffle=False, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=False
    )

    # model
    model = build_model(NUM_CLASSES).to(device)

    # loss
    weights_t: Optional[torch.Tensor] = None
    if args.use_auto_weights:
        hist = count_pixels_in_masks(masks_dir, NUM_CLASSES)
        weights_t = make_class_weights_from_hist(hist).to(device)
        share = (hist / max(1, hist.sum())).round(6)
        print("mask pixels:", hist.tolist())
        print("mask share :", share.tolist())
        print("ce weights :", weights_t.detach().cpu().numpy().round(4).tolist())

    ce = nn.CrossEntropyLoss(weight=weights_t)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    best = 1e9
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        tr = train_one_epoch(model, train_loader, opt, ce, device, float(args.dice))
        va = eval_one_epoch(model, val_loader, ce, device, float(args.dice))
        print(f"epoch={epoch}/{args.epochs} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), str(out_path))
            print(f"saved best -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
