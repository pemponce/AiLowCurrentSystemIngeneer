import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision

from app.ml.resplan_dataset import ResPlanRasterDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int):
    # Совместимо с разными версиями torchvision
    try:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )
    except TypeError:
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        n += 1

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()

        out = model(x)["out"]
        loss = F.cross_entropy(out, y)

        total += float(loss.item())
        n += 1

    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root with images/ and masks/")
    ap.add_argument("--out", required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=5, help="Override number of classes (default 5)")
    args = ap.parse_args()

    if args.batch < 2:
        raise SystemExit(
            "BatchNorm в DeepLabV3 может падать при batch_size=1 (особенно на 1x1 фичах). "
            "Поставь --batch 2 или больше."
        )

    set_seed(args.seed)

    data_root = Path(args.data)
    if not data_root.exists():
        raise SystemExit(f"Dataset not found: {data_root}")

    ds = ResPlanRasterDataset(str(data_root), augment=True)
    n_total = len(ds)
    if n_total < 2:
        raise SystemExit(f"Dataset too small: len={n_total}. Проверь, что images/ и masks/ содержат пары с одинаковыми именами.")

    # Более предсказуемый split: floor вместо round
    val_split = max(0.0, min(0.9, float(args.val_split)))
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    # Перемешиваем индексы
    idx = np.arange(n_total)
    np.random.shuffle(idx)

    # Если из-за split получается train с хвостом 1 на batch=2, подправим:
    # Это один из типовых источников ошибки BatchNorm (последний batch=1). :contentReference[oaicite:2]{index=2}
    if n_val > 0 and (n_train % args.batch) == 1:
        n_train -= 1
        n_val += 1

    if n_train < 2:
        raise SystemExit(f"After split train={n_train}, val={n_val}. Уменьши --val-split или увеличь датасет.")

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if n_val > 0 else None

    # Ключевой фикс: drop_last=True, чтобы в train никогда не было batch=1. :contentReference[oaicite:3]{index=3}
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # num_classes: либо из датасета, либо аргументом
    num_classes = getattr(ds, "num_classes", None)
    if num_classes is None:
        num_classes = int(args.num_classes)

    model = build_model(int(num_classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device)

        if val_loader is not None:
            vl = eval_one_epoch(model, val_loader, device)
        else:
            vl = tr

        print(f"epoch={epoch}/{args.epochs} train_loss={tr:.4f} val_loss={vl:.4f}")

        if best_val is None or vl < best_val:
            best_val = vl
            ckpt = {
                "model": model.state_dict(),
                "num_classes": int(num_classes),
                "epoch": int(epoch),
                "val_loss": float(vl),
            }
            torch.save(ckpt, str(out_path))
            print(f"saved best -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
