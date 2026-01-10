from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from app.ml.model_structure import build_structure_model
from app.ml.resplan_dataset import ResPlanRasterDataset


@dataclass
class TrainCfg:
    epochs: int = 10
    batch: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_split: float = 0.1
    seed: int = 42
    num_classes: int = 5  # bg, wall, door, window, front_door
    save_best_only: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)["out"]
        loss = F.cross_entropy(out, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def train_one_epoch(model, loader, opt, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        out = model(x)["out"]
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Папка resplan_raster (images/ + masks/)")
    ap.add_argument("--out", default="models/structure_resplan.pt", help="Куда сохранить веса")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg = TrainCfg(
        epochs=int(args.epochs),
        batch=int(args.batch),
        lr=float(args.lr),
        val_split=float(args.val_split),
        num_workers=int(args.num_workers),
    )

    set_seed(cfg.seed)

    ds = ResPlanRasterDataset(args.data, augment=True)
    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_structure_model(cfg.num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best = 1e18
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device)
        vl = evaluate(model, val_loader, device)
        print(f"epoch={epoch}/{cfg.epochs} train_loss={tr:.4f} val_loss={vl:.4f}")

        if cfg.save_best_only:
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "num_classes": cfg.num_classes}, args.out)
                print(f"saved best -> {args.out}")
        else:
            torch.save({"model": model.state_dict(), "num_classes": cfg.num_classes}, args.out)
            print(f"saved -> {args.out}")

    print("done.")


if __name__ == "__main__":
    main()
