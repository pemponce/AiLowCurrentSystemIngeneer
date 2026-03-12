"""
app/nn3/train.py — обучение NN-3 (GNN размещение устройств)

Запуск:
  python -m app.nn3.dataset_gen --out data/nn3_dataset --count 5000
  python -m app.nn3.train --data data/nn3_dataset --out models/nn3 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.nn3.model import PlanDataset, PlacementGNN, MAX_DEVICE
from app.nn3.dataset_gen import DEVICES


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    correct    = 0
    total      = 0

    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for feats, adj, labels, lengths in loader:
            feats  = feats.to(device)
            adj    = adj.to(device)
            labels = labels.to(device)

            logits = model(feats, adj)   # (B, N, D, C)
            B, N, D, C = logits.shape

            loss = torch.tensor(0.0, device=device)
            for b in range(B):
                n_rooms = lengths[b].item()
                for i in range(n_rooms):
                    for j in range(D):
                        loss = loss + ce(
                            logits[b, i, j].unsqueeze(0),
                            labels[b, i, j].unsqueeze(0),
                        )
            loss = loss / max(B, 1)
            total_loss += loss.item()
            n_batches  += 1

            # Точность
            preds = logits.argmax(dim=-1)   # (B, N, D)
            for b in range(B):
                n_rooms = lengths[b].item()
                p = preds[b, :n_rooms]
                g = labels[b, :n_rooms]
                correct += (p == g).sum().item()
                total   += p.numel()

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_acc":  correct / max(total, 1),
    }


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    os.makedirs(args.out, exist_ok=True)

    # ── датасеты ──────────────────────────────────────────────────
    train_ds = PlanDataset(os.path.join(args.data, "train.jsonl"))
    val_ds   = PlanDataset(os.path.join(args.data, "val.jsonl"))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── модель ────────────────────────────────────────────────────
    model = PlacementGNN(hidden=64, out_dim=128, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce        = nn.CrossEntropyLoss()

    best_acc  = 0.0
    best_ckpt = os.path.join(args.out, "nn3_best.pt")
    history   = []

    print(f"\nОбучение {args.epochs} эпох...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for feats, adj, labels, lengths in train_loader:
            feats  = feats.to(device)
            adj    = adj.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(feats, adj)   # (B, N, D, C)
            B = feats.size(0)

            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for b in range(B):
                n_rooms = lengths[b].item()
                for i in range(n_rooms):
                    for j in range(len(DEVICES)):
                        loss = loss + ce(
                            logits[b, i, j].unsqueeze(0),
                            labels[b, i, j].unsqueeze(0),
                        )
            loss = loss / max(B, 1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss/len(train_loader):.4f} | "
              f"val_loss={metrics['val_loss']:.4f} | "
              f"val_acc={metrics['val_acc']:.3f} | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss / len(train_loader), **metrics})

        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     best_acc,
            }, best_ckpt)
            print(f"  ✓ Лучшая модель (val_acc={best_acc:.3f})")

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nГотово! Лучшая val_acc={best_acc:.3f} → {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/nn3_dataset")
    parser.add_argument("--out",    default="models/nn3")
    parser.add_argument("--epochs", type=int,   default=50)
    parser.add_argument("--batch",  type=int,   default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()
    train(args)