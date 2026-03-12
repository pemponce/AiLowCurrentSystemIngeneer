"""
app/nn2/train.py — обучение NN-2 v2 (BiLSTM+CRF NER)

Запуск:
  python -m app.nn2.dataset_gen --out data/nn2_dataset --count 10000
  python -m app.nn2.train --data data/nn2_dataset --out models/nn2 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.nn2.model import NERDataset, NERModel, TagVocab, WordVocab, MAX_SEQ


def collate(batch):
    xs, ys, lengths = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.tensor(lengths)


def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            loss  = model.loss(x, y, lengths)
            preds = model.predict(x, lengths)
            total_loss += loss.item()
            n += 1

            for b, pred_seq in enumerate(preds):
                L     = lengths[b].item()
                gold  = y[b, :L].tolist()
                for p, g in zip(pred_seq, gold):
                    if g != 0:            # 0 = O тег
                        if p == g:
                            tp += 1
                        else:
                            fn += 1
                    elif p != 0:
                        fp += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "val_loss":  total_loss / max(n, 1),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    os.makedirs(args.out, exist_ok=True)

    # ── словари ───────────────────────────────────────────────────
    tag_path  = os.path.join(args.data, "tags.json")
    with open(tag_path, encoding="utf-8") as f:
        tag_list = json.load(f)

    tag_vocab  = TagVocab()
    tag_vocab.build(tag_list)
    tag_vocab.save(os.path.join(args.out, "tag_vocab.json"))

    word_vocab = WordVocab()
    all_samples: List[Dict] = []
    for split in ("train.jsonl", "val.jsonl"):
        path = os.path.join(args.data, split)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))
    word_vocab.build(all_samples)
    word_vocab.save(os.path.join(args.out, "word_vocab.json"))
    print(f"Слов: {word_vocab.size} | Тегов: {tag_vocab.size}")

    # ── датасеты ──────────────────────────────────────────────────
    train_ds = NERDataset(os.path.join(args.data, "train.jsonl"), word_vocab, tag_vocab)
    val_ds   = NERDataset(os.path.join(args.data, "val.jsonl"),   word_vocab, tag_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── модель ────────────────────────────────────────────────────
    model = NERModel(vocab_size=word_vocab.size, num_tags=tag_vocab.size,
                     emb_dim=64, hidden_dim=128, n_layers=2, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1   = 0.0
    best_ckpt = os.path.join(args.out, "nn2_best.pt")
    history   = []

    print(f"\nОбучение {args.epochs} эпох...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y, lengths)
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
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss / len(train_loader), **metrics})

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "vocab_size": word_vocab.size, "num_tags": tag_vocab.size,
                "f1": best_f1,
            }, best_ckpt)
            print(f"  ✓ Лучшая модель (F1={best_f1:.3f})")

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nГотово! Лучшая F1={best_f1:.3f} → {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/nn2_dataset")
    parser.add_argument("--out",    default="models/nn2")
    parser.add_argument("--epochs", type=int,   default=40)
    parser.add_argument("--batch",  type=int,   default=128)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()
    train(args)