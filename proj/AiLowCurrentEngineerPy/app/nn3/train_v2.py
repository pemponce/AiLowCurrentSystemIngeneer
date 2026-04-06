# app/nn3/train_v2.py
"""
Training script для PlacementNetV2

Обучает модель на датасете с ground truth координатами.

Метрики:
- count_acc: точность предсказания количества (после округления)
- count_mae: MAE количества
- coord_mae: MAE координат (только для существующих устройств)
- coord_within_threshold: % координат в пределах 10% от размера комнаты
- validity_score: % координат внутри [0,1]

Использование:
    python app/nn3/train_v2.py --data data/nn3_v2_dataset --epochs 80 --lr 0.01
"""

import os
import json
import time
import argparse
import torch

from model_v2 import PlacementNetV2, PlacementLoss, calculate_metrics


class PlacementDataLoader:
    """Custom DataLoader для PyTorch Geometric Data"""

    def __init__(self, data_list, batch_size=32, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data_list)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.data_list[idx] for idx in batch_indices]
            yield self._collate(batch)

    def _collate(self, batch):
        """Collate функция для батча"""
        # Простое батчирование — все примеры одинакового размера
        xs = torch.stack([data.x for data in batch])
        edge_indices = [data.edge_index for data in batch]

        targets = {
            "counts": torch.stack([data.target["counts"] for data in batch]),
            "coords": {},
            "masks": {}
        }

        from model_v2 import DEVICES
        for device_type in DEVICES:
            targets["coords"][device_type] = torch.stack([
                data.target["coords"][device_type] for data in batch
            ])
            targets["masks"][device_type] = torch.stack([
                data.target["masks"][device_type] for data in batch
            ])

        return {
            "x": xs,
            "edge_indices": edge_indices,
            "target": targets
        }

    def __len__(self):
        return (len(self.data_list) + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Одна эпоха обучения"""
    model.train()

    total_loss = 0
    total_count_loss = 0
    total_coord_loss = 0
    total_validity_loss = 0
    num_batches = 0

    for batch in dataloader:
        # Перенос на device
        x = batch["x"].to(device)
        target = {
            "counts": batch["target"]["counts"].to(device),
            "coords": {k: v.to(device) for k, v in batch["target"]["coords"].items()},
            "masks": {k: v.to(device) for k, v in batch["target"]["masks"].items()}
        }

        # Forward pass для каждого примера в батче
        batch_size = x.size(0)
        outputs = []

        for i in range(batch_size):
            edge_index = batch["edge_indices"][i].to(device)
            output = model(x[i], edge_index)
            outputs.append(output)

        # Собираем выходы в батч
        batch_output = {
            "counts": torch.stack([o["counts"] for o in outputs]),
            "coords": {}
        }

        from model_v2 import DEVICES
        for device_type in DEVICES:
            batch_output["coords"][device_type] = torch.stack([
                o["coords"][device_type] for o in outputs
            ])

        # Backward pass
        optimizer.zero_grad()
        losses = loss_fn(batch_output, target)
        losses["total_loss"].backward()
        optimizer.step()

        # Статистика
        total_loss += losses["total_loss"].item()
        total_count_loss += losses["count_loss"].item()
        total_coord_loss += losses["coord_loss"].item()
        total_validity_loss += losses["validity_loss"].item()
        num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "count_loss": total_count_loss / num_batches,
        "coord_loss": total_coord_loss / num_batches,
        "validity_loss": total_validity_loss / num_batches
    }


def evaluate(model, dataloader, loss_fn, device):
    """Оценка на валидационном наборе"""
    model.eval()

    total_loss = 0
    all_metrics = []
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Перенос на device
            x = batch["x"].to(device)
            target = {
                "counts": batch["target"]["counts"].to(device),
                "coords": {k: v.to(device) for k, v in batch["target"]["coords"].items()},
                "masks": {k: v.to(device) for k, v in batch["target"]["masks"].items()}
            }

            # Forward pass для каждого примера в батче
            batch_size = x.size(0)
            outputs = []

            for i in range(batch_size):
                edge_index = batch["edge_indices"][i].to(device)
                output = model(x[i], edge_index)
                outputs.append(output)

            # Собираем выходы в батч
            batch_output = {
                "counts": torch.stack([o["counts"] for o in outputs]),
                "coords": {}
            }

            from model_v2 import DEVICES
            for device_type in DEVICES:
                batch_output["coords"][device_type] = torch.stack([
                    o["coords"][device_type] for o in outputs
                ])

            # Loss
            losses = loss_fn(batch_output, target)
            total_loss += losses["total_loss"].item()

            # Метрики
            metrics = calculate_metrics(batch_output, target)
            all_metrics.append(metrics)
            num_batches += 1

    # Усреднённые метрики
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    avg_metrics["total_loss"] = total_loss / num_batches

    return avg_metrics


def train(
        data_dir: str,
        output_dir: str,
        epochs: int = 80,
        batch_size: int = 32,
        lr: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.05,
        gamma: float = 0.01,
        device: str = "auto"
):
    """
    Основная функция обучения

    Args:
        data_dir: Путь к датасету
        output_dir: Путь для сохранения моделей
        epochs: Количество эпох
        batch_size: Размер батча
        lr: Learning rate
        alpha, beta, gamma: Веса loss функции
        device: "cuda", "cpu" или "auto"
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ PlacementNetV2")
    print("=" * 60)
    print()

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}")
    print()

    # Создаём output директорию
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка датасета
    print("Загрузка датасета...")
    train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "val.pt"), weights_only=False)

    print(f"  Train: {len(train_data)} примеров")
    print(f"  Val:   {len(val_data)} примеров")
    print()

    # DataLoaders
    train_loader = PlacementDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PlacementDataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Модель
    print("Инициализация модели...")
    model = PlacementNetV2().to(device)

    # Подсчёт параметров
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Параметров: {num_params:,}")
    print()

    # Loss и optimizer
    loss_fn = PlacementLoss(alpha=alpha, beta=beta, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    print(f"Гиперпараметры:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Loss weights: α={alpha}, β={beta}, γ={gamma}")
    print()

    # История обучения
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_count_acc": [],
        "val_coord_mae": [],
        "val_coord_threshold": []
    }

    best_val_loss = float('inf')
    best_epoch = 0

    print("=" * 60)
    print("ОБУЧЕНИЕ")
    print("=" * 60)
    print()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Обучение
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Валидация
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Scheduler
        scheduler.step(val_metrics["total_loss"])

        # История
        history["train_loss"].append(train_metrics["total_loss"])
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_count_acc"].append(val_metrics["count_acc"])
        history["val_coord_mae"].append(val_metrics["coord_mae"])
        history["val_coord_threshold"].append(val_metrics["coord_within_threshold"])

        # Сохранение лучшей модели
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics
            }, os.path.join(output_dir, "nn3_v2_best.pt"))

        # Вывод прогресса
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1:3d}/{epochs} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['total_loss']:.4f}")
        print(f"  Val Metrics:")
        print(f"    Count Acc:       {val_metrics['count_acc']:.4f}")
        print(f"    Count MAE:       {val_metrics['count_mae']:.4f}")
        print(f"    Coord MAE:       {val_metrics['coord_mae']:.4f}")
        print(f"    Coord Threshold: {val_metrics['coord_within_threshold']:.4f}")

        # Каждые 10 эпох — более детальный вывод
        if (epoch + 1) % 10 == 0:
            print(f"  Best: Epoch {best_epoch + 1} | Val Loss: {best_val_loss:.4f}")
            print()

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Время: {total_time / 60:.1f} минут")
    print(f"Лучшая модель: Epoch {best_epoch + 1} | Val Loss: {best_val_loss:.4f}")
    print()

    # Финальная оценка на лучшей модели
    print("Загрузка лучшей модели для финальной оценки...")
    checkpoint = torch.load(os.path.join(output_dir, "nn3_v2_best.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(model, val_loader, loss_fn, device)

    print()
    print("Финальные метрики (validation set):")
    print(f"  Count Accuracy:      {final_metrics['count_acc']:.4f} (target: >0.90)")
    print(f"  Coord MAE:           {final_metrics['coord_mae']:.4f} (target: <0.05)")
    print(f"  Coord Threshold:     {final_metrics['coord_within_threshold']:.4f} (target: >0.85)")
    print()

    # Проверка целевых метрик
    targets_met = []
    if final_metrics['count_acc'] > 0.90:
        targets_met.append("✅ Count Accuracy")
    else:
        targets_met.append(f"❌ Count Accuracy ({final_metrics['count_acc']:.4f} < 0.90)")

    if final_metrics['coord_mae'] < 0.05:
        targets_met.append("✅ Coord MAE")
    else:
        targets_met.append(f"⚠️ Coord MAE ({final_metrics['coord_mae']:.4f} > 0.05)")

    if final_metrics['coord_within_threshold'] > 0.85:
        targets_met.append("✅ Coord Threshold")
    else:
        targets_met.append(f"⚠️ Coord Threshold ({final_metrics['coord_within_threshold']:.4f} < 0.85)")

    print("Целевые метрики:")
    for status in targets_met:
        print(f"  {status}")
    print()

    # Сохранение истории
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Модель сохранена: {output_dir}/nn3_v2_best.pt")
    print(f"История: {output_dir}/training_history.json")
    print()
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlacementNetV2")

    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--out", type=str, default="models/nn3_v2",
                        help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Count loss weight")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Coord loss weight")
    parser.add_argument("--gamma", type=float, default=0.01,
                        help="Validity loss weight")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")

    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        device=args.device
    )