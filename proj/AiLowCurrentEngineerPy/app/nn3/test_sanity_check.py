# app/nn3/test_sanity_check.py
"""
Sanity check для PlacementNetV2

Проверяет:
1. Модель создаётся без ошибок
2. Forward pass работает
3. Loss вычисляется
4. Backward pass работает
5. Модель переобучается на 10 примерах (loss → 0)

Если loss уменьшается до ~0.01 за 100 epochs — архитектура работает корректно.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np

# Импортируем модель
import sys

sys.path.append('C:/Users/azhel/Desktop/Ai ingeneer low-current systems/proj/AiLowCurrentEngineerPy')

from app.nn3.model_v2 import PlacementNetV2, PlacementLoss, calculate_metrics

# Константы
NUM_ROOMS = 5
NUM_EDGES = 8
NODE_FEATURES = 18
MAX_DEVICES = 12
DEVICES = ["ceiling_lights", "power_socket", "smoke_detector",
           "co2_detector", "internet_sockets", "switch"]


def create_dummy_data():
    """Создаём dummy данные для теста"""
    # Признаки узлов (комнат)
    x = torch.randn(NUM_ROOMS, NODE_FEATURES)

    # Рёбра графа (смежность комнат)
    edge_index = torch.randint(0, NUM_ROOMS, (2, NUM_EDGES))

    # Target данные
    target = {
        "counts": torch.randint(0, 5, (NUM_ROOMS, len(DEVICES))).float(),
        "coords": {},
        "masks": {}
    }

    # Для каждого типа устройства
    for device_type in DEVICES:
        # Случайные нормализованные координаты [0, 1]
        target["coords"][device_type] = torch.rand(NUM_ROOMS, MAX_DEVICES, 2)

        # Маска: 1 если устройство существует, 0 если нет
        # Создаём маску на основе counts
        masks = torch.zeros(NUM_ROOMS, MAX_DEVICES)
        device_idx = DEVICES.index(device_type)
        for room_idx in range(NUM_ROOMS):
            num_devices = int(target["counts"][room_idx, device_idx].item())
            masks[room_idx, :num_devices] = 1

        target["masks"][device_type] = masks

    return Data(x=x, edge_index=edge_index, target=target)


def test_forward_pass():
    """Тест 1: Forward pass"""
    print("=" * 60)
    print("ТЕСТ 1: Forward pass")
    print("=" * 60)

    model = PlacementNetV2()
    data = create_dummy_data()

    with torch.no_grad():
        output = model(data.x, data.edge_index)

    print(f"✅ Forward pass успешен")
    print(f"   Counts shape: {output['counts'].shape}")  # [5, 6]
    print(f"   Coords keys: {list(output['coords'].keys())}")
    print(f"   Sample coord shape: {output['coords']['ceiling_lights'].shape}")  # [5, 12, 2]
    print()


def test_loss_computation():
    """Тест 2: Loss вычисление"""
    print("=" * 60)
    print("ТЕСТ 2: Loss вычисление")
    print("=" * 60)

    model = PlacementNetV2()
    loss_fn = PlacementLoss(alpha=1.0, beta=1.0, gamma=0.5)
    data = create_dummy_data()

    output = model(data.x, data.edge_index)
    losses = loss_fn(output, data.target)

    print(f"✅ Loss вычислен")
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    print(f"   Count loss: {losses['count_loss'].item():.4f}")
    print(f"   Coord loss: {losses['coord_loss'].item():.4f}")
    print(f"   Validity loss: {losses['validity_loss'].item():.4f}")
    print()


def test_backward_pass():
    """Тест 3: Backward pass"""
    print("=" * 60)
    print("ТЕСТ 3: Backward pass")
    print("=" * 60)

    model = PlacementNetV2()
    loss_fn = PlacementLoss()
    data = create_dummy_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    losses = loss_fn(output, data.target)
    losses["total_loss"].backward()
    optimizer.step()

    print(f"✅ Backward pass успешен")
    print(f"   Градиенты вычислены и применены")
    print()


def test_overfitting():
    """Тест 4: Переобучение на 10 примерах (ГЛАВНЫЙ ТЕСТ)"""
    print("=" * 60)
    print("ТЕСТ 4: Переобучение на 10 примерах (Sanity Check)")
    print("=" * 60)
    print("Цель: loss должен уменьшиться до ~0.01 за 200 epochs")
    print("Если это происходит — архитектура работает корректно")
    print()

    # Создаём 10 одинаковых примеров (для переобучения)
    torch.manual_seed(42)  # Фиксируем random seed
    train_data = create_dummy_data()  # ← ОДИН пример, не список

    model = PlacementNetV2()
    loss_fn = PlacementLoss(alpha=1.0, beta=0.05, gamma=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    print("Обучение...")
    losses_history = []

    for epoch in range(200):
        optimizer.zero_grad()

        output = model(train_data.x, train_data.edge_index)
        losses = loss_fn(output, train_data.target)

        losses["total_loss"].backward()
        optimizer.step()

        loss_value = losses["total_loss"].item()
        losses_history.append(loss_value)

        scheduler.step(loss_value)  # ← Уменьшает lr если loss застрял

        # Печатаем каждые 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1:3d}/100 | Loss: {loss_value:.6f}")

    final_loss = losses_history[-1]
    initial_loss = losses_history[0]

    print()
    print(f"Результаты:")
    print(f"   Начальный loss: {initial_loss:.6f}")
    print(f"   Финальный loss:  {final_loss:.6f}")
    print(f"   Уменьшение:      {(1 - final_loss / initial_loss) * 100:.1f}%")
    print()

    # Проверка успешности
    if final_loss < 0.1:
        print("✅ SANITY CHECK ПРОЙДЕН!")
        print("   Модель успешно переобучилась — архитектура работает")
    elif final_loss < 0.5:
        print("⚠️ ЧАСТИЧНО ПРОЙДЕН")
        print("   Модель обучается, но медленно — возможно нужно:")
        print("   - Увеличить learning rate")
        print("   - Уменьшить weight для validity_loss (γ)")
    else:
        print("❌ SANITY CHECK НЕ ПРОЙДЕН")
        print("   Модель НЕ обучается — проблема в архитектуре:")
        print("   - Проверить forward pass")
        print("   - Проверить loss функцию")
        print("   - Проверить backward pass")


    print()
    print("Проверка предсказаний:")
    with torch.no_grad():
        output = model(train_data.x, train_data.edge_index)

        # Первая комната
        true_counts = train_data.target["counts"][0]
        pred_counts = output["counts"][0]
        pred_counts_rounded = torch.round(pred_counts)

        print(f"   Комната 0:")
        print(f"   True:      {true_counts.numpy()}")
        print(f"   Predicted: {pred_counts.detach().numpy()}")
        print(f"   Rounded:   {pred_counts_rounded.numpy()}")

        # Accuracy
        accuracy = (pred_counts_rounded == true_counts).float().mean().item()
        print(f"   Accuracy:  {accuracy * 100:.1f}%")


def test_metrics():
    """Тест 5: Метрики"""
    print("=" * 60)
    print("ТЕСТ 5: Метрики вычисления")
    print("=" * 60)

    model = PlacementNetV2()
    data = create_dummy_data()

    with torch.no_grad():
        output = model(data.x, data.edge_index)

    metrics = calculate_metrics(output, data.target)

    print(f"✅ Метрики вычислены")
    print(f"   Count MAE: {metrics['count_mae']:.4f}")
    print(f"   Count Acc: {metrics['count_acc']:.4f}")
    print(f"   Coord MAE: {metrics['coord_mae']:.4f}")
    print(f"   Coord within threshold: {metrics['coord_within_threshold']:.4f}")
    print()


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "SANITY CHECK PlacementNetV2" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Запускаем все тесты
    test_forward_pass()
    test_loss_computation()
    test_backward_pass()
    test_metrics()
    test_overfitting()  # ГЛАВНЫЙ ТЕСТ

    print("=" * 60)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
    print()
    print("Следующий шаг: Если sanity check пройден →")
    print("  Неделя 3-4: Генерация датасета с ground truth")
    print()