# app/nn3/dataset_gen_v2.py
"""
Dataset Generator v2 для PlacementNetV2

Генерирует датасет с ground truth координатами устройств:
- Случайные планы квартир (2-10 комнат)
- Реалистичные полигоны комнат
- Ground truth координаты по ГОСТ/ПУЭ/СП
- Нормализованные координаты [0, 1]
- Маски для существующих устройств

Формат датасета:
- node_features: [num_rooms, 18] — признаки комнат
- edge_index: [2, num_edges] — граф смежности
- counts: [num_rooms, 6] — количество устройств (target)
- coords: {device: [num_rooms, max_devices, 2]} — координаты (target)
- masks: {device: [num_rooms, max_devices]} — маски существования
"""

import os
import json
import random
import numpy as np
import torch
from torch_geometric.data import Data
from shapely.geometry import Polygon, Point
from typing import List, Dict, Tuple
import pickle

from ground_truth_generator import GroundTruthGenerator, normalize_positions

# Константы
DEVICES = [
    "ceiling_lights",
    "power_socket",
    "smoke_detector",
    "co2_detector",
    "internet_sockets",
    "switch"
]

ROOM_TYPES = [
    "living_room",
    "bedroom",
    "kitchen",
    "bathroom",
    "toilet",
    "corridor",
    "balcony"
]

MAX_ROOMS = 12
MAX_DEVICES = 12
NODE_FEATURES = len(ROOM_TYPES) + 4 + len(DEVICES)


class DatasetGeneratorV2:
    """Генератор датасета v2 с ground truth координатами"""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.generator = GroundTruthGenerator()

    def generate_sample(self, sample_id: int) -> Data:
        """
        Генерирует один обучающий пример

        Returns:
            PyTorch Geometric Data объект
        """
        # 1. Генерируем план квартиры
        num_rooms = random.randint(3, 8)  # 3-8 комнат
        rooms = self._generate_apartment_plan(num_rooms)

        # 2. Генерируем граф смежности (топология)
        edge_index = self._generate_topology(num_rooms)

        # 3. Для каждой комнаты генерируем устройства
        node_features = []
        counts_target = []
        coords_target = {device: [] for device in DEVICES}
        masks_target = {device: [] for device in DEVICES}

        for room in rooms:
            # Признаки комнаты
            features = self._compute_node_features(room)
            node_features.append(features)

            # Ground truth устройства
            positions = self.generator.generate_device_positions(
                room["polygon"],
                room["room_type"],
                room["area_m2"]
            )

            # Нормализация координат
            poly = Polygon(room["polygon"])
            normalized = normalize_positions(positions, poly)

            # Формируем targets
            room_counts = []

            for device_type in DEVICES:
                coords = normalized.get(device_type, [])
                num_devices = len(coords)

                # Count target
                room_counts.append(num_devices)

                # Coords target (padding до MAX_DEVICES)
                padded_coords = coords + [(0.5, 0.5)] * (MAX_DEVICES - num_devices)
                coords_target[device_type].append(padded_coords[:MAX_DEVICES])

                # Mask (1 если устройство существует)
                mask = [1.0] * num_devices + [0.0] * (MAX_DEVICES - num_devices)
                masks_target[device_type].append(mask[:MAX_DEVICES])

            counts_target.append(room_counts)

        # Padding до MAX_ROOMS
        while len(node_features) < MAX_ROOMS:
            # Padding комната (все нули)
            node_features.append([0.0] * NODE_FEATURES)
            counts_target.append([0.0] * len(DEVICES))

            for device_type in DEVICES:
                coords_target[device_type].append([(0.5, 0.5)] * MAX_DEVICES)
                masks_target[device_type].append([0.0] * MAX_DEVICES)

        # Конвертация в tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        counts = torch.tensor(counts_target, dtype=torch.float32)

        coords = {}
        masks = {}
        for device_type in DEVICES:
            coords[device_type] = torch.tensor(coords_target[device_type], dtype=torch.float32)
            masks[device_type] = torch.tensor(masks_target[device_type], dtype=torch.float32)

        target = {
            "counts": counts,
            "coords": coords,
            "masks": masks
        }

        return Data(x=x, edge_index=edge_index, target=target, sample_id=sample_id)

    def _generate_apartment_plan(self, num_rooms: int) -> List[Dict]:
        """
        Генерирует случайный план квартиры

        Returns:
            [
                {
                    "room_type": "living_room",
                    "area_m2": 25.0,
                    "polygon": [(x1,y1), (x2,y2), ...],
                    "is_exterior": True,
                    "num_windows": 2,
                    "num_doors": 1
                },
                ...
            ]
        """
        rooms = []

        # Определяем типы комнат
        room_types_pool = self._select_room_types(num_rooms)

        # Генерируем комнаты на сетке
        grid_size = int(np.ceil(np.sqrt(num_rooms)))
        cell_size = 500  # Размер ячейки сетки

        for i, room_type in enumerate(room_types_pool):
            # Позиция на сетке
            row = i // grid_size
            col = i % grid_size

            # Базовая позиция
            base_x = col * cell_size + random.randint(50, 100)
            base_y = row * cell_size + random.randint(50, 100)

            # Размер комнаты (в зависимости от типа)
            if room_type == "living_room":
                width = random.randint(300, 450)
                height = random.randint(300, 450)
            elif room_type in ("bathroom", "toilet", "balcony"):
                width = random.randint(150, 250)
                height = random.randint(150, 250)
            else:
                width = random.randint(250, 400)
                height = random.randint(250, 400)

            # Полигон (прямоугольник с небольшим искажением)
            polygon = self._generate_room_polygon(base_x, base_y, width, height)

            # Площадь
            poly = Polygon(polygon)
            area_px = poly.area
            # Примерная конвертация: 10px = 1m (для масштаба)
            area_m2 = area_px / 100.0

            # Внешняя стена (комнаты по периметру квартиры)
            is_exterior = (row == 0 or col == 0 or
                           row == grid_size - 1 or col == grid_size - 1)

            # Окна и двери
            num_windows = random.randint(1, 2) if is_exterior else 0
            num_doors = 1

            rooms.append({
                "room_type": room_type,
                "area_m2": area_m2,
                "polygon": polygon,
                "is_exterior": is_exterior,
                "num_windows": num_windows,
                "num_doors": num_doors
            })

        return rooms

    def _select_room_types(self, num_rooms: int) -> List[str]:
        """Выбирает типы комнат для квартиры"""
        # Обязательные комнаты
        required = ["living_room", "bedroom", "kitchen", "bathroom"]

        # Дополнительные комнаты
        optional = ["bedroom", "bedroom", "toilet", "corridor", "balcony"]

        # Составляем набор
        room_types = required.copy()

        remaining = num_rooms - len(required)
        if remaining > 0:
            room_types.extend(random.sample(optional, min(remaining, len(optional))))

        # Перемешиваем
        random.shuffle(room_types)

        return room_types[:num_rooms]

    def _generate_room_polygon(
            self,
            base_x: float,
            base_y: float,
            width: float,
            height: float
    ) -> List[Tuple[float, float]]:
        """
        Генерирует полигон комнаты с небольшими искажениями
        """
        # Базовый прямоугольник
        corners = [
            (base_x, base_y),
            (base_x + width, base_y),
            (base_x + width, base_y + height),
            (base_x, base_y + height)
        ]

        # Добавляем небольшие искажения (±5% от размера)
        distortion = min(width, height) * 0.05

        distorted = []
        for x, y in corners:
            dx = random.uniform(-distortion, distortion)
            dy = random.uniform(-distortion, distortion)
            distorted.append((x + dx, y + dy))

        return distorted

    def _generate_topology(self, num_rooms: int) -> List[List[int]]:
        """
        Генерирует граф смежности комнат

        Returns:
            [[source_nodes], [target_nodes]]
        """
        edges = []

        # Соединяем комнаты в цепочку (минимальная связность)
        for i in range(num_rooms - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Неориентированный граф

        # Добавляем случайные рёбра (дополнительные двери)
        num_extra = random.randint(1, max(1, num_rooms // 3))
        for _ in range(num_extra):
            i = random.randint(0, num_rooms - 1)
            j = random.randint(0, num_rooms - 1)
            if i != j:
                edges.append([i, j])
                edges.append([j, i])

        # Конвертация в формат [2, num_edges]
        if edges:
            edge_index = list(zip(*edges))
        else:
            # Fallback: хотя бы одно ребро
            edge_index = [[0, 1], [1, 0]]

        return edge_index

    def _compute_node_features(self, room: Dict) -> List[float]:
        """
        Вычисляет признаки комнаты (18 признаков)

        Features:
        - One-hot room type (7 типов)
        - Площадь (нормализованная)
        - Количество окон
        - Количество дверей
        - Флаг внешней стены
        - Пожелания пользователя (6 устройств) — пока нули
        """
        features = []

        # One-hot room type
        for rt in ROOM_TYPES:
            features.append(1.0 if room["room_type"] == rt else 0.0)

        # Площадь (нормализованная 0-100m²)
        area_norm = min(room["area_m2"] / 100.0, 1.0)
        features.append(area_norm)

        # Окна (нормализованные 0-3)
        windows_norm = min(room["num_windows"] / 3.0, 1.0)
        features.append(windows_norm)

        # Двери (нормализованные 0-3)
        doors_norm = min(room["num_doors"] / 3.0, 1.0)
        features.append(doors_norm)

        # Внешняя стена
        features.append(1.0 if room["is_exterior"] else 0.0)

        # Пожелания пользователя (пока нули — можно добавить позже)
        features.extend([0.0] * 6)

        return features


def generate_dataset(
        output_dir: str,
        num_samples: int = 8000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
):
    """
    Генерирует полный датасет

    Args:
        output_dir: Путь для сохранения
        num_samples: Количество примеров
        train_ratio: Доля train set (по умолчанию 80%)
        val_ratio: Доля val set (по умолчанию 10%)
    """
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА v2")
    print("=" * 60)
    print(f"Количество примеров: {num_samples}")
    print(f"Train: {int(num_samples * train_ratio)}")
    print(f"Val:   {int(num_samples * val_ratio)}")
    print(f"Test:  {int(num_samples * (1 - train_ratio - val_ratio))}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    generator = DatasetGeneratorV2()

    train_data = []
    val_data = []
    test_data = []

    for i in range(num_samples):
        sample = generator.generate_sample(i)

        # Разделение на train/val/test
        rand = random.random()
        if rand < train_ratio:
            train_data.append(sample)
        elif rand < train_ratio + val_ratio:
            val_data.append(sample)
        else:
            test_data.append(sample)

        # Прогресс
        if (i + 1) % 100 == 0:
            print(f"Сгенерировано: {i + 1}/{num_samples} ({(i + 1) / num_samples * 100:.1f}%)")

    print()
    print("Сохранение датасета...")

    # Сохраняем
    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(val_data, os.path.join(output_dir, "val.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))

    # Метаданные
    metadata = {
        "num_samples": num_samples,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "max_rooms": MAX_ROOMS,
        "max_devices": MAX_DEVICES,
        "node_features": NODE_FEATURES,
        "devices": DEVICES,
        "room_types": ROOM_TYPES
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("✅ Датасет сохранён:")
    print(f"   {output_dir}/train.pt ({len(train_data)} примеров)")
    print(f"   {output_dir}/val.pt ({len(val_data)} примеров)")
    print(f"   {output_dir}/test.pt ({len(test_data)} примеров)")
    print(f"   {output_dir}/metadata.json")
    print()
    print("=" * 60)


# ============================================================================
# CLI интерфейс
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset v2 for PlacementNetV2")
    parser.add_argument("--out", type=str, default="data/nn3_v2_dataset",
                        help="Output directory")
    parser.add_argument("--count", type=int, default=8000,
                        help="Number of samples to generate")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Val set ratio")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.out,
        num_samples=args.count,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )