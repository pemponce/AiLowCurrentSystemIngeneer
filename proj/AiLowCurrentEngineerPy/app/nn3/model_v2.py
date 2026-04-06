# app/nn3/model_v2.py
"""
PlacementNetV2: GraphSAGE с dual-head архитектурой

КРИТИЧЕСКОЕ УЛУЧШЕНИЕ:
- Head 1: Количество устройств (как v1)
- Head 2: Координаты устройств (НОВОЕ) - regression head

Модель предсказывает ДВА выхода:
1. counts: [num_rooms, 6] - количество каждого типа устройств
2. coords: {device_type: [num_rooms, max_devices, 2]} - нормализованные координаты
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

# Константы (совместимость с v1)
DEVICES = [
    "ceiling_lights",
    "power_socket",
    "smoke_detector",
    "co2_detector",
    "internet_sockets",
    "switch"  # Добавлен в v2 (в v1 добавлялся постпроцессингом)
]
MAX_DEVICES_PER_ROOM = 12
MAX_ROOMS = 12
NODE_FEATURES = 17

class PlacementNetV2(nn.Module):
    """
    GraphSAGE модель с dual head для размещения устройств

    Архитектура:
    - Shared GNN encoder (2 слоя GraphSAGE)
    - Classification head: количество устройств
    - Regression head: нормализованные координаты [0, 1]

    Args:
        node_features: Размерность входных признаков узла (по умолчанию 18)
        hidden_dim: Размерность скрытого слоя (по умолчанию 128)
        max_devices_per_room: Максимум устройств одного типа (по умолчанию 12)
        dropout: Вероятность dropout (по умолчанию 0.2)
    """

    def __init__(
            self,
            node_features=NODE_FEATURES,
            hidden_dim=128,
            max_devices_per_room=MAX_DEVICES_PER_ROOM,
            dropout=0.2
    ):
        super(PlacementNetV2, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.max_devices_per_room = max_devices_per_room
        self.num_device_types = len(DEVICES)

        # ============================================================
        # Shared Encoder: GraphSAGE
        # ============================================================
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # ============================================================
        # Head 1: Classification (количество устройств)
        # ============================================================
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, self.num_device_types)
            # Выход: [num_rooms, 6] - количество каждого типа устройств
        )

        # ============================================================
        # Head 2: Regression (координаты устройств)
        # ============================================================
        # Для каждого типа устройства - отдельная сеть
        # Это позволяет учитывать специфику размещения:
        # - ceiling_lights: по центру, сетка
        # - power_socket: у стен
        # - smoke_detector: центроид
        # и т.д.

        self.coord_heads = nn.ModuleDict()

        for device_type in DEVICES:
            self.coord_heads[device_type] = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64, max_devices_per_room * 2),  # (x, y) пары
                nn.Sigmoid()  # Нормализация в [0, 1]
            )
            # Выход: [num_rooms, max_devices*2]
            # После reshape: [num_rooms, max_devices, 2]

    def forward(self, x, edge_index):
        """
        Forward pass

        Args:
            x: Признаки узлов [num_rooms, node_features]
            edge_index: Рёбра графа [2, num_edges]

        Returns:
            {
                "counts": Tensor[num_rooms, 6],
                "coords": {
                    "ceiling_lights": Tensor[num_rooms, max_devices, 2],
                    "power_socket": Tensor[num_rooms, max_devices, 2],
                    ...
                }
            }
        """
        # ============================================================
        # Shared encoding
        # ============================================================
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)
        # h: [num_rooms, hidden_dim]

        # ============================================================
        # Head 1: Device counts
        # ============================================================
        device_counts = self.count_head(h)
        # device_counts: [num_rooms, 6]

        # ============================================================
        # Head 2: Device coordinates
        # ============================================================
        device_coords = {}

        for device_type in DEVICES:
            coords = self.coord_heads[device_type](h)
            # coords: [num_rooms, max_devices*2]

            # Reshape в (x, y) пары
            num_rooms = coords.size(0)
            coords = coords.view(num_rooms, self.max_devices_per_room, 2)
            # coords: [num_rooms, max_devices, 2]

            device_coords[device_type] = coords

        return {
            "counts": device_counts,
            "coords": device_coords
        }

    def predict_with_threshold(self, x, edge_index, count_threshold=0.5):
        """
        Inference с округлением количества устройств

        Args:
            x: Признаки узлов
            edge_index: Рёбра графа
            count_threshold: Порог для округления количества (по умолчанию 0.5)

        Returns:
            {
                "counts": Tensor[num_rooms, 6] (int),
                "coords": {device_type: Tensor[num_rooms, max_devices, 2]},
                "counts_raw": Tensor[num_rooms, 6] (float) - до округления
            }
        """
        output = self.forward(x, edge_index)

        # Округление количества
        counts_raw = output["counts"]
        counts_rounded = torch.clamp(
            torch.round(counts_raw),
            min=0,
            max=self.max_devices_per_room
        ).long()

        return {
            "counts": counts_rounded,
            "coords": output["coords"],
            "counts_raw": counts_raw
        }


# ============================================================================
# Loss функции
# ============================================================================

class PlacementLoss(nn.Module):
    """
    Combined loss для dual-head модели

    Loss = α * count_loss + β * coord_loss + γ * validity_loss

    - count_loss: MSE для количества устройств
    - coord_loss: MSE для координат (только для существующих устройств)
    - validity_loss: Penalty за координаты вне полигона
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        super(PlacementLoss, self).__init__()
        self.alpha = alpha  # Вес для count loss
        self.beta = beta  # Вес для coord loss
        self.gamma = gamma  # Вес для validity loss

    def forward(self, output, target):
        """
        Args:
            output: {
                "counts": [num_rooms, 6],
                "coords": {device_type: [num_rooms, max_devices, 2]}
            }
            target: {
                "counts": [num_rooms, 6],
                "coords": {device_type: [num_rooms, max_devices, 2]},
                "masks": {device_type: [num_rooms, max_devices]}  # 1 если устройство существует
            }

        Returns:
            {
                "total_loss": Tensor,
                "count_loss": Tensor,
                "coord_loss": Tensor,
                "validity_loss": Tensor
            }
        """
        # ============================================================
        # Count loss (MSE)
        # ============================================================
        count_loss = F.mse_loss(output["counts"], target["counts"])

        # ============================================================
        # Coord loss (MSE только для существующих устройств)
        # ============================================================
        coord_loss = 0.0
        num_devices_total = 0

        for device_type in DEVICES:
            pred_coords = output["coords"][device_type]  # [num_rooms, max_devices, 2]
            true_coords = target["coords"][device_type]  # [num_rooms, max_devices, 2]
            mask = target["masks"][device_type]  # [num_rooms, max_devices]

            # Применяем маску - считаем loss только для существующих устройств
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_coords)  # [num_rooms, max_devices, 2]

            diff = (pred_coords - true_coords) ** 2
            masked_diff = diff * mask_expanded

            coord_loss += masked_diff.sum()
            num_devices_total += mask.sum()

        if num_devices_total > 0:
            coord_loss = coord_loss / num_devices_total

        # ============================================================
        # Validity loss (penalty за координаты вне [0, 1])
        # ============================================================
        validity_loss = 0.0

        for device_type in DEVICES:
            coords = output["coords"][device_type]  # [num_rooms, max_devices, 2]
            mask = target["masks"][device_type]  # [num_rooms, max_devices]

            # Координаты должны быть в [0, 1]
            # Penalty за выход за границы
            lower_violation = torch.clamp(-coords, min=0)  # Если coords < 0
            upper_violation = torch.clamp(coords - 1, min=0)  # Если coords > 1

            violations = lower_violation + upper_violation
            mask_expanded = mask.unsqueeze(-1).expand_as(violations)

            validity_loss += (violations * mask_expanded).sum()

        if num_devices_total > 0:
            validity_loss = validity_loss / num_devices_total

        # ============================================================
        # Total loss
        # ============================================================
        total_loss = (
                self.alpha * count_loss +
                self.beta * coord_loss +
                self.gamma * validity_loss
        )

        return {
            "total_loss": total_loss,
            "count_loss": count_loss,
            "coord_loss": coord_loss,
            "validity_loss": validity_loss
        }


# ============================================================================
# Метрики
# ============================================================================

def calculate_metrics(output, target):
    """
    Расчёт метрик для валидации (работает с батчами)

    Args:
        output: Выход модели (батч)
        target: Ground truth (батч)

    Returns:
        {
            "count_mae": float,
            "count_acc": float,
            "coord_mae": float,
            "coord_within_threshold": float
        }
    """
    # Count MAE
    count_mae = torch.abs(output["counts"] - target["counts"]).mean().item()

    # Count accuracy (точное совпадение после округления)
    counts_pred_rounded = torch.round(output["counts"])
    count_acc = (counts_pred_rounded == target["counts"]).float().mean().item()

    # Coord MAE (только для существующих устройств)
    coord_mae_sum = 0.0
    coord_count = 0
    within_threshold_sum = 0

    for device_type in DEVICES:
        pred_coords = output["coords"][device_type]  # [batch, num_rooms, max_devices, 2]
        true_coords = target["coords"][device_type]
        mask = target["masks"][device_type]  # [batch, num_rooms, max_devices]

        # Вычисляем разницу
        diff = torch.abs(pred_coords - true_coords)  # [batch, num_rooms, max_devices, 2]

        # Применяем маску
        mask_expanded = mask.unsqueeze(-1)  # [batch, num_rooms, max_devices, 1]

        # MAE (только для существующих устройств)
        masked_diff = diff * mask_expanded
        coord_mae_sum += masked_diff.sum().item()
        coord_count += mask.sum().item() * 2  # *2 потому что x и y

        # Within threshold (10%)
        max_diff = torch.max(diff[..., 0], diff[..., 1])  # [batch, num_rooms, max_devices]
        within = (max_diff < 0.1).float() * mask
        within_threshold_sum += within.sum().item()

    coord_mae = coord_mae_sum / coord_count if coord_count > 0 else 0.0
    coord_accuracy = within_threshold_sum / (coord_count / 2) if coord_count > 0 else 0.0

    return {
        "count_mae": count_mae,
        "count_acc": count_acc,
        "coord_mae": coord_mae,
        "coord_within_threshold": coord_accuracy
    }

# ============================================================================
# Пример использования
# ============================================================================

if __name__ == "__main__":
    """
    Тестирование архитектуры модели
    """
    import torch

    # Создание dummy данных
    num_rooms = 5
    num_edges = 8

    x = torch.randn(num_rooms, NODE_FEATURES)
    edge_index = torch.randint(0, num_rooms, (2, num_edges))

    # Инициализация модели
    model = PlacementNetV2(
        node_features=NODE_FEATURES,
        hidden_dim=128,
        max_devices_per_room=MAX_DEVICES_PER_ROOM
    )

    # Forward pass
    output = model(x, edge_index)

    print("Model output:")
    print(f"  Counts shape: {output['counts'].shape}")  # [5, 6]
    print(f"  Coords keys: {list(output['coords'].keys())}")
    print(f"  Coords['ceiling_lights'] shape: {output['coords']['ceiling_lights'].shape}")  # [5, 12, 2]

    # Создание dummy target для loss
    target = {
        "counts": torch.randint(0, 5, (num_rooms, len(DEVICES))).float(),
        "coords": {},
        "masks": {}
    }

    for device_type in DEVICES:
        target["coords"][device_type] = torch.rand(num_rooms, MAX_DEVICES_PER_ROOM, 2)
        target["masks"][device_type] = torch.randint(0, 2, (num_rooms, MAX_DEVICES_PER_ROOM)).float()

    # Loss calculation
    loss_fn = PlacementLoss(alpha=1.0, beta=1.0, gamma=0.5)
    losses = loss_fn(output, target)

    print("\nLosses:")
    print(f"  Total: {losses['total_loss'].item():.4f}")
    print(f"  Count: {losses['count_loss'].item():.4f}")
    print(f"  Coord: {losses['coord_loss'].item():.4f}")
    print(f"  Validity: {losses['validity_loss'].item():.4f}")

    # Metrics
    metrics = calculate_metrics(output, target)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Количество параметров
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    """
    Ожидаемый вывод:

    Model output:
      Counts shape: torch.Size([5, 6])
      Coords keys: ['ceiling_lights', 'power_socket', 'smoke_detector', 'co2_detector', 'internet_sockets', 'switch']
      Coords['ceiling_lights'] shape: torch.Size([5, 12, 2])

    Losses:
      Total: X.XXXX
      Count: X.XXXX
      Coord: X.XXXX
      Validity: X.XXXX

    Metrics:
      count_mae: X.XXXX
      count_acc: X.XXXX
      coord_mae: X.XXXX
      coord_within_threshold: X.XXXX

    Total parameters: ~XXX,XXX
    """