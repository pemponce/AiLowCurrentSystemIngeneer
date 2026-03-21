"""
app/nn3/model.py

NN-3 — GraphSAGE для размещения устройств.

Архитектура:
  Каждая комната = узел графа с признаками:
    [room_type_onehot(6), area_norm, n_windows_norm,
     n_doors_norm, is_exterior, has_entrance,
     pref_tv, pref_internet, pref_smoke, pref_co2,
     pref_lights, pref_night]
    → итого 18 признаков

  GraphSAGE(2 слоя):
    Каждый узел агрегирует признаки соседей (mean pooling)
    → понимает контекст: спальня рядом с коридором

  Головы (для каждого устройства):
    Регрессия → сколько штук поставить (округляем)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from app.nn3.dataset_gen import DEVICES, ROOM_TYPES

# ─── константы ───────────────────────────────────────────────────────────────

N_ROOM_TYPES  = len(ROOM_TYPES)   # 6
N_EXTRA_FEATS = 5                 # area, n_windows, n_doors, is_exterior, has_entrance
N_PREF_FEATS  = len(DEVICES)      # 6 — пожелания по каждому устройству
N_NODE_FEATS  = N_ROOM_TYPES + N_EXTRA_FEATS + N_PREF_FEATS  # = 17

MAX_ROOMS     = 12   # максимум комнат в квартире
MAX_DEVICE    = 12   # максимальное количество устройств одного типа (до 10 SVT + запас)

ROOM_TYPE_IDX = {r: i for i, r in enumerate(ROOM_TYPES)}


# ─── кодирование узла ─────────────────────────────────────────────────────────

def encode_node(
    node: Dict,
    room_prefs: Optional[Dict] = None,
) -> List[float]:
    """
    Кодирует одну комнату в вектор признаков.

    room_prefs: пожелания для этой комнаты из PreferencesGraph
                {"tv_sockets": 2, "smoke_detector": true, ...}
    """
    # One-hot тип комнаты
    onehot = [0.0] * N_ROOM_TYPES
    idx    = ROOM_TYPE_IDX.get(node["room_type"], 0)
    onehot[idx] = 1.0

    # Числовые признаки (нормализованные)
    area     = min(node.get("area_m2", 10.0), 200.0) / 200.0  # расширен до 200m²
    n_win    = min(node.get("n_windows", 0), 5) / 5.0
    n_doors  = min(node.get("n_doors", 1), 3) / 3.0
    is_ext   = float(node.get("is_exterior", False))
    has_entr = float(node.get("has_entrance", False))

    # Пожелания клиента (0 если нет)
    prefs = room_prefs or {}
    pref_feats = []
    for device in DEVICES:
        val = prefs.get(device, 0)
        if isinstance(val, bool):
            val = float(val)
        pref_feats.append(min(float(val), MAX_DEVICE) / MAX_DEVICE)

    return onehot + [area, n_win, n_doors, is_ext, has_entr] + pref_feats


# ─── датасет ─────────────────────────────────────────────────────────────────

class PlanDataset(Dataset):
    """
    Загружает JSONL из nn3/dataset_gen.py.
    Возвращает (node_feats, adj_matrix, labels, n_rooms).
    """

    def __init__(self, jsonl_path: str) -> None:
        self.samples: List[Dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s     = self.samples[idx]
        nodes = s["nodes"]
        edges = s["edges"]
        prefs = s.get("preferences", {})
        n     = len(nodes)

        # ── признаки узлов ────────────────────────────────────────
        feats = torch.zeros(MAX_ROOMS, N_NODE_FEATS)
        for i, node in enumerate(nodes[:MAX_ROOMS]):
            rtype      = node["room_type"]
            room_prefs = prefs.get(rtype, {})
            feats[i]   = torch.tensor(encode_node(node, room_prefs))

        # ── матрица смежности (нормализованная) ───────────────────
        adj = torch.zeros(MAX_ROOMS, MAX_ROOMS)
        for e in edges:
            fi, ti = e["from"], e["to"]
            if fi < MAX_ROOMS and ti < MAX_ROOMS:
                adj[fi, ti] = 1.0
                adj[ti, fi] = 1.0
        # Self-loops
        for i in range(MAX_ROOMS):
            adj[i, i] = 1.0
        # Нормализация D^{-1/2} A D^{-1/2}
        deg  = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj  = adj / deg

        # ── метки ─────────────────────────────────────────────────
        # labels[room_idx][device_idx] = count (0..MAX_DEVICE)
        labels = torch.zeros(MAX_ROOMS, len(DEVICES), dtype=torch.long)
        for i, node in enumerate(nodes[:MAX_ROOMS]):
            room_id    = node["room_id"]
            room_labels = s["labels"].get(room_id, {})
            for j, device in enumerate(DEVICES):
                val = room_labels.get(device, 0)
                labels[i, j] = min(int(val), MAX_DEVICE)

        return feats, adj, labels, n


# ─── GraphSAGE слой ───────────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    Один слой GraphSAGE с mean aggregation.

    h_v = ReLU(W * [h_v || mean(h_u для u ∈ N(v))])
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (B, N, in_dim)
        # adj: (B, N, N) нормализованная
        agg = torch.bmm(adj, x)                          # (B, N, in_dim)
        out = self.linear(torch.cat([x, agg], dim=-1))   # (B, N, out_dim)
        out = self.norm(out)
        return F.relu(out)


# ─── GNN модель ──────────────────────────────────────────────────────────────

class PlacementGNN(nn.Module):
    """
    GraphSAGE (2 слоя) + головы для каждого устройства.

    Вход:  node_feats (B, MAX_ROOMS, N_NODE_FEATS)
           adj        (B, MAX_ROOMS, MAX_ROOMS)
    Выход: logits     (B, MAX_ROOMS, N_DEVICES, MAX_DEVICE+1)
           — для каждой комнаты × устройства → распределение по количеству
    """

    def __init__(
        self,
        in_dim:    int = N_NODE_FEATS,
        hidden:    int = 64,
        out_dim:   int = 128,
        dropout:   float = 0.3,
        n_devices: int = len(DEVICES),
        max_count: int = MAX_DEVICE,
    ) -> None:
        super().__init__()

        self.sage1   = SAGEConv(in_dim, hidden)
        self.sage2   = SAGEConv(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

        # Головы — одна на каждое устройство
        # Предсказывает количество (классификация 0..max_count)
        self.heads = nn.ModuleList([
            nn.Linear(out_dim, max_count + 1)
            for _ in range(n_devices)
        ])

    def forward(
        self,
        x:   torch.Tensor,   # (B, N, in_dim)
        adj: torch.Tensor,   # (B, N, N)
    ) -> torch.Tensor:       # (B, N, n_devices, max_count+1)

        h = self.sage1(x,   adj)   # (B, N, hidden)
        h = self.dropout(h)
        h = self.sage2(h,   adj)   # (B, N, out_dim)
        h = self.dropout(h)

        # Прогоняем через каждую голову
        outs = []
        for head in self.heads:
            outs.append(head(h))   # (B, N, max_count+1)

        return torch.stack(outs, dim=2)  # (B, N, n_devices, max_count+1)