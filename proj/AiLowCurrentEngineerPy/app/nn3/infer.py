"""
app/nn3/infer.py — инференс NN-3

Использование:
  from app.nn3.infer import run_placement
  design = run_placement(plan_graph, preferences_graph, project_id="p001")
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import torch

from app.nn3.model import PlacementGNN, PlanDataset, encode_node, MAX_DEVICE, MAX_ROOMS, N_NODE_FEATS
from app.nn3.dataset_gen import DEVICES, ROOM_TYPES


class PlacementInfer:
    def __init__(self, model: PlacementGNN, device: str) -> None:
        self.model  = model
        self.device = device
        self.model.eval()

    @classmethod
    def load(cls, model_dir: str) -> "PlacementInfer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(os.path.join(model_dir, "nn3_best.pt"), map_location=device)
        model  = PlacementGNN().to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"NN-3 загружена: val_acc={ckpt.get('val_acc', '?'):.3f}")
        return cls(model, device)

    @torch.no_grad()
    def predict(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        preferences: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Предсказывает количество устройств для каждой комнаты.
        Возвращает {room_id: {device: count}}.
        """
        prefs = preferences or {}
        n     = min(len(nodes), MAX_ROOMS)

        # Признаки узлов
        feats = torch.zeros(1, MAX_ROOMS, N_NODE_FEATS)
        for i, node in enumerate(nodes[:MAX_ROOMS]):
            rtype      = node.get("room_type", "unknown")
            room_prefs = prefs.get(rtype, {})
            feats[0, i] = torch.tensor(encode_node(node, room_prefs))

        # Матрица смежности
        adj = torch.zeros(1, MAX_ROOMS, MAX_ROOMS)
        for e in edges:
            fi, ti = e["from"], e["to"]
            if fi < MAX_ROOMS and ti < MAX_ROOMS:
                adj[0, fi, ti] = 1.0
                adj[0, ti, fi] = 1.0
        for i in range(MAX_ROOMS):
            adj[0, i, i] = 1.0
        deg = adj.sum(dim=2, keepdim=True).clamp(min=1.0)
        adj = adj / deg

        feats = feats.to(self.device)
        adj   = adj.to(self.device)

        logits = self.model(feats, adj)   # (1, N, D, C)
        preds  = logits.argmax(dim=-1)    # (1, N, D)

        result: Dict[str, Dict[str, int]] = {}
        for i, node in enumerate(nodes[:n]):
            room_id    = node["room_id"]
            room_preds = {}
            for j, device in enumerate(DEVICES):
                count = int(preds[0, i, j].item())
                if count > 0:
                    room_preds[device] = count
            result[room_id] = room_preds

        return result


def _to_design_graph(
    placement:  Dict[str, Dict[str, int]],
    nodes:      List[Dict],
    project_id: str,
) -> Dict[str, Any]:
    """Конвертирует предсказания в формат DesignGraph."""
    devices_list = []
    room_designs = []

    for node in nodes:
        room_id   = node["room_id"]
        room_type = node["room_type"]
        counts    = placement.get(room_id, {})
        device_ids = []

        for device, count in counts.items():
            for k in range(count):
                dev_id = f"{room_id}_{device}_{k}"
                devices_list.append({
                    "id":       dev_id,
                    "kind":     device,
                    "roomRef":  room_id,
                    "mount":    "ceiling" if device == "ceiling_light" else "wall",
                    "heightMm": 0 if device == "ceiling_light" else 300,
                    "label":    device.replace("_", " ").title(),
                    "reason":   "NN-3 prediction",
                })
                device_ids.append(dev_id)

        room_designs.append({
            "roomId":    room_id,
            "roomType":  room_type,
            "deviceIds": device_ids,
            "violations": [],
        })

    return {
        "version":     "design-1.0",
        "projectId":   project_id,
        "devices":     devices_list,
        "routes":      [],
        "roomDesigns": room_designs,
        "totalDevices": len(devices_list),
        "explain":     ["Размещение выполнено NN-3 (GNN GraphSAGE)"],
    }


# ─── публичный интерфейс ─────────────────────────────────────────────────────

_infer_instance: Optional[PlacementInfer] = None

def get_infer(model_dir: str = "models/nn3") -> Optional[PlacementInfer]:
    global _infer_instance
    if _infer_instance is None:
        try:
            _infer_instance = PlacementInfer.load(model_dir)
        except Exception as e:
            print(f"NN-3 не загружена ({e}), используем эвристику")
    return _infer_instance


def run_placement(
    plan_graph:   Dict[str, Any],
    prefs_graph:  Optional[Dict[str, Any]] = None,
    project_id:   str = "unknown",
) -> Dict[str, Any]:
    """
    Главная функция. Принимает PlanGraph и PreferencesGraph dict.
    Возвращает DesignGraph dict.
    """
    # Конвертируем PlanGraph → nodes/edges для GNN
    rooms = plan_graph.get("rooms", [])
    nodes = []
    for room in rooms:
        nodes.append({
            "room_id":    room.get("id", str(uuid.uuid4())),
            "room_type":  room.get("roomType", "unknown"),
            "area_m2":    room.get("areaM2") or 10.0,
            "n_windows":  sum(1 for o in plan_graph.get("openings", [])
                             if o.get("kind") == "window"
                             and room.get("id") in o.get("roomRefs", [])),
            "n_doors":    sum(1 for o in plan_graph.get("openings", [])
                             if o.get("kind") == "door"
                             and room.get("id") in o.get("roomRefs", [])),
            "is_exterior": room.get("isExterior", False),
            "has_entrance": False,
        })

    # Рёбра из topology
    edges = []
    topology = plan_graph.get("topology") or {}
    for adj in topology.get("roomAdjacency", []):
        from_id = adj.get("from", "")
        to_id   = adj.get("to", "")
        from_idx = next((i for i, n in enumerate(nodes) if n["room_id"] == from_id), None)
        to_idx   = next((i for i, n in enumerate(nodes) if n["room_id"] == to_id),   None)
        if from_idx is not None and to_idx is not None:
            edges.append({"from": from_idx, "to": to_idx})

    # Если нет рёбер — строим цепочку
    if not edges and len(nodes) > 1:
        for i in range(len(nodes) - 1):
            edges.append({"from": i, "to": i + 1})

    # Пожелания
    prefs: Dict[str, Any] = {}
    if prefs_graph:
        for room_pref in prefs_graph.get("rooms", []):
            rtype = room_pref.get("roomType", "")
            prefs[rtype] = {k: v for k, v in room_pref.items() if k != "roomType"}

    infer = get_infer()
    if infer:
        placement = infer.predict(nodes, edges, prefs)
    else:
        # Эвристика fallback
        from app.nn3.dataset_gen import BASE_RULES, _lights_from_area
        import random
        placement = {}
        for node in nodes:
            rtype  = node["room_type"]
            rules  = BASE_RULES.get(rtype, {})
            result = {}
            for device, (mn, mx) in rules.items():
                if mx > 0:
                    if device == "ceiling_lights":
                        result[device] = _lights_from_area(node["area_m2"])
                    else:
                        result[device] = random.randint(mn, mx)
            placement[node["room_id"]] = result

    return _to_design_graph(placement, nodes, project_id)