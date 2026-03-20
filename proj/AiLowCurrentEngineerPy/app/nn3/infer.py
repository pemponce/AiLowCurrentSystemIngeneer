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


import math as _math


def _get_walls(poly: list) -> list:
    walls = []
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i+1)%n][0]), float(poly[(i+1)%n][1])
        length = _math.hypot(x2-x1, y2-y1)
        if length < 15:
            continue
        cx2, cy2 = (x1+x2)/2, (y1+y2)/2
        dx, dy = (x2-x1)/length, (y2-y1)/length
        nx, ny = dy, -dx
        angle = _math.degrees(_math.atan2(y2-y1, x2-x1))
        walls.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                       "length":length,"cx":cx2,"cy":cy2,
                       "nx":nx,"ny":ny,"angle":angle})
    return walls


def _wall_point(kind: str, poly: list, room_cx: float, room_cy: float,
                offset: int = 22, n_device: int = 0):
    """Возвращает (px, py) для устройства на стене или потолке."""
    if not poly or len(poly) < 3:
        return None, None
    walls = _get_walls(poly)
    if not walls:
        return None, None
    # Нормали внутрь
    for w in walls:
        tx, ty   = w["cx"] + w["nx"]*20, w["cy"] + w["ny"]*20
        tx2, ty2 = w["cx"] - w["nx"]*20, w["cy"] - w["ny"]*20
        if _math.hypot(tx-room_cx, ty-room_cy) > _math.hypot(tx2-room_cx, ty2-room_cy):
            w["nx"], w["ny"] = -w["nx"], -w["ny"]
    by_len  = sorted(walls, key=lambda w: w["length"], reverse=True)
    h_walls = [w for w in walls if abs(w["angle"]) < 35 or abs(w["angle"]) > 145]
    k = kind.lower()
    if "ceiling" in k or "smoke" in k or "co2" in k or "motion" in k:
        # Равномерная сетка по площади комнаты
        if poly and len(poly) >= 3:
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w_r = x1 - x0
            h_r = y1 - y0
            aspect = w_r / max(1.0, h_r)
            if aspect > 1.5:
                cols, rows = 3, 2
            elif aspect < 0.67:
                cols, rows = 2, 3
            else:
                cols, rows = 2, 2
            col = n_device % cols
            row = (n_device // cols) % rows
            pad_x = w_r * 0.15
            pad_y = h_r * 0.15
            step_x = (w_r - 2*pad_x) / max(1, cols-1) if cols > 1 else 0
            step_y = (h_r - 2*pad_y) / max(1, rows-1) if rows > 1 else 0
            return int(x0 + pad_x + col*step_x), int(y0 + pad_y + row*step_y)
        step = 70
        col, row = n_device % 3, n_device // 3
        return int(room_cx + (col-1)*step), int(room_cy + (row-0.5)*step)
    elif "tv" in k:
        # TV — самая длинная стена далеко от центроида (внешняя стена)
        # score = length * distance_from_centroid — предпочитаем дальние длинные стены
        def _tv_score(w):
            dist = _math.hypot(w["cx"] - room_cx, w["cy"] - room_cy)
            return w["length"] * dist
        target = max(walls, key=_tv_score)
        t = 0.5 + (n_device * 70) / max(1.0, target["length"])
        t = min(t, 0.85)
        mid_x = target["x1"] + (target["x2"]-target["x1"]) * t
        mid_y = target["y1"] + (target["y2"]-target["y1"]) * t
        return int(mid_x + target["nx"]*offset), int(mid_y + target["ny"]*offset)
    elif "night" in k:
        # Ночник — короткая дальняя стена (изголовье кровати)
        def _nch_score(w):
            dist = _math.hypot(w["cx"] - room_cx, w["cy"] - room_cy)
            return dist / max(1.0, w["length"])
        target = max(walls, key=_nch_score)
        side = 0.25 + (n_device % 2) * 0.5
        return int(target["x1"]+(target["x2"]-target["x1"])*side + target["nx"]*offset),                int(target["y1"]+(target["y2"]-target["y1"])*side + target["ny"]*offset)
    elif "internet" in k or "lan" in k:
        target = by_len[0]
        return int(target["x1"]+(target["x2"]-target["x1"])*0.1 + target["nx"]*offset),                int(target["y1"]+(target["y2"]-target["y1"])*0.1 + target["ny"]*offset)
    elif "power" in k or "socket" in k:
        # Розетки — распределяем по РАЗНЫМ стенам
        # n_device-я розетка идёт на n_device-ю стену по длине
        sorted_walls = sorted(walls, key=lambda w: w["length"], reverse=True)
        # Берём стену по циклу — каждая следующая розетка на другой стене
        wall_idx = n_device % len(sorted_walls)
        target = sorted_walls[wall_idx]
        # Позиция вдоль стены — чередуем начало/середину/конец
        pos_t = [0.25, 0.75, 0.5, 0.15, 0.85]
        t = pos_t[(n_device // len(sorted_walls)) % len(pos_t)]
        return int(target["x1"]+(target["x2"]-target["x1"])*t + target["nx"]*offset),                int(target["y1"]+(target["y2"]-target["y1"])*t + target["ny"]*offset)
    else:
        return int(room_cx), int(room_cy)


def _to_design_graph(
    placement:  Dict[str, Dict[str, int]],
    nodes:      List[Dict],
    project_id: str,
) -> Dict[str, Any]:
    """Конвертирует предсказания в формат DesignGraph с координатами на стенах."""
    devices_list = []
    room_designs = []

    # Строим карту room_id → (centroid, polygon)
    node_map = {n["room_id"]: n for n in nodes}

    for node in nodes:
        room_id   = node["room_id"]
        room_type = node["room_type"]
        counts    = placement.get(room_id, {})
        device_ids = []
        poly      = node.get("polygonPx") or []
        # Центроид — из centroidPx или вычисляем из bbox
        cp = node.get("centroidPx") or []
        if cp and len(cp) >= 2:
            cx, cy = float(cp[0]), float(cp[1])
        elif poly:
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        else:
            cx, cy = 0.0, 0.0

        for device, count in counts.items():
            for k in range(count):
                dev_id = f"{room_id}_{device}_{k}"
                px, py = _wall_point(device, poly, cx, cy, offset=22, n_device=k)
                dev_entry = {
                    "id":       dev_id,
                    "kind":     device,
                    "roomRef":  room_id,
                    "mount":    "ceiling" if "light" in device else "wall",
                    "heightMm": 0 if "light" in device else 300,
                    "label":    device.replace("_", " ").title(),
                    "reason":   "NN-3 prediction",
                }
                if px is not None:
                    # Clamp внутри bbox комнаты
                    px, py = _clamp_to_bbox(px, py, poly, margin=12)
                    dev_entry["xPx"] = px
                    dev_entry["yPx"] = py
                devices_list.append(dev_entry)
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
            "polygonPx":  room.get("polygonPx") or [],
            "centroidPx": room.get("centroidPx") or [],
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


def _clamp_to_bbox(px: int, py: int, poly: list, margin: int = 10):
    """Ограничиваем координаты внутри bbox полигона с отступом."""
    if not poly or len(poly) < 3:
        return px, py
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs) + margin, max(xs) - margin
    y0, y1 = min(ys) + margin, max(ys) - margin
    return int(max(x0, min(x1, px))), int(max(y0, min(y1, py)))