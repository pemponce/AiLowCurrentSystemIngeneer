from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─── иконки устройств ─────────────────────────────────────────────────────────

DEVICE_LABELS: Dict[str, str] = {
    "tv_sockets":       "TV",
    "internet_sockets": "LAN",
    "smoke_detector":   "DYM",
    "co2_detector":     "CO2",
    "ceiling_lights":   "SVT",
    "night_lights":     "NCH",
    "motion_sensor":    "MOV",
    "intercom":         "DOM",
    "alarm":            "OXR",
}

DEVICE_COLORS: Dict[str, Tuple[int,int,int]] = {
    "tv_sockets":       (200,  50,  50),
    "internet_sockets": ( 50,  50, 200),
    "smoke_detector":   (200, 100,   0),
    "co2_detector":     (  0, 150, 150),
    "ceiling_lights":   (200, 200,   0),
    "night_lights":     (180, 100, 200),
    "motion_sensor":    (  0, 180,  80),
    "intercom":         (100, 100, 100),
    "alarm":            (220,   0,   0),
}
DEFAULT_COLOR = (80, 80, 80)


def _draw_device_icon(
    img: np.ndarray,
    kind: str,
    cx: int,
    cy: int,
    r: int,
    label: str,
) -> None:
    color = DEVICE_COLORS.get(kind, DEFAULT_COLOR)
    bg    = (255, 255, 255)
    k     = kind.lower()

    # Белый фон-кружок
    cv2.circle(img, (cx, cy), r + 2, bg, -1)

    if "light" in k:
        # Светильник — крест в круге
        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.line(img, (cx - r, cy), (cx + r, cy), color, 2)
        cv2.line(img, (cx, cy - r), (cx, cy + r), color, 2)

    elif "smoke" in k:
        # Датчик дыма — круг с точкой
        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.circle(img, (cx, cy), max(2, r // 3), color, -1)

    elif "co2" in k:
        # CO2 — квадрат
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        cv2.putText(img, "CO2", (cx - r + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 30.0, color, 1, cv2.LINE_AA)

    elif "internet" in k or "lan" in k:
        # LAN — квадрат с диагональю
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        cv2.line(img, (cx - r, cy + r), (cx + r, cy - r), color, 1)

    elif "tv" in k:
        # ТВ — круг с горизонтальной линией
        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.line(img, (cx - r, cy), (cx + r, cy), color, 2)

    elif "night" in k:
        # Ночник — треугольник
        pts = np.array([
            [cx, cy - r],
            [cx + int(r * 0.87), cy + r // 2],
            [cx - int(r * 0.87), cy + r // 2],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    elif "intercom" in k or "alarm" in k:
        # Ромб
        pts = np.array([
            [cx, cy - r], [cx + r, cy],
            [cx, cy + r], [cx - r, cy],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    else:
        cv2.circle(img, (cx, cy), r, color, 2)

    # Подпись снизу
    font_scale = max(0.3, r / 20.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(img, label,
                (cx - tw // 2, cy + r + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


# ─── геометрия ────────────────────────────────────────────────────────────────

def _poly_centroid(pts: List) -> Optional[Tuple[float, float]]:
    """Центроид полигона."""
    if not pts or len(pts) < 3:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _poly_area(pts: List) -> float:
    n = len(pts)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(a) / 2.0


def _build_room_centroids(rooms: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """room_id → (cx_px, cy_px)"""
    result: Dict[str, Tuple[float, float]] = {}
    for r in rooms or []:
        if not isinstance(r, dict):
            continue
        rid  = str(r.get("id") or r.get("roomId") or r.get("room_id") or "")
        poly = r.get("polygonPx") or r.get("polygon") or r.get("points")
        if rid and poly and len(poly) >= 3:
            c = _poly_centroid(poly)
            if c:
                result[rid] = c
    return result


def _place_devices_on_plan(
    devices: List[Dict],
    room_centroids: Dict[str, Tuple[float, float]],
    rooms: List[Dict],
) -> List[Tuple[str, int, int, str]]:
    """
    Возвращает (kind, x_px, y_px, label) для каждого устройства.
    Раскладывает устройства сеткой внутри комнаты.
    """
    out: List[Tuple[str, int, int, str]] = []
    room_device_count: Dict[str, int] = {}

    # Строим map room_id → bbox для вычисления шага сетки
    room_bbox: Dict[str, Tuple[float, float, float, float]] = {}
    for r in rooms or []:
        if not isinstance(r, dict):
            continue
        rid  = str(r.get("id") or r.get("roomId") or "")
        poly = r.get("polygonPx") or r.get("polygon") or []
        if rid and len(poly) >= 3:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            room_bbox[rid] = (min(xs), min(ys), max(xs), max(ys))

    for d in devices or []:
        if not isinstance(d, dict):
            continue
        kind    = str(d.get("kind") or d.get("type") or "device")
        room_id = str(d.get("roomRef") or d.get("room_id") or d.get("roomId") or "")
        label   = DEVICE_LABELS.get(kind, kind[:3].upper())

        # Явные координаты (не нулевые)
        x = d.get("x")
        y = d.get("y")
        if x is not None and y is not None and (float(x) != 0 or float(y) != 0):
            out.append((kind, int(float(x)), int(float(y)), label))
            continue

        # Координаты из центроида комнаты + сетка
        if room_id and room_id in room_centroids:
            cx, cy = room_centroids[room_id]
            n = room_device_count.get(room_id, 0)
            room_device_count[room_id] = n + 1

            # Шаг сетки из bbox комнаты
            if room_id in room_bbox:
                x0, y0, x1, y1 = room_bbox[room_id]
                step_x = max(25.0, (x1 - x0) / 5.0)
                step_y = max(25.0, (y1 - y0) / 5.0)
            else:
                step_x = step_y = 30.0

            col = n % 3
            row = n // 3
            px = int(cx + (col - 1) * step_x)
            py = int(cy + row * step_y * 0.8)

            out.append((kind, px, py, label))

    return out


# ─── главная функция ─────────────────────────────────────────────────────────

def export_overlay_png(
    src_image_path: str,
    rooms:   List[Dict],
    devices: List[Dict],
    routes:  List[Any],
    out_path: str,
) -> str:
    """
    Рисует устройства поверх оригинального плана.
    rooms  — список комнат с polygonPx (для центроидов)
    devices — список устройств с roomRef
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не могу прочитать: {src_image_path}")

    h, w = img.shape[:2]
    icon_r = max(10, min(20, w // 80))  # радиус иконки — адаптивный

    # Центроиды комнат
    room_centroids = _build_room_centroids(rooms)

    # Размещаем устройства
    placed = _place_devices_on_plan(devices, room_centroids, rooms)

    # Рисуем трассы (если есть)
    for route in routes or []:
        points = None
        if isinstance(route, dict):
            points = route.get("points") or route.get("polyline")
        elif isinstance(route, (list, tuple)) and len(route) >= 2:
            line = route[1]
            if hasattr(line, "coords"):
                points = [(int(x), int(y)) for x, y in line.coords]
        if not points or len(points) < 2:
            continue
        for a, b in zip(points, points[1:]):
            try:
                cv2.line(img,
                         (int(a[0] if isinstance(a, (list,tuple)) else a["x"]),
                          int(a[1] if isinstance(a, (list,tuple)) else a["y"])),
                         (int(b[0] if isinstance(b, (list,tuple)) else b["x"]),
                          int(b[1] if isinstance(b, (list,tuple)) else b["y"])),
                         (0, 0, 200), 2)
            except Exception:
                pass

    # Рисуем устройства
    for kind, px, py, label in placed:
        # Проверяем что координаты в пределах изображения
        if 0 <= px < w and 0 <= py < h:
            _draw_device_icon(img, kind, px, py, icon_r, label)

    # Легенда в нижнем левом углу
    legend_kinds = sorted(set(k for k, _, _, _ in placed))
    lx, ly = 10, h - 10 - len(legend_kinds) * (icon_r * 3)
    cv2.rectangle(img, (lx - 5, ly - 10),
                  (lx + 130, ly + len(legend_kinds) * icon_r * 3 + 10),
                  (255, 255, 255), -1)
    cv2.rectangle(img, (lx - 5, ly - 10),
                  (lx + 130, ly + len(legend_kinds) * icon_r * 3 + 10),
                  (180, 180, 180), 1)
    for i, kind in enumerate(legend_kinds):
        iy = ly + i * icon_r * 3 + icon_r
        _draw_device_icon(img, kind, lx + icon_r + 5, iy, icon_r, "")
        label = DEVICE_LABELS.get(kind, kind)
        cv2.putText(img, label,
                    (lx + icon_r * 2 + 10, iy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)

    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise ValueError(f"Не могу записать: {out_path}")
    return out_path