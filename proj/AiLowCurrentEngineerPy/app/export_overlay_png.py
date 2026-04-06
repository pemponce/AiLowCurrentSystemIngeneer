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
    "power_socket":     "RZT",
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
    "power_socket":     (255, 140,   0),
    "motion_sensor":    (  0, 180,  80),
    "intercom":         (100, 100, 100),
    "alarm":            (220,   0,   0),
}


import math as _math


def _get_walls(poly: list) -> list:
    """Рёбра полигона с нормалями."""
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
                offset: int = 22, n_device: int = 0) -> Tuple[int, int]:
    """Точка размещения устройства на стене или потолке комнаты."""
    if not poly or len(poly) < 3:
        return int(room_cx), int(room_cy)
    walls = _get_walls(poly)
    if not walls:
        return int(room_cx), int(room_cy)
    # Направляем нормали внутрь
    for w in walls:
        tx, ty   = w["cx"] + w["nx"]*20, w["cy"] + w["ny"]*20
        tx2, ty2 = w["cx"] - w["nx"]*20, w["cy"] - w["ny"]*20
        if _math.hypot(tx-room_cx, ty-room_cy) > _math.hypot(tx2-room_cx, ty2-room_cy):
            w["nx"], w["ny"] = -w["nx"], -w["ny"]
    by_len  = sorted(walls, key=lambda w: w["length"], reverse=True)
    h_walls = [w for w in walls if abs(w["angle"]) < 35 or abs(w["angle"]) > 145]
    k = kind.lower()
    if "ceiling" in k or "smoke" in k or "co2" in k or "motion" in k:
        # Потолочные — центр, сетка 3×N
        step = 70
        col  = n_device % 3
        row  = n_device // 3
        return int(room_cx + (col-1)*step), int(room_cy + (row-0.5)*step)
    elif "tv" in k:
        target = max(h_walls, key=lambda w: w["length"]) if h_walls else by_len[0]
        px = int(target["cx"] + n_device*80 + target["nx"]*offset)
        py = int(target["cy"] + target["ny"]*offset)
        return px, py
    elif "night" in k:
        target = by_len[-1] if len(by_len) > 1 else by_len[0]
        side   = 0.25 + (n_device % 2) * 0.5
        px = int(target["x1"]+(target["x2"]-target["x1"])*side + target["nx"]*offset)
        py = int(target["y1"]+(target["y2"]-target["y1"])*side + target["ny"]*offset)
        return px, py
    elif "internet" in k or "lan" in k:
        target = by_len[0]
        px = int(target["x1"]+(target["x2"]-target["x1"])*0.1 + target["nx"]*offset)
        py = int(target["y1"]+(target["y2"]-target["y1"])*0.1 + target["ny"]*offset)
        return px, py
    elif "power" in k or "socket" in k:
        target  = by_len[0]
        pos_t   = [0.2, 0.5, 0.8, 0.35, 0.65]
        t       = pos_t[n_device % len(pos_t)]
        px = int(target["x1"]+(target["x2"]-target["x1"])*t + target["nx"]*offset)
        py = int(target["y1"]+(target["y2"]-target["y1"])*t + target["ny"]*offset)
        return px, py
    else:
        return int(room_cx), int(room_cy)

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

    elif "power_socket" in k or "socket" in k:
        # Розетка — квадрат с двумя вертикальными линиями (контакты)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, 2)
        pin = max(2, r // 3)
        cv2.line(img, (cx - pin, cy - pin), (cx - pin, cy + pin), color, 2)
        cv2.line(img, (cx + pin, cy - pin), (cx + pin, cy + pin), color, 2)

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
    """Центроид полигона — используем центроид Грина (точный геометрический центр),
    затем проверяем что точка внутри, иначе берём среднее bbox."""
    if not pts or len(pts) < 3:
        return None
    # Центроид Грина (формула shoelace)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    n = len(xs)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = xs[i] * ys[j] - xs[j] * ys[i]
        area  += cross
        cx    += (xs[i] + xs[j]) * cross
        cy    += (ys[i] + ys[j]) * cross
    area /= 2.0
    if abs(area) < 1e-6:
        # Вырожденный полигон — центр bbox
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    # Проверяем что центроид внутри bbox (грубая проверка)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x0 <= cx <= x1 and y0 <= cy <= y1:
        return cx, cy
    # Если центроид вне bbox (самопересекающийся полигон) — центр bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def _poly_area(pts: List) -> float:
    n = len(pts)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(a) / 2.0


def _build_room_centroids(rooms: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """room_id → (cx_px, cy_px)"""
    import logging
    _log = logging.getLogger("planner")
    result: Dict[str, Tuple[float, float]] = {}
    for r in rooms or []:
        if not isinstance(r, dict):
            continue
        _raw_id = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        if isinstance(_raw_id, int):
            rid = f"room_{_raw_id:03d}"
        else:
            rid = str(_raw_id)
        poly = r.get("polygonPx") or r.get("polygon") or r.get("points")
        if rid and poly and len(poly) >= 3:
            c = _poly_centroid(poly)
            if c:
                result[rid] = c
                _log.debug("Centroid %s: (%.0f, %.0f) poly_pts=%d", rid, c[0], c[1], len(poly))
        else:
            _log.debug("No poly for %s: poly=%s", rid, type(poly))
    # Коррекция центроидов для L-образных комнат:
    # если центроид слишком близко к верхнему краю bbox — сдвигаем вниз
    for rid, (cx, cy) in list(result.items()):
        # Найдём bbox этой комнаты
        for r in (rooms or []):
            _raw = r.get("id") or r.get("roomId") or r.get("room_id") or ""
            _rid = f"room_{_raw:03d}" if isinstance(_raw, int) else str(_raw)
            if _rid != rid:
                continue
            poly = r.get("polygonPx") or r.get("polygon") or []
            if len(poly) < 3:
                break
            ys = [float(p[1]) for p in poly]
            y0, y1 = min(ys), max(ys)
            h = y1 - y0
            # Если центроид в верхней четверти bbox — сдвигаем в центр по Y
            if h > 300 and cy < y0 + h * 0.35:
                new_cy = y0 + h * 0.45
                _log.debug("Centroid correction %s: cy %.0f→%.0f (bbox y=[%.0f..%.0f])",
                           rid, cy, new_cy, y0, y1)
                result[rid] = (cx, new_cy)
            break

    _log.info("Built %d centroids from %d rooms", len(result), len(rooms or []))
    return result


def _build_room_bbox(rooms: List[Dict]) -> Dict[str, Tuple[float, float, float, float]]:
    """room_id → (x0, y0, x1, y1) bbox полигона"""
    import logging
    _log = logging.getLogger("planner")
    result: Dict[str, Tuple[float, float, float, float]] = {}
    for r in rooms or []:
        if not isinstance(r, dict):
            continue
        _raw_id = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        if isinstance(_raw_id, int):
            rid = f"room_{_raw_id:03d}"
        else:
            rid = str(_raw_id)
        poly = r.get("polygonPx") or r.get("polygon") or r.get("points")
        if rid and poly and len(poly) >= 3:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            result[rid] = bbox
            _log.debug("BBox %s: x=[%.0f,%.0f] y=[%.0f,%.0f] w=%.0f h=%.0f",
                       rid, bbox[0], bbox[2], bbox[1], bbox[3],
                       bbox[2]-bbox[0], bbox[3]-bbox[1])
    return result


def _place_devices_on_plan(
    devices: List[Dict],
    room_centroids: Dict[str, Tuple[float, float]],
    rooms: List[Dict],
    icon_r: int = 15,
) -> List[Tuple[str, int, int, str]]:
    """
    Возвращает (kind, x_px, y_px, label) для каждого устройства.
    Раскладывает устройства сеткой внутри комнаты.
    """
    out: List[Tuple[str, int, int, str]] = []
    room_device_count: Dict[str, int] = {}

    import logging as _logging
    _log2 = _logging.getLogger("planner")
    # Строим map room_id → bbox и polygon
    room_bbox:  Dict[str, Tuple[float, float, float, float]] = {}
    room_polys: Dict[str, list] = {}
    for r in rooms or []:
        if not isinstance(r, dict):
            continue
        _raw_id2 = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        rid = f"room_{_raw_id2:03d}" if isinstance(_raw_id2, int) else str(_raw_id2)
        poly = r.get("polygonPx") or r.get("polygon") or []
        if rid and len(poly) >= 3:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            room_bbox[rid]  = bbox
            room_polys[rid] = poly
            _log2.debug("BBox %s: x=[%.0f..%.0f] y=[%.0f..%.0f] w=%.0f h=%.0f",
                        rid, bbox[0], bbox[2], bbox[1], bbox[3],
                        bbox[2]-bbox[0], bbox[3]-bbox[1])

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

            # Координаты из NN-3 (xPx/yPx) или fallback на центроид
            if d.get("xPx") is not None and d.get("yPx") is not None:
                px, py = int(d["xPx"]), int(d["yPx"])
            else:
                # Fallback: wall placement если нет координат
                room_poly = room_polys.get(room_id, [])
                px, py = _wall_point(kind, room_poly, cx, cy,
                                     offset=icon_r + 8, n_device=n)

            out.append((kind, px, py, label))

    return out


# ─── главная функция ─────────────────────────────────────────────────────────



def _point_in_polygon(px: float, py: float, poly: list) -> bool:
    """Ray casting — точка внутри полигона."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside


def _nearest_interior_point(cx: float, cy: float, poly: list, step: int = 4) -> tuple:
    """
    Если точка (cx, cy) вне полигона — возвращает ближайшую внутреннюю точку.
    Поиск по спирали вокруг исходной точки с шагом step px.
    """
    if _point_in_polygon(cx, cy, poly):
        return int(cx), int(cy)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    best_dist = float("inf")
    best_pt = (int(cx), int(cy))
    # Перебираем кандидатов внутри bbox с шагом step
    for gx in range(int(bx0) + step, int(bx1), step):
        for gy in range(int(by0) + step, int(by1), step):
            if _point_in_polygon(gx, gy, poly):
                dist = (gx - cx) ** 2 + (gy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_pt = (gx, gy)
    return best_pt


def _build_lighting_zones(poly: list, area_m2: float, room_type: str) -> list:
    """
    Строит сетку зон освещения внутри полигона комнаты.
    Каждая зона — прямоугольник (x0,y0,x1,y1) с центром (cx,cy).
    Зоны не выходят за bbox, центр каждой проверяется ray casting.
    """
    if not poly or len(poly) < 3:
        return []
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx0, bx1 = min(xs), max(xs)
    by0, by1 = min(ys), max(ys)
    w, h = bx1 - bx0, by1 - by0

    # Динамический margin: больше отступ для больших комнат
    if area_m2 > 100:
        margin = 120  # ← Увеличили для больших комнат
    elif area_m2 > 50:
        margin = 40
    else:
        margin = 30

    # Применяем margin к bbox
    bx0 += margin
    bx1 -= margin
    by0 += margin
    by1 -= margin
    w, h = bx1 - bx0, by1 - by0

    # Проверка что margin не съел всю комнату
    if w < 50 or h < 50:
        # Fallback: минимальный margin
        bx0 = min(xs) + 30
        bx1 = max(xs) - 30
        by0 = min(ys) + 30
        by1 = max(ys) - 30
        w, h = bx1 - bx0, by1 - by0

    if room_type in ("bathroom", "toilet"):
        n = 1
    elif room_type == "corridor":
        n = max(1, min(4, round(area_m2 / 8.0)))
    elif room_type == "kitchen":
        n = max(1, min(4, round(area_m2 / 10.0)))
    else:
        n = max(1, min(8, round(area_m2 / 30.0)))

    aspect = w / max(1.0, h)
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = (2, 1) if aspect >= 1.0 else (1, 2)
    elif n <= 4:
        cols, rows = 2, 2
    elif n <= 6:
        cols, rows = (3, 2) if aspect >= 1.0 else (2, 3)
    else:
        cols, rows = 3, 3

    zw, zh = w / cols, h / rows
    zones = []
    for row in range(rows):
        for col in range(cols):
            zx0 = bx0 + zw * col
            zy0 = by0 + zh * row
            cx  = zx0 + zw / 2
            cy  = zy0 + zh / 2
            if _point_in_polygon(cx, cy, poly):
                zones.append({
                    "rect":   (int(zx0), int(zy0), int(zx0+zw), int(zy0+zh)),
                    "center": (int(cx), int(cy)),
                })
            else:
                # Fallback для L-образных и нестандартных комнат:
                # сдвигаем центр к ближайшей внутренней точке
                fx, fy = _nearest_interior_point(cx, cy, poly, step=4)
                zones.append({
                    "rect":   (int(zx0), int(zy0), int(zx0+zw), int(zy0+zh)),
                    "center": (fx, fy),
                })
    return zones


def export_zones_preview(
    base_img_path: str,
    rooms: List[Dict],
    out_path: str,
) -> None:
    """
    Рисует план с сеткой зон освещения.
    Зоны — прямоугольники строго внутри полигонов комнат.
    Используется после NN-1 для визуализации до /design.
    """
    if not (base_img_path and os.path.exists(base_img_path)):
        return
    img = cv2.imread(base_img_path)
    if img is None:
        return

    overlay = img.copy()
    palette = [
        (180, 230, 180), (180, 200, 240), (240, 220, 160),
        (220, 180, 240), (160, 230, 230), (240, 180, 200),
        (200, 240, 180), (240, 200, 160),
    ]

    for ri, r in enumerate(rooms or []):
        if not isinstance(r, dict):
            continue
        poly  = r.get("polygonPx") or []
        area  = float(r.get("areaM2") or r.get("area_m2") or 0)
        rtype = r.get("roomType") or r.get("room_type") or "bedroom"
        if rtype in ("bathroom", "toilet", "balcony") or len(poly) < 3 or area < 3:
            continue

        color  = palette[ri % len(palette)]
        dark   = tuple(max(0, c - 80) for c in color)
        zones  = _build_lighting_zones(poly, area, rtype)

        # 1. Заливаем полигон комнаты
        pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)

        # 2. Рисуем линии сетки ТОЛЬКО внутри полигона через маску
        h_img2, w_img2 = overlay.shape[:2]
        mask = np.zeros((h_img2, w_img2), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        grid_layer = overlay.copy()
        for z in zones:
            zx0, zy0, zx1, zy1 = z["rect"]
            cx,  cy             = z["center"]
            cv2.rectangle(grid_layer, (zx0, zy0), (zx1, zy1), dark, 1)

        # Применяем маску — линии только внутри полигона
        overlay = np.where(mask[:,:,np.newaxis] > 0, grid_layer, overlay)

        # 3. Точки SVT
        for z in zones:
            cx, cy = z["center"]
            cv2.circle(overlay, (cx, cy), 10, (0, 130, 130), -1)
            cv2.circle(overlay, (cx, cy), 14, (0, 90, 90), 2)

        # 4. Контур комнаты
        cv2.polylines(overlay, [pts], isClosed=True, color=dark, thickness=2)

        # 5. Подпись
        xs2 = [p[0] for p in poly]; ys2 = [p[1] for p in poly]
        label_x = int((min(xs2)+max(xs2))/2) - 40
        label_y = int(min(ys2)) + 25
        label   = f"{area:.0f}m2 / {len(zones)}SVT"
        cv2.putText(overlay, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, dark, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

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
    icon_r = max(4, min(10, w // 160))  # радиус иконки — адаптивный

    # Центроиды комнат
    room_centroids = _build_room_centroids(rooms)

    # Размещаем устройства
    placed = _place_devices_on_plan(devices, room_centroids, rooms, icon_r=icon_r)

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
    # Убираем наложения — минимальный шаг между иконками
    MIN_DIST = icon_r * 3
    placed_final = []
    for kind, px, py, label in placed:
        if not (0 <= px < w and 0 <= py < h):
            continue
        # Смещаем если слишком близко к уже нарисованным
        for attempt in range(20):
            too_close = False
            for _, ox, oy, _ in placed_final:
                if abs(px - ox) < MIN_DIST and abs(py - oy) < MIN_DIST:
                    too_close = True
                    px += MIN_DIST
                    if px >= w - icon_r:
                        px -= MIN_DIST * 2
                    break
            if not too_close:
                break
        if 0 <= px < w and 0 <= py < h:
            placed_final.append((kind, px, py, label))

    for kind, px, py, label in placed_final:
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