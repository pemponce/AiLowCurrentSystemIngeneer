"""
app/placement.py

Движок размещения слаботочных устройств.

Поддерживаемые типы устройств (из rules.json):
  - TV_SOCKET       — розетка ТВ (настенная, h=300мм)
  - INTERNET_SOCKET — розетка интернет RJ-45 (настенная, h=300мм)
  - SMOKE_DETECTOR  — датчик дыма (потолочный, ГОСТ Р 53325-2012)
  - CO2_DETECTOR    — датчик CO2/воздуха (настенный, h=1500мм)
  - CEILING_LIGHT   — основное освещение (потолочный центр)
  - NIGHT_LIGHT     — ночник/бра (настенный, h=1400мм)

Входные данные:
  DB["rooms"][project_id]     — list[dict] комнаты из structure_postprocess
  DB["structure"][project_id] — проёмы (двери/окна) если есть

Выходные данные:
  DB["devices"][project_id]   — list[PlacedDevice]
  JSON + PNG через export_*
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from app.geometry import DB, along_wall_points, coerce_polygon, room_walls
from app.rules import get_rules


# ─────────────────────────── типы ────────────────────────────────

@dataclass
class PlacedDevice:
    device_type: str          # TV_SOCKET, SMOKE_DETECTOR, …
    room_id:     str          # room_000, room_001, …
    room_type:   str          # living_room, bedroom, …
    x:           float        # пиксели (image coords)
    y:           float
    mount:       str          # "wall" | "ceiling"
    height_mm:   int          # высота монтажа от пола
    label:       str          # человекочитаемое название
    symbol:      str          # короткий символ для чертежа


# ─────────────────────────── утилиты ─────────────────────────────

def _room_id(idx: int) -> str:
    return f"room_{idx:03d}"


def _room_size_hint(poly: Polygon) -> float:
    return max(1.0, math.sqrt(abs(poly.area)))


def _rooms_as_polygons(project_id: str) -> List[Tuple[int, dict, Polygon]]:
    """
    Возвращает [(idx, raw_room_dict, Polygon), ...].
    Поддерживает как dict из structure_postprocess (с полем 'contour'),
    так и Polygon напрямую.
    """
    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    out = []
    for idx, r in enumerate(rooms_raw):
        if isinstance(r, Polygon):
            out.append((idx, {}, r))
            continue
        if isinstance(r, dict):
            # Пробуем polygon из контура postprocess
            poly = None
            pts = r.get("polygon") or r.get("polygonPx") or r.get("points")
            if pts and isinstance(pts, list) and len(pts) >= 3:
                try:
                    poly = Polygon([(float(p[0]), float(p[1])) for p in pts])
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                except Exception:
                    poly = None
            if poly is None or poly.is_empty or poly.area < 1:
                poly = coerce_polygon(r)
            if poly is not None and poly.area > 1:
                out.append((idx, r, poly))
    return out


def _openings_for_room(project_id: str, room_id: str) -> List[Dict[str, Any]]:
    struct = DB.get("structure", {}).get(project_id)
    if not struct:
        return []
    openings = struct.get("openings") or []
    return [o for o in openings if room_id in (o.get("roomRefs") or [])]


def _opening_seg(o: Dict[str, Any]) -> Optional[LineString]:
    seg = o.get("segmentPx") or o.get("segment")
    if not seg or not isinstance(seg, list) or len(seg) != 2:
        return None
    try:
        (x1, y1), (x2, y2) = seg
        return LineString([(float(x1), float(y1)), (float(x2), float(y2))])
    except Exception:
        return None


def _offset_inside(room: Polygon, base: Point, seg: LineString) -> Point:
    """Сдвигает точку вглубь комнаты от стены-сегмента."""
    x1, y1 = seg.coords[0]
    x2, y2 = seg.coords[-1]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    off = max(0.02 * _room_size_hint(room), 20.0)  # минимум 20px
    p1 = Point(base.x + nx * off, base.y + ny * off)
    p2 = Point(base.x - nx * off, base.y - ny * off)
    if room.contains(p1):
        return p1
    if room.contains(p2):
        return p2
    return base


def _ceiling_grid_points(poly: Polygon, step_px: float) -> List[Point]:
    """Сетка точек внутри полигона для потолочных устройств."""
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    x = minx + step_px / 2
    while x < maxx:
        y = miny + step_px / 2
        while y < maxy:
            p = Point(x, y)
            if poly.contains(p):
                pts.append(p)
            y += step_px
        x += step_px
    return pts


def _best_wall_point(poly: Polygon, opening_segs: List[LineString],
                     min_from_opening_px: float = 50.0) -> Optional[Point]:
    """
    Находит лучшую точку вдоль стен комнаты:
    как можно дальше от проёмов и углов.
    """
    best_pt = None
    best_score = -1.0
    for wall in room_walls(poly):
        L = float(wall.length)
        if L < 80:
            continue
        for t in [0.25, 0.5, 0.75]:
            p = wall.interpolate(t, normalized=True)
            if not poly.contains(p):
                continue
            # минимальное расстояние до проёмов
            min_d = min((seg.distance(p) for seg in opening_segs), default=999.0)
            if min_d < min_from_opening_px:
                continue
            score = min_d
            if score > best_score:
                best_score = score
                best_pt = p
    return best_pt


# ─────────────────────── генерация устройств ─────────────────────

def _place_wall_device(
    device_type: str,
    rule: dict,
    room_idx: int,
    room_dict: dict,
    poly: Polygon,
    opening_segs: List[LineString],
    count: int,
) -> List[PlacedDevice]:
    """Размещает настенное устройство вдоль стен комнаты."""
    results = []
    room_id   = _room_id(room_idx)
    room_type = room_dict.get("room_type", "unknown")
    mount     = rule.get("mount", "wall")
    h_mm      = int(rule.get("height_mm", 300))
    label     = rule.get("label", device_type)
    symbol    = rule.get("symbol", "?")
    min_op    = float(rule.get("min_from_door_mm", 150)) * 1.0  # в px (масштаб ~1px=14мм → /14)
    # Переводим мм → px (эталон 14мм/px)
    mm_per_px = 14.0
    min_op_px = min_op / mm_per_px
    min_corner_px = float(rule.get("min_from_corner_mm", 150)) / mm_per_px

    placed = 0
    used_pts: List[Point] = []

    for wall in room_walls(poly):
        if placed >= count:
            break
        L = float(wall.length)
        if L < min_corner_px * 2:
            continue
        step = max(L / (count - placed + 1), 80.0)
        t = min_corner_px
        while t < L - min_corner_px and placed < count:
            p = wall.interpolate(t)
            # проверяем отступы от проёмов
            min_d_op = min((s.distance(p) for s in opening_segs), default=999.0)
            if min_d_op < min_op_px:
                t += step
                continue
            # проверяем что не слишком близко к уже размещённым
            min_d_used = min((p.distance(u) for u in used_pts), default=999.0)
            if used_pts and min_d_used < 60.0:
                t += step
                continue
            if poly.contains(p) or poly.boundary.distance(p) < 5:
                results.append(PlacedDevice(
                    device_type=device_type,
                    room_id=room_id,
                    room_type=room_type,
                    x=float(p.x), y=float(p.y),
                    mount=mount, height_mm=h_mm,
                    label=label, symbol=symbol,
                ))
                used_pts.append(p)
                placed += 1
            t += step

    # Если не удалось разместить все — fallback на centroid
    while placed < count:
        c = poly.centroid
        results.append(PlacedDevice(
            device_type=device_type,
            room_id=room_id,
            room_type=room_type,
            x=float(c.x), y=float(c.y),
            mount=mount, height_mm=h_mm,
            label=label, symbol=symbol,
        ))
        placed += 1

    return results


def _place_ceiling_device(
    device_type: str,
    rule: dict,
    room_idx: int,
    room_dict: dict,
    poly: Polygon,
    count: int,
) -> List[PlacedDevice]:
    """Размещает потолочное устройство (датчик дыма, светильник)."""
    results = []
    room_id   = _room_id(room_idx)
    room_type = room_dict.get("room_type", "unknown")
    label     = rule.get("label", device_type)
    symbol    = rule.get("symbol", "?")
    min_wall_px = float(rule.get("min_from_wall_mm", 500)) / 14.0

    if count == 1:
        # Один датчик — в центр
        c = poly.centroid
        if not poly.contains(c):
            c = poly.representative_point()
        results.append(PlacedDevice(
            device_type=device_type,
            room_id=room_id, room_type=room_type,
            x=float(c.x), y=float(c.y),
            mount="ceiling", height_mm=0,
            label=label, symbol=symbol,
        ))
    else:
        # Несколько — равномерная сетка
        area_px2 = poly.area
        step = math.sqrt(area_px2 / count) * 0.9
        pts = _ceiling_grid_points(poly, step)
        # Сортируем по расстоянию от центра
        c = poly.centroid
        pts.sort(key=lambda p: p.distance(c))
        placed = 0
        used: List[Point] = []
        min_sep = float(rule.get("min_from_other_detector_mm", 3000)) / 14.0
        for p in pts:
            if placed >= count:
                break
            if min_wall_px > 0 and poly.boundary.distance(p) < min_wall_px:
                continue
            if used and min((p.distance(u) for u in used)) < min_sep:
                continue
            results.append(PlacedDevice(
                device_type=device_type,
                room_id=room_id, room_type=room_type,
                x=float(p.x), y=float(p.y),
                mount="ceiling", height_mm=0,
                label=label, symbol=symbol,
            ))
            used.append(p)
            placed += 1
        # fallback
        while placed < count:
            rp = poly.representative_point()
            results.append(PlacedDevice(
                device_type=device_type,
                room_id=room_id, room_type=room_type,
                x=float(rp.x), y=float(rp.y),
                mount="ceiling", height_mm=0,
                label=label, symbol=symbol,
            ))
            placed += 1

    return results


def _compute_count(device_rule: dict, room_type: str, area_m2: float) -> int:
    """Вычисляет количество устройств для комнаты по правилам."""
    room_rules = device_rule.get("rooms", {})
    if room_type not in room_rules:
        # Используем default_per_m2 если есть
        per_m2 = device_rule.get("default_per_m2")
        if per_m2 and area_m2 > 0:
            return max(1, math.ceil(area_m2 / per_m2))
        return 0

    rr = room_rules[room_type]
    if rr.get("count") is not None:
        return int(rr["count"])
    if rr.get("per_m2") and area_m2 > 0:
        cnt = math.ceil(area_m2 / rr["per_m2"])
        cnt = max(rr.get("min", 1), cnt)
        if "max" in rr:
            cnt = min(rr["max"], cnt)
        return cnt
    return 0


# ─────────────────────── главная функция ─────────────────────────

def generate_placements(project_id: str) -> List[PlacedDevice]:
    """
    Основная функция движка размещения.
    Размещает все устройства по всем комнатам согласно rules.json.

    Возвращает список PlacedDevice и сохраняет в DB["devices"][project_id].
    """
    rules = get_rules()
    device_rules: Dict[str, dict] = rules.get("devices", {})
    mm_per_px = 14.0  # масштаб plan_001 эталон

    room_tuples = _rooms_as_polygons(project_id)
    all_devices: List[PlacedDevice] = []

    for room_idx, room_dict, poly in room_tuples:
        room_id   = _room_id(room_idx)
        room_type = room_dict.get("room_type", "unknown")
        area_m2   = float(room_dict.get("area_m2", poly.area / (1000 / mm_per_px) ** 2))

        # Проёмы для этой комнаты
        opening_segs = [
            s for o in _openings_for_room(project_id, room_id)
            if (s := _opening_seg(o)) is not None
        ]

        for device_type, drule in device_rules.items():
            count = _compute_count(drule, room_type, area_m2)
            if count <= 0:
                continue

            mount = drule.get("mount", "wall")

            if mount == "ceiling":
                devices = _place_ceiling_device(
                    device_type, drule, room_idx, room_dict, poly, count
                )
            else:
                devices = _place_wall_device(
                    device_type, drule, room_idx, room_dict, poly,
                    opening_segs, count
                )

            all_devices.extend(devices)

    DB.setdefault("devices", {})[project_id] = all_devices
    return all_devices


def placements_to_json(devices: List[PlacedDevice]) -> List[Dict[str, Any]]:
    """Сериализует PlacedDevice в список dict для JSON-ответа."""
    return [
        {
            "device_type": d.device_type,
            "room_id":     d.room_id,
            "room_type":   d.room_type,
            "x":           round(d.x, 1),
            "y":           round(d.y, 1),
            "mount":       d.mount,
            "height_mm":   d.height_mm,
            "label":       d.label,
            "symbol":      d.symbol,
        }
        for d in devices
    ]


# ──────── Обратная совместимость со старым API ────────────────────

Candidate = Tuple[str, str, Point]


def generate_candidates(project_id: str) -> List[Candidate]:
    """Устаревший интерфейс — оставлен для совместимости."""
    devices = generate_placements(project_id)
    return [(d.device_type, d.room_id, Point(d.x, d.y)) for d in devices]


def select_devices(project_id: str, candidates: List[Candidate]):
    """Устаревший интерфейс — оставлен для совместимости."""
    DB.setdefault("candidates", {})[project_id] = candidates
    return candidates