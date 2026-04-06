# app/svt_validator.py
"""
Валидация и коррекция позиций потолочных светильников (SVT)

Правила (СП 52.13330.2016):
- Минимум 80px от стен (≈0.8м)
- Минимум 120px между светильниками (≈1.2м)
- Для L-комнат: проверка "тёмных зон" и добавление SVT
"""

import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# Константы по ГОСТ/СП
MIN_WALL_DISTANCE = 80  # Минимум 80px от стен (≈0.8м)
MIN_SVT_DISTANCE = 120  # Минимум 120px между SVT (≈1.2м)
DARK_ZONE_RADIUS = 250  # Радиус освещения одного SVT (≈2.5м)


def _distance_to_polygon_edge(point: Tuple[float, float], polygon: Polygon) -> float:
    """
    Вычисляет минимальное расстояние от точки до края полигона

    Returns:
        float: Расстояние в пикселях
    """
    px, py = point
    pt = Point(px, py)

    # Расстояние до внешнего контура
    boundary = polygon.boundary
    return pt.distance(boundary)


def _move_point_away_from_walls(
        point: Tuple[float, float],
        polygon: Polygon,
        min_distance: float = MIN_WALL_DISTANCE
) -> Optional[Tuple[float, float]]:
    """
    Сдвигает точку от стен если она слишком близко

    Returns:
        Скорректированная точка или None если невозможно скорректировать
    """
    px, py = point
    dist_to_wall = _distance_to_polygon_edge(point, polygon)

    if dist_to_wall >= min_distance:
        return point  # Всё ОК, коррекция не нужна

    # Нужно сдвинуть точку к центру комнаты
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Вектор от точки к центру
    dx = cx - px
    dy = cy - py
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1:
        return None  # Точка совпадает с центром

    # Нормализуем вектор
    dx /= length
    dy /= length

    # Сдвигаем точку на нужное расстояние
    needed_shift = min_distance - dist_to_wall + 10  # +10px запас
    new_px = px + dx * needed_shift
    new_py = py + dy * needed_shift

    # Проверяем что новая точка внутри полигона
    if polygon.contains(Point(new_px, new_py)):
        return (int(new_px), int(new_py))

    return None  # Не удалось скорректировать


def _are_svts_too_close(
        svt1: Tuple[float, float],
        svt2: Tuple[float, float],
        min_distance: float = MIN_SVT_DISTANCE
) -> bool:
    """
    Проверяет что два светильника не слишком близко друг к другу

    Returns:
        True если слишком близко (нужно удалить один)
    """
    x1, y1 = svt1
    x2, y2 = svt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist < min_distance


def validate_and_correct_svt_positions(
        svt_positions: List[Tuple[float, float]],
        polygon_coords: List[Tuple[float, float]],
        area_m2: float,
        room_type: str = "living_room"
) -> List[Tuple[float, float]]:
    """
    Валидирует и корректирует позиции SVT с учётом ГОСТ/СП

    Шаги:
    1. Проверка расстояния от стен → коррекция или удаление
    2. Проверка расстояния между SVT → удаление дублей
    3. Проверка "тёмных зон" → добавление SVT если нужно

    Args:
        svt_positions: Список позиций (x, y)
        polygon_coords: Координаты полигона комнаты
        area_m2: Площадь комнаты
        room_type: Тип комнаты

    Returns:
        Валидированный и скорректированный список позиций
    """

    if room_type in ("bathroom", "toilet", "balcony"):
        return svt_positions  # Возвращаем как есть, без валидации

    if not svt_positions or not polygon_coords:
        return []

    # Создаём Shapely полигон
    try:
        polygon = Polygon(polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Исправляем невалидный полигон
    except Exception:
        return svt_positions  # Fallback: возвращаем как есть

    # ── ШАГ 1: Проверка расстояния от стен ─────────────────────────────
    corrected_positions = []

    for pos in svt_positions:
        dist_to_wall = _distance_to_polygon_edge(pos, polygon)

        if dist_to_wall >= MIN_WALL_DISTANCE:
            # Всё ОК
            corrected_positions.append(pos)
        else:
            # Пытаемся скорректировать
            corrected = _move_point_away_from_walls(pos, polygon)
            if corrected:
                corrected_positions.append(corrected)
            # Если не удалось скорректировать — удаляем этот SVT

    # ── ШАГ 2: Проверка расстояния между SVT ───────────────────────────
    final_positions = []

    for i, pos1 in enumerate(corrected_positions):
        # Проверяем со всеми уже добавленными SVT
        too_close = False
        for pos2 in final_positions:
            if _are_svts_too_close(pos1, pos2):
                too_close = True
                break

        if not too_close:
            final_positions.append(pos1)

    # ── ШАГ 3: Проверка "тёмных зон" (для L-комнат) ────────────────────
    if room_type not in ("bathroom", "toilet", "balcony") and area_m2 > 30:
        final_positions = _add_svt_for_dark_zones(
            final_positions,
            polygon,
            area_m2
        )

    return final_positions


def _add_svt_for_dark_zones(
        svt_positions: List[Tuple[float, float]],
        polygon: Polygon,
        area_m2: float
) -> List[Tuple[float, float]]:
    """
    Находит "тёмные зоны" в L-комнатах и добавляет SVT

    Алгоритм:
    1. Создаём круги освещения вокруг каждого SVT
    2. Объединяем круги в одну зону покрытия
    3. Находим разницу между комнатой и покрытием = тёмные зоны
    4. Если тёмная зона > 20% площади → добавляем SVT в центр зоны
    """
    if not svt_positions:
        return svt_positions

    # Круги освещения
    coverage_zones = []
    for x, y in svt_positions:
        circle = Point(x, y).buffer(DARK_ZONE_RADIUS)
        coverage_zones.append(circle)

    # Объединяем все зоны покрытия
    try:
        total_coverage = unary_union(coverage_zones)
    except Exception:
        return svt_positions  # Fallback

    # Находим тёмные зоны
    try:
        dark_zones = polygon.difference(total_coverage)
    except Exception:
        return svt_positions

    # Если тёмная зона слишком большая — добавляем SVT
    if dark_zones.is_empty:
        return svt_positions

    # Площадь тёмных зон
    dark_area_px = dark_zones.area
    room_area_px = polygon.area
    dark_percentage = dark_area_px / room_area_px if room_area_px > 0 else 0

    # Порог: если больше 15% комнаты в темноте — добавляем SVT
    if dark_percentage > 0.15:
        # Находим центр самой большой тёмной зоны
        if hasattr(dark_zones, 'geoms'):  # MultiPolygon
            largest_zone = max(dark_zones.geoms, key=lambda z: z.area)
        else:  # Polygon
            largest_zone = dark_zones

        # Центр тёмной зоны
        dark_center = largest_zone.centroid
        new_svt = (int(dark_center.x), int(dark_center.y))

        # Проверяем что новый SVT не слишком близко к существующим
        too_close = False
        for existing_svt in svt_positions:
            if _are_svts_too_close(new_svt, existing_svt, MIN_SVT_DISTANCE):
                too_close = True
                break

        if not too_close:
            # Проверяем расстояние от стен
            dist_to_wall = _distance_to_polygon_edge(new_svt, polygon)
            if dist_to_wall >= MIN_WALL_DISTANCE:
                svt_positions.append(new_svt)
            else:
                # Пытаемся скорректировать
                corrected = _move_point_away_from_walls(new_svt, polygon)
                if corrected:
                    svt_positions.append(corrected)

    return svt_positions


# ============================================================================
# Интеграция с placement.py
# ============================================================================

def apply_svt_validation_to_design_graph(
        design_graph: dict,
        rooms: List[dict]
) -> dict:
    """
    Применяет валидацию SVT ко всем комнатам в design_graph

    Использование:
        # В placement.py, функция _apply_hard_rules, после добавления SVT:
        from app.svt_validator import apply_svt_validation_to_design_graph
        design_graph = apply_svt_validation_to_design_graph(design_graph, rooms)
    """
    devices = design_graph.get("devices", [])
    room_designs = design_graph.get("roomDesigns", [])

    # Карта room_id → polygon
    room_polygons = {}
    room_areas = {}
    room_types = {}

    for r in rooms:
        rid = r.get("id") or r.get("roomId") or ""
        if isinstance(rid, int):
            rid = f"room_{rid:03d}"
        poly = r.get("polygonPx") or []
        area = r.get("areaM2") or r.get("area_m2") or 0
        if poly:
            room_polygons[rid] = poly
            room_areas[rid] = area

    for rd in room_designs:
        rid = rd["roomId"]
        rtype = rd.get("roomType", "bedroom")
        room_types[rid] = rtype

    # Группируем SVT по комнатам
    svt_by_room = {}
    other_devices = []

    for d in devices:
        if d.get("kind") == "ceiling_lights":
            room_id = d.get("roomRef") or d.get("room_id", "")
            svt_by_room.setdefault(room_id, []).append(d)
        else:
            other_devices.append(d)

    # Валидируем SVT для каждой комнаты
    validated_devices = other_devices.copy()

    for room_id, svt_devices in svt_by_room.items():
        poly = room_polygons.get(room_id)
        area = room_areas.get(room_id, 0)
        rtype = room_types.get(room_id, "living_room")

        if not poly:
            # Нет полигона — оставляем как есть
            validated_devices.extend(svt_devices)
            continue

        # Извлекаем позиции
        positions = []
        for d in svt_devices:
            x = d.get("xPx")
            y = d.get("yPx")
            if x is not None and y is not None:
                positions.append((x, y))

        # Валидация и коррекция
        validated_positions = validate_and_correct_svt_positions(
            positions, poly, area, rtype
        )

        # Создаём новые устройства с валидированными позициями
        for k, (x, y) in enumerate(validated_positions):
            validated_devices.append({
                "id": f"{room_id}_ceiling_lights_validated_{k}",
                "kind": "ceiling_lights",
                "roomRef": room_id,
                "mount": "ceiling",
                "heightMm": 0,
                "label": "Ceiling Lights",
                "reason": f"validated: {len(validated_positions)} SVT, min {MIN_WALL_DISTANCE}px from walls",
                "xPx": int(x),
                "yPx": int(y),
            })

    # Обновляем design_graph
    design_graph["devices"] = validated_devices
    design_graph["totalDevices"] = len(validated_devices)

    # Пересчитываем deviceIds в roomDesigns
    dev_ids_by_room = {}
    for d in validated_devices:
        rid = d.get("roomRef") or d.get("room_id", "")
        dev_ids_by_room.setdefault(rid, []).append(d["id"])

    new_room_designs = []
    for rd in room_designs:
        rid = rd["roomId"]
        new_room_designs.append({
            **rd,
            "deviceIds": dev_ids_by_room.get(rid, []),
        })

    design_graph["roomDesigns"] = new_room_designs
    design_graph["explain"] = design_graph.get("explain", []) + [
        f"SVT валидация: min {MIN_WALL_DISTANCE}px от стен, min {MIN_SVT_DISTANCE}px между SVT"
    ]

    return design_graph