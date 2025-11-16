from __future__ import annotations

from typing import Dict, Any, List, Optional

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from .geometry import DB
from .models import (
    LightingRequest,
    LightingResponse,
    RoomLightingResult,
    FixturePlacement,
)

# Базовые нормы освещённости по типу помещений (можно расширять)
TARGET_LUX_BY_ROOM_TYPE: Dict[str, float] = {
    "living": 150.0,
    "bedroom": 100.0,
    "kitchen": 200.0,
}


# --------- Вспомогательные функции ---------

def _extract_geom(room: Any) -> BaseGeometry:
    """
    Достаём shapely-геометрию из объекта комнаты, хранящегося в DB['rooms'].
    Поддерживаем варианты:
      - чистая shapely-геометрия (Polygon/MultiPolygon)
      - dict с ключами 'polygon' / 'geom' / 'geometry'
    """
    from shapely.geometry.base import BaseGeometry as _BG
    if isinstance(room, _BG):
        return room
    if isinstance(room, dict):
        g = room.get("polygon") or room.get("geom") or room.get("geometry")
        if isinstance(g, _BG):
            return g
    raise ValueError(f"Unsupported room structure: {room!r}")


def _room_area(room: Any) -> float:
    return float(_extract_geom(room).area)


def _room_display_name(room_obj: Any, fallback: str) -> str:
    """
    Читаем "человеческое" имя комнаты для логики норм:
    - GeoJSON: properties.name
    - верхний уровень: name или id
    - иначе fallback (room_id)
    """
    if isinstance(room_obj, dict):
        props = room_obj.get("properties") or {}
        if "name" in props and props["name"]:
            return str(props["name"])
        for k in ("name", "id"):
            if k in room_obj and room_obj[k]:
                return str(room_obj[k])
    return fallback


def _pick_target_lux_for_room(
        room_name: str,
        req_default_target: Optional[float],
        overrides: Optional[Dict[str, float]],
) -> float:
    """
    Приоритет:
      1) per_room_target_lux (точное совпадение по имени комнаты или room_id)
      2) общий target_lux из запроса
      3) TARGET_LUX_BY_ROOM_TYPE по нормализованному названию (living/bedroom/...)
      4) дефолт 150 лк
    """
    overrides = overrides or {}
    if room_name in overrides:
        return float(overrides[room_name])

    # пробуем без регистра
    low_map = {k.lower(): v for k, v in overrides.items()}
    if room_name.lower() in low_map:
        return float(low_map[room_name.lower()])

    if req_default_target is not None:
        return float(req_default_target)

    key = room_name.strip().lower()
    if key in TARGET_LUX_BY_ROOM_TYPE:
        return float(TARGET_LUX_BY_ROOM_TYPE[key])

    return 150.0


def _distribute_fixtures_by_area(rooms: Dict[str, Any], total_fixtures: int) -> Dict[str, int]:
    """
    Пропорционально площади распределяем заданное количество светильников по комнатам.
    Сумма точно равна total_fixtures (подгонка после округлений).
    """
    areas = {room_id: _room_area(room) for room_id, room in rooms.items()}
    total_area = sum(areas.values())

    if total_area <= 0 or total_fixtures <= 0:
        return {room_id: 0 for room_id in rooms.keys()}

    raw = {room_id: area / total_area * total_fixtures for room_id, area in areas.items()}
    fixtures = {room_id: int(round(v)) for room_id, v in raw.items()}

    diff = total_fixtures - sum(fixtures.values())
    if diff != 0:
        # распределяем недостающие/лишние штуки от самых больших помещений
        sorted_rooms = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0 and sorted_rooms:
            room_id = sorted_rooms[i % len(sorted_rooms)][0]
            new_val = fixtures[room_id] + step
            if new_val >= 0:
                fixtures[room_id] = new_val
                diff -= step
            i += 1

    return fixtures


def _grid_points_inside(geom: BaseGeometry, count: int) -> List[Point]:
    """
    Простейшее равномерное размещение:
    - берём bbox, строим сетку ~sqrt(N) x sqrt(N),
    - отбираем точки, попавшие внутрь полигона,
    - ограничиваем количеством count.
    """
    if count <= 0:
        return []

    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if width <= 0 or height <= 0:
        return []

    import math
    n_side = max(1, int(math.ceil(math.sqrt(count))))
    nx, ny = n_side, n_side
    dx = width / (nx + 1)
    dy = height / (ny + 1)

    pts: List[Point] = []
    for ix in range(nx):
        for iy in range(ny):
            x = minx + (ix + 1) * dx
            y = miny + (iy + 1) * dy
            p = Point(x, y)
            # слегка «раздуем» полигон, чтобы точка на границе считалась внутри
            if geom.buffer(1e-6).contains(p):
                pts.append(p)

    return pts[:count]


# --------- Основная функция ---------

def design_lighting(req: LightingRequest) -> LightingResponse:
    """
    1) Берём комнаты проекта из DB['rooms'][project_id]
    2) Распределяем total_fixtures пропорционально площади
    3) Для каждой комнаты считаем суммарные люмены под её норму (target_lux)
       с учётом MF и UF, затем на 1 светильник
    4) Размещаем светильники сеткой (x,y)
    5) Возвращаем LightingResponse в формате моделей из app.models
    """

    # --- 1) комнаты проекта ---
    rooms_store = DB["rooms"].get(req.project_id)
    if rooms_store is None:
        raise ValueError(f"Project '{req.project_id}' not found in DB['rooms']")

    # Приводим к словарю {room_id -> room_obj}
    rooms: Dict[str, Any] = {}
    if isinstance(rooms_store, dict):
        for key, val in rooms_store.items():
            rooms[str(key)] = val
    elif isinstance(rooms_store, list):
        for i, val in enumerate(rooms_store):
            rid = None
            if isinstance(val, dict):
                rid = val.get("name") or val.get("id")
            if not rid:
                rid = f"room_{i}"
            rooms[str(rid)] = val
    else:
        raise ValueError(f"Unsupported rooms container type: {type(rooms_store)}")

    # --- 2) распределяем светильники ---
    fixtures_per_room = _distribute_fixtures_by_area(rooms, req.total_fixtures)

    fixtures_out: List[FixturePlacement] = []
    rooms_map: Dict[str, RoomLightingResult] = {}

    # --- 3) расчёты по каждому помещению ---
    for room_id, room_obj in rooms.items():
        geom = _extract_geom(room_obj)
        area = float(geom.area)

        # имя комнаты для норм
        display_name = _room_display_name(room_obj, fallback=room_id)

        # выбираем норму (персональная -> общий target -> карта типов -> дефолт)
        target_lux_used = _pick_target_lux_for_room(
            room_name=display_name,
            req_default_target=req.target_lux,
            overrides=req.per_room_target_lux or {},
        )

        # lumen = lux * m² / (MF * UF)
        total_lumens = target_lux_used * area / (req.maintenance_factor * req.utilization_factor)

        n_fix = max(0, int(fixtures_per_room.get(room_id, 0)))
        if n_fix > 0:
            lumens_per_fixture = total_lumens / n_fix
            power_w_per_fixture = lumens_per_fixture / req.fixture_efficacy_lm_per_w
        else:
            lumens_per_fixture = 0.0
            power_w_per_fixture = 0.0

        # координаты светильников (x, y) — как в модели FixturePlacement
        pts = _grid_points_inside(geom, n_fix) if n_fix > 0 else []
        for p in pts:
            fixtures_out.append(
                FixturePlacement(
                    room_id=room_id,
                    x=float(p.x),
                    y=float(p.y),
                    lumens=lumens_per_fixture,
                    power_w=power_w_per_fixture,
                )
            )

        # Результаты для комнаты — строго по RoomLightingResult
        rooms_map[room_id] = RoomLightingResult(
            area_m2=area,
            fixtures=n_fix,
            lumens_total=total_lumens,
            lumens_per_fixture=lumens_per_fixture,
            power_w_per_fixture=power_w_per_fixture,
        )

    # Сохраним раскладку — может пригодиться далее
    DB.setdefault("fixtures", {})[req.project_id] = fixtures_out

    # --- 4) финальный ответ ---
    return LightingResponse(
        project_id=req.project_id,
        total_fixtures=req.total_fixtures,
        target_lux=req.target_lux or 300.0,
        rooms=rooms_map,        # Dict[str, RoomLightingResult]
        fixtures=fixtures_out,  # List[FixturePlacement]
    )
