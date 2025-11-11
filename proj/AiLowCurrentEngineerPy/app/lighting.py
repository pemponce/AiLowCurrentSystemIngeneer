from typing import Dict, Any, List
from pydantic import BaseModel
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point

from .geometry import DB


# Нормы освещённости по типу помещения (пока не используем в логике,
# но константа уже есть и потом подцепим)
TARGET_LUX_BY_ROOM_TYPE = {
    "living": 150.0,
    "bedroom": 100.0,
    "kitchen": 200.0,
    # и т.д.
}


class FixturePlacement(BaseModel):
    room_id: str
    x: float
    y: float
    lumens: float
    power_w: float


class LightingRequest(BaseModel):
    project_id: str
    total_fixtures: int
    target_lux: float
    fixture_efficacy_lm_per_w: float
    maintenance_factor: float
    utilization_factor: float


class RoomLightingResult(BaseModel):
    area_m2: float
    fixtures: int
    lumens_total: float
    lumens_per_fixture: float
    power_w_per_fixture: float


class LightingResponse(BaseModel):
    project_id: str
    total_fixtures: int
    target_lux: float
    rooms: Dict[str, RoomLightingResult]
    fixtures: List[FixturePlacement]


def _extract_geom(room: Any) -> BaseGeometry:
    """
    Достаём shapely-геометрию из того, что хранится в DB["rooms"].
    Поддерживаем несколько форматов:
    - просто Polygon / MultiPolygon
    - dict с ключами "polygon" / "geom" / "geometry"
    """
    if isinstance(room, BaseGeometry):
        return room
    if isinstance(room, dict):
        g = room.get("polygon") or room.get("geom") or room.get("geometry")
        if isinstance(g, BaseGeometry):
            return g
    raise ValueError(f"Unsupported room structure: {room!r}")


def _room_area(room: Any) -> float:
    geom = _extract_geom(room)
    return float(geom.area)


def _distribute_fixtures_by_area(rooms: Dict[str, Any], total_fixtures: int) -> Dict[str, int]:
    # Считаем площади
    areas = {room_id: _room_area(room) for room_id, room in rooms.items()}
    total_area = sum(areas.values())

    # если почему-то площади нет — всем по нулям
    if total_area <= 0 or total_fixtures <= 0:
        return {room_id: 0 for room_id in rooms.keys()}

    # "сырое" распределение по пропорции площади
    raw = {room_id: area / total_area * total_fixtures for room_id, area in areas.items()}
    fixtures = {room_id: int(round(v)) for room_id, v in raw.items()}

    # Подгоняем, чтобы сумма точно была total_fixtures
    diff = total_fixtures - sum(fixtures.values())
    if diff != 0:
        # сортируем по площади, корректируем крупные комнаты
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
    Примитивное размещение светильников по сетке внутри комнаты:
    - берём bbox,
    - строим сетку точек,
    - выбираем те, что попали внутрь полигона,
    - обрезаем до нужного количества.
    """
    if count <= 0:
        return []

    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny

    if width <= 0 or height <= 0:
        return []

    # Пробуем сделать сетку примерно sqrt(count) x sqrt(count)
    import math

    n_side = max(1, int(math.ceil(math.sqrt(count))))
    # Немного больше точек, чтобы было из чего фильтровать
    nx, ny = n_side, n_side

    dx = width / (nx + 1)
    dy = height / (ny + 1)

    pts: List[Point] = []
    for ix in range(nx):
        for iy in range(ny):
            x = minx + (ix + 1) * dx
            y = miny + (iy + 1) * dy
            p = Point(x, y)
            # Немного "расширим" полигон на случай граничных эффектов
            if geom.buffer(1e-6).contains(p):
                pts.append(p)

    # Если точек меньше, чем нужно — просто возвращаем сколько есть
    # Если больше — берём первые count
    return pts[:count]


def design_lighting(req: LightingRequest) -> LightingResponse:
    # 1. Достаём комнаты проекта из in-memory DB
    rooms_store = DB["rooms"].get(req.project_id)
    if rooms_store is None:
        raise ValueError(f"Project '{req.project_id}' not found in DB['rooms']")

    # 2. Приводим к словарю {room_id -> room_obj}
    rooms: Dict[str, Any] = {}

    # вариант: уже dict
    if isinstance(rooms_store, dict):
        for key, val in rooms_store.items():
            room_id = str(key)
            rooms[room_id] = val

    # вариант: список (как у тебя с simple_apartment)
    elif isinstance(rooms_store, list):
        for i, val in enumerate(rooms_store):
            room_id = None

            if isinstance(val, dict):
                room_id = val.get("name") or val.get("id")

            if not room_id:
                room_id = f"room_{i}"

            rooms[str(room_id)] = val

    else:
        raise ValueError(f"Unsupported rooms container type: {type(rooms_store)}")

    # 3. Распределяем светильники по комнатам
    fixtures_per_room = _distribute_fixtures_by_area(rooms, req.total_fixtures)

    # 4. Считаем светотехнику по каждой комнате
    rooms_result: Dict[str, RoomLightingResult] = {}
    fixtures_result: List[FixturePlacement] = []

    for room_id, room in rooms.items():
        geom = _extract_geom(room)
        area = float(geom.area)

        # lumen = lux * m² / (MF * UF)
        total_lumens = req.target_lux * area / (req.maintenance_factor * req.utilization_factor)

        fixtures = fixtures_per_room.get(room_id, 0)
        if fixtures > 0:
            lumens_per_fixture = total_lumens / fixtures
            power_w_per_fixture = lumens_per_fixture / req.fixture_efficacy_lm_per_w
        else:
            lumens_per_fixture = 0.0
            power_w_per_fixture = 0.0

        rooms_result[room_id] = RoomLightingResult(
            area_m2=area,
            fixtures=fixtures,
            lumens_total=total_lumens,
            lumens_per_fixture=lumens_per_fixture,
            power_w_per_fixture=power_w_per_fixture,
        )

        # 5. Генерируем координаты светильников в этой комнате
        if fixtures > 0:
            pts = _grid_points_inside(geom, fixtures)
            for p in pts:
                fixtures_result.append(
                    FixturePlacement(
                        room_id=room_id,
                        x=float(p.x),
                        y=float(p.y),
                        lumens=lumens_per_fixture,
                        power_w=power_w_per_fixture,
                    )
                )

    # (опционально) сохраняем в DB для экспорта позже
    DB.setdefault("fixtures", {})[req.project_id] = fixtures_result

    return LightingResponse(
        project_id=req.project_id,
        total_fixtures=req.total_fixtures,
        target_lux=req.target_lux,
        rooms=rooms_result,
        fixtures=fixtures_result,
    )
