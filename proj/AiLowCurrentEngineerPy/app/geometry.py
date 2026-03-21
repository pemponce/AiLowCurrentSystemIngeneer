from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from shapely.geometry import LineString, Polygon, shape


# In-memory DB (MVP)
DB: Dict[str, Dict[str, Any]] = {
    "rooms": {},
    "devices": {},
    "routes": {},
    "candidates": {},
    "structure": {},
    "plan_graph": {},
    "source_meta": {},
    # иногда используются в DXF импорте/роутинге:
    "walls": {},
    "doors": {},
}


def ensure_project(project_id: str) -> None:
    DB.setdefault("rooms", {}).setdefault(project_id, [])
    DB.setdefault("devices", {}).setdefault(project_id, [])
    DB.setdefault("routes", {}).setdefault(project_id, [])
    DB.setdefault("candidates", {}).setdefault(project_id, [])
    DB.setdefault("structure", {}).setdefault(project_id, {})
    DB.setdefault("plan_graph", {}).setdefault(project_id, None)
    DB.setdefault("source_meta", {}).setdefault(project_id, {})
    DB.setdefault("walls", {}).setdefault(project_id, [])
    DB.setdefault("doors", {}).setdefault(project_id, [])


def load_sample_rooms(project_id: str, path: str = "../../samples/geojson/simple_apartment.geojson"):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    rooms = [shape(feat["geometry"]) for feat in gj["features"]]
    ensure_project(project_id)
    DB["rooms"][project_id] = rooms
    return rooms


def coerce_polygon(obj: Any) -> Optional[Polygon]:
    """
    Приводит разные представления комнаты к shapely.Polygon.

    Поддерживаем:
      - Polygon
      - GeoJSON dict {"type":"Polygon","coordinates":[...]}
      - dict с polygon / polygonPx / points: list[[x,y],...]
      - list[[x,y],...]
    """
    if obj is None:
        return None

    poly: Optional[Polygon] = None

    if isinstance(obj, Polygon):
        poly = obj
    elif isinstance(obj, dict):
        if "type" in obj and "coordinates" in obj:
            try:
                poly = shape(obj)
            except Exception:
                return None
        else:
            pts = obj.get("polygon") or obj.get("polygonPx") or obj.get("points")
            if not pts or not isinstance(pts, list) or len(pts) < 3:
                return None
            try:
                poly = Polygon([(float(p[0]), float(p[1])) for p in pts])
            except Exception:
                return None
    elif isinstance(obj, list):
        if len(obj) < 3:
            return None
        try:
            poly = Polygon([(float(p[0]), float(p[1])) for p in obj])
        except Exception:
            return None
    else:
        return None

    if poly is None or poly.is_empty:
        return None

    # чинит самопересечения
    try:
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return None

    if poly.is_empty or poly.area <= 0:
        return None

    return poly


def normalize_project_geometry(project_id: str) -> List[Polygon]:
    """
    Нормализует DB['rooms'][project_id] к List[Polygon] и кладёт обратно.
    Нужно, потому что PNG-ingest может хранить комнаты как dict.
    """
    ensure_project(project_id)
    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    if not rooms_raw:
        DB["rooms"][project_id] = []
        return []

    normalized: List[Polygon] = []
    for r in rooms_raw:
        poly = r if isinstance(r, Polygon) else coerce_polygon(r)
        if poly is not None:
            normalized.append(poly)

    DB["rooms"][project_id] = normalized
    return normalized


def set_geometry(
    project_id: str,
    rooms: Any = None,
    walls: Any = None,
    doors: Any = None,
    source_meta: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, int]:
    """
    Универсальный setter, чтобы разные модули (DXF/PNG/инжестеры) могли
    задавать геометрию через единый контракт.
    """
    ensure_project(project_id)

    if rooms is not None:
        DB["rooms"][project_id] = rooms
        normalize_project_geometry(project_id)

    if walls is not None:
        DB.setdefault("walls", {})[project_id] = walls

    if doors is not None:
        DB.setdefault("doors", {})[project_id] = doors

    if source_meta is not None:
        DB.setdefault("source_meta", {}).setdefault(project_id, {})
        DB["source_meta"][project_id].update(source_meta)

    for k, v in extra.items():
        if k is None:
            continue
        DB.setdefault(k, {})
        try:
            DB[k][project_id] = v
        except Exception:
            pass

    return {
        "rooms": len(DB.get("rooms", {}).get(project_id, [])),
        "walls": len(DB.get("walls", {}).get(project_id, [])),
        "doors": len(DB.get("doors", {}).get(project_id, [])),
    }


def room_walls(room: Union[Polygon, dict, list]) -> List[LineString]:
    """
    Возвращает стены комнаты как список LineString по внешнему контуру.
    """
    poly = room if isinstance(room, Polygon) else coerce_polygon(room)
    if poly is None:
        return []
    xs, ys = poly.exterior.coords.xy
    return [
        LineString([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])
        for i in range(len(xs) - 1)
    ]


def detect_doorways(rooms: list, tolerance: float = 18.0) -> List[Dict]:
    """
    Вычисляет дверные проёмы как общие участки границ смежных комнат.

    Алгоритм:
    - Для каждой пары комнат строим Shapely Polygon
    - Если расстояние между границами < tolerance px — это общая стена
    - Находим ближайшие точки двух контуров → середина = центр проёма
    - Возвращает список {"room_a", "room_b", "cx", "cy", "dist"}

    Используется для SWI-выключателей когда NN-1 не возвращает doors.
    """
    from shapely.geometry import Polygon as ShPoly
    from shapely.ops import nearest_points

    doorways = []
    n = len(rooms)
    for i in range(n):
        for j in range(i + 1, n):
            ra = rooms[i]
            rb = rooms[j]
            poly_a = ra.get("polygonPx") or []
            poly_b = rb.get("polygonPx") or []
            if len(poly_a) < 3 or len(poly_b) < 3:
                continue
            try:
                sha = ShPoly([(float(p[0]), float(p[1])) for p in poly_a])
                shb = ShPoly([(float(p[0]), float(p[1])) for p in poly_b])
                if not sha.is_valid:
                    sha = sha.buffer(0)
                if not shb.is_valid:
                    shb = shb.buffer(0)
                dist = sha.distance(shb)
                if dist > tolerance:
                    continue
                pa, pb = nearest_points(sha.exterior, shb.exterior)
                cx = (pa.x + pb.x) / 2
                cy = (pa.y + pb.y) / 2
                rid_a = ra.get("id") or ra.get("room_id") or f"room_{i:03d}"
                rid_b = rb.get("id") or rb.get("room_id") or f"room_{j:03d}"
                doorways.append({
                    "room_a": rid_a,
                    "room_b": rid_b,
                    "cx": cx,
                    "cy": cy,
                    "dist": dist,
                })
            except Exception:
                continue
    return doorways


def along_wall_points(wall: LineString, step: float = 1.5, offsets: float = 0.2):
    """
    Генерирует точки вдоль стены через step,
    с отступом от углов offsets.
    """
    L = float(wall.length)
    t = float(offsets)
    pts = []
    while t < L - offsets:
        pts.append(wall.interpolate(t))
        t += step
    return pts