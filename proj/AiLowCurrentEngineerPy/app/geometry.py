from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from shapely.geometry import LineString, Polygon, shape


DB: Dict[str, Any] = {
    "plan_graph": {},      # project_id -> dict|None
    "rooms": {},           # project_id -> List[Polygon]
    "walls": {},           # project_id -> List[LineString]
    "doors": {},           # project_id -> List[LineString]
    "room_meta": {},       # project_id -> List[dict]
    "devices": {},         # project_id -> List[(type, room_idx, shapely Point)]
    "routes": {},          # project_id -> List[(type, shapely LineString, length)]
    "candidates": {},      # project_id -> List[(type, room_idx, shapely Point)]
    "source_meta": {},     # project_id -> {"type": "png|dxf", "local_path": str|None, "src_key": str|None}
}


def reset_project(project_id: str) -> None:
    DB.setdefault("plan_graph", {}).setdefault(project_id, None)
    DB.setdefault("rooms", {}).setdefault(project_id, [])
    DB.setdefault("walls", {}).setdefault(project_id, [])
    DB.setdefault("doors", {}).setdefault(project_id, [])
    DB.setdefault("room_meta", {}).setdefault(project_id, [])
    DB.setdefault("devices", {}).setdefault(project_id, [])
    DB.setdefault("routes", {}).setdefault(project_id, [])
    DB.setdefault("candidates", {}).setdefault(project_id, [])
    DB.setdefault("source_meta", {}).setdefault(project_id, {"type": None, "local_path": None, "src_key": None})


def set_source_meta(project_id: str, *, file_type: str, local_path: Optional[str], src_key: Optional[str]) -> None:
    reset_project(project_id)
    DB["source_meta"][project_id] = {"type": file_type, "local_path": local_path, "src_key": src_key}


def coerce_polygon(obj: Any) -> Polygon:
    if isinstance(obj, Polygon):
        return obj
    if isinstance(obj, dict):
        if "polygon" in obj and isinstance(obj["polygon"], (list, tuple)):
            return Polygon(obj["polygon"])
        if "geometry" in obj:
            return shape(obj["geometry"])
        if "type" in obj and obj.get("type") in ("Polygon", "MultiPolygon", "GeometryCollection"):
            return shape(obj)
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return Polygon(obj)
    raise TypeError(f"Cannot coerce to Polygon: {type(obj)}")


def coerce_linestring(obj: Any) -> LineString:
    if isinstance(obj, LineString):
        return obj
    if isinstance(obj, dict):
        if "polyline" in obj and isinstance(obj["polyline"], (list, tuple)):
            return LineString(obj["polyline"])
        if "geometry" in obj:
            return shape(obj["geometry"])
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return LineString(obj)
    raise TypeError(f"Cannot coerce to LineString: {type(obj)}")


def normalize_project_geometry(project_id: str) -> None:
    reset_project(project_id)

    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    if rooms_raw and not isinstance(rooms_raw[0], Polygon):
        DB["rooms"][project_id] = [coerce_polygon(r) for r in rooms_raw]

    walls_raw = DB.get("walls", {}).get(project_id, [])
    if walls_raw and not isinstance(walls_raw[0], LineString):
        DB["walls"][project_id] = [coerce_linestring(w) for w in walls_raw]

    doors_raw = DB.get("doors", {}).get(project_id, [])
    if doors_raw and not isinstance(doors_raw[0], LineString):
        DB["doors"][project_id] = [coerce_linestring(d) for d in doors_raw]


def set_geometry(
    project_id: str,
    *,
    plan_graph: Optional[Dict[str, Any]],
    rooms: List[Any],
    room_meta: Optional[List[Dict[str, Any]]] = None,
    walls: Optional[List[Any]] = None,
    doors: Optional[List[Any]] = None,
) -> None:
    reset_project(project_id)

    DB["plan_graph"][project_id] = plan_graph
    DB["rooms"][project_id] = [coerce_polygon(r) for r in (rooms or [])]
    DB["room_meta"][project_id] = room_meta or []
    DB["walls"][project_id] = [coerce_linestring(w) for w in (walls or [])]
    DB["doors"][project_id] = [coerce_linestring(d) for d in (doors or [])]

    DB["devices"][project_id] = []
    DB["routes"][project_id] = []
    DB["candidates"][project_id] = []


def load_sample_rooms(project_id: str, path: str = "../../samples/geojson/simple_apartment.geojson"):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    rooms = [shape(feat["geometry"]) for feat in gj["features"]]
    set_geometry(project_id, plan_graph=None, rooms=rooms, room_meta=[])
    return rooms


def room_walls(room: Any) -> List[LineString]:
    poly = coerce_polygon(room)
    xs, ys = poly.exterior.coords.xy
    return [
        LineString([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])
        for i in range(len(xs) - 1)
    ]


def along_wall_points(wall: LineString, step: float = 1.5, offsets: float = 0.2):
    L = wall.length
    t = offsets
    pts = []
    while t < L - offsets:
        pts.append(wall.interpolate(t))
        t += step
    return pts
