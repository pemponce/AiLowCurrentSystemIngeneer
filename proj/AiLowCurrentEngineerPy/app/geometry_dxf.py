from shapely.geometry import Polygon, LineString, Point
from shapely.ops import linemerge, polygonize
import ezdxf
from app.geometry import DB
from typing import List, Tuple

# Ожидаем: контуры помещений на слоях ROOM_* или POLY_ROOM
ROOM_LAYERS = {"ROOM", "ROOMS", "ROOM_OUTLINE", "POLY_ROOM"}
WALL_LAYERS = {"WALL", "WALLS"}
DOOR_LAYERS = {"DOOR", "OPENING"}


def _lwpoly_to_lines(entity) -> List[LineString]:
    pts = list(entity.get_points('xy'))
    lines = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
    lines.append(LineString([a, b]))
    return lines


def _collect_lines(msp, layers: set) -> List[LineString]:
    lines = []
    for e in msp.query("LINE,LWPOLYLINE"):  # можно добавить SPLINE/POLYLINE при необходимости
        if e.dxf.layer.upper() in layers:
            if e.dxftype() == 'LINE':
                lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
    else:
        lines.extend(_lwpoly_to_lines(e))
    return lines


def dxf_to_rooms_walls(dxf_path: str) -> Tuple[List[Polygon], List[LineString], List[LineString]]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    room_lines = _collect_lines(msp, ROOM_LAYERS)
    wall_lines = _collect_lines(msp, WALL_LAYERS)
    door_lines = _collect_lines(msp, DOOR_LAYERS)

    rooms = []
    if room_lines:
        merged = linemerge(room_lines)
    # polygonize ожидает замкнутый граф
    for poly in polygonize(merged):
        if poly.area > 0.05:  # отсечём мусор
            rooms.append(Polygon(poly.exterior.coords))

    # Если явных комнат нет, попробуем по стенам (вторичный путь)
    if not rooms and wall_lines:
        merged = linemerge(wall_lines)
    for poly in polygonize(merged):
        if poly.area > 0.05:
            rooms.append(Polygon(poly.exterior.coords))

    return rooms, wall_lines, door_lines


def ingest_dxf(project_id: str, dxf_path: str):
    rooms, walls, doors = dxf_to_rooms_walls(dxf_path)
    DB['rooms'][project_id] = rooms
    DB['walls'] = DB.get('walls', {})
    DB['doors'] = DB.get('doors', {})
    DB['walls'][project_id] = walls
    DB['doors'][project_id] = doors
    return {"rooms": len(rooms), "walls": len(walls), "doors": len(doors)}
