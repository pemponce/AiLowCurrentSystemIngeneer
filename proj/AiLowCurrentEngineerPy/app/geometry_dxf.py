from __future__ import annotations

"""DXF ingestion (MVP).

Цель: получить shapely-геометрию комнат (Polygon) и линий стен/дверей,
и сформировать PlanGraph для единого контракта.

Важно:
- DXF единицы зависят от файла (мм/см/м). На текущем шаге координаты берём
  "как есть".
- В PlanGraph мы явно ставим coordinateSystem.units.image = "cad", чтобы
  не путать с пикселями.
"""

from typing import Any, Dict, List, Optional, Tuple

import math
import ezdxf
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union

from app.contracts import (
    CoordinateSystem,
    Opening,
    PlanElements,
    PlanGraph,
    PlanSource,
    Room,
    Wall,
)
from app.geometry import set_geometry

# Ожидаемые слои (можно расширять)
ROOM_LAYERS = {"ROOM", "ROOMS", "ROOM_OUTLINE", "POLY_ROOM"}
WALL_LAYERS = {"WALL", "WALLS"}
DOOR_LAYERS = {"DOOR", "OPENING"}
WINDOW_LAYERS = {"WINDOW", "WIN"}


def _lwpoly_to_lines(entity) -> List[LineString]:
    # ezdxf LWPOLYLINE: get_points('xy') → [(x,y), ...]
    pts = list(entity.get_points("xy"))
    if len(pts) < 2:
        return []
    lines: List[LineString] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        lines.append(LineString([a, b]))
    if getattr(entity, "closed", False) and len(pts) >= 3:
        lines.append(LineString([pts[-1], pts[0]]))
    return lines


def _collect_lines(msp, layers: set[str]) -> List[LineString]:
    lines: List[LineString] = []

    # LINE
    for e in msp.query("LINE"):
        if e.dxf.layer.upper() not in layers:
            continue
        lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))

    # LWPOLYLINE
    for e in msp.query("LWPOLYLINE"):
        if e.dxf.layer.upper() not in layers:
            continue
        lines.extend(_lwpoly_to_lines(e))

    return lines


def _polygons_from_lines(lines: List[LineString]) -> List[Polygon]:
    if not lines:
        return []

    try:
        merged = linemerge(lines)
    except Exception:
        merged = unary_union(lines)

    polys: List[Polygon] = []
    for poly in polygonize(merged):
        try:
            p = Polygon(poly.exterior.coords)
            if p.is_valid and p.area > 1e-6:
                polys.append(p)
        except Exception:
            continue
    return polys


def dxf_to_rooms_walls(dxf_path: str) -> Tuple[List[Polygon], List[LineString], List[LineString], List[LineString]]:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    room_lines = _collect_lines(msp, ROOM_LAYERS)
    wall_lines = _collect_lines(msp, WALL_LAYERS)
    door_lines = _collect_lines(msp, DOOR_LAYERS)
    window_lines = _collect_lines(msp, WINDOW_LAYERS)

    rooms = _polygons_from_lines(room_lines)
    if not rooms:
        rooms = _polygons_from_lines(wall_lines)

    return rooms, wall_lines, door_lines, window_lines


def _bbox_from_geoms(rooms: List[Polygon], lines: List[LineString]) -> Tuple[int, int]:
    geoms: List[Any] = []
    geoms.extend(rooms)
    geoms.extend(lines)
    if not geoms:
        return 1, 1

    u = unary_union(geoms)
    minx, miny, maxx, maxy = u.bounds
    w = max(1, int(math.ceil(maxx - minx)))
    h = max(1, int(math.ceil(maxy - miny)))
    return w, h


def ingest_dxf(project_id: str, dxf_path: str, src_key: Optional[str] = None) -> Dict[str, Any]:
    rooms, walls, doors, windows = dxf_to_rooms_walls(dxf_path)

    w, h = _bbox_from_geoms(rooms, walls + doors + windows)

    room_models: List[Room] = []
    room_meta: List[Dict[str, Any]] = []
    for i, r in enumerate(rooms):
        rid = f"room_{i:03d}"
        coords = [(float(x), float(y)) for x, y in list(r.exterior.coords)[:-1]]
        area = float(r.area)
        room_models.append(Room(id=rid, polygon=coords, label=None, roomType=None, areaPx2=area, confidence=0.7))
        room_meta.append({"id": rid, "label": None, "roomType": None, "areaPx2": area, "confidence": 0.7})

    wall_models: List[Wall] = []
    for i, l in enumerate(walls):
        wall_models.append(
            Wall(
                id=f"w-{i:04d}",
                polyline=[(float(l.coords[0][0]), float(l.coords[0][1])), (float(l.coords[-1][0]), float(l.coords[-1][1]))],
                thicknessPx=None,
                type="unknown",
                confidence=0.7,
            )
        )

    opening_models: List[Opening] = []
    # В DXF двери/окна обычно задаются линиями/дугами; здесь MVP: линия → прямоугольник (buffer)
    def _openings_from_lines(lines: List[LineString], kind: str, start_idx: int) -> List[Opening]:
        out: List[Opening] = []
        for j, l in enumerate(lines):
            # небольшой буфер (в тех же единицах, что и DXF)
            buff = l.buffer(0.05, cap_style=2)
            poly = [(float(x), float(y)) for x, y in list(buff.exterior.coords)[:-1]]
            cx, cy = float(l.centroid.x), float(l.centroid.y)
            (x1, y1), (x2, y2) = l.coords[0], l.coords[-1]
            orientation = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            out.append(
                Opening(
                    id=f"{kind[0]}-{start_idx + j:04d}",
                    kind=kind,  # type: ignore[arg-type]
                    polygon=poly,
                    center=(cx, cy),
                    widthPx=float(l.length),
                    orientationDeg=float(orientation),
                    wallRef=None,
                    confidence=0.65,
                )
            )
        return out

    opening_models.extend(_openings_from_lines(doors, "door", 0))
    opening_models.extend(_openings_from_lines(windows, "window", len(doors)))

    plan_graph = PlanGraph(
        source=PlanSource(srcKey=src_key or "", imageWidth=w, imageHeight=h, dpi=None, scale=None),
        coordinateSystem=CoordinateSystem(origin="unknown", units={"image": "cad", "world": "m"}),
        elements=PlanElements(walls=wall_models, openings=opening_models, rooms=room_models),
        topology=None,
        artifacts=None,
    ).model_dump()

    set_geometry(
        project_id,
        plan_graph=plan_graph,
        rooms=rooms,
        room_meta=room_meta,
        walls=walls,
        doors=doors,
    )

    return {
        "rooms": len(rooms),
        "walls": len(walls),
        "doors": len(doors),
        "windows": len(windows),
        "note": "parsed from DXF (layers + polygonize); units are 'cad'",
    }
