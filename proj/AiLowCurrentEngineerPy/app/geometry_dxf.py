from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from app.geometry import DB


# ============================================================
# DXF ingest (MVP)
# ============================================================
# Цель: прочитать DXF, достать линии/полилинии, попытаться
# восстановить комнаты (как полигоны) и стены (как набор сегментов),
# и положить это в DB в "легковесном" формате (dict + координаты).
#
# Важно: DXF планы бывают очень разными по слоям/масштабу, поэтому
# здесь стоят безопасные эвристики по умолчанию.


DEFAULT_ROOM_LAYERS = {
    "ROOM", "ROOMS", "A-ROOM", "A-AREA", "AREA", "Z-ROOM",
}
DEFAULT_WALL_LAYERS = {
    "WALL", "WALLS", "A-WALL", "A-WALLS", "Z-WALL",
}
DEFAULT_OPENING_LAYERS = {
    "DOOR", "DOORS", "A-DOOR", "WINDOW", "WINDOWS", "A-WINDOW",
}


def _ensure_project(project_id: str) -> None:
    """Гарантирует наличие секций в DB для project_id."""
    for key in ("rooms", "devices", "routes", "candidates", "structure", "plan_graph", "source_meta"):
        DB.setdefault(key, {})
        DB[key].setdefault(project_id, None)


def _as_xy_pairs(points: Sequence[Any]) -> List[Tuple[float, float]]:
    """Унифицирует список точек DXF -> [(x,y), ...]."""
    out: List[Tuple[float, float]] = []
    for p in points:
        # ezdxf LWPOLYLINE: p обычно tuple(x, y, ...) или (x,y)
        if isinstance(p, (tuple, list)) and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
        else:
            # vec2/vec3
            x = float(getattr(p, "x"))
            y = float(getattr(p, "y"))
            out.append((x, y))
    return out


def _lines_from_points(pts: List[Tuple[float, float]], closed: bool) -> List[LineString]:
    if len(pts) < 2:
        return []
    lines: List[LineString] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if a != b:
            lines.append(LineString([a, b]))
    if closed and pts[0] != pts[-1]:
        lines.append(LineString([pts[-1], pts[0]]))
    return lines


def _entity_to_lines(e: Any) -> List[LineString]:
    t = e.dxftype()
    if t == "LINE":
        a = e.dxf.start
        b = e.dxf.end
        return [LineString([(float(a.x), float(a.y)), (float(b.x), float(b.y))])]
    if t == "LWPOLYLINE":
        pts = _as_xy_pairs(list(e.get_points("xy")))
        return _lines_from_points(pts, closed=bool(getattr(e, "closed", False)))
    if t == "POLYLINE":
        verts = []
        for v in e.vertices():
            loc = v.dxf.location
            verts.append((float(loc.x), float(loc.y)))
        return _lines_from_points(verts, closed=bool(getattr(e, "is_closed", False)))
    return []


def _collect_lines(msp: Any, *, layers: Optional[set[str]] = None) -> List[LineString]:
    lines: List[LineString] = []
    for e in msp:
        layer = str(getattr(e.dxf, "layer", "")).upper()
        if layers is not None and layer not in layers:
            continue
        lines.extend(_entity_to_lines(e))
    return lines


def _polygon_to_room_dict(poly: Polygon, idx: int) -> Dict[str, Any]:
    coords = [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]
    return {
        "id": f"room_{idx}",
        "polygon": [[x, y] for x, y in coords],
        # DXF units могут быть мм/см/м — поэтому это "units^2"
        "area_units2": float(poly.area),
        "meta": {"source": "dxf_polygonize"},
    }


def _lines_to_wall_dicts(lines: List[LineString]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        coords = list(ln.coords)
        if len(coords) != 2:
            continue
        (x1, y1), (x2, y2) = coords[0], coords[1]
        out.append(
            {
                "id": f"wall_{i}",
                "a": {"x": float(x1), "y": float(y1)},
                "b": {"x": float(x2), "y": float(y2)},
            }
        )
    return out


def ingest_dxf(project_id: str, dxf_path: str) -> Dict[str, Any]:
    """
    Загружает DXF с диска, извлекает линии/полилинии, пытается
    восстановить комнаты и стены, и сохраняет в DB.

    Возвращает plan_graph (в стиле /ingest).
    """
    _ensure_project(project_id)

    # Lazy import — чтобы сервис мог стартовать даже без ezdxf.
    try:
        import ezdxf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DXF ingest is unavailable because dependency 'ezdxf' is not installed."
        ) from e

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # 1) Пытаемся взять "явные" слои
    room_lines = _collect_lines(msp, layers=DEFAULT_ROOM_LAYERS)
    wall_lines = _collect_lines(msp, layers=DEFAULT_WALL_LAYERS)

    # 2) Если слои не размечены — берём все линии
    if not room_lines and not wall_lines:
        wall_lines = _collect_lines(msp, layers=None)

    # 3) Комнаты через polygonize
    rooms: List[Dict[str, Any]] = []
    base = room_lines if room_lines else wall_lines

    if base:
        merged = unary_union(base)
        polys = list(polygonize(merged))

        if polys:
            minx, miny, maxx, maxy = merged.bounds
            bbox_area = max(1e-9, (maxx - minx) * (maxy - miny))

            filtered: List[Polygon] = []
            for p in polys:
                a = float(p.area)
                if a <= 0:
                    continue
                if a < 0.0005 * bbox_area:
                    continue
                if a > 0.95 * bbox_area:
                    continue
                filtered.append(p)

            polys_to_use = filtered if filtered else polys

            for i, p in enumerate(polys_to_use, start=1):
                rooms.append(_polygon_to_room_dict(p, i))

    walls_out = _lines_to_wall_dicts(wall_lines)

    # 4) plan_graph (минимальный)
    plan_graph: Dict[str, Any] = {
        "version": "plan-graph-0.1",
        "source": {
            "srcKey": None,
            "localPath": dxf_path,
            "imageWidth": None,
            "imageHeight": None,
            "dpi": None,
            "scale": {"pxPerMeter": None, "confidence": 0},
        },
        "coordinateSystem": {
            "origin": "dxf_world",
            "units": {"image": "px", "world": "m"},
        },
        "elements": {"walls": walls_out, "openings": [], "rooms": rooms},
        "topology": None,
        "artifacts": {"previewOverlayPngKey": None, "masks": None},
    }

    # 5) DB запись
    DB["rooms"][project_id] = rooms
    DB.setdefault("walls", {})
    DB["walls"][project_id] = walls_out
    DB["plan_graph"][project_id] = plan_graph
    DB["source_meta"][project_id] = {"localPath": dxf_path, "kind": "dxf"}

    return plan_graph


__all__ = ["ingest_dxf"]
