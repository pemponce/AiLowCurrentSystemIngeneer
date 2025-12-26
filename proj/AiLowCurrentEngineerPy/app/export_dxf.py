# app/export_dxf.py
"""
DXF export for AiLowCurrentEngineerPy.

Note about /detect-structure (structure_detect):
- Endpoint /detect-structure runs "structure detection" over the uploaded plan.
- Its job is to extract higher-level geometry from the raster plan (walls/rooms/openings),
  and persist a normalized representation into the project DB (e.g., rooms as polygons in pixel coords),
  so later stages (device placement, routing, exports) can work with stable geometry.

This module exports:
- Rooms (polylines)
- Routes (polylines)
- Devices (circles + optional text labels)

Compatibility:
- Works with different internal representations:
  * rooms as shapely Polygon OR dict like {"polygonPx": [[x,y],...]} / {"polygon": ...}
  * routes as shapely LineString OR dict with points OR tuples like (type, LineString, length)
  * devices as shapely Point OR dict {"x","y"} OR tuples like (type, meta, Point)

Also avoids ezdxf Text.set_pos() (not available in some ezdxf versions) by setting dxf.insert.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import ezdxf
from shapely.geometry import LineString, Point, Polygon


LABEL_BY_DEVICE_TYPE: Dict[str, str] = {
    "PANEL": "Щит",
    "LIGHT": "Светильник",
    "SWITCH": "Выключатель",
    "SOCKET": "Розетка",
    "SENSOR": "Датчик",
}


def _to_point(obj: Any) -> Optional[Point]:
    """Coerce to shapely Point from dict/tuple/list/Point."""
    if obj is None:
        return None
    if isinstance(obj, Point):
        return obj
    if isinstance(obj, dict):
        x = obj.get("x", obj.get("X", None))
        y = obj.get("y", obj.get("Y", None))
        if x is None or y is None:
            return None
        return Point(float(x), float(y))
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        try:
            return Point(float(obj[0]), float(obj[1]))
        except Exception:
            return None
    return None


def _to_linestring(obj: Any) -> Optional[LineString]:
    """Coerce to shapely LineString from LineString/dict/list."""
    if obj is None:
        return None
    if isinstance(obj, LineString):
        return obj
    if isinstance(obj, dict):
        pts = obj.get("points") or obj.get("path") or obj.get("polyline") or obj.get("coords")
        if pts is None:
            geom = obj.get("geometry")
            if isinstance(geom, LineString):
                return geom
            pts = geom
        if isinstance(pts, (list, tuple)) and len(pts) >= 2:
            coords: List[Tuple[float, float]] = []
            for p in pts:
                pp = _to_point(p)
                if pp is not None:
                    coords.append((float(pp.x), float(pp.y)))
            if len(coords) >= 2:
                return LineString(coords)
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        coords2: List[Tuple[float, float]] = []
        for p in obj:
            pp = _to_point(p)
            if pp is not None:
                coords2.append((float(pp.x), float(pp.y)))
        if len(coords2) >= 2:
            return LineString(coords2)
    return None


def _to_polygon(obj: Any) -> Optional[Polygon]:
    """Coerce to shapely Polygon from Polygon/dict/list."""
    if obj is None:
        return None
    if isinstance(obj, Polygon):
        return obj
    if isinstance(obj, dict):
        pts = obj.get("polygonPx") or obj.get("polygon") or obj.get("points") or obj.get("coords")
        if pts is None:
            geom = obj.get("geometry")
            if isinstance(geom, Polygon):
                return geom
            pts = geom
        if isinstance(pts, (list, tuple)) and len(pts) >= 3:
            coords: List[Tuple[float, float]] = []
            for p in pts:
                pp = _to_point(p)
                if pp is None:
                    continue
                coords.append((float(pp.x), float(pp.y)))
            if len(coords) >= 3:
                # Ensure closed ring
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                try:
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_empty:
                        return None
                    return poly
                except Exception:
                    return None
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        coords2: List[Tuple[float, float]] = []
        for p in obj:
            pp = _to_point(p)
            if pp is not None:
                coords2.append((float(pp.x), float(pp.y)))
        if len(coords2) >= 3:
            if coords2[0] != coords2[-1]:
                coords2.append(coords2[0])
            try:
                poly2 = Polygon(coords2)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                if poly2.is_empty:
                    return None
                return poly2
            except Exception:
                return None
    return None


def _normalize_rooms(rooms: Sequence[Any]) -> List[Polygon]:
    out: List[Polygon] = []
    for r in rooms or []:
        poly = _to_polygon(r)
        if poly is not None and not poly.is_empty:
            out.append(poly)
    return out


def _normalize_routes(routes: Sequence[Any]) -> List[Tuple[str, LineString]]:
    """
    Returns list of (route_type, LineString).
    Accepts:
      - LineString
      - dict with points/path/coords
      - tuple like (type, LineString, length) OR (type, meta, LineString, ...)
    """
    out: List[Tuple[str, LineString]] = []
    for r in routes or []:
        if isinstance(r, LineString):
            out.append(("ROUTE", r))
            continue
        if isinstance(r, dict):
            t = str(r.get("type", "ROUTE"))
            ls = _to_linestring(r)
            if ls is not None:
                out.append((t, ls))
            continue
        if isinstance(r, (list, tuple)) and len(r) >= 2:
            # common: (t, LineString, length)
            t = str(r[0])
            ls = None
            for item in r[1:]:
                ls = _to_linestring(item)
                if ls is not None:
                    break
            if ls is not None:
                out.append((t, ls))
            continue
    return out


def _normalize_devices(devices: Sequence[Any]) -> List[Tuple[str, Point]]:
    """
    Returns list of (device_type, Point).
    Accepts:
      - Point
      - dict with x,y and optional type
      - tuple like (type, meta, Point)
      - tuple like (type, Point)
    """
    out: List[Tuple[str, Point]] = []
    for d in devices or []:
        if isinstance(d, Point):
            out.append(("DEVICE", d))
            continue
        if isinstance(d, dict):
            t = str(d.get("type", "DEVICE"))
            pt = _to_point(d)
            if pt is not None:
                out.append((t, pt))
            continue
        if isinstance(d, (list, tuple)) and len(d) >= 2:
            t = str(d[0])
            pt = None
            for item in d[1:]:
                pt = _to_point(item)
                if pt is not None:
                    break
            if pt is not None:
                out.append((t, pt))
            continue
    return out


def _bbox_from_geometry(rooms: List[Polygon], routes: List[Tuple[str, LineString]], devs: List[Tuple[str, Point]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []

    for poly in rooms:
        for x, y in list(poly.exterior.coords):
            xs.append(float(x))
            ys.append(float(y))

    for _, ls in routes:
        for x, y in list(ls.coords):
            xs.append(float(x))
            ys.append(float(y))

    for _, pt in devs:
        xs.append(float(pt.x))
        ys.append(float(pt.y))

    if not xs or not ys:
        return 0.0, 0.0, 1.0, 1.0

    return min(xs), min(ys), max(xs), max(ys)


def _is_pixel_space(rooms: List[Polygon], routes: List[Tuple[str, LineString]], devs: List[Tuple[str, Point]]) -> bool:
    """
    Heuristic: if bbox is large (hundreds/thousands), assume pixel coords.
    """
    minx, miny, maxx, maxy = _bbox_from_geometry(rooms, routes, devs)
    w = maxx - minx
    h = maxy - miny
    return max(w, h) > 50.0


def export_dxf(
    project_id: str,
    rooms: Sequence[Any],
    devices: Sequence[Any],
    routes: Sequence[Any],
    out_path: str,
) -> str:
    rooms_poly = _normalize_rooms(rooms or [])
    routes_ls = _normalize_routes(routes or [])
    devs_pt = _normalize_devices(devices or [])

    minx, miny, maxx, maxy = _bbox_from_geometry(rooms_poly, routes_ls, devs_pt)

    def tr(x: float, y: float) -> Tuple[float, float]:
        # translate to positive space
        return float(x - minx), float(y - miny)

    px_space = _is_pixel_space(rooms_poly, routes_ls, devs_pt)

    # Device sizes in drawing units
    device_radius = 8.0 if px_space else 0.08
    text_height = 10.0 if px_space else 0.10
    text_offset = 10.0 if px_space else 0.10

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Layers (safe even if repeated)
    for name in ["ROOMS", "ROUTES", "DEVICES", "DEV_TEXT"]:
        if name not in doc.layers:
            doc.layers.new(name=name)

    # Rooms
    for poly in rooms_poly:
        pts = [tr(float(x), float(y)) for x, y in list(poly.exterior.coords)]
        # Use LWPOLYLINE if available, otherwise POLYLINE
        try:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "ROOMS", "closed": True})
        except Exception:
            pl = msp.add_polyline2d(pts, dxfattribs={"layer": "ROOMS"})
            pl.close(True)

    # Routes
    for rtype, ls in routes_ls:
        pts = [tr(float(x), float(y)) for x, y in list(ls.coords)]
        try:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "ROUTES"})
        except Exception:
            msp.add_polyline2d(pts, dxfattribs={"layer": "ROUTES"})

    # Devices (circle + label)
    for t, pt in devs_pt:
        x, y = tr(float(pt.x), float(pt.y))
        msp.add_circle((x, y), device_radius, dxfattribs={"layer": "DEVICES"})

        label = LABEL_BY_DEVICE_TYPE.get(t, t)
        txt = msp.add_text(label, dxfattribs={"layer": "DEV_TEXT", "height": text_height})
        # DO NOT use .set_pos() (missing in some ezdxf versions)
        txt.dxf.insert = (x + text_offset, y + text_offset)

    doc.saveas(out_path)
    return out_path
