# app/export_preview_png.py
from __future__ import annotations

from dataclasses import is_dataclass, asdict
from typing import Any, Iterable, List, Tuple, Optional

import cv2
import numpy as np


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _point_xy(p: Any) -> Tuple[float, float]:
    """
    Accepts:
      - shapely Point: has .x/.y
      - dict: {"x":..,"y":..} or {"pt":{...}} or {"point":{...}}
      - tuple/list: (x,y)
    """
    if p is None:
        return 0.0, 0.0

    # shapely Point-like
    if hasattr(p, "x") and hasattr(p, "y"):
        return _as_float(p.x), _as_float(p.y)

    # dataclass point-like
    if is_dataclass(p):
        d = asdict(p)
        if "x" in d and "y" in d:
            return _as_float(d["x"]), _as_float(d["y"])

    # dict point-like
    if isinstance(p, dict):
        if "x" in p and "y" in p:
            return _as_float(p.get("x")), _as_float(p.get("y"))
        if "pt" in p:
            return _point_xy(p["pt"])
        if "point" in p:
            return _point_xy(p["point"])
        if "p" in p:
            return _point_xy(p["p"])

    # tuple/list
    if isinstance(p, (tuple, list)) and len(p) >= 2:
        return _as_float(p[0]), _as_float(p[1])

    return 0.0, 0.0


def _poly_coords(room: Any) -> List[Tuple[float, float]]:
    """
    Accepts:
      - shapely Polygon: has .exterior.coords
      - dict with "polygon"/"poly"/"points"/"corners"/"coords"
      - list of points
    Returns list of (x,y) including closing point if present.
    """
    if room is None:
        return []

    # shapely Polygon-like
    if hasattr(room, "exterior") and hasattr(room.exterior, "coords"):
        pts = list(room.exterior.coords)
        return [(_as_float(x), _as_float(y)) for x, y in pts]

    if is_dataclass(room):
        room = asdict(room)

    if isinstance(room, dict):
        for k in ("polygon", "poly", "points", "corners", "coords"):
            if k in room and room[k] is not None:
                pts_any = room[k]
                if hasattr(pts_any, "exterior") and hasattr(pts_any.exterior, "coords"):
                    pts = list(pts_any.exterior.coords)
                    return [(_as_float(x), _as_float(y)) for x, y in pts]
                if isinstance(pts_any, (list, tuple)):
                    return [_point_xy(p) for p in pts_any]

        # some schemas: room["geometry"] -> {"points":[...]}
        if "geometry" in room and room["geometry"] is not None:
            return _poly_coords(room["geometry"])

    if isinstance(room, (list, tuple)):
        return [_point_xy(p) for p in room]

    return []


def _iter_devices(devices: Any) -> Iterable[Tuple[str, str, Tuple[float, float]]]:
    """
    Yields tuples: (device_type, room_id, (x,y))

    Accepts devices formats:
      - list of dicts: {"type": "...", "roomId":"...", "point": {...}}
      - list of tuples: ("SWITCH","room_000", Point)
      - list of dataclasses
    """
    if devices is None:
        return []
    if is_dataclass(devices):
        devices = asdict(devices)

    if isinstance(devices, dict):
        # e.g. {"items":[...]}
        for k in ("items", "devices", "data"):
            if k in devices:
                devices = devices[k]
                break

    out = []
    if isinstance(devices, (list, tuple)):
        for d in devices:
            if is_dataclass(d):
                d = asdict(d)

            if isinstance(d, tuple) and len(d) >= 3:
                dtype = str(d[0])
                room_id = str(d[1])
                pt = _point_xy(d[2])
                out.append((dtype, room_id, pt))
                continue

            if isinstance(d, dict):
                dtype = str(d.get("type") or d.get("deviceType") or d.get("kind") or "DEVICE")
                room_id = str(d.get("roomId") or d.get("room_id") or d.get("room") or "")
                pt = _point_xy(d.get("point") or d.get("pt") or d.get("p") or d.get("pos"))
                out.append((dtype, room_id, pt))
                continue

    return out


def _iter_routes(routes: Any) -> Iterable[List[Tuple[float, float]]]:
    """
    Yields polyline points for each route.
    Accepts:
      - list of shapely LineString
      - list of dicts: {"points":[...]} or {"polyline":[...]}
      - list of list-of-points
    """
    if routes is None:
        return []
    if is_dataclass(routes):
        routes = asdict(routes)

    if isinstance(routes, dict):
        for k in ("items", "routes", "data"):
            if k in routes:
                routes = routes[k]
                break

    out: List[List[Tuple[float, float]]] = []
    if isinstance(routes, (list, tuple)):
        for r in routes:
            if r is None:
                continue

            if hasattr(r, "coords"):
                pts = list(r.coords)
                out.append([(_as_float(x), _as_float(y)) for x, y in pts])
                continue

            if is_dataclass(r):
                r = asdict(r)

            if isinstance(r, dict):
                pts_any = r.get("points") or r.get("polyline") or r.get("coords")
                if pts_any is None and "line" in r:
                    pts_any = r["line"]
                if pts_any is not None:
                    out.append([_point_xy(p) for p in pts_any])
                continue

            if isinstance(r, (list, tuple)):
                out.append([_point_xy(p) for p in r])
                continue

    return out


def export_preview_png(
    base_image_path: str,
    rooms: Any,
    devices: Any,
    routes: Any,
    out_path: str,
    room_label_prefix: str = "room",
) -> str:
    """
    Draws overlay:
      - room polygons
      - device points and labels
      - routing polylines

    NOTE: Coordinates are assumed to be in the same coordinate system as the base image (pixels, origin top-left).
    """
    img = cv2.imread(base_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read base image: {base_image_path}")

    h, w = img.shape[:2]
    overlay = img.copy()

    # --- Rooms ---
    if rooms is None:
        rooms_list = []
    elif isinstance(rooms, dict) and "rooms" in rooms:
        rooms_list = rooms["rooms"]
    else:
        rooms_list = rooms if isinstance(rooms, (list, tuple)) else [rooms]

    for idx, room in enumerate(rooms_list):
        coords = _poly_coords(room)
        if len(coords) < 3:
            continue

        pts = np.array([[int(round(x)), int(round(y))] for x, y in coords], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # draw polygon outline
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # label at centroid-ish (mean)
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        cx = int(round(sum(xs) / max(len(xs), 1)))
        cy = int(round(sum(ys) / max(len(ys), 1)))

        rid = ""
        if isinstance(room, dict):
            rid = str(room.get("id") or room.get("roomId") or room.get("name") or "")
        if not rid:
            rid = f"{room_label_prefix}_{idx:03d}"

        cx = max(5, min(w - 5, cx))
        cy = max(15, min(h - 5, cy))
        cv2.putText(
            overlay,
            rid,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    # --- Routes ---
    for poly in _iter_routes(routes):
        if len(poly) < 2:
            continue
        pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=(255, 0, 255), thickness=2)

    # --- Devices ---
    for dtype, room_id, (x, y) in _iter_devices(devices):
        px = int(round(x))
        py = int(round(y))
        if px < 0 or py < 0 or px >= w or py >= h:
            continue

        cv2.circle(overlay, (px, py), 6, (0, 0, 255), -1)
        label = dtype if not room_id else f"{dtype}:{room_id}"
        cv2.putText(
            overlay,
            label,
            (min(w - 5, px + 10), min(h - 5, py + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # blend
    out = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise RuntimeError(f"Cannot write preview png: {out_path}")
    return out_path


def export_preview_canvas_png(
    *,
    rooms: Any,
    devices: Any,
    routes: Any,
    out_path: str,
    canvas_size: int = 1400,
    pad: int = 40,
    room_label_prefix: str = "room",
) -> str:
    """Рендерит preview на белом холсте, когда исходного изображения нет (например, DXF).

    Координаты могут быть в "world" единицах DXF. Мы приводим их к пикселям
    через affine-преобразование по bounds всех геометрий.
    """

    # --- Collect bounds ---
    pts_all: List[Tuple[float, float]] = []

    rooms_list = []
    if rooms is None:
        rooms_list = []
    elif isinstance(rooms, dict) and "rooms" in rooms:
        rooms_list = rooms["rooms"]
    else:
        rooms_list = rooms if isinstance(rooms, (list, tuple)) else [rooms]

    for room in rooms_list:
        coords = _poly_coords(room)
        for x, y in coords:
            pts_all.append((float(x), float(y)))

    for poly in _iter_routes(routes):
        for x, y in poly:
            pts_all.append((float(x), float(y)))

    for _, _, (x, y) in _iter_devices(devices):
        pts_all.append((float(x), float(y)))

    # empty -> blank canvas
    img = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    overlay = img.copy()

    if not pts_all:
        ok = cv2.imwrite(out_path, img)
        if not ok:
            raise RuntimeError(f"Cannot write preview png: {out_path}")
        return out_path

    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    usable = max(canvas_size - 2 * pad, 1)
    s = usable / max(w, h)

    def tx(x: float, y: float) -> Tuple[int, int]:
        X = (x - minx) * s + pad
        Y = (y - miny) * s + pad
        return int(round(X)), int(round(Y))

    # --- Rooms ---
    for idx, room in enumerate(rooms_list):
        coords = _poly_coords(room)
        if len(coords) < 3:
            continue
        pts = np.array([tx(x, y) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # label
        xs0 = [p[0] for p in coords]
        ys0 = [p[1] for p in coords]
        cxw = float(sum(xs0) / max(len(xs0), 1))
        cyw = float(sum(ys0) / max(len(ys0), 1))
        cx, cy = tx(cxw, cyw)

        rid = ""
        if isinstance(room, dict):
            rid = str(room.get("id") or room.get("roomId") or room.get("name") or "")
        if not rid:
            rid = f"{room_label_prefix}_{idx:03d}"

        cx = max(5, min(canvas_size - 5, cx))
        cy = max(15, min(canvas_size - 5, cy))
        cv2.putText(
            overlay,
            rid,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    # --- Routes ---
    for poly in _iter_routes(routes):
        if len(poly) < 2:
            continue
        pts = np.array([tx(x, y) for x, y in poly], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=(255, 0, 255), thickness=2)

    # --- Devices ---
    for dtype, room_id, (x, y) in _iter_devices(devices):
        px, py = tx(float(x), float(y))
        if px < 0 or py < 0 or px >= canvas_size or py >= canvas_size:
            continue
        cv2.circle(overlay, (px, py), 6, (0, 0, 255), -1)
        label = dtype if not room_id else f"{dtype}:{room_id}"
        cv2.putText(
            overlay,
            label,
            (min(canvas_size - 5, px + 10), min(canvas_size - 5, py + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    out = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise RuntimeError(f"Cannot write preview png: {out_path}")
    return out_path
