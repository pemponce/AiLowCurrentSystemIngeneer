from __future__ import annotations

from typing import List, Tuple, Optional, Any
import math

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union


def _as_int_pt(x: float, y: float) -> tuple[int, int]:
    return int(round(float(x))), int(round(float(y)))


def _draw_polygon(cv2, img, poly: Polygon, color: tuple[int, int, int], thickness: int = 2) -> None:
    coords = list(poly.exterior.coords)
    pts = np.array([_as_int_pt(x, y) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _draw_line(cv2, img, ls: LineString, color: tuple[int, int, int], thickness: int = 2) -> None:
    coords = list(ls.coords)
    pts = np.array([_as_int_pt(x, y) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def _make_canvas_for_cad(rooms: List[Polygon], routes: List[LineString], devices: List[Point]) -> tuple[Any, Any, float, float, float]:
    """
    Для DXF (CAD координаты): рисуем на белом холсте и применяем affine:
    (x,y) -> ((x-minx)*s + pad, (y-miny)*s + pad)

    Возвращает: (cv2, img, minx, miny, scale)
    """
    import cv2  # type: ignore

    geoms = []
    geoms.extend(rooms)
    geoms.extend(routes)
    geoms.extend(devices)
    u = unary_union([g for g in geoms if g is not None])
    minx, miny, maxx, maxy = u.bounds
    w = maxx - minx
    h = maxy - miny
    diag = math.hypot(w, h)

    canvas = 1400
    pad = 40
    usable = canvas - 2 * pad
    scale = usable / max(w, h, 1e-6)

    img = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    return cv2, img, float(minx), float(miny), float(scale)


def _tx(x: float, y: float, minx: float, miny: float, s: float, pad: int = 40) -> tuple[int, int]:
    X = (x - minx) * s + pad
    Y = (y - miny) * s + pad
    return _as_int_pt(X, Y)


def export_png(
    project_id: str,
    *,
    out_path: str,
    src_image_path: Optional[str],
    rooms: List[Polygon],
    devices_raw: List[tuple[str, int, Any]],
    routes_raw: List[tuple[str, LineString, float]],
    mode: str = "design",  # "parsed" | "design"
) -> str:
    """
    Делает PNG оверлей.
    - Если есть src_image_path (PNG/JPG) -> рисуем поверх оригинала в пиксельных координатах.
    - Если src_image_path None (DXF) -> рисуем на белом холсте с масштабированием.
    """
    import cv2  # type: ignore

    # Подготовим списки для рендера
    devices_pts: List[Point] = []
    for _, _, p in devices_raw:
        if isinstance(p, Point):
            devices_pts.append(p)
        else:
            try:
                devices_pts.append(Point(float(p[0]), float(p[1])))
            except Exception:
                continue

    routes_ls: List[LineString] = []
    for _, ls, _ in routes_raw:
        if isinstance(ls, LineString):
            routes_ls.append(ls)

    if src_image_path:
        img = cv2.imread(src_image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {src_image_path}")

        # rooms
        for r in rooms:
            _draw_polygon(cv2, img, r, color=(0, 200, 0), thickness=2)

        if mode == "design":
            # routes
            for ls in routes_ls:
                _draw_line(cv2, img, ls, color=(0, 180, 255), thickness=2)

            # devices
            for i, (dtype, room_idx, p) in enumerate(devices_raw):
                if not isinstance(p, Point):
                    try:
                        p = Point(float(p[0]), float(p[1]))
                    except Exception:
                        continue

                x, y = _as_int_pt(p.x, p.y)
                t = str(dtype).upper()

                if t == "SOCKET":
                    cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
                elif t == "SWITCH":
                    cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), -1)
                else:
                    cv2.circle(img, (x, y), 6, (0, 0, 0), 2)

        cv2.imwrite(out_path, img)
        return out_path

    # CAD fallback canvas
    cv2, img, minx, miny, s = _make_canvas_for_cad(rooms, routes_ls, devices_pts)

    # draw rooms
    for r in rooms:
        coords = list(r.exterior.coords)
        pts = np.array([_tx(x, y, minx, miny, s) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

    if mode == "design":
        # draw routes
        for ls in routes_ls:
            coords = list(ls.coords)
            pts = np.array([_tx(x, y, minx, miny, s) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=(0, 180, 255), thickness=2)

        # draw devices
        for dtype, _, p in devices_raw:
            if not isinstance(p, Point):
                try:
                    p = Point(float(p[0]), float(p[1]))
                except Exception:
                    continue
            x, y = _tx(p.x, p.y, minx, miny, s)
            t = str(dtype).upper()
            if t == "SOCKET":
                cv2.circle(img, (x, y), 6, (255, 0, 0), -1)
            elif t == "SWITCH":
                cv2.rectangle(img, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), -1)
            else:
                cv2.circle(img, (x, y), 6, (0, 0, 0), 2)

    cv2.imwrite(out_path, img)
    return out_path
