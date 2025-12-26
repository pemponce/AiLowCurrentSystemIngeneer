from __future__ import annotations

import os
import os.path as osp
import math
import heapq
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, download_file


# ------------------------------------------------------------
# Routing (MVP)
# ------------------------------------------------------------
#
# Цель: получить трассы кабелей, которые "не проходят через стены".
# Для этого используем маску стен (wallsMask) из structure_detect и
# строим путь по сетке (A*). Если маски нет или путь не найден —
# делаем безопасный fallback (прямая линия), чтобы сервис не падал.


# Якорь для щита по умолчанию (в пикселях, top-left origin)
DEFAULT_PANEL_ANCHOR_PX = (30.0, 30.0)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _coerce_polygon(room_like) -> Optional[Polygon]:
    """Поддерживаем: shapely.Polygon, dict с polygon/polygonPx, list[pts]."""
    if isinstance(room_like, Polygon):
        return room_like
    if isinstance(room_like, dict):
        pts = room_like.get("polygon") or room_like.get("polygonPx") or room_like.get("points")
        if not pts or not isinstance(pts, list) or len(pts) < 3:
            return None
        try:
            poly = Polygon([(float(x), float(y)) for x, y in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.is_valid and poly.area > 1.0 else None
        except Exception:
            return None
    if isinstance(room_like, list) and len(room_like) >= 3:
        try:
            poly = Polygon([(float(x), float(y)) for x, y in room_like])
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.is_valid and poly.area > 1.0 else None
        except Exception:
            return None
    return None


def _rooms_map(project_id: str) -> Dict[str, Polygon]:
    """room_id -> Polygon."""
    out: Dict[str, Polygon] = {}

    # Приоритет: plan_graph (там уже нормализованные комнаты)
    pg = DB.get("plan_graph", {}).get(project_id)
    if isinstance(pg, dict):
        rooms = (pg.get("elements") or {}).get("rooms") or []
        if isinstance(rooms, list):
            for i, r in enumerate(rooms):
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("id") or f"room_{i:03d}")
                poly = _coerce_polygon(r)
                if poly is not None:
                    out[rid] = poly

    if out:
        return out

    # Fallback: DB['rooms'] (может быть shapely или dict)
    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    if isinstance(rooms_raw, list):
        for i, r in enumerate(rooms_raw):
            rid = f"room_{i:03d}"
            if isinstance(r, dict) and r.get("id"):
                rid = str(r.get("id"))
            poly = _coerce_polygon(r)
            if poly is not None:
                out[rid] = poly
    return out


def _download_walls_mask(project_id: str) -> Optional[str]:
    """Возвращает локальный путь к walls mask (PNG), если она доступна."""
    pg = DB.get("plan_graph", {}).get(project_id)
    key = None
    if isinstance(pg, dict):
        key = ((pg.get("artifacts") or {}).get("masks") or {}).get("wallsMaskKey")
    if not key:
        st = DB.get("structure", {}).get(project_id)
        if isinstance(st, dict):
            key = st.get("walls_mask_key") or (st.get("masks") or {}).get("wallsMaskKey")
    if not key:
        return None

    local_dir = os.getenv("LOCAL_DOWNLOAD_DIR_ROUTING", "/tmp/routing")
    _ensure_dir(local_dir)
    local = osp.join(local_dir, f"{project_id}_wallsMask.png")

    if osp.exists(local):
        return local

    try:
        download_file(EXPORT_BUCKET, str(key), local)
        return local
    except Exception:
        return None


def _footprint_mask_from_walls(walls_mask: np.ndarray) -> Optional[np.ndarray]:
    """Пытаемся получить "внутренность здания" как залитый внешний контур."""
    cnts, _ = cv2.findContours(walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None
    fp = np.zeros_like(walls_mask)
    cv2.drawContours(fp, [cnt], -1, 255, thickness=-1)
    return fp


def _build_occupancy(
    walls_mask: np.ndarray,
    downsample: int,
    dilate_px: int,
) -> Tuple[np.ndarray, int]:
    """
    Возвращает occupancy grid (True = blocked) и downsample factor.
    """
    m = (walls_mask > 0).astype(np.uint8) * 255

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)

    fp = _footprint_mask_from_walls(m)
    outside = (fp == 0) if fp is not None else None

    h, w = m.shape[:2]
    ds = max(2, int(downsample))
    nh = max(1, h // ds)
    nw = max(1, w // ds)
    m_small = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    blocked = (m_small > 0)

    if outside is not None:
        fp_small = cv2.resize(fp, (nw, nh), interpolation=cv2.INTER_NEAREST)
        blocked = np.logical_or(blocked, fp_small == 0)

    return blocked.astype(bool), ds


def _astar(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* по 8-связной сетке. occ[y,x] == True => blocked."""
    h, w = occ.shape[:2]

    sx, sy = start
    gx, gy = goal
    if sx < 0 or sy < 0 or sx >= w or sy >= h:
        return None
    if gx < 0 or gy < 0 or gx >= w or gy >= h:
        return None
    if occ[sy, sx] or occ[gy, gx]:
        return None

    def heur(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    nbrs = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heur(start, goal), 0.0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            path = [cur]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        cx, cy = cur
        for dx, dy, cost in nbrs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if occ[ny, nx]:
                continue
            nxt = (nx, ny)
            ng = g + cost
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                came_from[nxt] = cur
                f = ng + heur(nxt, goal)
                heapq.heappush(open_heap, (f, ng, nxt))

    return None


def _nearest_free_cell(occ: np.ndarray, cell: Tuple[int, int], max_r: int = 25) -> Optional[Tuple[int, int]]:
    """Ищем ближайшую свободную клетку (BFS по кольцам)."""
    h, w = occ.shape[:2]
    x0, y0 = cell
    if 0 <= x0 < w and 0 <= y0 < h and not occ[y0, x0]:
        return (x0, y0)

    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < w and 0 <= y < h and not occ[y, x]:
                    return (x, y)
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < w and 0 <= y < h and not occ[y, x]:
                    return (x, y)
    return None


def _point_to_cell(p: Point, ds: int) -> Tuple[int, int]:
    return (int(round(p.x / ds)), int(round(p.y / ds)))


def _cell_to_point(cell: Tuple[int, int], ds: int) -> Tuple[float, float]:
    x, y = cell
    return (x * ds + ds * 0.5, y * ds + ds * 0.5)


def _choose_panel_point(
    rooms: Dict[str, Polygon],
    openings: List[dict],
    anchor_px: Tuple[float, float],
) -> Point:
    """
    1) Если есть дверь "наружу" (roomRefs <= 1) — ставим щит чуть внутри помещения рядом с этой дверью.
    2) Иначе — centroid первого помещения.
    3) Иначе — anchor.
    """
    best = None
    for o in openings or []:
        if (o.get("kind") or "") != "door":
            continue
        refs = o.get("roomRefs") or []
        if not isinstance(refs, list):
            refs = []
        if len(refs) > 1:
            continue
        seg = o.get("segmentPx") or o.get("segment")
        if not seg or not isinstance(seg, list) or len(seg) != 2:
            continue
        try:
            (x1, y1), (x2, y2) = seg
            mid = Point((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
        except Exception:
            continue

        rid = str(refs[0]) if refs else None
        if rid and rid in rooms:
            poly = rooms[rid]
            dx = float(x2) - float(x1)
            dy = float(y2) - float(y1)
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            off = max(0.02 * math.sqrt(abs(poly.area)), 8.0)
            p1 = Point(mid.x + nx * off, mid.y + ny * off)
            p2 = Point(mid.x - nx * off, mid.y - ny * off)
            if poly.contains(p1):
                best = p1
                break
            if poly.contains(p2):
                best = p2
                break

        if best is None:
            best = mid

    if best is not None:
        return best

    if rooms:
        first = next(iter(rooms.values()))
        c = first.centroid
        return Point(float(c.x), float(c.y))

    return Point(float(anchor_px[0]), float(anchor_px[1]))


def route_all(project_id: str):
    """Строим трассы от устройств до щита."""
    devices = DB.get("devices", {}).get(project_id, [])
    if not devices:
        DB.setdefault("routes", {})[project_id] = []
        return []

    openings = []
    pg = DB.get("plan_graph", {}).get(project_id)
    if isinstance(pg, dict):
        openings = ((pg.get("elements") or {}).get("openings")) or []
    if not openings:
        st = DB.get("structure", {}).get(project_id)
        if isinstance(st, dict):
            openings = st.get("openings") or []

    rooms = _rooms_map(project_id)

    mask_path = _download_walls_mask(project_id)
    if not mask_path or not osp.exists(mask_path):
        panel = Point(*DEFAULT_PANEL_ANCHOR_PX)
        routes = []
        for t, _, p in devices:
            ls = LineString([(float(p.x), float(p.y)), (panel.x, panel.y)])
            routes.append((t, ls, float(ls.length)))
        DB.setdefault("routes", {})[project_id] = routes
        return routes

    walls = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if walls is None:
        panel = Point(*DEFAULT_PANEL_ANCHOR_PX)
        routes = []
        for t, _, p in devices:
            ls = LineString([(float(p.x), float(p.y)), (panel.x, panel.y)])
            routes.append((t, ls, float(ls.length)))
        DB.setdefault("routes", {})[project_id] = routes
        return routes

    panel_point_guess = _choose_panel_point(rooms, openings, DEFAULT_PANEL_ANCHOR_PX)

    attempts = [
        (8, 3),
        (8, 2),
        (6, 2),
        (6, 1),
        (4, 1),
        (4, 0),
    ]

    routes = []
    for t, _, p in devices:
        p = Point(float(p.x), float(p.y))

        best_ls: Optional[LineString] = None
        for ds, dil in attempts:
            occ, ds = _build_occupancy(walls, downsample=ds, dilate_px=dil)

            s0 = _point_to_cell(p, ds)
            g0 = _point_to_cell(panel_point_guess, ds)

            s = _nearest_free_cell(occ, s0, max_r=35)
            g = _nearest_free_cell(occ, g0, max_r=60)
            if s is None or g is None:
                continue

            path = _astar(occ, s, g)
            if path is None or len(path) < 2:
                continue

            coords = [_cell_to_point(c, ds) for c in path]
            coords[0] = (float(p.x), float(p.y))
            coords[-1] = (float(panel_point_guess.x), float(panel_point_guess.y))

            ls = LineString(coords)
            if ls.length > 0:
                best_ls = ls
                break

        if best_ls is None:
            best_ls = LineString([(float(p.x), float(p.y)), (float(panel_point_guess.x), float(panel_point_guess.y))])

        routes.append((t, best_ls, float(best_ls.length)))

    DB.setdefault("routes", {})[project_id] = routes
    return routes
