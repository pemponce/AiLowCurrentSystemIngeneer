from __future__ import annotations

import math
import os
import os.path as osp
import heapq
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, download_file


# ------------------------------------------------------------
# Routing (MVP)
# ------------------------------------------------------------
# Цель: построить трассы кабелей, которые НЕ проходят через стены.
# Используем wallsMask + (опционально) freeSpaceMask из structure_detect.
# Если масок нет или путь не найден — делаем безопасный fallback (прямая линия),
# чтобы сервис не падал.
# ------------------------------------------------------------

DEFAULT_PANEL_ANCHOR_PX = (30.0, 30.0)  # "щит" по умолчанию (в px, origin=top_left)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _coerce_polygon(room_like: Any) -> Optional[Polygon]:
    """Поддерживаем: shapely.Polygon, dict с polygon/polygonPx/points, list[(x,y)]."""
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

    # 1) Приоритет: plan_graph (если комнаты уже там)
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

    # 2) Fallback: DB["rooms"]
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


def _get_px_per_meter(project_id: str) -> Optional[float]:
    """Если известен масштаб (pxPerMeter) — вернём его."""
    pg = DB.get("plan_graph", {}).get(project_id)
    if not isinstance(pg, dict):
        return None
    src = pg.get("source") or {}
    scale = (src.get("scale") or {}) if isinstance(src, dict) else {}
    ppm = scale.get("pxPerMeter")
    try:
        v = float(ppm)
        return v if v > 0 else None
    except Exception:
        return None


def _download_mask_by_key(key: str, project_id: str, suffix: str) -> Optional[str]:
    local_dir = os.getenv("LOCAL_DOWNLOAD_DIR_ROUTING", "/tmp/routing")
    _ensure_dir(local_dir)
    local = osp.join(local_dir, f"{project_id}_{suffix}.png")
    if osp.exists(local):
        return local
    try:
        download_file(EXPORT_BUCKET, str(key), local)
        return local
    except Exception:
        return None


def _download_walls_mask(project_id: str) -> Optional[str]:
    pg = DB.get("plan_graph", {}).get(project_id)
    key = None
    if isinstance(pg, dict):
        key = ((pg.get("artifacts") or {}).get("masks") or {}).get("wallsMaskKey")

    if not key:
        st = DB.get("structure", {}).get(project_id)
        if isinstance(st, dict):
            key = st.get("walls_mask_key") or ((st.get("masks") or {}).get("wallsMaskKey"))

    if not key:
        return None
    return _download_mask_by_key(str(key), project_id, "wallsMask")


def _download_free_space_mask(project_id: str) -> Optional[str]:
    pg = DB.get("plan_graph", {}).get(project_id)
    key = None
    if isinstance(pg, dict):
        key = ((pg.get("artifacts") or {}).get("masks") or {}).get("freeSpaceMaskKey")

    if not key:
        st = DB.get("structure", {}).get(project_id)
        if isinstance(st, dict):
            # поддержим разные названия ключа (на случай старых итераций)
            key = (
                st.get("free_space_mask_key")
                or st.get("free_space_key")
                or ((st.get("masks") or {}).get("freeSpaceMaskKey"))
            )

    if not key:
        return None
    return _download_mask_by_key(str(key), project_id, "freeSpaceMask")


def _footprint_mask_from_walls(walls_mask: np.ndarray) -> Optional[np.ndarray]:
    """Пытаемся получить 'контур здания' как залитый внешний контур стен."""
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
    free_space_mask: Optional[np.ndarray],
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

    fp: Optional[np.ndarray] = None
    if free_space_mask is None:
        fp = _footprint_mask_from_walls(m)

    h, w = m.shape[:2]
    ds = max(2, int(downsample))
    nh = max(1, h // ds)
    nw = max(1, w // ds)

    m_small = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    blocked = (m_small > 0)

    if free_space_mask is not None:
        fs_small = cv2.resize(free_space_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        blocked = np.logical_or(blocked, fs_small == 0)
    elif fp is not None:
        fp_small = cv2.resize(fp, (nw, nh), interpolation=cv2.INTER_NEAREST)
        blocked = np.logical_or(blocked, fp_small == 0)

    return blocked.astype(bool), ds


def _astar(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* по 8-связной сетке. occ[y,x] == True => blocked."""
    h, w = occ.shape[:2]
    sx, sy = start
    gx, gy = goal

    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
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
            if not (0 <= nx < w and 0 <= ny < h):
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


def _normalize_devices(project_id: str) -> List[Tuple[str, str, Point]]:
    """
    Приводим DB["devices"][project_id] к единому формату:
      (device_type, room_id, Point(x,y))
    Поддерживаем:
      - tuple/list: (type, room_id, Point)
      - dict: {"type": "...", "roomId": "...", "x": ..., "y": ...}
    """
    raw = DB.get("devices", {}).get(project_id, [])
    out: List[Tuple[str, str, Point]] = []

    if not isinstance(raw, list):
        return out

    for d in raw:
        # tuple/list
        if isinstance(d, (tuple, list)) and len(d) >= 3:
            try:
                t = str(d[0])
                rid = str(d[1])
                p = d[2]
                if isinstance(p, Point):
                    out.append((t, rid, Point(float(p.x), float(p.y))))
                    continue
                if isinstance(p, (tuple, list)) and len(p) == 2:
                    out.append((t, rid, Point(float(p[0]), float(p[1]))))
                    continue
            except Exception:
                pass

        # dict
        if isinstance(d, dict):
            try:
                t = str(d.get("type") or d.get("device_type") or d.get("kind") or "DEVICE")
                rid = str(d.get("roomId") or d.get("room_id") or d.get("room") or "room_000")
                x = float(d.get("x"))
                y = float(d.get("y"))
                out.append((t, rid, Point(x, y)))
                continue
            except Exception:
                pass

    return out


def route_all(project_id: str):
    """Строим трассы от устройств до щита."""
    devices = _normalize_devices(project_id)
    if not devices:
        DB.setdefault("routes", {})[project_id] = []
        return []

    # openings (если есть)
    openings = []
    pg = DB.get("plan_graph", {}).get(project_id)
    if isinstance(pg, dict):
        openings = ((pg.get("elements") or {}).get("openings")) or []
    if not openings:
        st = DB.get("structure", {}).get(project_id)
        if isinstance(st, dict):
            openings = st.get("openings") or []

    rooms = _rooms_map(project_id)

    # masks -> occupancy grid
    walls_path = _download_walls_mask(project_id)
    free_path = _download_free_space_mask(project_id)

    # панель (щит) — пока просто якорь; позже привяжем к входной двери/тамбуру и т.п.
    panel = Point(float(DEFAULT_PANEL_ANCHOR_PX[0]), float(DEFAULT_PANEL_ANCHOR_PX[1]))

    if not walls_path or not osp.exists(walls_path):
        # Без маски: fallback — прямые линии, чтобы не падать
        routes = []
        px_per_m = _get_px_per_meter(project_id)
        for t, _, p in devices:
            ls = LineString([(float(p.x), float(p.y)), (panel.x, panel.y)])
            length = float(ls.length)
            if px_per_m:
                length = length / px_per_m
            routes.append((t, ls, length))
        DB.setdefault("routes", {})[project_id] = routes
        return routes

    walls = cv2.imread(walls_path, cv2.IMREAD_GRAYSCALE)
    if walls is None:
        routes = []
        px_per_m = _get_px_per_meter(project_id)
        for t, _, p in devices:
            ls = LineString([(float(p.x), float(p.y)), (panel.x, panel.y)])
            length = float(ls.length)
            if px_per_m:
                length = length / px_per_m
            routes.append((t, ls, length))
        DB.setdefault("routes", {})[project_id] = routes
        return routes

    free_space = None
    if free_path and osp.exists(free_path):
        free_space = cv2.imread(free_path, cv2.IMREAD_GRAYSCALE)

    # Несколько попыток (дискретизация/дилатация стен)
    attempts = [
        (8, 3),
        (8, 2),
        (6, 2),
        (6, 1),
        (4, 1),
        (4, 0),
    ]

    px_per_m = _get_px_per_meter(project_id)
    routes = []

    for t, _, p in devices:
        p = Point(float(p.x), float(p.y))

        best_ls: Optional[LineString] = None
        for ds, dil in attempts:
            occ, ds = _build_occupancy(walls, free_space, downsample=ds, dilate_px=dil)

            s0 = _point_to_cell(p, ds)
            g0 = _point_to_cell(panel, ds)

            s = _nearest_free_cell(occ, s0, max_r=35)
            g = _nearest_free_cell(occ, g0, max_r=60)
            if s is None or g is None:
                continue

            path = _astar(occ, s, g)
            if path is None or len(path) < 2:
                continue

            coords = [_cell_to_point(c, ds) for c in path]
            coords[0] = (float(p.x), float(p.y))
            coords[-1] = (float(panel.x), float(panel.y))

            ls = LineString(coords)
            if ls.length > 0:
                best_ls = ls
                break

        if best_ls is None:
            best_ls = LineString([(float(p.x), float(p.y)), (float(panel.x), float(panel.y))])

        length = float(best_ls.length)
        if px_per_m:
            length = length / px_per_m

        routes.append((t, best_ls, length))

    DB.setdefault("routes", {})[project_id] = routes
    return routes
