from __future__ import annotations

import math
import os
import os.path as osp
import heapq
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, download_file

logger = logging.getLogger("planner")

# ─────────────────────────────────────────────────────────────────────────────
# ГРУППЫ ЦЕПЕЙ
# ─────────────────────────────────────────────────────────────────────────────

DEVICE_GROUP: Dict[str, str] = {
    "ceiling_lights":   "lighting",
    "switch":           "lighting",
    "night_lights":     "lighting",
    "power_socket":     "sockets",
    "internet_sockets": "sockets",
    "tv_sockets":       "sockets",
    "smoke_detector":   "low_voltage",
    "co2_detector":     "low_voltage",
    "motion_sensor":    "low_voltage",
    "intercom":         "low_voltage",
    "alarm":            "low_voltage",
}

# Цвет трассы по группе (BGR для OpenCV)
GROUP_COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    "lighting":    (0,   0,   220),   # Красный  — освещение
    "sockets":     (220, 50,    0),   # Синий    — розетки
    "low_voltage": (0,   200, 220),   # Жёлтый   — слаботочка
}

GROUP_COLOR_RGB: Dict[str, str] = {
    "lighting":    "#DC0000",
    "sockets":     "#0032DC",
    "low_voltage": "#00C8DC",
}

GROUP_LABEL_RU: Dict[str, str] = {
    "lighting":    "Освещение (SVT/SWI)",
    "sockets":     "Розетки (RZT/LAN/TV)",
    "low_voltage": "Слаботочка (DYM/CO2)",
}

GROUP_LINE_THICKNESS: Dict[str, int] = {
    "lighting":    2,
    "sockets":     2,
    "low_voltage": 1,
}

DEFAULT_PANEL_ANCHOR_PX = (30.0, 30.0)


# ─────────────────────────────────────────────────────────────────────────────
# Маска стен — читаем ЛОКАЛЬНО, не через MinIO
# ─────────────────────────────────────────────────────────────────────────────

def _get_walls_mask(project_id: str) -> Optional[np.ndarray]:
    """
    Получаем маску стен для проекта.

    Приоритет:
    1. Локальный файл /tmp/exports/{project_id}_walls.png (пишется при /ingest)
    2. Строим на лету из исходного PNG через build_walls_mask()
    3. None — нет данных, A* уйдёт в fallback (прямая линия)
    """
    # 1. Локальный файл
    local_walls = f"/tmp/exports/{project_id}_walls.png"
    if osp.exists(local_walls):
        img = cv2.imread(local_walls, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            logger.info("routing: walls mask loaded from local file (%s)", local_walls)
            return img

    # 2. Строим из исходника
    # Сначала ищем в in-memory DB, потом в SQLite (после рестарта контейнера)
    source = DB.get("source", {}).get(project_id, {})
    img_path = source.get("local_path")
    if not img_path:
        try:
            from app.db import get_project
            row = get_project(project_id)
            if row:
                img_path = row.get("local_path")
                logger.info("routing: local_path loaded from SQLite: %s", img_path)
        except Exception as e:
            logger.warning("routing: SQLite local_path lookup failed: %s", e)
    if img_path and osp.exists(img_path):
        try:
            from app.structure_detect import build_walls_mask
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                mask = build_walls_mask(img_bgr)
                # Кешируем чтобы не строить каждый раз
                os.makedirs("/tmp/exports", exist_ok=True)
                cv2.imwrite(local_walls, mask)
                logger.info("routing: walls mask built from source image (%s)", img_path)
                return mask
        except Exception as e:
            logger.warning("routing: failed to build walls mask: %s", e)

    # 3. Fallback через MinIO (старый код)
    try:
        pg = DB.get("plan_graph", {}).get(project_id)
        key = None
        if isinstance(pg, dict):
            key = ((pg.get("artifacts") or {}).get("masks") or {}).get("wallsMaskKey")
        if not key:
            st = DB.get("structure", {}).get(project_id)
            if isinstance(st, dict):
                key = st.get("walls_mask_key")
        if key:
            local_dir = "/tmp/routing"
            os.makedirs(local_dir, exist_ok=True)
            local = osp.join(local_dir, f"{project_id}_walls.png")
            download_file(EXPORT_BUCKET, str(key), local)
            img = cv2.imread(local, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                logger.info("routing: walls mask loaded from MinIO key=%s", key)
                return img
    except Exception as e:
        logger.warning("routing: MinIO walls mask fallback failed: %s", e)

    logger.warning("routing: no walls mask for project %s — routes will be straight lines", project_id)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Щиток — позиция
# ─────────────────────────────────────────────────────────────────────────────

def _get_panel_position(project_id: str) -> Point:
    """
    Возвращает позицию щитка (точки входа проводки).

    Приоритет:
    1. DB["panel"][project_id] — задано явно через /set_panel или парсером [щиток]
    2. DEFAULT_PANEL_ANCHOR_PX — (30, 30) заглушка
    """
    panel_raw = DB.get("panel", {}).get(project_id)
    if isinstance(panel_raw, dict):
        x = panel_raw.get("x", DEFAULT_PANEL_ANCHOR_PX[0])
        y = panel_raw.get("y", DEFAULT_PANEL_ANCHOR_PX[1])
        logger.info("routing: panel at (%.0f, %.0f) from DB", x, y)
        return Point(float(x), float(y))

    logger.warning(
        "routing: panel position not set for project %s — using default (30,30). "
        "Add '[щиток]' to preferencesText or call POST /set_panel.", project_id
    )
    return Point(float(DEFAULT_PANEL_ANCHOR_PX[0]), float(DEFAULT_PANEL_ANCHOR_PX[1]))


# ─────────────────────────────────────────────────────────────────────────────
# A* по маске стен
# ─────────────────────────────────────────────────────────────────────────────

def _build_occupancy(walls_mask: np.ndarray, downsample: int, dilate_px: int) -> Tuple[np.ndarray, int]:
    """Строим occupancy grid из маски стен."""
    m = (walls_mask > 0).astype(np.uint8) * 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)

    h, w = m.shape[:2]
    ds = max(2, int(downsample))
    nh = max(1, h // ds)
    nw = max(1, w // ds)
    m_small = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    blocked = (m_small > 0)
    return blocked.astype(bool), ds


def _astar(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    h, w = occ.shape[:2]
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if occ[sy, sx] or occ[gy, gx]:
        return None

    def heur(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    nbrs = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
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


def _nearest_free_cell(occ: np.ndarray, cell: Tuple[int, int], max_r: int = 30) -> Optional[Tuple[int, int]]:
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


def _route_one(
    walls_mask: np.ndarray,
    device_pt: Point,
    panel_pt: Point,
) -> List[Tuple[float, float]]:
    """
    Строит A*-маршрут от устройства до щитка.
    Пробует несколько параметров (грубо → точно).
    Fallback: прямая линия — ТОЛЬКО если маски нет вообще.
    """
    attempts = [
        (8, 3),
        (6, 2),
        (4, 1),
        (4, 0),
    ]

    for ds, dil in attempts:
        occ, ds_used = _build_occupancy(walls_mask, downsample=ds, dilate_px=dil)

        s0 = _point_to_cell(device_pt, ds_used)
        g0 = _point_to_cell(panel_pt,  ds_used)

        s = _nearest_free_cell(occ, s0, max_r=40)
        g = _nearest_free_cell(occ, g0, max_r=60)

        if s is None or g is None:
            continue

        path_cells = _astar(occ, s, g)
        if path_cells is None or len(path_cells) < 2:
            continue

        pts = [_cell_to_point(c, ds_used) for c in path_cells]
        # Восстанавливаем точные координаты начала и конца
        pts[0]  = (float(device_pt.x), float(device_pt.y))
        pts[-1] = (float(panel_pt.x),  float(panel_pt.y))
        return pts

    # Fallback — прямая (только если A* не нашёл путь ни при каком параметре)
    logger.warning(
        "routing: A* failed for device at (%.0f, %.0f) — using straight line fallback",
        device_pt.x, device_pt.y
    )
    return [
        (float(device_pt.x), float(device_pt.y)),
        (float(panel_pt.x),  float(panel_pt.y)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Нормализация устройств из DesignGraph
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_devices(project_id: str) -> List[Tuple[str, str, Point]]:
    out: List[Tuple[str, str, Point]] = []

    design = DB.get("design", {}).get(project_id)
    if isinstance(design, dict):
        for d in design.get("devices", []):
            if not isinstance(d, dict):
                continue
            try:
                kind = str(d.get("kind") or d.get("type") or "DEVICE")
                rid  = str(d.get("roomRef") or d.get("room_id") or "room_000")
                x    = float(d.get("xPx") or d.get("x") or 0)
                y    = float(d.get("yPx") or d.get("y") or 0)
                out.append((kind, rid, Point(x, y)))
            except Exception:
                pass
        if out:
            return out

    # Legacy fallback
    raw = DB.get("devices", {}).get(project_id, [])
    if not isinstance(raw, list):
        return out
    for d in raw:
        if isinstance(d, dict):
            try:
                t   = str(d.get("type") or d.get("kind") or "DEVICE")
                rid = str(d.get("roomId") or d.get("room_id") or "room_000")
                x   = float(d.get("x") or d.get("xPx") or 0)
                y   = float(d.get("y") or d.get("yPx") or 0)
                out.append((t, rid, Point(x, y)))
            except Exception:
                pass
    return out


def _get_px_per_meter(project_id: str) -> Optional[float]:
    pg = DB.get("plan_graph", {}).get(project_id)
    if not isinstance(pg, dict):
        return None
    ppm = ((pg.get("source") or {}).get("scale") or {}).get("pxPerMeter")
    try:
        v = float(ppm)
        return v if v > 0 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция маршрутизации
# ─────────────────────────────────────────────────────────────────────────────

def route_all(project_id: str) -> List[Dict]:
    """
    Строит трассы от каждого устройства до щитка, с обходом стен (A*).

    Возвращает список маршрутов:
    {
        "device_kind": str,
        "room_id": str,
        "group": str,
        "color_bgr": tuple,
        "line_thickness": int,
        "points": [(x,y), ...],
        "length_px": float,
        "length_m": float,
        "length_cm": float,
        "length_mm": float,
    }
    """
    devices = _normalize_devices(project_id)
    if not devices:
        logger.warning("routing: no devices for project %s", project_id)
        DB.setdefault("routes", {})[project_id] = []
        return []

    panel   = _get_panel_position(project_id)
    walls   = _get_walls_mask(project_id)
    px_per_m = _get_px_per_meter(project_id)

    logger.info(
        "routing: project=%s devices=%d panel=(%.0f,%.0f) walls=%s px_per_m=%s",
        project_id, len(devices),
        panel.x, panel.y,
        "OK" if walls is not None else "NONE (straight lines)",
        px_per_m,
    )

    routes: List[Dict] = []

    for kind, room_id, p in devices:
        group     = DEVICE_GROUP.get(kind, "low_voltage")
        color_bgr = GROUP_COLOR_BGR[group]
        thickness = GROUP_LINE_THICKNESS[group]

        if walls is not None:
            pts = _route_one(walls, p, panel)
        else:
            # Нет маски — прямая линия
            pts = [(float(p.x), float(p.y)), (float(panel.x), float(panel.y))]

        length_px = sum(
            math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )

        if px_per_m and px_per_m > 0:
            length_m = length_px / px_per_m
        else:
            # Эвристика: типичный план ~100px/м
            length_m = length_px / 100.0

        routes.append({
            "device_kind":    kind,
            "room_id":        room_id,
            "group":          group,
            "color_bgr":      color_bgr,
            "line_thickness": thickness,
            "points":         pts,
            "length_px":      round(length_px, 1),
            "length_m":       round(length_m, 3),
            "length_cm":      round(length_m * 100, 1),
            "length_mm":      round(length_m * 1000, 0),
        })

    DB.setdefault("routes", {})[project_id] = routes
    logger.info("routing: built %d routes for project %s", len(routes), project_id)
    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Спецификация кабеля
# ─────────────────────────────────────────────────────────────────────────────

def build_cable_spec(routes: List[Dict], project_id: str = "") -> Dict:
    groups: Dict[str, Dict] = {}

    for route in routes:
        g    = route.get("group", "low_voltage")
        kind = route.get("device_kind", "unknown")
        rid  = route.get("room_id", "")
        lm   = route.get("length_m",  0.0)
        lcm  = route.get("length_cm", 0.0)
        lmm  = route.get("length_mm", 0.0)

        if g not in groups:
            groups[g] = {
                "label_ru":     GROUP_LABEL_RU.get(g, g),
                "color_hex":    GROUP_COLOR_RGB.get(g, "#888888"),
                "devices":      [],
                "total_m":      0.0,
                "total_cm":     0.0,
                "total_mm":     0.0,
                "device_count": 0,
            }

        groups[g]["devices"].append({
            "kind":      kind,
            "room_id":   rid,
            "length_m":  lm,
            "length_cm": lcm,
            "length_mm": lmm,
        })
        groups[g]["total_m"]      += lm
        groups[g]["total_cm"]     += lcm
        groups[g]["total_mm"]     += lmm
        groups[g]["device_count"] += 1

    for g in groups:
        groups[g]["total_m"]  = round(groups[g]["total_m"],  2)
        groups[g]["total_cm"] = round(groups[g]["total_cm"], 1)
        groups[g]["total_mm"] = round(groups[g]["total_mm"], 0)

    grand_m  = round(sum(groups[g]["total_m"]  for g in groups), 2)
    grand_cm = round(sum(groups[g]["total_cm"] for g in groups), 1)
    grand_mm = round(sum(groups[g]["total_mm"] for g in groups), 0)
    total_d  = sum(groups[g]["device_count"]   for g in groups)

    return {
        "project_id":     project_id,
        "groups":         groups,
        "grand_total_m":  grand_m,
        "grand_total_cm": grand_cm,
        "grand_total_mm": grand_mm,
        "total_devices":  total_d,
    }


def build_cable_spec_text(spec: Dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  СПЕЦИФИКАЦИЯ КАБЕЛЯ")
    pid = spec.get("project_id", "")
    if pid:
        lines.append(f"  Проект: {pid}")
    lines.append("=" * 60)
    lines.append("")

    for group_key in ("lighting", "sockets", "low_voltage"):
        g = spec.get("groups", {}).get(group_key)
        if not g:
            continue
        lines.append("─" * 60)
        lines.append(f"  {g['label_ru']}")
        lines.append(f"  Цвет трассы: {g['color_hex']}")
        lines.append("─" * 60)
        lines.append(f"  {'Устройство':<20} {'Комната':<12} {'м':>6} {'см':>8} {'мм':>8}")
        lines.append(f"  {'-'*20} {'-'*12} {'-'*6} {'-'*8} {'-'*8}")
        for d in g["devices"]:
            lines.append(
                f"  {d['kind']:<20} {d['room_id']:<12} "
                f"{d['length_m']:>6.2f} {d['length_cm']:>7.1f} {d['length_mm']:>8.0f}"
            )
        lines.append(
            f"  {'':20} {'ИТОГО':<12} "
            f"{g['total_m']:>6.2f} {g['total_cm']:>7.1f} {g['total_mm']:>8.0f}"
        )
        lines.append(f"  Устройств в группе: {g['device_count']}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  ИТОГО ПО ВСЕМ ГРУППАМ:")
    lines.append(
        f"    {spec['grand_total_m']:.2f} м  /  "
        f"{spec['grand_total_cm']:.1f} см  /  "
        f"{spec['grand_total_mm']:.0f} мм"
    )
    lines.append(f"  Всего устройств: {spec['total_devices']}")
    lines.append("=" * 60)

    return "\n".join(lines)