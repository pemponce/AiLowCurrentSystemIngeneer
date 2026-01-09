from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from app.geometry import DB


def _pt_xy(p: Any) -> Tuple[float, float]:
    if p is None:
        return 0.0, 0.0
    if isinstance(p, dict):
        if "x" in p and "y" in p:
            return float(p.get("x", 0.0)), float(p.get("y", 0.0))
        if "pt" in p:
            return _pt_xy(p["pt"])
        if "point" in p:
            return _pt_xy(p["point"])
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    return 0.0, 0.0


def _polyline_length(points: Any) -> float:
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return 0.0
    total = 0.0
    prev = _pt_xy(points[0])
    for p in points[1:]:
        cur = _pt_xy(p)
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
    return float(total)


def _route_length(route: Any) -> float:
    """Поддерживаем и старый, и новый формат маршрутов.

    Новый (JSON):
      {"type":"SOCKET","length_m":12.3,"points":[{"x":..,"y":..}, ...]}
    Старый (tuple/list):
      (type, line, length)
    """
    if route is None:
        return 0.0
    if isinstance(route, dict):
        try:
            if route.get("length_m") is not None:
                return float(route.get("length_m"))
        except Exception:
            pass
        return _polyline_length(route.get("points") or route.get("polyline") or route.get("coords"))

    if isinstance(route, (list, tuple)):
        if len(route) >= 3 and isinstance(route[2], (int, float)):
            return float(route[2])
        if len(route) >= 2 and hasattr(route[1], "coords"):
            try:
                pts = list(route[1].coords)
                return _polyline_length(pts)
            except Exception:
                return 0.0
    return 0.0


def _route_type(route: Any) -> str:
    if route is None:
        return "UNKNOWN"
    if isinstance(route, dict):
        return str(route.get("type") or route.get("kind") or "UNKNOWN")
    if isinstance(route, (list, tuple)) and route:
        return str(route[0])
    return "UNKNOWN"


def make_bom(project_id: str) -> Dict[str, float]:
    """Грубая BOM: суммируем длину трасс.

    По умолчанию:
      - SOCKET/SWITCH -> UTP Cat5e
      - остальное     -> Cable

    IMPORTANT: ожидается, что DB хранит JSON-совместимые структуры.
    """
    routes = DB.get("routes", {}).get(project_id, []) or []
    bom = defaultdict(float)
    for r in routes:
        t = _route_type(r).upper()
        length = _route_length(r)
        if t in ("SOCKET", "SWITCH"):
            bom["UTP Cat5e (m)"] += length
        else:
            bom["Cable (m)"] += length

    return {k: round(float(v), 1) for k, v in bom.items()}
