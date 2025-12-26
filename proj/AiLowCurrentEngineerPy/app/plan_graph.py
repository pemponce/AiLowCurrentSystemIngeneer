from __future__ import annotations

from typing import Any, Dict, List, Optional

from shapely.geometry import Polygon

from app.geometry import DB, coerce_polygon, normalize_project_geometry


def get_plan_graph(project_id: str) -> Optional[Dict[str, Any]]:
    pg = DB.get("plan_graph", {}).get(project_id)
    return pg


def ensure_plan_graph(project_id: str) -> Dict[str, Any]:
    """
    Если plan_graph не построен через /detect-structure, соберём минимальный
    граф только из rooms (чтобы downstream не падал).
    """
    pg = get_plan_graph(project_id)
    if pg:
        return pg

    rooms: List[Polygon] = normalize_project_geometry(project_id)
    rooms_out = []
    for i, poly in enumerate(rooms):
        rooms_out.append(
            {
                "id": f"room_{i:03d}",
                "polygon": [[float(x), float(y)] for (x, y) in list(poly.exterior.coords)],
                "confidence": 0.3,
            }
        )

    pg = {
        "version": "plan-graph-0.2",
        "source": DB.get("source_meta", {}).get(project_id, {}),
        "elements": {"rooms": rooms_out, "walls": [], "openings": []},
        "artifacts": {},
    }
    DB.setdefault("plan_graph", {})[project_id] = pg
    return pg


def rooms_from_plan_graph(project_id: str) -> List[Polygon]:
    """
    Возвращает комнаты как Polygon, пытаясь сначала из DB['rooms'],
    иначе из plan_graph.elements.rooms[*].polygon
    """
    rooms = DB.get("rooms", {}).get(project_id) or []
    if rooms:
        return normalize_project_geometry(project_id)

    pg = ensure_plan_graph(project_id)
    out: List[Polygon] = []
    for r in (pg.get("elements", {}).get("rooms") or []):
        poly = coerce_polygon(r.get("polygon") if isinstance(r, dict) else None)
        if poly is not None:
            out.append(poly)
    DB.setdefault("rooms", {})[project_id] = out
    return out


def openings_from_plan_graph(project_id: str) -> List[Dict[str, Any]]:
    pg = ensure_plan_graph(project_id)
    return list(pg.get("elements", {}).get("openings") or [])
