from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model
from shapely.geometry import LineString, Point, Polygon

from app.geometry import DB, along_wall_points, room_walls
from app.rules import get_rules


Candidate = Tuple[str, str, Point]  # (device_type, room_id, point)


def _room_id(idx: int) -> str:
    return f"room_{idx:03d}"


def _room_size_hint(room: Polygon) -> float:
    a = float(abs(room.area))
    return max(1.0, math.sqrt(a))


def _offset_inside(room: Polygon, base: Point, seg: LineString) -> Point:
    x1, y1 = seg.coords[0]
    x2, y2 = seg.coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy) or 1.0
    nx = -dy / L
    ny = dx / L

    off = max(0.02 * _room_size_hint(room), 0.2)
    p1 = Point(base.x + nx * off, base.y + ny * off)
    p2 = Point(base.x - nx * off, base.y - ny * off)

    if room.contains(p1):
        return p1
    if room.contains(p2):
        return p2
    return base


def _openings_for_room(project_id: str, room_id: str) -> List[Dict[str, Any]]:
    struct = DB.get("structure", {}).get(project_id)
    if not struct:
        return []
    openings = struct.get("openings") or []
    out = []
    for o in openings:
        refs = o.get("roomRefs") or []
        if room_id in refs:
            out.append(o)
    return out


def _opening_seg(o: Dict[str, Any]) -> Optional[LineString]:
    seg = o.get("segmentPx") or o.get("segment")
    if not seg or not isinstance(seg, list) or len(seg) != 2:
        return None
    try:
        (x1, y1), (x2, y2) = seg
        return LineString([(float(x1), float(y1)), (float(x2), float(y2))])
    except Exception:
        return None


def generate_candidates(project_id: str) -> List[Candidate]:
    rules = get_rules()
    rooms: List[Polygon] = DB["rooms"].get(project_id, [])
    candidates: List[Candidate] = []

    for idx, room in enumerate(rooms):
        rid = _room_id(idx)

        per_meter = rules["per_room_requirements"].get("LIVING", {}).get("socket_per_wall_meter", 0.3)
        step = 1.0 / max(per_meter, 0.2)

        openings = _openings_for_room(project_id, rid)
        opening_segs = [s for s in (_opening_seg(o) for o in openings) if s is not None]

        clearance = max(0.03 * _room_size_hint(room), 0.25)

        for wall in room_walls(room):
            for p in along_wall_points(wall, step=step, offsets=0.3):
                if opening_segs:
                    if min(s.distance(p) for s in opening_segs) < clearance:
                        continue
                candidates.append(("SOCKET", rid, p))

        door_openings = [o for o in openings if (o.get("kind") == "door")]
        if door_openings:
            best = None
            best_len = -1.0
            for o in door_openings:
                seg = _opening_seg(o)
                if seg is None:
                    continue
                L = float(seg.length)
                if L > best_len:
                    best_len = L
                    best = seg
            if best is not None:
                mid = best.interpolate(0.5, normalized=True)
                sw = _offset_inside(room, mid, best)
                candidates.append(("SWITCH", rid, sw))
                continue

        x, y = room.exterior.coords[0]
        candidates.append(("SWITCH", rid, Point(float(x), float(y))))

    return candidates


def select_devices(project_id: str, candidates: List[Candidate]):
    model = cp_model.CpModel()
    xs = []

    for i, _ in enumerate(candidates):
        xs.append(model.NewBoolVar(f"x_{i}"))

    room_ids = sorted({rid for _, rid, _ in candidates})

    for rid in room_ids:
        idxs_socket = [i for i, (t, r, _) in enumerate(candidates) if r == rid and t == "SOCKET"]
        idxs_switch = [i for i, (t, r, _) in enumerate(candidates) if r == rid and t == "SWITCH"]

        if idxs_switch:
            model.Add(sum(xs[i] for i in idxs_switch) >= 1)

        if idxs_socket:
            model.Add(sum(xs[i] for i in idxs_socket) <= 6)

    model.Minimize(sum(xs))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0

    res = solver.Solve(model)
    chosen: List[Candidate] = []

    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, var in enumerate(xs):
            if solver.Value(var) == 1:
                chosen.append(candidates[i])

    DB["devices"][project_id] = chosen
    return chosen
