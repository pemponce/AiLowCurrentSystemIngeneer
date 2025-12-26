from __future__ import annotations

from typing import Any, Dict, List, Optional

from shapely.geometry import Point, Polygon

from app.geometry import DB
from app.rules import get_rules


def _coerce_polygon(room_like: Any) -> Optional[Polygon]:
    if isinstance(room_like, Polygon):
        return room_like

    pts = None
    if isinstance(room_like, dict):
        pts = room_like.get("polygonPx") or room_like.get("polygon") or room_like.get("points")
    elif isinstance(room_like, (list, tuple)):
        pts = room_like

    if not pts or len(pts) < 3:
        return None

    try:
        poly = Polygon([(float(x), float(y)) for x, y in pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if (not poly.is_valid) or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


# MVP checks:
# - min distance from corners and doors
# - simple per-room minimums
def validate_project(project_id: str):
    rules = get_rules()

    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    devices = DB.get("devices", {}).get(project_id, [])
    doors = DB.get("doors", {}).get(project_id, [])

    violations: List[Dict[str, Any]] = []

    min_off = rules.get("min_offsets_cm", {})
    off_door = float(min_off.get("door", 15)) / 100.0
    off_corner = float(min_off.get("corner", 10)) / 100.0

    # requirements (keep your previous fallback behavior)
    default_req = rules.get("per_room_requirements", {}).get("BEDROOM", {})
    min_switches = int(default_req.get("min_switches", 1))

    for idx, room_like in enumerate(rooms_raw):
        poly = _coerce_polygon(room_like)
        if poly is None:
            continue

        room_id = idx
        if isinstance(room_like, dict):
            room_id = room_like.get("id", idx)

        # corners as polygon vertices (excluding duplicate last)
        corners = [Point(x, y) for x, y in list(poly.exterior.coords)[:-1]]

        # devices in this room (support idx/int/string ids)
        ds = []
        for d in devices:
            if not (isinstance(d, tuple) and len(d) == 3):
                continue
            t, r, p = d
            if r == idx or r == room_id or str(r) == str(room_id):
                ds.append((t, r, p))

        # minimum switches
        if sum(1 for (t, _, _) in ds if t == "SWITCH") < min_switches:
            violations.append({"room": room_id, "type": "SWITCH_MIN", "msg": "Не хватает выключателей"})

        for (t, _, p) in ds:
            try:
                if corners and min(c.distance(p) for c in corners) < off_corner:
                    violations.append({"room": room_id, "type": "OFFSET_CORNER", "msg": f"{t} слишком близко к углу"})
            except Exception:
                pass

            if doors:
                try:
                    dmin = min(d.distance(p) for d in doors)
                    if dmin < off_door:
                        violations.append({"room": room_id, "type": "OFFSET_DOOR", "msg": f"{t} слишком близко к двери"})
                except Exception:
                    pass

    return violations
