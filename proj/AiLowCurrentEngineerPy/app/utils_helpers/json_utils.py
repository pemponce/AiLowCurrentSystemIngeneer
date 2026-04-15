# app/utils_helpers/json_utils.py
"""
JSON serialization helpers для API responses.
Конвертирует Shapely геометрию, numpy типы и кастомные объекты в JSON.
"""

from typing import Any, Dict, List


def _is_point(obj: Any) -> bool:
    """Проверка что объект — Shapely Point."""
    return hasattr(obj, "x") and hasattr(obj, "y")


def _point_to_xy(obj: Any) -> Dict[str, float]:
    """Конвертация Point в {x, y}."""
    return {"x": float(getattr(obj, "x")), "y": float(getattr(obj, "y"))}


def _json_default(o: Any):
    """
    JSON encoder для нестандартных типов:
    - Shapely геометрия (Point, LineString, Polygon)
    - Numpy scalar types
    """
    # Shapely-like geometries
    if _is_point(o):
        return _point_to_xy(o)

    if hasattr(o, "geom_type") and hasattr(o, "coords"):
        # LineString-like
        return [{"x": float(x), "y": float(y)} for x, y in list(o.coords)]

    if hasattr(o, "geom_type") and hasattr(o, "exterior"):
        # Polygon-like
        ext = [{"x": float(x), "y": float(y)} for x, y in list(o.exterior.coords)]
        holes = []
        for ring in getattr(o, "interiors", []) or []:
            holes.append([{"x": float(x), "y": float(y)} for x, y in list(ring.coords)])
        return {"exterior": ext, "holes": holes}

    # Numpy scalars
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass

    # Fallback
    return str(o)


def _devices_to_json(devices: List[Any]) -> List[Dict[str, Any]]:
    """
    Конвертация списка устройств в JSON-совместимый формат.

    Поддерживаемые форматы:
    - dict
    - tuple: (TYPE, room_id, Point) или (TYPE, room_id, x, y) или (TYPE, Point)
    """
    out: List[Dict[str, Any]] = []
    for d in devices or []:
        if isinstance(d, dict):
            dd = dict(d)
            # normalize embedded point fields if present
            if "pt" in dd and _is_point(dd["pt"]):
                dd["x"] = float(dd["pt"].x)
                dd["y"] = float(dd["pt"].y)
                dd.pop("pt", None)
            if "point" in dd and _is_point(dd["point"]):
                dd["x"] = float(dd["point"].x)
                dd["y"] = float(dd["point"].y)
                dd.pop("point", None)
            out.append(dd)
            continue

        # tuple patterns:
        # (TYPE, room_id, Point)
        # (TYPE, room_id, x, y)
        # (TYPE, Point)
        if isinstance(d, (tuple, list)):
            if len(d) >= 3 and _is_point(d[2]):
                dev_type = str(d[0])
                room_id = str(d[1])
                pt = d[2]
                out.append({
                    "type": dev_type,
                    "label": dev_type,
                    "room_id": room_id,
                    "x": float(pt.x),
                    "y": float(pt.y),
                })
                continue
            if len(d) >= 4 and isinstance(d[2], (int, float)) and isinstance(d[3], (int, float)):
                dev_type = str(d[0])
                room_id = str(d[1])
                out.append({
                    "type": dev_type,
                    "label": dev_type,
                    "room_id": room_id,
                    "x": float(d[2]),
                    "y": float(d[3]),
                })
                continue
            if len(d) >= 2 and _is_point(d[1]):
                dev_type = str(d[0])
                pt = d[1]
                out.append({"type": dev_type, "label": dev_type, "x": float(pt.x), "y": float(pt.y)})
                continue

        # unknown device type: stringify
        out.append({"type": "UNKNOWN", "label": "UNKNOWN", "raw": str(d)})
    return out


def _routes_to_json(routes: List[Any]) -> List[Dict[str, Any]]:
    """
    Конвертация списка маршрутов в JSON-совместимый формат.

    Поддерживаемые форматы:
    - dict
    - tuple: (type, LineString, length_m)
    """
    out: List[Dict[str, Any]] = []
    for r in routes or []:
        if isinstance(r, dict):
            out.append(r)
            continue
        if isinstance(r, tuple) and len(r) >= 3:
            t = str(r[0])
            line = r[1]
            length_m = float(r[2])
            pts = []
            coords = getattr(line, "coords", None)
            if coords is not None:
                for x, y in list(coords):
                    pts.append({"x": float(x), "y": float(y)})
            out.append({"type": t, "length_m": length_m, "points": pts})
    return out