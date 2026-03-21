"""
app/placement.py — нормативное размещение устройств.

Содержит:
- _point_in_polygon      — ray casting
- _build_lighting_zones  — сетка зон SVT внутри полигона (импорт из export_overlay_png)
- _svt_grid_positions    — позиции SVT по зонам
- _apply_hard_rules      — постпроцессинг: ГОСТ, ПУЭ, СП 484
- _wall_positions_rzt    — распределение RZT по периметру с учётом дверей
"""

from __future__ import annotations
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("planner")

import re as import_re

def _point_in_polygon(px: float, py: float, poly: list) -> bool:
    """Ray casting — точка внутри полигона."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _svt_grid_positions(poly: list, area_m2: float, room_type: str = "living_room") -> list:
    """
    Возвращает список (px, py) центров зон освещения.
    Использует те же зоны что и export_zones_preview — единый алгоритм.
    """
    if not poly or len(poly) < 3:
        return []
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0

    if room_type in ("bathroom", "toilet", "balcony"):
        needed = 1
    elif room_type == "corridor":
        needed = max(1, min(4, round(area_m2 / 8.0)))
    elif room_type == "kitchen":
        needed = max(1, min(4, round(area_m2 / 10.0)))
    else:
        needed = max(1, min(8, round(area_m2 / 16.0)))

    aspect = w / max(1.0, h)
    if needed == 1:
        cols, rows = 1, 1
    elif needed == 2:
        cols, rows = (2, 1) if aspect >= 1.0 else (1, 2)
    elif needed <= 4:
        cols, rows = 2, 2
    elif needed <= 6:
        cols, rows = (3, 2) if aspect >= 1.0 else (2, 3)
    else:
        cols, rows = (3, 3) if needed <= 9 else (4, 3)

    try:
        from app.export_overlay_png import _build_lighting_zones
        zones = _build_lighting_zones(poly, area_m2, room_type)
        positions = [z["center"] for z in zones]
        if positions:
            return positions
    except Exception:
        pass

    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [(int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))]


def _apply_hard_rules(design_graph: dict, forced_devices: dict = None, rooms: list = None) -> dict:
    """
    Постпроцессинг после NN-3: применяем жёсткие правила которые нельзя нарушать.

    Правила:
    - internet_sockets: ОДИН на всю квартиру (corridor или living_room)
    - smoke_detector: НЕ ставим на кухне и в санузлах, максимум 1 на комнату
    - co2_detector: ТОЛЬКО кухня (и гостиная по желанию)
    - night_lights: ТОЛЬКО спальня и коридор
    - tv_sockets: НЕ ставим в corridor/bathroom/toilet/kitchen
    """
    forced_devices = forced_devices or {}

    NO_SMOKE   = {"kitchen", "bathroom", "toilet", "balcony"}
    NO_CO2     = {"bedroom", "bathroom", "toilet", "corridor", "balcony"}
    NO_NIGHT   = {"living_room", "kitchen", "bathroom", "toilet"}
    NO_TV      = {"corridor", "bathroom", "toilet"}
    NO_SOCKET  = {"bathroom", "toilet", "balcony"}
    ONLY_LIGHT = {"bathroom", "toilet", "balcony"}
    DISABLED_DEVICES = {"tv_sockets", "night_lights"}

    devices      = design_graph.get("devices", [])
    room_designs = design_graph.get("roomDesigns", [])

    room_areas = {}
    for r in (rooms or []):
        if isinstance(r, dict):
            _rid = r.get("id") or r.get("roomId") or r.get("room_id") or ""
            if isinstance(_rid, int):
                _rid = f"room_{_rid:03d}"
            _area = r.get("areaM2") or r.get("area_m2") or 999
            if _rid:
                room_areas[_rid] = float(_area)
    for rd in room_designs:
        rid = rd["roomId"]
        if rid not in room_areas:
            room_areas[rid] = rd.get("areaM2") or rd.get("area_m2") or 999

    room_type_map = {}
    for rd in room_designs:
        rid   = rd["roomId"]
        rtype = rd.get("roomType", "bedroom")
        area  = room_areas.get(rid, 999)
        if rtype == "kitchen" and area < 10:
            rtype = "balcony"
            logger.debug("Hard rules: %s reclassified kitchen→balcony (area=%.1f m²)", rid, area)
        room_type_map[rid] = rtype

    internet_room = None
    for priority in ["corridor", "living_room"]:
        for rd in room_designs:
            if rd.get("roomType") == priority:
                internet_room = rd["roomId"]
                break
        if internet_room:
            break
    if not internet_room and room_designs:
        internet_room = room_designs[0]["roomId"]

    filtered_devices = []
    internet_placed  = False

    for d in devices:
        kind    = d.get("kind", "")
        if kind in DISABLED_DEVICES:
            continue
        room_id = d.get("roomRef", "") or d.get("room_id", "")
        rtype   = room_type_map.get(room_id, "bedroom")

        if rtype in ONLY_LIGHT and kind != "ceiling_lights":
            continue

        if rtype == "balcony" and kind == "ceiling_lights":
            already = sum(1 for x in filtered_devices
                          if x.get("kind") == "ceiling_lights"
                          and (x.get("roomRef") == room_id or x.get("room_id") == room_id))
            if already >= 1:
                continue

        if kind == "smoke_detector" and rtype in NO_SMOKE:
            continue
        if kind == "co2_detector" and rtype in NO_CO2:
            continue
        if kind == "night_lights" and rtype in NO_NIGHT:
            continue
        if kind == "power_socket" and rtype in NO_SOCKET:
            continue
        if kind == "tv_sockets" and rtype in NO_TV:
            continue

        # LAN — строго одна на квартиру (любой, включая forced)
        if kind == "internet_sockets":
            if internet_placed:
                continue
            internet_placed = True

        # DYM — максимум 1 на комнату (СП 484.1311500.2020 п.6.2)
        if kind == "smoke_detector":
            already_dym = sum(
                1 for x in filtered_devices
                if x.get("kind") == "smoke_detector"
                and (x.get("roomRef") == room_id or x.get("room_id") == room_id)
            )
            if already_dym >= 1:
                continue

        filtered_devices.append(d)

    existing_ids = {d["id"] for d in filtered_devices}
    from app.nn3.infer import _wall_point as _wp
    _room_geo = {}
    for r in (rooms or []):
        if not isinstance(r, dict):
            continue
        _rid = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        if isinstance(_rid, int):
            _rid = f"room_{_rid:03d}"
        _poly = r.get("polygonPx") or []
        _cp   = r.get("centroidPx") or []
        if _poly:
            if _cp and len(_cp) >= 2:
                _cx, _cy = float(_cp[0]), float(_cp[1])
            else:
                _xs = [p[0] for p in _poly]; _ys = [p[1] for p in _poly]
                _cx, _cy = sum(_xs)/len(_xs), sum(_ys)/len(_ys)
            _room_geo[_rid] = (_poly, _cx, _cy)

    for room_id, devs in forced_devices.items():
        rtype = room_type_map.get(room_id, "bedroom")
        for device, count in devs.items():
            if count <= 0:
                continue
            if device == "smoke_detector" and rtype in NO_SMOKE:
                continue
            if device == "co2_detector" and rtype in NO_CO2:
                continue
            # LAN — один на квартиру, forced тоже не исключение
            if device == "internet_sockets":
                if internet_placed:
                    continue
                internet_placed = True
            filtered_devices = [d for d in filtered_devices
                                 if not (d.get("roomRef") == room_id and d.get("kind") == device)]
            for k in range(int(count)):
                dev_id = f"{room_id}_{device}_{k}_forced"
                if dev_id not in existing_ids:
                    dev_entry = {
                        "id":       dev_id,
                        "kind":     device,
                        "roomRef":  room_id,
                        "room_id":  room_id,
                        "mount":    "wall",
                        "heightMm": 300,
                        "label":    device.replace("_", " ").title(),
                        "reason":   "user request",
                    }
                    if room_id in _room_geo:
                        _poly, _cx, _cy = _room_geo[room_id]
                        _px, _py = _wp(device, _poly, _cx, _cy, offset=22, n_device=k)
                        if _px is not None:
                            if _poly:
                                _xs = [p[0] for p in _poly]; _ys = [p[1] for p in _poly]
                                _x0, _x1 = min(_xs)+12, max(_xs)-12
                                _y0, _y1 = min(_ys)+12, max(_ys)-12
                                _px = int(max(_x0, min(_x1, _px)))
                                _py = int(max(_y0, min(_y1, _py)))
                            dev_entry["xPx"] = _px
                            dev_entry["yPx"] = _py
                    filtered_devices.append(dev_entry)

    if not internet_placed and internet_room:
        filtered_devices.append({
            "id":       f"{internet_room}_internet_sockets_0",
            "kind":     "internet_sockets",
            "roomRef":  internet_room,
            "room_id":  internet_room,
            "mount":    "wall",
            "heightMm": 300,
            "label":    "Internet Sockets",
            "reason":   "rule: one LAN per apartment",
        })

    # ── Нормативная коррекция количества SVT по СП 52.13330 ─────────────────
    room_area_map = {}
    for r in (rooms or []):
        if not isinstance(r, dict):
            continue
        _rid = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        if isinstance(_rid, int):
            _rid = f"room_{_rid:03d}"
        _area = float(r.get("areaM2") or r.get("area_m2") or 0)
        if _rid:
            room_area_map[_rid] = _area

    import math as _math
    for room_id, rtype in room_type_map.items():
        if rtype in ("bathroom", "toilet", "balcony"):
            continue
        area = room_area_map.get(room_id, 0)
        if area <= 0:
            continue

        _room_poly_norm = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw  = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
            _rid_norm = f"room_{_rid_raw:03d}" if isinstance(_rid_raw, int) else str(_rid_raw)
            if _rid_norm == room_id or _rid_raw == room_id:
                _room_poly_norm = _r.get("polygonPx") or []
                break

        svt_positions = _svt_grid_positions(_room_poly_norm, area, rtype)
        needed_svt    = len(svt_positions)

        if _room_poly_norm and needed_svt > 0:
            filtered_devices = [
                d for d in filtered_devices
                if not (d.get("kind") == "ceiling_lights"
                        and (d.get("roomRef") == room_id or d.get("room_id") == room_id))
            ]
            for k, (px_svt, py_svt) in enumerate(svt_positions):
                dev_id = f"{room_id}_ceiling_lights_zone_{k}"
                filtered_devices.append({
                    "id":       dev_id,
                    "kind":     "ceiling_lights",
                    "roomRef":  room_id,
                    "mount":    "ceiling",
                    "heightMm": 0,
                    "label":    "Ceiling Lights",
                    "reason":   f"zone: {needed_svt} SVT for {area:.0f}m²",
                    "xPx":      px_svt,
                    "yPx":      py_svt,
                })

    # ── Коррекция позиций DYM/CO2 — потолок, центр комнаты (СП 484) ──────────
    for d in filtered_devices:
        kind = d.get("kind", "")
        if kind not in ("smoke_detector", "co2_detector"):
            continue
        room_id      = d.get("roomRef") or d.get("room_id") or ""
        d["mount"]    = "ceiling"
        d["heightMm"] = 0
        _poly_dym = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw  = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
            _rid_norm = f"room_{_rid_raw:03d}" if isinstance(_rid_raw, int) else str(_rid_raw)
            if _rid_norm == room_id or _rid_raw == room_id:
                _poly_dym = _r.get("polygonPx") or []
                break
        if not _poly_dym:
            continue
        _xs_d = [p[0] for p in _poly_dym]
        _ys_d = [p[1] for p in _poly_dym]
        _n_d  = len(_poly_dym)
        _area_d = _acx_d = _acy_d = 0.0
        for _j in range(_n_d):
            _jj    = (_j + 1) % _n_d
            _cross = _xs_d[_j] * _ys_d[_jj] - _xs_d[_jj] * _ys_d[_j]
            _area_d += _cross
            _acx_d  += (_xs_d[_j] + _xs_d[_jj]) * _cross
            _acy_d  += (_ys_d[_j] + _ys_d[_jj]) * _cross
        _area_d /= 2.0
        if abs(_area_d) > 1e-6:
            _cx_d = _acx_d / (6 * _area_d)
            _cy_d = _acy_d / (6 * _area_d)
        else:
            _cx_d = sum(_xs_d) / _n_d
            _cy_d = sum(_ys_d) / _n_d
        if not _point_in_polygon(_cx_d, _cy_d, _poly_dym):
            _cx_d = (min(_xs_d) + max(_xs_d)) / 2
            _cy_d = (min(_ys_d) + max(_ys_d)) / 2
            if not _point_in_polygon(_cx_d, _cy_d, _poly_dym):
                try:
                    from app.export_overlay_png import _nearest_interior_point as _nip
                    _cx_d, _cy_d = _nip(_cx_d, _cy_d, _poly_dym, step=4)
                except Exception:
                    pass
        d["xPx"] = int(_cx_d)
        d["yPx"] = int(_cy_d)

    # ── Нормативная коррекция количества RZT по ПУЭ ──────────────────────────
    from app.nn3.infer import _wall_point as _wp_norm2
    for room_id, rtype in room_type_map.items():
        if rtype in ("bathroom", "toilet", "corridor", "balcony"):
            continue
        area = room_area_map.get(room_id, 0)
        if area <= 0:
            continue

        _room_poly_rzt = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw  = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
            _rid_norm = f"room_{_rid_raw:03d}" if isinstance(_rid_raw, int) else str(_rid_raw)
            if _rid_norm == room_id or _rid_raw == room_id:
                _room_poly_rzt = _r.get("polygonPx") or []
                break

        needed_rzt = max(1, min(6, round(area / 12.0)))

        current_rzt = sum(
            1 for d in filtered_devices
            if d.get("kind") == "power_socket"
            and (d.get("roomRef") == room_id or d.get("room_id") == room_id)
        )

        if current_rzt < needed_rzt:
            existing_rzt_count = current_rzt
            _cx_n = sum(p[0] for p in _room_poly_rzt) / len(_room_poly_rzt) if _room_poly_rzt else 0.0
            _cy_n = sum(p[1] for p in _room_poly_rzt) / len(_room_poly_rzt) if _room_poly_rzt else 0.0
            for k in range(needed_rzt - current_rzt):
                dev_id = f"{room_id}_power_socket_auto_{k}"
                n_idx  = existing_rzt_count + k
                _px2, _py2 = _wp_norm2("power_socket", _room_poly_rzt,
                                        _cx_n, _cy_n, offset=22, n_device=n_idx)
                entry = {
                    "id":       dev_id,
                    "kind":     "power_socket",
                    "roomRef":  room_id,
                    "mount":    "wall",
                    "heightMm": 300,
                    "label":    "Power Socket",
                    "reason":   f"norm: {needed_rzt} RZT for {area:.0f}m²",
                }
                if _px2 is not None:
                    entry["xPx"] = _px2
                    entry["yPx"] = _py2
                filtered_devices.append(entry)

    # Пересчитываем deviceIds в roomDesigns
    dev_ids_by_room: dict = {}
    for d in filtered_devices:
        rid = d.get("roomRef") or d.get("room_id", "")
        dev_ids_by_room.setdefault(rid, []).append(d["id"])

    new_room_designs = []
    for rd in room_designs:
        rid            = rd["roomId"]
        corrected_type = room_type_map.get(rid, rd.get("roomType", "bedroom"))
        new_room_designs.append({
            **rd,
            "roomType":  corrected_type,
            "deviceIds": dev_ids_by_room.get(rid, []),
        })

    design_graph = {
        **design_graph,
        "devices":      filtered_devices,
        "roomDesigns":  new_room_designs,
        "totalDevices": len(filtered_devices),
        "explain":      design_graph.get("explain", []) + ["Постпроцессинг: жёсткие правила применены"],
    }
    return design_graph