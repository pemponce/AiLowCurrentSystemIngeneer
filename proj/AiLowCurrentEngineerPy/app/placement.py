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
from app.nn3.infer import _wall_point as _wp_norm2
from app.export_overlay_png import _build_lighting_zones

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
    - smoke_detector: НЕ ставим на кухне и в санузлах
    - co2_detector: ТОЛЬКО кухня (и гостиная по желанию)
    - night_lights: ТОЛЬКО спальня и коридор
    - tv_sockets: НЕ ставим в corridor/bathroom/toilet/kitchen
    """
    forced_devices = forced_devices or {}

    NO_SMOKE = {"kitchen", "bathroom", "toilet", "balcony"}
    NO_CO2 = {"bedroom", "bathroom", "toilet", "corridor", "balcony"}
    NO_NIGHT = {"living_room", "kitchen", "bathroom", "toilet"}
    NO_TV = {"corridor", "bathroom", "toilet"}
    NO_SOCKET = {"bathroom", "toilet", "balcony"}
    ONLY_LIGHT = {"bathroom", "toilet", "balcony"}  # только свет, ничего другого
    DISABLED_DEVICES = {"tv_sockets", "night_lights"}  # временно отключены

    devices = design_graph.get("devices", [])
    room_designs = design_graph.get("roomDesigns", [])

    # Карта roomId → roomType с коррекцией балкона
    # Берём площадь из переданных rooms
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
        rid = rd["roomId"]
        rtype = rd.get("roomType", "bedroom")
        area = room_areas.get(rid, 999)
        # Коррекция: маленькая "кухня" (< 10 m²) → балкон/лоджия
        if rtype == "kitchen" and area < 10:
            rtype = "balcony"
            logger.debug("Hard rules: %s reclassified kitchen→balcony (area=%.1f m²)", rid, area)
        room_type_map[rid] = rtype

    # Найдём комнату для роутера: corridor > living_room
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

    # Фильтруем устройства
    filtered_devices = []
    internet_placed = False

    for d in devices:
        kind = d.get("kind", "")
        # Временно отключённые устройства
        if kind in DISABLED_DEVICES:
            continue
        room_id = d.get("roomRef", "") or d.get("room_id", "")
        rtype = room_type_map.get(room_id, "bedroom")

        # Санузел/туалет/балкон — только ceiling_lights
        if rtype in ONLY_LIGHT and kind != "ceiling_lights":
            continue
        # Балкон — максимум 1 светильник
        if rtype == "balcony" and kind == "ceiling_lights":
            already = sum(1 for d in filtered_devices
                          if d.get("kind") == "ceiling_lights"
                          and (d.get("roomRef") == room_id or d.get("room_id") == room_id))
            if already >= 1:
                continue

        # Датчик дыма — не на кухне и не в санузлах
        if kind == "smoke_detector" and rtype in NO_SMOKE:
            continue

        # CO2 — только кухня
        if kind == "co2_detector" and rtype in NO_CO2:
            continue

        # Ночник — только спальня и коридор
        if kind == "night_lights" and rtype in NO_NIGHT:
            continue

        # TV — не в коридоре/санузле
        if kind == "power_socket" and rtype in NO_SOCKET:
            continue
        if kind == "tv_sockets" and rtype in NO_TV:
            continue

        # LAN — строго одна на квартиру
        # forced устройства (явный запрос пользователя) имеют приоритет над auto-placed
        if kind == "internet_sockets":
            is_forced = d.get("reason") == "user request"
            if internet_placed:
                # Если уже есть LAN и это не forced — пропускаем
                if not is_forced:
                    continue
                # Если forced — заменяем предыдущий auto-placed LAN
                filtered_devices = [x for x in filtered_devices if x.get("kind") != "internet_sockets"]
                internet_placed = False
            internet_placed = True

        filtered_devices.append(d)

    # Добавляем устройства явно запрошенные пользователем по номеру комнаты
    existing_ids = {d["id"] for d in filtered_devices}
    # Строим карту room_id → (polygonPx, centroidPx) для wall placement
    from app.nn3.infer import _wall_point as _wp
    _room_geo = {}
    for r in (rooms or []):
        if not isinstance(r, dict):
            continue
        _rid = r.get("id") or r.get("roomId") or r.get("room_id") or ""
        if isinstance(_rid, int):
            _rid = f"room_{_rid:03d}"
        _poly = r.get("polygonPx") or []
        _cp = r.get("centroidPx") or []
        if _poly:
            if _cp and len(_cp) >= 2:
                _cx, _cy = float(_cp[0]), float(_cp[1])
            else:
                _xs = [p[0] for p in _poly];
                _ys = [p[1] for p in _poly]
                _cx, _cy = sum(_xs) / len(_xs), sum(_ys) / len(_ys)
            _room_geo[_rid] = (_poly, _cx, _cy)

    for room_id, devs in forced_devices.items():
        rtype = room_type_map.get(room_id, "bedroom")
        for device, count in devs.items():
            if count <= 0:
                continue
            # Проверяем жёсткие запреты
            if device == "smoke_detector" and rtype in NO_SMOKE:
                continue
            if device == "co2_detector" and rtype in NO_CO2:
                continue
            # Убираем старые устройства этого типа в этой комнате и ставим нужное кол-во
            filtered_devices = [d for d in filtered_devices
                                if not (d.get("roomRef") == room_id and d.get("kind") == device)]
            for k in range(int(count)):
                dev_id = f"{room_id}_{device}_{k}_forced"
                if dev_id not in existing_ids:
                    dev_entry = {
                        "id": dev_id,
                        "kind": device,
                        "roomRef": room_id,
                        "room_id": room_id,
                        "mount": "wall",
                        "heightMm": 300,
                        "label": device.replace("_", " ").title(),
                        "reason": "user request",
                    }
                    # Добавляем координаты wall placement
                    if room_id in _room_geo:
                        _poly, _cx, _cy = _room_geo[room_id]
                        _px, _py = _wp(device, _poly, _cx, _cy, offset=22, n_device=k)
                        if _px is not None:
                            # Clamp внутри bbox комнаты
                            if _poly:
                                _xs = [p[0] for p in _poly];
                                _ys = [p[1] for p in _poly]
                                _x0, _x1 = min(_xs) + 12, max(_xs) - 12
                                _y0, _y1 = min(_ys) + 12, max(_ys) - 12
                                _px = int(max(_x0, min(_x1, _px)))
                                _py = int(max(_y0, min(_y1, _py)))
                            dev_entry["xPx"] = _px
                            dev_entry["yPx"] = _py
                    filtered_devices.append(dev_entry)

    # Если LAN не попал автоматически — добавляем в living_room/corridor
    if not internet_placed and internet_room:
        filtered_devices.append({
            "id": f"{internet_room}_internet_sockets_0",
            "kind": "internet_sockets",
            "roomRef": internet_room,
            "room_id": internet_room,
            "mount": "wall",
            "heightMm": 300,
            "label": "Internet Sockets",
            "reason": "rule: one LAN per apartment",
        })

    # ── Нормативная коррекция количества SVT по СП 52.13330 ─────────────────
    # Если NN-3 поставила меньше светильников чем нужно по норме — добавляем
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

        # Получаем polygonPx комнаты
        _room_poly_norm = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
            _rid_norm = f"room_{_rid_raw:03d}" if isinstance(_rid_raw, int) else str(_rid_raw)
            if _rid_norm == room_id or _rid_raw == room_id:
                _room_poly_norm = _r.get("polygonPx") or []
                break

        # Вычисляем оптимальные позиции SVT по сетке зон освещения
        svt_positions = _svt_grid_positions(_room_poly_norm, area, rtype)
        needed_svt = len(svt_positions)

        if _room_poly_norm and needed_svt > 0:
            # Удаляем ВСЕ старые SVT этой комнаты — заменяем на zone grid
            filtered_devices = [
                d for d in filtered_devices
                if not (d.get("kind") == "ceiling_lights"
                        and (d.get("roomRef") == room_id or d.get("room_id") == room_id))
            ]
            # Добавляем SVT с правильными координатами зон освещения
            for k, (px_svt, py_svt) in enumerate(svt_positions):
                dev_id = f"{room_id}_ceiling_lights_zone_{k}"
                filtered_devices.append({
                    "id": dev_id,
                    "kind": "ceiling_lights",
                    "roomRef": room_id,
                    "mount": "ceiling",
                    "heightMm": 0,
                    "label": "Ceiling Lights",
                    "reason": f"zone: {needed_svt} SVT for {area:.0f}m²",
                    "xPx": px_svt,
                    "yPx": py_svt,
                })

    # ── Коррекция позиций DYM/CO2 — потолок, центр комнаты (СП 484) ──────────
    for d in filtered_devices:
        kind = d.get("kind", "")
        if kind not in ("smoke_detector", "co2_detector"):
            continue
        room_id = d.get("roomRef") or d.get("room_id") or ""
        d["mount"] = "ceiling"
        d["heightMm"] = 0
        _poly_dym = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
            _rid_norm = f"room_{_rid_raw:03d}" if isinstance(_rid_raw, int) else str(_rid_raw)
            if _rid_norm == room_id or _rid_raw == room_id:
                _poly_dym = _r.get("polygonPx") or []
                break
        if not _poly_dym:
            continue
        _xs_d = [p[0] for p in _poly_dym]
        _ys_d = [p[1] for p in _poly_dym]
        _n_d = len(_poly_dym)
        _area_d = _acx_d = _acy_d = 0.0
        for _j in range(_n_d):
            _jj = (_j + 1) % _n_d
            _cross = _xs_d[_j] * _ys_d[_jj] - _xs_d[_jj] * _ys_d[_j]
            _area_d += _cross
            _acx_d += (_xs_d[_j] + _xs_d[_jj]) * _cross
            _acy_d += (_ys_d[_j] + _ys_d[_jj]) * _cross
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

        # ВАЖНО: получаем polygonPx именно для ТЕКУЩЕЙ комнаты
        _room_poly_rzt = []
        for _r in (rooms or []):
            if not isinstance(_r, dict):
                continue
            _rid_raw = _r.get("id") or _r.get("roomId") or _r.get("room_id") or ""
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
                n_idx = existing_rzt_count + k
                _px2, _py2 = _wp_norm2("power_socket", _room_poly_rzt,
                                       _cx_n, _cy_n, offset=5, n_device=n_idx)
                entry = {
                    "id": dev_id,
                    "kind": "power_socket",
                    "roomRef": room_id,
                    "mount": "wall",
                    "heightMm": 300,
                    "label": "Power Socket",
                    "reason": f"norm: {needed_rzt} RZT for {area:.0f}m²",
                }
                if _px2 is not None:
                    entry["xPx"] = _px2
                    entry["yPx"] = _py2
                filtered_devices.append(entry)

    # ── Выключатели SWI — у каждого дверного проёма (СП 256) ──────────────────
    # Если NN-1 не вернула doors — вычисляем проёмы через смежность полигонов
    try:
        from app.geometry import detect_doorways as _detect_doorways
        _all_doorways = _detect_doorways(rooms or [], tolerance=18.0)
    except Exception as _dw_err:
        logger.warning("SWI: detect_doorways failed: %s", _dw_err)
        _all_doorways = []

    # Строим карту room_id → список проёмов
    _doorway_map: dict = {}
    for _dw in _all_doorways:
        for _side in ("room_a", "room_b"):
            _rid = _dw[_side]
            _doorway_map.setdefault(_rid, []).append({"cx": _dw["cx"], "cy": _dw["cy"]})

    for room_id, rtype in room_type_map.items():
        if rtype in ("bathroom", "toilet"):
            continue
        # Находим дверные проёмы комнаты из геометрии
        _room_data = None
        for _r2 in (rooms or []):
            _rid2 = _r2.get("id") or _r2.get("room_id") or ""
            _rid2n = f"room_{_rid2:03d}" if isinstance(_rid2, int) else str(_rid2)
            if _rid2n == room_id or _rid2 == room_id:
                _room_data = _r2
                break
        if not _room_data:
            continue
        poly_swi = _room_data.get("polygonPx") or []
        if not poly_swi:
            continue

        # Приоритет: doors из NN-1 → detect_doorways → fallback на центроид входной стены
        doors = _room_data.get("doors") or []
        if doors:
            door_points = [
                {"cx": float(d.get("x") or d.get("cx") or 0),
                 "cy": float(d.get("y") or d.get("cy") or 0)}
                for d in doors
                if (d.get("x") or d.get("cx"))
            ]
        else:
            door_points = _doorway_map.get(room_id, [])

        # Финальный fallback: кратчайшее ребро полигона (вероятный проём)
        if not door_points and len(poly_swi) >= 3:
            _walls_swi = []
            _n_swi = len(poly_swi)
            for _wi in range(_n_swi):
                _x1, _y1 = float(poly_swi[_wi][0]), float(poly_swi[_wi][1])
                _x2, _y2 = float(poly_swi[(_wi+1) % _n_swi][0]), float(poly_swi[(_wi+1) % _n_swi][1])
                _wlen = (((_x2-_x1)**2 + (_y2-_y1)**2) ** 0.5)
                if _wlen >= 15:
                    _walls_swi.append({"cx": (_x1+_x2)/2, "cy": (_y1+_y2)/2, "len": _wlen})
            if _walls_swi:
                _shortest = min(_walls_swi, key=lambda w: w["len"])
                door_points = [{"cx": _shortest["cx"], "cy": _shortest["cy"]}]

        for di, door in enumerate(door_points):
            dx = float(door.get("cx") or door.get("x") or 0)
            dy = float(door.get("cy") or door.get("y") or 0)
            if dx == 0 and dy == 0:
                continue
            # SWI ставим вплотную к стене рядом с проёмом (сдвиг вдоль стены)
            best_wall_pt = None
            best_dist = float("inf")
            _n_swi2 = len(poly_swi)
            for _wi2 in range(_n_swi2):
                _x1s = float(poly_swi[_wi2][0])
                _y1s = float(poly_swi[_wi2][1])
                _x2s = float(poly_swi[(_wi2 + 1) % _n_swi2][0])
                _y2s = float(poly_swi[(_wi2 + 1) % _n_swi2][1])
                _wlen2 = ((_x2s - _x1s) ** 2 + (_y2s - _y1s) ** 2) ** 0.5
                if _wlen2 < 15:
                    continue
                _tx = (_x2s - _x1s) / _wlen2
                _ty = (_y2s - _y1s) / _wlen2
                _proj = (_tx * (dx - _x1s) + _ty * (dy - _y1s))
                _proj = max(0.1 * _wlen2, min(0.9 * _wlen2, _proj))
                _px_w = _x1s + _tx * _proj
                _py_w = _y1s + _ty * _proj
                _d = (_px_w - dx) ** 2 + (_py_w - dy) ** 2
                if _d < best_dist:
                    best_dist = _d
                    _side_offset = min(25.0, _wlen2 * 0.15)
                    _proj2 = min(_proj + _side_offset, 0.9 * _wlen2)
                    best_wall_pt = (
                        int(_x1s + _tx * _proj2),
                        int(_y1s + _ty * _proj2)
                    )
            if best_wall_pt:
                swi_x, swi_y = best_wall_pt
            else:
                swi_x, swi_y = int(dx), int(dy)

            filtered_devices.append({
                "id": f"{room_id}_switch_{di}",
                "kind": "switch",
                "roomRef": room_id,
                "mount": "wall",
                "heightMm": 900,
                "label": "Switch",
                "reason": "rule: SWI at door" if doors else "rule: SWI at detected opening",
                "xPx": swi_x,
                "yPx": swi_y,
            })
    # Пересчитываем deviceIds в roomDesigns
    dev_ids_by_room: dict = {}
    for d in filtered_devices:
        rid = d.get("roomRef") or d.get("room_id", "")
        dev_ids_by_room.setdefault(rid, []).append(d["id"])

    new_room_designs = []
    for rd in room_designs:
        rid = rd["roomId"]
        # Используем скорректированный roomType (kitchen→balcony для маленьких комнат)
        corrected_type = room_type_map.get(rid, rd.get("roomType", "bedroom"))
        new_room_designs.append({
            **rd,
            "roomType": corrected_type,
            "deviceIds": dev_ids_by_room.get(rid, []),
        })

    design_graph = {
        **design_graph,
        "devices": filtered_devices,
        "roomDesigns": new_room_designs,
        "totalDevices": len(filtered_devices),
        "explain": design_graph.get("explain", []) + ["Постпроцессинг: жёсткие правила применены"],
    }
    return design_graph
