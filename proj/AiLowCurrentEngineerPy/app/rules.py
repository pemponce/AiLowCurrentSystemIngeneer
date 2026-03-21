"""
app/rules.py — правила размещения слаботочных устройств.
СП 52.13330.2016, СП 484.1311500.2020, ПУЭ 7-е издание, ГОСТ 31565-2012
"""
from __future__ import annotations
from typing import Dict, Set, Tuple

DEVICE_KINDS = [
    "ceiling_lights", "smoke_detector", "co2_detector",
    "internet_sockets", "power_socket", "switch",
]

DISABLED_DEVICES: Set[str] = {"tv_sockets", "night_lights"}

ROOM_TYPES = [
    "living_room", "bedroom", "kitchen",
    "bathroom", "toilet", "corridor", "balcony",
]

ONLY_LIGHT:  Set[str] = {"bathroom", "toilet", "balcony"}
NO_SMOKE:    Set[str] = {"kitchen", "bathroom", "toilet", "balcony"}
NO_CO2:      Set[str] = {"bathroom", "toilet", "balcony", "corridor", "bedroom", "living_room"}
NO_RZT:      Set[str] = {"bathroom", "toilet", "corridor", "balcony"}

MIN_ROOM_AREA_M2 = 3.0
MAX_ROOM_AREA_M2 = 300.0

def is_valid_room(area_m2: float) -> bool:
    return MIN_ROOM_AREA_M2 <= area_m2 <= MAX_ROOM_AREA_M2

def svt_count(area_m2: float, room_type: str) -> int:
    """СП 52.13330.2016"""
    if room_type in ("bathroom", "toilet", "balcony"): return 1
    if room_type == "corridor": return max(1, min(4, round(area_m2 / 8.0)))
    if room_type == "kitchen":  return max(1, min(4, round(area_m2 / 10.0)))
    return max(1, min(8, round(area_m2 / 16.0)))

def rzt_count(area_m2: float, room_type: str) -> int:
    """ПУЭ — не более 6 на комнату"""
    if room_type in NO_RZT: return 0
    return max(1, min(6, round(area_m2 / 12.0)))

def dym_needed(area_m2: float, room_type: str) -> int:
    """СП 484.1311500"""
    if room_type in NO_SMOKE: return 0
    if room_type == "corridor": return 1
    if area_m2 < 12: return 0
    return 1 if area_m2 < 85 else 2

DEVICE_LABELS: Dict[str, str] = {
    "ceiling_lights": "SVT", "smoke_detector": "DYM",
    "co2_detector": "CO2", "internet_sockets": "LAN",
    "power_socket": "RZT", "switch": "SWI",
}

DEVICE_COLORS: Dict[str, Tuple[int,int,int]] = {
    "ceiling_lights":   (0, 200, 200),
    "smoke_detector":   (0, 100, 220),
    "co2_detector":     (80, 160, 0),
    "internet_sockets": (200, 60, 60),
    "power_socket":     (30, 120, 220),
    "switch":           (180, 100, 20),
}


# ── Совместимость со старым validator.py ─────────────────────────────────────
def get_rules(room_type: str = None) -> dict:
    """Возвращает правила для типа комнаты. Заглушка для совместимости."""
    rules = {
        "ceiling_lights": True,
        "smoke_detector": room_type not in NO_SMOKE if room_type else True,
        "co2_detector":   room_type == "kitchen",
        "power_socket":   room_type not in NO_RZT if room_type else True,
        "internet_sockets": room_type in ("living_room", "corridor"),
        "switch":         room_type not in ONLY_LIGHT if room_type else True,
    }
    return rules