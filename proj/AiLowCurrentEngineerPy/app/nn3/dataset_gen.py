"""
app/nn3/dataset_gen.py — правила размещения слаботочных устройств
по СП 52.13330, СП 484.1311500, ГОСТ 31565
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

ROOM_TYPES = [
    "living_room", "bedroom", "kitchen",
    "bathroom", "toilet", "corridor",
]

DEVICES = [
    "tv_sockets",
    "internet_sockets",
    "smoke_detector",
    "co2_detector",
    "ceiling_lights",
    "night_lights",
]

# ─── НОРМЫ по ГОСТ/СП ────────────────────────────────────────────────────────

def _lights_from_area(area_m2: float, room_type: str = "living_room") -> int:
    """
    СП 52.13330.2016 — нормируемая освещённость жилых помещений.
    Принцип: 1 светильник (люстра/спот) на каждые 15-20 м².
    Минимум: 1 в любой комнате.
    """
    if room_type in ("bathroom", "toilet"):
        return 1
    if room_type == "corridor":
        return max(1, round(area_m2 / 8.0))   # коридор — чаще точечные, каждые 8м²
    if room_type == "kitchen":
        return max(1, round(area_m2 / 10.0))  # кухня — рабочая зона требует больше
    # Жилые комнаты: 1 светильник на 15-20 м²
    base = area_m2 / 16.0
    count = max(1, round(base))
    # Добавляем случайность ±1
    count = count + random.randint(-1, 1)
    return max(1, count)


def _sockets_from_area(area_m2: float, room_type: str) -> int:
    """
    ПУЭ и практика: 1 розетка на каждые 6 м² площади комнаты, минимум 1.
    Для гостиной/спальни — дополнительно у каждой стены минимум 1.
    """
    if room_type in ("bathroom", "toilet", "corridor"):
        return 0  # power_socket не в санузлах и коридоре
    base = area_m2 / 6.0
    count = max(1, round(base))
    count = count + random.randint(0, 1)
    return max(1, min(count, 12))  # не больше 12


def _smoke_from_area(area_m2: float, room_type: str) -> int:
    """
    СП 484.1311500.2020:
    - Жилые комнаты >12м² — 1 датчик
    - Коридор — 1 датчик (путь эвакуации)
    - Кухня, санузел — НЕ ставится
    - Площадь защиты 1 датчика = 85м², но при высоте 2.5м практически 1 на комнату
    """
    if room_type in ("kitchen", "bathroom", "toilet", "balcony"):
        return 0
    if room_type == "corridor":
        return 1
    if area_m2 < 12:
        return random.randint(0, 1)
    if area_m2 < 85:
        return 1
    # Большие помещения >85м² — 2 датчика
    return 2


# ─── Базовые правила (min, max) ──────────────────────────────────────────────

BASE_RULES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "living_room": {
        "tv_sockets":       (1, 2),
        "internet_sockets": (1, 1),
        "smoke_detector":   (1, 2),
        "co2_detector":     (0, 1),
        "ceiling_lights":   (1, 10),  # до 10 для больших комнат
        "night_lights":     (0, 0),
    },
    "bedroom": {
        "tv_sockets":       (0, 1),
        "internet_sockets": (0, 0),
        "smoke_detector":   (0, 1),
        "co2_detector":     (0, 0),
        "ceiling_lights":   (1, 4),
        "night_lights":     (1, 2),
    },
    "kitchen": {
        "tv_sockets":       (0, 1),
        "internet_sockets": (0, 0),
        "smoke_detector":   (0, 0),
        "co2_detector":     (1, 1),
        "ceiling_lights":   (1, 3),
        "night_lights":     (0, 0),
    },
    "bathroom": {
        "tv_sockets":       (0, 0),
        "internet_sockets": (0, 0),
        "smoke_detector":   (0, 0),
        "co2_detector":     (0, 0),
        "ceiling_lights":   (1, 1),
        "night_lights":     (0, 0),
    },
    "toilet": {
        "tv_sockets":       (0, 0),
        "internet_sockets": (0, 0),
        "smoke_detector":   (0, 0),
        "co2_detector":     (0, 0),
        "ceiling_lights":   (1, 1),
        "night_lights":     (0, 0),
    },
    "corridor": {
        "tv_sockets":       (0, 0),
        "internet_sockets": (0, 1),
        "smoke_detector":   (1, 1),
        "co2_detector":     (0, 0),
        "ceiling_lights":   (1, 3),
        "night_lights":     (1, 1),
    },
}


def _assign_internet_socket(nodes: List[Dict]) -> Dict[str, int]:
    priority = ["corridor", "living_room"]
    result = {n["room_id"]: 0 for n in nodes}
    for ptype in priority:
        candidates = [n for n in nodes if n["room_type"] == ptype]
        if candidates:
            chosen = random.choice(candidates)
            result[chosen["room_id"]] = 1
            return result
    result[nodes[0]["room_id"]] = 1
    return result


AREA_RANGES: Dict[str, Tuple[float, float]] = {
    "living_room": (15.0, 50.0),
    "bedroom":     (9.0,  25.0),
    "kitchen":     (6.0,  15.0),
    "bathroom":    (3.0,  8.0),
    "toilet":      (1.5,  4.0),
    "corridor":    (3.0,  12.0),
}

APARTMENT_CONFIGS = [
    ("1к",  ["living_room", "kitchen", "bathroom", "corridor"]),
    ("2к",  ["living_room", "bedroom", "kitchen", "bathroom", "corridor"]),
    ("2к+", ["living_room", "bedroom", "kitchen", "bathroom", "toilet", "corridor"]),
    ("3к",  ["living_room", "bedroom", "bedroom", "kitchen", "bathroom", "toilet", "corridor"]),
    ("3к+", ["living_room", "bedroom", "bedroom", "kitchen", "bathroom", "toilet", "corridor", "corridor"]),
    ("4к",  ["living_room", "bedroom", "bedroom", "bedroom", "kitchen", "bathroom", "toilet", "corridor"]),
]


def _gen_apartment() -> Tuple[List[Dict], List[Dict]]:
    _, room_types = random.choice(APARTMENT_CONFIGS)
    nodes = []
    for i, rtype in enumerate(room_types):
        area = round(random.uniform(*AREA_RANGES[rtype]), 1)
        is_ext = rtype in ("living_room", "bedroom", "kitchen")
        nodes.append({
            "room_id":      f"room_{i}",
            "room_type":    rtype,
            "area_m2":      area,
            "n_windows":    random.randint(1, 3) if is_ext else 0,
            "n_doors":      random.randint(1, 2),
            "is_exterior":  is_ext,
            "has_entrance": (rtype == "corridor" and i == len(room_types) - 1),
        })

    edges = []
    corridor_ids  = [i for i, n in enumerate(nodes) if n["room_type"] == "corridor"]
    non_corridor  = [i for i, n in enumerate(nodes) if n["room_type"] != "corridor"]

    if corridor_ids:
        c = corridor_ids[0]
        for i in non_corridor:
            edges += [{"from": c, "to": i}, {"from": i, "to": c}]
        for i in range(1, len(corridor_ids)):
            edges += [{"from": corridor_ids[0], "to": corridor_ids[i]},
                      {"from": corridor_ids[i], "to": corridor_ids[0]}]
    else:
        for i in range(len(nodes) - 1):
            edges += [{"from": i, "to": i+1}, {"from": i+1, "to": i}]

    return nodes, edges


def _gen_labels(nodes: List[Dict], preferences: Optional[Dict] = None) -> Dict[str, Dict[str, int]]:
    prefs = preferences or {}
    internet_map = _assign_internet_socket(nodes)
    labels: Dict[str, Dict[str, int]] = {}

    for node in nodes:
        rtype   = node["room_type"]
        room_id = node["room_id"]
        area    = node["area_m2"]
        rules   = BASE_RULES.get(rtype, {})
        room_prefs = prefs.get(rtype, {})
        room_labels: Dict[str, int] = {}

        for device in DEVICES:
            mn, mx = rules.get(device, (0, 0))

            if device == "internet_sockets":
                room_labels[device] = internet_map.get(room_id, 0)
                continue

            if mx == 0:
                room_labels[device] = 0
                continue

            # Пожелание пользователя имеет приоритет
            if device in room_prefs:
                room_labels[device] = max(mn, min(int(room_prefs[device]), mx))
                continue

            # Нормативный расчёт по площади
            if device == "ceiling_lights":
                count = _lights_from_area(area, rtype)
                room_labels[device] = max(mn, min(count, mx))
                continue

            if device == "smoke_detector":
                count = _smoke_from_area(area, rtype)
                room_labels[device] = max(0, min(count, mx))
                continue

            room_labels[device] = random.randint(mn, mx)

        labels[room_id] = room_labels

    return labels


def _gen_preferences() -> Dict[str, Any]:
    prefs: Dict[str, Any] = {"global": {}}
    if random.random() < 0.3:
        prefs["living_room"] = {"tv_sockets": random.randint(1, 2)}
    if random.random() < 0.2:
        prefs["bedroom"] = {"tv_sockets": 1}
    return prefs


def _gen_sample() -> Dict[str, Any]:
    nodes, edges = _gen_apartment()
    preferences  = _gen_preferences()
    labels       = _gen_labels(nodes, preferences)
    return {"nodes": nodes, "edges": edges, "labels": labels, "preferences": preferences}


def generate_dataset(out_dir: str, count: int, seed: int = 42) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    samples  = [_gen_sample() for _ in range(count)]
    random.shuffle(samples)
    n_val    = max(1, count // 10)
    n_train  = len(samples) - n_val

    def write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write(os.path.join(out_dir, "train.jsonl"), samples[:n_train])
    write(os.path.join(out_dir, "val.jsonl"),   samples[n_train:])

    meta = {"devices": DEVICES, "room_types": ROOM_TYPES,
            "count": count, "n_train": n_train, "n_val": n_val}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset generated: train={n_train}, val={n_val}")
    print("\nExample (first apartment):")
    s = samples[0]
    for node in s["nodes"]:
        rid = node["room_id"]
        print(f"  {rid} ({node['room_type']:12s}) area={node['area_m2']:5.1f}m2 -> {s['labels'][rid]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="data/nn3_dataset")
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out, args.count, args.seed)