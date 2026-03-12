"""
app/nn3/dataset_gen.py — реалистичные правила размещения слаботочных устройств
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# ─── устройства и типы комнат ─────────────────────────────────────────────────

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

# ─── реалистичные правила (min, max) ─────────────────────────────────────────
#
# internet_sockets = розетка RJ-45 под патч-корд к роутеру.
# Роутер ОДИН на квартиру — ставится в гостиной или коридоре.
# В спальнях розетка RJ-45 не нужна — там WiFi.
#
# smoke_detector — потолочный, по СП 484.1311500:
#   обязателен в жилых комнатах и коридорах.
#   НЕ ставится: кухня (пар/дым от готовки = ложные срабатывания),
#                санузел, туалет.
#
# co2_detector — актуален там где газ или плохая вентиляция:
#   кухня (газовая плита), иногда гостиная.
#   В спальнях по желанию (умный дом).
#
# night_lights — розеточный ночник, только спальня и коридор.
# tv_sockets   — только там где реально ставят TV.

BASE_RULES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "living_room": {
        "tv_sockets":       (1, 2),   # 1-2 точки под TV
        "internet_sockets": (1, 1),   # 1 розетка RJ-45 — сюда идёт роутер
        "smoke_detector":   (1, 1),   # обязателен
        "co2_detector":     (0, 1),   # по желанию
        "ceiling_lights":   (1, 4),   # от площади
        "night_lights":     (0, 0),   # не нужен в гостиной
    },
    "bedroom": {
        "tv_sockets":       (0, 1),   # иногда TV в спальне
        "internet_sockets": (0, 0),   # WiFi достаточно, RJ-45 не нужен
        "smoke_detector":   (1, 1),   # обязателен по СП
        "co2_detector":     (0, 0),   # не актуален в спальне
        "ceiling_lights":   (1, 2),
        "night_lights":     (1, 2),   # ночник у кровати — стандарт
    },
    "kitchen": {
        "tv_sockets":       (0, 1),   # маленький TV на кухне — бывает
        "internet_sockets": (0, 0),   # не нужен
        "smoke_detector":   (0, 0),   # НЕ ставится (ложные срабатывания от готовки)
        "co2_detector":     (1, 1),   # обязателен при газовой плите
        "ceiling_lights":   (1, 2),
        "night_lights":     (0, 0),
    },
    "bathroom": {
        "tv_sockets":       (0, 0),
        "internet_sockets": (0, 0),
        "smoke_detector":   (0, 0),   # влажность = ложные срабатывания
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
        "internet_sockets": (0, 1),   # альтернативное место роутера
        "smoke_detector":   (1, 1),   # обязателен — путь эвакуации
        "co2_detector":     (0, 0),
        "ceiling_lights":   (1, 2),
        "night_lights":     (1, 1),   # ночник в коридоре — стандарт
    },
}

# ─── интернет-розетка: только одна на всю квартиру ───────────────────────────

def _assign_internet_socket(nodes: List[Dict]) -> Dict[str, int]:
    """
    Один RJ-45 на квартиру — в гостиной или коридоре.
    Возвращает {room_id: 1} для выбранной комнаты, остальные = 0.
    """
    # Приоритет: corridor → living_room → первая попавшаяся из допустимых
    priority = ["corridor", "living_room"]
    result = {n["room_id"]: 0 for n in nodes}

    for ptype in priority:
        candidates = [n for n in nodes if n["room_type"] == ptype]
        if candidates:
            chosen = random.choice(candidates)
            result[chosen["room_id"]] = 1
            return result

    # Fallback — вообще нет коридора и гостиной (редкость)
    result[nodes[0]["room_id"]] = 1
    return result


# ─── площадь → количество светильников ───────────────────────────────────────

def _lights_from_area(area_m2: float) -> int:
    if area_m2 < 8:
        return 1
    elif area_m2 < 15:
        return random.randint(1, 2)
    elif area_m2 < 25:
        return random.randint(2, 3)
    else:
        return random.randint(3, 4)


# ─── генератор квартиры ───────────────────────────────────────────────────────

AREA_RANGES: Dict[str, Tuple[float, float]] = {
    "living_room": (15.0, 40.0),
    "bedroom":     (9.0,  20.0),
    "kitchen":     (6.0,  15.0),
    "bathroom":    (3.0,  8.0),
    "toilet":      (1.5,  4.0),
    "corridor":    (3.0,  10.0),
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
            "room_id":     f"room_{i}",
            "room_type":   rtype,
            "area_m2":     area,
            "n_windows":   random.randint(1, 3) if is_ext else 0,
            "n_doors":     random.randint(1, 2),
            "is_exterior": is_ext,
            "has_entrance": (rtype == "corridor" and i == len(room_types) - 1),
        })

    edges = []
    corridor_ids = [i for i, n in enumerate(nodes) if n["room_type"] == "corridor"]
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
    internet_map = _assign_internet_socket(nodes)  # один RJ-45 на квартиру
    labels: Dict[str, Dict[str, int]] = {}

    for node in nodes:
        rtype   = node["room_type"]
        room_id = node["room_id"]
        rules   = BASE_RULES.get(rtype, {})
        room_prefs = prefs.get(rtype, {})
        room_labels: Dict[str, int] = {}

        for device in DEVICES:
            mn, mx = rules.get(device, (0, 0))

            # internet_sockets — жёсткий контроль: только из internet_map
            if device == "internet_sockets":
                room_labels[device] = internet_map.get(room_id, 0)
                continue

            if mx == 0:
                room_labels[device] = 0
                continue

            # Пожелание клиента
            if device in room_prefs:
                pref_val = int(room_prefs[device])
                room_labels[device] = max(mn, min(pref_val, mx + 1))
                continue

            # Светильники от площади
            if device == "ceiling_lights":
                lights = _lights_from_area(node["area_m2"])
                room_labels[device] = max(mn, min(lights, mx))
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
    if random.random() < 0.15:
        prefs["global"]["alarm"] = True
    return prefs


def _gen_sample() -> Dict[str, Any]:
    nodes, edges = _gen_apartment()
    preferences  = _gen_preferences()
    labels       = _gen_labels(nodes, preferences)
    return {"nodes": nodes, "edges": edges, "labels": labels, "preferences": preferences}


def generate_dataset(out_dir: str, count: int, seed: int = 42) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    samples = [_gen_sample() for _ in range(count)]
    random.shuffle(samples)
    n_val   = max(1, count // 10)
    n_train = len(samples) - n_val

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

    print(f"Датасет NN-3 сгенерирован: train={n_train}, val={n_val}")
    print("\nПример (первая квартира):")
    s = samples[0]
    for node in s["nodes"]:
        rid = node["room_id"]
        print(f"  {rid} ({node['room_type']:12s}) area={node['area_m2']:5.1f}м² → {s['labels'][rid]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="data/nn3_dataset")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out, args.count, args.seed)