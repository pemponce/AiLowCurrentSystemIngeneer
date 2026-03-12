"""
app/nn2/dataset_gen.py

Генератор датасета для NN-2 v2 (NER / sequence labeling).

Формат: JSONL, каждая строка:
{
  "tokens": ["нужен", "интернет", "в", "спальне", "и", "на", "кухне"],
  "tags":   ["O", "DEV-B-internet_socket", "O", "ROOM-B-bedroom", "O", "O", "ROOM-B-kitchen"]
}

Схема тегов:
  O                    — не важный токен
  ROOM-{type}-B        — первый токен названия комнаты
  ROOM-{type}-I        — продолжение названия комнаты
  DEV-{device}-B       — первый токен устройства
  DEV-{device}-I       — продолжение
  COUNT-{n}            — число (1..6)
  NEG                  — отрицание (кроме, без, не)
  GLOBAL-{device}-B/I  — глобальное устройство (домофон, сигнализация)

Запуск:
  python -m app.nn2.dataset_gen --out data/nn2_dataset --count 10000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

Token  = str
Tag    = str
Sample = Dict

# ─── словари комнат ───────────────────────────────────────────────────────────

ROOM_TOKENS: Dict[str, List[List[str]]] = {
    "living_room": [
        ["в", "гостиной"], ["в", "зале"], ["в", "гостинке"],
        ["в", "гостиную"], ["в", "большой", "комнате"], ["в", "зал"],
    ],
    "bedroom": [
        ["в", "спальне"], ["в", "спальню"], ["в", "детской"],
        ["в", "детскую"], ["в", "главной", "спальне"],
    ],
    "kitchen": [
        ["на", "кухне"], ["в", "кухне"], ["на", "кухню"],
        ["на", "кухонном", "острове"],
    ],
    "bathroom": [
        ["в", "ванной"], ["в", "ванной", "комнате"],
        ["в", "совмещённом", "санузле"], ["в", "ванну"],
    ],
    "toilet": [
        ["в", "туалете"], ["в", "уборной"], ["в", "санузле"],
    ],
    "corridor": [
        ["в", "коридоре"], ["в", "прихожей"], ["в", "холле"],
        ["у", "входа"], ["в", "прихожую"],
    ],
}

DEVICE_TOKENS: Dict[str, List[List[str]]] = {
    "tv_socket": [
        ["розетку", "для", "телевизора"], ["ТВ-розетку"],
        ["розетку", "под", "телик"], ["антенну", "для", "телевизора"],
        ["розетку", "для", "ТВ"], ["телевизионную", "розетку"],
    ],
    "internet_socket": [
        ["интернет-розетку"], ["розетку", "для", "интернета"],
        ["LAN-розетку"], ["розетку", "под", "компьютер"],
        ["сетевую", "розетку"], ["розетку", "RJ-45"],
        ["витую", "пару"], ["интернет"],
    ],
    "smoke_detector": [
        ["датчик", "дыма"], ["датчики", "дыма"],
        ["пожарный", "датчик"], ["детектор", "дыма"],
        ["датчик", "пожара"], ["дымовой", "датчик"],
    ],
    "co2_detector": [
        ["датчик", "CO2"], ["датчик", "углекислого", "газа"],
        ["датчик", "воздуха"], ["датчик", "качества", "воздуха"],
    ],
    "ceiling_light": [
        ["светильник"], ["светильники"], ["люстру"],
        ["освещение"], ["потолочный", "свет"],
        ["точечные", "светильники"],
    ],
    "night_light": [
        ["ночник"], ["бра"], ["прикроватный", "светильник"],
    ],
    "motion_sensor": [
        ["датчик", "движения"], ["датчики", "движения"],
        ["сенсор", "движения"],
    ],
}

GLOBAL_DEVICE_TOKENS: Dict[str, List[List[str]]] = {
    "intercom": [
        ["домофон"], ["видеодомофон"],
        ["панель", "домофона"], ["переговорное", "устройство"],
    ],
    "alarm": [
        ["охранную", "сигнализацию"], ["сигнализацию"], ["охрану"],
    ],
    "smart_home": [
        ["умный", "дом"], ["систему", "умного", "дома"],
        ["автоматизацию"],
    ],
}

COUNT_TOKENS: Dict[int, List[str]] = {
    1: ["одну", "один", "одно", "1"],
    2: ["две", "два", "2", "пару"],
    3: ["три", "3"],
    4: ["четыре", "4"],
    5: ["пять", "5"],
}

CONNECTORS  = [["и"], [","], ["также"], ["плюс"], ["ещё"], ["а", "также"]]
INTROS      = [["хочу"], ["нужно"], ["прошу", "поставить"],
               ["установите"], ["нужны"], ["поставьте"], ["хотелось", "бы"], []]
NEG_TOKENS  = [["кроме"], ["без"], ["исключая"], ["не", "включая"]]

# Родительные падежи комнат — используются после "кроме/без"
ROOM_GENITIVE: Dict[str, List[List[str]]] = {
    "living_room": [["гостиной"], ["зала"], ["гостинки"]],
    "bedroom":     [["спальни"], ["детской"]],
    "kitchen":     [["кухни"], ["кухонной", "зоны"]],
    "bathroom":    [["ванной"], ["санузла"]],
    "toilet":      [["туалета"], ["уборной"]],
    "corridor":    [["коридора"], ["прихожей"], ["холла"]],
}


# ─── хелперы ─────────────────────────────────────────────────────────────────

def bio(tokens: List[str], prefix: str) -> Tuple[List[Token], List[Tag]]:
    if not tokens:
        return [], []
    tags = [f"{prefix}-B"] + [f"{prefix}-I"] * (len(tokens) - 1)
    return tokens, tags

def o(tokens: List[str]) -> Tuple[List[Token], List[Tag]]:
    return tokens, ["O"] * len(tokens)

def room(rtype: str) -> Tuple[List[Token], List[Tag]]:
    return bio(random.choice(ROOM_TOKENS[rtype]), f"ROOM-{rtype}")

def device(dtype: str) -> Tuple[List[Token], List[Tag]]:
    return bio(random.choice(DEVICE_TOKENS[dtype]), f"DEV-{dtype}")

def gdevice(dtype: str) -> Tuple[List[Token], List[Tag]]:
    return bio(random.choice(GLOBAL_DEVICE_TOKENS[dtype]), f"GLOBAL-{dtype}")

def count(n: int) -> Tuple[List[Token], List[Tag]]:
    return [random.choice(COUNT_TOKENS[n])], [f"COUNT-{n}"]

def neg() -> Tuple[List[Token], List[Tag]]:
    p = random.choice(NEG_TOKENS)
    return p, ["NEG"] * len(p)

def build(parts) -> Sample:
    tokens, tags = [], []
    for t, g in parts:
        if t:
            tokens.extend(t)
            tags.extend(g)
    assert len(tokens) == len(tags)
    return {"tokens": tokens, "tags": tags}

# ─── генераторы ───────────────────────────────────────────────────────────────

def gen_single() -> Sample:
    """в гостиной [2] розетки для телевизора"""
    rtype = random.choice(list(ROOM_TOKENS))
    dtype = random.choice(list(DEVICE_TOKENS))
    parts = [o(random.choice(INTROS)), room(rtype)]
    if random.random() < 0.4:
        parts.append(count(random.randint(1, 3)))
    parts.append(device(dtype))
    if random.random() < 0.3:
        dtype2 = random.choice([d for d in DEVICE_TOKENS if d != dtype])
        parts.append(o(random.choice(CONNECTORS)))
        parts.append(device(dtype2))
    return build(parts)

def gen_two_rooms() -> Sample:
    """в спальне интернет и на кухне датчик дыма"""
    rooms = random.sample(list(ROOM_TOKENS), 2)
    parts = [o(random.choice(INTROS))]
    for i, rtype in enumerate(rooms):
        if i > 0:
            parts.append(o(random.choice(CONNECTORS)))
        parts.append(room(rtype))
        dtype = random.choice(list(DEVICE_TOKENS))
        if random.random() < 0.35:
            parts.append(count(random.randint(1, 3)))
        parts.append(device(dtype))
    return build(parts)

def gen_three_rooms() -> Sample:
    """в гостиной ТВ, в спальне интернет, в коридоре датчик"""
    rooms = random.sample(list(ROOM_TOKENS), 3)
    parts = []
    for i, rtype in enumerate(rooms):
        if i > 0:
            parts.append(o([","]))
        parts.append(room(rtype))
        parts.append(device(random.choice(list(DEVICE_TOKENS))))
    return build(parts)

def gen_device_multi_rooms() -> Sample:
    """датчики дыма в спальне и в коридоре"""
    dtype = random.choice(list(DEVICE_TOKENS))
    rooms = random.sample(list(ROOM_TOKENS), random.randint(2, 3))
    parts = [o(random.choice(INTROS)), device(dtype)]
    for i, rtype in enumerate(rooms):
        if i > 0:
            parts.append(o(random.choice(CONNECTORS)))
        parts.append(room(rtype))
    return build(parts)

def room_genitive(rtype: str) -> Tuple[List[Token], List[Tag]]:
    """Комната в родительном падеже для конструкций 'кроме X'."""
    return bio(random.choice(ROOM_GENITIVE[rtype]), f"ROOM-{rtype}")

def gen_negation() -> Sample:
    """датчики дыма везде кроме кухни [и охрану]"""
    dtype = random.choice(["smoke_detector", "co2_detector", "motion_sensor"])
    rtype = random.choice(list(ROOM_TOKENS))
    parts = [
        o(random.choice(INTROS)),
        device(dtype),
        o(["везде"]),
        neg(),
        room_genitive(rtype),
    ]
    # Иногда добавляем глобальное устройство
    if random.random() < 0.4:
        gtype = random.choice(list(GLOBAL_DEVICE_TOKENS))
        parts.append(o(random.choice(CONNECTORS)))
        parts.append(gdevice(gtype))
    return build(parts)

def gen_global() -> Sample:
    """нужен домофон и охранная сигнализация"""
    devs = random.sample(list(GLOBAL_DEVICE_TOKENS), random.randint(1, 2))
    parts = [o(random.choice(INTROS))]
    for i, d in enumerate(devs):
        if i > 0:
            parts.append(o(random.choice(CONNECTORS)))
        parts.append(gdevice(d))
    return build(parts)

def gen_mixed() -> Sample:
    """хочу в гостиной ТВ и домофон"""
    rtype = random.choice(list(ROOM_TOKENS))
    dtype = random.choice(list(DEVICE_TOKENS))
    gtype = random.choice(list(GLOBAL_DEVICE_TOKENS))
    return build([
        o(random.choice(INTROS)),
        room(rtype),
        device(dtype),
        o(random.choice(CONNECTORS)),
        gdevice(gtype),
    ])

def gen_count_explicit() -> Sample:
    """поставьте 3 датчика дыма в гостиной"""
    dtype = random.choice(list(DEVICE_TOKENS))
    rtype = random.choice(list(ROOM_TOKENS))
    n     = random.randint(1, 4)
    return build([
        o(random.choice(INTROS)),
        count(n),
        device(dtype),
        room(rtype),
    ])

# ─── сборка ───────────────────────────────────────────────────────────────────

GENERATORS = [
    (gen_single,            30),
    (gen_two_rooms,         22),
    (gen_device_multi_rooms,15),
    (gen_three_rooms,       10),
    (gen_count_explicit,     8),
    (gen_negation,           5),
    (gen_global,             5),
    (gen_mixed,              5),
]

def generate_dataset(out_dir: str, count_: int, seed: int = 42) -> None:
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    fns, weights = zip(*GENERATORS)
    samples = []
    for _ in range(count_):
        fn = random.choices(fns, weights=weights, k=1)[0]
        s  = fn()
        if s["tokens"]:
            samples.append(s)

    random.shuffle(samples)
    n_val   = max(1, count_ // 10)
    n_train = len(samples) - n_val

    def write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write(os.path.join(out_dir, "train.jsonl"), samples[:n_train])
    write(os.path.join(out_dir, "val.jsonl"),   samples[n_train:])

    tag_set = set()
    for s in samples:
        tag_set.update(s["tags"])
    tag_list = sorted(tag_set)
    with open(os.path.join(out_dir, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(tag_list, f, ensure_ascii=False, indent=2)

    print(f"Датасет сгенерирован:")
    print(f"  train: {n_train} → {out_dir}/train.jsonl")
    print(f"  val:   {n_val}   → {out_dir}/val.jsonl")
    print(f"  тегов: {len(tag_list)}")
    print("\nПримеры:")
    for s in samples[:4]:
        print(f"\n  TOKENS: {s['tokens']}")
        print(f"  TAGS:   {s['tags']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="data/nn2_dataset")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out, args.count, args.seed)