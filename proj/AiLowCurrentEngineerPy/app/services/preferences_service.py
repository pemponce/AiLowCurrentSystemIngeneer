# app/services/preferences_service.py
"""
Парсинг пожеланий клиента (numbered preferences).

Поддерживает формат:
  "1: телевизор, свет; 2: свет, 2 ночника; 3: ничего"
"""

import re
import logging

logger = logging.getLogger("planner")


def parse_numbered_preferences(text: str, room_map: dict) -> dict:
    """
    Парсит текст пожеланий с номерами комнат.

    Args:
        text: Текст вида "1: свет розетки; 2: ничего; 3: свет 2 розетки"
        room_map: Маппинг {1: "room_000", 2: "room_001", ...}

    Returns:
        PreferencesGraph dict с маркерами "_skip" для комнат с "ничего"

    Supported patterns:
        - "2 ночника" → night_lights: 2
        - "2-4 источника света" → ceiling_lights: 3 (среднее)
        - "телевизор" → tv_sockets: 1
        - "ничего", "пусто", "без" → {"_skip": True}
    """

    # Устройства: список (паттерн_regex, device_key)
    DEVICE_PATTERNS = [
        (r"датчик\s*дыма", "smoke_detector"),
        (r"датчик\s*co2", "co2_detector"),
        (r"датчик\s*угарного", "co2_detector"),
        (r"углекислый", "co2_detector"),
        (r"co2", "co2_detector"),
        (r"газовый\s*датчик", "co2_detector"),
        (r"источник(?:а|ов)?\s*света", "ceiling_lights"),
        (r"светильник(?:а|ов)?", "ceiling_lights"),
        (r"люстр(?:а|ы)?", "ceiling_lights"),
        (r"лампоч(?:ка|ки|ек)?", "ceiling_lights"),
        (r"свет(?:овых|овые)?", "ceiling_lights"),
        (r"подсветк(?:а|и)?", "night_lights"),
        (r"розетк(?:а|и|у|ой)?", "power_socket"),
        (r"\bsocket\b", "power_socket"),
        (r"тв\b", "tv_sockets"),
        (r"\btv\b", "tv_sockets"),
        (r"интернет", "internet_sockets"),
        (r"роутер", "internet_sockets"),
        (r"\blan\b", "internet_sockets"),
        (r"вайфай", "internet_sockets"),
        (r"\bwifi\b", "internet_sockets"),
        (r"дым\b", "smoke_detector"),
        (r"пожарн", "smoke_detector"),
    ]

    def _parse_count(token: str) -> int:
        """Извлекает число из токена: '2', '2-4' → среднее=3, 'два'=2."""
        WORDS = {
            "один": 1, "одна": 1, "одного": 1,
            "два": 2, "две": 2,
            "трёх": 3, "три": 3,
            "четыре": 4, "четырёх": 4,
            "пять": 5
        }
        t = token.strip().lower()

        # диапазон "2-4"
        m = re.match(r"(\d+)\s*[-–—]\s*(\d+)", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return max(1, round((a + b) / 2))

        # просто число
        m = re.match(r"(\d+)", t)
        if m:
            return max(1, int(m.group(1)))

        # слово
        for w, n in WORDS.items():
            if w in t:
                return n
        return 1

    def _parse_room_segment(seg: str) -> dict:
        """Парсит строку одной комнаты → {device: count}."""
        result = {}
        seg_lo = seg.lower()

        for pattern, device in DEVICE_PATTERNS:
            for m in re.finditer(pattern, seg_lo):
                start = m.start()
                # Смотрим что стоит ПЕРЕД паттерном (число/диапазон)
                prefix = seg_lo[max(0, start - 12):start].strip()
                prefix = re.sub(r"[,;]", " ", prefix).strip()
                tokens = prefix.split()
                count = 1
                if tokens:
                    count = _parse_count(tokens[-1])
                result[device] = max(result.get(device, 0), count)

        return result

    # Разбиваем на сегменты по ";" или "\n"
    segments = [s.strip() for s in re.split(r"[;\n]", text) if s.strip()]
    rooms_prefs = {}

    for seg in segments:
        # Ищем номер комнаты: "1:", "комната 2:", "1 -"
        m = re.match(
            r"(?:комната\s*)?(\d+)\s*[:\-–—]?\s*(.*)",
            seg.strip(),
            re.IGNORECASE | re.DOTALL
        )
        if not m:
            continue

        num = int(m.group(1))
        room_body = m.group(2).strip()
        room_id = room_map.get(num)

        if not room_id or not room_body:
            continue

        # Проверка на "ничего", "пусто", "без"
        room_body_lower = room_body.lower()
        if any(word in room_body_lower for word in ["ничего", "пусто", "без", "none", "empty", "skip"]):
            rooms_prefs[room_id] = {"_skip": True}
            continue

        devs = _parse_room_segment(room_body)
        if devs:
            rooms_prefs[room_id] = devs

    rooms_list = [{"roomId": rid, "devices": devs} for rid, devs in rooms_prefs.items()]

    return {
        "version": "preferences-1.0",
        "sourceText": text,
        "global": {},
        "rooms": rooms_list,
        "_by_room_id": rooms_prefs,
    }