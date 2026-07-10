# app/services/preferences_service.py
"""
Парсинг пожеланий клиента (numbered preferences) v2.

НОВОЕ: Поддержка названий комнат!

Поддерживает форматы:
  "1: телевизор, свет; 2: свет, 2 ночника"           ← цифры (старый)
  "прихожая: свет, розетки; туалет: свет"           ← названия (новый!)
  "1=прихожая: свет; 2=туалет: свет"                ← смешанный
"""

import re
import logging

logger = logging.getLogger("planner")

# ========== СЛОВАРЬ СИНОНИМОВ КОМНАТ ==========

ROOM_TYPE_ALIASES = {
    # Гостиная
    "гостиная": "living_room",
    "гостинная": "living_room",
    "гостинка": "living_room",
    "зал": "living_room",
    "холл": "living_room",
    "гостиную": "living_room",
    "зала": "living_room",

    # Спальня
    "спальня": "bedroom",
    "спалня": "bedroom",
    "спальню": "bedroom",
    "комната": "bedroom",
    "детская": "bedroom",
    "детскую": "bedroom",

    # Кухня
    "кухня": "kitchen",
    "кух": "kitchen",
    "кухню": "kitchen",
    "кухне": "kitchen",

    # Ванная
    "ванная": "bathroom",
    "ванна": "bathroom",
    "ванну": "bathroom",
    "ванной": "bathroom",

    # Туалет
    "туалет": "toilet",
    "санузел": "toilet",
    "уборная": "toilet",
    "wc": "toilet",
    "туалете": "toilet",

    # Прихожая / Коридор
    "прихожая": "corridor",
    "коридор": "corridor",
    "прихожей": "corridor",
    "прих": "corridor",
    "коридоре": "corridor",

    # Балкон
    "балкон": "balcony",
    "лоджия": "balcony",
    "балконе": "balcony",
}


def parse_numbered_preferences(text: str, room_map: dict, room_type_map: dict = None, project_id: str = None) -> dict:
    """
    Парсит текст пожеланий с номерами или названиями комнат.

    Args:
        text: Текст вида:
            - "1: свет розетки; 2: ничего; 3: свет 2 розетки"          (цифры)
            - "прихожая: свет, розетки; туалет: свет"                  (названия)
            - "1=прихожая: свет; 2=туалет: свет"                       (смешанный)
            - "1 (Прихожая): свет, розетки; 2: свет"                   (номер + подпись)
            - "1 (Прихожая) [щиток]: свет, розетки; 2: свет"           (с указанием щитка)
        room_map: Маппинг {1: "room_000", 2: "room_001", ...}
        room_type_map: Маппинг {"room_000": "living_room", ...}
        project_id: ID проекта (для сохранения позиции щитка в DB["panel"])

    Returns:
        PreferencesGraph dict с маркерами "_skip" для комнат с "ничего"
    """
    room_type_map = room_type_map or {}


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

    logger.info(f"PARSE START: text='{text}'")
    logger.info(f"  room_map={room_map}")
    logger.info(f"  room_type_map={room_type_map}")


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

    def _find_room_by_type(room_type: str) -> str:
        """Находит первую комнату указанного типа."""
        for room_id, rtype in room_type_map.items():
            if rtype == room_type:
                return room_id
        # Fallback: если не нашли — возвращаем первую комнату
        if room_map:
            return room_map.get(1, "")
        return ""

    # Разбиваем на сегменты по ";" или "\n"
    segments = [s.strip() for s in re.split(r"[;\n]", text) if s.strip()]
    rooms_prefs = {}
    panel_room_id = None   # Комната со щитком (если указано [щиток])

    for seg in segments:
        # ========== ПАРСИНГ КОМНАТЫ ==========

        # Вариант 1: "1=прихожая: свет" (явное указание)
        m_explicit = re.match(
            r"(\d+)\s*=\s*([а-яА-ЯёЁa-zA-Z]+)\s*:\s*(.*)",
            seg.strip(),
            re.IGNORECASE | re.DOTALL
        )

        # Вариант 2: "прихожая: свет" (только название)
        m_name = re.match(
            r"([а-яА-ЯёЁa-zA-Z]+)\s*:\s*(.*)",
            seg.strip(),
            re.IGNORECASE | re.DOTALL
        )

        # Вариант 3: "1 (Прихожая): свет" или "1: свет" (номер, опционально с подписью)
        m_num = re.match(
            r"(\d+)\s*(?:\([^)]*\))?\s*(?:\[[^\]]*\])?\s*:\s*(.*)",
            seg.strip(),
            re.IGNORECASE | re.DOTALL
        )

        room_id = None
        room_body = None

        if m_explicit:
            # Формат: "1=прихожая: свет"
            num = int(m_explicit.group(1))
            room_name = m_explicit.group(2).strip().lower()
            room_body = m_explicit.group(3).strip()

            # Используем номер из room_map
            room_id = room_map.get(num)

            logger.debug(f"Parsed explicit: num={num}, name={room_name}, room_id={room_id}")

        elif m_name and not m_num:
            # Формат: "прихожая: свет" (только название, без цифры)
            room_name = m_name.group(1).strip().lower()
            room_body = m_name.group(2).strip()

            # Ищем тип комнаты по названию
            room_type = ROOM_TYPE_ALIASES.get(room_name)
            if room_type:
                # Находим первую комнату этого типа
                room_id = _find_room_by_type(room_type)
                logger.debug(f"Parsed name: name={room_name}, type={room_type}, room_id={room_id}")
            else:
                logger.warning(f"Unknown room name: {room_name}")
                continue

        elif m_num:
            # Формат: "1 (Прихожая) [щиток]: свет" или "1: свет"
            num = int(m_num.group(1))
            room_body = m_num.group(2).strip()
            room_id = room_map.get(num)

            # Проверяем наличие [щиток] в оригинальном сегменте
            if re.search(r"\[\s*щиток\s*\]", seg, re.IGNORECASE):
                panel_room_id = room_id
                logger.info(f"Panel room detected: room_id={room_id} (from '[щиток]' tag)")

            logger.debug(f"Parsed number: num={num}, room_id={room_id}")

        if not room_id or not room_body:
            continue

        # ========== ПАРСИНГ УСТРОЙСТВ ==========

        # Проверка на "ничего", "пусто", "без"
        room_body_lower = room_body.lower()
        if any(word in room_body_lower for word in ["ничего", "пусто", "без", "none", "empty", "skip"]):
            rooms_prefs[room_id] = {"_skip": True}
            logger.info(f"Parsed room {room_id}: SKIP")
            continue

        devs = _parse_room_segment(room_body)
        if devs:
            logger.info(f"Parsed room {room_id}: {devs}")
            rooms_prefs[room_id] = devs

    rooms_list = [{"roomId": rid, "devices": devs} for rid, devs in rooms_prefs.items()]

    # Сохраняем позицию щитка (центроид комнаты со [щиток]) в DB
    if panel_room_id and project_id:
        _save_panel_from_room(project_id, panel_room_id)

    return {
        "version": "preferences-1.0",
        "sourceText": text,
        "global": {},
        "rooms": rooms_list,
        "_by_room_id": rooms_prefs,
    }

def _save_panel_from_room(project_id: str, room_id: str) -> None:
    """
    Берёт центроид комнаты room_id и сохраняет его как позицию щитка в DB["panel"].
    Вызывается автоматически если пользователь указал [щиток] в preferencesText.
    """
    try:
        from app.geometry import DB
        rooms = DB.get("rooms", {}).get(project_id, [])
        for room in rooms:
            if not isinstance(room, dict):
                continue
            rid = room.get("id") or room.get("roomId") or ""
            if rid != room_id:
                continue
            # Берём готовый центроид
            cp = room.get("centroidPx")
            if cp and len(cp) == 2:
                x, y = float(cp[0]), float(cp[1])
            else:
                # Считаем из полигона
                import numpy as np
                pts = room.get("polygonPx") or room.get("polygon") or []
                if len(pts) < 3:
                    return
                arr = np.array(pts, dtype=float)
                x, y = float(arr[:, 0].mean()), float(arr[:, 1].mean())

            DB.setdefault("panel", {})[project_id] = {
                "x": x, "y": y,
                "room_id": room_id,
                "source": "preferences_text",
            }
            logger.info(
                "Panel position set from room %s centroid: (%.0f, %.0f) for project %s",
                room_id, x, y, project_id
            )
            return

        logger.warning("_save_panel_from_room: room_id=%s not found for project %s", room_id, project_id)
    except Exception as e:
        logger.warning("_save_panel_from_room error: %s", e)