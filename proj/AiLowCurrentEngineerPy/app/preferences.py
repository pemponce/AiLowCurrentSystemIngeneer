import re
from typing import Dict, Optional
from .models import PreferenceParseRequest, PreferenceParseResponse

# словарь с “синонимами” комнат -> ключи модели освещенности/наших норм
ROOM_SYNONYMS = {
    r"\bгостин(ая|ой|ую|ке|иц)\b": "living",
    r"\bзал\b": "living",
    r"\bспальн(я|е|ю|и|ях)\b": "bedroom",
    r"\bкухн(я|е|ю|и)\b": "kitchen",
    r"\bдетск(ая|ой|ую)\b": "bedroom",  # временно маппим к bedroom
}

LUX_RE = re.compile(r"(\d{2,4})\s*(лк|люкс|\blux\b)", re.IGNORECASE)
INT_RE = re.compile(r"\b(\d{1,3})\b")

def parse_preferences(req: PreferenceParseRequest) -> PreferenceParseResponse:
    text = req.text.lower()
    per_room: Dict[str, float] = {}

    # 1) ищем конструкции вида: "для кухни 300 лк", "в спальне 150 люкс"
    for patt, norm_room in ROOM_SYNONYMS.items():
        for m_room in re.finditer(patt, text):
            # берем ближайшее число-люкс после упоминания комнаты
            m_lux = LUX_RE.search(text, m_room.end(), m_room.end() + 80)
            if m_lux:
                try:
                    per_room[norm_room] = float(m_lux.group(1))
                except ValueError:
                    pass

    # 2) общий target_lux (если встречается без упоминания комнаты)
    # возьмем первое упоминание "N лк" в тексте, если ещё нет per_room и общая цель не задана
    target_lux: Optional[float] = None
    if not per_room:
        m = LUX_RE.search(text)
        if m:
            try:
                target_lux = float(m.group(1))
            except ValueError:
                target_lux = None

    # 3) подсказка на общее кол-во светильников (на случай: "поставить 18 светильников")
    total_fixtures_hint: Optional[int] = None
    if "светильник" in text or "светильников" in text:
        # грубо: первое маленькое целое в тексте
        m = INT_RE.search(text)
        if m:
            try:
                val = int(m.group(1))
                if 1 <= val <= 500:
                    total_fixtures_hint = val
            except ValueError:
                pass

    # 4) эффективность, если явно встречается "... лм/вт"
    eff: Optional[float] = None
    m_eff = re.search(r"(\d{2,3})\s*(лм/вт|lm/w)\b", text, flags=re.IGNORECASE)
    if m_eff:
        try:
            eff = float(m_eff.group(1))
        except ValueError:
            pass

    return PreferenceParseResponse(
        per_room_target_lux=per_room or None,
        target_lux=target_lux,
        total_fixtures_hint=total_fixtures_hint,
        fixture_efficacy_lm_per_w=eff,
        notes=None if (per_room or target_lux) else "Не удалось извлечь явные нормы; использую значения по умолчанию."
    )
