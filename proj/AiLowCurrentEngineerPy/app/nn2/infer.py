"""
app/nn2/infer.py — инференс NN-2 v2 (NER)

Использование:
  from app.nn2.infer import parse_text
  result = parse_text("хочу в гостиной 2 розетки для телевизора и датчик дыма")
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import torch

from app.nn2.model import NERModel, TagVocab, WordVocab, MAX_SEQ
from app.nn2.dataset_gen import ROOM_TOKENS as ROOM_TOKENS_REF


# ─── постпроцессинг тегов → PreferencesGraph ─────────────────────────────────

def _parse_tags(tokens: List[str], tags: List[str]) -> Dict[str, Any]:
    """
    Собирает пары (комната → устройства) из BIO тегов.

    Поддерживаемые паттерны:
      ROOM DEV           → устройство в комнате
      DEV ROOM           → устройство в комнате (устройство перед комнатой)
      DEV ROOM1 ROOM2    → устройство в обеих комнатах
      ROOM1 DEV ROOM2    → каждая комната получает своё устройство
      DEV везде NEG ROOM → устройство везде кроме указанной комнаты
      COUNT DEV ROOM     → количество устройств в комнате
    """
    rooms: Dict[str, Dict] = {}
    global_devs: Dict[str, bool] = {}
    negated_rooms: List[str] = []

    # ── 1. Собираем сущности ──────────────────────────────────────
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.endswith("-B"):
            prefix = tag[:-2]
            span   = [tokens[i]]
            j = i + 1
            while j < len(tags) and tags[j] == prefix + "-I":
                span.append(tokens[j])
                j += 1
            entities.append({"type": prefix, "tokens": span, "pos": i})
            i = j
        elif tag.startswith("COUNT-"):
            n = int(tag.split("-")[1])
            entities.append({"type": "COUNT", "value": n, "pos": i})
            i += 1
        elif tag == "NEG":
            entities.append({"type": "NEG", "pos": i})
            i += 1
        else:
            i += 1

    # ── 2. Определяем режим "везде" ───────────────────────────────
    ALL_ROOMS = list(ROOM_TOKENS_REF)
    has_all = any(t == "везде" for t in tokens)

    # ── 3. Проходим по сущностям ──────────────────────────────────
    # pending_device: устройство которое ещё не привязано к комнате
    pending_device: Optional[str] = None
    pending_count:  Optional[int] = None
    current_room:   Optional[str] = None
    neg_active = False

    for idx, ent in enumerate(entities):
        etype = ent["type"]

        # --- Отрицание ---
        if etype == "NEG":
            neg_active = True

        # --- Число ---
        elif etype == "COUNT":
            pending_count = ent["value"]

        # --- Комната ---
        elif etype.startswith("ROOM-"):
            room_type = etype[5:]
            if neg_active:
                negated_rooms.append(room_type)
                neg_active = False
            else:
                current_room = room_type
                if current_room not in rooms:
                    rooms[current_room] = {}
                # Если висит pending_device — привязываем к этой комнате тоже
                if pending_device:
                    _assign_device(rooms[current_room], pending_device, pending_count)
                    # НЕ сбрасываем pending_device — он может прилипнуть к следующим комнатам

        # --- Устройство ---
        elif etype.startswith("DEV-"):
            device = etype[4:]
            # Сбрасываем старый pending если новое устройство
            pending_device = device

            # Если есть текущая комната — сразу привязываем
            if current_room:
                _assign_device(rooms[current_room], device, pending_count)
                pending_count = None
            else:
                # Устройство перед комнатой — ищем все следующие комнаты
                # до следующего устройства
                for future in entities[idx + 1:]:
                    if future["type"].startswith("DEV-") or future["type"].startswith("GLOBAL-"):
                        break
                    if future["type"].startswith("ROOM-"):
                        rtype = future["type"][5:]
                        if rtype not in rooms:
                            rooms[rtype] = {}
                        _assign_device(rooms[rtype], device, pending_count)
                pending_count = None
                pending_device = None

        # --- Глобальное устройство ---
        elif etype.startswith("GLOBAL-"):
            gdev = etype[7:]
            global_devs[gdev] = True
            pending_device = None

    # ── 4. Обрабатываем "везде кроме X" ──────────────────────────
    if has_all and negated_rooms:
        # Находим устройство которое должно быть везде
        for ent in entities:
            if ent["type"].startswith("DEV-"):
                device = ent["type"][4:]
                for rtype in ALL_ROOMS:
                    if rtype not in negated_rooms:
                        if rtype not in rooms:
                            rooms[rtype] = {}
                        _assign_device(rooms[rtype], device, None)
                break

    return {"rooms": rooms, "global": global_devs, "negated": negated_rooms}


def _assign_device(room_data: Dict, device: str, count: Optional[int]) -> None:
    """Записывает устройство в данные комнаты."""
    BOOL_DEVICES = {"smoke_detector", "co2_detector", "night_light", "motion_sensor"}
    COUNT_DEVICES = {"tv_socket": "tv_sockets", "internet_socket": "internet_sockets",
                     "ceiling_light": "ceiling_lights"}

    if device in BOOL_DEVICES:
        room_data[device] = True
    elif device in COUNT_DEVICES:
        key = COUNT_DEVICES[device]
        room_data[key] = count if count else 1
    else:
        room_data[device] = count if count else True


def _to_preferences_graph(parsed: Dict, source_text: str, project_id: str) -> Dict:
    """Конвертирует в формат PreferencesGraph."""
    rooms_list = []
    for room_type, slots in parsed["rooms"].items():
        entry = {"roomType": room_type}
        entry.update(slots)
        rooms_list.append(entry)

    return {
        "version":    "preferences-1.0",
        "projectId":  project_id,
        "sourceText": source_text,
        "global":     parsed["global"],
        "rooms":      rooms_list,
        "negated_rooms": parsed.get("negated", []),
    }


# ─── инференс ────────────────────────────────────────────────────────────────

class PreferencesInfer:
    def __init__(self, model: NERModel,
                 word_vocab: WordVocab,
                 tag_vocab:  TagVocab,
                 device: str) -> None:
        self.model      = model
        self.word_vocab = word_vocab
        self.tag_vocab  = tag_vocab
        self.device     = device
        self.model.eval()

    @classmethod
    def load(cls, model_dir: str) -> "PreferencesInfer":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        word_vocab = WordVocab.load(os.path.join(model_dir, "word_vocab.json"))
        tag_vocab  = TagVocab.load(os.path.join(model_dir, "tag_vocab.json"))
        ckpt       = torch.load(os.path.join(model_dir, "nn2_best.pt"), map_location=device)
        model = NERModel(vocab_size=ckpt["vocab_size"],
                         num_tags=ckpt["num_tags"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"NN-2 загружена: F1={ckpt.get('f1', '?'):.3f}")
        return cls(model, word_vocab, tag_vocab, device)

    @torch.no_grad()
    def parse(self, text: str, project_id: str = "unknown") -> Dict[str, Any]:
        # токенизируем по пробелам (simple word tokenizer)
        tokens  = re.findall(r"[а-яёА-ЯЁa-zA-Z0-9][а-яёА-ЯЁa-zA-Z0-9\-\.]*|[,!?;]", text)
        if not tokens:
            return _to_preferences_graph({"rooms": {}, "global": {}, "negated": []}, text, project_id)

        length = min(len(tokens), MAX_SEQ)
        x      = torch.tensor([self.word_vocab.encode(tokens)], dtype=torch.long).to(self.device)
        lens   = torch.tensor([length]).to(self.device)

        pred_ids = self.model.predict(x, lens)[0]
        pred_tags = self.tag_vocab.decode(pred_ids)

        parsed = _parse_tags(tokens[:length], pred_tags[:length])
        return _to_preferences_graph(parsed, text, project_id)


# ─── публичный интерфейс ─────────────────────────────────────────────────────

_infer_instance: Optional[PreferencesInfer] = None

def get_infer(model_dir: str = "models/nn2") -> Optional[PreferencesInfer]:
    global _infer_instance
    if _infer_instance is None:
        try:
            _infer_instance = PreferencesInfer.load(model_dir)
        except Exception as e:
            print(f"NN-2 не загружена ({e}), используем regex fallback")
    return _infer_instance

def parse_text(text: str, project_id: str = "unknown") -> Dict[str, Any]:
    infer = get_infer()
    if infer:
        return infer.parse(text, project_id)
    # fallback
    return {"version": "preferences-1.0", "projectId": project_id,
            "sourceText": text, "global": {}, "rooms": []}