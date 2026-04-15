# app/api_endpoints/design.py
"""POST /design и /design_nn3 - размещение устройств через NN-3."""

import re
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional

from app.placement import _apply_hard_rules
from app.services.preferences_service import parse_numbered_preferences

logger = logging.getLogger("planner")
router = APIRouter(tags=["design"])

# Глобальное хранилище и NN-2/NN-3
DB = None
nn2_parse_text = None
nn3_run_placement = None
_NN2_AVAILABLE = False
_NN3_AVAILABLE = False


def set_dependencies(db, nn2_fn, nn3_fn, nn2_ok, nn3_ok):
    """Инициализация из main.py."""
    global DB, nn2_parse_text, nn3_run_placement, _NN2_AVAILABLE, _NN3_AVAILABLE
    DB = db
    nn2_parse_text = nn2_fn
    nn3_run_placement = nn3_fn
    _NN2_AVAILABLE = nn2_ok
    _NN3_AVAILABLE = nn3_ok


class DesignRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    preferences_text: Optional[str] = Field(default=None,
                                            validation_alias=AliasChoices("preferencesText", "preferences_text"))


def _parse_preferences(text: str, project_id: str) -> dict:
    """Парсинг через NN-2 (fallback если нет номеров комнат)."""
    if not _NN2_AVAILABLE or not nn2_parse_text:
        return {"version": "preferences-1.0", "sourceText": text, "rooms": [], "_by_room_id": {}}
    try:
        result = nn2_parse_text(text, project_id=project_id)
        logger.info("NN-2: parsed %d device mentions", len(result.get("rooms", [])))
        return result
    except Exception as e:
        logger.error("NN-2 failed: %s", e)
        return {"version": "preferences-1.0", "sourceText": text, "rooms": [], "_by_room_id": {}}


@router.post("/design")
async def design(req: DesignRequest):
    """
    Основной эндпоинт размещения устройств.
    NN-2 → NN-3 → постпроцессинг (жёсткие правила ГОСТ/ПУЭ).
    """
    project_id = req.project_id

    # Получаем комнаты из NN-1
    rooms = DB.get("rooms", {}).get(project_id, [])
    if not rooms:
        return {"error": "No rooms found. Run /ingest first.", "project_id": project_id}

    # Строим PlanGraph
    plan_graph = {
        "projectId": project_id,
        "rooms": rooms,
        "openings": [],
        "topology": {"roomAdjacency": []},
    }

    # NN-2: парсим пожелания
    prefs_graph = DB.get("preferences", {}).get(project_id)
    if req.preferences_text:
        room_map = DB.get("room_map", {}).get(project_id, {})

        # Если есть номера комнат в тексте и room_map — используем прямой парсер
        has_numbers = bool(room_map and re.search(r"\d+\s*[:–-]", req.preferences_text))
        if has_numbers:
            prefs_graph = parse_numbered_preferences(req.preferences_text, room_map)
            logger.info("NN-2: numbered preferences parsed, %d rooms", len(prefs_graph.get("rooms", [])))
        else:
            prefs_graph = _parse_preferences(req.preferences_text, project_id=project_id)

        DB.setdefault("preferences", {})[project_id] = prefs_graph

    # NN-3: размещение устройств
    if not _NN3_AVAILABLE or not nn3_run_placement:
        return {"error": "NN-3 unavailable", "project_id": project_id}

    design_graph = nn3_run_placement(
        plan_graph=plan_graph if isinstance(plan_graph, dict) else plan_graph.dict(),
        prefs_graph=prefs_graph,
        project_id=project_id,
    )

    # Удаляем устройства из комнат с маркером "_skip" (пользователь написал "ничего")
    by_room_id = (prefs_graph or {}).get("_by_room_id", {})
    skip_rooms = [rid for rid, devs in by_room_id.items() if isinstance(devs, dict) and devs.get("_skip")]

    if skip_rooms:
        logger.info("Skipping devices for rooms: %s", skip_rooms)
        devices_before = design_graph.get("devices", [])
        devices_after = [
            d for d in devices_before
            if (d.get("roomRef") or d.get("room_id", "")) not in skip_rooms
        ]
        design_graph["devices"] = devices_after
        design_graph["totalDevices"] = len(devices_after)
        logger.info("Removed %d devices from skip rooms", len(devices_before) - len(devices_after))

    # Применяем жёсткие правила + пожелания по номерам комнат
    _rooms_for_rules = DB.get("rooms", {}).get(project_id, [])
    design_graph = _apply_hard_rules(design_graph, forced_devices=by_room_id, rooms=_rooms_for_rules,
                                     skip_rooms=skip_rooms)

    # Сохраняем в DB
    DB.setdefault("design", {})[project_id] = design_graph
    logger.info("NN-3: placed %d devices for project %s", design_graph.get("totalDevices", 0), project_id)

    return design_graph


@router.post("/design_nn3")
async def design_nn3(req: DesignRequest):
    """
    Чистый NN-3 БЕЗ постпроцессинга.
    Для тестирования и сравнения с /design.
    """
    project_id = req.project_id

    rooms = DB.get("rooms", {}).get(project_id, [])
    if not rooms:
        return {"error": "No rooms found. Run /ingest first.", "project_id": project_id}

    plan_graph = {
        "projectId": project_id,
        "rooms": rooms,
        "openings": [],
        "topology": {"roomAdjacency": []},
    }

    prefs_graph = DB.get("preferences", {}).get(project_id)
    if req.preferences_text:
        room_map = DB.get("room_map", {}).get(project_id, {})
        has_numbers = bool(room_map and re.search(r"\d+\s*[:–-]", req.preferences_text))
        if has_numbers:
            prefs_graph = parse_numbered_preferences(req.preferences_text, room_map)
        else:
            prefs_graph = _parse_preferences(req.preferences_text, project_id=project_id)
        DB.setdefault("preferences", {})[project_id] = prefs_graph

    if not _NN3_AVAILABLE or not nn3_run_placement:
        return {"error": "NN-3 unavailable", "project_id": project_id}

    design_graph = nn3_run_placement(
        plan_graph=plan_graph if isinstance(plan_graph, dict) else plan_graph.dict(),
        prefs_graph=prefs_graph,
        project_id=project_id,
    )

    # Минимальная валидация (без постпроцессинга)
    # Просто убираем дубли и санузлы с >1 SVT
    from app.svt_validator import apply_svt_validation_to_design_graph
    design_graph = apply_svt_validation_to_design_graph(design_graph, rooms or [])

    DB.setdefault("design", {})[project_id] = design_graph
    logger.info("NN-3 (pure): placed %d devices for project %s", design_graph.get("totalDevices", 0), project_id)

    return design_graph