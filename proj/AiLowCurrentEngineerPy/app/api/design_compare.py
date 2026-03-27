"""
A/B сравнение /design vs /design_nn3
"""
from fastapi import APIRouter, HTTPException
import time
from typing import Dict, List

router = APIRouter()


def count_by_reason(devices: List[Dict]) -> Dict[str, int]:
    """Группировка устройств по источнику размещения"""
    groups = {
        "NN-3": 0,
        "zone-grid": 0,
        "normative": 0,
        "user-forced": 0,
        "rule-based": 0,
        "other": 0
    }

    for d in devices:
        reason = d.get("reason", "")

        if "NN-3" in reason:
            groups["NN-3"] += 1
        elif "zone:" in reason:
            groups["zone-grid"] += 1
        elif "norm:" in reason:
            groups["normative"] += 1
        elif "user request" in reason:
            groups["user-forced"] += 1
        elif "rule:" in reason:
            groups["rule-based"] += 1
        else:
            groups["other"] += 1

    return groups


def calculate_nn3_usage(devices: List[Dict]) -> float:
    """% устройств от NN-3 (не перезаписанных постпроцессингом)"""
    total = len(devices)
    if total == 0:
        return 0.0

    nn3_count = sum(1 for d in devices if "NN-3" in d.get("reason", ""))
    return round((nn3_count / total) * 100, 2)


@router.post("/design_compare")
async def design_compare(req: dict):
    """
    Сравнение /design (с постпроцессингом) vs /design_nn3 (без постпроцессинга)
    """
    project_id = req.get("projectId")
    prefs_text = req.get("preferencesText", "")

    if not project_id:
        raise HTTPException(400, "projectId required")

    # ── Загрузка плана из DB ──────────────────────────────────────────────────
    from app.main import DB
    rooms = DB.get("rooms", {}).get(project_id)
    if not rooms:
        raise HTTPException(404, f"Plan not found: {project_id}")

    # ── Метод A: /design (ТЕКУЩИЙ - с постпроцессингом) ───────────────────────
    start_a = time.time()

    try:
        from app.db import get_design
        design_a = get_design(project_id)

        if not design_a:
            raise HTTPException(404, f"Design not found for {project_id}. Run /design first.")
    except Exception as e:
        raise HTTPException(500, f"Method A failed: {str(e)}")

    time_a_ms = int((time.time() - start_a) * 1000)

    # ── Метод B: /design_nn3 (БЕЗ постпроцессинга) ────────────────────────────
    # Для этого нужен эндпоинт /design_nn3 - пока возвращаем None
    start_b = time.time()

    design_b = None
    time_b_ms = 0

    # TODO: После создания /design_nn3 эндпоинта - раскомментировать:
    try:
        design_b = get_design(f"{project_id}_nn3")
        if not design_b:
            raise HTTPException(404, "Run /design_nn3 first")
    except Exception as e:
        raise HTTPException(500, f"Method B failed: {str(e)}")

    time_b_ms = int((time.time() - start_b) * 1000)

    # ── Анализ результатов ────────────────────────────────────────────────────
    devices_a = design_a.get("devices", [])

    reasons_a = count_by_reason(devices_a)
    nn3_usage_a = calculate_nn3_usage(devices_a)

    devices_b = design_b.get("devices", [])

    reasons_b = count_by_reason(devices_b)
    nn3_usage_b = calculate_nn3_usage(devices_b)

    # ── Результат (пока только Method A) ─────────────────────────────────────
    return {
        "project_id": project_id,
        "method_a": {
            "name": "/design (with postprocessing)",
            "total_devices": len(devices_a),
            "execution_time_ms": time_a_ms,
            "reason_breakdown": reasons_a,
            "nn3_usage_pct": nn3_usage_a,
            "postprocessing_override_pct": round(100 - nn3_usage_a, 2)
        },
        "method_b": {
            "name": "/design_nn3 (pure NN-3)",
            "total_devices": len(devices_b),
            "execution_time_ms": time_b_ms,
            "reason_breakdown": reasons_b,
            "nn3_usage_pct": nn3_usage_b
        },
        "recommendation": (
            f"⚠️ Postprocessing overrides {round(100 - nn3_usage_a, 1)}% of NN-3 predictions. "
            f"NN-3 is only used for {nn3_usage_a}% of devices."
        )
    }
