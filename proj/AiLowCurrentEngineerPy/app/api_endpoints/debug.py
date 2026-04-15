# app/api_endpoints/debug.py
"""Debug endpoints для просмотра состояния DB."""

from fastapi import APIRouter
from app.db import list_projects

router = APIRouter(tags=["debug"])

# Глобальное хранилище (импортируется из main.py)
DB = None

def set_db(db_instance):
    """Инициализация DB из main.py"""
    global DB
    DB = db_instance

@router.get("/projects")
async def list_all_projects():
    """Список всех проектов из SQLite."""
    try:
        return {"projects": list_projects()}
    except Exception:
        return {"projects": list(DB.get("rooms", {}).keys()) if DB else []}

@router.get("/debug/db/{project_id}")
async def debug_db(project_id: str):
    """Показывает что лежит в DB для project_id."""
    if not DB:
        return {"error": "DB not initialized"}
    return {
        "has_rooms": bool(DB.get("rooms", {}).get(project_id)),
        "n_rooms": len(DB.get("rooms", {}).get(project_id) or []),
        "has_plan_graph": bool(DB.get("plan_graph", {}).get(project_id)),
        "has_preferences": bool(DB.get("preferences", {}).get(project_id)),
        "rooms_sample": (DB.get("rooms", {}).get(project_id) or [])[:2],
        "db_keys": list(DB.keys()),
    }