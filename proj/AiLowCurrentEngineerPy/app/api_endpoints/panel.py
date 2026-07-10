# app/api_endpoints/panel.py
"""
POST /set_panel   — установить позицию щитка (координаты клика по плану)
GET  /panel/{id} — получить текущую позицию щитка
GET  /panel_ui   — HTML интерфейс для визуального выбора щитка
"""
from __future__ import annotations

import logging
import os
import os.path as osp
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, presigned_get_url

logger = logging.getLogger("planner")
router = APIRouter(prefix="/panel", tags=["panel"])


# ── Модели запросов ───────────────────────────────────────────────────────────

class SetPanelRequest(BaseModel):
    projectId: str
    xPx: float
    yPx: float
    label: Optional[str] = "Щиток"   # произвольная метка для UI


# ── Эндпоинты ────────────────────────────────────────────────────────────────

@router.post("/set")
async def set_panel(req: SetPanelRequest):
    """
    Устанавливает позицию щитка для проекта.
    Координаты в пикселях относительно исходного PNG плана.

    После вызова /export автоматически использует эту точку
    как начало всех кабельных трасс.
    """
    DB.setdefault("panel", {})[req.projectId] = {
        "x":      req.xPx,
        "y":      req.yPx,
        "label":  req.label,
        "source": "manual",
    }
    logger.info(
        "Panel set manually: project=%s x=%.0f y=%.0f",
        req.projectId, req.xPx, req.yPx,
    )
    return {
        "project_id": req.projectId,
        "panel": {"x": req.xPx, "y": req.yPx, "label": req.label},
        "message": "Позиция щитка сохранена. Запустите /export для перестройки трасс.",
    }


@router.get("/info/{project_id}")
async def get_panel(project_id: str):
    """Возвращает текущую позицию щитка для проекта."""
    panel = DB.get("panel", {}).get(project_id)
    if not panel:
        return JSONResponse(
            status_code=404,
            content={"error": f"Щиток не задан для проекта {project_id}"},
        )
    return {"project_id": project_id, "panel": panel}


@router.get("/ui", response_class=HTMLResponse)
async def panel_ui(project_id: str = ""):
    """
    Возвращает HTML-страницу для визуального выбора позиции щитка.
    Открывать в браузере: http://localhost:8000/panel_ui?project_id=plan001
    """
    html_path = osp.join(osp.dirname(__file__), "..", "static", "panel_ui.html")
    if osp.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
        # Подставляем projectId если передан
        if project_id:
            html = html.replace('value=""', f'value="{project_id}"', 1)
        return HTMLResponse(content=html)
    return HTMLResponse(content="<h1>panel_ui.html not found</h1>", status_code=404)


@router.get("/numbered_url/{project_id}")
async def get_numbered_url(project_id: str):
    """
    Возвращает presigned URL numbered плана для отображения в UI.
    Numbered план создаётся при /ingest.
    """
    # Локальный файл (быстрее)
    local = f"/tmp/exports/{project_id}_numbered.png"
    if osp.exists(local):
        # Отдаём через /static_file эндпоинт
        return {
            "project_id": project_id,
            "url": f"/panel/static_file/{project_id}_numbered.png",
            "source": "local",
        }

    # MinIO presigned URL
    key = f"previews/{project_id}_numbered.png"
    try:
        url = presigned_get_url(EXPORT_BUCKET, key, expires_seconds=3600)
        return {"project_id": project_id, "url": url, "source": "minio"}
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": f"Numbered план не найден: {e}"},
        )


@router.get("/static_file/{filename}")
async def serve_static_file(filename: str):
    """Отдаёт файлы из /tmp/exports/ напрямую (для panel_ui)."""
    from fastapi.responses import FileResponse
    safe = filename.replace("..", "").replace("/", "").replace("\\", "")
    path = f"/tmp/exports/{safe}"
    if osp.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": f"File not found: {safe}"})