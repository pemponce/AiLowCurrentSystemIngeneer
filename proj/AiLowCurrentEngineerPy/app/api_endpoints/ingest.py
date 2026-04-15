# app/api_endpoints/ingest.py
"""POST /ingest - сегментация плана через NN-1."""

import os
import os.path as osp
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional

from app.geometry_png import ingest_png
from app.geometry_dxf import ingest_dxf
from app.minio_client import download_file, upload_file, RAW_BUCKET, EXPORT_BUCKET
from app.services.nn1_service import nn1_get_rooms, make_numbered_plan

logger = logging.getLogger("planner")
router = APIRouter(tags=["ingest"])

LOCAL_RAW_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)

# Глобальное хранилище (инициализируется из main.py)
DB = None
_SQLITE_OK = False
save_project = None
save_rooms = None
save_room_map = None


def set_db(db_instance, sqlite_ok, save_proj_fn, save_rooms_fn, save_map_fn):
    """Инициализация DB и SQLite функций из main.py."""
    global DB, _SQLITE_OK, save_project, save_rooms, save_room_map
    DB = db_instance
    _SQLITE_OK = sqlite_ok
    save_project = save_proj_fn
    save_rooms = save_rooms_fn
    save_room_map = save_map_fn


class APIIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: str = Field(validation_alias=AliasChoices("srcKey", "src_s3_key"))


def _strip_bucket_prefix(key: str, bucket: str) -> str:
    """Убирает префикс bucket/ из S3 ключа."""
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1:]
    return key


@router.post("/ingest")
async def ingest(req: APIIngestRequest):
    """
    Запускает NN-1 сегментацию плана и возвращает список комнат.
    Генерирует numbered plan с номерами комнат для пользователя.
    """
    DB.setdefault("rooms", {}).setdefault(req.project_id, [])
    DB.setdefault("devices", {}).setdefault(req.project_id, [])
    DB.setdefault("routes", {}).setdefault(req.project_id, [])
    DB.setdefault("candidates", {}).setdefault(req.project_id, [])
    DB.setdefault("exports", {}).setdefault(req.project_id, {})
    DB.setdefault("source", {}).setdefault(req.project_id, {})

    key = _strip_bucket_prefix(req.src_s3_key, RAW_BUCKET)
    basename = osp.basename(key)
    local_path = osp.join(LOCAL_RAW_DIR, basename)

    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{key}]: {e}")

    DB["source"][req.project_id] = {"src_key": req.src_s3_key, "local_path": local_path}

    low = local_path.lower()
    if low.endswith((".dxf", ".dwg")):
        stats = ingest_dxf(req.project_id, local_path)
    elif low.endswith((".png", ".jpg", ".jpeg")):
        stats = ingest_png(req.project_id, local_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")

    # NN-1: запускаем сегментацию и нумеруем комнаты
    numbered_url = None
    room_map = {}
    nn1_rooms = []

    if low.endswith((".png", ".jpg", ".jpeg")):
        nn1_rooms = nn1_get_rooms(local_path, req.project_id)
        if nn1_rooms:
            DB.setdefault("rooms", {})[req.project_id] = nn1_rooms
            _, numbered_url, room_map = make_numbered_plan(local_path, nn1_rooms, req.project_id)
            DB.setdefault("room_map", {})[req.project_id] = room_map
            logger.info("Ingest room_map: %s", {k: v for k, v in room_map.items()})

            # Сохраняем в SQLite
            if _SQLITE_OK and save_project and save_rooms and save_room_map:
                try:
                    save_project(req.project_id, req.src_s3_key or "", local_path or "")
                    save_rooms(req.project_id, nn1_rooms)
                    save_room_map(req.project_id, room_map)
                except Exception as _se:
                    logger.debug("SQLite save failed: %s", _se)
            logger.info("Ingest: NN-1 нашла %d комнат, numbered plan создан", len(nn1_rooms))

    # Генерируем preview зон освещения
    try:
        from app.export_overlay_png import export_zones_preview
        zones_path = f"/tmp/exports/{req.project_id}_zones.png"
        export_zones_preview(local_path, nn1_rooms, zones_path)
        _zones_uri = upload_file(EXPORT_BUCKET, zones_path, f"previews/{req.project_id}_zones.png")
        logger.info("Zones preview: %s", _zones_uri)
    except Exception as _ze:
        logger.debug("Zones preview failed: %s", _ze)

    # Строим краткое описание комнат для пользователя
    rooms_info = []
    for i, r in enumerate(nn1_rooms):
        rooms_info.append({
            "num": i + 1,
            "room_id": r.get("id"),
            "room_type": r.get("roomType", "?"),
            "area_m2": r.get("areaM2", 0),
        })
    logger.info(f"INGEST srcKey = {key}")

    return {
        "project_id": req.project_id,
        "rooms_found": len(nn1_rooms),
        "rooms": rooms_info,
        "numbered_plan": numbered_url,
        "hint": "Посмотри на numbered_plan и в /design укажи: \"1: свет датчик дыма; 2: телевизор свет; 3: свет\"",
        **stats,
    }