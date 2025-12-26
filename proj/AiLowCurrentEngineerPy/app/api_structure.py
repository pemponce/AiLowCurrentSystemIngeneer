from __future__ import annotations

import os
import os.path as osp
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

from app.geometry import DB
from app.geometry_png import ingest_png
from app.geometry_dxf import ingest_dxf
from app.minio_client import RAW_BUCKET, download_file
from app.structure_detect import detect_structure


router = APIRouter(tags=["structure"])

LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_STRUCT", "/tmp/structure-downloads")
os.makedirs(LOCAL_DL_DIR, exist_ok=True)


class APIDetectStructureRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key"))
    debug: bool = True


def _strip_bucket_prefix(key: str, bucket: str) -> str:
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1:]
    return key


def _download_from_raw(src_key: str) -> str:
    clean_key = _strip_bucket_prefix(src_key, RAW_BUCKET)
    basename = osp.basename(clean_key)
    local_path = osp.join(LOCAL_DL_DIR, basename)
    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, clean_key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{clean_key}]: {e}")
    return local_path


def _ensure_geometry(project_id: str, src_key: Optional[str]) -> Optional[str]:
    """
    Если геометрии ещё нет — попробуем скачать исходник и вызвать ingest.
    Возвращаем локальный путь к картинке (если есть).
    """
    rooms = DB.get("rooms", {}).get(project_id)
    if rooms:
        # если уже есть rooms, всё равно нужен image_path для детектора
        # попробуем найти в source_meta (если у тебя он есть), иначе потребуется src_key
        meta = DB.get("source_meta", {}).get(project_id, {}) if isinstance(DB.get("source_meta", {}), dict) else {}
        for k in ("localPath", "local_path", "path"):
            lp = meta.get(k)
            if lp and osp.exists(lp):
                return lp
        return None

    if not src_key:
        return None

    local_path = _download_from_raw(src_key)
    low = local_path.lower()

    # поддержим dxf/dwg (но детектор структуры работает только на image — это нормально)
    if low.endswith((".dxf", ".dwg")):
        ingest_dxf(project_id, local_path)
        return None

    if low.endswith((".png", ".jpg", ".jpeg")):
        ingest_png(project_id, local_path)
        return local_path

    raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")


@router.post("/detect-structure")
async def detect_structure_endpoint(req: APIDetectStructureRequest):
    DB.setdefault("rooms", {}).setdefault(req.project_id, [])
    DB.setdefault("devices", {}).setdefault(req.project_id, [])
    DB.setdefault("routes", {}).setdefault(req.project_id, [])
    DB.setdefault("candidates", {}).setdefault(req.project_id, [])

    # 1) гарантируем rooms (если надо — ingest)
    image_path = _ensure_geometry(req.project_id, req.src_s3_key)

    # 2) если image_path не нашли (а rooms есть), то без srcKey мы не сможем построить маску
    if image_path is None:
        if req.src_s3_key:
            image_path = _download_from_raw(req.src_s3_key)
        else:
            raise HTTPException(
                status_code=400,
                detail="Нет локального пути к изображению. Передай srcKey (RAW png/jpg) в /detect-structure."
            )

    # 3) детект структуры
    try:
        return detect_structure(
            project_id=req.project_id,
            image_path=image_path,
            src_key=req.src_s3_key,
            debug=req.debug
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure detection failed: {e}")
