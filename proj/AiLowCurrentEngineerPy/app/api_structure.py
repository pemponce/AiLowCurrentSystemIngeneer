from __future__ import annotations

import os
import os.path as osp
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.geometry import DB
from app.geometry_dxf import ingest_dxf
from app.geometry_png import ingest_png
from app.minio_client import RAW_BUCKET, download_file
from app.structure_detect import detect_structure

router = APIRouter(tags=["structure"])

LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_STRUCT", "/tmp/structure-downloads")
os.makedirs(LOCAL_DL_DIR, exist_ok=True)


class APIDetectStructureRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key"),
    )
    debug: bool = Field(default=True)


def _strip_bucket_prefix(key: str, bucket: str) -> str:
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1 :]
    return key


def _download_from_raw(src_key: str) -> str:
    clean = _strip_bucket_prefix(src_key, RAW_BUCKET)
    local = osp.join(LOCAL_DL_DIR, osp.basename(clean))
    if not osp.exists(local):
        try:
            download_file(RAW_BUCKET, clean, local)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{clean}]: {e}")
    return local


def _ensure_rooms(project_id: str, src_key: Optional[str]) -> Optional[str]:
    DB.setdefault("rooms", {}).setdefault(project_id, [])
    DB.setdefault("devices", {}).setdefault(project_id, [])
    DB.setdefault("routes", {}).setdefault(project_id, [])
    DB.setdefault("candidates", {}).setdefault(project_id, [])

    if DB["rooms"].get(project_id):
        meta = DB.get("source_meta", {}).get(project_id, {}) if isinstance(DB.get("source_meta", {}), dict) else {}
        for k in ("localPath", "local_path", "path"):
            lp = meta.get(k)
            if lp and osp.exists(lp):
                return lp
        return None

    if not src_key:
        return None

    local = _download_from_raw(src_key)
    low = local.lower()
    if low.endswith((".dxf", ".dwg")):
        ingest_dxf(project_id, local)
        return None
    if low.endswith((".png", ".jpg", ".jpeg")):
        ingest_png(project_id, local)
        return local

    raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")


@router.post("/detect-structure")
async def detect_structure_endpoint(req: APIDetectStructureRequest):
    local_image = _ensure_rooms(req.project_id, req.src_s3_key)

    if local_image is None:
        if not req.src_s3_key:
            raise HTTPException(
                status_code=400,
                detail="Нет локального пути к изображению. Передай srcKey (RAW png/jpg) в /detect-structure.",
            )
        local_image = _download_from_raw(req.src_s3_key)

    try:
        return detect_structure(req.project_id, local_image, src_key=req.src_s3_key, debug=req.debug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structure detection failed: {e}")
