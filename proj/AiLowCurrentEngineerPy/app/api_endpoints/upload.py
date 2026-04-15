# app/api_endpoints/upload.py
"""POST /upload - загрузка PNG плана в MinIO."""

import base64
import tempfile
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from app.minio_client import upload_file, RAW_BUCKET

logger = logging.getLogger("planner")
router = APIRouter(tags=["ingest"])


class UploadPlanRequest(BaseModel):
    projectId: str = "plan001"
    imageBase64: str
    srcKey: Optional[str] = None


@router.post("/upload")
async def upload_plan(req: UploadPlanRequest):
    """Загружает PNG план (base64) в MinIO и возвращает srcKey."""
    try:
        img_bytes = base64.b64decode(req.imageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    key = f"raw_plans/{req.projectId}/{uuid.uuid4()}.png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        upload_file(RAW_BUCKET, tmp_path, key)
        logger.info("Uploaded plan to MinIO: %s/%s", RAW_BUCKET, key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {"uploaded": True, "srcKey": key, "projectId": req.projectId}