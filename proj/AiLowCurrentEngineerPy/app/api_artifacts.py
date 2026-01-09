from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.minio_client import presigned_get_url
from app.artifacts_index import build_artifacts_index

router = APIRouter(tags=["Artifacts"])


class ArtifactUrlRequest(BaseModel):
    # Mode A (legacy): bucket+key -> presigned URL
    bucket: Optional[str] = Field(None, description="S3/MinIO bucket name")
    key: Optional[str] = Field(None, description="Object key inside the bucket")

    # Mode B: projectId -> unified artifacts index (preview/overlay/files/result_json)
    projectId: Optional[str] = Field(None, description="Project id to resolve artifacts")
    expiresSeconds: int = Field(3600, ge=60, le=7 * 24 * 3600)


@router.post("/artifact-url")
def artifact_url(req: ArtifactUrlRequest) -> Any:
    if req.projectId:
        return {
            "projectId": req.projectId,
            "expiresSeconds": int(req.expiresSeconds),
            "artifacts": build_artifacts_index(req.projectId, expires_seconds=int(req.expiresSeconds)),
        }

    if not req.bucket or not req.key:
        return {
            "detail": "Provide either projectId, or bucket+key.",
        }

    url = presigned_get_url(req.bucket, req.key, int(req.expiresSeconds))
    return {
        "bucket": req.bucket,
        "key": req.key,
        "url": url,
        "expiresSeconds": int(req.expiresSeconds),
    }
