from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.minio_client import presigned_get_url

router = APIRouter(tags=["Artifacts"])


class ArtifactUrlRequest(BaseModel):
    bucket: str = Field(..., description="S3/MinIO bucket name")
    key: str = Field(..., description="Object key inside the bucket")
    expiresSeconds: int = Field(3600, ge=60, le=7 * 24 * 3600)


class ArtifactUrlResponse(BaseModel):
    bucket: str
    key: str
    url: str
    expiresSeconds: int


@router.post("/artifact-url", response_model=ArtifactUrlResponse)
def artifact_url(req: ArtifactUrlRequest) -> ArtifactUrlResponse:
    url = presigned_get_url(req.bucket, req.key, req.expiresSeconds)
    return ArtifactUrlResponse(bucket=req.bucket, key=req.key, url=url, expiresSeconds=req.expiresSeconds)
