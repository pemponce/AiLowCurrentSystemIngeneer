# app/api_endpoints/health.py
"""
Health check эндпоинты для мониторинга системы.
"""

import os
from fastapi import APIRouter

from app.minio_client import CLIENT, RAW_BUCKET, EXPORT_BUCKET


router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Базовая проверка здоровья сервиса."""
    return {"status": "ok"}


@router.get("/health/s3")
def health_s3():
    """
    Проверка подключения к MinIO (S3).
    Показывает конфигурацию endpoints и buckets.
    """
    try:
        ok = CLIENT.bucket_exists(RAW_BUCKET)
        internal = os.getenv("S3_ENDPOINTPY", "minio:9000")
        public = (
            os.getenv("S3_PUBLIC_ENDPOINT")
            or os.getenv("S3_PUBLIC_ENDPOINTPY")
            or internal
        )

        warning = None
        if (
            ("minio" in str(public).lower())
            and not os.getenv("S3_PUBLIC_ENDPOINT")
            and not os.getenv("S3_PUBLIC_ENDPOINTPY")
        ):
            warning = (
                "S3_PUBLIC_ENDPOINT is not set; presigned URLs may use internal host "
                "(e.g., minio:9000) and may not open in browser. "
                "Set S3_PUBLIC_ENDPOINT=localhost:9000 for local dev."
            )

        return {
            "ok": ok,
            "raw_bucket": RAW_BUCKET,
            "export_bucket": EXPORT_BUCKET,
            "endpoint_internal": internal,
            "endpoint_public": public,
            "warning": warning,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}