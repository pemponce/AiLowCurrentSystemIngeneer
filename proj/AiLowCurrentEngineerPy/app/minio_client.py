"""app/minio_client.py

Единый MinIO/S3-клиент для Python-сервиса.

Важное:
1) Java-сервис читает переменные S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET_RAW/S3_BUCKET_EXPORTS.
2) В вашем Python проекте ранее использовался S3_ENDPOINTPY и другие дефолты.

Этот модуль делает совместимость:
- принимает S3_ENDPOINTPY как override,
- иначе берёт S3_ENDPOINT,
- аккуратно обрабатывает http/https в endpoint,
- дефолтные бакеты совпадают с main.py (raw-plans / exports).
"""

from __future__ import annotations

import os
from typing import Optional

from minio import Minio


def _pick_env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def _normalize_endpoint(endpoint: str) -> tuple[str, bool]:
    """Возвращает (host:port, secure)."""
    ep = endpoint.strip()
    if ep.startswith("http://"):
        return ep[len("http://") :], False
    if ep.startswith("https://"):
        return ep[len("https://") :], True
    # без схемы
    secure = os.getenv("S3_MINIO_SECURE", "false").lower() == "true"
    return ep, secure


def _mk_client() -> Minio:
    endpoint_raw = _pick_env("S3_ENDPOINTPY", "S3_ENDPOINT", default="minio:9000") or "minio:9000"
    endpoint, secure_from_scheme = _normalize_endpoint(endpoint_raw)

    access = _pick_env("S3_ACCESS_KEY", default="minioadmin") or "minioadmin"
    secret = _pick_env("S3_SECRET_KEY", default="minioadmin") or "minioadmin"

    # secure_from_scheme имеет приоритет (если указали http/https)
    secure = secure_from_scheme
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


CLIENT = _mk_client()

# Дефолты должны совпадать с app/main.py
RAW_BUCKET = _pick_env("S3_BUCKET_RAW", default="raw-plans") or "raw-plans"
EXPORT_BUCKET = _pick_env("S3_BUCKET_EXPORTS", default="exports") or "exports"


def ensure_bucket(bucket: str) -> None:
    if not CLIENT.bucket_exists(bucket):
        CLIENT.make_bucket(bucket)


def upload_file(bucket: str, local_path: str, key: str) -> str:
    ensure_bucket(bucket)
    CLIENT.fput_object(bucket, key, local_path)
    return f"s3://{bucket}/{key}"


def download_file(bucket: str, key: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    CLIENT.fget_object(bucket, key, local_path)
    return local_path
