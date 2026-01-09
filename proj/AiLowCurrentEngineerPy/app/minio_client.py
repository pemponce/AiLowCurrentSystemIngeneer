import os
import datetime
from minio import Minio


def _mk_client() -> Minio:
    endpoint = os.getenv("S3_ENDPOINTPY", "minio:9000")  # без схемы
    access = os.getenv("S3_ACCESS_KEY", "minioadmin")
    secret = os.getenv("S3_SECRET_KEY", "minioadmin")
    secure = os.getenv("S3_MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


def _mk_public_client() -> Minio:
    """Клиент для генерации presigned URL, чтобы ссылки открывались в браузере.

    IMPORTANT: presigned URL завязан на host/endpoint. Поэтому для "публичных" ссылок
    используем отдельный endpoint (например, localhost:9000), если он задан.
    """
    endpoint = (
        os.getenv("S3_PUBLIC_ENDPOINT")
        or os.getenv("S3_PUBLIC_ENDPOINTPY")
        or os.getenv("S3_ENDPOINTPY", "minio:9000")
    )
    access = os.getenv("S3_ACCESS_KEY", "minioadmin")
    secret = os.getenv("S3_SECRET_KEY", "minioadmin")
    secure = os.getenv("S3_PUBLIC_MINIO_SECURE", os.getenv("S3_MINIO_SECURE", "false")).lower() == "true"
    return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)


CLIENT = _mk_client()
PUBLIC_CLIENT = _mk_public_client()

# Поддерживаем несколько исторических имён переменных (чтобы compose/env не ломались при рефакторинге).
RAW_BUCKET = os.getenv("RAW_BUCKET") or os.getenv("S3_BUCKET_RAW", "ai-lowcurrent-raw")
EXPORT_BUCKET = os.getenv("EXPORT_BUCKET") or os.getenv("S3_BUCKET_EXPORTS", "ai-lowcurrent-exports")


def upload_file(bucket: str, local_path: str, key: str) -> str:
    # гарантируем наличие бакета (идемпотентно)
    if not CLIENT.bucket_exists(bucket):
        CLIENT.make_bucket(bucket)
    CLIENT.fput_object(bucket, key, local_path)
    return f"s3://{bucket}/{key}"


def download_file(bucket: str, key: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    CLIENT.fget_object(bucket, key, local_path)
    return local_path


def presigned_get_url(bucket: str, key: str, expires_seconds: int = 3600) -> str:
    """Generate a presigned HTTP GET URL for an object."""
    return PUBLIC_CLIENT.presigned_get_object(
        bucket_name=bucket,
        object_name=key,
        expires=datetime.timedelta(seconds=int(expires_seconds)),
    )
