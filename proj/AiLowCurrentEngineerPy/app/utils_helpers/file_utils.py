# app/utils_helpers/file_utils.py
"""
File handling utilities для загрузки и конвертации файлов.
"""

import os
import os.path as osp
from typing import Tuple
from fastapi import HTTPException

from app.minio_client import download_file, RAW_BUCKET, EXPORT_BUCKET

LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
os.makedirs(LOCAL_DL_DIR, exist_ok=True)


def _strip_bucket_prefix(key: str, bucket: str) -> str:
    """Убирает префикс bucket/ из S3 ключа."""
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1:]
    return key


def _download_from_raw(key: str) -> str:
    """
    Скачивает файл из MinIO (RAW_BUCKET или EXPORT_BUCKET).

    Args:
        key: S3 ключ файла (с или без префикса bucket/)

    Returns:
        Локальный путь к скачанному файлу

    Raises:
        HTTPException: Если файл не найден в обоих buckets
    """

    def _try(bucket: str) -> Tuple[bool, str]:
        clean = _strip_bucket_prefix(key, bucket)
        basename = osp.basename(clean)
        local = osp.join(LOCAL_DL_DIR, basename)
        if osp.exists(local):
            return True, local
        try:
            download_file(bucket, clean, local)
            return True, local
        except Exception as e:
            return False, f"S3 download failed [{bucket}/{clean}]: {e}"

    ok, res = _try(RAW_BUCKET)
    if ok:
        return res
    ok2, res2 = _try(EXPORT_BUCKET)
    if ok2:
        return res2
    raise HTTPException(status_code=400, detail=f"{res}; fallback failed [{res2}]")


def _ensure_png_path(local_path: str) -> str:
    """
    Конвертирует JPG/JPEG в PNG если нужно.
    export_preview_png ожидает PNG формат.

    Args:
        local_path: Путь к изображению

    Returns:
        Путь к PNG файлу (оригинал или сконвертированный)
    """
    low = (local_path or "").lower()
    if low.endswith(".png"):
        return local_path

    if low.endswith((".jpg", ".jpeg")):
        try:
            from PIL import Image  # type: ignore
            out_path = osp.join(
                LOCAL_DL_DIR,
                osp.splitext(osp.basename(local_path))[0] + ".png"
            )
            if osp.exists(out_path):
                return out_path
            Image.open(local_path).convert("RGBA").save(out_path)
            return out_path
        except Exception:
            return local_path

    return local_path