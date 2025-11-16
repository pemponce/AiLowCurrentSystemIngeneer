from __future__ import annotations

import os
from typing import Tuple
from fastapi import UploadFile
from .geometry import DB
from .minio_client import upload_file

RAW_BUCKET = os.getenv("S3_BUCKET_RAW", "raw-plans")


def register_project(project_id: str, filename: str | None) -> None:
    DB.setdefault("projects", {})[project_id] = {
        "filename": filename,
        "status": "created"
    }


def store_upload_to_minio(project_id: str, fileobj: UploadFile) -> Tuple[str, str]:
    """
    Кладём файл в MinIO: s3://RAW_BUCKET/plans/<project_id>/<original_filename>
    Возвращаем (local_tmp_path, s3_uri)
    """
    os.makedirs("/tmp/uploads", exist_ok=True)
    local_path = f"/tmp/uploads/{project_id}__{fileobj.filename}"
    with open(local_path, "wb") as f:
        f.write(fileobj.file.read())

    key = f"plans/{project_id}/{fileobj.filename}"
    uri = upload_file(RAW_BUCKET, local_path, key)

    # Запишем в DB
    DB.setdefault("uploads", {})[project_id] = {
        "local_path": local_path,
        "s3_key": key,
        "s3_uri": uri,
        "original_name": fileobj.filename,
    }
    DB["projects"][project_id]["status"] = "uploaded"
    return local_path, uri


def ensure_dxf(local_path: str) -> str:
    """
    Гарантируем DXF. Если уже DXF — возвращаем путь.
    Если DWG — здесь место для конвертации DWG->DXF (ODA/LibreDWG).
    Пока просто возвращаем 400 на уровне хендлера, если расширение .dwg.
    """
    ext = os.path.splitext(local_path)[1].lower()
    if ext == ".dxf":
        return local_path
    if ext == ".dwg":
        # TODO: подключить dwg2dxf (ODAFileConverter) внутри контейнера
        raise ValueError("DWG конвертация не настроена. Сохрани чертеж как DXF и загрузи заново.")
    raise ValueError(f"Неподдерживаемый формат '{ext}'. Загрузите .dxf (или .dwg после включения конвертера).")
