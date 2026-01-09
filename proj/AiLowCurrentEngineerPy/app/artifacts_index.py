from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, presigned_get_url


def _safe_presigned(bucket: str, key: str, expires_seconds: int) -> Optional[str]:
    try:
        return presigned_get_url(bucket, key, expires_seconds)
    except Exception:
        return None


def _as_key(v: Any) -> Optional[str]:
    if not v:
        return None
    if isinstance(v, str):
        return v
    return str(v)


def _artifact_obj(bucket: str, key: Optional[str], expires_seconds: int) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    url = _safe_presigned(bucket, key, expires_seconds)
    obj: Dict[str, Any] = {"bucket": bucket, "key": key}
    if url:
        obj["url"] = url
    obj["s3_uri"] = f"s3://{bucket}/{key}"
    return obj


def _infer_format_from_key(key: str) -> str:
    k = key.lower()
    if k.endswith(".pdf"):
        return "PDF"
    if k.endswith(".dxf"):
        return "DXF"
    if k.endswith(".png"):
        return "PNG"
    if k.endswith(".json"):
        return "JSON"
    return "FILE"


def build_artifacts_index(project_id: str, expires_seconds: int = 3600) -> Dict[str, Any]:
    """Единый индекс артефактов для API.

    Формат:
      artifacts.preview.key/url
      artifacts.overlay.key/url
      artifacts.files[] (PDF/DXF/PNG/…)
      artifacts.result_json.key/url
    """

    exports: Dict[str, Any] = DB.get("exports", {}).get(project_id, {}) or {}
    keys: List[str] = []

    # Список всех ключей, которые у нас есть
    raw_keys = exports.get("keys") or exports.get("export_keys") or exports.get("exportKeys")
    if isinstance(raw_keys, list):
        keys.extend([str(k) for k in raw_keys if k])

    # legacy поля
    preview_key = _as_key(exports.get("preview_key") or exports.get("previewKey"))
    overlay_key = _as_key(exports.get("overlay_key") or exports.get("overlayKey"))
    dxf_key = _as_key(exports.get("dxf_key") or exports.get("dxfKey"))
    pdf_key = _as_key(exports.get("pdf_key") or exports.get("pdfKey"))
    result_key = _as_key(exports.get("result_json_key") or exports.get("resultJsonKey"))

    # fallback по keys
    if not result_key:
        for k in keys:
            if k.startswith("results/") and k.endswith(f"{project_id}.json"):
                result_key = k
                break
        if not result_key:
            result_key = f"results/{project_id}.json"

    if not preview_key:
        for k in keys:
            if k.endswith("/preview.png") and f"/{project_id}/" in k:
                preview_key = k
                break

    if not overlay_key:
        for k in keys:
            if "overlay" in k.lower() and k.lower().endswith(".png"):
                overlay_key = k
                break

    if not dxf_key:
        for k in keys:
            if k.lower().endswith(".dxf"):
                dxf_key = k
                break

    if not pdf_key:
        for k in keys:
            if k.lower().endswith(".pdf"):
                pdf_key = k
                break

    files: List[Dict[str, Any]] = []
    for k in keys:
        if k == preview_key or k == overlay_key or k == result_key:
            continue
        fmt = _infer_format_from_key(k)
        if fmt in ("PDF", "DXF", "PNG"):
            o = _artifact_obj(EXPORT_BUCKET, k, expires_seconds)
            if o:
                o["format"] = fmt
                files.append(o)

    # если keys пустые, но известны dxf/pdf — добавим
    if dxf_key and not any(f.get("key") == dxf_key for f in files):
        o = _artifact_obj(EXPORT_BUCKET, dxf_key, expires_seconds)
        if o:
            o["format"] = "DXF"
            files.append(o)

    if pdf_key and not any(f.get("key") == pdf_key for f in files):
        o = _artifact_obj(EXPORT_BUCKET, pdf_key, expires_seconds)
        if o:
            o["format"] = "PDF"
            files.append(o)

    return {
        "project_id": project_id,
        "preview": _artifact_obj(EXPORT_BUCKET, preview_key, expires_seconds),
        "overlay": _artifact_obj(EXPORT_BUCKET, overlay_key, expires_seconds),
        "files": files,
        "result_json": _artifact_obj(EXPORT_BUCKET, result_key, expires_seconds),
    }


def build_artifacts_manifest(project_id: str) -> Dict[str, Any]:
    """Стабильная версия индекса (без presigned URL), пригодна для results/*.json."""
    exports: Dict[str, Any] = DB.get("exports", {}).get(project_id, {}) or {}
    keys: List[str] = []
    raw_keys = exports.get("keys") or exports.get("export_keys") or exports.get("exportKeys")
    if isinstance(raw_keys, list):
        keys.extend([str(k) for k in raw_keys if k])

    preview_key = _as_key(exports.get("preview_key") or exports.get("previewKey"))
    overlay_key = _as_key(exports.get("overlay_key") or exports.get("overlayKey"))
    result_key = _as_key(exports.get("result_json_key") or exports.get("resultJsonKey")) or f"results/{project_id}.json"

    def stable(key: Optional[str]) -> Optional[Dict[str, Any]]:
        if not key:
            return None
        return {"bucket": EXPORT_BUCKET, "key": key, "s3_uri": f"s3://{EXPORT_BUCKET}/{key}"}

    files = []
    for k in keys:
        if k in (preview_key, overlay_key, result_key):
            continue
        fmt = _infer_format_from_key(k)
        if fmt in ("PDF", "DXF", "PNG"):
            o = stable(k)
            if o:
                o["format"] = fmt
                files.append(o)

    return {
        "project_id": project_id,
        "preview": stable(preview_key),
        "overlay": stable(overlay_key),
        "files": files,
        "result_json": stable(result_key),
    }
