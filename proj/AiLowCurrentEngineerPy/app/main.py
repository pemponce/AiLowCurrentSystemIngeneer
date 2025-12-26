import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, AliasChoices
from starlette.middleware.cors import CORSMiddleware

from app.geometry import DB
from app.placement import generate_candidates, select_devices
from app.routing import route_all
from app.bom import make_bom
from app.validate import validate_project
from app.export_dxf import export_dxf
from app.export_pdf import export_pdf
from app.geometry_dxf import ingest_dxf
from app.geometry_png import ingest_png
from app.lighting import LightingRequest, LightingResponse, design_lighting
from app.minio_client import upload_file, download_file, CLIENT, RAW_BUCKET, EXPORT_BUCKET
from app.api_structure import router as structure_router
from app.structure_detect import detect_structure


# -------------------------
# Config
# -------------------------
LOCAL_RAW_DIR = os.getenv("LOCAL_RAW_DIR", "/tmp/raw")
LOCAL_DL_DIR = os.getenv("LOCAL_DL_DIR", "/tmp/downloads")
LOCAL_EXPORT_DIR = os.getenv("LOCAL_EXPORT_DIR", "/tmp/exports")

os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_DL_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)


# -------------------------
# API models (Pydantic v2)
# -------------------------
class APIIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: str = Field(validation_alias=AliasChoices("srcKey", "srcKey", "src_s3_key", "srcS3Key"))


class APIInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("srcKey", "src_s3_key", "srcS3Key"),
    )
    preferences_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("preferencesText", "preferences_text"),
    )
    export_formats: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("exportFormats", "export_formats"),
    )


class PreferenceParseResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    per_room_target_lux: Dict[str, int] = {}
    target_lux: Optional[int] = None
    notes: Optional[str] = None


# =======================================
#                Helpers
# =======================================

def _strip_bucket_prefix(key: str, bucket: str) -> str:
    """Если ключ пришёл как 'bucket/key', убрать префикс 'bucket/'."""
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1:]
    return key


def _download_from_raw(key: str) -> str:
    """Скачать файл из RAW в локальную папку и вернуть локальный путь."""
    clean_key = _strip_bucket_prefix(key, RAW_BUCKET)
    basename = osp.basename(clean_key)
    local_path = osp.join(LOCAL_DL_DIR, basename)
    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, clean_key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{clean_key}]: {e}")
    return local_path


def _ensure_geometry_from_key(project_id: str, src_key: Optional[str]) -> Optional[str]:
    """
    Если геометрии нет — подтянуть из RAW по ключу.
    Возвращает local_path (если png/jpg скачали) — полезно для structure_detect.
    """
    if DB.get("rooms", {}).get(project_id):
        return None

    if not src_key:
        return None

    local_path = _download_from_raw(src_key)
    low = local_path.lower()

    if low.endswith((".dxf", ".dwg")):
        ingest_dxf(project_id, local_path)
        return None

    if low.endswith((".png", ".jpg", ".jpeg")):
        ingest_png(project_id, local_path)
        return local_path

    raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")


def _ensure_structure(project_id: str, src_key: Optional[str], local_path: Optional[str]) -> None:
    """Если structure/plan_graph ещё нет — попытаться детектировать из PNG/JPG."""
    if DB.get("structure", {}).get(project_id) and DB.get("plan_graph", {}).get(project_id):
        return

    lp = None
    if local_path and osp.exists(local_path):
        lp = local_path
    elif src_key:
        try:
            lp = _download_from_raw(src_key)
        except Exception:
            lp = None

    if not lp:
        return

    low = lp.lower()
    if not low.endswith((".png", ".jpg", ".jpeg")):
        return

    try:
        detect_structure(project_id, lp, src_key=src_key, debug=True)
    except Exception:
        # Не блокируем infer: placement/routing имеют fallback.
        return


def _parse_preferences(text: Optional[str]) -> PreferenceParseResponse:
    """
    Простой парсер строк вида:
    'кухня 500 лк, гостиная 300 лк'
    """
    resp = PreferenceParseResponse(per_room_target_lux={})
    if not text:
        resp.notes = "preferencesText is empty"
        return resp

    pattern = re.compile(
        r"([A-Za-zА-Яа-яЁё0-9 _\-]+?)\s*[:\-]?\s*(\d+)\s*(?:лк|lx|lux)\b",
        flags=re.IGNORECASE,
    )

    for m in pattern.finditer(text):
        name = m.group(1).strip().lower()
        val = int(m.group(2))
        resp.per_room_target_lux[name] = val

    m2 = re.search(r"(\d+)\s*(?:лк|lx|lux)\b", text, flags=re.IGNORECASE)
    if m2:
        resp.target_lux = int(m2.group(1))

    if not resp.per_room_target_lux:
        resp.notes = "No per-room lux found (expected 'кухня 500 лк, ...')."
    return resp


# =======================================
#                App
# =======================================

app = FastAPI(title="AiLowCurrentEngineerPy", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутер структуры
app.include_router(structure_router)


# =======================================
#                Routes
# =======================================

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/plan-graph/{project_id}")
def get_plan_graph(project_id: str):
    pg = DB.get("plan_graph", {}).get(project_id)
    if not pg:
        raise HTTPException(status_code=404, detail="plan_graph not found (run /detect-structure first)")
    return pg


@app.get("/health/s3")
def health_s3():
    try:
        ok = CLIENT.bucket_exists(RAW_BUCKET)
        return {"ok": ok, "endpoint": os.getenv("S3_ENDPOINTPY"), "raw_bucket": RAW_BUCKET}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/ingest")
async def ingest(req: APIIngestRequest):
    DB.setdefault("rooms", {}).setdefault(req.project_id, [])
    DB.setdefault("devices", {}).setdefault(req.project_id, [])
    DB.setdefault("routes", {}).setdefault(req.project_id, [])
    DB.setdefault("candidates", {}).setdefault(req.project_id, [])

    key = _strip_bucket_prefix(req.src_s3_key, RAW_BUCKET)
    basename = osp.basename(key)
    local_path = osp.join(LOCAL_RAW_DIR, basename)

    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{key}]: {e}")

    low = local_path.lower()
    if low.endswith((".dxf", ".dwg")):
        stats = ingest_dxf(req.project_id, local_path)
    elif low.endswith((".png", ".jpg", ".jpeg")):
        stats = ingest_png(req.project_id, local_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")

    return {"project_id": req.project_id, **stats}


@app.post("/lighting", response_model=LightingResponse)
async def lighting(req: LightingRequest) -> LightingResponse:
    return design_lighting(req)


@app.post("/infer")
async def infer(req: APIInferRequest):
    DB.setdefault("rooms", {}).setdefault(req.project_id, [])
    DB.setdefault("devices", {}).setdefault(req.project_id, [])
    DB.setdefault("routes", {}).setdefault(req.project_id, [])
    DB.setdefault("candidates", {}).setdefault(req.project_id, [])

    # 1) Если нет помещений — попробуем авто-инжест из RAW
    local_path = _ensure_geometry_from_key(req.project_id, req.src_s3_key)

    if not DB["rooms"].get(req.project_id):
        raise HTTPException(
            status_code=400,
            detail="Нет геометрии помещений. Передай srcKey в /infer или сначала вызови /ingest.",
        )

    # 2) Детект структуры (стены/проёмы) — чтобы placement/routing работали точнее
    _ensure_structure(req.project_id, req.src_s3_key, local_path)

    # 3) Разобрать предпочтения
    parsed = _parse_preferences(req.preferences_text)

    # 4) Освещение
    lighting_summary: Dict[str, Any] = {}
    try:
        kwargs: Dict[str, Any] = {"project_id": req.project_id}
        if parsed.per_room_target_lux:
            kwargs["per_room_target_lux"] = parsed.per_room_target_lux
        if parsed.target_lux:
            kwargs["target_lux"] = parsed.target_lux

        lr = LightingRequest(**kwargs)  # type: ignore[arg-type]
        lighting_resp = design_lighting(lr)
        lighting_summary = lighting_resp.model_dump()
    except Exception:
        rooms = DB["rooms"].get(req.project_id, [])
        lighting_summary = {"rooms_count": len(rooms), "note": "LightingResponse не сформирован (fallback)."}

    # 5) Расстановка и трассировка
    cands = generate_candidates(req.project_id)
    _ = select_devices(req.project_id, cands)
    routes = route_all(req.project_id)
    _ = validate_project(req.project_id)

    # 6) BOM
    bom = {}
    try:
        bom = make_bom(req.project_id)
    except Exception:
        bom = {}

    # 7) Экспорт
    wants = set(req.export_formats or [])
    exported_files: List[str] = []
    uploaded_uris: List[str] = []

    if "DXF" in wants:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, DB["rooms"][req.project_id], DB["devices"][req.project_id], DB["routes"][req.project_id], dxf_path)
        exported_files.append(dxf_path)
        try:
            uploaded_uris.append(upload_file(EXPORT_BUCKET, dxf_path, f"drawings/{osp.basename(dxf_path)}"))
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    if "PDF" in wants:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, DB["rooms"][req.project_id], DB["devices"][req.project_id], DB["routes"][req.project_id], pdf_path)
        exported_files.append(pdf_path)
        try:
            uploaded_uris.append(upload_file(EXPORT_BUCKET, pdf_path, f"drawings/{osp.basename(pdf_path)}"))
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    return {
        "project_id": req.project_id,
        "parsed": parsed.model_dump(),
        "lighting_summary": lighting_summary,
        "routes_count": len(routes),
        "bom": bom,
        "exported_files": exported_files,
        "uploaded_uris": uploaded_uris,
    }
