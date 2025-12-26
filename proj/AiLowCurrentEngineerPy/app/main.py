from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, List, Literal, Dict
import re as regex
import os
import os.path as osp

from app.geometry_png import ingest_png
from app.lighting import design_lighting, LightingRequest, LightingResponse
from app.geometry import DB
from app.placement import generate_candidates, select_devices
from app.routing import route_all
from app.export_dxf import export_dxf
from app.export_pdf import export_pdf
from app.geometry_dxf import ingest_dxf
from app.validator import validate_project
from app.bom import make_bom
from app.minio_client import upload_file, download_file, CLIENT, RAW_BUCKET, EXPORT_BUCKET
from app.api_structure import router as structure_router

app = FastAPI(title="Low-Current Planner")
app.include_router(structure_router)

# --------- Константы окружения ----------
# Buckets берём из app.minio_client (единый источник правды)
LOCAL_RAW_DIR = os.getenv("LOCAL_DOWNLOAD_DIR", "/data")
LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
LOCAL_EXPORT_DIR = "/tmp/exports"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_DL_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)

# =======================================
#            Pydantic модели (API)
# =======================================

class APIIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: str = Field(
        validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key")
    )

class APIPlaceRequest(BaseModel):
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))

class APIRouteRequest(BaseModel):
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))

class APIExportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    formats: List[Literal['PDF', 'DXF', 'PNG']] = Field(
        default_factory=list,
        validation_alias=AliasChoices("formats", "exportFormats", "export_formats")
    )

class APIInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key")
    )
    preferences_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("preferencesText", "preferences_text", "user_preferences_text")
    )
    export_formats: List[Literal['PDF', 'DXF', 'PNG']] = Field(
        default_factory=list,
        validation_alias=AliasChoices("exportFormats", "export_formats")
    )

class PreferenceParseResponse(BaseModel):
    per_room_target_lux: Optional[Dict[str, float]] = None
    target_lux: Optional[float] = None
    total_fixtures_hint: Optional[int] = None
    fixture_efficacy_lm_per_w: Optional[float] = None
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

def _parse_preferences(text: Optional[str]) -> PreferenceParseResponse:
    """
    Очень простой парсер строк вида:
    'Кухня 500 лк, гостиная 300 лк, коридор 150 лк'
    """
    resp = PreferenceParseResponse(per_room_target_lux={})
    if not text:
        resp.notes = "preferences_text is empty"
        return resp

    pattern = regex.compile(r"([A-Za-zА-Яа-яЁё0-9 _\\-]+?)\\s*[:\\-]?\\s*(\\d+)\\s*(?:лк|lx|lux)\\b", regex.IGNORECASE)
    for name, val in pattern.findall(text):
        room = name.strip().lower()
        try:
            lux = float(val)
            resp.per_room_target_lux[room] = lux  # type: ignore[index]
        except Exception:
            pass

    if not resp.per_room_target_lux:
        only_num = regex.search(r"\\b(\\d+)\\s*(?:лк|lx|lux)\\b", text, flags=regex.IGNORECASE)
        if only_num:
            resp.target_lux = float(only_num.group(1))

    if not resp.per_room_target_lux and not resp.target_lux:
        resp.notes = "Could not parse lux targets"
    return resp

def _ensure_geometry_from_key(project_id: str, src_key: Optional[str]) -> None:
    """
    Если геометрии нет — пытаемся подтянуть из RAW по ключу.
    """
    if DB.get('rooms', {}).get(project_id):
        return

    if not src_key:
        return

    local_path = _download_from_raw(src_key)
    low = local_path.lower()
    if low.endswith(('.dxf', '.dwg')):
        ingest_dxf(project_id, local_path)
    elif low.endswith(('.png', '.jpg', '.jpeg')):
        ingest_png(project_id, local_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")

# =======================================
#                Маршруты
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

@app.post('/ingest')
async def ingest(req: APIIngestRequest):
    DB.setdefault('rooms', {}).setdefault(req.project_id, [])
    DB.setdefault('devices', {}).setdefault(req.project_id, [])
    DB.setdefault('routes', {}).setdefault(req.project_id, [])
    DB.setdefault('candidates', {}).setdefault(req.project_id, [])

    key = _strip_bucket_prefix(req.src_s3_key, RAW_BUCKET)
    basename = osp.basename(key)
    local_path = osp.join(LOCAL_RAW_DIR, basename)

    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{key}]: {e}")

    low = local_path.lower()
    if low.endswith(('.dxf', '.dwg')):
        stats = ingest_dxf(req.project_id, local_path)
    elif low.endswith(('.png', '.jpg', '.jpeg')):
        stats = ingest_png(req.project_id, local_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")

    return {"project_id": req.project_id, **stats}

@app.post("/lighting", response_model=LightingResponse)
async def lighting(req: LightingRequest) -> LightingResponse:
    return design_lighting(req)

@app.post('/place')
async def place(req: APIPlaceRequest):
    cands = generate_candidates(req.project_id)
    chosen = select_devices(req.project_id, cands)
    violations = validate_project(req.project_id)
    return {"project_id": req.project_id, "devices": len(chosen), "violations": violations}

@app.post('/route')
async def route(req: APIRouteRequest):
    routes = route_all(req.project_id)
    total = sum(l for _, _, l in routes)
    bom = make_bom(req.project_id)
    return {"project_id": req.project_id, "routes": len(routes), "length_sum": total, "bom": bom}

@app.post('/export')
async def export(req: APIExportRequest):
    rooms = DB['rooms'].get(req.project_id, [])
    devices = DB['devices'].get(req.project_id, [])
    routes = DB['routes'].get(req.project_id, [])

    out_paths: List[str] = []

    if 'DXF' in req.formats:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, rooms, devices, routes, dxf_path)
        out_paths.append(dxf_path)

    if 'PDF' in req.formats:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, rooms, devices, routes, pdf_path)
        out_paths.append(pdf_path)

    uploaded: List[str] = []
    for p in out_paths:
        key = f"drawings/{osp.basename(p)}"
        try:
            uri = upload_file(EXPORT_BUCKET, p, key)
            uploaded.append(uri)
        except Exception as e:
            uploaded.append(f"ERROR:{e}")

    return {"project_id": req.project_id, "files": out_paths, "uploaded": uploaded}

@app.post('/infer')
async def infer(req: APIInferRequest):
    DB.setdefault('rooms', {}).setdefault(req.project_id, [])
    DB.setdefault('devices', {}).setdefault(req.project_id, [])
    DB.setdefault('routes', {}).setdefault(req.project_id, [])
    DB.setdefault('candidates', {}).setdefault(req.project_id, [])

    _ensure_geometry_from_key(req.project_id, req.src_s3_key)

    if not DB['rooms'].get(req.project_id):
        raise HTTPException(
            status_code=400,
            detail="Нет геометрии помещений. Передай srcKey в /infer или сначала вызови /ingest."
        )

    parsed = _parse_preferences(req.preferences_text)

    lighting_summary = {}
    try:
        kwargs = {"project_id": req.project_id}
        if parsed.per_room_target_lux:
            kwargs["per_room_target_lux"] = parsed.per_room_target_lux
        if parsed.target_lux:
            kwargs["target_lux"] = parsed.target_lux

        lr = LightingRequest(**kwargs)  # type: ignore[arg-type]
        lighting_resp = design_lighting(lr)
        lighting_summary = lighting_resp.model_dump() if isinstance(lighting_resp, BaseModel) else lighting_resp  # type: ignore[attr-defined]
    except Exception:
        rooms = DB['rooms'].get(req.project_id, [])
        lighting_summary = {
            "rooms_count": len(rooms),
            "note": "LightingResponse не сформирован — вернулась сводка по помещениям."
        }

    cands = generate_candidates(req.project_id)
    _ = select_devices(req.project_id, cands)
    _routes = route_all(req.project_id)
    _ = validate_project(req.project_id)

    wants = set(req.export_formats or [])
    exported_files: List[str] = []
    uploaded_uris: List[str] = []

    if 'DXF' in wants:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, DB['rooms'][req.project_id], DB['devices'][req.project_id], DB['routes'][req.project_id], dxf_path)
        exported_files.append(dxf_path)
        try:
            uploaded_uris.append(upload_file(EXPORT_BUCKET, dxf_path, f"drawings/{osp.basename(dxf_path)}"))
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    if 'PDF' in wants:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, DB['rooms'][req.project_id], DB['devices'][req.project_id], DB['routes'][req.project_id], pdf_path)
        exported_files.append(pdf_path)
        try:
            uploaded_uris.append(upload_file(EXPORT_BUCKET, pdf_path, f"drawings/{osp.basename(pdf_path)}"))
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    return {
        "project_id": req.project_id,
        "parsed": parsed.model_dump(),
        "lighting_summary": lighting_summary,
        "exported_files": exported_files,
        "uploaded_uris": uploaded_uris
    }
