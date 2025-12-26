from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, List, Literal, Tuple
import os
import os.path as osp

from app.geometry_png import ingest_png
from app.geometry_dxf import ingest_dxf
from app.geometry import DB
from app.placement import generate_candidates, select_devices
from app.routing import route_all
from app.export_dxf import export_dxf
from app.export_pdf import export_pdf
from app.validator import validate_project
from app.bom import make_bom
from app.minio_client import upload_file, download_file, CLIENT, RAW_BUCKET, EXPORT_BUCKET
from app.plan_graph_store import save_plan_graph, load_plan_graph, upload_plan_graph


app = FastAPI(title="Low-Current Planner")

LOCAL_RAW_DIR = os.getenv("LOCAL_DOWNLOAD_DIR", "/data")
LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
LOCAL_EXPORT_DIR = "/tmp/exports"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_DL_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)

UPLOAD_PLAN_GRAPH = (os.getenv("UPLOAD_PLAN_GRAPH", "true").strip().lower() == "true")


class APIIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: str = Field(validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key"))


class APIInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key"))
    preferences_text: Optional[str] = Field(default=None, validation_alias=AliasChoices("preferencesText", "preferences_text"))
    export_formats: List[Literal["PDF", "DXF"]] = Field(default_factory=list, validation_alias=AliasChoices("exportFormats", "export_formats"))


EXPORT_PREFIXES = ("overlays/", "drawings/", "masks/", "plan-graphs/")


def _parse_src(src: str) -> Tuple[str, str]:
    s = (src or "").strip()

    if s.startswith("s3://"):
        rest = s[len("s3://"):]
        if "/" not in rest:
            raise HTTPException(status_code=400, detail=f"Bad srcKey: {src}")
        b, k = rest.split("/", 1)
        return b, k

    if "/" in s:
        first, rest = s.split("/", 1)
        if first in (RAW_BUCKET, EXPORT_BUCKET):
            return first, rest
        if s.startswith(EXPORT_PREFIXES):
            return EXPORT_BUCKET, s
        return RAW_BUCKET, s

    if s.startswith(EXPORT_PREFIXES):
        return EXPORT_BUCKET, s
    return RAW_BUCKET, s


def _download(bucket: str, key: str, local_dir: str) -> str:
    basename = osp.basename(key)
    local_path = osp.join(local_dir, basename)
    if not osp.exists(local_path):
        try:
            download_file(bucket, key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{bucket}/{key}]: {e}")
    return local_path


def _ensure_project_slots(project_id: str) -> None:
    DB.setdefault("rooms", {}).setdefault(project_id, [])
    DB.setdefault("room_meta", {}).setdefault(project_id, [])
    DB.setdefault("devices", {}).setdefault(project_id, [])
    DB.setdefault("routes", {}).setdefault(project_id, [])
    DB.setdefault("candidates", {}).setdefault(project_id, [])
    DB.setdefault("plan_graph", {}).setdefault(project_id, None)
    DB.setdefault("source_meta", {}).setdefault(project_id, {})


def _has_geometry(project_id: str) -> bool:
    rooms = DB.get("rooms", {}).get(project_id, [])
    return bool(rooms)


def _ingest_from_src(project_id: str, src_key: str) -> dict:
    bucket, key = _parse_src(src_key)

    if key.startswith(("overlays/", "drawings/", "masks/")) and bucket == RAW_BUCKET:
        raise HTTPException(
            status_code=400,
            detail="Ты передал overlays/drawings/masks в RAW. Передай исходный план из RAW (raw-plans/...png|dxf) или сначала /ingest, а в /infer srcKey не меняй."
        )

    local_path = _download(bucket, key, LOCAL_RAW_DIR)
    low = local_path.lower()

    if low.endswith((".dxf", ".dwg")):
        return ingest_dxf(project_id, local_path)
    if low.endswith((".png", ".jpg", ".jpeg")):
        return ingest_png(project_id, local_path, src_key=src_key)

    raise HTTPException(status_code=415, detail="Unsupported file type")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/health/s3", tags=["health"])
def health_s3():
    try:
        ok_raw = CLIENT.bucket_exists(RAW_BUCKET)
        ok_exp = CLIENT.bucket_exists(EXPORT_BUCKET)
        return {"raw_bucket": RAW_BUCKET, "exports_bucket": EXPORT_BUCKET, "raw_exists": ok_raw, "exports_exists": ok_exp}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/plan-graph/{project_id}", tags=["pipeline"])
def get_plan_graph(project_id: str):
    # сначала пытаемся с диска (источник истины)
    pg = load_plan_graph(project_id)
    if pg is not None:
        return pg

    # fallback: из памяти, если сервис не перезапускался
    pg = DB.get("plan_graph", {}).get(project_id)
    if pg is None:
        raise HTTPException(status_code=404, detail="PlanGraph not found. Call /ingest first.")
    return pg


@app.post("/ingest", tags=["pipeline"])
async def ingest(req: APIIngestRequest):
    _ensure_project_slots(req.project_id)

    stats = _ingest_from_src(req.project_id, req.src_s3_key)
    plan_graph = DB.get("plan_graph", {}).get(req.project_id)

    if not plan_graph:
        raise HTTPException(status_code=500, detail="Ingest finished but plan_graph is empty.")

    # 1) сохраняем PlanGraph локально
    pg_local = save_plan_graph(req.project_id, plan_graph)

    # 2) (опционально) грузим PlanGraph в exports
    pg_uri = None
    if UPLOAD_PLAN_GRAPH:
        try:
            pg_uri = upload_plan_graph(req.project_id, pg_local)
        except Exception:
            pg_uri = None

    return {
        "project_id": req.project_id,
        "plan_graph": plan_graph,
        "plan_graph_local": pg_local,
        "plan_graph_uploaded": pg_uri,
        **stats,
    }


@app.post("/infer", tags=["pipeline"])
async def infer(req: APIInferRequest):
    _ensure_project_slots(req.project_id)

    # если геометрии нет — подтянем исходник (raw)
    if not _has_geometry(req.project_id):
        if not req.src_s3_key:
            raise HTTPException(
                status_code=400,
                detail="Нет геометрии. Сначала вызови /ingest, либо передай srcKey=путь в RAW на /infer."
            )
        _ingest_from_src(req.project_id, req.src_s3_key)

    if not _has_geometry(req.project_id):
        raise HTTPException(status_code=400, detail="PNG/DXF ingestion не выделил помещения (rooms=0).")

    # placement + routing
    cands = generate_candidates(req.project_id)
    _ = select_devices(req.project_id, cands)
    routes = route_all(req.project_id)

    violations = validate_project(req.project_id)
    bom = make_bom(req.project_id)

    wants = set(req.export_formats or [])
    uploaded_uris: List[str] = []

    if "DXF" in wants:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, DB["rooms"][req.project_id], DB["devices"][req.project_id], DB["routes"][req.project_id], dxf_path)
        uploaded_uris.append(upload_file(EXPORT_BUCKET, dxf_path, f"drawings/{osp.basename(dxf_path)}"))

    if "PDF" in wants:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, DB["rooms"][req.project_id], DB["devices"][req.project_id], DB["routes"][req.project_id], pdf_path)
        uploaded_uris.append(upload_file(EXPORT_BUCKET, pdf_path, f"drawings/{osp.basename(pdf_path)}"))

    return {
        "project_id": req.project_id,
        "rooms": len(DB["rooms"][req.project_id]),
        "devices": len(DB["devices"][req.project_id]),
        "routes": len(routes),
        "violations": violations,
        "bom": bom,
        "uploaded_uris": uploaded_uris,
        "plan_graph": DB.get("plan_graph", {}).get(req.project_id),
    }
