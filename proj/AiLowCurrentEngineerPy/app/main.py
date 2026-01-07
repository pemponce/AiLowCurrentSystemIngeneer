# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, List, Literal, Dict, Tuple, Any
import re as regex
import os
import os.path as osp
import json

from app.geometry_png import ingest_png
from app.lighting import design_lighting, LightingRequest, LightingResponse
from app.geometry import DB
from app.placement import generate_candidates, select_devices
from app.routing import route_all
from app.export_dxf import export_dxf
from app.export_pdf import export_pdf
from app.export_overlay_png import export_overlay_png
from app.geometry_dxf import ingest_dxf
from app.validator import validate_project
from app.bom import make_bom
from app.minio_client import upload_file, download_file, CLIENT, RAW_BUCKET, EXPORT_BUCKET
from app.api_structure import router as structure_router
from app.api_artifacts import router as artifacts_router
from app.api_state import router as state_router
from app.structure_detect import detect_structure

app = FastAPI(title="Low-Current Planner")
app.include_router(structure_router)
app.include_router(artifacts_router)
app.include_router(state_router)

LOCAL_RAW_DIR = os.getenv("LOCAL_DOWNLOAD_DIR", "/data")
LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
LOCAL_EXPORT_DIR = "/tmp/exports"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_DL_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)


# ---------------------------
# JSON normalization helpers
# ---------------------------

def _is_point(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y")


def _point_to_xy(obj: Any) -> Dict[str, float]:
    return {"x": float(getattr(obj, "x")), "y": float(getattr(obj, "y"))}


def _json_default(o: Any):
    # Shapely-like geometries
    if _is_point(o):
        return _point_to_xy(o)
    if hasattr(o, "geom_type") and hasattr(o, "coords"):
        # LineString-like
        return [{"x": float(x), "y": float(y)} for x, y in list(o.coords)]
    if hasattr(o, "geom_type") and hasattr(o, "exterior"):
        # Polygon-like
        ext = [{"x": float(x), "y": float(y)} for x, y in list(o.exterior.coords)]
        holes = []
        for ring in getattr(o, "interiors", []) or []:
            holes.append([{"x": float(x), "y": float(y)} for x, y in list(ring.coords)])
        return {"exterior": ext, "holes": holes}

    # Numpy scalars
    if hasattr(o, "item") and callable(getattr(o, "item")):
        try:
            return o.item()
        except Exception:
            pass

    # Fallback
    return str(o)


def _devices_to_json(devices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in devices or []:
        if isinstance(d, dict):
            dd = dict(d)
            # normalize embedded point fields if present
            if "pt" in dd and _is_point(dd["pt"]):
                dd["x"] = float(dd["pt"].x)
                dd["y"] = float(dd["pt"].y)
                dd.pop("pt", None)
            if "point" in dd and _is_point(dd["point"]):
                dd["x"] = float(dd["point"].x)
                dd["y"] = float(dd["point"].y)
                dd.pop("point", None)
            out.append(dd)
            continue

        # tuple patterns:
        # (TYPE, room_id, Point)
        # (TYPE, room_id, x, y)
        # (TYPE, Point)
        if isinstance(d, (tuple, list)):
            if len(d) >= 3 and _is_point(d[2]):
                dev_type = str(d[0])
                room_id = str(d[1])
                pt = d[2]
                out.append(
                    {
                        "type": dev_type,
                        "label": dev_type,
                        "room_id": room_id,
                        "x": float(pt.x),
                        "y": float(pt.y),
                    }
                )
                continue
            if len(d) >= 4 and isinstance(d[2], (int, float)) and isinstance(d[3], (int, float)):
                dev_type = str(d[0])
                room_id = str(d[1])
                out.append(
                    {
                        "type": dev_type,
                        "label": dev_type,
                        "room_id": room_id,
                        "x": float(d[2]),
                        "y": float(d[3]),
                    }
                )
                continue
            if len(d) >= 2 and _is_point(d[1]):
                dev_type = str(d[0])
                pt = d[1]
                out.append({"type": dev_type, "label": dev_type, "x": float(pt.x), "y": float(pt.y)})
                continue

        # unknown device type: stringify
        out.append({"type": "UNKNOWN", "label": "UNKNOWN", "raw": str(d)})
    return out


def _routes_to_json(routes: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in routes or []:
        if isinstance(r, dict):
            out.append(r)
            continue
        if isinstance(r, tuple) and len(r) >= 3:
            t = str(r[0])
            line = r[1]
            length_m = float(r[2])
            pts = []
            coords = getattr(line, "coords", None)
            if coords is not None:
                for x, y in list(coords):
                    pts.append({"x": float(x), "y": float(y)})
            out.append({"type": t, "length_m": length_m, "points": pts})
    return out


def _strip_bucket_prefix(key: str, bucket: str) -> str:
    if key.startswith(f"{bucket}/"):
        return key[len(bucket) + 1 :]
    return key


def _download_from_raw(key: str) -> str:
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


def _ensure_structure(project_id: str, src_key: Optional[str], local_path: Optional[str]) -> None:
    if DB.get("structure", {}).get(project_id) and DB.get("plan_graph", {}).get(project_id):
        return
    if not src_key and not local_path:
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
        detect_structure(project_id, lp, src_key=src_key)
    except Exception:
        return


class APIIngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: str = Field(validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key"))


class APIPlaceRequest(BaseModel):
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))


class APIRouteRequest(BaseModel):
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))


class APIExportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    formats: List[Literal["PDF", "DXF", "PNG"]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("formats", "exportFormats", "export_formats"),
    )


class APIInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    src_s3_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("srcKey", "src_key", "srcS3Key", "src_s3_key")
    )
    preferences_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("preferencesText", "preferences_text", "user_preferences_text"),
    )
    export_formats: List[str] = Field(default_factory=list, validation_alias=AliasChoices("exportFormats", "export_formats"))


class PreferenceParseResponse(BaseModel):
    per_room_target_lux: Optional[Dict[str, float]] = None
    target_lux: Optional[float] = None
    total_fixtures_hint: Optional[int] = None
    fixture_efficacy_lm_per_w: Optional[float] = None
    notes: Optional[str] = None


def _parse_preferences(text: Optional[str]) -> PreferenceParseResponse:
    resp = PreferenceParseResponse(per_room_target_lux={})
    if not text:
        resp.notes = "preferences_text is empty"
        return resp

    pattern = regex.compile(
        r"([A-Za-zА-Яа-яЁё0-9 _\-]+?)\s*[:\-]?\s*(\d+)\s*(?:лк|lx|lux)\b",
        regex.IGNORECASE,
    )
    for name, val in pattern.findall(text):
        room = name.strip().lower()
        try:
            lux = float(val)
            resp.per_room_target_lux[room] = lux  # type: ignore[index]
        except Exception:
            pass

    if not resp.per_room_target_lux:
        only_num = regex.search(r"\b(\d+)\s*(?:лк|lx|lux)\b", text, flags=regex.IGNORECASE)
        if only_num:
            resp.target_lux = float(only_num.group(1))

    if not resp.per_room_target_lux and not resp.target_lux:
        resp.notes = "Could not parse lux targets"
    return resp


def _ensure_geometry_from_key(project_id: str, src_key: Optional[str]) -> None:
    if DB.get("rooms", {}).get(project_id):
        return
    if not src_key:
        return

    local_path = _download_from_raw(src_key)
    low = local_path.lower()
    if low.endswith((".dxf", ".dwg")):
        ingest_dxf(project_id, local_path)
    elif low.endswith((".png", ".jpg", ".jpeg")):
        ingest_png(project_id, local_path)
    else:
        raise HTTPException(status_code=415, detail="Unsupported file type (expected .dxf/.dwg/.png/.jpg/.jpeg)")


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
    DB.setdefault("exports", {}).setdefault(req.project_id, {})
    DB.setdefault("source", {}).setdefault(req.project_id, {})

    key = _strip_bucket_prefix(req.src_s3_key, RAW_BUCKET)
    basename = osp.basename(key)
    local_path = osp.join(LOCAL_RAW_DIR, basename)

    if not osp.exists(local_path):
        try:
            download_file(RAW_BUCKET, key, local_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"S3 download failed [{RAW_BUCKET}/{key}]: {e}")

    DB["source"][req.project_id] = {"src_key": req.src_s3_key, "local_path": local_path}

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


@app.post("/place")
async def place(req: APIPlaceRequest):
    cands = generate_candidates(req.project_id)
    chosen = select_devices(req.project_id, cands)
    DB["devices"][req.project_id] = _devices_to_json(chosen if isinstance(chosen, list) else DB["devices"].get(req.project_id, []))
    violations = validate_project(req.project_id)
    return {"project_id": req.project_id, "devices": len(DB["devices"][req.project_id]), "violations": violations}


@app.post("/route")
async def route(req: APIRouteRequest):
    routes_raw = route_all(req.project_id)
    routes_json = _routes_to_json(routes_raw)
    DB["routes"][req.project_id] = routes_json

    total = 0.0
    for r in routes_json:
        total += float(r.get("length_m", 0.0))

    bom = make_bom(req.project_id)
    return {"project_id": req.project_id, "routes": len(routes_json), "length_sum": total, "bom": bom}


@app.post("/export")
async def export(req: APIExportRequest):
    rooms = DB.get("rooms", {}).get(req.project_id, [])
    devices = _devices_to_json(DB.get("devices", {}).get(req.project_id, []))
    routes = DB.get("routes", {}).get(req.project_id, [])
    DB["devices"][req.project_id] = devices  # normalize

    out_paths: List[str] = []
    uploaded: List[str] = []
    keys: List[str] = []

    if "DXF" in req.formats:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, rooms, devices, routes, dxf_path)
        out_paths.append(dxf_path)

    if "PDF" in req.formats:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, rooms, devices, routes, pdf_path)
        out_paths.append(pdf_path)

    for p in out_paths:
        key = f"drawings/{osp.basename(p)}"
        try:
            uri = upload_file(EXPORT_BUCKET, p, key)
            uploaded.append(uri)
            keys.append(key)
        except Exception as e:
            uploaded.append(f"ERROR:{e}")

    DB.setdefault("exports", {}).setdefault(req.project_id, {})
    DB["exports"][req.project_id].update({"files": out_paths, "uploaded_uris": uploaded, "keys": keys})

    return {"project_id": req.project_id, "files": out_paths, "uploaded": uploaded, "keys": keys}


@app.post("/infer")
async def infer(req: APIInferRequest):
    DB.setdefault("rooms", {}).setdefault(req.project_id, [])
    DB.setdefault("devices", {}).setdefault(req.project_id, [])
    DB.setdefault("routes", {}).setdefault(req.project_id, [])
    DB.setdefault("candidates", {}).setdefault(req.project_id, [])
    DB.setdefault("exports", {}).setdefault(req.project_id, {})
    DB.setdefault("source", {}).setdefault(req.project_id, {})

    if req.src_s3_key:
        DB["source"][req.project_id] = {"src_key": req.src_s3_key}

    _ensure_geometry_from_key(req.project_id, req.src_s3_key)
    _ensure_structure(req.project_id, req.src_s3_key, None)

    if not DB.get("rooms", {}).get(req.project_id):
        raise HTTPException(
            status_code=400,
            detail="Нет геометрии помещений. Передай srcKey в /infer или сначала вызови /ingest.",
        )

    parsed = _parse_preferences(req.preferences_text)

    lighting_summary: dict = {}
    try:
        kwargs = {"project_id": req.project_id}
        if parsed.per_room_target_lux:
            kwargs["per_room_target_lux"] = parsed.per_room_target_lux
        if parsed.target_lux:
            kwargs["target_lux"] = parsed.target_lux

        lr = LightingRequest(**kwargs)  # type: ignore[arg-type]
        lighting_resp = design_lighting(lr)
        lighting_summary = lighting_resp.model_dump() if isinstance(lighting_resp, BaseModel) else lighting_resp
    except Exception:
        rooms = DB.get("rooms", {}).get(req.project_id, [])
        lighting_summary = {"rooms_count": len(rooms), "note": "LightingResponse не сформирован — вернулась сводка."}

    # placement
    cands = generate_candidates(req.project_id)
    chosen = select_devices(req.project_id, cands)
    DB["devices"][req.project_id] = _devices_to_json(chosen if isinstance(chosen, list) else DB["devices"].get(req.project_id, []))

    # routing
    routes_raw = route_all(req.project_id)
    routes_json = _routes_to_json(routes_raw)
    DB["routes"][req.project_id] = routes_json

    violations = validate_project(req.project_id)

    wants = set([str(x).upper() for x in (req.export_formats or [])])
    exported_files: List[str] = []
    uploaded_uris: List[str] = []
    exported_keys: List[str] = []

    rooms_json = DB.get("rooms", {}).get(req.project_id, [])
    devices_json = DB.get("devices", {}).get(req.project_id, [])
    routes_json = DB.get("routes", {}).get(req.project_id, [])

    # --- DXF ---
    if "DXF" in wants:
        dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
        export_dxf(req.project_id, rooms_json, devices_json, routes_json, dxf_path)
        exported_files.append(dxf_path)
        try:
            key = f"drawings/{osp.basename(dxf_path)}"
            uploaded_uris.append(upload_file(EXPORT_BUCKET, dxf_path, key))
            exported_keys.append(key)
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    # --- PDF ---
    if "PDF" in wants:
        pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
        export_pdf(req.project_id, rooms_json, devices_json, routes_json, pdf_path)
        exported_files.append(pdf_path)
        try:
            key = f"drawings/{osp.basename(pdf_path)}"
            uploaded_uris.append(upload_file(EXPORT_BUCKET, pdf_path, key))
            exported_keys.append(key)
        except Exception as e:
            uploaded_uris.append(f"ERROR:{e}")

    # --- PNG overlay ---
    overlay_key = None
    if "PNG" in wants:
        src_key = req.src_s3_key
        if not src_key:
            src_key = (DB.get("source", {}).get(req.project_id) or {}).get("src_key")
        if not src_key:
            src_key = (DB.get("source_meta", {}).get(req.project_id) or {}).get("srcKey")

        if src_key:
            src_path = _download_from_raw(src_key)
            overlay_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}_overlay.png"
            export_overlay_png(src_path, rooms_json, devices_json, routes_json, overlay_path)
            exported_files.append(overlay_path)
            try:
                overlay_key = f"overlays/{osp.basename(overlay_path)}"
                uploaded_uris.append(upload_file(EXPORT_BUCKET, overlay_path, overlay_key))
                exported_keys.append(overlay_key)
            except Exception as e:
                uploaded_uris.append(f"ERROR:{e}")

    # --- JSON result (ALWAYS) ---
    result_key = f"results/{req.project_id}.json"
    result_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.json"
    result_payload = {
        "project_id": req.project_id,
        "plan_graph": DB.get("plan_graph", {}).get(req.project_id),
        "structure": DB.get("structure", {}).get(req.project_id),
        "rooms": rooms_json,
        "devices": devices_json,
        "routes": routes_json,
        "lighting_summary": lighting_summary,
        "violations": violations,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2, default=_json_default)

    try:
        uploaded_uris.append(upload_file(EXPORT_BUCKET, result_path, result_key))
        exported_keys.append(result_key)
    except Exception as e:
        uploaded_uris.append(f"ERROR:{e}")

    DB["exports"][req.project_id] = {
        "exported_files": exported_files,
        "uploaded_uris": uploaded_uris,
        "keys": exported_keys,
        "overlay_key": overlay_key,
        "result_json_key": result_key,
    }

    return {
        "project_id": req.project_id,
        "parsed": parsed.model_dump(),
        "lighting_summary": lighting_summary,
        "rooms": rooms_json,
        "devices": devices_json,
        "routes": routes_json,
        "plan_graph": DB.get("plan_graph", {}).get(req.project_id),
        "structure": DB.get("structure", {}).get(req.project_id),
        "violations": violations,
        "exports": DB.get("exports", {}).get(req.project_id, {}),
    }
