# app/main.py

from fastapi import FastAPI, HTTPException
from app.api.design_compare import router as compare_router
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import Optional, List, Literal, Dict, Tuple, Any
import re as regex
import os
import os.path as osp
import json
import logging

from app.geometry_png import ingest_png
from app.export_preview_png import export_preview_png, export_preview_canvas_png
from app.lighting import design_lighting, LightingRequest, LightingResponse
from app.geometry import DB
from app.placement import _apply_hard_rules
# placement.py — рефакторинг в процессе
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
from app.artifacts_index import build_artifacts_index, build_artifacts_manifest

# NN-2: парсинг пожеланий клиента
try:
    from app.nn2.infer import parse_text as nn2_parse_text

    _NN2_AVAILABLE = True
except Exception as _nn2_err:
    _NN2_AVAILABLE = False
    logger_tmp = __import__("logging").getLogger("planner")
    logger_tmp.warning(f"NN-2 недоступна: {_nn2_err}")

# NN-3: размещение устройств
try:
    from app.nn3.infer import run_placement as nn3_run_placement

    _NN3_AVAILABLE = True
except Exception as _nn3_err:
    _NN3_AVAILABLE = False
    logger_tmp2 = __import__("logging").getLogger("planner")
    logger_tmp2.warning(f"NN-3 недоступна: {_nn3_err}")

# NN-1: сегментация + классификация комнат
try:
    import torch as _torch
    import cv2 as _cv2
    from app.ml.structure_infer import _load_checkpoint as _nn1_load_ckpt, infer_one as _nn1_infer_one, \
        _preprocess as _nn1_preprocess
    from app.ml.structure_postprocess import extract_geometry as _nn1_extract_geometry

    _NN1_CKPT = "models/structure_rf_v4_bestreal.pt"
    _nn1_model = None
    _NN1_AVAILABLE = True
except Exception as _nn1_err:
    import logging as _logging

    _logging.getLogger("planner").warning(f"NN-1 недоступна: {_nn1_err}")
    _NN1_AVAILABLE = False


def _nn1_get_rooms(image_path: str, project_id: str) -> list:
    """Запускает NN-1 и возвращает список комнат с типами и polygonPx."""
    if not _NN1_AVAILABLE:
        return []
    global _nn1_model
    try:
        import os, cv2
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        if _nn1_model is None:
            _nn1_model, _, _ = _nn1_load_ckpt(_NN1_CKPT, device)
        img = cv2.imread(image_path)
        if img is None:
            return []
        img_prep = _nn1_preprocess(img, mode="binarize", invert=False)
        pred_mask = _nn1_infer_one(_nn1_model, img_prep, device)
        # Сохраняем маску во временный файл
        mask_path = f"/tmp/exports/{project_id}_nn1_mask.png"
        os.makedirs("/tmp/exports", exist_ok=True)
        cv2.imwrite(mask_path, pred_mask)
        # Извлекаем геометрию с классификацией комнат
        geom = _nn1_extract_geometry(mask_path, image_path=image_path)
        rooms_out = []
        for i, r in enumerate(geom.get("rooms", [])):
            contour = r.get("contour")
            poly_px = contour.reshape(-1, 2).tolist() if contour is not None else []
            # Центроид Грина
            _cx, _cy = 0.0, 0.0
            if len(poly_px) >= 3:
                import math as _m
                _xs = [p[0] for p in poly_px];
                _ys = [p[1] for p in poly_px]
                _n = len(_xs);
                _area = _acx = _acy = 0.0
                for _j in range(_n):
                    _jj = (_j + 1) % _n
                    _cross = _xs[_j] * _ys[_jj] - _xs[_jj] * _ys[_j]
                    _area += _cross;
                    _acx += (_xs[_j] + _xs[_jj]) * _cross;
                    _acy += (_ys[_j] + _ys[_jj]) * _cross
                _area /= 2.0
                if abs(_area) > 1e-6:
                    _cx = _acx / (6 * _area);
                    _cy = _acy / (6 * _area)
                else:
                    _cx, _cy = sum(_xs) / len(_xs), sum(_ys) / len(_ys)
            rooms_out.append({
                "id": f"room_{i:03d}",
                "polygonPx": poly_px,
                "centroidPx": [_cx, _cy],
                "roomType": r.get("room_type", "bedroom"),
                "areaM2": r.get("area_m2", 10.0),
                "areaPx": r.get("area_px", 0),
                "isExterior": r.get("room_type") in {"living_room", "bedroom", "kitchen"},
            })
        logger.info("NN-1: %d комнат для project %s", len(rooms_out), project_id)
        return rooms_out
    except Exception as e:
        logger.warning("NN-1 extract_rooms failed: %s", e)
        return []


app = FastAPI(title="Low-Current Planner")
app.include_router(structure_router)
app.include_router(artifacts_router)
app.include_router(state_router)
app.include_router(compare_router, tags=["Comparison"])


LOCAL_RAW_DIR = os.getenv("LOCAL_DOWNLOAD_DIR", "/data")
LOCAL_DL_DIR = os.getenv("LOCAL_DOWNLOAD_DIR_INFER", "/tmp/downloads")
LOCAL_EXPORT_DIR = "/tmp/exports"
os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
os.makedirs(LOCAL_DL_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPORT_DIR, exist_ok=True)

# ---------------------------
# Logging / diagnostic mode
# ---------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "debug").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("planner")

# SQLite персистентная БД (дополняет in-memory DB для новых проектов)
try:
    from app.db import (
        init_db, save_project, get_project,
        save_rooms, get_rooms, save_design, get_design,
        save_room_map, get_room_map, save_export, list_projects
    )

    init_db()
    _SQLITE_OK = True
    logger.info("SQLite DB ready")
except Exception as _dbe:
    _SQLITE_OK = False
    logger.warning("SQLite unavailable: %s — using in-memory only", _dbe)


def _artifact_ttl_seconds() -> int:
    """TTL (seconds) for presigned artifact URLs returned by API."""
    try:
        v = int(os.getenv("ARTIFACT_URL_TTL", "3600"))
        return v if v > 0 else 3600
    except Exception:
        return 3600


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
        return key[len(bucket) + 1:]
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


def _ensure_png_path(local_path: str) -> str:
    """
    export_preview_png ожидает PNG. Если пришел JPG/JPEG — попробуем конвертировать.
    Если Pillow не установлен или конвертация не удалась — вернем исходный путь.
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
    export_formats: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("exportFormats", "export_formats")
    )


class PreferenceParseResponse(BaseModel):
    per_room_target_lux: Optional[Dict[str, float]] = None
    target_lux: Optional[float] = None
    total_fixtures_hint: Optional[int] = None
    fixture_efficacy_lm_per_w: Optional[float] = None
    notes: Optional[str] = None


def _parse_preferences(text: Optional[str], project_id: str = "unknown") -> dict:
    """Парсит пожелания через NN-2. Возвращает PreferencesGraph dict."""
    if not text:
        return {"version": "preferences-1.0", "projectId": project_id,
                "sourceText": "", "global": {}, "rooms": []}
    if _NN2_AVAILABLE:
        return nn2_parse_text(text, project_id=project_id)
    # fallback: пустой граф
    return {"version": "preferences-1.0", "projectId": project_id,
            "sourceText": text, "global": {}, "rooms": []}


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


@app.get("/health/s3")
def health_s3():
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
            warning = "S3_PUBLIC_ENDPOINT is not set; presigned URLs may use internal host (e.g., minio:9000) and may not open in browser. Set S3_PUBLIC_ENDPOINT=localhost:9000 for local dev."

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


def _make_numbered_plan(image_path: str, rooms: list, project_id: str) -> tuple:
    """
    Рисует поверх плана номера комнат (1, 2, 3...) в центре каждой.
    Возвращает (local_path, s3_uri, room_map) где room_map = {номер: room_id}
    """
    import cv2, numpy as np, os
    img = cv2.imread(image_path)
    if img is None:
        return None, None, {}

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    room_map = {}  # {1: "room_000", 2: "room_001", ...}
    colors = [
        (220, 80, 80), (80, 150, 220), (80, 200, 80),
        (200, 160, 40), (160, 80, 200), (40, 200, 200),
        (220, 120, 40), (120, 220, 120), (80, 80, 200),
        (200, 80, 160),
    ]

    for i, room in enumerate(rooms):
        num = i + 1
        room_id = room.get("id") or f"room_{i:03d}"
        room_map[num] = room_id
        poly = room.get("polygonPx") or []
        if len(poly) < 3:
            continue

        # Центроид
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        color = colors[i % len(colors)]
        label = str(num)

        # Полупрозрачная заливка полигона
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        mask = img.copy()
        cv2.fillPoly(mask, [pts], color)
        img = cv2.addWeighted(img, 0.75, mask, 0.25, 0)

        # Контур
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # Номер — белый текст с тёмной обводкой
        fs = max(0.8, min(2.0, (max(max(xs) - min(xs), max(ys) - min(ys))) / 200))
        thick = max(2, int(fs * 2.5))
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        tx, ty = cx - tw // 2, cy + th // 2
        # Обводка
        cv2.putText(img, label, (tx, ty), font, fs, (0, 0, 0), thick + 2, cv2.LINE_AA)
        # Текст
        cv2.putText(img, label, (tx, ty), font, fs, (255, 255, 255), thick, cv2.LINE_AA)

        # Площадь под номером
        area = room.get("areaM2") or 0
        if area:
            alabel = f"{area:.0f}m2"
            afs = fs * 0.55
            (aw, ah), _ = cv2.getTextSize(alabel, font, afs, 1)
            cv2.putText(img, alabel, (cx - aw // 2, cy + th + 5), font, afs, (30, 30, 30), 1, cv2.LINE_AA)

    out_path = f"/tmp/exports/{project_id}_numbered.png"
    os.makedirs("/tmp/exports", exist_ok=True)
    cv2.imwrite(out_path, img)

    # Загружаем в MinIO
    s3_key = f"previews/{project_id}_numbered.png"
    try:
        uri = upload_file(EXPORT_BUCKET, out_path, s3_key)
    except Exception:
        uri = None

    return out_path, uri, room_map


class UploadPlanRequest(BaseModel):
    projectId: str = "plan001"
    imageBase64: str  # PNG в base64
    srcKey: Optional[str] = None


@app.post("/upload", tags=["ingest"])
async def upload_plan(req: UploadPlanRequest):
    """
    Загружает PNG план (base64) в MinIO и запускает ingest.
    Использование:
      import base64
      data = base64.b64encode(open("plan.png","rb").read()).decode()
      POST /upload  {"projectId":"plan001","imageBase64": data}
    """
    import base64, tempfile
    try:
        img_bytes = base64.b64decode(req.imageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    key = req.srcKey or f"raw_plans/{req.projectId}_input.png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        upload_file(RAW_BUCKET, tmp_path, key)
        logger.info("Uploaded plan to MinIO: %s/%s", RAW_BUCKET, key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {e}")
    finally:
        os.unlink(tmp_path)

    ingest_req = APIIngestRequest(projectId=req.projectId, srcKey=key)
    result = await ingest(ingest_req)
    return {"uploaded": key, "ingest": result}


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

    # NN-1: запускаем сегментацию и нумеруем комнаты
    numbered_url = None
    room_map = {}
    nn1_rooms = []

    if low.endswith((".png", ".jpg", ".jpeg")):
        nn1_rooms = _nn1_get_rooms(local_path, req.project_id)
        if nn1_rooms:
            DB.setdefault("rooms", {})[req.project_id] = nn1_rooms
            _, numbered_url, room_map = _make_numbered_plan(local_path, nn1_rooms, req.project_id)
            DB.setdefault("room_map", {})[req.project_id] = room_map
            logger.info("Ingest room_map: %s", {k: v for k, v in room_map.items()})
            # Сохраняем в SQLite
            if _SQLITE_OK:
                try:
                    save_project(req.project_id, req.srcKey or "", local_path or "")
                    save_rooms(req.project_id, nn1_rooms)
                    save_room_map(req.project_id, room_map)
                except Exception as _se:
                    logger.debug("SQLite save failed: %s", _se)
            logger.info("Ingest: NN-1 нашла %d комнат, numbered plan создан", len(nn1_rooms))

    # Генерируем preview зон освещения
    try:
        from app.export_overlay_png import export_zones_preview
        zones_path = f"/tmp/exports/{req.project_id}_zones.png"
        export_zones_preview(local_path, nn1_rooms, zones_path)
        # Загружаем в MinIO
        _zones_uri = upload_file(EXPORT_BUCKET, zones_path, f"previews/{req.project_id}_zones.png")
        logger.info("Zones preview: %s", _zones_uri)
    except Exception as _ze:
        logger.debug("Zones preview failed: %s", _ze)

    # Строим краткое описание комнат для пользователя
    rooms_info = []
    for i, r in enumerate(nn1_rooms):
        rooms_info.append({
            "num": i + 1,
            "room_id": r.get("id"),
            "room_type": r.get("roomType", "?"),
            "area_m2": r.get("areaM2", 0),
        })

    return {
        "project_id": req.project_id,
        "rooms_found": len(nn1_rooms),
        "rooms": rooms_info,
        "numbered_plan": numbered_url,
        "hint": "Посмотри на numbered_plan и в /design укажи: \"1: свет датчик дыма; 2: телевизор свет; 3: свет\"",
        **stats,
    }


@app.post("/place")
async def place(req: APIPlaceRequest):
    cands = generate_candidates(req.project_id)
    chosen = select_devices(req.project_id, cands)
    DB["devices"][req.project_id] = _devices_to_json(
        chosen if isinstance(chosen, list) else DB["devices"].get(req.project_id, [])
    )
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
    try:
        routes = DB.get("routes", {}).get(req.project_id, [])

        # ── 1. Комнаты: объединяем геометрию из DB["rooms"] с типами из DesignGraph ──
        legacy_rooms = DB.get("rooms", {}).get(req.project_id, []) or []
        # Fallback из SQLite после рестарта
        if not legacy_rooms and _SQLITE_OK:
            legacy_rooms = get_rooms(req.project_id)
            if legacy_rooms:
                DB.setdefault("rooms", {})[req.project_id] = legacy_rooms
                logger.info("Rooms restored from SQLite for %s", req.project_id)
        design_graph = DB.get("design", {}).get(req.project_id)

        # Строим map room_id → roomType из DesignGraph.roomDesigns
        room_type_map: dict = {}
        if design_graph:
            for rd in design_graph.get("roomDesigns", []):
                rid = rd.get("roomId", "")
                rtype = rd.get("roomType", "bedroom")
                if rid:
                    room_type_map[rid] = rtype

        # Обогащаем legacy_rooms типами из DesignGraph
        # Сортируем по площади чтобы совпасть с порядком назначения типов в /design
        def _area_from_room(r):
            a = r.get("areaM2") or r.get("area")
            if a:
                return float(a)
            poly = r.get("polygonPx") or []
            if len(poly) >= 3:
                n = len(poly)
                area = 0
                for ii in range(n):
                    jj = (ii + 1) % n
                    area += poly[ii][0] * poly[jj][1]
                    area -= poly[jj][0] * poly[ii][1]
                return abs(area) / 2 * 0.000196
            return float(r.get("areaPx", 0)) * 0.000196

        MIN_AREA_M2 = 4.5
        MAX_AREA_M2 = 300.0
        ROOM_TYPES_CYCLE = [
            "living_room", "bedroom", "bedroom", "kitchen",
            "bathroom", "corridor", "toilet",
        ]

        # Фильтруем и сортируем как в /design
        filtered = sorted(
            [(r, _area_from_room(r)) for r in legacy_rooms
             if MIN_AREA_M2 <= _area_from_room(r) <= MAX_AREA_M2],
            key=lambda x: x[1], reverse=True
        )

        rooms = []
        for i, (r, area_m2) in enumerate(filtered):
            rid = r.get("id") or f"room_{i:03d}"
            rtype = room_type_map.get(rid) or ROOM_TYPES_CYCLE[i % len(ROOM_TYPES_CYCLE)]
            # Коррекция: если NN-1 классифицировала как kitchen но площадь < 10m² → скорее всего балкон/лоджия
            if rtype == "kitchen" and area_m2 < 10:
                rtype = "balcony"
            rooms.append({**r, "id": rid, "roomType": rtype, "areaM2": area_m2})

        # ── 2. Устройства из DesignGraph с room_id (без координат — export_pdf сам считает центроид) ──
        if design_graph and "devices" in design_graph:
            devices = []
            for d in design_graph["devices"]:
                dev_entry = {
                    "kind": d.get("kind", "UNKNOWN"),
                    "type": d.get("kind", "UNKNOWN"),  # legacy compat
                    "label": d.get("label", ""),
                    "roomRef": d.get("roomRef", ""),
                    "room_id": d.get("roomRef", ""),  # legacy compat
                    "id": d.get("id", ""),
                    "reason": d.get("reason", ""),
                }
                # Передаём координаты если есть (для PNG overlay)
                if d.get("xPx") is not None:
                    dev_entry["xPx"] = d["xPx"]
                    dev_entry["yPx"] = d["yPx"]
                devices.append(dev_entry)
        else:
            devices = _devices_to_json(DB.get("devices", {}).get(req.project_id, []))

        DB.setdefault("devices", {})[req.project_id] = devices

        out_paths: List[str] = []
        uploaded: List[str] = []
        keys: List[str] = []

        # Источник для overlay PNG
        _src_info = DB.get("source", {}).get(req.project_id) or {}
        local_src_png: Optional[str] = None
        # 1. Кэш — local_path сохранён при /ingest
        _lp = _src_info.get("local_path")
        if _lp and osp.exists(_lp):
            local_src_png = _ensure_png_path(_lp)
        # 2. Fallback — numbered plan уже лежит в /tmp/exports
        if not local_src_png:
            _numbered = f"/tmp/exports/{req.project_id}_numbered.png"
            if osp.exists(_numbered):
                local_src_png = _numbered
        # 3. Fallback — скачать из MinIO
        if not local_src_png:
            src_key = req.src_s3_key if hasattr(req, "src_s3_key") else None
            src_key = src_key or _src_info.get("src_key")
            if src_key:
                try:
                    local_src_png = _ensure_png_path(_download_from_raw(src_key))
                except Exception:
                    local_src_png = None
        logger.info("Export PNG: base image = %s", local_src_png or "NONE → canvas fallback")

        if "DXF" in req.formats:
            dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
            export_dxf(req.project_id, rooms, devices, routes, dxf_path)
            out_paths.append(dxf_path)

        if "PDF" in req.formats:
            pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
            export_pdf(req.project_id, rooms, devices, routes, pdf_path)
            out_paths.append(pdf_path)

        if "PNG" in req.formats:
            try:
                png_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}_overlay.png"
                # Используем DB["rooms"] — там есть polygonPx от NN-1
                db_rooms = DB.get("rooms", {}).get(req.project_id, []) or []
                rooms_with_poly = db_rooms if db_rooms else rooms
                _with_poly_count = sum(
                    1 for r in rooms_with_poly if r.get("polygonPx") and len(r.get("polygonPx", [])) >= 3)
                logger.info("PNG export: %d rooms, %d with polygonPx, base=%s",
                            len(rooms_with_poly), _with_poly_count, local_src_png or "None")
                if local_src_png and osp.exists(local_src_png):
                    export_overlay_png(local_src_png, rooms_with_poly, devices, routes, png_path)
                else:
                    # Нет фонового изображения — numbered plan как база
                    _numbered = f"/tmp/exports/{req.project_id}_numbered.png"
                    if osp.exists(_numbered):
                        export_overlay_png(_numbered, rooms_with_poly, devices, routes, png_path)
                    else:
                        export_preview_canvas_png(rooms=rooms_with_poly, devices=devices,
                                                  routes=routes, out_path=png_path)
                out_paths.append(png_path)
            except Exception as e:
                logger.warning("PNG export failed: %s", e, exc_info=True)

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

        ttl = _artifact_ttl_seconds()
        artifacts = build_artifacts_index(req.project_id, expires_seconds=ttl)

        return {
            "project_id": req.project_id,
            "files": out_paths,
            "uploaded": uploaded,
            "keys": keys,
            "artifacts": artifacts,
            "expiresSeconds": ttl,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("export failed project_id=%s", getattr(req, "project_id", None))
        raise HTTPException(status_code=500, detail=f"Export failed: {type(e).__name__}: {e}")


class PreferencesParseRequest(BaseModel):
    project_id: str = Field(default="unknown", validation_alias=AliasChoices("projectId", "project_id"))
    text: str


@app.post("/preferences/parse", tags=["preferences"])
async def preferences_parse(req: PreferencesParseRequest):
    """Парсит пожелания клиента через NN-2 → PreferencesGraph."""
    result = _parse_preferences(req.text, project_id=req.project_id)
    DB.setdefault("preferences", {})[req.project_id] = result
    return result


class DesignRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    preferences_text: Optional[str] = Field(default=None,
                                            validation_alias=AliasChoices("preferencesText", "preferences_text"))


@app.get("/projects", tags=["debug"])
async def list_all_projects():
    """Список всех проектов из SQLite."""
    if _SQLITE_OK:
        return {"projects": list_projects()}
    return {"projects": list(DB.get("rooms", {}).keys())}


@app.get("/debug/db/{project_id}", tags=["debug"])
async def debug_db(project_id: str):
    """Показывает что лежит в DB для project_id."""
    return {
        "has_rooms": bool(DB.get("rooms", {}).get(project_id)),
        "n_rooms": len(DB.get("rooms", {}).get(project_id) or []),
        "has_plan_graph": bool(DB.get("plan_graph", {}).get(project_id)),
        "has_preferences": bool(DB.get("preferences", {}).get(project_id)),
        "rooms_sample": (DB.get("rooms", {}).get(project_id) or [])[:2],
        "db_keys": list(DB.keys()),
    }


def _parse_numbered_preferences(text: str, room_map: dict) -> dict:
    """
    Парсит текст вида:
      "1: телевизор, свет; 2: свет, 2 ночника, телевизор; 3: свет, датчик дыма; 4: 2-4 источника света, интернет"

    Поддерживает:
      - "2 ночника"           → night_lights: 2
      - "2-4 источника света" → ceiling_lights: 3  (берём среднее)
      - "телевизор"           → tv_sockets: 1
      - запятые и пробелы как разделители внутри комнаты
      - ";" или "\n" как разделители между комнатами

    room_map: {1: "room_000", 2: "room_001", ...}
    """

    # Устройства: список (паттерн_regex, device_key)
    # Порядок важен — длинные паттерны раньше коротких
    DEVICE_PATTERNS = [
        (r"датчик\s*дыма", "smoke_detector"),
        (r"датчик\s*co2", "co2_detector"),
        (r"датчик\s*угарного", "co2_detector"),
        (r"углекислый", "co2_detector"),
        (r"co2", "co2_detector"),
        (r"газовый\s*датчик", "co2_detector"),
        (r"источник(?:а|ов)?\s*света", "ceiling_lights"),
        (r"светильник(?:а|ов)?", "ceiling_lights"),
        (r"люстр(?:а|ы)?", "ceiling_lights"),
        (r"лампоч(?:ка|ки|ек)?", "ceiling_lights"),
        (r"свет(?:овых|овые)?", "ceiling_lights"),
        # [DISABLED] (r"ночник(?:а|ов)?",             "night_lights"),
        (r"подсветк(?:а|и)?", "night_lights"),
        (r"розетк(?:а|и|у|ой)?", "power_socket"),
        (r"\bsocket\b", "power_socket"),
        # [DISABLED] (r"телевизор(?:а)?",             "tv_sockets"),
        (r"тв\b", "tv_sockets"),
        (r"\btv\b", "tv_sockets"),
        (r"интернет", "internet_sockets"),
        (r"роутер", "internet_sockets"),
        (r"\blan\b", "internet_sockets"),
        (r"вайфай", "internet_sockets"),
        (r"\bwifi\b", "internet_sockets"),
        (r"дым\b", "smoke_detector"),
        (r"пожарн", "smoke_detector"),
    ]

    def _parse_count(token: str) -> int:
        """Извлекает число из токена: '2', '2-4' → среднее=3, 'два'=2 и т.д."""
        WORDS = {"один": 1, "одна": 1, "одного": 1, "два": 2, "две": 2, "трёх": 3, "три": 3,
                 "четыре": 4, "четырёх": 4, "пять": 5}
        t = token.strip().lower()
        # диапазон "2-4"
        m = import_re.match(r"(\d+)\s*[-–—]\s*(\d+)", t)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return max(1, round((a + b) / 2))
        # просто число
        m = import_re.match(r"(\d+)", t)
        if m:
            return max(1, int(m.group(1)))
        # слово
        for w, n in WORDS.items():
            if w in t:
                return n
        return 1

    def _parse_room_segment(seg: str) -> dict:
        """Парсит строку одной комнаты → {device: count}"""
        result = {}
        seg_lo = seg.lower()

        for pattern, device in DEVICE_PATTERNS:
            # Ищем все вхождения паттерна
            for m in import_re.finditer(pattern, seg_lo):
                start = m.start()
                # Смотрим что стоит ПЕРЕД паттерном (число/диапазон)
                prefix = seg_lo[max(0, start - 12):start].strip()
                # Убираем запятые и лишнее
                prefix = import_re.sub(r"[,;]", " ", prefix).strip()
                # Берём последний токен перед паттерном
                tokens = prefix.split()
                count = 1
                if tokens:
                    count = _parse_count(tokens[-1])

                result[device] = max(result.get(device, 0), count)

        return result

    # Разбиваем на сегменты по ";" или "\n"
    segments = [s.strip() for s in import_re.split(r"[;\n]", text) if s.strip()]
    rooms_prefs = {}

    for seg in segments:
        # Ищем номер комнаты в начале: "1:", "комната 2:", "1 -"
        m = import_re.match(
            r"(?:комната\s*)?(\d+)\s*[:\-–—]?\s*(.*)",
            seg.strip(), import_re.IGNORECASE | import_re.DOTALL
        )
        if not m:
            continue
        num = int(m.group(1))
        room_body = m.group(2).strip()
        room_id = room_map.get(num)
        if not room_id or not room_body:
            continue

        devs = _parse_room_segment(room_body)
        if devs:
            rooms_prefs[room_id] = devs

    rooms_list = [{"roomId": rid, "devices": devs} for rid, devs in rooms_prefs.items()]

    return {
        "version": "preferences-1.0",
        "sourceText": text,
        "global": {},
        "rooms": rooms_list,
        "_by_room_id": rooms_prefs,
    }


import re as import_re


def _point_in_polygon(px: float, py: float, poly: list) -> bool:
    """Ray casting — точка внутри полигона."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _svt_grid_positions(poly: list, area_m2: float, room_type: str = "living_room") -> list:
    """
    Возвращает список (px, py) центров зон освещения.
    Использует те же зоны что и export_zones_preview — единый алгоритм.
    """
    if not poly or len(poly) < 3:
        return []
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0

    # Количество по норме
    if room_type in ("bathroom", "toilet", "balcony"):
        needed = 1
    elif room_type == "corridor":
        needed = max(1, min(4, round(area_m2 / 8.0)))
    elif room_type == "kitchen":
        needed = max(1, min(4, round(area_m2 / 10.0)))
    else:
        needed = max(1, min(8, round(area_m2 / 16.0)))

    # Сетка cols×rows
    aspect = w / max(1.0, h)
    if needed == 1:
        cols, rows = 1, 1
    elif needed == 2:
        cols, rows = (2, 1) if aspect >= 1.0 else (1, 2)
    elif needed <= 4:
        cols, rows = 2, 2
    elif needed <= 6:
        cols, rows = (3, 2) if aspect >= 1.0 else (2, 3)
    else:
        cols, rows = (3, 3) if needed <= 9 else (4, 3)

    # Используем те же зоны что рисуются в zones preview
    try:
        from app.export_overlay_png import _build_lighting_zones
        zones = _build_lighting_zones(poly, area_m2, room_type)
        positions = [z["center"] for z in zones]
        if positions:
            return positions
    except Exception:
        pass

    # Fallback — центр комнаты
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [(int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))]


@app.post("/design", tags=["design"])
async def design(req: DesignRequest):
    """
    NN-2 + NN-3 pipeline:
      1. Берём PlanGraph из DB (должен быть после /ingest или /structure)
      2. NN-2 парсит пожелания → PreferencesGraph
      3. NN-3 размещает устройства → DesignGraph
      4. Сохраняем в DB
    """
    project_id = req.project_id

    # Берём PlanGraph из DB
    # Приоритет: plan_graph (новый формат) → NN-1 → rooms (старый /ingest формат)
    plan_graph = DB.get("plan_graph", {}).get(project_id)

    # NN-1: если нет plan_graph — запускаем NN-1 для получения комнат с типами
    if not plan_graph:
        src_key_for_nn1 = (DB.get("source", {}).get(project_id) or {}).get("src_key")
        local_path_for_nn1 = (DB.get("source_meta", {}).get(project_id) or {}).get("localPath")
        if not local_path_for_nn1 and src_key_for_nn1:
            try:
                local_path_for_nn1 = _download_from_raw(src_key_for_nn1)
            except Exception:
                local_path_for_nn1 = None
        if local_path_for_nn1:
            nn1_rooms = _nn1_get_rooms(local_path_for_nn1, project_id)
            if nn1_rooms:
                # Сохраняем в DB и строим plan_graph
                DB.setdefault("rooms", {})[project_id] = nn1_rooms
                plan_graph = {
                    "projectId": project_id,
                    "rooms": nn1_rooms,
                    "openings": [],
                    "topology": {"roomAdjacency": []},
                }
                DB.setdefault("plan_graph", {})[project_id] = plan_graph
                logger.info("NN-1: построен plan_graph с %d комнатами", len(nn1_rooms))

    if not plan_graph:
        # Fallback: конвертируем старый DB["rooms"] в PlanGraph-совместимый dict
        rooms_legacy = DB.get("rooms", {}).get(project_id)
        if rooms_legacy:
            converted = []
            # Карта label → room_type (из geometry_png)
            LABEL_MAP = {
                "living_room": "living_room", "гостиная": "living_room",
                "bedroom": "bedroom", "спальня": "bedroom",
                "kitchen": "kitchen", "кухня": "kitchen",
                "bathroom": "bathroom", "ванная": "bathroom",
                "toilet": "toilet", "туалет": "toilet",
                "corridor": "corridor", "коридор": "corridor",
                "hall": "corridor", "прихожая": "corridor",
            }
            EXT_TYPES = {"living_room", "bedroom", "kitchen"}
            MIN_AREA_M2 = 3.0  # отфильтровываем артефакты < 3м²
            MAX_AREA_M2 = 200.0  # включаем даже большие полигоны
            ROOM_TYPES_CYCLE = [
                "living_room", "bedroom", "bedroom", "kitchen",
                "bathroom", "corridor", "toilet",
            ]

            def _poly_area_px(pts):
                """Площадь полигона по формуле Гаусса."""
                n = len(pts)
                area = 0
                for ii in range(n):
                    jj = (ii + 1) % n
                    area += pts[ii][0] * pts[jj][1]
                    area -= pts[jj][0] * pts[ii][1]
                return abs(area) / 2

            filtered = []
            for r in rooms_legacy:
                if not isinstance(r, dict):
                    continue
                # Считаем площадь
                area_m2 = r.get("areaM2") or r.get("area")
                if not area_m2:
                    poly = r.get("polygonPx") or []
                    if poly and len(poly) >= 3:
                        area_px = _poly_area_px(poly)
                    else:
                        area_px = r.get("areaPx") or 0
                    area_m2 = round(float(area_px) * 0.000196, 1)
                area_m2 = float(area_m2)
                # Фильтруем мусор
                if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
                    continue
                filtered.append((r, area_m2))

            # Сортируем по площади убыванию — большие комнаты вперёд
            filtered.sort(key=lambda x: x[1], reverse=True)

            room_idx = 0
            for r, area_m2 in filtered:
                label = str(r.get("label") or r.get("room_type") or r.get("roomType") or "").lower()
                if label in LABEL_MAP:
                    rtype = LABEL_MAP[label]
                else:
                    # Назначаем тип по порядку из типичной конфигурации
                    rtype = ROOM_TYPES_CYCLE[room_idx % len(ROOM_TYPES_CYCLE)]
                converted.append({
                    "id": r.get("id") or f"room_{room_idx}",
                    "roomType": rtype,
                    "areaM2": area_m2,
                    "isExterior": rtype in EXT_TYPES,
                })
                room_idx += 1
            plan_graph = {
                "projectId": project_id,
                "rooms": converted,
                "openings": [],
                "topology": {"roomAdjacency": []},
            }
            logger.info("design: используем legacy rooms (%d комнат) для project %s",
                        len(converted), project_id)

    if not plan_graph:
        # Детальная диагностика
        rooms_raw = DB.get("rooms", {}).get(project_id)
        logger.error(
            "design 400: plan_graph=%s, rooms_raw type=%s, len=%s, project=%s",
            DB.get("plan_graph", {}).get(project_id),
            type(rooms_raw).__name__,
            len(rooms_raw) if rooms_raw else 0,
            project_id,
        )
        raise HTTPException(
            status_code=400,
            detail=f"PlanGraph не найден. rooms_raw={bool(rooms_raw)}, n={len(rooms_raw) if rooms_raw else 0}. Сначала вызови /ingest или /structure/detect."
        )

    # NN-2: парсим пожелания
    prefs_graph = DB.get("preferences", {}).get(project_id)
    if req.preferences_text:
        room_map = DB.get("room_map", {}).get(project_id, {})
        # Если есть номера комнат в тексте и room_map — используем прямой парсер
        has_numbers = bool(room_map and import_re.search(r"\d+\s*[:–-]", req.preferences_text))
        if has_numbers:
            prefs_graph = _parse_numbered_preferences(req.preferences_text, room_map)
            logger.info("NN-2: numbered preferences parsed, %d rooms", len(prefs_graph.get("rooms", [])))
        else:
            prefs_graph = _parse_preferences(req.preferences_text, project_id=project_id)
            logger.info("NN-2: preferences parsed for project %s", project_id)
        DB.setdefault("preferences", {})[project_id] = prefs_graph

    # NN-3: размещаем устройства
    if not _NN3_AVAILABLE:
        raise HTTPException(status_code=503, detail="NN-3 недоступна — модель не загружена")

    design_graph = nn3_run_placement(
        plan_graph=plan_graph if isinstance(plan_graph, dict) else plan_graph.dict(),
        prefs_graph=prefs_graph,
        project_id=project_id,
    )

    # Применяем жёсткие правила + пожелания по номерам комнат
    by_room_id = (prefs_graph or {}).get("_by_room_id", {})
    # Передаём rooms из DB чтобы _apply_hard_rules знал площади комнат
    _rooms_for_rules = DB.get("rooms", {}).get(project_id, [])
    design_graph = _apply_hard_rules(design_graph, forced_devices=by_room_id, rooms=_rooms_for_rules)

    # Сохраняем в DB
    DB.setdefault("design", {})[project_id] = design_graph
    if _SQLITE_OK:
        try:
            save_design(project_id, design_graph)
        except Exception as _se:
            logger.debug("SQLite design save: %s", _se)
    DB.setdefault("devices", {})[project_id] = design_graph.get("devices", [])

    logger.info(
        "NN-3: placed %d devices for project %s",
        design_graph.get("totalDevices", 0), project_id
    )

    return design_graph


@app.post("/design_nn3", tags=["design"])
async def design_nn3(req: Dict[str, Any]):
    """
    ЧИСТЫЙ NN-3:
    - без placement.py
    - без _apply_hard_rules
    - без override

    Только:
    NN-1 → NN-2 → NN-3 → результат
    """

    project_id = req.get("projectId")
    prefs_text = req.get("preferencesText", "")

    if not project_id:
        raise HTTPException(400, "projectId required")

    # ─────────────────────────────────────────────
    # 1. PLAN GRAPH (как в /design)
    # ─────────────────────────────────────────────
    plan_graph = DB.get("plan_graph", {}).get(project_id)

    if not plan_graph:
        rooms = DB.get("rooms", {}).get(project_id)
        if not rooms:
            raise HTTPException(404, f"Plan not found: {project_id}")

        # fallback как в /design
        converted = []
        for i, r in enumerate(rooms):
            converted.append({
                "id": r.get("id") or f"room_{i}",
                "roomType": r.get("roomType") or r.get("room_type") or "living_room",
                "areaM2": r.get("areaM2") or r.get("area") or 10.0,
                "isExterior": r.get("isExterior", True),
            })

        plan_graph = {
            "projectId": project_id,
            "rooms": converted,
            "openings": [],
            "topology": {"roomAdjacency": []},
        }

    # ─────────────────────────────────────────────
    # 2. NN-2 (preferences)
    # ─────────────────────────────────────────────
    prefs_graph = DB.get("preferences", {}).get(project_id)

    if prefs_text:
        from app.main import _parse_preferences
        prefs_graph = _parse_preferences(prefs_text, project_id=project_id)
        DB.setdefault("preferences", {})[project_id] = prefs_graph

    # ─────────────────────────────────────────────
    # 3. NN-3 (ГЛАВНОЕ)
    # ─────────────────────────────────────────────
    design_graph = nn3_run_placement(
        plan_graph=plan_graph,
        prefs_graph=prefs_graph,
        project_id=project_id,
    )

    # КРИТИЧНО: Передаём rooms для проверки координат и санузлов
    rooms_list = []
    if "rooms" in DB and project_id in DB["rooms"]:
        rooms_list = DB["rooms"][project_id]

    design_graph = _validate_nn3_output(design_graph, rooms=rooms_list)

    # ❌ НЕТ ЭТОГО:
    # design_graph = _apply_hard_rules(...)
    # placement.py не используется

    # ─────────────────────────────────────────────
    # 4. Сохраняем отдельно (ВАЖНО!)
    # ─────────────────────────────────────────────
    DB.setdefault("design_nn3", {})[project_id] = design_graph

    # (опционально для compare через SQLite)
    try:
        from app.db import save_design
        save_design(f"{project_id}_nn3", design_graph)
    except Exception:
        pass

    return {
        "project_id": project_id,
        "mode": "NN-3 only (no postprocessing)",
        "totalDevices": design_graph.get("totalDevices", 0),
        "devices": design_graph.get("devices", []),
    }


def _validate_nn3_output(design_graph: dict, rooms: list = None) -> dict:
    """
    Улучшенная валидация NN-3 без замены устройств.

    Что делает:
    - Удаляет временно отключённые устройства (tv_sockets, night_lights)
    - Дедупликация устройств по координатам
    - Санузел: max 1 SVT (оставляем ближайший к центру)
    - Координаты внутри bbox полигона
    - Max 1 DYM на комнату

    Что НЕ делает:
    - НЕ заменяет SVT на zone-grid
    - НЕ добавляет RZT по нормативу
    - НЕ добавляет SWI у дверей
    """
    devices = design_graph.get("devices", [])
    room_designs = design_graph.get("roomDesigns", [])

    # Константы
    DISABLED_DEVICES = {"tv_sockets", "night_lights"}  # Временно отключены
    BATHROOM_TYPES = {"bathroom", "toilet", "balcony"}

    # Карта типов комнат
    room_type_map = {rd["roomId"]: rd.get("roomType", "bedroom") for rd in room_designs}

    # Карта полигонов комнат (для проверки координат)
    room_polygons = {}
    if rooms:
        for r in rooms:
            if not isinstance(r, dict):
                continue
            _rid = r.get("id") or r.get("roomId") or r.get("room_id") or ""
            if isinstance(_rid, int):
                _rid = f"room_{_rid:03d}"
            _poly = r.get("polygonPx") or []
            if _poly and len(_poly) >= 3:
                room_polygons[_rid] = _poly

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 1: Удаляем временно отключённые устройства
    # ══════════════════════════════════════════════════════════════════════════
    filtered = []
    removed_disabled = 0

    for d in devices:
        kind = d.get("kind", "")
        if kind in DISABLED_DEVICES:
            removed_disabled += 1
            continue
        filtered.append(d)

    devices = filtered

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 2: Дедупликация по ID
    # ══════════════════════════════════════════════════════════════════════════
    seen_ids = set()
    deduped = []
    removed_duplicate_ids = 0

    for d in devices:
        dev_id = d.get("id", "")
        if dev_id in seen_ids:
            removed_duplicate_ids += 1
            continue
        seen_ids.add(dev_id)
        deduped.append(d)

    devices = deduped

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 3: Дедупликация по координатам (одинаковые позиции)
    # ══════════════════════════════════════════════════════════════════════════
    position_keys = set()
    final_devices = []
    removed_duplicate_positions = 0

    for d in devices:
        room_id = d.get("roomRef") or d.get("room_id", "")
        kind = d.get("kind", "")
        x = d.get("xPx", 0)
        y = d.get("yPx", 0)

        # Ключ: room + kind + координаты (округлённые до 5px)
        # Это позволяет убрать устройства в одной точке
        x_rounded = round(x / 5) * 5
        y_rounded = round(y / 5) * 5
        key = f"{room_id}_{kind}_{x_rounded}_{y_rounded}"

        if key in position_keys:
            removed_duplicate_positions += 1
            continue

        position_keys.add(key)
        final_devices.append(d)

    devices = final_devices

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 4: Max 1 DYM на комнату
    # ══════════════════════════════════════════════════════════════════════════
    dym_seen = set()
    deduped_dym = []
    removed_dym = 0

    for d in devices:
        if d.get("kind") == "smoke_detector":
            rid = d.get("roomRef") or d.get("room_id", "")
            if rid in dym_seen:
                removed_dym += 1
                continue
            dym_seen.add(rid)
        deduped_dym.append(d)

    devices = deduped_dym

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 5: Санузел max 1 SVT
    # ══════════════════════════════════════════════════════════════════════════
    removed_bathroom_svt = 0

    for room_id, rtype in room_type_map.items():
        if rtype not in BATHROOM_TYPES:
            continue

        # Находим все SVT в этой комнате
        svt_list = [d for d in devices
                    if d.get("kind") == "ceiling_lights"
                    and (d.get("roomRef") == room_id or d.get("room_id") == room_id)]

        if len(svt_list) <= 1:
            continue

        # Вычисляем центр комнаты
        cx, cy = 0, 0
        if room_id in room_polygons:
            poly = room_polygons[room_id]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
        else:
            # Fallback: среднее по всем SVT
            if svt_list:
                cx = sum(s.get("xPx", 0) for s in svt_list) / len(svt_list)
                cy = sum(s.get("yPx", 0) for s in svt_list) / len(svt_list)

        # Сортируем по расстоянию до центра
        def _dist(d):
            dx = d.get("xPx", 0) - cx
            dy = d.get("yPx", 0) - cy
            return dx * dx + dy * dy

        svt_sorted = sorted(svt_list, key=_dist)
        keep = svt_sorted[0]  # ближайший к центру

        # Удаляем остальные
        devices = [d for d in devices
                   if not (d.get("kind") == "ceiling_lights"
                           and (d.get("roomRef") == room_id or d.get("room_id") == room_id)
                           and d["id"] != keep["id"])]

        removed_bathroom_svt += len(svt_list) - 1

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 6: Проверка координат внутри bbox комнаты
    # ══════════════════════════════════════════════════════════════════════════
    corrected_positions = 0

    for d in devices:
        room_id = d.get("roomRef") or d.get("room_id", "")
        if room_id not in room_polygons:
            continue

        poly = room_polygons[room_id]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x = d.get("xPx", 0)
        y = d.get("yPx", 0)

        # Clamp внутри bbox с небольшим margin
        margin = 10
        x_clamped = max(x_min + margin, min(x_max - margin, x))
        y_clamped = max(y_min + margin, min(y_max - margin, y))

        if x != x_clamped or y != y_clamped:
            d["xPx"] = int(x_clamped)
            d["yPx"] = int(y_clamped)
            corrected_positions += 1

    # ══════════════════════════════════════════════════════════════════════════
    # ШАГ 7: Пересчитываем deviceIds в roomDesigns
    # ══════════════════════════════════════════════════════════════════════════
    dev_ids_by_room = {}
    for d in devices:
        rid = d.get("roomRef") or d.get("room_id", "")
        dev_ids_by_room.setdefault(rid, []).append(d["id"])

    new_room_designs = []
    for rd in room_designs:
        rid = rd["roomId"]
        new_room_designs.append({
            **rd,
            "deviceIds": dev_ids_by_room.get(rid, []),
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Логирование статистики
    # ══════════════════════════════════════════════════════════════════════════
    total_removed = (removed_disabled + removed_duplicate_ids +
                     removed_duplicate_positions + removed_dym + removed_bathroom_svt)

    logger.info(
        "NN-3 validation: removed %d devices "
        "(disabled: %d, dup_ids: %d, dup_pos: %d, dym: %d, bathroom_svt: %d), "
        "corrected %d positions",
        total_removed, removed_disabled, removed_duplicate_ids,
        removed_duplicate_positions, removed_dym, removed_bathroom_svt,
        corrected_positions
    )

    return {
        **design_graph,
        "devices": devices,
        "roomDesigns": new_room_designs,
        "totalDevices": len(devices),
        "explain": design_graph.get("explain", []) + [
            f"Валидация NN-3: удалено {total_removed} некорректных устройств"
        ]
    }

@app.post("/infer")
async def infer(req: APIInferRequest):
    try:
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

        # NN-2: сохраняем PreferencesGraph в DB
        if req.preferences_text:
            prefs = _parse_preferences(req.preferences_text, project_id=req.project_id)
            DB.setdefault("preferences", {})[req.project_id] = prefs
            logger.info("preferences parsed: %d rooms, global=%s",
                        len(prefs.get("rooms", [])), prefs.get("global", {}))

        if not DB.get("rooms", {}).get(req.project_id):
            raise HTTPException(
                status_code=400,
                detail="Нет геометрии помещений. Передай srcKey в /infer или сначала вызови /ingest.",
            )

        # placement + routing
        cands = generate_candidates(req.project_id)
        chosen = select_devices(req.project_id, cands)
        DB["devices"][req.project_id] = _devices_to_json(
            chosen if isinstance(chosen, list) else DB["devices"].get(req.project_id, [])
        )

        routes_raw = route_all(req.project_id)
        routes_json = _routes_to_json(routes_raw)
        DB["routes"][req.project_id] = routes_json

        violations = validate_project(req.project_id)
        bom = make_bom(req.project_id)

        wants = set([str(x).upper() for x in (req.export_formats or [])])
        exported_files: List[str] = []
        uploaded_uris: List[str] = []
        exported_keys: List[str] = []

        rooms_json = DB.get("rooms", {}).get(req.project_id, [])
        devices_json = DB.get("devices", {}).get(req.project_id, [])
        routes_json = DB.get("routes", {}).get(req.project_id, [])

        # Debug: логируем что имеем для экспорта
        _rooms_with_poly = [r for r in rooms_json if r.get("polygonPx") and len(r.get("polygonPx", [])) >= 3]
        logger.info("Export: %d rooms total, %d with polygonPx, %d devices",
                    len(rooms_json), len(_rooms_with_poly), len(devices_json))
        if not _rooms_with_poly and rooms_json:
            logger.warning("Export: rooms have no polygonPx! Keys: %s",
                           list(rooms_json[0].keys()) if rooms_json else "empty")

        # Determine source image path once
        # 1. Сначала берём local_path из DB (быстро, без MinIO)
        src_info = DB.get("source", {}).get(req.project_id) or {}
        local_src_png_path: Optional[str] = None
        _cached_local = src_info.get("local_path")
        if _cached_local and osp.exists(_cached_local):
            local_src_png_path = _ensure_png_path(_cached_local)

        # 2. Fallback — скачиваем из MinIO
        if not local_src_png_path:
            src_key = req.src_s3_key or src_info.get("src_key")
            if src_key:
                try:
                    local_src_path = _download_from_raw(src_key)
                    local_src_png_path = _ensure_png_path(local_src_path)
                except Exception:
                    local_src_png_path = None

        # --- PREVIEW PNG (ALWAYS) — используем overlay рендер (читает polygonPx) ---
        preview_key: Optional[str] = None
        try:
            preview_path = f"/tmp/preview_{req.project_id}.png"
            if local_src_png_path and osp.exists(local_src_png_path):
                export_overlay_png(local_src_png_path, rooms_json, devices_json, routes_json, preview_path)
            else:
                # Нет исходника — рисуем numbered plan как фон
                numbered_path = f"/tmp/exports/{req.project_id}_numbered.png"
                if osp.exists(numbered_path):
                    export_overlay_png(numbered_path, rooms_json, devices_json, routes_json, preview_path)
                else:
                    export_preview_canvas_png(rooms=rooms_json, devices=devices_json,
                                              routes=routes_json, out_path=preview_path)
            preview_key = f"exports/{req.project_id}/preview.png"
            exported_files.append(preview_path)
            uploaded_uris.append(upload_file(EXPORT_BUCKET, preview_path, preview_key))
            exported_keys.append(preview_key)
        except Exception as e:
            uploaded_uris.append(f"ERROR:preview:{e}")

        # --- DXF ---
        dxf_key: Optional[str] = None
        if "DXF" in wants:
            dxf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.dxf"
            export_dxf(req.project_id, rooms_json, devices_json, routes_json, dxf_path)
            exported_files.append(dxf_path)
            try:
                dxf_key = f"drawings/{osp.basename(dxf_path)}"
                uploaded_uris.append(upload_file(EXPORT_BUCKET, dxf_path, dxf_key))
                exported_keys.append(dxf_key)
            except Exception as e:
                uploaded_uris.append(f"ERROR:dxf:{e}")

        # --- PDF ---
        pdf_key: Optional[str] = None
        if "PDF" in wants:
            pdf_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.pdf"
            export_pdf(req.project_id, rooms_json, devices_json, routes_json, pdf_path)
            exported_files.append(pdf_path)
            try:
                pdf_key = f"drawings/{osp.basename(pdf_path)}"
                uploaded_uris.append(upload_file(EXPORT_BUCKET, pdf_path, pdf_key))
                exported_keys.append(pdf_key)
            except Exception as e:
                uploaded_uris.append(f"ERROR:pdf:{e}")

        # --- PNG overlay (only if requested) ---
        overlay_key: Optional[str] = None
        if "PNG" in wants:
            try:
                overlay_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}_overlay.png"
                # Выбираем лучший фоновый PNG: оригинал > numbered > canvas
                _base_for_overlay = None
                if local_src_png_path and osp.exists(local_src_png_path):
                    _base_for_overlay = local_src_png_path
                else:
                    _numbered = f"/tmp/exports/{req.project_id}_numbered.png"
                    if osp.exists(_numbered):
                        _base_for_overlay = _numbered

                if _base_for_overlay:
                    export_overlay_png(_base_for_overlay, rooms_json, devices_json, routes_json, overlay_path)
                else:
                    export_preview_canvas_png(rooms=rooms_json, devices=devices_json,
                                              routes=routes_json, out_path=overlay_path)

                exported_files.append(overlay_path)
                overlay_key = f"overlays/{osp.basename(overlay_path)}"
                uploaded_uris.append(upload_file(EXPORT_BUCKET, overlay_path, overlay_key))
                exported_keys.append(overlay_key)
            except Exception as e:
                uploaded_uris.append(f"ERROR:overlay:{e}")

        # --- JSON result (ALWAYS) ---
        result_key = f"results/{req.project_id}.json"
        result_path = f"{LOCAL_EXPORT_DIR}/{req.project_id}.json"

        DB["exports"][req.project_id] = {
            "exported_files": exported_files,
            "uploaded_uris": uploaded_uris,
            "keys": exported_keys,
            "preview_key": preview_key,
            "dxf_key": dxf_key,
            "pdf_key": pdf_key,
            "overlay_key": overlay_key,
            "result_json_key": result_key,
        }

        artifacts_manifest = build_artifacts_manifest(req.project_id)

        result_payload = {
            "project_id": req.project_id,
            "rooms": rooms_json,
            "devices": devices_json,
            "routes": routes_json,
            "violations": violations,
            "bom": bom,
            "preview_key": preview_key,
            "overlay_key": overlay_key,
            "pdf_key": pdf_key,
            "dxf_key": dxf_key,
            "result_json_key": result_key,
            "export_keys": exported_keys,
            "artifacts": artifacts_manifest,
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2, default=_json_default)

        try:
            uploaded_uris.append(upload_file(EXPORT_BUCKET, result_path, result_key))
            exported_keys.append(result_key)
        except Exception as e:
            uploaded_uris.append(f"ERROR:result_json:{e}")

        DB["exports"][req.project_id].update(
            {
                "uploaded_uris": uploaded_uris,
                "keys": exported_keys,
                "result_json_key": result_key,
            }
        )

        ttl = _artifact_ttl_seconds()
        artifacts = build_artifacts_index(req.project_id, expires_seconds=ttl)

        return {
            "project_id": req.project_id,
            "rooms": rooms_json,
            "devices": devices_json,
            "routes": routes_json,
            "violations": violations,
            "bom": bom,
            "exports": DB.get("exports", {}).get(req.project_id, {}),
            "artifacts": artifacts,
            "expiresSeconds": ttl,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("infer failed project_id=%s", getattr(req, "project_id", None))
        raise HTTPException(status_code=500, detail=f"Infer failed: {type(e).__name__}: {e}")