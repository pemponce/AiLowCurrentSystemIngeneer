from typing import Optional, Dict, Literal, List

from fastapi import FastAPI
from pydantic import BaseModel

from app.lighting import design_lighting, LightingRequest, LightingResponse
from app.models import IngestRequest, PlaceRequest, RouteRequest, ExportRequest
from app.geometry import load_sample_rooms, DB
from app.placement import generate_candidates, select_devices
from app.routing import route_all
from app.export_dxf import export_dxf
from app.export_pdf import export_pdf
from app.geometry_dxf import ingest_dxf
from app.validator import validate_project
from app.bom import make_bom
from app.minio_client import upload_file
import os

app = FastAPI(
    title="Low-Current Planner"
)


@app.post('/ingest')
async def ingest(req: IngestRequest):
    # Если путь указывает на локальный файл — читаем его; иначе fallback на sample
    path = f"/data/{os.path.basename(req.src_s3_key)}"
    if os.path.exists(path):
        stats = ingest_dxf(req.project_id, path)
        return {"project_id": req.project_id, **stats}
    else:
        rooms = load_sample_rooms(req.project_id)
        return {"project_id": req.project_id, "rooms": len(rooms), "note": "fallback to sample geojson"}


@app.post("/lighting", response_model=LightingResponse)
async def lighting(req: LightingRequest) -> LightingResponse:
    """
    Расчёт схемы освещения:
    - распределяет заданное количество светильников по комнатам,
    - рассчитывает поток и мощность,
    - возвращает координаты размещения.
    """
    return design_lighting(req)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post('/place')
async def place(req: PlaceRequest):
    cands = generate_candidates(req.project_id)
    chosen = select_devices(req.project_id, cands)
    violations = validate_project(req.project_id)
    return {"project_id": req.project_id, "devices": len(chosen), "violations": violations}


@app.post('/route')
async def route(req: RouteRequest):
    routes = route_all(req.project_id)
    total = sum(l for _, _, l in routes)
    bom = make_bom(req.project_id)
    return {"project_id": req.project_id, "routes": len(routes), "length_sum": total, "bom": bom}


@app.post('/export')
async def export(req: ExportRequest):
    rooms = DB['rooms'][req.project_id]
    devices = DB['devices'][req.project_id]
    routes = DB['routes'][req.project_id]
    os.makedirs('/tmp/exports', exist_ok=True)
    out = []
    if 'DXF' in req.formats:
        dxf_path = f"/tmp/exports/{req.project_id}.dxf"
        export_dxf(req.project_id, rooms, devices, routes, dxf_path)
        out.append(dxf_path)
    if 'PDF' in req.formats:
        pdf_path = f"/tmp/exports/{req.project_id}.pdf"
        export_pdf(req.project_id, rooms, devices, routes, pdf_path)
        out.append(pdf_path)
    # Зальём в MinIO, если заданы переменные окружения бакетов
    bucket = os.getenv('S3_BUCKET_EXPORTS', 'exports')
    uploaded = []

    for p in out:
        key = f"drawings/{os.path.basename(p)}"
        try:
            uri = upload_file(bucket, p, key)
            uploaded.append(uri)
        except Exception as e:
            uploaded.append(f"ERROR:{e}")
    return {"project_id": req.project_id, "files": out, "uploaded": uploaded}


class PreferenceParseRequest(BaseModel):
    text: str
    # опционально, фиксированные единицы измерения/язык
    locale: Optional[str] = "ru"


class PreferenceParseResponse(BaseModel):
    # нормализованные цели освещенности (если конкретные комнаты названы в тексте)
    per_room_target_lux: Optional[Dict[str, float]] = None
    # общий target_lux, если нет явного деления по комнатам
    target_lux: Optional[float] = None
    # желаемое кол-во светильников (если указал пользователь)
    total_fixtures_hint: Optional[int] = None
    # эффективность (если написал “лампы 100 лм/Вт” и т.п.)
    fixture_efficacy_lm_per_w: Optional[float] = None
    # комментарии/несмогли распарсить
    notes: Optional[str] = None


class ProjectCreateRequest(BaseModel):
    project_id: str
    # имя файла, который ожидаем загрузить (для ссылки и проверки)
    filename: Optional[str] = None


class ProjectCreateResponse(BaseModel):
    project_id: str
    upload_kind: Literal["multipart", "s3"] = "multipart"
    # если когда-нибудь сделаем presigned URL — сюда положим
    upload_url: Optional[str] = None


class InferRequest(BaseModel):
    """
    Единая точка входа из UI: пользователь загрузил DXF/DWG и написал пожелания.
    """
    project_id: str
    user_preferences_text: str
    # fallback-значения, если из текста ничего не распарсили
    default_total_fixtures: int = 20
    default_target_lux: float = 300.0
    fixture_efficacy_lm_per_w: float = 110.0
    maintenance_factor: float = 0.8
    utilization_factor: float = 0.6
    export_formats: List[Literal['PDF', 'DXF', 'PNG']] = ['PDF']


class InferResponse(BaseModel):
    project_id: str
    parsed: PreferenceParseResponse
    lighting: LightingResponse
    exported_files: List[str] = []
    uploaded_uris: List[str] = []
