from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Dict
import re as regex

from app.geometry_png import ingest_png
from app.lighting import design_lighting, LightingRequest, LightingResponse
from app.models import IngestRequest, PlaceRequest, RouteRequest, ExportRequest, InferResponse, InferRequest
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
    # защитим структуру БД от KeyError
    DB.setdefault('rooms', {}).setdefault(req.project_id, [])
    DB.setdefault('devices', {}).setdefault(req.project_id, [])
    DB.setdefault('routes', {}).setdefault(req.project_id, [])
    DB.setdefault('candidates', {}).setdefault(req.project_id, [])

    path = f"/data/{os.path.basename(req.src_s3_key)}"

    if os.path.exists(path):
        low = path.lower()
        if low.endswith(('.dxf', '.dwg')):
            stats = ingest_dxf(req.project_id, path)
            return {"project_id": req.project_id, **stats}
        elif low.endswith(('.png', '.jpg', '.jpeg')):
            stats = ingest_png(req.project_id, path)     # <<< НОВОЕ
            return {"project_id": req.project_id, **stats}
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type")
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


from fastapi import HTTPException


@app.post('/infer', response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    DB.setdefault('rooms', {})
    DB.setdefault('devices', {})
    DB.setdefault('routes', {})
    DB.setdefault('candidates', {})

    if req.project_id not in DB['rooms'] or not DB['rooms'][req.project_id]:
        DB['rooms'][req.project_id] = load_sample_rooms(req.project_id)
        DB['devices'][req.project_id] = []
        DB['routes'][req.project_id] = []
        DB['candidates'][req.project_id] = []

    # 1) распарсим пожелания пользователя
    text = req.user_preferences_text or ''
    parsed = PreferenceParseResponse()

    # пример: вытащить target_lux из текста "300 люкс"
    m = regex.search(r'(\d{2,4})\s*люкс', text.lower())
    if m:
        parsed.target_lux = float(m.group(1))

    # fallback если не было в тексте
    target_lux = parsed.target_lux or req.default_target_lux
    total = req.default_total_fixtures

    # 2) lighting
    lighting_req = LightingRequest(
        project_id=req.project_id,
        total_fixtures=total,
        target_lux=target_lux,
        efficacy_lm_per_w=req.fixture_efficacy_lm_per_w,
        maintenance_factor=req.maintenance_factor,
        utilization_factor=req.utilization_factor,
    )
    lighting_res = design_lighting(lighting_req)

    # 3) place + route
    _ = generate_candidates(req.project_id)
    _ = select_devices(req.project_id, DB['candidates'][req.project_id])
    _ = validate_project(req.project_id)
    routes = route_all(req.project_id)

    # 4) export
    export_req = ExportRequest(project_id=req.project_id, formats=req.export_formats)
    exported = await export(export_req)  # если export у тебя sync — вызови синхронно

    return InferResponse(
        project_id=req.project_id,
        parsed=parsed,
        lighting=lighting_res,
        exported_files=exported.get('files', []),
        uploaded_uris=exported.get('uploaded', []),
    )


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
    # алиасы под вход от Java (camelCase)
    project_id: str = Field(alias='projectId')
    user_preferences_text: str = Field('', alias='preferencesText')

    default_total_fixtures: int = Field(20, alias='totalFixtures')
    default_target_lux: float = Field(300.0, alias='targetLux')
    fixture_efficacy_lm_per_w: float = Field(110.0, alias='efficacyLmPerW')
    maintenance_factor: float = Field(0.8, alias='maintenanceFactor')
    utilization_factor: float = Field(0.6, alias='utilizationFactor')

    export_formats: List[Literal['PDF', 'DXF', 'PNG']] = Field(default_factory=lambda: ['PDF'], alias='exportFormats')

    # позволяем наполнять поля по алиасам (camelCase)
    model_config = ConfigDict(populate_by_name=True)


class PreferenceParseResponse(BaseModel):
    per_room_target_lux: Optional[dict[str, float]] = None
    target_lux: Optional[float] = None
    total_fixtures_hint: Optional[int] = None
    fixture_efficacy_lm_per_w: Optional[float] = None
    notes: Optional[str] = None


class InferResponse(BaseModel):
    project_id: str
    parsed: PreferenceParseResponse
    lighting: "LightingResponse"
    exported_files: List[str] = []
    uploaded_uris: List[str] = []
