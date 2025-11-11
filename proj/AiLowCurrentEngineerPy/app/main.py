from fastapi import FastAPI

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
