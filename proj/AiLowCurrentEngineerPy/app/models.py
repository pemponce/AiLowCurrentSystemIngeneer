# app/models.py
from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# ----------------------- Уже существующие модели -----------------------

class IngestRequest(BaseModel):
    project_id: str
    src_s3_key: str  # путь к DXF в raw-plans (или имя файла в /data)


class PlaceRequest(BaseModel):
    project_id: str
    preferences: dict | None = None


class RouteRequest(BaseModel):
    project_id: str


class ExportRequest(BaseModel):
    project_id: str
    formats: List[Literal['PDF', 'DXF', 'PNG']] = ['PDF']


# ---------- Освещение: запрос/ответ (совместимо с твоим lighting.py) ----------

class LightingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "simple_apartment",
                "total_fixtures": 20,
                "target_lux": 300,
                "per_room_target_lux": {
                    "living": 200,
                    "bedroom": 120
                },
                "fixture_efficacy_lm_per_w": 110,
                "maintenance_factor": 0.8,
                "utilization_factor": 0.6
            }
        }
    )

    project_id: str = Field(description="ID проекта, тот же, что в /ingest")
    total_fixtures: int = Field(gt=0, description="Общее количество светильников по проекту")

    # общий target (если нет пер-комнатных)
    target_lux: Optional[float] = Field(
        default=300.0,
        description="Целевая освещённость, люкс (если per_room_target_lux не задан)"
    )

    # переопределения по комнатам: { 'living': 200, 'bedroom': 120 }
    per_room_target_lux: Optional[Dict[str, float]] = Field(
        default=None,
        description="Нормы по люксам для конкретных комнат (по name/типу)"
    )

    fixture_efficacy_lm_per_w: float = Field(
        default=110.0,
        description="Световая отдача источника, лм/Вт"
    )

    maintenance_factor: float = Field(
        default=0.8,
        description="Коэффициент запаса (MF)"
    )

    utilization_factor: float = Field(
        default=0.6,
        description="Коэффициент использования светового потока (UF)"
    )


class FixturePlacement(BaseModel):
    room_id: str
    x: float
    y: float
    lumens: float
    power_w: float


class RoomLightingResult(BaseModel):
    area_m2: float
    fixtures: int
    lumens_total: float
    lumens_per_fixture: float
    power_w_per_fixture: float
    # опционально — если в lighting.py ты добавил расчёт фактически используемого target_lux:
    # target_lux_used: Optional[float] = None


class RoomLightingSummary(BaseModel):
    room_id: str
    area_m2: float
    fixtures: int
    target_lux: float
    total_lumens: float
    total_watts: float


class LightingResponse(BaseModel):
    project_id: str
    total_fixtures: int
    target_lux: float
    rooms: Dict[str, RoomLightingResult]   # ВАЖНО: dict, как возвращает твой lighting.py
    fixtures: List[FixturePlacement] = []  # lighting.py теперь может наполнять координаты


# ----------------------- Парсер предпочтений / NLP -----------------------

class PreferenceParseRequest(BaseModel):
    text: str
    locale: Optional[str] = "ru"


class PreferenceParseResponse(BaseModel):
    per_room_target_lux: Optional[Dict[str, float]] = None
    target_lux: Optional[float] = None
    total_fixtures_hint: Optional[int] = None
    fixture_efficacy_lm_per_w: Optional[float] = None
    notes: Optional[str] = None


# ----------------------- Проекты / единый конвейер -----------------------

class ProjectCreateRequest(BaseModel):
    project_id: str
    filename: Optional[str] = None


class ProjectCreateResponse(BaseModel):
    project_id: str
    upload_kind: Literal["multipart","s3"] = "multipart"
    upload_url: Optional[str] = None


class InferRequest(BaseModel):
    """
    Единая точка входа из UI: пользователь загрузил DXF/DWG и написал пожелания.
    """
    project_id: str
    user_preferences_text: str
    default_total_fixtures: int = 20
    default_target_lux: float = 300.0
    fixture_efficacy_lm_per_w: float = 110.0
    maintenance_factor: float = 0.8
    utilization_factor: float = 0.6
    export_formats: List[Literal['PDF','DXF','PNG']] = ['PDF']


class InferResponse(BaseModel):
    project_id: str
    parsed: PreferenceParseResponse
    lighting: LightingResponse
    exported_files: List[str] = []
    uploaded_uris: List[str] = []
