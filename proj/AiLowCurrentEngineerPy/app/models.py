from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Literal


class IngestRequest(BaseModel):
    project_id: str
    src_s3_key: str  # путь к DXF в raw-plans


class PlaceRequest(BaseModel):
    project_id: str
    preferences: dict


class RouteRequest(BaseModel):
    project_id: str


class ExportRequest(BaseModel):
    project_id: str
    formats: List[Literal['PDF', 'DXF', 'PNG']] = ['PDF']


class LightingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "simple_apartment",
                "total_fixtures": 20,
                "target_lux": 300,
                "fixture_efficacy_lm_per_w": 110,
                "maintenance_factor": 0.8,
                "utilization_factor": 0.6
            }
        }
    )

    project_id: str = Field(description="ID проекта, тот же, что в /ingest")
    total_fixtures: int = Field(gt=0, description="Общее количество светильников по проекту")

    # общая цель по люксам (если не задано для комнат отдельно)
    target_lux: Optional[float] = Field(
        default=300.0,
        description="Целевая освещённость, люкс (если per_room_target_lux не задан)"
    )

    # опционально: разные нормы по комнатам: {room_id: lux}
    per_room_target_lux: Optional[Dict[str, float]] = Field(
        default=None,
        description="Нормы по люксам для конкретных комнат"
    )

    # принятое светосодержание светильника (Лм/Вт) — по умолчанию для LED
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
    z: float = 0.0

    lumens: float
    watts: float
    target_lux: float


class RoomLightingSummary(BaseModel):
    room_id: str
    area_m2: float
    fixtures: int
    target_lux: float
    total_lumens: float
    total_watts: float


class LightingResponse(BaseModel):
    project_id: str
    fixtures: List[FixturePlacement]
    rooms: List[RoomLightingSummary]
    assumed_voltage: float = Field(default=230.0, description="Принятое напряжение сети, В")
