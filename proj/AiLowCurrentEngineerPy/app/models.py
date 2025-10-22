from pydantic import BaseModel
from typing import List, Literal


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
