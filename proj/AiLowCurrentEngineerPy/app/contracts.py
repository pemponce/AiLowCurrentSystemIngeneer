"""app/contracts.py

Контракты данных между модулями:
- PlanGraph: результат парсинга плана (PNG/DXF → структурная геометрия)
- Design: результат планирования (устройства + трассы)

Сейчас контракт фиксируем в формате v0.1.
Важно: координаты по умолчанию трактуем как "image"-координаты.
Для PNG это пиксели, для DXF — модельные координаты (CAD единицы),
но API и планировщик работают одинаково: это просто плоскость X-Y.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

Point2D = Tuple[float, float]


class PlanSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    srcKey: str
    imageWidth: int
    imageHeight: int
    dpi: Optional[float] = None
    scale: Optional[Dict[str, Any]] = None


class CoordinateSystem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: Literal["top_left", "unknown"] = "unknown"
    units: Dict[str, str] = Field(default_factory=lambda: {"image": "px", "world": "m"})


class Wall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    polyline: List[Point2D]
    thicknessPx: Optional[float] = None
    type: Literal["unknown", "bearing", "partition"] = "unknown"
    confidence: float = 0.7


class Opening(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["door", "window"]
    polygon: List[Point2D]
    center: Point2D
    widthPx: Optional[float] = None
    orientationDeg: Optional[float] = None
    wallRef: Optional[str] = None
    confidence: float = 0.7


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    polygon: List[Point2D]
    label: Optional[str] = None
    roomType: Optional[str] = None
    areaPx2: Optional[float] = None
    confidence: float = 0.7


class PlanElements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    walls: List[Wall] = Field(default_factory=list)
    openings: List[Opening] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)


class Topology(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roomAdjacency: List[Dict[str, str]] = Field(default_factory=list)
    exteriorWalls: List[str] = Field(default_factory=list)


class Artifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    previewOverlayPngKey: Optional[str] = None
    masks: Optional[Dict[str, Optional[str]]] = None


class PlanGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["plan-graph-0.1"] = "plan-graph-0.1"
    source: PlanSource
    coordinateSystem: CoordinateSystem
    elements: PlanElements
    topology: Optional[Topology] = None
    artifacts: Optional[Artifacts] = None


# ---------------- Design ----------------

class Device(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal[
        "ceiling_light",
        "switch",
        "socket",
        "smoke_sensor",
        "motion_sensor",
        "camera",
        "access_point",
        "router",
        "intercom_panel",
        "panel",
    ]
    roomRef: Optional[str] = None
    posPx: Point2D
    posM: Optional[Point2D] = None
    params: Optional[Dict[str, Any]] = None
    confidence: float = 0.7


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["power", "low_current", "lighting"] = "power"
    polylinePx: List[Point2D]
    constraints: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class Design(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["design-0.1"] = "design-0.1"
    devices: List[Device] = Field(default_factory=list)
    routes: List[Route] = Field(default_factory=list)
    explain: Optional[List[str]] = None
