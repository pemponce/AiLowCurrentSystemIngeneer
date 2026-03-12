"""
app/contracts.py

Единые контракты данных между тремя нейронными сетями.
Версия: 1.0

Поток данных:
    PNG/DXF → [NN-1] → PlanGraph → [NN-2] → PreferencesGraph
                                          ↘
                                    [NN-3] → DesignGraph → PDF/DXF/PNG

Правила контракта:
  - Все координаты в пикселях (image-space, origin top-left)
  - Реальные размеры через scale_mm_per_px
  - Каждый контракт имеет поле version — при изменении схемы версия растёт
  - Каждая NN валидирует вход через pydantic перед работой
  - confidence — уверенность модели [0.0 .. 1.0]
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────── базовые типы ────────────────────────────────────

Point2D    = Tuple[float, float]           # (x, y) в пикселях
PolygonPx  = List[Point2D]                 # замкнутый контур в пикселях
RoomType   = Literal[
    "living_room", "bedroom", "kitchen",
    "bathroom", "toilet", "corridor",
    "balcony", "storage", "unknown",
]
DeviceKind = Literal[
    # слаботочка
    "tv_socket", "internet_socket", "intercom_panel",
    "smoke_detector", "co2_detector", "motion_sensor", "alarm_siren",
    # освещение
    "ceiling_light", "night_light", "switch",
    # силовая (для совместимости)
    "socket", "panel",
    # сеть
    "access_point", "router",
    # камеры
    "camera",
]
RouteKind  = Literal["power", "low_current", "lighting", "network", "alarm"]
MountType  = Literal["wall", "ceiling", "floor"]


# ══════════════════════════════════════════════════════════════════════════════
#  БЛОК 1 — PlanGraph (выход NN-1)
# ══════════════════════════════════════════════════════════════════════════════

class PlanSource(BaseModel):
    """Метаданные источника плана."""
    model_config = ConfigDict(extra="forbid")

    srcKey:      str                     # S3-ключ или путь к файлу
    fileType:    Literal["png", "dxf", "pdf"] = "png"
    imageWidth:  int                     # px
    imageHeight: int                     # px
    dpi:         Optional[float] = None
    scaleMmPerPx: Optional[float] = None # масштаб: сколько мм в 1 пикселе
                                         # None = неизвестен (нужна калибровка)


class CoordinateSystem(BaseModel):
    """Система координат плана."""
    model_config = ConfigDict(extra="forbid")

    origin: Literal["top_left", "bottom_left", "unknown"] = "top_left"
    unitsPx: str = "px"
    unitsWorld: str = "mm"


class Wall(BaseModel):
    """Стена — отрезок или полилиния."""
    model_config = ConfigDict(extra="forbid")

    id:          str
    polyline:    List[Point2D]           # список точек стены
    thicknessPx: Optional[float] = None
    type:        Literal["unknown", "bearing", "partition"] = "unknown"
    confidence:  float = Field(default=0.7, ge=0.0, le=1.0)


class Opening(BaseModel):
    """Дверной или оконный проём."""
    model_config = ConfigDict(extra="forbid")

    id:             str
    kind:           Literal["door", "window", "front_door"]
    polygon:        PolygonPx            # контур проёма
    center:         Point2D
    widthPx:        Optional[float] = None
    orientationDeg: Optional[float] = None  # угол открытия двери
    wallRef:        Optional[str]  = None   # id стены которой принадлежит
    roomRefs:       List[str]      = Field(default_factory=list)  # id комнат по обе стороны
    confidence:     float          = Field(default=0.7, ge=0.0, le=1.0)


class Room(BaseModel):
    """
    Комната — замкнутый полигон с типом и метриками.
    Центральный узел графа для NN-3.
    """
    model_config = ConfigDict(extra="forbid")

    id:           str
    polygon:      PolygonPx
    roomType:     RoomType = "unknown"
    label:        Optional[str]  = None   # текстовая метка если была на плане
    areaPx2:      Optional[float] = None  # площадь в пикселях²
    areaM2:       Optional[float] = None  # площадь в м² (если известен масштаб)
    centroid:     Optional[Point2D] = None
    # связи — заполняет topology builder
    neighborIds:  List[str] = Field(default_factory=list)  # смежные комнаты
    openingIds:   List[str] = Field(default_factory=list)  # проёмы этой комнаты
    isExterior:   bool = False            # выходит на улицу (есть окна)
    confidence:   float = Field(default=0.7, ge=0.0, le=1.0)


class Topology(BaseModel):
    """
    Граф смежности комнат.
    Именно этот граф подаётся на вход NN-3 как GNN.
    """
    model_config = ConfigDict(extra="forbid")

    # список рёбер: {"from": "room_001", "to": "room_002", "via": "door_001"}
    roomAdjacency: List[Dict[str, str]] = Field(default_factory=list)
    exteriorRoomIds: List[str]          = Field(default_factory=list)
    entranceRoomId:  Optional[str]      = None  # комната с входной дверью


class PlanGraph(BaseModel):
    """
    Выход NN-1. Полное структурное описание плана.
    Передаётся в NN-2 (для контекста) и в NN-3 (как основа размещения).
    """
    model_config = ConfigDict(extra="forbid")

    version:          Literal["plan-graph-1.0"] = "plan-graph-1.0"
    projectId:        str
    source:           PlanSource
    coordinateSystem: CoordinateSystem = Field(default_factory=CoordinateSystem)
    walls:            List[Wall]       = Field(default_factory=list)
    openings:         List[Opening]    = Field(default_factory=list)
    rooms:            List[Room]       = Field(default_factory=list)
    topology:         Optional[Topology] = None
    # сырые артефакты NN-1
    nn1Confidence:    Optional[float]  = None
    maskS3Key:        Optional[str]    = None   # S3-ключ маски сегментации

    def room_by_id(self, room_id: str) -> Optional[Room]:
        for r in self.rooms:
            if r.id == room_id:
                return r
        return None

    def openings_for_room(self, room_id: str) -> List[Opening]:
        return [o for o in self.openings if room_id in o.roomRefs]


# ══════════════════════════════════════════════════════════════════════════════
#  БЛОК 2 — PreferencesGraph (выход NN-2)
# ══════════════════════════════════════════════════════════════════════════════

class RoomPreference(BaseModel):
    """Пожелания клиента для одной комнаты."""
    model_config = ConfigDict(extra="ignore")  # NN-2 может добавлять новые поля

    roomType:       RoomType
    roomId:         Optional[str]  = None  # если удалось сматчить с PlanGraph

    # устройства — желаемое количество (None = по умолчанию из rules.json)
    tvSockets:      Optional[int]  = None
    internetSockets: Optional[int] = None
    smokeDet:       Optional[bool] = None
    co2Det:         Optional[bool] = None
    ceilingLights:  Optional[int]  = None
    nightLights:    Optional[int]  = None
    motionSensors:  Optional[int]  = None
    cameras:        Optional[int]  = None

    # освещение
    targetLux:      Optional[float] = None

    # особые пожелания в свободной форме (для логов и объяснений)
    rawNotes:       Optional[str]   = None
    confidence:     float           = Field(default=0.8, ge=0.0, le=1.0)


class GlobalPreference(BaseModel):
    """Глобальные пожелания на весь проект."""
    model_config = ConfigDict(extra="ignore")

    style:          Optional[str]  = None  # "минимализм", "smart home", ...
    budget:         Optional[str]  = None  # "эконом", "стандарт", "премиум"
    smartHome:      bool           = False # нужна ли интеграция умного дома
    intercom:       bool           = True  # нужен ли домофон
    alarm:          bool           = False # нужна ли охранная сигнализация
    targetLux:      float          = 300.0 # глобальный дефолт освещённости


class PreferencesGraph(BaseModel):
    """
    Выход NN-2. Структурированные пожелания клиента.
    Передаётся в NN-3 вместе с PlanGraph.
    """
    model_config = ConfigDict(extra="forbid")

    version:        Literal["preferences-1.0"] = "preferences-1.0"
    projectId:      str
    sourceText:     Optional[str]          = None  # исходный текст клиента
    global_:        GlobalPreference       = Field(
                        default_factory=GlobalPreference,
                        alias="global"
                    )
    rooms:          List[RoomPreference]   = Field(default_factory=list)
    nn2Confidence:  Optional[float]        = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def for_room_type(self, room_type: RoomType) -> Optional[RoomPreference]:
        for r in self.rooms:
            if r.roomType == room_type:
                return r
        return None

    def for_room_id(self, room_id: str) -> Optional[RoomPreference]:
        for r in self.rooms:
            if r.roomId == room_id:
                return r
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  БЛОК 3 — DesignGraph (выход NN-3)
# ══════════════════════════════════════════════════════════════════════════════

class PlacedDevice(BaseModel):
    """Размещённое устройство на плане."""
    model_config = ConfigDict(extra="forbid")

    id:         str
    kind:       DeviceKind
    roomRef:    Optional[str]   = None   # id комнаты из PlanGraph
    posPx:      Point2D                  # позиция в пикселях
    posM:       Optional[Point2D] = None # позиция в метрах (если масштаб известен)
    mount:      MountType       = "wall"
    heightMm:   int             = 300    # высота монтажа от пола
    symbol:     Optional[str]   = None   # символ для чертежа ("SD", "TV", ...)
    label:      Optional[str]   = None   # человекочитаемое название
    params:     Optional[Dict[str, Any]] = None
    # объяснение почему модель поставила сюда (для XAI)
    reason:     Optional[str]   = None
    confidence: float           = Field(default=0.8, ge=0.0, le=1.0)


class CableRoute(BaseModel):
    """Трасса кабеля между устройствами или до щита."""
    model_config = ConfigDict(extra="forbid")

    id:          str
    kind:        RouteKind = "low_current"
    fromDeviceId: Optional[str] = None
    toDeviceId:   Optional[str] = None
    polylinePx:  List[Point2D]
    lengthM:     Optional[float] = None  # длина в метрах
    constraints: Optional[Dict[str, Any]] = None


class RoomDesign(BaseModel):
    """
    Результат проектирования одной комнаты.
    Промежуточная структура которую NN-3 строит для каждого узла графа.
    """
    model_config = ConfigDict(extra="forbid")

    roomId:      str
    roomType:    RoomType
    deviceIds:   List[str] = Field(default_factory=list)  # ссылки на PlacedDevice
    violations:  List[str] = Field(default_factory=list)  # нарушения ГОСТ/СП
    confidence:  float     = Field(default=0.8, ge=0.0, le=1.0)


class DesignGraph(BaseModel):
    """
    Выход NN-3. Полный проект размещения устройств и трасс.
    Передаётся в рендер (PDF/DXF/PNG).

    Содержит ссылки на входные данные чтобы рендер мог
    восстановить полный контекст без дополнительных запросов.
    """
    model_config = ConfigDict(extra="forbid")

    version:       Literal["design-1.0"] = "design-1.0"
    projectId:     str

    # ссылки на входные данные
    planGraphVersion:  str = "plan-graph-1.0"
    prefGraphVersion:  str = "preferences-1.0"

    # результат
    devices:       List[PlacedDevice] = Field(default_factory=list)
    routes:        List[CableRoute]   = Field(default_factory=list)
    roomDesigns:   List[RoomDesign]   = Field(default_factory=list)

    # метрики и объяснения
    totalDevices:  int            = 0
    totalCableM:   Optional[float] = None
    violations:    List[str]      = Field(default_factory=list)  # глобальные нарушения
    explain:       List[str]      = Field(default_factory=list)  # объяснения решений
    nn3Confidence: Optional[float] = None

    # утилиты
    def devices_in_room(self, room_id: str) -> List[PlacedDevice]:
        return [d for d in self.devices if d.roomRef == room_id]

    def devices_by_kind(self, kind: DeviceKind) -> List[PlacedDevice]:
        return [d for d in self.devices if d.kind == kind]


# ══════════════════════════════════════════════════════════════════════════════
#  БЛОК 4 — Pipeline контейнер (хранится в БД по project_id)
# ══════════════════════════════════════════════════════════════════════════════

class ProjectPipeline(BaseModel):
    """
    Полное состояние проекта — все три графа вместе.
    Именно это сохраняется в БД (SQLite/PostgreSQL).
    """
    model_config = ConfigDict(extra="forbid")

    projectId:   str
    status:      Literal[
        "created",       # проект создан
        "nn1_done",      # NN-1 отработала, PlanGraph готов
        "nn2_done",      # NN-2 отработала, PreferencesGraph готов
        "nn3_done",      # NN-3 отработала, DesignGraph готов
        "exported",      # PDF/DXF экспортированы
        "failed",        # ошибка на одном из этапов
    ] = "created"

    planGraph:        Optional[PlanGraph]        = None  # выход NN-1
    preferencesGraph: Optional[PreferencesGraph] = None  # выход NN-2
    designGraph:      Optional[DesignGraph]      = None  # выход NN-3

    errorMsg:    Optional[str] = None
    createdAt:   Optional[str] = None
    updatedAt:   Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  Обратная совместимость со старым кодом
# ══════════════════════════════════════════════════════════════════════════════

# старые имена → новые
PlanElements = None   # заменён на отдельные поля в PlanGraph
Artifacts    = None   # заменён на maskS3Key в PlanGraph
Device       = PlacedDevice
Route        = CableRoute
Design       = DesignGraph