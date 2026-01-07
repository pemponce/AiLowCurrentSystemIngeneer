# app/api_state.py

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.geometry import DB

router = APIRouter(tags=["State"])


def _is_point(obj: Any) -> bool:
    return hasattr(obj, "x") and hasattr(obj, "y")


def _devices_to_json(devices: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in devices or []:
        if isinstance(d, dict):
            dd = dict(d)
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

        out.append({"type": "UNKNOWN", "label": "UNKNOWN", "raw": str(d)})
    return out


def _routes_to_json(routes: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in routes or []:
        if isinstance(r, dict):
            out.append(r)
            continue
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            r_type = str(r[0])
            line = r[1]
            length_m = float(r[2]) if len(r) >= 3 else 0.0
            pts = []
            if hasattr(line, "coords"):
                for x, y in list(line.coords):
                    pts.append({"x": float(x), "y": float(y)})
            out.append({"type": r_type, "length_m": length_m, "points": pts})
    return out


class ProjectStateResponse(BaseModel):
    projectId: str = Field(..., description="Project id")
    source: dict | None = Field(None, description="Source ingestion metadata")
    structure: dict | None = Field(None, description="Structure detection output")
    planGraph: dict | None = Field(None, description="Plan graph (walls/openings/rooms)")
    rooms: list[dict[str, Any]] = Field(default_factory=list)
    devices: list[dict[str, Any]] = Field(default_factory=list)
    routes: list[dict[str, Any]] = Field(default_factory=list)
    exports: dict = Field(default_factory=dict)


@router.get("/state/{project_id}", response_model=ProjectStateResponse)
def get_state(project_id: str):
    if (
        project_id not in DB.get("rooms", {})
        and project_id not in DB.get("source_meta", {})
        and project_id not in DB.get("source", {})
    ):
        raise HTTPException(status_code=404, detail=f"Unknown projectId: {project_id}")

    exports = DB.get("exports", {}).get(project_id, {})

    rooms = DB.get("rooms", {}).get(project_id, [])
    devices_raw = DB.get("devices", {}).get(project_id, [])
    routes_raw = DB.get("routes", {}).get(project_id, [])

    devices = _devices_to_json(devices_raw)
    routes = _routes_to_json(routes_raw) if (routes_raw and not isinstance(routes_raw[0], dict)) else (routes_raw or [])

    # Optionally persist normalized devices in DB to avoid repeated conversions
    DB.setdefault("devices", {})[project_id] = devices

    return ProjectStateResponse(
        projectId=project_id,
        source=DB.get("source", {}).get(project_id) or DB.get("source_meta", {}).get(project_id),
        structure=DB.get("structure", {}).get(project_id),
        planGraph=DB.get("plan_graph", {}).get(project_id),
        rooms=rooms,
        devices=devices,
        routes=routes,
        exports=exports,
    )
