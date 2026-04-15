# app/api_endpoints/export.py
"""POST /export - экспорт чертежей в PNG/PDF/DXF."""

import os
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict, AliasChoices
from typing import List, Literal

from app.export_overlay_png import export_overlay_png
from app.export_pdf import export_pdf
from app.export_dxf import export_dxf
from app.minio_client import upload_file, EXPORT_BUCKET

logger = logging.getLogger("planner")
router = APIRouter(tags=["export"])

DB = None

def set_db(db_instance):
    global DB
    DB = db_instance

class APIExportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    project_id: str = Field(validation_alias=AliasChoices("projectId", "project_id"))
    formats: List[Literal["PNG", "PDF", "DXF"]] = ["PNG"]

@router.post("/export")
async def export(req: APIExportRequest):
    """Экспорт чертежей."""
    try:
        routes = DB.get("routes", {}).get(req.project_id, [])
        legacy_rooms = DB.get("rooms", {}).get(req.project_id, []) or []
        design_graph = DB.get("design", {}).get(req.project_id)

        room_type_map = {}
        if design_graph:
            for rd in design_graph.get("roomDesigns", []):
                rid = rd.get("roomId", "")
                rtype = rd.get("roomType", "bedroom")
                if rid:
                    room_type_map[rid] = rtype

        MIN_AREA_M2 = 4.5
        MAX_AREA_M2 = 300.0
        ROOM_TYPES_CYCLE = ["living_room", "bedroom", "bedroom", "kitchen", "bathroom", "corridor", "toilet"]

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
            return 0.0

        filtered_rooms = [r for r in legacy_rooms if MIN_AREA_M2 <= _area_from_room(r) <= MAX_AREA_M2]
        sorted_rooms = sorted(filtered_rooms, key=_area_from_room, reverse=True)

        for idx, room in enumerate(sorted_rooms):
            rid = room.get("id") or room.get("room_id", "")
            if rid in room_type_map:
                room["roomType"] = room_type_map[rid]
            else:
                room["roomType"] = ROOM_TYPES_CYCLE[idx % len(ROOM_TYPES_CYCLE)]

        devices = (design_graph or {}).get("devices", [])
        base_image_path = DB.get("source", {}).get(req.project_id, {}).get("local_path")

        if not base_image_path:
            raise Exception(f"No source image for project {req.project_id}")

        exports = {}
        os.makedirs("/tmp/exports", exist_ok=True)

        for fmt in req.formats:
            if fmt == "PNG":
                out_path = f"/tmp/exports/{req.project_id}_overlay.png"
                export_overlay_png(
                    base_image_path=base_image_path,
                    rooms=sorted_rooms,
                    devices=devices,
                    routes=routes,
                    output_path=out_path,
                    project_id=req.project_id,
                )
                uri = upload_file(EXPORT_BUCKET, out_path, f"drawings/{req.project_id}_overlay.png")
                exports["PNG"] = uri

            elif fmt == "PDF":
                out_path = f"/tmp/exports/{req.project_id}_drawing.pdf"
                export_pdf(
                    base_image_path=base_image_path,
                    rooms=sorted_rooms,
                    devices=devices,
                    routes=routes,
                    output_path=out_path,
                )
                uri = upload_file(EXPORT_BUCKET, out_path, f"drawings/{req.project_id}_drawing.pdf")
                exports["PDF"] = uri

            elif fmt == "DXF":
                out_path = f"/tmp/exports/{req.project_id}_drawing.dxf"
                export_dxf(
                    rooms=sorted_rooms,
                    devices=devices,
                    routes=routes,
                    output_path=out_path,
                )
                uri = upload_file(EXPORT_BUCKET, out_path, f"drawings/{req.project_id}_drawing.dxf")
                exports["DXF"] = uri

        DB.setdefault("exports", {})[req.project_id] = exports

        return {
            "project_id": req.project_id,
            "exports": exports,
            "devices_count": len(devices),
            "rooms_count": len(sorted_rooms),
        }

    except Exception as e:
        logger.error("Export failed: %s", e, exc_info=True)
        return {"error": str(e), "project_id": req.project_id}