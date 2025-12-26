from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from shapely.geometry import LineString, Point, Polygon


LABEL_BY_DEVICE_TYPE: Dict[str, str] = {
    "SOCKET": "Розетка",
    "SWITCH": "Выключатель",
    "LAMP": "Светильник",
    "SENSOR": "Датчик",
    "MODEM": "Модем/роутер",
    "PANEL": "Щит",
}


def _coerce_polygon(room_like: Any) -> Optional[Polygon]:
    if isinstance(room_like, Polygon):
        return room_like

    pts = None
    if isinstance(room_like, dict):
        pts = room_like.get("polygonPx") or room_like.get("polygon") or room_like.get("points")
    elif isinstance(room_like, (list, tuple)):
        pts = room_like

    if not pts or len(pts) < 3:
        return None

    try:
        coords = [(float(x), float(y)) for x, y in pts]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if (not poly.is_valid) or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def _normalize_devices(devices: Any) -> List[Tuple[str, str, Point]]:
    out: List[Tuple[str, str, Point]] = []
    for d in (devices or []):
        if isinstance(d, tuple) and len(d) == 3 and isinstance(d[2], Point):
            t, room_ref, p = d
            out.append((str(t), str(room_ref), Point(float(p.x), float(p.y))))
            continue

        if isinstance(d, dict):
            t = str(d.get("type") or d.get("deviceType") or "DEVICE")
            room_ref = str(d.get("room") or d.get("roomId") or d.get("room_idx") or "room_000")

            x = d.get("x")
            y = d.get("y")

            pt = d.get("point")
            if (x is None or y is None) and isinstance(pt, dict):
                x = pt.get("x")
                y = pt.get("y")
            if (x is None or y is None) and isinstance(pt, (list, tuple)) and len(pt) == 2:
                x, y = pt

            if x is None or y is None:
                continue

            out.append((t, room_ref, Point(float(x), float(y))))
    return out


def _normalize_routes(routes: Any) -> List[Tuple[str, LineString]]:
    out: List[Tuple[str, LineString]] = []
    for r in (routes or []):
        if isinstance(r, tuple) and len(r) >= 2 and isinstance(r[1], LineString):
            out.append((str(r[0]), r[1]))
            continue

        if isinstance(r, dict):
            t = str(r.get("type") or r.get("t") or "CABLE")
            coords = r.get("coords") or r.get("points")
            if isinstance(coords, list) and len(coords) >= 2:
                try:
                    out.append((t, LineString([(float(x), float(y)) for x, y in coords])))
                except Exception:
                    pass
    return out


def _compute_bbox(polys: List[Polygon], devs: List[Tuple[str, str, Point]], rts: List[Tuple[str, LineString]]) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []

    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        xs.extend([minx, maxx])
        ys.extend([miny, maxy])

    for _, _, pt in devs:
        xs.append(float(pt.x))
        ys.append(float(pt.y))

    for _, ls in rts:
        for x, y in ls.coords:
            xs.append(float(x))
            ys.append(float(y))

    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def export_pdf(project_id: str, rooms: Any, devices: Any, routes: Any, out_path: str) -> None:
    polys = [_coerce_polygon(r) for r in (rooms or [])]
    polys = [p for p in polys if p is not None]
    devs = _normalize_devices(devices)
    rts = _normalize_routes(routes)

    bbox = _compute_bbox(polys, devs, rts)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    c = canvas.Canvas(out_path, pagesize=A4)
    page_w, page_h = A4

    if bbox is None:
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, page_h - 2 * cm, f"Project {project_id}: nothing to export")
        c.showPage()
        c.save()
        return

    minx, miny, maxx, maxy = bbox
    span_x = max(maxx - minx, 1e-6)
    span_y = max(maxy - miny, 1e-6)

    margin = 1.2 * cm
    draw_w = page_w - 2 * margin
    draw_h = page_h - 2 * margin

    # координаты как пиксели (origin top-left), поэтому Y инвертируем
    s = min(draw_w / span_x, draw_h / span_y)

    def tr(x: float, y: float) -> Tuple[float, float]:
        return (
            margin + (x - minx) * s,
            margin + (maxy - y) * s,
        )

    device_radius = max(1.5, 3.0 * s)

    # title
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_h - margin + 0.2 * cm, f"AILCE TECH — Project {project_id}")

    # rooms
    c.setLineWidth(1)
    for poly in polys:
        coords = list(poly.exterior.coords)
        if len(coords) < 2:
            continue
        x0, y0 = tr(float(coords[0][0]), float(coords[0][1]))
        path = c.beginPath()
        path.moveTo(x0, y0)
        for x, y in coords[1:]:
            x1, y1 = tr(float(x), float(y))
            path.lineTo(x1, y1)
        path.close()
        c.drawPath(path, stroke=1, fill=0)

    # routes
    c.setLineWidth(0.8)
    for _, ls in rts:
        coords = list(ls.coords)
        if len(coords) < 2:
            continue
        x0, y0 = tr(float(coords[0][0]), float(coords[0][1]))
        path = c.beginPath()
        path.moveTo(x0, y0)
        for x, y in coords[1:]:
            x1, y1 = tr(float(x), float(y))
            path.lineTo(x1, y1)
        c.drawPath(path, stroke=1, fill=0)

    # devices
    c.setLineWidth(1)
    for t, _, pt in devs:
        x, y = tr(float(pt.x), float(pt.y))
        c.circle(x, y, device_radius, stroke=1, fill=0)
        # подпись рядом (по желанию)
        # c.setFont("Helvetica", 7)
        # c.drawString(x + device_radius + 2, y + device_radius + 2, LABEL_BY_DEVICE_TYPE.get(t, t))

    # legend
    c.setFont("Helvetica", 9)
    legend = ", ".join(sorted({LABEL_BY_DEVICE_TYPE.get(t, t) for t, _, _ in devs})) or "—"
    c.drawString(margin, margin - 0.5 * cm, f"Устройства: {legend}")

    c.showPage()
    c.save()
