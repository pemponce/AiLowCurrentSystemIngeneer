from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from shapely.geometry import LineString, Point, Polygon, MultiPolygon

# Регистрируем DejaVu (кириллица)
_FONT_PATHS = [
    "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    "/usr/local/lib/python3.11/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_NAME       = "Helvetica"       # fallback если TTF не найден
_FONT_NAME_BOLD  = "Helvetica-Bold"  # fallback если TTF не найден

for _fp in _FONT_PATHS:
    if os.path.exists(_fp):
        try:
            pdfmetrics.registerFont(TTFont("DejaVu",     _fp))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", _fp))
            _FONT_NAME      = "DejaVu"
            _FONT_NAME_BOLD = "DejaVu-Bold"
            break
        except Exception:
            pass

# ─── словари подписей ─────────────────────────────────────────────────────────

DEVICE_LABELS: Dict[str, str] = {
    "tv_socket":        "ТВ",
    "tv_sockets":       "ТВ",
    "internet_socket":  "LAN",
    "internet_sockets": "LAN",
    "smoke_detector":   "Дым",
    "co2_detector":     "CO₂",
    "ceiling_light":    "Свет",
    "ceiling_lights":   "Свет",
    "night_light":      "Ночн",
    "night_lights":     "Ночн",
    "motion_sensor":    "Дв",
    "intercom":         "Дом",
    "alarm":            "Охр",
    # legacy
    "SOCKET": "Роз",
    "SWITCH": "Выкл",
    "LAMP":   "Свет",
    "SENSOR": "Датч",
    "MODEM":  "LAN",
    "PANEL":  "Щит",
}

ROOM_LABELS: Dict[str, str] = {
    "living_room": "Гостиная",
    "bedroom":     "Спальня",
    "kitchen":     "Кухня",
    "bathroom":    "Ванная",
    "toilet":      "Туалет",
    "corridor":    "Коридор",
}


# ─── геометрия ───────────────────────────────────────────────────────────────

def _coerce_polygon(room_like: Any) -> Optional[Polygon]:
    if isinstance(room_like, Polygon):
        return room_like if room_like.is_valid and room_like.area > 0 else None

    pts = None
    if isinstance(room_like, dict):
        pts = (room_like.get("polygonPx")
               or room_like.get("polygon")
               or room_like.get("points"))
    elif isinstance(room_like, (list, tuple)):
        pts = room_like

    if not pts or len(pts) < 3:
        return None

    try:
        coords = [(float(p[0]), float(p[1])) for p in pts]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly if poly.is_valid and poly.area > 0 else None
    except Exception:
        return None


def _build_room_map(rooms: Any) -> Dict[str, Tuple[Polygon, str, str]]:
    """
    Строит словарь room_id → (polygon, room_type, label).
    Поддерживает оба формата: legacy rooms list и DesignGraph roomDesigns.
    """
    result: Dict[str, Tuple[Polygon, str, str]] = {}
    if not rooms:
        return result
    for r in rooms:
        if not isinstance(r, dict):
            continue
        rid   = str(r.get("id") or r.get("roomId") or r.get("room_id") or "")
        rtype = str(r.get("roomType") or r.get("room_type") or r.get("label") or "bedroom").lower()
        poly  = _coerce_polygon(r)
        if rid and poly:
            label = ROOM_LABELS.get(rtype, rtype.replace("_", " ").title())
            result[rid] = (poly, rtype, label)
    return result


def _place_devices(
    devices: Any,
    room_map: Dict[str, Tuple[Polygon, str, str]],
) -> List[Tuple[str, str, Point, str]]:
    """
    Возвращает список (device_kind, room_id, Point, label).
    Координаты: центроид комнаты + небольшой offset по индексу устройства в комнате.
    """
    out: List[Tuple[str, str, Point, str]] = []
    if not devices:
        return out

    # Считаем сколько устройств уже размещено в каждой комнате (для offset)
    room_device_count: Dict[str, int] = {}

    for d in devices:
        if not isinstance(d, dict):
            continue

        kind    = str(d.get("kind") or d.get("type") or d.get("deviceType") or "DEVICE")
        room_id = str(d.get("roomRef") or d.get("room_id") or d.get("room") or d.get("roomId") or "")
        label   = DEVICE_LABELS.get(kind, kind[:4])

        # Координаты явные
        x = d.get("x")
        y = d.get("y")

        if (x is not None and y is not None
                and float(x) != 0.0 and float(y) != 0.0):
            out.append((kind, room_id, Point(float(x), float(y)), label))
            continue

        # Координаты из центроида комнаты
        if room_id and room_id in room_map:
            poly, _, _ = room_map[room_id]
            cx = poly.centroid.x
            cy = poly.centroid.y

            # Раскладываем устройства по сетке внутри комнаты
            n = room_device_count.get(room_id, 0)
            room_device_count[room_id] = n + 1

            # Сетка 3×N: offset в пикселях
            step = max(15.0, poly.bounds[2] - poly.bounds[0]) / 5
            col  = n % 3
            row  = n // 3
            ox   = (col - 1) * step
            oy   = row * step * 0.8

            pt = Point(cx + ox, cy + oy)
            # Если точка вышла за пределы полигона — используем centroid
            if not poly.contains(pt):
                pt = Point(cx, cy)

            out.append((kind, room_id, pt, label))

    return out


# ─── bbox ─────────────────────────────────────────────────────────────────────

def _compute_bbox(
    polys: List[Polygon],
    devs:  List[Tuple[str, str, Point, str]],
    rts:   List[Tuple[str, LineString]],
) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for p in polys:
        b = p.bounds
        xs += [b[0], b[2]]
        ys += [b[1], b[3]]
    for _, _, pt, _ in devs:
        xs.append(pt.x)
        ys.append(pt.y)
    for _, ls in rts:
        for x, y in ls.coords:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    pad = 20
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


# ─── нормализация маршрутов ───────────────────────────────────────────────────

def _normalize_routes(routes: Any) -> List[Tuple[str, LineString]]:
    out = []
    for r in (routes or []):
        if isinstance(r, tuple) and len(r) >= 2 and isinstance(r[1], LineString):
            out.append((str(r[0]), r[1]))
            continue
        if isinstance(r, dict):
            t      = str(r.get("type") or "CABLE")
            coords = r.get("coords") or r.get("points")
            if isinstance(coords, list) and len(coords) >= 2:
                try:
                    out.append((t, LineString([(float(c[0]), float(c[1])) for c in coords])))
                except Exception:
                    pass
    return out


# ─── символы устройств ───────────────────────────────────────────────────────

def _draw_device(c: canvas.Canvas, kind: str, x: float, y: float, r: float, label: str) -> None:
    """Рисует символ устройства на чертеже."""
    kind_lower = kind.lower()

    if "light" in kind_lower:
        # Светильник — крестик в круге
        c.circle(x, y, r, stroke=1, fill=0)
        c.line(x - r * 0.7, y, x + r * 0.7, y)
        c.line(x, y - r * 0.7, x, y + r * 0.7)

    elif "smoke" in kind_lower or "detector" in kind_lower and "co2" not in kind_lower:
        # Датчик дыма — круг с точкой
        c.circle(x, y, r, stroke=1, fill=0)
        c.circle(x, y, r * 0.25, stroke=0, fill=1)

    elif "co2" in kind_lower:
        # CO2 — квадрат
        c.rect(x - r, y - r, r * 2, r * 2, stroke=1, fill=0)

    elif "internet" in kind_lower or "lan" in kind_lower:
        # Розетка интернет — квадрат с диагональю
        c.rect(x - r, y - r, r * 2, r * 2, stroke=1, fill=0)
        c.line(x - r * 0.6, y + r * 0.6, x + r * 0.6, y - r * 0.6)

    elif "tv" in kind_lower:
        # ТВ розетка — круг с горизонтальной линией
        c.circle(x, y, r, stroke=1, fill=0)
        c.line(x - r * 0.7, y, x + r * 0.7, y)

    elif "night" in kind_lower:
        # Ночник — треугольник
        path = c.beginPath()
        path.moveTo(x, y + r)
        path.lineTo(x + r * 0.87, y - r * 0.5)
        path.lineTo(x - r * 0.87, y - r * 0.5)
        path.close()
        c.drawPath(path, stroke=1, fill=0)

    else:
        # Прочее — ромб
        path = c.beginPath()
        path.moveTo(x, y + r)
        path.lineTo(x + r, y)
        path.lineTo(x, y - r)
        path.lineTo(x - r, y)
        path.close()
        c.drawPath(path, stroke=1, fill=0)

    # Подпись под символом
    c.setFont(_FONT_NAME, max(4, r * 0.9))
    c.drawCentredString(x, y - r - max(3, r * 0.8), label)


# ─── главная функция ─────────────────────────────────────────────────────────

def export_pdf(
    project_id: str,
    rooms:   Any,
    devices: Any,
    routes:  Any,
    out_path: str,
) -> None:
    # Строим карту комнат
    room_map = _build_room_map(rooms)
    polys    = [v[0] for v in room_map.values()]

    # Если room_map пустой — пробуем старый путь
    if not polys:
        polys = [_coerce_polygon(r) for r in (rooms or [])]
        polys = [p for p in polys if p is not None]

    # Размещаем устройства
    devs = _place_devices(devices, room_map)
    rts  = _normalize_routes(routes)
    bbox = _compute_bbox(polys, devs, rts)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    c = canvas.Canvas(out_path, pagesize=A4)
    page_w, page_h = A4

    if bbox is None:
        c.setFont(_FONT_NAME, 12)
        c.drawString(2 * cm, page_h - 2 * cm, f"Project {project_id}: нет данных для экспорта")
        c.showPage()
        c.save()
        return

    minx, miny, maxx, maxy = bbox
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)

    margin    = 1.5 * cm
    title_h   = 1.0 * cm
    legend_h  = 1.2 * cm
    draw_w    = page_w - 2 * margin
    draw_h    = page_h - 2 * margin - title_h - legend_h

    s = min(draw_w / span_x, draw_h / span_y)
    dev_r = max(4.0, min(10.0, 6.0 * s))

    # Y: PDF origin снизу, план origin сверху → инвертируем
    def tr(x: float, y: float) -> Tuple[float, float]:
        return (
            margin + (x - minx) * s,
            margin + legend_h + (maxy - y) * s,
        )

    # ── заголовок ────────────────────────────────────────────────
    c.setFont(_FONT_NAME_BOLD, 11)
    c.drawString(margin, page_h - margin * 0.6, f"AILCE TECH  —  Проект {project_id}")
    c.setFont(_FONT_NAME, 8)
    c.drawString(margin, page_h - margin * 0.6 - 12,
                 "Схема размещения слаботочных устройств")

    # ── комнаты ──────────────────────────────────────────────────
    c.setLineWidth(1.2)
    c.setStrokeColorRGB(0, 0, 0)

    for rid, (poly, rtype, rlabel) in room_map.items():
        coords = list(poly.exterior.coords)
        if len(coords) < 2:
            continue
        x0, y0 = tr(coords[0][0], coords[0][1])
        path = c.beginPath()
        path.moveTo(x0, y0)
        for px, py in coords[1:]:
            path.lineTo(*tr(px, py))
        path.close()
        c.drawPath(path, stroke=1, fill=0)

        # Подпись типа комнаты в центре
        cx, cy = tr(poly.centroid.x, poly.centroid.y)
        c.setFont(_FONT_NAME, max(5, min(8, dev_r * 0.8)))
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawCentredString(cx, cy, rlabel)
        c.setFillColorRGB(0, 0, 0)

    # ── маршруты ─────────────────────────────────────────────────
    c.setLineWidth(0.6)
    c.setStrokeColorRGB(0.4, 0.4, 0.8)
    for _, ls in rts:
        coords = list(ls.coords)
        if len(coords) < 2:
            continue
        path = c.beginPath()
        path.moveTo(*tr(coords[0][0], coords[0][1]))
        for px, py in coords[1:]:
            path.lineTo(*tr(px, py))
        c.drawPath(path, stroke=1, fill=0)
    c.setStrokeColorRGB(0, 0, 0)

    # ── устройства ───────────────────────────────────────────────
    c.setLineWidth(0.8)
    for kind, room_id, pt, label in devs:
        dx, dy = tr(pt.x, pt.y)
        _draw_device(c, kind, dx, dy, dev_r, label)

    # ── легенда ──────────────────────────────────────────────────
    legend_y = margin * 0.4
    c.setFont(_FONT_NAME_BOLD, 7)
    c.drawString(margin, legend_y + 4, "Условные обозначения:")

    legend_items = sorted(set(d[0] for d in devs))
    lx = margin + 90
    c.setLineWidth(0.6)
    for kind in legend_items:
        label = DEVICE_LABELS.get(kind, kind[:6])
        _draw_device(c, kind, lx, legend_y + 5, 4.0, "")
        c.setFont(_FONT_NAME, 6)
        c.drawString(lx + 7, legend_y + 3, label)
        lx += 45
        if lx > page_w - margin:
            break

    # ── рамка ────────────────────────────────────────────────────
    c.setLineWidth(0.5)
    c.setStrokeColorRGB(0.7, 0.7, 0.7)
    c.rect(margin * 0.5, margin * 0.5,
           page_w - margin, page_h - margin,
           stroke=1, fill=0)

    c.showPage()
    c.save()