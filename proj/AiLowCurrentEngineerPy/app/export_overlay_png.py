from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


def export_overlay_png(
    src_image_path: str,
    rooms: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    routes: list[Any],
    out_path: str,
) -> str:
    """Рисует overlay поверх исходного плана: комнаты, трассы, устройства.
    Координаты — в пикселях (origin top-left), как у исходной картинки.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {src_image_path}")

    # --- rooms (light outline) ---
    for r in rooms or []:
        poly = r.get("polygon") or r.get("poly")
        if not poly or not isinstance(poly, list) or len(poly) < 3:
            continue
        pts = []
        for p in poly:
            try:
                pts.append([int(round(float(p.get("x")))), int(round(float(p.get("y"))))])
            except Exception:
                continue
        if len(pts) < 3:
            continue
        arr = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [arr], isClosed=True, color=(200, 200, 200), thickness=2)

    # --- routes (red lines) ---
    for route in routes or []:
        points = None
        if isinstance(route, dict):
            points = route.get("points") or route.get("polyline") or route.get("path")
        elif isinstance(route, (list, tuple)) and len(route) >= 2:
            line = route[1]
            if hasattr(line, "coords"):
                points = [{"x": x, "y": y} for x, y in list(line.coords)]

        if not points or len(points) < 2:
            continue

        for a, b in zip(points, points[1:]):
            try:
                x0, y0 = int(round(float(a["x"]))), int(round(float(a["y"])))
                x1, y1 = int(round(float(b["x"]))), int(round(float(b["y"])))
            except Exception:
                continue
            cv2.line(img, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=3)

    # --- devices (green dots + label) ---
    for d in devices or []:
        try:
            x, y = int(round(float(d.get("x", 0)))), int(round(float(d.get("y", 0))))
        except Exception:
            continue
        label = str(d.get("label") or d.get("type") or "DEV")
        cv2.circle(img, (x, y), 8, color=(0, 255, 0), thickness=-1)
        cv2.putText(img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise ValueError(f"Could not write overlay: {out_path}")
    return out_path
