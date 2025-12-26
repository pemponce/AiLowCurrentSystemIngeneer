from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from shapely.geometry import Polygon

from app.geometry import DB


# Подсказки для русских названий
KNOWN_ROOM_WORDS = [
    "комната", "кухня", "ванная", "прихожая", "балкон", "лоджия",
    "санузел", "туалет", "коридор", "гостиная", "спальня"
]


def _ensure_dirs() -> None:
    os.makedirs("/tmp/exports", exist_ok=True)


def _poly_points(poly: np.ndarray) -> List[Tuple[float, float]]:
    pts = np.squeeze(poly)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return []
    return [(float(x), float(y)) for x, y in pts.tolist()]


def _safe_ocr_labels(img_bgr) -> List[Tuple[str, Tuple[int, int]]]:
    """
    OCR может падать, если в контейнере нет языкового пакета.
    Тогда просто вернём пусто (не ломаем ingest).
    """
    try:
        ocr = pytesseract.image_to_data(img_bgr, lang="rus+eng", output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    labels: List[Tuple[str, Tuple[int, int]]] = []
    for i in range(len(ocr.get("text", []))):
        txt = (ocr["text"][i] or "").strip().lower()
        if not txt:
            continue
        if any(word in txt for word in KNOWN_ROOM_WORDS):
            x, y, w0, h0 = (
                int(ocr["left"][i]),
                int(ocr["top"][i]),
                int(ocr["width"][i]),
                int(ocr["height"][i]),
            )
            cx, cy = x + w0 // 2, y + h0 // 2
            labels.append((txt, (cx, cy)))
    return labels


def _draw_overlay(src_path: str, rooms_polys: List[np.ndarray], room_names: List[str], out_path: str) -> None:
    img = cv2.imread(src_path)
    if img is None:
        raise RuntimeError(f"Cannot read image for overlay: {src_path}")

    for poly, name in zip(rooms_polys, room_names):
        pts = np.squeeze(poly)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)

        # подпись в первой вершине
        x0, y0 = int(pts[0][0]), int(pts[0][1])
        cv2.putText(img, name, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


def ingest_png(project_id: str, path: str, src_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращаемся к контурному методу, который у тебя работал:
    bilateral + adaptiveThreshold + findContours(RETR_CCOMP) + фильтр по площади.
    Важно: комнаты сохраняем как Shapely Polygon (иначе placement/geometry ломаются).
    Плюс формируем overlay PNG и кладём в S3 exports.
    """
    _ensure_dirs()

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    h, w = img.shape[:2]

    # 1) бинаризация (как в твоём предыдущем варианте)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    # 2) контуры
    contours, _ = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rooms_polys: List[np.ndarray] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 2000:
            continue
        if area > 0.25 * float(w * h):
            continue

        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)

        pts = np.squeeze(poly)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue

        rooms_polys.append(poly)

    # 3) OCR (не обязателен)
    labels = _safe_ocr_labels(img)

    def point_in_poly(pt: Tuple[int, int], poly: np.ndarray) -> bool:
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    # 4) формируем комнаты (Shapely) + meta
    rooms_geom: List[Polygon] = []
    room_meta: List[Dict[str, Any]] = []
    room_names: List[str] = []
    used = set()

    for idx, poly in enumerate(rooms_polys):
        name = f"room_{idx:03d}"
        label = None

        for j, (txt, (cx, cy)) in enumerate(labels):
            if j in used:
                continue
            if point_in_poly((cx, cy), poly):
                name = txt
                label = txt
                used.add(j)
                break

        pts = _poly_points(poly)
        if not pts:
            continue

        geom = Polygon(pts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if (not geom.is_valid) or geom.area <= 1.0:
            continue

        rooms_geom.append(geom)
        room_names.append(name)
        room_meta.append({
            "id": name,
            "label": label,
            "roomType": None,
            "areaPx2": float(geom.area),
            "confidence": 0.60
        })

    # 5) сохраняем в DB
    DB.setdefault("rooms", {})[project_id] = rooms_geom
    DB.setdefault("room_meta", {})[project_id] = room_meta
    DB.setdefault("devices", {})[project_id] = []
    DB.setdefault("routes", {})[project_id] = []
    DB.setdefault("candidates", {})[project_id] = []
    DB.setdefault("source_meta", {})[project_id] = {
        "srcKey": src_key or "",
        "fileType": "png",
        "localPath": path,
        "imageWidth": w,
        "imageHeight": h
    }

    # 6) overlay + upload в exports
    overlay_local = f"/tmp/exports/{project_id}_parsed.png"
    _draw_overlay(path, rooms_polys, room_names, overlay_local)

    overlay_key = f"overlays/{project_id}_parsed.png"
    parsed_overlay_uri = None
    try:
        from app.minio_client import upload_file, EXPORT_BUCKET
        parsed_overlay_uri = upload_file(EXPORT_BUCKET, overlay_local, overlay_key)
    except Exception:
        # если S3 временно недоступен — ingest всё равно успешен
        parsed_overlay_uri = None

    # 7) plan_graph (как dict, без зависимости от отдельных pydantic-контрактов)
    plan_graph = {
        "version": "plan-graph-0.1",
        "source": {
            "srcKey": src_key or "",
            "imageWidth": w,
            "imageHeight": h,
            "dpi": None,
            "scale": {"pxPerMeter": None, "confidence": 0},
        },
        "coordinateSystem": {"origin": "top_left", "units": {"image": "px", "world": "m"}},
        "elements": {
            "walls": [],
            "openings": [],
            "rooms": [
                {
                    "id": m["id"],
                    "polygon": [(float(x), float(y)) for x, y in list(rooms_geom[i].exterior.coords)],
                    "label": m.get("label"),
                    "roomType": None,
                    "areaPx2": m.get("areaPx2"),
                    "confidence": m.get("confidence", 0.0),
                }
                for i, m in enumerate(room_meta)
            ],
        },
        "topology": None,
        "artifacts": {"previewOverlayPngKey": overlay_key, "masks": None},
    }
    DB.setdefault("plan_graph", {})[project_id] = plan_graph

    return {
        "project_id": project_id,
        "plan_graph": plan_graph,
        "rooms": len(rooms_geom),
        "method": "contours",
        "note": "parsed from PNG via contours+OCR (stable variant)",
        "parsed_overlay": parsed_overlay_uri,
        "parsed_overlay_key": overlay_key
    }
