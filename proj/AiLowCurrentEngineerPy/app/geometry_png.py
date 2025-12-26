from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.geometry import DB

KNOWN_ROOM_WORDS = [
    "комната",
    "кухня",
    "ванная",
    "прихожая",
    "балкон",
    "лоджия",
    "санузел",
    "туалет",
    "гостиная",
    "спальня",
    "коридор",
]


def _try_ocr_room_labels(img_bgr: np.ndarray) -> List[Tuple[str, Tuple[int, int]]]:
    """Пытаемся получить подписи помещений OCR (опционально).

    pytesseract/tesseract могут отсутствовать — тогда просто вернём пустой список.
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return []

    try:
        ocr = pytesseract.image_to_data(img_bgr, lang="rus+eng", output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    labels: List[Tuple[str, Tuple[int, int]]] = []
    texts = ocr.get("text", [])
    for i in range(len(texts)):
        txt = (texts[i] or "").strip().lower()
        if not txt:
            continue
        if any(word in txt for word in KNOWN_ROOM_WORDS):
            x = int(ocr.get("left", [0])[i])
            y = int(ocr.get("top", [0])[i])
            w0 = int(ocr.get("width", [0])[i])
            h0 = int(ocr.get("height", [0])[i])
            labels.append((txt, (x + w0 // 2, y + h0 // 2)))
    return labels


def ingest_png(project_id: str, path: str) -> Dict[str, Any]:
    """MVP PNG-ingest.

    Пробует найти помещения по контурам + опционально OCR для label.
    IDs помещений нормализуются в формат room_000, room_001, ...
    """
    DB.setdefault("rooms", {}).setdefault(project_id, [])
    DB.setdefault("devices", {}).setdefault(project_id, [])
    DB.setdefault("routes", {}).setdefault(project_id, [])
    DB.setdefault("candidates", {}).setdefault(project_id, [])
    DB.setdefault("source_meta", {}).setdefault(project_id, {})

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5,
    )

    contours, _ = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    h, w = thr.shape[:2]

    room_polys: List[np.ndarray] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 2000:
            continue
        if area > 0.25 * float(w * h):
            continue
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        if poly is None or len(poly) < 3:
            continue
        room_polys.append(poly)

    labels = _try_ocr_room_labels(img)

    def point_in_poly(pt: Tuple[int, int], poly: np.ndarray) -> bool:
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    rooms: List[Dict[str, Any]] = []
    used = set()
    for idx, poly in enumerate(room_polys):
        rid = f"room_{idx:03d}"
        label: Optional[str] = None
        for j, (txt, (cx, cy)) in enumerate(labels):
            if j in used:
                continue
            if point_in_poly((cx, cy), poly):
                label = txt
                used.add(j)
                break

        pts = np.squeeze(poly).tolist()
        area_px = float(cv2.contourArea(poly))
        rooms.append(
            {"id": rid, "polygonPx": pts, "label": label, "areaPx": area_px, "areaM2": None}
        )

    DB["rooms"][project_id] = rooms
    DB["devices"][project_id] = []
    DB["routes"][project_id] = []
    DB["candidates"][project_id] = []
    DB["source_meta"][project_id] = {"localPath": path, "imageWidth": w, "imageHeight": h}

    return {
        "rooms": len(rooms),
        "method": "contours+optional-ocr",
        "note": "PNG parsed (heuristic contours). Для более стабильного результата используй /detect-structure: там есть wallsMask + auto rooms + freeSpaceMask.",
    }
