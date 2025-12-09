from __future__ import annotations
import cv2
import pytesseract
import numpy as np
from typing import Dict, Any
from .geometry import DB

# Подсказки для русских названий
KNOWN_ROOM_WORDS = ["комната", "кухня", "ванная", "прихожая", "балкон", "лоджия", "санузел", "туалет"]


def ingest_png(project_id: str, path: str) -> Dict[str, Any]:
    # 1) читаем картинку и готовим бинарь для контуров
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # сгладим и порог
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 5)

    # 2) находим контуры помещений (замкнутые области внутри внешней рамки)
    contours, _ = cv2.findContours(thr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rooms_polys = []
    h, w = thr.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:  # отбрасываем мелкие объекты/текст
            continue
        # аппроксим до полигона (для компактности)
        eps = 0.01 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)

        # отсекаем внешнюю рамку плана (слишком большая площадь)
        if area > 0.25 * (w * h):
            continue
        rooms_polys.append(poly)

    # 3) OCR по картинке (русский язык) и привязка текста к ближайшему «помещению»
    #    image_to_data вернет блоки с bbox — их центры «отнесем» к ближайшему полигону
    ocr = pytesseract.image_to_data(img, lang="rus+eng", output_type=pytesseract.Output.DICT)
    labels = []
    for i in range(len(ocr["text"])):
        txt = (ocr["text"][i] or "").strip().lower()
        if not txt:
            continue
        if any(word in txt for word in KNOWN_ROOM_WORDS):
            x, y, w0, h0 = ocr["left"][i], ocr["top"][i], ocr["width"][i], ocr["height"][i]
            cx, cy = x + w0 // 2, y + h0 // 2
            labels.append((txt, (cx, cy)))

    def point_in_poly(pt, poly):
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    rooms = []
    used = set()
    for idx, poly in enumerate(rooms_polys):
        name = f"room_{idx}"
        # ищем подпись, попавшую внутрь полигона
        for j, (txt, (cx, cy)) in enumerate(labels):
            if j in used:
                continue
            if point_in_poly((cx, cy), poly):
                name = txt
                used.add(j)
                break

        # площадь пока в px^2 (перевод в м² можно добавить после калибровки масштаба)
        area_px = float(cv2.contourArea(poly))
        rooms.append({
            "id": name,
            "polygon": np.squeeze(poly).tolist(),  # [[x,y], ...]
            "area_px2": area_px
        })

    # 4) сохраняем в DB
    DB['rooms'][project_id] = rooms
    # Обнулим зависимые сущности
    DB['devices'][project_id] = []
    DB['routes'][project_id] = []
    DB['candidates'][project_id] = []

    return {"rooms": len(rooms), "note": "parsed from PNG via contours+OCR"}
