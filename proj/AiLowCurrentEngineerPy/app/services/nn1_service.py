# app/services/nn1_service.py
"""NN-1 сервис: сегментация плана и нумерация комнат."""

import os
import logging
import cv2

logger = logging.getLogger("planner")

# NN-1 модель (lazy load)
try:
    import torch as _torch
    from app.ml.structure_infer import _load_checkpoint, infer_one, _preprocess
    from app.ml.structure_postprocess import extract_geometry

    _NN1_CKPT = "models/structure_rf_v4_bestreal.pt"
    _nn1_model = None
    _NN1_AVAILABLE = True
except Exception as e:
    logger.warning(f"NN-1 недоступна: {e}")
    _NN1_AVAILABLE = False

from app.minio_client import upload_file, EXPORT_BUCKET


# TODO: FIX: нумерация комнат в момент ingest
def nn1_get_rooms(image_path: str, project_id: str) -> list:
    """Запускает NN-1 и возвращает список комнат с типами и polygonPx."""
    if not _NN1_AVAILABLE:
        return []
    global _nn1_model
    try:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        if _nn1_model is None:
            _nn1_model, _, _ = _load_checkpoint(_NN1_CKPT, device)
        img = cv2.imread(image_path)
        if img is None:
            return []
        img_prep = _preprocess(img, mode="binarize", invert=False)
        pred_mask = infer_one(_nn1_model, img_prep, device)
        mask_path = f"/tmp/exports/{project_id}_nn1_mask.png"
        os.makedirs("/tmp/exports", exist_ok=True)
        cv2.imwrite(mask_path, pred_mask)
        geom = extract_geometry(mask_path, image_path=image_path)

        rooms_out = []
        for i, r in enumerate(geom.get("rooms", [])):
            contour = r.get("contour")
            poly_px = contour.reshape(-1, 2).tolist() if contour is not None else []

            # ИСПРАВЛЕНО: Упрощённый расчёт центроида
            _cx, _cy = 0.0, 0.0
            if len(poly_px) >= 3:
                # Простое среднее координат
                _xs = [p[0] for p in poly_px]
                _ys = [p[1] for p in poly_px]
                _cx = sum(_xs) / len(_xs)
                _cy = sum(_ys) / len(_ys)

            rooms_out.append({
                "id": f"room_{i:03d}",
                "roomType": r.get("label", "bedroom"),
                "areaM2": round(r.get("area_m2", 0), 1),
                "polygonPx": poly_px,
                "centroidPx": [round(_cx, 1), round(_cy, 1)],
                "isExterior": r.get("exterior", False),
            })
        logger.info("NN-1: %d комнат для project %s", len(rooms_out), project_id)
        return rooms_out
    except Exception as e:
        logger.error("NN-1 failed: %s", e)
        return []


def make_numbered_plan(image_path: str, rooms: list, project_id: str):
    """Создаёт numbered plan с цветными зонами и белыми номерами."""
    import numpy as np  # ← ДОБАВЬ ИМПОРТ

    logger.info("make_numbered_plan: image_path=%s, rooms=%d", image_path, len(rooms))

    if not rooms:
        return None, None, {}

    img = cv2.imread(image_path)
    if img is None:
        logger.error("make_numbered_plan: Failed to read image at %s", image_path)
        return None, None, {}

    h, w = img.shape[:2]
    logger.info("make_numbered_plan: Image size=%dx%d", w, h)

    overlay = img.copy()

    colors = [
        (200, 150, 255), (255, 180, 150), (150, 255, 150), (150, 200, 255),
        (255, 150, 200), (200, 255, 150), (255, 200, 150), (180, 150, 255),
        (150, 255, 200), (255, 150, 150),
    ]

    room_map = {}
    sorted_rooms = sorted(enumerate(rooms, 1), key=lambda x: x[1].get("areaM2", 0), reverse=True)

    for num, room in sorted_rooms:
        room_id = room.get("id")
        room_map[num] = room_id

        # Рисуем цветную зону
        poly_px = room.get("polygonPx", [])
        if poly_px and len(poly_px) >= 3:
            # ИСПРАВЛЕНО: конвертация в numpy array
            pts = np.array([[int(p[0]), int(p[1])] for p in poly_px], dtype=np.int32)
            color = colors[(num - 1) % len(colors)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (100, 100, 200), 2)

        cx, cy = room.get("centroidPx", [0, 0])
        cx, cy = int(cx), int(cy)

        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            logger.warning("make_numbered_plan: Room %d centroid OUT OF BOUNDS (%d, %d)", num, cx, cy)
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 1.0
        text = str(num)
        (tw, th), _ = cv2.getTextSize(text, font, fs, 2)

        radius = max(25, tw // 2 + 8)
        cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), radius, (80, 80, 80), 2)
        cv2.putText(overlay, text, (cx - tw // 2, cy + th // 2), font, fs, (0, 0, 0), 2, cv2.LINE_AA)

        area = room.get("areaM2", 0)
        alabel = f"{area:.0f}m2"
        afs = 0.45
        (aw, ah), _ = cv2.getTextSize(alabel, font, afs, 1)

        cv2.rectangle(overlay, (cx - aw // 2 - 3, cy + th + 5),
                      (cx + aw // 2 + 3, cy + th + ah + 8), (255, 255, 255), -1)
        cv2.putText(overlay, alabel, (cx - aw // 2, cy + th + ah + 5),
                    font, afs, (40, 40, 40), 1, cv2.LINE_AA)

    alpha = 0.6
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    out_path = f"/tmp/exports/{project_id}_numbered.png"
    os.makedirs("/tmp/exports", exist_ok=True)

    success = cv2.imwrite(out_path, result)
    logger.info("make_numbered_plan: File saved=%s, success=%s", out_path, success)

    s3_key = f"previews/{project_id}_numbered.png"
    try:
        uri = upload_file(EXPORT_BUCKET, out_path, s3_key)
        logger.info("make_numbered_plan: Uploaded to MinIO: %s", uri)
    except Exception as e:
        logger.error("make_numbered_plan: MinIO upload failed: %s", e)
        uri = None

    return out_path, uri, room_map