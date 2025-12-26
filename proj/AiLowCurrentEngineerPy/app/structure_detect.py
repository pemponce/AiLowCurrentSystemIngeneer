from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, upload_file


def _ensure_dirs() -> None:
    os.makedirs("/tmp/exports", exist_ok=True)


def build_walls_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Делает бинарную маску стен (255 = стена).
    Это эвристика для «типовых» чертежей/сканов: толстые линии стен должны выделяться.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5,
    )

    # Убираем мелкий шум
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k1, iterations=1)

    # Склеиваем стены (утолщаем/соединяем разрывы)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    walls = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k2, iterations=2)

    return walls


def build_free_space_mask(walls_mask: np.ndarray) -> np.ndarray:
    """
    Маска свободного пространства (255 = можно ходить/класть трассу).
    Делаем инверсию и слегка эрозируем, чтобы свободное пространство не прилипало к стенам.
    """
    free = cv2.bitwise_not(walls_mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free = cv2.erode(free, k, iterations=1)
    return free


def extract_rooms_from_free_space(free_mask: np.ndarray) -> List[List[List[float]]]:
    """
    Извлекаем комнаты как компоненты связности свободного пространства,
    которые НЕ касаются границ картинки (граница — это внешний фон).
    Возвращаем список полигонов в px: [ [ [x,y], ... ], ... ].
    """
    h, w = free_mask.shape[:2]
    bin_img = (free_mask > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    polys: List[List[List[float]]] = []

    # порог площади комнаты (настраивается)
    min_area = max(1500, int(0.002 * (w * h)))

    for lbl in range(1, num):
        x, y, ww, hh, area = stats[lbl]
        if area < min_area:
            continue

        # если компонент касается края — это скорее внешний фон/коридор за пределами
        touches_border = (x <= 0) or (y <= 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if touches_border:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[labels == lbl] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
            continue

        eps = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2).tolist()
        polys.append([[float(px), float(py)] for px, py in pts])

    return polys


def draw_overlay(img_bgr: np.ndarray, walls_mask: np.ndarray, rooms_polys: List[List[List[float]]]) -> np.ndarray:
    out = img_bgr.copy()

    # красим стены поверх (полупрозрачно)
    walls_col = out.copy()
    walls_col[walls_mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(out, 0.80, walls_col, 0.20, 0)

    # рисуем контуры комнат
    for poly in rooms_polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return out


def detect_structure(
    project_id: str,
    image_path: str,
    src_key: Optional[str] = None,
    debug: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    """
    Главная функция: строит walls/free-space/rooms, сохраняет в DB и грузит артефакты в S3.

    Важно: принимает src_key/debug, чтобы не было 500 при вызове из API.
    """
    _ensure_dirs()

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    walls = build_walls_mask(img)
    free = build_free_space_mask(walls)

    # если rooms уже есть из /ingest — оставим, иначе попробуем извлечь автоматически
    rooms_existing = DB.get("rooms", {}).get(project_id) or []
    rooms_polys: List[List[List[float]]] = []

    if rooms_existing:
        # поддержим dict-формат ingest_png: polygonPx
        for r in rooms_existing:
            if isinstance(r, dict):
                pts = r.get("polygonPx") or r.get("polygon") or r.get("points")
                if isinstance(pts, list) and len(pts) >= 3:
                    rooms_polys.append([[float(x), float(y)] for x, y in pts])
    else:
        rooms_polys = extract_rooms_from_free_space(free)
        rooms_dicts: List[Dict[str, Any]] = []
        for i, poly in enumerate(rooms_polys):
            rooms_dicts.append(
                {
                    "id": f"room_{i:03d}",
                    "polygonPx": poly,
                    "areaPx": float(abs(cv2.contourArea(np.array(poly, dtype=np.float32)))),
                }
            )
        DB.setdefault("rooms", {})[project_id] = rooms_dicts

    openings: List[Dict[str, Any]] = []  # пока пусто (следующим шагом добавим детект дверей/окон)

    overlay = draw_overlay(img, walls, rooms_polys)

    walls_local = f"/tmp/exports/{project_id}_walls.png"
    free_local = f"/tmp/exports/{project_id}_free.png"
    overlay_local = f"/tmp/exports/{project_id}_structure.png"
    cv2.imwrite(walls_local, walls)
    cv2.imwrite(free_local, free)
    cv2.imwrite(overlay_local, overlay)

    walls_key = f"masks/{project_id}_walls.png"
    free_key = f"masks/{project_id}_free_space.png"
    overlay_key = f"overlays/{project_id}_structure.png"

    walls_uri = upload_file(EXPORT_BUCKET, walls_local, walls_key)
    free_uri = upload_file(EXPORT_BUCKET, free_local, free_key)
    overlay_uri = upload_file(EXPORT_BUCKET, overlay_local, overlay_key)

    # plan_graph — единый контракт
    rooms_out: List[Dict[str, Any]] = []
    for i, poly in enumerate(rooms_polys):
        rooms_out.append({"id": f"room_{i:03d}", "polygonPx": poly})

    plan_graph: Dict[str, Any] = {
        "version": "plan-graph-0.3",
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
            "openings": openings,
            "rooms": rooms_out,
        },
        "topology": None,
        "artifacts": {
            "previewOverlayPngKey": overlay_key,
            "masks": {"wallsMaskKey": walls_key, "freeSpaceMaskKey": free_key},
        },
        "debug": {"enabled": bool(debug)},
    }

    DB.setdefault("structure", {})[project_id] = {
        "walls_mask_key": walls_key,
        "free_space_mask_key": free_key,
        "overlay_key": overlay_key,
        "openings": openings,
    }
    DB.setdefault("plan_graph", {})[project_id] = plan_graph
    DB.setdefault("source_meta", {}).setdefault(project_id, {})
    DB["source_meta"][project_id].update({"localPath": image_path, "srcKey": src_key or "", "imageWidth": w, "imageHeight": h})

    return {
        "project_id": project_id,
        "image": {"w": w, "h": h},
        "rooms": len(rooms_out),
        "openings": len(openings),
        "masks": {"walls": walls_uri, "free_space": free_uri},
        "overlay": overlay_uri,
        "plan_graph": plan_graph,
    }
