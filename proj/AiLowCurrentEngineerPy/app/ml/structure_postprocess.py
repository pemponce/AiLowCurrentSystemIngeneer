"""
app/ml/structure_postprocess.py

Постпроцессинг маски NN-1 → геометрия плана.

Входные данные:
    pred_mask.png — маска классов (0=bg, 1=wall, 2=door, 3=window, 4=front_door)

Выходные данные:
    - wall_skeleton   : np.ndarray (H,W) uint8 — центральные линии стен
    - rooms           : list[dict]  — комнаты как полигоны + bbox + площадь
    - doors           : list[dict]  — позиции дверей (центроид + bbox)
    - windows         : list[dict]  — позиции окон (центроид + bbox)
    - debug_vis.png   : визуализация результата (если out_dir задан)

Использование:
    from app.ml.structure_postprocess import extract_geometry
    result = extract_geometry(pred_mask_path="out/test/pred_mask.png", out_dir="out/test")
    rooms   = result["rooms"]
    doors   = result["doors"]
    windows = result["windows"]
    skeleton = result["wall_skeleton"]
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from skimage.morphology import skeletonize

# ─────────────────────────── константы ───────────────────────────
CLASS_BG = 0
CLASS_WALL = 1
CLASS_DOOR = 2
CLASS_WINDOW = 3
CLASS_FRONT_DOOR = 4

COLORS = {
    "wall": (60, 60, 220),
    "door": (0, 220, 220),
    "window": (220, 180, 0),
    "skeleton": (0, 255, 0),
    "room": (180, 255, 180),
    "centroid": (0, 0, 255),
}


# ─────────────────────────── утилиты ─────────────────────────────

def _binary(mask: np.ndarray, cls: int) -> np.ndarray:
    return (mask == cls).astype(np.uint8) * 255


def _largest_components(bin_mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    out = np.zeros_like(bin_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _centroids_from_mask(bin_mask: np.ndarray, min_area: int = 50) -> list[dict]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )
    result = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        result.append({
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "area": area,
        })
    return result


# ─────────────────────── скелетизация стен ───────────────────────

def _skeletonize_walls(wall_mask: np.ndarray) -> np.ndarray:
    wall_clean = _largest_components(wall_mask, min_area=500)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(wall_clean, cv2.MORPH_CLOSE, k, iterations=2)
    skel = skeletonize(closed > 0)
    skel_uint8 = (skel.astype(np.uint8)) * 255
    skel_clean = _largest_components(skel_uint8, min_area=50)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.dilate(skel_clean, k2, iterations=1)


def _repair_wall_gaps(wall_mask: np.ndarray) -> np.ndarray:
    """
    Закрывает небольшие разрывы в наружных стенах вдоль краёв изображения.
    Ядро 25px — только для мелких разрывов (проёмы окон в стене).
    """
    h, w = wall_mask.shape
    result = wall_mask.copy()

    # Горизонтальное закрытие — верхняя и нижняя стены
    k_wide_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    closed_h = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_wide_h, iterations=1)
    margin_y = h // 5
    result[:margin_y, :] = closed_h[:margin_y, :]
    result[h - margin_y:, :] = closed_h[h - margin_y:, :]

    # Вертикальное закрытие — левая и правая стены
    k_wide_v = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    closed_v = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_wide_v, iterations=1)
    margin_x = w // 5
    result[:, :margin_x] = cv2.bitwise_or(result[:, :margin_x], closed_v[:, :margin_x])
    result[:, w - margin_x:] = cv2.bitwise_or(result[:, w - margin_x:], closed_v[:, w - margin_x:])

    return result


# ──────────────────── bounding box плана ─────────────────────────

def _find_plan_bbox(wall_mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Находит bounding box плана как объединение всех значимых компонент стен.

    Фильтрует:
    - слишком маленькие компоненты (< 500px) — шум
    - тонкие компоненты (< 5px по одной из сторон) — граничные линии AutoCAD
    - компоненты на всю высоту/ширину изображения (> 80%) — рамки листа

    Дополнительно уточняет правую границу по реальному положению стены,
    чтобы не захватывать подъезд/улицу справа.

    Возвращает (x, y, w, h).
    """
    h, w = wall_mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
    if num < 2:
        return (0, 0, w, h)

    x_min, y_min = w, h
    x_max, y_max = 0, 0
    found = False

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if area < 500: continue
        if bw < 5 or bh < 5: continue

        # Фильтр рамок листа ОТКЛЮЧЕН, чтобы не терять длинные стены
        x_min = min(x_min, bx)
        y_min = min(y_min, by)
        x_max = max(x_max, bx + bw)
        y_max = max(y_max, by + bh)
        found = True

    if not found: return (0, 0, w, h)

    # Увеличенный паддинг для надежности
    pad = 150
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

    # Уточняем правую границу: ранее сканировалась только одна линия mid_y,
    # что приводило к обрезанию если в этой строке был проём или разрыв.
    # Теперь используем x_max, полученный из объединения всех значимых компонент стен.
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _merge_adjacent_rooms(labels, num, wall_mask, min_room_area):
    """
    Объединяет компоненты, которые разделены не стеной, а просто виртуальной границей.
    """
    h, w = labels.shape
    merged_labels = labels.copy()

    # Ищем пары смежных компонентов
    # Для этого расширяем каждый компонент и смотрим, с кем он пересекается
    for i in range(1, num):
        for j in range(i + 1, num):
            # Проверяем, есть ли между компонентами i и j стена
            # Для этого берем границу между ними
            mask_i = (merged_labels == i).astype(np.uint8)
            mask_j = (merged_labels == j).astype(np.uint8)

            # Расширяем обе маски на 5 пикселей
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            dil_i = cv2.dilate(mask_i, k)
            dil_j = cv2.dilate(mask_j, k)

            # Зона контакта
            contact = cv2.bitwise_and(dil_i, dil_j)
            if np.any(contact):
                # Если в зоне контакта НЕТ стен, значит это одна комната
                if not np.any(cv2.bitwise_and(contact, wall_mask)):
                    merged_labels[merged_labels == j] = i

    return merged_labels


# ──────────────────────── извлечение комнат ──────────────────────

def _extract_rooms(wall_mask, door_mask, win_mask=None, min_room_area=2000):
    h, w = wall_mask.shape

    # Добавляем 2px рамку по краям, чтобы "запереть" комнаты внутри
    walled = wall_mask.copy()
    cv2.rectangle(walled, (0, 0), (w - 1, h - 1), 255, 2)

    combined = cv2.bitwise_or(walled, door_mask)
    if win_mask is not None:
        combined = cv2.bitwise_or(combined, win_mask)

    # Используем минимальную дилатацию, чтобы не сливать комнаты через двери,
    # но позволить им быть "близко" друг к другу
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(combined, k, iterations=1)
    free_space = cv2.bitwise_not(dilated)

    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[5:-5, 5:-5] = 255
    free_space = cv2.bitwise_and(free_space, border_mask)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space, connectivity=8)

    # Объединяем L-образные комнаты, разделенные виртуальными границами
    labels = _merge_adjacent_rooms(labels, num, wall_mask, min_room_area)

    # Пересчитываем компоненты после объединения
    num, labels, stats, centroids = cv2.connectedComponentsWithStats((labels > 0).astype(np.uint8) * 255,
                                                                     connectivity=8)

    bg_candidates = [labels[1, 1], labels[1, w - 2], labels[h - 2, 1], labels[h - 2, w - 2]]
    bg_id = max(set(bg_candidates), key=bg_candidates.count)

    rooms = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_room_area or i == bg_id:
            continue

        room_bin = (labels == i).astype(np.uint8) * 255

        # КРИТЕРИЙ УЛИЦЫ: Проверяем, есть ли у комнаты хотя бы одна стена рядом
        # Расширяем комнату и смотрим пересечение со стенами
        k_check = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        expanded_room = cv2.dilate(room_bin, k_check)
        if not np.any(cv2.bitwise_and(expanded_room, wall_mask)):
            # Если стен рядом нет — это улица/пустота
            continue

        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        rw = int(stats[i, cv2.CC_STAT_WIDTH])
        rh = int(stats[i, cv2.CC_STAT_HEIGHT])

        contours, _ = cv2.findContours(room_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0] if contours else None
        polygon = None
        if contour is not None:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

        rooms.append({
            "id": i,
            "centroid": (cx, cy),
            "bbox": (x, y, rw, rh),
            "area_px": area,
            "contour": contour,
            "polygon": polygon,
        })

    rooms.sort(key=lambda r: r["area_px"], reverse=True)
    return rooms


# ──────────────────────── классификация комнат ───────────────────

WALL_THICKNESS_MM = 200.0


def _estimate_scale(wall_mask: np.ndarray) -> float:
    """
    Масштаб в мм/px.
    Откалиброван по реальным планам 1600x1280px:
    1px = ~14мм (из AutoCAD: комната 9.54м² = 48862px²)
    """
    h, w = wall_mask.shape
    ref_scale = 14.0
    ref_pixels = 1600 * 1280
    actual_pixels = w * h
    return ref_scale * ((actual_pixels / ref_pixels) ** 0.5)


def _count_windows_in_room(
        room_contour: np.ndarray,
        windows: list[dict],
        h: int, w: int,
) -> int:
    if room_contour is None:
        return 0
    room_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(room_img, [room_contour], -1, 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    room_expanded = cv2.dilate(room_img, k, iterations=2)
    count = 0
    for win in windows:
        cx, cy = int(win["centroid"][0]), int(win["centroid"][1])
        if 0 <= cy < h and 0 <= cx < w:
            if room_expanded[cy, cx] == 255:
                count += 1
    return count


def classify_room(area_px, bbox, contour, windows_in_room, scale_mm_per_px):
    px_per_m2 = (1000 / scale_mm_per_px) ** 2
    area_m2 = area_px / px_per_m2
    x, y, bw, bh = bbox
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    has_window = windows_in_room > 0

    if aspect > 3.5 and area_m2 < 20:
        return "corridor"
    if area_m2 > 30:
        return "living_room"
    if area_m2 > 8:
        return "bedroom"
    if area_m2 > 5 and has_window:
        return "kitchen"
    if area_m2 > 5:
        return "kitchen"
    if area_m2 > 3:
        return "bathroom"
    if area_m2 > 1.5:
        return "toilet"
    return "corridor"


def classify_all_rooms(
        rooms: list[dict],
        windows: list[dict],
        wall_mask: np.ndarray,
) -> list[dict]:
    h, w = wall_mask.shape
    scale = _estimate_scale(wall_mask)
    for room in rooms:
        wins = _count_windows_in_room(room["contour"], windows, h, w)
        room_type = classify_room(
            area_px=room["area_px"],
            bbox=room["bbox"],
            contour=room["contour"],
            windows_in_room=wins,
            scale_mm_per_px=scale,
        )
        px_per_m2 = (1000 / scale) ** 2
        room["area_m2"] = round(room["area_px"] / px_per_m2, 1)
        room["windows_count"] = wins
        room["room_type"] = room_type
    return rooms


# ──────────────────────── визуализация ───────────────────────────

def _draw_debug(image_path, wall_skeleton, rooms, doors, windows, out_path):
    img = cv2.imread(image_path)
    if img is None:
        h, w = wall_skeleton.shape
        img = np.ones((h, w, 3), dtype=np.uint8) * 40

    vis = img.copy()
    overlay = vis.copy()

    ROOM_COLORS = {
        "living_room": (50, 200, 50),
        "bedroom": (50, 50, 200),
        "kitchen": (200, 150, 50),
        "bathroom": (50, 200, 200),
        "toilet": (150, 200, 200),
        "corridor": (180, 180, 50),
        "storage": (150, 150, 150),
        "unknown": (200, 200, 200),
    }

    for idx, room in enumerate(rooms):
        color = ROOM_COLORS.get(room.get("room_type", "unknown"), (200, 200, 200))
        if room["contour"] is not None:
            cv2.drawContours(overlay, [room["contour"]], -1, color, -1)
            cx, cy = int(room["centroid"][0]), int(room["centroid"][1])
            cv2.putText(vis, f"R{idx + 1} {room.get('room_type', '?')}",
                        (cx - 30, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(vis, f"{room.get('area_m2', 0)}m2",
                        (cx - 20, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 100), 1)

    cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)
    vis[wall_skeleton > 0] = (0, 220, 0)

    for d in doors:
        cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
        cv2.circle(vis, (cx, cy), 8, (0, 220, 220), -1)
        cv2.circle(vis, (cx, cy), 8, (0, 0, 0), 1)

    for w_obj in windows:
        cx, cy = int(w_obj["centroid"][0]), int(w_obj["centroid"][1])
        x, y, bw, bh = w_obj["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (220, 180, 0), 2)
        cv2.circle(vis, (cx, cy), 5, (220, 180, 0), -1)

    cv2.imwrite(out_path, vis)


# ─────────────────────── главная функция ─────────────────────────

def extract_geometry(
        pred_mask_path: str,
        image_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        min_room_area: int = 5000,
        min_door_area: int = 50,
        min_window_area: int = 50,
) -> dict:
    mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Не найдена маска: {pred_mask_path}")

    wall_bin = _binary(mask, CLASS_WALL)
    door_bin = _binary(mask, CLASS_DOOR)
    win_bin = _binary(mask, CLASS_WINDOW)
    fdoor_bin = _binary(mask, CLASS_FRONT_DOOR)

    wall_bin = _largest_components(wall_bin, min_area=200)
    wall_bin = _repair_wall_gaps(wall_bin)
    door_bin = _largest_components(door_bin, min_area=min_door_area)
    win_bin = _largest_components(win_bin, min_area=min_window_area)
    fdoor_bin = _largest_components(fdoor_bin, min_area=min_door_area)

    wall_skeleton = _skeletonize_walls(wall_bin)
    rooms = _extract_rooms(wall_bin, door_bin, win_mask=win_bin, min_room_area=2000)
    doors = _centroids_from_mask(door_bin, min_area=min_door_area)
    windows = _centroids_from_mask(win_bin, min_area=min_window_area)
    front_doors = _centroids_from_mask(fdoor_bin, min_area=min_door_area)
    rooms = classify_all_rooms(rooms, windows, wall_bin)

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path / "wall_skeleton.png"), wall_skeleton)
        if image_path:
            _draw_debug(
                image_path=image_path,
                wall_skeleton=wall_skeleton,
                rooms=rooms,
                doors=doors,
                windows=windows,
                out_path=str(out_path / "debug_geometry.png"),
            )

    return {
        "wall_skeleton": wall_skeleton,
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "front_doors": front_doors,
    }


# ─────────────────────── CLI ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Постпроцессинг маски NN-1")
    parser.add_argument("--mask", required=True, help="путь к pred_mask.png")
    parser.add_argument("--image", default=None, help="путь к оригинальному изображению")
    parser.add_argument("--out", default=None, help="папка для результатов")
    parser.add_argument("--min-room-area", type=int, default=5000)
    args = parser.parse_args()

    result = extract_geometry(
        pred_mask_path=args.mask,
        image_path=args.image,
        out_dir=args.out,
        min_room_area=args.min_room_area,
    )

    print(f"Комнат найдено:  {len(result['rooms'])}")
    print(f"Дверей найдено:  {len(result['doors'])}")
    print(f"Окон найдено:    {len(result['windows'])}")
    print(f"Вх. дверей:      {len(result['front_doors'])}")

    if result["rooms"]:
        print("\nКомнаты:")
        for i, r in enumerate(result["rooms"]):
            cx, cy = r["centroid"]
            print(
                f"  R{i + 1}: {r['room_type']:<15} {r['area_m2']}м²"
                f"  окон={r['windows_count']}  центр=({cx:.0f},{cy:.0f})"
            )