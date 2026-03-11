import argparse
import os
import glob
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CLASSES = ["bg", "wall", "door", "window", "front_door"]  # ids: 0..4


def _imread_gray(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        return None
    g = gray.copy()
    if float(g.mean()) < 127.0:
        g = 255 - g
    _, th_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, th_fixed = cv2.threshold(g, 230, 255, cv2.THRESH_BINARY_INV)
    th = cv2.bitwise_or(th_otsu, th_fixed)
    # medianBlur убран — он уничтожал тонкие линии
    return th

def _binarize_input(gray: np.ndarray) -> np.ndarray:
    """
    Optional: binarize the INPUT image too (to match inference --preprocess binarize).
    Returns BGR uint8.
    """
    if gray is None:
        raise ValueError("base is None")
    g = gray.copy()
    if float(g.mean()) < 127.0:
        g = 255 - g
    # binary (not inverse): make paper white, ink black
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _morph_close(mask255: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(mask255, cv2.MORPH_CLOSE, kernel, iterations=1)


def _morph_open(mask255: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(mask255, cv2.MORPH_OPEN, kernel, iterations=1)


def _dilate(mask255: np.ndarray, k: int, iters: int = 1) -> np.ndarray:
    if k <= 0:
        return mask255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask255, kernel, iterations=max(1, int(iters)))


def _tight_bbox_from_any(masks255: List[np.ndarray], pad: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Return bbox (x0,y0,x1,y1) for union of all masks. Coordinates inclusive-exclusive.
    """
    union = None
    for m in masks255:
        if m is None:
            continue
        mm = (m > 0).astype(np.uint8)
        union = mm if union is None else np.maximum(union, mm)

    if union is None:
        return None

    ys, xs = np.where(union > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(union.shape[1], x1 + pad)
    y1 = min(union.shape[0], y1 + pad)
    return (x0, y0, x1, y1)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _scan_plans(src: str) -> List[Dict[str, str]]:
    """
    Supports two layouts:
    A) src/plan_001/{input,wall,door,window,front_door}.png
    B) src/plan_001_base.png, src/plan_001_wall.png, ...
    """
    plans: List[Dict[str, str]] = []

    # Layout A: subfolders
    subdirs = [d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d)]
    for d in sorted(subdirs):
        pid = os.path.basename(d)
        cand = {
            "id": pid,
            "base": os.path.join(d, "input.png"),
            "wall": os.path.join(d, "wall.png"),
            "door": os.path.join(d, "door.png"),
            "window": os.path.join(d, "window.png"),
            "front_door": os.path.join(d, "front_door.png"),
        }
        if os.path.exists(cand["base"]) and os.path.exists(cand["wall"]):
            plans.append(cand)

    # Layout B: flat files
    base_files = sorted(glob.glob(os.path.join(src, "*_base.png")))
    for b in base_files:
        pid = os.path.basename(b).replace("_base.png", "")
        cand = {
            "id": pid,
            "base": b,
            "wall": os.path.join(src, f"{pid}_wall.png"),
            "door": os.path.join(src, f"{pid}_door.png"),
            "window": os.path.join(src, f"{pid}_window.png"),
            "front_door": os.path.join(src, f"{pid}_front_door.png"),
        }
        if os.path.exists(cand["wall"]):
            plans.append(cand)

    # Deduplicate by id (prefer subfolder layout)
    uniq: Dict[str, Dict[str, str]] = {}
    for p in plans:
        if p["id"] not in uniq:
            uniq[p["id"]] = p
    return list(uniq.values())


def _remove_hatch(wall_255: np.ndarray) -> np.ndarray:
    """
    Убирает одиночные пиксели-шум из слоя стен.
    Порог min_area=10 — убирает только совсем мелкий мусор,
    все реальные линии (даже 5px) сохраняются.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wall_255, connectivity=8
    )
    min_area = 10
    out = np.zeros_like(wall_255)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out

def _compose_mask(
        wall_gray: Optional[np.ndarray],
        door_gray: Optional[np.ndarray],
        window_gray: Optional[np.ndarray],
        front_door_gray: Optional[np.ndarray],
        door_dilate: int,
        window_dilate: int,
        front_door_dilate: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    wall = _binarize_ink(wall_gray) if wall_gray is not None else None
    door = _binarize_ink(door_gray) if door_gray is not None else None
    window = _binarize_ink(window_gray) if window_gray is not None else None
    front = _binarize_ink(front_door_gray) if front_door_gray is not None else None

    if wall is None:
        raise RuntimeError("wall layer is required")

    # ── ГЛАВНОЕ ИСПРАВЛЕНИЕ: убираем штриховку из стен ──────────────────
    wall = _remove_hatch(wall)
    # ────────────────────────────────────────────────────────────────────

    # Стены: небольшое закрытие и dilate
    wall = _fill_wall_polygon(wall)

    # Двери
    if door is not None:
        door = _dilate(door, door_dilate, iters=2)  # убрали morph_open

    # Окна
    if window is not None:
        window = _dilate(window, window_dilate, iters=2)  # убрали morph_open

    if front is not None:
        front = _dilate(front, front_door_dilate, iters=2)  # убрали morph_open

    h, w = wall.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Приоритет: wall → door/window/front перекрывают wall
    mask[wall > 0] = 1
    if door is not None:
        mask[door > 0] = 2
    if window is not None:
        mask[window > 0] = 3
    if front is not None:
        mask[front > 0] = 4

    dbg = {
        "wall": wall,
        "door": door if door is not None else np.zeros((h, w), dtype=np.uint8),
        "window": window if window is not None else np.zeros((h, w), dtype=np.uint8),
        "front_door": front if front is not None else np.zeros((h, w), dtype=np.uint8),
    }
    return mask, dbg

def _fill_wall_polygon(wall_255: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wall_255, connectivity=8
    )
    clean = np.zeros_like(wall_255)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            clean[labels == i] = 255
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close, iterations=1)  # было 2
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  # было 30
    thick = cv2.dilate(closed, k_dilate, iterations=1)
    return thick

def _tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))

    if not xs:
        xs = [0]
    if not ys:
        ys = [0]

    if xs[-1] != max(0, W - tile):
        xs.append(max(0, W - tile))
    if ys[-1] != max(0, H - tile):
        ys.append(max(0, H - tile))

    out = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(W, x0 + tile)
            y1 = min(H, y0 + tile)
            if (x1 - x0) < tile:
                x0 = max(0, x1 - tile)
            if (y1 - y0) < tile:
                y0 = max(0, y1 - tile)
            out.append((x0, y0, x0 + tile, y0 + tile))

    out = list(dict.fromkeys(out))
    return out


def _mask_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    colors = {
        1: (0, 0, 255),     # wall - red
        2: (0, 255, 255),   # door - yellow
        3: (255, 255, 0),   # window - cyan
        4: (255, 0, 255),   # front_door - magenta
    }
    over = img_bgr.copy()
    for cid, col in colors.items():
        m = (mask == cid)
        if m.any():
            over[m] = (over[m] * (1.0 - alpha) + np.array(col, dtype=np.float32) * alpha).astype(np.uint8)
    return over


def _apply_aug_variant(img: np.ndarray, mask: np.ndarray, variant: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cheap deterministic geometric variants (keeps labels aligned):
      0: identity
      1: flip LR
      2: flip UD
      3: rot90
      4: rot180
      5: rot270
    """
    if variant == 0:
        return img, mask
    if variant == 1:
        return np.flip(img, axis=1).copy(), np.flip(mask, axis=1).copy()
    if variant == 2:
        return np.flip(img, axis=0).copy(), np.flip(mask, axis=0).copy()
    if variant == 3:
        return np.rot90(img, k=1).copy(), np.rot90(mask, k=1).copy()
    if variant == 4:
        return np.rot90(img, k=2).copy(), np.rot90(mask, k=2).copy()
    if variant == 5:
        return np.rot90(img, k=3).copy(), np.rot90(mask, k=3).copy()
    return img, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with exported layer PNGs")
    ap.add_argument("--out", required=True, help="Output dataset root (images/, masks/)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--crop-to-content", action="store_true")
    ap.add_argument("--pad", type=int, default=64, help="Padding when crop-to-content")
    ap.add_argument("--seed", type=int, default=42)

    # Morph params
    ap.add_argument("--wall-close", type=int, default=11, help="Close kernel for wall mask")
    ap.add_argument("--wall-open", type=int, default=3, help="Open kernel for wall mask")
    ap.add_argument("--wall-dilate", type=int, default=5, help="Dilate kernel for wall mask (keep modest)")
    ap.add_argument("--door-dilate", type=int, default=11)
    ap.add_argument("--window-dilate", type=int, default=11)
    ap.add_argument("--front-door-dilate", type=int, default=11)

    # Filtering / balancing
    ap.add_argument("--min-nonbg", type=float, default=0.002, help="Drop tiles with too little labeled pixels")
    ap.add_argument("--keep-wall-only", type=float, default=0.10, help="Probability to keep wall-only tiles")
    ap.add_argument("--dup-minority", type=int, default=1, help="Duplicate minority tiles with geometric variants (>=1)")
    ap.add_argument("--write-debug", action="store_true")
    ap.add_argument("--debug-max", type=int, default=120)

    # Optional: binarize input images too (match inference --preprocess binarize)
    ap.add_argument("--binarize-input", action="store_true")

    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masks")
    _ensure_dir(out_images)
    _ensure_dir(out_masks)

    plans = _scan_plans(args.src)
    if not plans:
        raise RuntimeError(
            f"No plans found in {args.src}. Expect *_base.png + *_wall.png or subfolders with input.png/wall.png"
        )

    total_tiles = 0
    kept_tiles = 0
    minority_tiles = 0
    wall_only_kept = 0
    debug_written = 0

    for plan in plans:
        pid = plan["id"]
        base = _imread_gray(plan.get("base", ""))
        wall = _imread_gray(plan.get("wall", ""))
        door = _imread_gray(plan.get("door", ""))
        window = _imread_gray(plan.get("window", ""))
        front = _imread_gray(plan.get("front_door", ""))

        if base is None or wall is None:
            print(f"[skip] {pid}: missing base or wall")
            continue

        mask, dbg_masks = _compose_mask(
            wall_gray=wall,
            door_gray=door,
            window_gray=window,
            front_door_gray=front,
            door_dilate=args.door_dilate,
            window_dilate=args.window_dilate,
            front_door_dilate=args.front_door_dilate,
        )

        # Prepare input image
        if args.binarize_input:
            img_bgr = _binarize_input(base)
        else:
            g = base.copy()
            if float(g.mean()) < 127.0:
                g = 255 - g
            img_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # Optional crop
        if args.crop_to_content:
            bbox = _tight_bbox_from_any(
                [dbg_masks["wall"], dbg_masks["door"], dbg_masks["window"], dbg_masks["front_door"]],
                pad=args.pad,
            )
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                img_bgr = img_bgr[y0:y1, x0:x1]
                mask = mask[y0:y1, x0:x1]

        H, W = mask.shape[:2]
        coords = _tile_coords(H, W, args.tile, args.overlap)

        plan_total = 0
        plan_kept = 0
        plan_minority = 0
        plan_wall_only_kept = 0

        for ti, (x0, y0, x1, y1) in enumerate(coords):
            tile_img = img_bgr[y0:y1, x0:x1]
            tile_mask = mask[y0:y1, x0:x1]

            if tile_img.shape[0] != args.tile or tile_img.shape[1] != args.tile:
                pad_y = args.tile - tile_img.shape[0]
                pad_x = args.tile - tile_img.shape[1]
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
                tile_mask = cv2.copyMakeBorder(tile_mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)

            plan_total += 1
            total_tiles += 1

            nonbg = float((tile_mask > 0).mean())
            if nonbg < args.min_nonbg:
                continue

            has_minority = bool(((tile_mask == 2) | (tile_mask == 3) | (tile_mask == 4)).any())
            if has_minority:
                keep = True
                plan_minority += 1
                minority_tiles += 1
            else:
                keep = (rng.random() < float(args.keep_wall_only))
                if keep:
                    plan_wall_only_kept += 1
                    wall_only_kept += 1

            if not keep:
                continue

            # Save base tile
            fn0 = f"{pid}_{ti:04d}.png"
            cv2.imwrite(os.path.join(out_images, fn0), tile_img)
            cv2.imwrite(os.path.join(out_masks, fn0), tile_mask)
            plan_kept += 1
            kept_tiles += 1

            if args.write_debug and debug_written < args.debug_max:
                over = _mask_overlay(tile_img, tile_mask, alpha=0.55)
                cv2.imwrite(os.path.join(out_masks, fn0.replace(".png", "_overlay.png")), over)
                debug_written += 1

            # Duplicate minority tiles (geometric variants)
            if has_minority and int(args.dup_minority) > 1:
                # variants 1..5 are flips/rots; choose deterministically but shuffled
                variants = [1, 2, 3, 4, 5]
                rng.shuffle(variants)
                need = int(args.dup_minority) - 1
                for k in range(min(need, len(variants))):
                    v = variants[k]
                    aug_img, aug_mask = _apply_aug_variant(tile_img, tile_mask, v)
                    fn = f"{pid}_{ti:04d}_a{v}.png"
                    cv2.imwrite(os.path.join(out_images, fn), aug_img)
                    cv2.imwrite(os.path.join(out_masks, fn), aug_mask)
                    plan_kept += 1
                    kept_tiles += 1

                    if args.write_debug and debug_written < args.debug_max:
                        over = _mask_overlay(aug_img, aug_mask, alpha=0.55)
                        cv2.imwrite(os.path.join(out_masks, fn.replace(".png", "_overlay.png")), over)
                        debug_written += 1

        print(
            f"[ok] {pid}: tiles total={plan_total} (minority={plan_minority} wall_only_kept={plan_wall_only_kept}) saved={plan_kept}"
        )

    print("Done.")
    print(f"Output images: {out_images}")
    print(f"Output masks : {out_masks}")
    print(f"Classes      : {CLASSES}")
    print(
        f"Stats        : tiles_total={total_tiles} saved={kept_tiles} minority={minority_tiles} wall_only_kept={wall_only_kept}"
    )


if __name__ == "__main__":
    main()
