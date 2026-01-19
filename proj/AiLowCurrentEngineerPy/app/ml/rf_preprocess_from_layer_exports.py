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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale to binary 'ink' mask.
    We assume drawings are mostly black on white, but we handle both by Otsu + heuristic.
    Returns uint8 mask {0,255} where 255=ink.
    """
    if gray is None:
        return None
    g = gray.copy()

    # Heuristic: if background is dark (mean < 127), invert for consistency
    if float(g.mean()) < 127.0:
        g = 255 - g

    # Otsu threshold: ink will be black-ish => after invert, ink is dark => threshold on inverse
    # We want ink as 255, so do binary inverse on g (white bg)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove tiny noise
    th = cv2.medianBlur(th, 3)
    return th


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
    Return bbox (x0,y0,x1,y1) for union of all masks. Coordinates are inclusive-exclusive.
    """
    union = None
    for m in masks255:
        if m is None:
            continue
        if union is None:
            union = (m > 0).astype(np.uint8)
        else:
            union = np.maximum(union, (m > 0).astype(np.uint8))
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

    A) src/plan_001/{base,wall,door,window,front_door}.png
    B) src/plan_001_base.png, src/plan_001_wall.png, ...

    Returns list of dicts:
      {"id": "plan_001", "base": "...", "wall": "...", ...}
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


def _compose_mask(
    wall_gray: Optional[np.ndarray],
    door_gray: Optional[np.ndarray],
    window_gray: Optional[np.ndarray],
    front_door_gray: Optional[np.ndarray],
    wall_close: int,
    wall_open: int,
    wall_dilate: int,
    door_dilate: int,
    window_dilate: int,
    front_door_dilate: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      mask uint8 with class ids
      debug dict of per-class masks {name: mask255}
    """
    wall = _binarize_ink(wall_gray) if wall_gray is not None else None
    door = _binarize_ink(door_gray) if door_gray is not None else None
    window = _binarize_ink(window_gray) if window_gray is not None else None
    front = _binarize_ink(front_door_gray) if front_door_gray is not None else None

    # Make walls more learnable: close holes (hatches), remove speckle, then dilate
    if wall is None:
        raise RuntimeError("wall layer is required")
    wall = _morph_close(wall, wall_close)
    wall = _morph_open(wall, wall_open)
    wall = _dilate(wall, wall_dilate, iters=1)

    if door is not None:
        door = _morph_open(door, 3)
        door = _dilate(door, door_dilate, iters=1)
    if window is not None:
        window = _morph_open(window, 3)
        window = _dilate(window, window_dilate, iters=1)
    if front is not None:
        front = _morph_open(front, 3)
        front = _dilate(front, front_door_dilate, iters=1)

    h, w = wall.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
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


def _tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
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
            # If at border and smaller than tile: pad by shifting start
            if (x1 - x0) < tile:
                x0 = max(0, x1 - tile)
            if (y1 - y0) < tile:
                y0 = max(0, y1 - tile)
            out.append((x0, y0, x0 + tile, y0 + tile))
    # Unique
    out = list(dict.fromkeys(out))
    return out


def _mask_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple visualization. Colors are fixed for debug only.
    """
    colors = {
        1: (0, 0, 255),     # wall - red
        2: (0, 255, 255),   # door - yellow
        3: (255, 255, 0),   # window - cyan
        4: (255, 0, 255),   # front_door - magenta
    }
    over = img_bgr.copy()
    for cid, col in colors.items():
        m = (mask == cid)
        over[m] = (over[m] * (1.0 - alpha) + np.array(col, dtype=np.float32) * alpha).astype(np.uint8)
    return over


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with exported layer PNGs")
    ap.add_argument("--out", required=True, help="Output dataset root (images/, masks/)")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--crop-to-content", action="store_true")
    ap.add_argument("--pad", type=int, default=32, help="Padding when crop-to-content")
    ap.add_argument("--seed", type=int, default=42)

    # Morph params
    ap.add_argument("--wall-close", type=int, default=9, help="Close kernel for wall mask")
    ap.add_argument("--wall-open", type=int, default=3, help="Open kernel for wall mask")
    ap.add_argument("--wall-dilate", type=int, default=7, help="Dilate kernel for wall mask")
    ap.add_argument("--door-dilate", type=int, default=5)
    ap.add_argument("--window-dilate", type=int, default=5)
    ap.add_argument("--front-door-dilate", type=int, default=5)

    # Filtering / balancing
    ap.add_argument("--min-nonbg", type=float, default=0.002, help="Drop tiles with too little labeled pixels")
    ap.add_argument("--keep-wall-only", type=float, default=0.35, help="Probability to keep wall-only tiles")
    ap.add_argument("--write-debug", action="store_true")
    ap.add_argument("--debug-max", type=int, default=80)

    args = ap.parse_args()
    random.seed(args.seed)

    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masks")
    _ensure_dir(out_images)
    _ensure_dir(out_masks)

    plans = _scan_plans(args.src)
    if not plans:
        raise RuntimeError(f"No plans found in {args.src}. Expect *_base.png + *_wall.png or subfolders with base.png/wall.png")

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
            wall_close=args.wall_close,
            wall_open=args.wall_open,
            wall_dilate=args.wall_dilate,
            door_dilate=args.door_dilate,
            window_dilate=args.window_dilate,
            front_door_dilate=args.front_door_dilate,
        )

        # Prepare input image (use base as grayscale -> 3ch for consistency)
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
                # pad to tile size (rare at borders)
                pad_y = args.tile - tile_img.shape[0]
                pad_x = args.tile - tile_img.shape[1]
                tile_img = cv2.copyMakeBorder(tile_img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(255, 255, 255))
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
                # wall-only tile: keep with probability
                keep = (random.random() < float(args.keep_wall_only))
                if keep:
                    plan_wall_only_kept += 1
                    wall_only_kept += 1

            if not keep:
                continue

            fn = f"{pid}_{ti:04d}.png"
            cv2.imwrite(os.path.join(out_images, fn), tile_img)
            cv2.imwrite(os.path.join(out_masks, fn), tile_mask)

            if args.write_debug and debug_written < args.debug_max:
                over = _mask_overlay(tile_img, tile_mask, alpha=0.55)
                cv2.imwrite(os.path.join(out_masks, fn.replace(".png", "_overlay.png")), over)
                debug_written += 1

            plan_kept += 1
            kept_tiles += 1

        print(f"[ok] {pid}: tiles total={plan_total} (minority={plan_minority} wall_only_kept={plan_wall_only_kept}) saved={plan_kept}")

    print("Done.")
    print(f"Output images: {out_images}")
    print(f"Output masks : {out_masks}")
    print(f"Classes      : {CLASSES}")
    print(f"Stats        : tiles_total={total_tiles} saved={kept_tiles} minority={minority_tiles} wall_only_kept={wall_only_kept}")


if __name__ == "__main__":
    main()
