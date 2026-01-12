from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from shapely import wkt
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, shape
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    wkt = None
    Polygon = MultiPolygon = GeometryCollection = None
    shape = None
    unary_union = None


CLASS_NAMES = ["bg", "wall", "door", "window", "front_door"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class RasterizeConfig:
    out_size: int = 512
    margin: int = 16


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_shapely_geom(obj: Any) -> bool:
    return hasattr(obj, "geom_type") and hasattr(obj, "is_empty") and hasattr(obj, "bounds")


def _iter_records(obj: Any) -> Iterable[Dict[str, Any]]:
    if obj is None:
        return
    if isinstance(obj, dict):
        if "plans" in obj and isinstance(obj["plans"], list):
            for it in obj["plans"]:
                if isinstance(it, dict):
                    yield it
            return
        if "data" in obj and isinstance(obj["data"], list):
            for it in obj["data"]:
                if isinstance(it, dict):
                    yield it
            return
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals[: min(10, len(vals))]):
            for v in vals:
                yield v
            return
        yield obj
        return
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return


def _parse_geom(val: Any):
    if val is None:
        return None
    if _is_shapely_geom(val):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s or "EMPTY" in s.upper():
            return None
        if wkt is not None:
            try:
                return wkt.loads(s)
            except Exception:
                return None
        return None
    if isinstance(val, dict):
        if shape is not None and "type" in val and "coordinates" in val:
            try:
                return shape(val)
            except Exception:
                return None
        return None
    try:
        if Polygon is None:
            return None
        if isinstance(val, list) and len(val) >= 3:
            if isinstance(val[0], (list, tuple)) and len(val[0]) == 2 and isinstance(val[0][0], (int, float)):
                return Polygon(val)
            if isinstance(val[0], list) and len(val[0]) >= 3 and isinstance(val[0][0], (list, tuple)):
                polys = []
                for ring in val:
                    if isinstance(ring, list) and len(ring) >= 3 and isinstance(ring[0], (list, tuple)) and len(ring[0]) == 2:
                        polys.append(Polygon(ring))
                if polys:
                    return MultiPolygon(polys)
    except Exception:
        return None
    return None


def _find_subdict_with_any_keys(obj: Any, keys: List[str], max_depth: int = 4) -> Optional[Dict[str, Any]]:
    if max_depth < 0:
        return None
    if isinstance(obj, dict):
        s = set(obj.keys())
        if any(k in s for k in keys):
            return obj
        for v in obj.values():
            found = _find_subdict_with_any_keys(v, keys, max_depth=max_depth - 1)
            if found is not None:
                return found
    if isinstance(obj, list):
        for it in obj:
            found = _find_subdict_with_any_keys(it, keys, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def _collect_geoms(rec: Dict[str, Any], keys: List[str]) -> List[Any]:
    out = []
    for k in keys:
        if k in rec:
            g = _parse_geom(rec.get(k))
            if g is not None and _is_shapely_geom(g) and not g.is_empty:
                out.append(g)
    if out:
        return out
    holder = _find_subdict_with_any_keys(rec, keys, max_depth=6)
    if holder:
        for k in keys:
            if k in holder:
                g = _parse_geom(holder.get(k))
                if g is not None and _is_shapely_geom(g) and not g.is_empty:
                    out.append(g)
    return out


def _bounds_union(geoms: List[Any]) -> Optional[Tuple[float, float, float, float]]:
    if not geoms:
        return None
    if unary_union is not None:
        try:
            u = unary_union(geoms)
            if u is None or u.is_empty:
                return None
            return u.bounds
        except Exception:
            pass
    xs, ys = [], []
    for g in geoms:
        try:
            b = g.bounds
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
        except Exception:
            continue
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _world_to_px_mapper(bounds: Tuple[float, float, float, float], cfg: RasterizeConfig):
    minx, miny, maxx, maxy = bounds
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    size = int(cfg.out_size)
    margin = int(cfg.margin)
    usable = max(1, size - 2 * margin)

    scale = usable / max(w, h)

    def map_pt(x: float, y: float) -> Tuple[int, int]:
        xp = int(round((x - minx) * scale + margin))
        yp = int(round((maxy - y) * scale + margin))
        xp = max(0, min(size - 1, xp))
        yp = max(0, min(size - 1, yp))
        return xp, yp

    return map_pt


def _fill_geom(mask: np.ndarray, geom: Any, value: int, map_pt) -> None:
    if geom is None or not _is_shapely_geom(geom) or geom.is_empty:
        return

    def fill_poly(poly):
        try:
            coords = list(poly.exterior.coords)
            pts = np.array([map_pt(float(x), float(y)) for x, y in coords], dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], int(value))
        except Exception:
            return

    gt = geom.geom_type
    if gt == "Polygon":
        fill_poly(geom)
    elif gt == "MultiPolygon":
        for p in geom.geoms:
            fill_poly(p)
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            _fill_geom(mask, g, value, map_pt)


def _stable_id(rec: Dict[str, Any], fallback: str, idx: int) -> str:
    for k in ["id", "plan_id", "uid", "name"]:
        v = rec.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return f"{fallback}_{idx:05d}"


def _looks_like_record(rec: Dict[str, Any]) -> bool:
    keys = ["wall", "walls", "door", "doors", "window", "windows", "front_door", "frontDoor", "frontdoor"]
    if any(k in rec for k in keys):
        return True
    return _find_subdict_with_any_keys(rec, keys, max_depth=6) is not None


def _render_synthetic_from_mask(mask: np.ndarray) -> np.ndarray:
    # white background + black walls, gray doors/windows
    img = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    img[mask == CLASS_TO_ID["wall"]] = (0, 0, 0)
    img[mask == CLASS_TO_ID["door"]] = (60, 60, 60)
    img[mask == CLASS_TO_ID["window"]] = (160, 160, 160)
    img[mask == CLASS_TO_ID["front_door"]] = (30, 30, 30)
    return img


def _add_dimension_lines(img: np.ndarray, rng: random.Random, n: int = 8) -> None:
    h, w = img.shape[:2]
    for _ in range(n):
        x1 = rng.randint(0, w - 1)
        y1 = rng.randint(0, h - 1)
        horiz = rng.random() < 0.5
        length = rng.randint(int(0.15 * min(w, h)), int(0.6 * min(w, h)))
        if horiz:
            x2 = min(w - 1, x1 + length)
            y2 = y1
        else:
            x2 = x1
            y2 = min(h - 1, y1 + length)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
        # ticks
        cv2.line(img, (x1, y1), (x1 + 6, y1 + 6), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - 6, y2 - 6), (0, 0, 0), 1, cv2.LINE_AA)


def _add_random_text(img: np.ndarray, rng: random.Random, n: int = 10) -> None:
    h, w = img.shape[:2]
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    tokens = ["3.66", "5.01", "1.35", "3.12", "1.26", "34", "18", "0.97", "1.89"]
    for _ in range(n):
        txt = rng.choice(tokens)
        x = rng.randint(0, max(0, w - 100))
        y = rng.randint(20, h - 5)
        font = rng.choice(fonts)
        scale = rng.uniform(0.4, 0.9)
        cv2.putText(img, txt, (x, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)


def _simulate_real_plan_style(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()

    # random line thickness (erode/dilate)
    if rng.random() < 0.5:
        k = rng.choice([1, 2, 3])
        kernel = np.ones((k, k), np.uint8)
        out = cv2.erode(out, kernel, iterations=1)
    else:
        k = rng.choice([1, 2, 3])
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)

    # add dimensions + text
    if rng.random() < 0.9:
        _add_dimension_lines(out, rng, n=rng.randint(6, 12))
    if rng.random() < 0.9:
        _add_random_text(out, rng, n=rng.randint(6, 16))

    # small blur/noise
    if rng.random() < 0.8:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.8:
        noise = rng.uniform(2.0, 10.0)
        n = np.random.normal(0, noise, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)

    # slight rotation
    if rng.random() < 0.3:
        h, w = out.shape[:2]
        angle = rng.uniform(-2.0, 2.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return out


def rasterize_one(rec: Dict[str, Any], out_img_path: str, out_mask_path: str, cfg: RasterizeConfig, seed: int) -> bool:
    walls = _collect_geoms(rec, ["wall", "walls"])
    doors = _collect_geoms(rec, ["door", "doors"])
    windows = _collect_geoms(rec, ["window", "windows"])
    front_doors = _collect_geoms(rec, ["front_door", "frontDoor", "frontdoor"])

    all_geoms = walls + doors + windows + front_doors
    b = _bounds_union(all_geoms)
    if b is None:
        return False

    map_pt = _world_to_px_mapper(b, cfg)
    size = int(cfg.out_size)

    mask = np.zeros((size, size), dtype=np.uint8)

    for g in walls:
        _fill_geom(mask, g, CLASS_TO_ID["wall"], map_pt)
    for g in doors:
        _fill_geom(mask, g, CLASS_TO_ID["door"], map_pt)
    for g in windows:
        _fill_geom(mask, g, CLASS_TO_ID["window"], map_pt)
    for g in front_doors:
        _fill_geom(mask, g, CLASS_TO_ID["front_door"], map_pt)

    base_img = _render_synthetic_from_mask(mask)
    rng = random.Random(seed)
    aug_img = _simulate_real_plan_style(base_img, rng)

    cv2.imwrite(out_img_path, aug_img)
    cv2.imwrite(out_mask_path, mask)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resplan-pkl", required=True, help="Path to ResPlan.pkl")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = RasterizeConfig(out_size=int(args.size), margin=int(args.margin))

    out_images = osp.join(args.out, "images")
    out_masks = osp.join(args.out, "masks")
    _ensure_dir(out_images)
    _ensure_dir(out_masks)

    with open(args.resplan_pkl, "rb") as f:
        obj = pickle.load(f)

    ok = 0
    idx = 0
    for rec in _iter_records(obj):
        if not isinstance(rec, dict) or not _looks_like_record(rec):
            continue

        sid = _stable_id(rec, "resplan", idx)
        idx += 1

        img_path = osp.join(out_images, f"{sid}.png")
        mask_path = osp.join(out_masks, f"{sid}.png")

        if osp.exists(img_path) and osp.exists(mask_path):
            ok += 1
            if ok >= int(args.limit):
                break
            continue

        seed = int(args.seed) + idx
        if rasterize_one(rec, img_path, mask_path, cfg, seed=seed):
            ok += 1
            if ok % 200 == 0:
                print(f"ok={ok}")
            if ok >= int(args.limit):
                break

    print(f"Done. Prepared {ok} samples.")
    print(f"Images: {out_images}")
    print(f"Masks : {out_masks}")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
