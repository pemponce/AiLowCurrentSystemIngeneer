from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

# Optional dependency: shapely (recommended for ResPlan)
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


def _load_any(path: str) -> Any:
    """
    Load JSON/JSON.GZ/PKL/GPICKLE as python objects.
    """
    p = path.lower()
    try:
        if p.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if p.endswith(".json") or p.endswith(".geojson"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if p.endswith(".jsonl"):
            items = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
            return items
        if p.endswith((".pkl", ".pickle", ".gpickle")):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    except Exception:
        return None


def _iter_records(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Normalize dataset container -> stream of record dicts.
    Supports:
      - dict: a record
      - list: list of records
      - dict with key 'plans' or 'data'
      - dict keyed by id -> record
    """
    if obj is None:
        return
    if isinstance(obj, dict):
        # common container patterns
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

        # if dict looks like id->record mapping
        # (heuristic: many keys, values are dicts)
        dict_values = list(obj.values())
        if dict_values and all(isinstance(v, dict) for v in dict_values[: min(10, len(dict_values))]):
            for v in dict_values:
                yield v
            return

        # single record
        yield obj
        return

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
        return


def _parse_geom(val: Any):
    """
    Return shapely geometry or None.
    Supports:
      - shapely objects
      - WKT string
      - GeoJSON dict
      - coordinates list
    """
    if val is None:
        return None

    if _is_shapely_geom(val):
        return val

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        if "EMPTY" in s.upper():
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
            # polygon coords: [[x,y],...]
            if isinstance(val[0], (list, tuple)) and len(val[0]) == 2 and isinstance(val[0][0], (int, float)):
                return Polygon(val)
            # multipolygon-ish: [ [[x,y],...], [[x,y],...], ... ]
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
    """
    Walk nested dicts to find first dict containing any of keys.
    """
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
    # direct keys
    for k in keys:
        if k in rec:
            g = _parse_geom(rec.get(k))
            if g is None:
                continue
            if _is_shapely_geom(g) and getattr(g, "is_empty", False):
                continue
            out.append(g)

    if out:
        return out

    # nested dict search
    holder = _find_subdict_with_any_keys(rec, keys, max_depth=5)
    if holder:
        for k in keys:
            if k in holder:
                g = _parse_geom(holder.get(k))
                if g is None:
                    continue
                if _is_shapely_geom(g) and getattr(g, "is_empty", False):
                    continue
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
        yp = int(round((maxy - y) * scale + margin))  # flip y
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


def _looks_like_record(rec: Dict[str, Any]) -> bool:
    # ResPlan содержит архитектурные элементы в JSON (walls/doors/windows/...), но могут быть вложены. :contentReference[oaicite:2]{index=2}
    keys = ["wall", "walls", "door", "doors", "window", "windows", "front_door", "frontDoor", "frontdoor"]
    if any(k in rec for k in keys):
        return True
    holder = _find_subdict_with_any_keys(rec, keys, max_depth=5)
    return holder is not None


def _stable_id(rec: Dict[str, Any], fallback: str, idx: int) -> str:
    for k in ["id", "plan_id", "uid", "name"]:
        v = rec.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    return f"{fallback}_{idx:05d}"


def rasterize_one(rec: Dict[str, Any], out_img_path: str, out_mask_path: str, cfg: RasterizeConfig) -> bool:
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
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # priority: wall -> door -> window -> front_door
    for g in walls:
        _fill_geom(mask, g, CLASS_TO_ID["wall"], map_pt)
    for g in doors:
        _fill_geom(mask, g, CLASS_TO_ID["door"], map_pt)
    for g in windows:
        _fill_geom(mask, g, CLASS_TO_ID["window"], map_pt)
    for g in front_doors:
        _fill_geom(mask, g, CLASS_TO_ID["front_door"], map_pt)

    # synthetic input
    img[mask == CLASS_TO_ID["wall"]] = (0, 0, 0)
    img[mask == CLASS_TO_ID["door"]] = (60, 60, 60)
    img[mask == CLASS_TO_ID["window"]] = (160, 160, 160)
    img[mask == CLASS_TO_ID["front_door"]] = (30, 30, 30)

    cv2.imwrite(out_img_path, img)
    cv2.imwrite(out_mask_path, mask)
    return True


def iter_candidate_files(root: str) -> Iterable[str]:
    patterns = [
        os.path.join(root, "**", "*.json"),
        os.path.join(root, "**", "*.jsonl"),
        os.path.join(root, "**", "*.geojson"),
        os.path.join(root, "**", "*.json.gz"),
        os.path.join(root, "**", "*.pkl"),
        os.path.join(root, "**", "*.pickle"),
        os.path.join(root, "**", "*.gpickle"),
    ]
    seen = set()
    for pat in patterns:
        for p in glob(pat, recursive=True):
            if p not in seen:
                seen.add(p)
                yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resplan-root", required=True, help="Путь к распакованному ResPlan")
    ap.add_argument("--out", required=True, help="Куда сохранить raster dataset")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Ограничить число примеров (0=без лимита)")
    ap.add_argument("--debug", action="store_true", help="Печатать причины пропусков")
    args = ap.parse_args()

    cfg = RasterizeConfig(out_size=int(args.size), margin=int(args.margin))

    out_images = os.path.join(args.out, "images")
    out_masks = os.path.join(args.out, "masks")
    _ensure_dir(out_images)
    _ensure_dir(out_masks)

    ok = 0
    scanned_files = 0
    scanned_records = 0

    for path in iter_candidate_files(args.resplan_root):
        scanned_files += 1
        obj = _load_any(path)
        if obj is None:
            if args.debug:
                print(f"skip file (cannot load): {path}")
            continue

        base = os.path.splitext(os.path.basename(path))[0]
        rec_idx = 0

        for rec in _iter_records(obj):
            scanned_records += 1
            if not _looks_like_record(rec):
                if args.debug:
                    print(f"skip rec (no keys): file={path}")
                continue

            sid = _stable_id(rec, base, rec_idx)
            rec_idx += 1

            img_path = os.path.join(out_images, f"{sid}.png")
            mask_path = os.path.join(out_masks, f"{sid}.png")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                ok += 1
                if args.limit and ok >= int(args.limit):
                    break
                continue

            if rasterize_one(rec, img_path, mask_path, cfg):
                ok += 1
                if args.debug and ok % 100 == 0:
                    print(f"ok={ok} last={sid}")
                if args.limit and ok >= int(args.limit):
                    break
            else:
                if args.debug:
                    print(f"skip rec (no bounds/geoms): sid={sid} file={path}")

        if args.limit and ok >= int(args.limit):
            break

    print(f"Done. Prepared {ok} samples.")
    print(f"Scanned files   : {scanned_files}")
    print(f"Scanned records : {scanned_records}")
    print(f"Images: {out_images}")
    print(f"Masks : {out_masks}")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
