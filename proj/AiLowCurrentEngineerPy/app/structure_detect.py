from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from app.geometry import DB
from app.minio_client import EXPORT_BUCKET, upload_file


@dataclass(frozen=True)
class DoorArc:
    cx: float
    cy: float
    r: float
    score: float


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _scale_params(h: int, w: int) -> Dict[str, int]:
    m = min(h, w)
    base = max(3, _odd(int(m / 520)))  # 3..5..7 на больших планах
    return {
        "k_open": _odd(base),
        "k_close": _odd(base * 9),
        "k_dilate": _odd(base * 3),
    }


def build_walls_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Эвристика для CAD-подобных PNG/JPG. Возвращает бинарную маску 0/255."""
    h, w = img_bgr.shape[:2]
    p = _scale_params(h, w)

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

    thr = cv2.morphologyEx(
        thr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (p["k_open"], p["k_open"])),
        iterations=1,
    )

    thr = cv2.morphologyEx(
        thr,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (p["k_close"], p["k_close"])),
        iterations=1,
    )

    walls = cv2.dilate(
        thr,
        cv2.getStructuringElement(cv2.MORPH_RECT, (p["k_dilate"], p["k_dilate"])),
        iterations=1,
    )
    return (walls > 0).astype(np.uint8) * 255


def vectorize_walls_hough(walls_mask: np.ndarray) -> List[Dict[str, Any]]:
    """Черновая векторизация стен (для отладки/первого роутинга)."""
    h, w = walls_mask.shape[:2]
    m = min(h, w)

    edges = cv2.Canny(walls_mask, 50, 150)

    threshold = max(80, int(m * 0.10))
    min_len = max(60, int(m * 0.08))
    max_gap = max(8, int(m * 0.01))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    out: List[Dict[str, Any]] = []
    if lines is None:
        return out

    for i, (x1, y1, x2, y2) in enumerate(lines[:, 0, :].tolist()):
        out.append(
            {
                "id": f"w-{i:04d}",
                "polylinePx": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "confidence": 0.45,
            }
        )
    return out


def _coerce_rooms_polygons(project_id: str) -> List[Tuple[str, Polygon]]:
    """Поддерживаем rooms как shapely.Polygon или как dict с polygon/polygonPx."""
    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    if not rooms_raw:
        return []

    out: List[Tuple[str, Polygon]] = []

    if isinstance(rooms_raw[0], Polygon):
        for i, poly in enumerate(rooms_raw):
            out.append((f"room_{i:03d}", poly))
        return out

    if isinstance(rooms_raw[0], dict):
        for i, r in enumerate(rooms_raw):
            rid = str(r.get("id") or f"room_{i:03d}")
            pts = r.get("polygon") or r.get("polygonPx") or r.get("points")
            if not pts or not isinstance(pts, list) or len(pts) < 3:
                continue
            try:
                poly = Polygon([(float(p[0]), float(p[1])) for p in pts])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and poly.area > 1.0:
                    out.append((rid, poly))
            except Exception:
                continue
        return out

    return []


def _outer_wall_polygon(walls_mask: np.ndarray) -> Optional[Polygon]:
    """Внешний контур стен (для классификации окна/наружного проёма)."""
    cnts, _ = cv2.findContours(walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None
    pts = cnt[:, 0, :].astype(float).tolist()
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.is_valid else None


def _edge_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 60, 160)


def _circle_arc_score(edges: np.ndarray, cx: float, cy: float, r: float) -> float:
    """
    Оцениваем «насколько окружность похожа на дугу»:
    берём N сэмплов по окружности и считаем долю попаданий в edge-map.
    """
    h, w = edges.shape[:2]
    n = 64
    hit = 0
    total = 0
    for i in range(n):
        ang = (2 * math.pi * i) / n
        x = int(round(cx + r * math.cos(ang)))
        y = int(round(cy + r * math.sin(ang)))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        total += 1
        if edges[y, x] > 0:
            hit += 1
    if total == 0:
        return 0.0
    return hit / total


def detect_door_arcs(img_bgr: np.ndarray, walls_mask: np.ndarray) -> List[DoorArc]:
    """Поиск дуг дверей через HoughCircles + валидация по edge coverage."""
    if os.getenv("STRUCT_DETECT_DOOR_ARCS", "true").lower() != "true":
        return []

    h, w = img_bgr.shape[:2]
    m = min(h, w)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = _edge_map(img_bgr)

    min_r = max(10, int(m * 0.018))
    max_r = max(min_r + 5, int(m * 0.12))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(25, int(m * 0.03)),
        param1=120,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    arcs: List[DoorArc] = []
    for (cx, cy, r) in circles[0, :, :].tolist():
        x = int(round(cx))
        y = int(round(cy))
        x0 = max(0, x - 3)
        x1 = min(w, x + 4)
        y0 = max(0, y - 3)
        y1 = min(h, y + 4)
        # дуга должна быть около стены
        if walls_mask[y0:y1, x0:x1].mean() < 10:
            continue

        score = _circle_arc_score(edges, cx, cy, r)
        # «полная окружность» и шум отсекаем
        if score < 0.05 or score > 0.55:
            continue

        arcs.append(DoorArc(float(cx), float(cy), float(r), float(score)))

    uniq: Dict[Tuple[int, int, int], DoorArc] = {}
    for a in arcs:
        key = (int(a.cx // 10), int(a.cy // 10), int(a.r // 8))
        prev = uniq.get(key)
        if prev is None or a.score > prev.score:
            uniq[key] = a

    out = sorted(uniq.values(), key=lambda z: z.score, reverse=True)
    return out[:120]


def _opening_candidates_from_room_boundaries(
    rooms: List[Tuple[str, Polygon]],
    walls_mask: np.ndarray,
    step_px: float = 5.0,
    probe_r: int = 5,
    wall_thr: float = 0.28,
    gap_min_px: float = 35.0,
    gap_max_px: float = 190.0,
    boundary_eps: float = 7.0,
) -> List[Dict[str, Any]]:
    """Ищем «провалы» стены вдоль границы комнаты."""
    h, w = walls_mask.shape[:2]

    def wall_presence(x: float, y: float) -> float:
        xi = int(round(x))
        yi = int(round(y))
        x0 = max(0, xi - probe_r)
        x1 = min(w, xi + probe_r + 1)
        y0 = max(0, yi - probe_r)
        y1 = min(h, yi + probe_r + 1)
        patch = walls_mask[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(patch.mean()) / 255.0

    def find_runs(flags: List[bool]) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        n = len(flags)
        i = 0
        while i < n:
            if not flags[i]:
                i += 1
                continue
            j = i
            while j < n and flags[j]:
                j += 1
            runs.append((i, j))
            i = j
        return runs

    cands: List[Dict[str, Any]] = []
    for room_id, poly in rooms:
        line = LineString(list(poly.exterior.coords))
        L = float(line.length)
        if L < 10:
            continue

        n = max(32, int(L // step_px))
        pts = [line.interpolate((i * L) / n) for i in range(n)]
        gaps = [wall_presence(p.x, p.y) < wall_thr for p in pts]

        for a, b in find_runs(gaps):
            per_len = (b - a) * (L / n)
            if per_len < gap_min_px or per_len > gap_max_px:
                continue
            p1 = pts[a]
            p2 = pts[b - 1]
            eu = math.hypot(p2.x - p1.x, p2.y - p1.y)
            if eu < gap_min_px * 0.45:
                continue
            mid = Point((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0)

            touched: List[str] = []
            for rid2, poly2 in rooms:
                if poly2.boundary.distance(mid) < boundary_eps:
                    touched.append(rid2)

            cands.append(
                {
                    "roomRefs": touched,
                    "segmentPx": [[float(p1.x), float(p1.y)], [float(p2.x), float(p2.y)]],
                    "mid": [float(mid.x), float(mid.y)],
                    "len": float(eu),
                }
            )
    return cands


def _classify_openings(
    candidates: List[Dict[str, Any]],
    arcs: List[DoorArc],
    outer: Optional[Polygon],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    outer_thr = 10.0
    arc_tol = 12.0

    for i, c in enumerate(candidates):
        (x1, y1), (x2, y2) = c["segmentPx"]
        mx, my = c["mid"]
        mid = Point(mx, my)
        touched = c.get("roomRefs") or []

        near_arc = False
        for a in arcs:
            d = abs(mid.distance(Point(a.cx, a.cy)) - a.r)
            if d < arc_tol:
                near_arc = True
                break

        on_outer = False
        if outer is not None:
            on_outer = outer.boundary.distance(mid) < outer_thr

        if near_arc or len(touched) >= 2:
            kind = "door"
            conf = 0.55 if near_arc else 0.45
        elif on_outer:
            kind = "window"
            conf = 0.45
        else:
            kind = "door"
            conf = 0.35

        out.append(
            {
                "id": f"{kind[0]}-{i:03d}",
                "kind": kind,
                "segmentPx": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "roomRefs": touched,
                "confidence": float(conf),
            }
        )

    # дедуп (по сетке midpoint)
    uniq: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for o in out:
        (x1, y1), (x2, y2) = o["segmentPx"]
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        key = (o["kind"], int(mx // 12), int(my // 12))
        prev = uniq.get(key)
        if prev is None:
            uniq[key] = o
            continue

        def seg_len(seg: Dict[str, Any]) -> float:
            (a1, b1), (a2, b2) = seg["segmentPx"]
            return math.hypot(a2 - a1, b2 - b1)

        if seg_len(o) > seg_len(prev):
            uniq[key] = o

    final: List[Dict[str, Any]] = []
    di = 0
    wi = 0
    for o in uniq.values():
        if o["kind"] == "door":
            o["id"] = f"d-{di:03d}"
            di += 1
        else:
            o["id"] = f"win-{wi:03d}"
            wi += 1
        final.append(o)

    return final


def draw_structure_overlay(
    img_bgr: np.ndarray,
    walls_mask: np.ndarray,
    wall_segments: List[Dict[str, Any]],
    openings: List[Dict[str, Any]],
    arcs: List[DoorArc],
    alpha: float = 0.35,
    draw_arcs: bool = True,
) -> np.ndarray:
    out = img_bgr.copy()

    overlay = out.copy()
    overlay[walls_mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    for w in wall_segments[:600]:
        (x1, y1), (x2, y2) = w["polylinePx"]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

    if draw_arcs:
        for a in arcs[:80]:
            cv2.circle(out, (int(a.cx), int(a.cy)), int(a.r), (255, 0, 255), 2)

    for o in openings:
        (x1, y1), (x2, y2) = o["segmentPx"]
        color = (0, 255, 0) if o["kind"] == "door" else (255, 0, 0)
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

    return out


def detect_structure(
    project_id: str,
    image_path: str,
    src_key: Optional[str] = None,
    debug: bool = True,  # <-- ВАЖНО: добавили, чтобы /detect-structure не падал
) -> Dict[str, Any]:
    """
    Строит структуру:
      - walls mask
      - wall segments (черновые)
      - door arcs (черновые)
      - openings (door/window)
      - overlay
      - plan-graph (0.2)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    os.makedirs("/tmp/exports", exist_ok=True)

    walls = build_walls_mask(img)

    # В debug=false можно упростить/ускорить (но оставим стены всегда)
    wall_segments = vectorize_walls_hough(walls) if debug else []

    rooms = _coerce_rooms_polygons(project_id)
    outer = _outer_wall_polygon(walls)

    arcs = detect_door_arcs(img, walls) if debug else []

    candidates = _opening_candidates_from_room_boundaries(rooms, walls) if (rooms and debug) else []
    openings = _classify_openings(candidates, arcs, outer) if candidates else []

    overlay = draw_structure_overlay(
        img, walls, wall_segments, openings, arcs,
        draw_arcs=bool(debug)
    )

    walls_local = f"/tmp/exports/{project_id}_walls_mask.png"
    overlay_local = f"/tmp/exports/{project_id}_structure_overlay.png"
    cv2.imwrite(walls_local, walls)
    cv2.imwrite(overlay_local, overlay)

    walls_key = f"masks/{project_id}_walls.png"
    overlay_key = f"overlays/{project_id}_structure.png"
    walls_uri = upload_file(EXPORT_BUCKET, walls_local, walls_key)
    overlay_uri = upload_file(EXPORT_BUCKET, overlay_local, overlay_key)

    h, w = img.shape[:2]
    plan_graph: Dict[str, Any] = {
        "version": "plan-graph-0.2",
        "source": {
            "srcKey": src_key or "",
            "imageWidth": w,
            "imageHeight": h,
            "scale": {"pxPerMeter": None, "confidence": 0},
        },
        "elements": {
            "rooms": [
                {
                    "id": rid,
                    "polygon": [[float(x), float(y)] for (x, y) in list(poly.exterior.coords)],
                    "confidence": 0.6,
                }
                for (rid, poly) in rooms
            ],
            "walls": wall_segments,
            "openings": openings,
        },
        "artifacts": {
            "masks": {"wallsMaskKey": walls_key},
            "previewOverlayPngKey": overlay_key,
        },
    }

    DB.setdefault("plan_graph", {})[project_id] = plan_graph
    DB.setdefault("structure", {})[project_id] = {
        "walls_mask_key": walls_key,
        "walls_mask_uri": walls_uri,
        "overlay_key": overlay_key,
        "overlay_uri": overlay_uri,
        "openings": openings,
        "door_arcs": [a.__dict__ for a in arcs],
        "wall_segments": wall_segments,
    }

    return {
        "project_id": project_id,
        "walls_mask_key": walls_key,
        "walls_mask_uri": walls_uri,
        "structure_overlay_key": overlay_key,
        "structure_overlay_uri": overlay_uri,
        "openings": openings,
        "door_arcs": [a.__dict__ for a in arcs],
        "plan_graph": plan_graph,
    }
