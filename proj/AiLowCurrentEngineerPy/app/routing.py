from __future__ import annotations

from typing import List, Tuple, Any, Dict
import math

import networkx as nx
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points, unary_union

from app.geometry import DB, normalize_project_geometry, coerce_polygon


def _norm_node(x: float, y: float, ndigits: int = 1) -> tuple[float, float]:
    """Нормализация координат узлов графа, чтобы уменьшить дубликаты."""
    return (round(float(x), ndigits), round(float(y), ndigits))


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_room_boundary_graph(rooms: List[Polygon]) -> nx.Graph:
    """
    Строит граф по границам всех комнат: вершины = точки контура,
    ребра = сегменты контура с весом = длина.
    """
    G = nx.Graph()
    for room in rooms:
        coords = list(room.exterior.coords)
        if len(coords) < 3:
            continue

        # coords обычно замкнуты: последняя = первая
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            n1 = _norm_node(x1, y1)
            n2 = _norm_node(x2, y2)
            w = _euclid(n1, n2)
            G.add_node(n1)
            G.add_node(n2)
            # накапливаем минимальный вес, если ребро повторяется
            if G.has_edge(n1, n2):
                G[n1][n2]["weight"] = min(G[n1][n2]["weight"], w)
            else:
                G.add_edge(n1, n2, weight=w)
    return G


def _connect_rooms_mvp(G: nx.Graph, rooms: List[Polygon]) -> None:
    """
    MVP-связность: соединяем комнаты "мостиками" между ближайшими точками границ,
    если комнаты соприкасаются или находятся достаточно близко.

    Это временная замена двери/проёма до внедрения нейронки door/window.
    """
    if len(rooms) < 2:
        return

    u = unary_union(rooms)
    minx, miny, maxx, maxy = u.bounds
    diag = math.hypot(maxx - minx, maxy - miny)

    # Порог соединения:
    # - для PNG (пиксели) обычно 10-40px, для DXF может быть 0..небольшие.
    connect_thresh = max(5.0, diag * 0.01)  # 1% диагонали, минимум 5

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            ri, rj = rooms[i], rooms[j]
            d = ri.distance(rj)
            if d > connect_thresh:
                continue

            # ближайшие точки на границах
            try:
                pi, pj = nearest_points(ri.exterior, rj.exterior)
            except Exception:
                continue

            a = _norm_node(pi.x, pi.y)
            b = _norm_node(pj.x, pj.y)

            # Добавляем "мост" с весом = расстояние * штраф (чтобы предпочитать внутри комнаты)
            w = max(0.0001, _euclid(a, b)) * 3.0  # штраф 3x
            G.add_node(a)
            G.add_node(b)
            if G.has_edge(a, b):
                G[a][b]["weight"] = min(G[a][b]["weight"], w)
            else:
                G.add_edge(a, b, weight=w)


def _pick_panel_point(rooms: List[Polygon]) -> Point:
    """
    Автовыбор щитка (panel) для MVP:
    - берём bounding box всего плана и ставим точку около minx/miny с отступом.
    Это гарантирует детерминированность и отсутствие (0.5,0.5) "в вакууме".
    """
    u = unary_union(rooms)
    minx, miny, maxx, maxy = u.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    offset = max(0.5, diag * 0.01)  # 1% диагонали, минимум 0.5
    return Point(minx + offset, miny + offset)


def _connect_point_to_graph(G: nx.Graph, p: Point) -> tuple[float, float]:
    """
    Добавляет узел точки p в граф и соединяет его с ближайшим узлом графа.
    Возвращает node-key (tuple) для этой точки.
    """
    pn = _norm_node(p.x, p.y)
    G.add_node(pn)

    # Если граф пустой — просто вернём pn
    if G.number_of_nodes() == 1:
        return pn

    # найдём ближайший существующий узел (кроме pn)
    best = None
    best_d = float("inf")
    for n in G.nodes:
        if n == pn:
            continue
        d = _euclid(pn, n)
        if d < best_d:
            best_d = d
            best = n

    if best is not None and best_d < float("inf"):
        # соединяем с ближайшим узлом
        G.add_edge(pn, best, weight=max(0.0001, best_d))
    return pn


def route_all(project_id: str) -> List[Tuple[str, LineString, float]]:
    """
    Строит маршруты от всех устройств до щитка.

    Гарантия:
    - не выбрасывает NetworkXNoPath наружу (никаких 500),
    - если топология плохая — делает fallback (прямая линия).
    """
    normalize_project_geometry(project_id)

    rooms_raw = DB.get("rooms", {}).get(project_id, [])
    rooms: List[Polygon] = [coerce_polygon(r) for r in rooms_raw if r is not None]

    devices = DB.get("devices", {}).get(project_id, [])

    if not rooms or not devices:
        DB["routes"][project_id] = []
        return []

    # 1) граф границ комнат
    G = _build_room_boundary_graph(rooms)

    # 2) соединяем комнаты (MVP)
    _connect_rooms_mvp(G, rooms)

    # 3) панель (щиток)
    panel_point = _pick_panel_point(rooms)
    panel_node = _connect_point_to_graph(G, panel_point)

    routes: List[Tuple[str, LineString, float]] = []

    for idx, (dtype, room_idx, p) in enumerate(devices):
        # p — shapely Point
        dev_point: Point = p if isinstance(p, Point) else Point(float(p[0]), float(p[1]))
        dev_node = _connect_point_to_graph(G, dev_point)

        # 4) shortest path (или fallback)
        polyline_nodes: List[tuple[float, float]]
        try:
            path = nx.shortest_path(G, dev_node, panel_node, weight="weight")
            polyline_nodes = [(float(x), float(y)) for x, y in path]
        except nx.NetworkXNoPath:
            # fallback: прямая линия
            polyline_nodes = [(float(dev_point.x), float(dev_point.y)), (float(panel_point.x), float(panel_point.y))]
        except Exception:
            polyline_nodes = [(float(dev_point.x), float(dev_point.y)), (float(panel_point.x), float(panel_point.y))]

        ls = LineString(polyline_nodes)
        routes.append((str(dtype), ls, float(ls.length)))

    DB["routes"][project_id] = routes
    return routes
