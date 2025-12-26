from __future__ import annotations

from shapely.geometry import Point, Polygon

from ortools.sat.python import cp_model

from app.geometry import DB, room_walls, along_wall_points, coerce_polygon, normalize_project_geometry
from app.rules import get_rules


def generate_candidates(project_id: str):
    """
    Генерация кандидатов для устройств.

    Важно:
    - Эта функция больше не падает, даже если в DB случайно лежат dict вместо Polygon:
      мы нормализуем геометрию и приводим каждую комнату через coerce_polygon().
    """
    normalize_project_geometry(project_id)

    rules = get_rules()
    rooms_raw = DB['rooms'].get(project_id, [])
    candidates = []  # (type, room_idx, point)

    for idx, room_any in enumerate(rooms_raw):
        room: Polygon = coerce_polygon(room_any)

        per_meter = rules.get('per_room_requirements', {}).get('LIVING', {}).get('socket_per_wall_meter', 0.3)
        step = 1.0 / max(per_meter, 0.2)

        for wall in room_walls(room):
            for p in along_wall_points(wall, step=step, offsets=0.3):
                candidates.append(('SOCKET', idx, p))

        # Выключатель для каждой комнаты (MVP):
        # ставим в первую вершину полигона комнаты.
        x, y = room.exterior.coords[0]
        candidates.append(('SWITCH', idx, Point(float(x), float(y))))

    return candidates


def select_devices(project_id: str, candidates):
    """
    Простейшая CP-SAT модель:
    - минимум 1 выключатель на комнату
    - максимум 6 розеток на комнату
    - минимизируем количество устройств
    """
    model = cp_model.CpModel()
    xs = []

    for i, (t, room_idx, p) in enumerate(candidates):
        v = model.NewBoolVar(f"x_{i}")
        xs.append(v)

    # Ограничения по комнатам
    room_indices = set(r for _, r, _ in candidates)
    for room_idx in room_indices:
        idxs_socket = [i for i, (t, r, _) in enumerate(candidates) if r == room_idx and t == 'SOCKET']
        idxs_switch = [i for i, (t, r, _) in enumerate(candidates) if r == room_idx and t == 'SWITCH']

        # минимум 1 выключатель
        if idxs_switch:
            model.Add(sum(xs[i] for i in idxs_switch) >= 1)

        # максимум 6 розеток
        if idxs_socket:
            model.Add(sum(xs[i] for i in idxs_socket) <= 6)

    model.Minimize(sum(xs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    res = solver.Solve(model)

    chosen = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, var in enumerate(xs):
            if solver.Value(var) == 1:
                chosen.append(candidates[i])

    DB['devices'][project_id] = chosen
    return chosen
