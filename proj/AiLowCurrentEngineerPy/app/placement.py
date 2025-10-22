from app.geometry import DB, room_walls, along_wall_points
from app.rules import get_rules
from shapely.geometry import Point
from ortools.sat.python import cp_model


def generate_candidates(project_id: str):
    rules = get_rules()
    rooms = DB['rooms'][project_id]
    candidates = []  # (type, room_idx, point)
    for idx, room in enumerate(rooms):
        per_meter = rules['per_room_requirements'].get('LIVING',
                                                       {}).get('socket_per_wall_meter', 0.3)
        step = 1.0 / max(per_meter, 0.2)
        for wall in room_walls(room):
            for p in along_wall_points(wall, step=step, offsets=0.3):
                candidates.append(('SOCKET', idx, p))

    # Добавим выключатель возле одной из дверей — упростим: первая вершина
    x, y = room.exterior.coords[0]
    candidates.append(('SWITCH', idx, Point(x, y)))
    return candidates


def select_devices(project_id: str, candidates):
    # Простейшая модель: берём до N сокетов на комнату + 1 выключатель
    rules = get_rules()
    model = cp_model.CpModel()
    xs = []
    for i, (t, room_idx, p) in enumerate(candidates):
        v = model.NewBoolVar(f"x_{i}")
        xs.append(v)
    # Ограничения по комнатам
    for room_idx in set(r for _, r, _ in candidates):
        idxs_socket = [i for i, (t, r, _) in enumerate(candidates) if
                       r == room_idx and t == 'SOCKET']
        idxs_switch = [i for i, (t, r, _) in enumerate(candidates) if
                       r == room_idx and t == 'SWITCH']
        # минимум 1 выключатель
        model.Add(sum(xs[i] for i in idxs_switch) >= 1)
        # максимум 6 сокетов для MVP
        model.Add(sum(xs[i] for i in idxs_socket) <= 6)
    # Цель: минимизировать количество устройств (как заглушка стоимости)
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
