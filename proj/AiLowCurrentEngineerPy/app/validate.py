from shapely.geometry import Point
from shapely.ops import nearest_points
from app.geometry import DB
from app.rules import get_rules


# Для MVP: проверяем отступ от дверей и углов, количество устройств на комнату


def validate_project(project_id: str):
    rules = get_rules()
    rooms = DB['rooms'][project_id]
    devices = DB['devices'][project_id]
    doors = DB.get('doors', {}).get(project_id, [])

    violations = []
    min_off = rules['min_offsets_cm']
    off_door = min_off.get('door', 15) / 100.0  # в метрах
    off_corner = min_off.get('corner', 10) / 100.0

    for idx, room in enumerate(rooms):
        # считаем углы как вершины периметра
        corners = [Point(x, y) for x, y in list(room.exterior.coords)[:-1]]
        # устройства этой комнаты (по индексу)
        ds = [(t, r, p) for (t, r, p) in devices if r == idx]
        # минимум 1 выключатель
        if sum(1 for (t, _, _) in ds if t == 'SWITCH') < rules['per_room_requirements'].get('BEDROOM', {}).get('min_switches',1):
            violations.append({"room": idx, "type": "SWITCH_MIN", "msg": "Не хватает выключателей"})
        for (t, _, p) in ds:
            # от углов
            if min(c.distance(p) for c in corners) < off_corner:
                violations.append({"room": idx, "type": "OFFSET_CORNER", "msg": f"{t} слишком близко к углу"})
            # от дверей
            if doors:
                dmin = min(d.distance(p) for d in doors)
                if dmin < off_door:
                    violations.append({"room": idx, "type": "OFFSET_DOOR", "msg": f"{t} слишком близко к двери"})
    return violations
