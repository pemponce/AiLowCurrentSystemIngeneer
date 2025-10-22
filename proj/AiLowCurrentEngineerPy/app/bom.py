from collections import defaultdict
from app.geometry import DB


# Грубая модель: суммируем длину всех трасс по типам устройств → общая длина кабеля
# Можно разделить по кабельным маркам в rules['routing'].


def make_bom(project_id: str):
    routes = DB['routes'][project_id]
    bom = defaultdict(float)
    for t, _, length in routes:
        # предположим один тип кабеля (витая пара) для слаботочки
        if t in ('SOCKET', 'SWITCH'):
            bom['UTP Cat5e (м)'] += length
        else:
            bom['Cable (м)'] += length
            # округлим до метров
    bom = {k: round(v, 1) for k, v in bom.items()}
    return bom
