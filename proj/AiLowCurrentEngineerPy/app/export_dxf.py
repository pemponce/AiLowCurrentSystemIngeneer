import ezdxf
from shapely.geometry import Polygon, Point, LineString
SYMBOLS = {
    'SOCKET': {'layer': 'EL_SOCKET'},
    'SWITCH': {'layer': 'EL_SWITCH'},
}
def export_dxf(project_id: str, rooms, devices, routes, out_path: str):
    doc = ezdxf.new()
    msp = doc.modelspace()
    # Комнаты (контуры)
    for room in rooms:
        pts = list(room.exterior.coords)
        msp.add_lwpolyline(pts, dxfattribs={'layer': 'ROOMS'})
    # Устройства
    for t, room_idx, p in devices:
        msp.add_circle((p.x, p.y), 0.05, dxfattribs={'layer': SYMBOLS[t]
        ['layer']})
    # Маршруты
    for t, ls, _ in routes:
        msp.add_lwpolyline(list(ls.coords), dxfattribs={'layer':
    'EL_CABLES'})
    doc.saveas(out_path)