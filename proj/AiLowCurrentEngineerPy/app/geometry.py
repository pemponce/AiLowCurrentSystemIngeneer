import json

from shapely.geometry import shape, Polygon, LineString

DB = {
    # для MVP храним в памяти: project_id -> rooms/devices/routes
    'rooms': {}, 'devices': {}, 'routes': {}
}


def load_sample_rooms(project_id: str, path: str = '/app/../samples/geojson/simple_apartment.geojson'):
    with open(path, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    rooms = [shape(feat['geometry']) for feat in gj['features']]
    DB['rooms'][project_id] = rooms
    return rooms


def room_walls(room: Polygon):
    xs, ys = room.exterior.coords.xy
    return [LineString([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]) for i in
            range(len(xs) - 1)]


def along_wall_points(wall: LineString, step: float = 1.5, offsets: float = 0.2):
    # генерирует точки через step метров, с отступами от углов
    L = wall.length
    t = offsets
    pts = []
    while t < L - offsets:
        pts.append(wall.interpolate(t))
        t += step
    return pts
