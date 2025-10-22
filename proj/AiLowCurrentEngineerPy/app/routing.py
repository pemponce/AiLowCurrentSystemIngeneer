from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import networkx as nx
from app.geometry import DB
# Примем точку щита фиксированной для MVP
PANEL = Point(0.5, 0.5)
def build_corridor_graph(rooms):
    # Граф: вершины — все вершины полигонов + PANEL; рёбра — стороны комнат
    G = nx.Graph()
    for room in rooms:
        coords = list(room.exterior.coords)
        for i in range(len(coords)-1):
            a = coords[i]; b = coords[i+1]
            G.add_node(a); G.add_node(b)
            length = Point(a).distance(Point(b))
            G.add_edge(a, b, weight=length)
    G.add_node((PANEL.x, PANEL.y))
    # соединим PANEL с ближайшей вершиной
    best = None; best_d = 1e9
    for n in G.nodes:
        d = Point(n).distance(PANEL)
        if d < best_d:
            best_d, best = d, n
    if best is not None:
        G.add_edge(best, (PANEL.x, PANEL.y), weight=best_d)
    return G

def route_all(project_id: str):
    rooms = DB['rooms'][project_id]
    devices = DB['devices'][project_id]
    G = build_corridor_graph(rooms)
    routes = []
    for t, room_idx, p in devices:
        start = (p.x, p.y)
        # привязываем к ближайшей вершине графа
        closest = min(G.nodes, key=lambda n: Point(n).distance(p))
        # путь до панели
        path = nx.shortest_path(G, closest, (PANEL.x, PANEL.y),
        weight='weight')
        coords = [start] + path
        ls = LineString(coords)
        length = ls.length
        routes.append((t, ls, length))
    DB['routes'][project_id] = routes
    return routes