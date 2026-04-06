import sys
sys.path.insert(0, '.')
from app.nn3.infer import run_placement
import json

# Тестовый план — 3к квартира
plan_graph = {
    "projectId": "test",
    "rooms": [
        {"id": "room_0", "roomType": "living_room", "areaM2": 22.0, "isExterior": True},
        {"id": "room_1", "roomType": "bedroom",     "areaM2": 14.0, "isExterior": True},
        {"id": "room_2", "roomType": "bedroom",     "areaM2": 12.0, "isExterior": True},
        {"id": "room_3", "roomType": "kitchen",     "areaM2": 9.0,  "isExterior": True},
        {"id": "room_4", "roomType": "bathroom",    "areaM2": 4.5,  "isExterior": False},
        {"id": "room_5", "roomType": "corridor",    "areaM2": 6.0,  "isExterior": False},
    ],
    "openings": [],
    "topology": {
        "roomAdjacency": [
            {"from": "room_5", "to": "room_0"},
            {"from": "room_5", "to": "room_1"},
            {"from": "room_5", "to": "room_2"},
            {"from": "room_5", "to": "room_3"},
            {"from": "room_5", "to": "room_4"},
        ]
    }
}

# Пожелания клиента (из NN-2)
prefs_graph = {
    "version": "preferences-1.0",
    "projectId": "test",
    "rooms": [
        {"roomType": "living_room", "tv_sockets": 2, "internet_sockets": 2},
        {"roomType": "bedroom",     "internet_sockets": 1},
    ],
    "global": {"alarm": True}
}

result = run_placement(plan_graph, prefs_graph, project_id="test")

print(f"Всего устройств: {result['totalDevices']}")
print()

# Группируем по комнатам
for rd in result["roomDesigns"]:
    devices = [d for d in result["devices"] if d["roomRef"] == rd["roomId"]]
    if devices:
        print(f"{rd['roomId']} ({rd['roomType']}):")
        from collections import Counter
        counts = Counter(d["kind"] for d in devices)
        for kind, cnt in sorted(counts.items()):
            print(f"  {kind}: {cnt}")