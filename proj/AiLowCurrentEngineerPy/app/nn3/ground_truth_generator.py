# app/nn3/ground_truth_generator.py
"""
GroundTruthGenerator — генератор реалистичных координат устройств

Генерирует позиции согласно нормативам ГОСТ/ПУЭ/СП:
- SVT (ceiling_lights): zone-grid с отступами от стен
- RZT (power_socket): по периметру внешних стен
- DYM (smoke_detector): центроид потолка
- CO2 (co2_detector): центроид (только кухня/гостиная)
- LAN (internet_sockets): длинная стена, 10% от края
- SWI (switch): у дверей (упрощённо — короткая стена)
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from typing import List, Tuple, Dict


class GroundTruthGenerator:
    """Генератор реалистичных позиций устройств"""

    def __init__(self):
        # Константы (в пикселях для bbox 1000x1000)
        self.MARGIN_SMALL = 50  # Отступ от стен для маленьких комнат
        self.MARGIN_LARGE = 80  # Отступ от стен для больших комнат

        # Нормативы
        self.SVT_DENSITY = 1 / 16.0  # 1 светильник на 16m²
        self.RZT_DENSITY = 1 / 12.0  # 1 розетка на 12m²

        # Запреты по типам комнат
        self.NO_SMOKE = {"kitchen", "bathroom", "toilet", "balcony"}
        self.NO_CO2 = {"bedroom", "bathroom", "toilet", "corridor", "balcony"}
        self.NO_SOCKET = {"bathroom", "toilet", "balcony"}

    def generate_device_positions(
            self,
            room_polygon: List[Tuple[float, float]],
            room_type: str,
            area_m2: float
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Генерирует позиции всех типов устройств для комнаты

        Args:
            room_polygon: Список (x, y) координат вершин полигона
            room_type: Тип комнаты (bedroom, kitchen, etc)
            area_m2: Площадь комнаты в м²

        Returns:
            {
                "ceiling_lights": [(x1, y1), (x2, y2), ...],
                "power_socket": [(x1, y1), ...],
                "smoke_detector": [(x, y)] или [],
                "co2_detector": [(x, y)] или [],
                "internet_sockets": [(x, y)] или [],
                "switch": [(x, y)]
            }
        """
        positions = {}

        # Создаём Shapely polygon
        poly = Polygon(room_polygon)

        # SVT (ceiling_lights) — zone-grid
        positions["ceiling_lights"] = self._generate_svt_grid(poly, area_m2, room_type)

        # RZT (power_socket) — по периметру
        positions["power_socket"] = self._generate_rzt_perimeter(poly, area_m2, room_type)

        # DYM (smoke_detector) — центроид
        if room_type not in self.NO_SMOKE:
            centroid = self._centroid(poly)
            positions["smoke_detector"] = [centroid] if centroid else []
        else:
            positions["smoke_detector"] = []

        # CO2 (co2_detector) — центроид (только кухня/гостиная)
        if room_type in ["living_room", "kitchen"]:
            centroid = self._centroid(poly)
            positions["co2_detector"] = [centroid] if centroid else []
        else:
            positions["co2_detector"] = []

        # LAN (internet_sockets) — одна на квартиру, но генерируем для каждой комнаты
        # (в датасете потом выберем только одну)
        positions["internet_sockets"] = self._generate_lan_position(poly)

        # SWI (switch) — у дверей (упрощённо — короткая стена)
        positions["switch"] = self._generate_switch_positions(poly)

        return positions

    def _generate_svt_grid(
            self,
            polygon: Polygon,
            area_m2: float,
            room_type: str
    ) -> List[Tuple[float, float]]:
        """
        Генерация zone-grid для светильников

        Логика:
        - Санузлы/балконы: только 1 SVT в центре
        - Остальные: сетка с отступами от стен
        """
        # Санузлы — только 1 SVT
        if room_type in ("bathroom", "toilet", "balcony"):
            centroid = self._centroid(polygon)
            return [centroid] if centroid else []

        # Нормативное количество
        num_svt = max(1, int(np.ceil(area_m2 * self.SVT_DENSITY)))

        # Bbox полигона
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Динамический margin
        margin = self.MARGIN_LARGE if area_m2 > 100 else self.MARGIN_SMALL

        # Проверяем что margin не слишком большой
        if margin * 2 >= min(width, height):
            margin = min(width, height) * 0.2  # 20% от меньшей стороны

        # Равномерная сетка
        grid_size = int(np.ceil(np.sqrt(num_svt)))
        positions = []

        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= num_svt:
                    break

                # Позиция в сетке
                if grid_size > 1:
                    x = minx + margin + (width - 2 * margin) * (i + 0.5) / grid_size
                    y = miny + margin + (height - 2 * margin) * (j + 0.5) / grid_size
                else:
                    # Если только 1 светильник — в центр
                    x = minx + width / 2
                    y = miny + height / 2

                point = Point(x, y)

                # Проверка что точка внутри полигона
                if polygon.contains(point):
                    positions.append((x, y))

        # Если не набралось — добавить центроид
        if not positions:
            centroid = self._centroid(polygon)
            if centroid:
                positions.append(centroid)

        return positions

    def _generate_rzt_perimeter(
            self,
            polygon: Polygon,
            area_m2: float,
            room_type: str
    ) -> List[Tuple[float, float]]:
        """
        Генерация розеток по периметру

        Логика:
        - Запрещённые зоны: bathroom, toilet, balcony
        - Размещение: равномерно по периметру bbox
        """
        # Запрещённые зоны
        if room_type in self.NO_SOCKET:
            return []

        # Нормативное количество
        num_rzt = min(6, max(1, int(np.ceil(area_m2 * self.RZT_DENSITY))))

        # Bbox полигона
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Позиции на периметре bbox
        # Распределяем по 4 сторонам
        perimeter_points = []

        # Левая стена (вертикальная)
        for i in range(max(1, num_rzt // 4)):
            y = miny + height * (i + 1) / (num_rzt // 4 + 1)
            perimeter_points.append((minx + 10, y))  # +10px от края

        # Правая стена (вертикальная)
        for i in range(max(1, num_rzt // 4)):
            y = miny + height * (i + 1) / (num_rzt // 4 + 1)
            perimeter_points.append((maxx - 10, y))

        # Нижняя стена (горизонтальная)
        for i in range(max(1, num_rzt // 4)):
            x = minx + width * (i + 1) / (num_rzt // 4 + 1)
            perimeter_points.append((x, miny + 10))

        # Верхняя стена (горизонтальная)
        for i in range(max(1, num_rzt // 4)):
            x = minx + width * (i + 1) / (num_rzt // 4 + 1)
            perimeter_points.append((x, maxy - 10))

        # Фильтр — только точки внутри полигона
        positions = []
        for pt in perimeter_points:
            if polygon.contains(Point(pt)) and len(positions) < num_rzt:
                positions.append(pt)

        # Если не набралось — добавить точки ближе к углам
        if len(positions) < num_rzt:
            corners = [
                (minx + 20, miny + 20),
                (maxx - 20, miny + 20),
                (minx + 20, maxy - 20),
                (maxx - 20, maxy - 20)
            ]
            for corner in corners:
                if len(positions) >= num_rzt:
                    break
                if polygon.contains(Point(corner)):
                    positions.append(corner)

        return positions[:num_rzt]

    def _generate_lan_position(self, polygon: Polygon) -> List[Tuple[float, float]]:
        """
        Генерация позиции интернет-розетки

        Логика:
        - На длинной стене, 10% от края
        """
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Определяем длинную стену
        if width > height:
            # Горизонтальная стена длиннее
            x = minx + width * 0.1
            y = miny + 10  # Нижняя стена
        else:
            # Вертикальная стена длиннее
            x = minx + 10  # Левая стена
            y = miny + height * 0.1

        point = Point(x, y)
        if polygon.contains(point):
            return [(x, y)]

        # Fallback — центроид
        centroid = self._centroid(polygon)
        return [centroid] if centroid else []

    def _generate_switch_positions(self, polygon: Polygon) -> List[Tuple[float, float]]:
        """
        Генерация позиции выключателя

        Логика:
        - У дверного проёма (упрощённо — короткая стена)
        """
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Короткая стена
        if width < height:
            # Короткая — горизонтальная
            x = minx + width * 0.15
            y = miny + 10
        else:
            # Короткая — вертикальная
            x = minx + 10
            y = miny + height * 0.15

        point = Point(x, y)
        if polygon.contains(point):
            return [(x, y)]

        # Fallback — угол
        corner = (minx + 20, miny + 20)
        if polygon.contains(Point(corner)):
            return [corner]

        return []

    def _centroid(self, polygon: Polygon) -> Tuple[float, float]:
        """
        Вычисление центроида полигона (формула Грина)

        Returns:
            (x, y) или None если полигон невалидный
        """
        try:
            centroid = polygon.centroid

            # Проверяем что центроид внутри полигона
            if polygon.contains(centroid):
                return (centroid.x, centroid.y)

            # Fallback — центр bbox
            minx, miny, maxx, maxy = polygon.bounds
            center = Point((minx + maxx) / 2, (miny + maxy) / 2)

            if polygon.contains(center):
                return (center.x, center.y)

            # Fallback 2 — первая точка внутри
            coords = list(polygon.exterior.coords)
            for coord in coords:
                if polygon.contains(Point(coord)):
                    return coord

            return None
        except Exception:
            return None


def normalize_positions(
        positions: Dict[str, List[Tuple[float, float]]],
        polygon: Polygon
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Нормализация координат в [0, 1] относительно bbox комнаты

    Args:
        positions: {device_type: [(x_px, y_px), ...]}
        polygon: Shapely полигон комнаты

    Returns:
        {device_type: [(x_norm, y_norm), ...]} где x,y ∈ [0,1]
    """
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    # Защита от деления на 0
    if width < 1:
        width = 1
    if height < 1:
        height = 1

    normalized = {}

    for device_type, coords in positions.items():
        normalized[device_type] = [
            ((x - minx) / width, (y - miny) / height)
            for x, y in coords
        ]

    return normalized


# ============================================================================
# Тестирование генератора
# ============================================================================

if __name__ == "__main__":
    """
    Тест генератора на простом квадратном полигоне
    """
    print("=" * 60)
    print("ТЕСТ GroundTruthGenerator")
    print("=" * 60)
    print()

    # Создаём тестовый полигон (квадрат 500x500)
    test_polygon = [
        (100, 100),
        (600, 100),
        (600, 600),
        (100, 600)
    ]

    generator = GroundTruthGenerator()

    # Тестируем разные типы комнат
    test_cases = [
        ("living_room", 50.0),
        ("bedroom", 20.0),
        ("bathroom", 5.0),
        ("kitchen", 12.0)
    ]

    for room_type, area in test_cases:
        print(f"Комната: {room_type} ({area}m²)")
        print("-" * 60)

        positions = generator.generate_device_positions(
            test_polygon,
            room_type,
            area
        )

        for device_type, coords in positions.items():
            if coords:
                print(f"  {device_type}: {len(coords)} устройств")
                for i, (x, y) in enumerate(coords[:3]):  # Показываем первые 3
                    print(f"    [{i}] ({x:.1f}, {y:.1f})")
                if len(coords) > 3:
                    print(f"    ... и ещё {len(coords) - 3}")

        print()

    # Тест нормализации
    print("=" * 60)
    print("ТЕСТ нормализации координат")
    print("=" * 60)
    print()

    poly = Polygon(test_polygon)
    positions = generator.generate_device_positions(test_polygon, "living_room", 50.0)
    normalized = normalize_positions(positions, poly)

    print("Исходные координаты (пиксели):")
    print(f"  SVT: {positions['ceiling_lights'][:2]}")
    print()
    print("Нормализованные координаты [0,1]:")
    print(f"  SVT: {normalized['ceiling_lights'][:2]}")
    print()

    # Проверка что все координаты в [0, 1]
    all_valid = True
    for device_type, coords in normalized.items():
        for x, y in coords:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                print(f"❌ ОШИБКА: {device_type} координаты вне [0,1]: ({x}, {y})")
                all_valid = False

    if all_valid:
        print("✅ Все нормализованные координаты в пределах [0, 1]")

    print()
    print("=" * 60)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 60)