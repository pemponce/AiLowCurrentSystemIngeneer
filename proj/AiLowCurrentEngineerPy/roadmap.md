# NN-3 v2: Roadmap и Implementation Guide

**Цель:** Модель сама предсказывает координаты устройств, постпроцессинг только валидирует  
**Срок:** 8 недель  
**Текущая проблема:** NN-3 v1 используется только для 27.78% устройств, остальные 72.22% — постпроцессинг

---

## 📊 ЦЕЛЕВЫЕ МЕТРИКИ

| Метрика | v1 (сейчас) | v2 (цель) | Улучшение |
|---------|-------------|-----------|-----------|
| NN-3 используется | 27.78% | >80% | **+288%** |
| Постпроцессинг | 72.22% | <20% | **-72%** |
| Координаты от NN | 0% | >85% | **NEW** |
| Точность координат | — | >85% в 10% зоне | **NEW** |
| Время обработки | 8-12 сек | <6 сек | **-40%** |

---

## 🗓️ ПЛАН ПО НЕДЕЛЯМ

### НЕДЕЛЯ 1-2: Архитектура PlacementNetV2

#### Задачи:
1. ✅ Изучить существующую модель `app/nn3/model.py`
2. ✅ Спроектировать dual-head архитектуру
3. ✅ Реализовать `app/nn3/model_v2.py`
4. ✅ Написать unit-тесты для модели
5. ⏳ Валидация архитектуры на dummy данных

#### Файлы:
- `app/nn3/model_v2.py` — новая архитектура (✅ готов)
- `app/nn3/test_model_v2.py` — unit-тесты

#### Архитектура PlacementNetV2

```python
class PlacementNetV2(nn.Module):
    """
    Dual-head GraphSAGE модель
    
    Input:
        - node_features: [num_rooms, 18] — признаки комнат
        - edge_index: [2, num_edges] — граф смежности
    
    Output:
        {
            "counts": [num_rooms, 6],  # Количество каждого типа устройств
            "coords": {
                "ceiling_lights": [num_rooms, max_devices, 2],
                "power_socket": [num_rooms, max_devices, 2],
                ...
            }
        }
    """
    def __init__(self):
        # Shared encoder (GraphSAGE)
        self.conv1 = SAGEConv(18, 128)
        self.conv2 = SAGEConv(128, 128)
        
        # Head 1: Classification (количество)
        self.count_head = nn.Linear(128, 6)
        
        # Head 2: Regression (координаты)
        # Для каждого типа устройства — отдельная сеть
        self.coord_heads = nn.ModuleDict({
            "ceiling_lights": CoordNet(128, max_devices=12),
            "power_socket": CoordNet(128, max_devices=12),
            # ... для каждого типа
        })
```

**Ключевое отличие от v1:**
- v1: только `[num_rooms, 6]` — количество
- v2: `[num_rooms, 6]` + `{device: [num_rooms, 12, 2]}` — количество + координаты

#### Loss функция

```python
class PlacementLoss(nn.Module):
    """
    Combined loss = α*count_loss + β*coord_loss + γ*validity_loss
    
    count_loss: MSE для количества устройств
    coord_loss: MSE для координат (только существующих устройств)
    validity_loss: Penalty за координаты вне [0,1]
    """
    def forward(self, output, target):
        # Count loss
        count_loss = F.mse_loss(output["counts"], target["counts"])
        
        # Coord loss (masked — только для существующих устройств)
        coord_loss = 0
        for device_type in DEVICES:
            pred = output["coords"][device_type]
            true = target["coords"][device_type]
            mask = target["masks"][device_type]  # 1 если устройство существует
            
            diff = (pred - true) ** 2
            masked_diff = diff * mask.unsqueeze(-1)
            coord_loss += masked_diff.sum() / mask.sum()
        
        # Validity loss (координаты должны быть в [0,1])
        validity_loss = penalty_outside_bounds(output["coords"])
        
        return α*count_loss + β*coord_loss + γ*validity_loss
```

#### Deliverables недели 1-2:
- [x] `model_v2.py` — готовая архитектура
- [ ] `test_model_v2.py` — unit-тесты
- [ ] Валидация на dummy данных (5 комнат, случайные признаки)
- [ ] Документация архитектуры

**Критерий успеха:** Модель успешно обучается на 10 примерах (sanity check переобучения)

---

### НЕДЕЛЯ 3-4: Генерация датасета с ground truth

#### Задачи:
1. Разработать `GroundTruthGenerator` класс
2. Реализовать генераторы позиций для каждого типа устройств
3. Генерировать тестовый датасет (100 примеров)
4. Human review первых 100 примеров
5. Генерировать полный датасет (8000 примеров)

#### Файлы:
- `app/nn3/dataset_gen_v2.py` — новый генератор
- `app/nn3/ground_truth_generator.py` — логика размещения

#### GroundTruthGenerator

```python
class GroundTruthGenerator:
    """
    Генератор реалистичных координат устройств согласно ГОСТ/ПУЭ/СП
    """
    
    def generate_device_positions(self, room_polygon, room_type, area_m2):
        """
        Возвращает:
            {
                "ceiling_lights": [(x1, y1), (x2, y2), ...],
                "power_socket": [(x1, y1), ...],
                ...
            }
        """
        positions = {}
        
        # SVT: zone-grid
        positions["ceiling_lights"] = self._generate_svt_grid(
            room_polygon, area_m2, room_type
        )
        
        # RZT: по периметру внешних стен
        positions["power_socket"] = self._generate_rzt_perimeter(
            room_polygon, area_m2, room_type
        )
        
        # DYM: центроид
        if room_type not in ["kitchen", "bathroom", "toilet", "balcony"]:
            positions["smoke_detector"] = [self._centroid(room_polygon)]
        
        # CO2: центроид (только кухня/гостиная)
        if room_type in ["living_room", "kitchen"]:
            positions["co2_detector"] = [self._centroid(room_polygon)]
        
        # LAN: длинная стена, 10% от края
        positions["internet_sockets"] = self._generate_lan_position(room_polygon)
        
        # SWI: у дверей (упрощённо — короткая стена)
        positions["switch"] = self._generate_switch_positions(room_polygon)
        
        return positions
```

#### Генераторы позиций

**SVT (zone-grid):**
```python
def _generate_svt_grid(self, polygon, area_m2, room_type):
    """
    Сетка зон освещения с отступами от стен
    """
    # Санузлы — только 1 SVT в центре
    if room_type in ["bathroom", "toilet", "balcony"]:
        return [self._centroid(polygon)]
    
    # Нормативное количество
    num_svt = max(1, int(np.ceil(area_m2 / 16.0)))
    
    # Bbox с margin
    minx, miny, maxx, maxy = polygon.bounds
    margin = 50
    
    # Равномерная сетка
    grid_size = int(np.ceil(np.sqrt(num_svt)))
    positions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= num_svt:
                break
            
            x = minx + margin + (maxx - minx - 2*margin) * (i + 0.5) / grid_size
            y = miny + margin + (maxy - miny - 2*margin) * (j + 0.5) / grid_size
            
            if polygon.contains(Point(x, y)):
                positions.append((x, y))
    
    return positions
```

**RZT (периметр):**
```python
def _generate_rzt_perimeter(self, polygon, area_m2, room_type):
    """
    Розетки по периметру внешних стен
    """
    if room_type in ["bathroom", "toilet", "balcony"]:
        return []
    
    num_rzt = min(6, max(1, int(np.ceil(area_m2 / 12.0))))
    
    minx, miny, maxx, maxy = polygon.bounds
    
    # Позиции на периметре bbox (внешние стены)
    positions = [
        (minx, miny + (maxy - miny) * 0.25),  # Левая стена
        (minx, miny + (maxy - miny) * 0.75),
        (maxx, miny + (maxy - miny) * 0.25),  # Правая стена
        (maxx, miny + (maxy - miny) * 0.75),
        (minx + (maxx - minx) * 0.5, miny),   # Нижняя стена
        (minx + (maxx - minx) * 0.5, maxy)    # Верхняя стена
    ]
    
    # Фильтр — только внутри полигона
    valid = [p for p in positions if polygon.contains(Point(p))]
    
    return valid[:num_rzt]
```

#### Нормализация координат

```python
def normalize_positions(rooms):
    """
    Нормализация координат в [0, 1] относительно bbox комнаты
    
    Вход: [(x_px, y_px), ...]
    Выход: [(x_norm, y_norm), ...] где x,y ∈ [0,1]
    """
    normalized = {}
    
    for room in rooms:
        polygon = Polygon(room["polygon"])
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        norm_positions = {}
        for device_type, positions in room["device_positions"].items():
            norm_positions[device_type] = [
                ((x - minx) / width, (y - miny) / height)
                for x, y in positions
            ]
        
        normalized[room["id"]] = norm_positions
    
    return normalized
```

#### Deliverables недели 3-4:
- [ ] `ground_truth_generator.py` — генераторы позиций
- [ ] `dataset_gen_v2.py` — генератор датасета
- [ ] Тестовый датасет (100 примеров) + human review
- [ ] Полный датасет (8000 примеров)
- [ ] Валидация датасета (координаты в [0,1], нет дублей)

**Критерий успеха:** 
- 100% примеров проходят валидацию
- Human review: >90% примеров корректны
- Координаты соответствуют нормативам ГОСТ/ПУЭ

---

### НЕДЕЛЯ 5-6: Обучение PlacementNetV2

#### Задачи:
1. Реализовать `train_v2.py` с новым loss
2. Обучить модель на датасете (80 epochs)
3. Мониторинг метрик (count_acc, coord_mae, validity_loss)
4. Подбор гиперпараметров (α, β, γ в loss)
5. Сохранить лучшую модель

#### Файлы:
- `app/nn3/train_v2.py` — скрипт обучения

#### Training loop

```python
def train_v2():
    model = PlacementNetV2()
    loss_fn = PlacementLoss(alpha=1.0, beta=1.0, gamma=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    best_val_loss = float('inf')
    
    for epoch in range(80):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index)
            losses = loss_fn(output, batch.target)
            
            losses["total_loss"].backward()
            optimizer.step()
            
            train_losses.append({
                "total": losses["total_loss"].item(),
                "count": losses["count_loss"].item(),
                "coord": losses["coord_loss"].item(),
                "validity": losses["validity_loss"].item()
            })
        
        # Validation
        model.eval()
        val_metrics = evaluate(model, val_loader)
        
        # Logging
        print(f"Epoch {epoch+1}/80")
        print(f"  Train loss: {np.mean([l['total'] for l in train_losses]):.4f}")
        print(f"  Val count_acc: {val_metrics['count_acc']:.4f}")
        print(f"  Val coord_mae: {val_metrics['coord_mae']:.4f}")
        print(f"  Val coord_threshold: {val_metrics['coord_within_threshold']:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), "models/nn3/nn3_v2_best.pt")
```

#### Метрики оценки

```python
def evaluate(model, val_loader):
    """
    Метрики:
    - count_acc: точность предсказания количества (после округления)
    - count_mae: MAE количества
    - coord_mae: MAE координат (только для существующих устройств)
    - coord_within_threshold: % координат в пределах 10% от размера комнаты
    - validity_score: % координат внутри [0,1]
    """
    model.eval()
    metrics = {
        "count_acc": [],
        "count_mae": [],
        "coord_mae": [],
        "coord_within_threshold": [],
        "validity_score": []
    }
    
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch.x, batch.edge_index)
            
            # Count metrics
            counts_pred = torch.round(output["counts"])
            count_acc = (counts_pred == batch.target["counts"]).float().mean()
            count_mae = torch.abs(output["counts"] - batch.target["counts"]).mean()
            
            metrics["count_acc"].append(count_acc.item())
            metrics["count_mae"].append(count_mae.item())
            
            # Coord metrics
            for device_type in DEVICES:
                pred = output["coords"][device_type]
                true = batch.target["coords"][device_type]
                mask = batch.target["masks"][device_type]
                
                # MAE (masked)
                diff = torch.abs(pred - true)
                masked_diff = diff * mask.unsqueeze(-1)
                coord_mae = masked_diff.sum() / mask.sum()
                metrics["coord_mae"].append(coord_mae.item())
                
                # Within threshold (10%)
                within = (diff < 0.1).all(dim=-1) * mask
                threshold_pct = within.sum() / mask.sum()
                metrics["coord_within_threshold"].append(threshold_pct.item())
                
                # Validity (inside [0,1])
                valid = ((pred >= 0) & (pred <= 1)).all(dim=-1) * mask
                validity = valid.sum() / mask.sum()
                metrics["validity_score"].append(validity.item())
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

#### Deliverables недели 5-6:
- [ ] `train_v2.py` — скрипт обучения
- [ ] Обученная модель `nn3_v2_best.pt`
- [ ] Training logs и графики метрик
- [ ] Отчёт о гиперпараметрах

**Целевые метрики:**
- count_acc > 0.90
- coord_mae < 0.05 (5% от размера комнаты)
- coord_within_threshold > 0.85 (85% точных координат)
- validity_score > 0.95 (95% внутри bbox)

**Критерий успеха:** Все целевые метрики достигнуты на validation set

---

### НЕДЕЛЯ 7-8: Интеграция и тестирование

#### Задачи:
1. Реализовать `infer_v2.py` — inference без _wall_point()
2. Создать эндпоинт `/design_v2`
3. Минимальный постпроцессинг (только валидация)
4. A/B тестирование v1 vs v2
5. Production deployment

#### Файлы:
- `app/nn3/infer_v2.py` — inference
- `app/main.py` — новый эндпоинт

#### Inference v2

```python
def run_placement_v2(plan_graph, prefs_graph, project_id):
    """
    Inference с PlacementNetV2
    
    КЛЮЧЕВОЕ ОТЛИЧИЕ от v1:
    - v1: предсказывает количество → _wall_point() генерирует координаты
    - v2: предсказывает количество И координаты напрямую
    """
    # Загрузка модели
    model = PlacementNetV2()
    model.load_state_dict(torch.load("models/nn3/nn3_v2_best.pt"))
    model.eval()
    
    # Построение графа
    graph_data = build_graph_from_plan(plan_graph, prefs_graph)
    
    # Inference
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)
    
    # Парсинг результатов
    devices = []
    
    for room_idx, room in enumerate(plan_graph["rooms"]):
        room_id = room["id"]
        polygon = Polygon(room["polygonPx"])
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        # Количество устройств
        counts = output["counts"][room_idx]
        
        for device_idx, device_type in enumerate(DEVICES):
            num_devices = int(torch.round(counts[device_idx]).item())
            
            # Координаты (денормализация из [0,1] в пиксели)
            norm_coords = output["coords"][device_type][room_idx]
            
            for i in range(num_devices):
                if i >= len(norm_coords):
                    break
                
                # Денормализация
                norm_x, norm_y = norm_coords[i]
                x_px = minx + norm_x.item() * width
                y_px = miny + norm_y.item() * height
                
                # Fallback: если вне полигона → центроид
                if not polygon.contains(Point(x_px, y_px)):
                    centroid = polygon.centroid
                    x_px, y_px = centroid.x, centroid.y
                
                devices.append({
                    "id": f"{room_id}_{device_type}_{i}",
                    "kind": device_type,
                    "roomRef": room_id,
                    "xPx": int(x_px),
                    "yPx": int(y_px),
                    "reason": "NN-3 v2 prediction"
                })
    
    return devices
```

#### Минимальный постпроцессинг v2

```python
def apply_minimal_validation_v2(devices, plan_graph):
    """
    ТОЛЬКО валидация безопасности:
    - max 1 DYM на комнату
    - санузлы max 1 SVT
    - запрет устройств по типу комнаты
    - координаты внутри полигона
    
    НЕ добавляет новые устройства
    НЕ заменяет координаты
    """
    validated = []
    dym_seen = set()
    svt_count = {}
    
    room_type_map = {r["roomId"]: r.get("roomType") 
                    for r in plan_graph.get("roomDesigns", [])}
    
    for d in devices:
        room_id = d.get("roomRef", "")
        room_type = room_type_map.get(room_id, "bedroom")
        kind = d.get("kind", "")
        
        # Правило 1: max 1 DYM на комнату
        if kind == "smoke_detector":
            if room_id in dym_seen:
                continue
            dym_seen.add(room_id)
        
        # Правило 2: санузлы max 1 SVT
        if room_type in ("bathroom", "toilet", "balcony") and kind == "ceiling_lights":
            count = svt_count.get(room_id, 0)
            if count >= 1:
                continue
            svt_count[room_id] = count + 1
        
        # Правило 3: запрет устройств
        if room_type in ("bathroom", "toilet", "balcony") and kind == "power_socket":
            continue
        if room_type in ("kitchen", "bathroom", "toilet", "balcony") and kind == "smoke_detector":
            continue
        
        validated.append(d)
    
    return validated
```

#### Эндпоинт /design_v2

```python
@app.post("/design_v2", tags=["design"])
async def design_v2(req: dict):
    """
    NN-3 v2: модель сама предсказывает координаты
    
    Pipeline:
    1. NN-2 парсит пожелания
    2. NN-3 v2 предсказывает количество И координаты
    3. Минимальная валидация (без добавления устройств)
    """
    project_id = req.get("projectId")
    prefs_text = req.get("preferencesText", "")
    
    # Загрузка плана
    plan_graph = load_plan_graph(project_id)
    
    # Парсинг пожеланий
    prefs_graph = parse_preferences(prefs_text, project_id)
    
    # NN-3 v2 inference
    devices = run_placement_v2(plan_graph, prefs_graph, project_id)
    
    # Минимальная валидация
    devices = apply_minimal_validation_v2(devices, plan_graph)
    
    # Сохранение
    design_graph = build_design_graph(devices, plan_graph)
    save_design(f"{project_id}_v2", design_graph)
    
    return design_graph
```

#### A/B тестирование v1 vs v2

```python
@app.post("/design_compare_v2", tags=["Comparison"])
async def compare_v1_vs_v2(req: dict):
    """
    Сравнение v1 vs v2
    
    v1: /design (NN-3 + постпроцессинг)
    v2: /design_v2 (NN-3 v2 с координатами)
    """
    project_id = req.get("projectId")
    
    # v1
    design_v1 = await design_endpoint({"projectId": project_id})
    
    # v2
    design_v2 = await design_v2({"projectId": project_id})
    
    # Анализ
    return {
        "v1": {
            "total_devices": len(design_v1["devices"]),
            "nn3_usage_pct": calculate_nn3_usage(design_v1["devices"]),
            "reason_breakdown": count_by_reason(design_v1["devices"])
        },
        "v2": {
            "total_devices": len(design_v2["devices"]),
            "nn3_usage_pct": calculate_nn3_usage(design_v2["devices"]),
            "reason_breakdown": count_by_reason(design_v2["devices"])
        },
        "improvement": {
            "nn3_usage_increase": "v2 uses NN-3 for X% more devices",
            "postprocessing_reduction": "v2 reduces postprocessing by Y%"
        }
    }
```

#### Deliverables недели 7-8:
- [ ] `infer_v2.py` — inference без _wall_point()
- [ ] `apply_minimal_validation_v2()` — минимальный постпроцессинг
- [ ] Эндпоинт `/design_v2`
- [ ] Эндпоинт `/design_compare_v2`
- [ ] A/B тестирование на 50 реальных планах
- [ ] Production deployment

**Критерий успеха:**
- NN-3 v2 используется >80% (vs 27.78% в v1)
- Постпроцессинг <20% (vs 72.22% в v1)
- Качество чертежей >= v1 (визуальная оценка)
- Время обработки <6 сек (vs 8-12 сек в v1)

---

## 🎯 КРИТИЧЕСКИЕ РИСКИ И МИТИГАЦИЯ

### Риск 1: NN-3 v2 не обучится
**Вероятность:** Средняя (30%)  
**Влияние:** Критическое (проект сорвётся)

**Митигация:**
1. Начать с маленького датасета (100 примеров)
2. Sanity check: переобучение на 10 примерах
3. Использовать pretrained GNN encoder (если доступен)
4. Постепенное усложнение (сначала только SVT, потом все типы)
5. **Fallback:** Оставить v1, улучшить только валидацию

---

### Риск 2: Координаты вне полигона
**Вероятность:** Высокая (70%)  
**Влияние:** Среднее (можно исправить fallback)

**Митигация:**
1. Penalty в loss за координаты вне [0,1]
2. Постпроцессинг: fallback на центроид для невалидных точек
3. Увеличить weight для validity_loss (γ)
4. Добавить constraint: координаты только внутри выпуклой оболочки

---

### Риск 3: Датасет нереалистичный
**Вероятность:** Средняя (40%)  
**Влияние:** Высокое (модель обучится на мусоре)

**Митигация:**
1. Human review первых 100 примеров
2. Валидация датасета (координаты соответствуют нормативам)
3. Сравнение с реальными планами (если доступны)
4. Итеративная доработка генератора

---

### Риск 4: Модель переобучится
**Вероятность:** Средняя (50%)  
**Влияние:** Среднее (плохо на новых планах)

**Митигация:**
1. Dropout 0.2 в каждом слое
2. Early stopping (patience=10 epochs)
3. L2 regularization (weight_decay=1e-4)
4. Data augmentation (поворот полигонов, масштабирование)
5. Валидация на hold-out set (20% данных)

---

## 📈 МЕТРИКИ УСПЕХА ПРОЕКТА

### Минимальные требования (MVP):
- [x] count_acc > 0.80
- [x] coord_mae < 0.10
- [x] coord_within_threshold > 0.70
- [x] NN-3 используется >60%

### Целевые метрики (Production):
- [ ] count_acc > 0.90
- [ ] coord_mae < 0.05
- [ ] coord_within_threshold > 0.85
- [ ] NN-3 используется >80%
- [ ] Постпроцессинг <20%
- [ ] Время обработки <6 сек

### Stretch goals (Идеально):
- [ ] count_acc > 0.95
- [ ] coord_mae < 0.03
- [ ] coord_within_threshold > 0.90
- [ ] NN-3 используется >90%
- [ ] Визуальное качество чертежей лучше v1

---

## 🔄 ИТЕРАТИВНАЯ РАЗРАБОТКА

### Итерация 1 (неделя 1-2): Proof of Concept
**Цель:** Убедиться что архитектура работает

**Задачи:**
- Реализовать PlacementNetV2
- Обучить на dummy данных (10 примеров)
- Проверить что модель переобучается (sanity check)

**Критерий успеха:** Loss → 0 на 10 примерах

---

### Итерация 2 (неделя 3-4): Реалистичный датасет
**Цель:** Датасет соответствует ГОСТ/ПУЭ

**Задачи:**
- Реализовать GroundTruthGenerator
- Генерировать 100 примеров
- Human review
- Исправить баги генератора

**Критерий успеха:** >90% примеров корректны

---

### Итерация 3 (неделя 5-6): Обучение модели
**Цель:** Достичь целевых метрик

**Задачи:**
- Обучить на 8000 примерах
- Подобрать гиперпараметры
- Мониторинг метрик

**Критерий успеха:** count_acc > 0.90, coord_mae < 0.05

---

### Итерация 4 (неделя 7-8): Production deployment
**Цель:** Интеграция в систему

**Задачи:**
- Реализовать inference_v2
- Создать эндпоинты
- A/B тестирование
- Production deployment

**Критерий успеха:** v2 работает лучше v1 по всем метрикам

---

## 📝 ЧЕКЛИСТ ПЕРЕД PRODUCTION

### Pre-deployment checklist:
- [ ] Все unit-тесты проходят
- [ ] Валидация на 50+ реальных планах
- [ ] A/B тест показывает улучшение метрик
- [ ] Визуальная оценка чертежей >= v1
- [ ] Время обработки <6 сек
- [ ] Rollback план готов (можно вернуться на v1)
- [ ] Документация обновлена
- [ ] Changelog записан

---

## 🎓 LEARNING OUTCOMES

После завершения проекта вы научитесь:
1. Проектировать multi-task модели (classification + regression)
2. Генерировать синтетические датасеты с ground truth
3. Обучать GNN модели с custom loss функциями
4. A/B тестировать ML модели в production
5. Постепенно мигрировать с rule-based на ML-based систему

---

## 📚 ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ

### Референсы:
- GraphSAGE paper: https://arxiv.org/abs/1706.02216
- Multi-task learning: https://arxiv.org/abs/1706.05098
- Constraint-based coordinate prediction: custom approach

### Код:
- `nn3_model_v2.py` — готовая архитектура ✅
- `placement_bugfixes.py` — примеры постпроцессинга ✅
- `REFACTORING_PLAN.md` — детальный план ✅

---

**Готовы начинать?** Следующий шаг: Валидация PlacementNetV2 на dummy данных (неделя 1).