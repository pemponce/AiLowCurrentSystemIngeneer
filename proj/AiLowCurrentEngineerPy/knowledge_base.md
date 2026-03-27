# База знаний: AI Low-Current Engineer

**Система автоматизированного проектирования слаботочных систем**  
Версия документа: 1.0  
Дата: 2025-03-25

---

## 1. ОБЩАЯ АРХИТЕКТУРА

### 1.1 Назначение системы
Автоматическая расстановка устройств слаботочных систем (освещение, розетки, датчики) на плане квартиры согласно ГОСТ/СП/ПУЭ.

**Входные данные:** PNG план квартиры  
**Выходные данные:** Чертёж с устройствами (PNG/PDF/DXF)

### 1.2 Pipeline обработки
```
PNG план → NN-1 сегментация → NN-2 парсинг пожеланий → NN-3 размещение → Постпроцессинг → Экспорт
```

### 1.3 Стек технологий
- **Backend:** FastAPI (Python), Docker
- **Storage:** MinIO (S3-совместимое хранилище), SQLite
- **ML:** PyTorch, DeepLabV3 (NN-1), BiLSTM+CRF (NN-2), GraphSAGE (NN-3)
- **Геометрия:** Shapely, OpenCV

---

## 2. КОМПОНЕНТЫ СИСТЕМЫ

### 2.1 NN-1: Сегментация плана (ml/structure_infer.py)
**Модель:** DeepLabV3 ResNet-50  
**Задача:** Разделение плана на комнаты  
**Метрики:** val_iou=0.94  
**Модель:** `models/structure_rf_v4_bestreal.pt`

**Вход:** PNG изображение плана  
**Выход:** Маска сегментации + список комнат с полигонами

**Ключевые функции:**
- `_nn1_get_rooms(image_path, project_id)` — запуск сегментации
- `_nn1_extract_geometry()` — извлечение полигонов из маски

**Известные проблемы:**
- Не всегда делит квартиру на отдельные комнаты
- Большая комната (134m²) может объединять гостиную + прихожую в один полигон

---

### 2.2 NN-2: Парсинг пожеланий (nn2/infer.py)
**Модель:** BiLSTM + CRF (NER)  
**Задача:** Извлечение требований клиента из текста  
**Метрики:** F1=1.0  
**Модель:** `models/nn2/nn2_best.pt`

**Вход:** Текст вида `"1: свет розетки; 2: свет розетки датчик дыма"`  
**Выход:** `PreferencesGraph` — структура `{room_id: {device: count}}`

**Поддерживаемые запросы:**
- "2 ночника" → `night_lights: 2`
- "2-4 источника света" → `ceiling_lights: 3` (среднее)
- "телевизор" → `tv_sockets: 1`

**Парсер:** `_parse_numbered_preferences()` в `main.py`

---

### 2.3 NN-3: Размещение устройств (nn3/)

#### Архитектура модели (nn3/model.py)
**Модель:** GraphSAGE (2 слоя GNN)  
**Метрики:** val_acc=0.892  
**Модель:** `models/nn3/nn3_best.pt`

**Входные признаки узла (комнаты):**
- One-hot тип комнаты (6 типов)
- Площадь (нормализованная)
- Количество окон/дверей
- Флаг внешней стены
- Пожелания клиента (6 устройств)
**Всего:** 18 признаков

**Выход модели:**
- Для каждой комнаты: количество устройств каждого типа
- **НЕ координаты!** Только `{room_id: {device: count}}`

**Константы:**
- `MAX_ROOMS = 12` — максимум комнат в квартире
- `MAX_DEVICE = 12` — максимум устройств одного типа
- `DEVICES = ["ceiling_lights", "power_socket", "smoke_detector", "co2_detector", "internet_sockets", "tv_sockets"]`

#### Inference (nn3/infer.py)
**Ключевая функция:** `run_placement(plan_graph, prefs_graph, project_id)`

**Генерация координат:**
- `_wall_point(kind, poly, cx, cy, offset, n_device)` — размещение на стенах/потолке
- SVT (ceiling_lights): сетка по bbox с отступами
- RZT (power_socket): только внешние стены, позиции 0.25/0.75/0.5/0.15/0.85
- DYM (smoke_detector): центроид потолка
- CO2: центроид потолка
- LAN (internet_sockets): длинная стена, позиция 0.1
- SWI (switch): НЕТ в NN-3 (добавляется постпроцессингом)

**КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ:**
NN-3 в текущей архитектуре **НЕ учится размещать устройства** — она только предсказывает количество. Координаты генерируются функцией `_wall_point()` по жёстким правилам.

---

### 2.4 Постпроцессинг (placement.py)

#### Функция `_apply_hard_rules()`
**Задача:** Применение нормативных правил ГОСТ/ПУЭ/СП

**Что делает:**
1. **Заменяет все SVT от NN-3** на zone-grid (`_svt_grid_positions()`)
2. **Добавляет нормативные RZT** (1 на 12m², max 6 на комнату)
3. **Добавляет SWI** у каждого дверного проёма
4. **Фильтрует устройства** по типу комнаты
5. **Дедупликация** (max 1 DYM на комнату, max 1 SVT в санузле)

**Жёсткие правила:**
```python
NO_SMOKE = {"kitchen", "bathroom", "toilet", "balcony"}  # Датчик дыма запрещён
NO_CO2 = {"bedroom", "bathroom", "toilet", "corridor", "balcony"}  # CO2 только кухня
NO_SOCKET = {"bathroom", "toilet", "balcony"}  # Розетки запрещены
ONLY_LIGHT = {"bathroom", "toilet", "balcony"}  # Только свет, ничего другого
```

**Проблема:**
Постпроцессинг **полностью перезаписывает** результат NN-3 для SVT/RZT/SWI. NN-3 фактически игнорируется.

---

## 3. УСТРОЙСТВА И НОРМАТИВЫ

### 3.1 Типы устройств

| Код | kind | Русское название | Монтаж | Высота |
|-----|------|------------------|--------|--------|
| SVT | ceiling_lights | Светильник | потолок | 0 |
| RZT | power_socket | Розетка | стена | 300mm |
| DYM | smoke_detector | Датчик дыма | потолок | 0 |
| CO2 | co2_detector | Датчик CO2 | потолок | 0 |
| LAN | internet_sockets | Интернет-розетка | стена | 300mm |
| SWI | switch | Выключатель | стена | 900mm |

### 3.2 Нормативы размещения

#### SVT (Светильники)
- **Норма:** 1 на 16m²
- **Размещение:** Сетка зон, отступ от стен 50-80px
- **Санузел/туалет:** max 1 SVT
- **Балкон:** max 1 SVT

#### RZT (Розетки)
- **Норма:** 1 на 12m², max 6 на комнату
- **Размещение:** Только внешние стены (периметр bbox)
- **Запрещено:** bathroom, toilet, balcony

#### DYM (Датчик дыма)
- **Норма:** max 1 на комнату
- **Размещение:** Центроид потолка (формула Грина)
- **Запрещено:** kitchen, bathroom, toilet, balcony

#### CO2 (Датчик CO2)
- **Разрешено:** Только living_room и kitchen
- **Размещение:** Центроид потолка

#### LAN (Интернет)
- **Норма:** Строго 1 на квартиру
- **Приоритет комнаты:** corridor > living_room
- **Forced-запрос не обходит ограничение**

#### SWI (Выключатели)
- **Норма:** У каждого дверного проёма
- **Размещение:** Offset 25px вдоль стены от проёма
- **Высота:** 900mm
- **Fallback:** Кратчайшее ребро полигона (если проёмы не найдены)

### 3.3 Специальные правила

**Bathroom/Toilet:**
- Только 1 SVT
- Никаких RZT/DYM/CO2

**Balcony:**
- Только 1 SVT
- Никаких RZT/DYM

**Kitchen (маленькая < 10m²):**
- Переклассифицируется в balcony

**L-образные комнаты:**
- SVT: zone-grid может давать позиции близко к стенам
- Fallback: `_nearest_interior_point()` из `export_overlay_png.py`

---

## 4. API ЭНДПОИНТЫ

### 4.1 Основной pipeline

#### POST /upload
**Задача:** Загрузка PNG плана в MinIO  
**Вход:** `{projectId, imageBase64}`  
**Выход:** `{srcKey: "raw_plans/input.png"}`

#### POST /ingest
**Задача:** NN-1 сегментация плана → комнаты с полигонами  
**Вход:** `{projectId, srcKey}`  
**Выход:** `PlanGraph` — список комнат с типами и polygonPx

#### POST /design
**Задача:** NN-2 + NN-3 + постпроцессинг → размещение устройств  
**Вход:** `{projectId, preferencesText?}`  
**Выход:** `DesignGraph` — устройства с координатами

**Pipeline:**
1. NN-2 парсит `preferencesText` → `PreferencesGraph`
2. NN-3 предсказывает количество устройств + генерирует координаты
3. `_apply_hard_rules()` **заменяет** SVT/RZT/SWI по нормативам
4. Сохраняет в DB

#### POST /design_nn3
**Задача:** Чистый NN-3 БЕЗ постпроцессинга  
**Вход:** `{projectId, preferencesText?}`  
**Выход:** `DesignGraph` — только результат NN-3

**Отличия от /design:**
- НЕ заменяет SVT на zone-grid
- НЕ добавляет RZT по нормативу
- НЕ добавляет SWI у дверей
- Только минимальная валидация (дедупликация, санузел max 1 SVT)

**Валидация:** `_validate_nn3_output()`

#### POST /export
**Задача:** Экспорт чертежа  
**Вход:** `{projectId, formats: ["PNG"|"PDF"|"DXF"]}`  
**Выход:** Ссылки на файлы в MinIO

### 4.2 Вспомогательные эндпоинты

#### GET /projects
Список всех проектов из SQLite

#### GET /debug/db/{project_id}
Показывает что лежит в DB для проекта

#### GET /health
Статус сервиса

---

## 5. СТРУКТУРА ДАННЫХ

### 5.1 PlanGraph (после /ingest)
```json
{
  "projectId": "plan001",
  "rooms": [
    {
      "id": "room_000",
      "roomType": "living_room",
      "areaM2": 134.0,
      "polygonPx": [[x1,y1], [x2,y2], ...],
      "centroidPx": [cx, cy],
      "isExterior": true
    }
  ],
  "openings": [],
  "topology": {
    "roomAdjacency": [{"from": "room_000", "to": "room_001"}]
  }
}
```

### 5.2 PreferencesGraph (после NN-2)
```json
{
  "version": "preferences-1.0",
  "sourceText": "1: свет розетки; 2: датчик дыма",
  "rooms": [
    {
      "roomId": "room_000",
      "devices": {
        "ceiling_lights": 1,
        "power_socket": 1
      }
    }
  ],
  "_by_room_id": {
    "room_000": {"ceiling_lights": 1, "power_socket": 1}
  }
}
```

### 5.3 DesignGraph (после /design)
```json
{
  "version": "design-1.0",
  "projectId": "plan001",
  "devices": [
    {
      "id": "room_000_ceiling_lights_zone_0",
      "kind": "ceiling_lights",
      "roomRef": "room_000",
      "mount": "ceiling",
      "heightMm": 0,
      "label": "Ceiling Lights",
      "reason": "zone: 4 SVT for 134m²",
      "xPx": 314,
      "yPx": 391
    }
  ],
  "roomDesigns": [
    {
      "roomId": "room_000",
      "roomType": "living_room",
      "deviceIds": ["room_000_ceiling_lights_zone_0", ...],
      "violations": []
    }
  ],
  "totalDevices": 36,
  "explain": ["Размещение выполнено NN-3", "Постпроцессинг применён"]
}
```

**Поле `reason`:**
- `"NN-3 prediction"` — устройство от нейросети
- `"zone: N SVT for Xm²"` — zone-grid светильники
- `"norm: N RZT for Xm²"` — нормативные розетки
- `"user request"` — forced-устройства из пожеланий
- `"rule: SWI at door"` — выключатели у дверей

---

## 6. ФАЙЛОВАЯ СТРУКТУРА

```
app/
├── main.py                  # FastAPI эндпоинты (1900+ строк)
│                            # СОДЕРЖИТ дублированные функции из других модулей
│
├── placement.py             # Постпроцессинг нормативных правил
│   ├── _apply_hard_rules()       # Фильтрация и коррекция устройств
│   ├── _svt_grid_positions()     # Сетка зон SVT по полигону
│   └── _point_in_polygon()       # Ray casting
│
├── geometry.py              # Shapely-утилиты
│   ├── detect_doorways()         # Поиск дверных проёмов
│   └── coerce_polygon()          # Коррекция полигонов
│
├── export_overlay_png.py    # Рендер PNG чертежа
│   ├── _build_lighting_zones()   # Сетка SVT внутри полигона
│   └── _nearest_interior_point() # Fallback для L-комнат
│
├── rules.py                 # Нормативные константы (ГОСТ/ПУЭ/СП)
├── db.py                    # SQLite персистентность
├── minio_client.py          # MinIO (S3-совместимое хранилище)
│
├── nn1/ (ml/)               # NN-1: сегментация плана
│   └── structure_infer.py        # DeepLabV3, val_iou=0.94
│
├── nn2/                     # NN-2: парсинг пожеланий
│   └── infer.py                  # BiLSTM+CRF, F1=1.0
│
└── nn3/                     # NN-3: размещение устройств
    ├── infer.py                  # Inference, val_acc=0.892
    │   └── _wall_point()         # Генерация координат по правилам
    ├── model.py                  # GraphSAGE архитектура
    ├── dataset_gen.py            # Генератор датасета
    └── train.py                  # Обучение модели
```

---

## 7. ИЗВЕСТНЫЕ ПРОБЛЕМЫ И РЕШЕНИЯ

### 7.1 Room_009 (toilet): 2× SVT вместо 1
**Причина:** Фильтр в `placement.py` строка 163 проверял только `balcony`, но не `bathroom/toilet`

**Решение:**
```python
# Строка 163 placement.py
if rtype in ("balcony", "bathroom", "toilet") and kind == "ceiling_lights":
    already = sum(1 for d in filtered_devices
                  if d.get("kind") == "ceiling_lights"
                  and (d.get("roomRef") == room_id or d.get("room_id") == room_id))
    if already >= 1:
        continue
```

### 7.2 Room_000: 2× DYM в одной точке
**Причина:** NN-3 может вернуть несколько DYM с одинаковыми координатами

**Решение:** Дедупликация в `placement.py` перед строкой 340:
```python
# Дедупликация: max 1 DYM на комнату
dym_seen = set()
deduped = []
for d in filtered_devices:
    if d.get("kind") == "smoke_detector":
        rid = d.get("roomRef") or d.get("room_id", "")
        if rid in dym_seen:
            continue
        dym_seen.add(rid)
    deduped.append(d)
filtered_devices = deduped
```

### 7.3 Room_000 (134m²): SVT слишком близко к стенам
**Причина:** `_build_lighting_zones()` использует bbox-разбиение для L-комнат, недостаточный margin

**Возможное решение:**
```python
# placement.py, функция _svt_grid_positions()
margin = 80 if area_m2 > 100 else 50  # Увеличенный отступ для больших комнат
```

### 7.4 NN-3 не учится размещать устройства
**Проблема:** NN-3 предсказывает только количество, координаты генерируются `_wall_point()` по правилам

**Решение:** Вариант 1 (архитектурное изменение)
1. Добавить regression head в модель для координат
2. Генерировать датасет с координатами
3. Убрать `_wall_point()` из inference
4. Минимальный постпроцессинг — только валидация

**Временное решение:** Эндпоинт `/design_nn3` — чистый NN-3 без постпроцессинга

---

## 8. DEVELOPMENT WORKFLOW

### 8.1 Запуск системы
```powershell
cd "C:\Users\azhel\Desktop\Ai ingeneer low-current systems\proj"
docker compose up -d
```

### 8.2 Деплой изменений
```powershell
.\deploy.ps1  # Копирует app/*.py и app/nn3/*.py в контейнер
```

**Если файл не применяется:**
```powershell
docker cp app\placement.py proj-planner-1:/app/app/placement.py
docker restart proj-planner-1
```

### 8.3 Тестирование
**Полный цикл:**
```powershell
# 1. Загрузить план
$bytes = [System.IO.File]::ReadAllBytes("C:\путь\план.png")
$b64   = [Convert]::ToBase64String($bytes)
$body  = @{ projectId="plan002"; imageBase64=$b64 } | ConvertTo-Json
curl -X POST http://localhost:8000/upload -H "Content-Type: application/json" -d $body

# 2. Сегментация
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"projectId\":\"plan002\",\"srcKey\":\"raw_plans/input.png\"}"

# 3. Размещение (с постпроцессингом)
curl -X POST http://localhost:8000/design -H "Content-Type: application/json" -d "{\"projectId\":\"plan002\",\"preferencesText\":\"1: свет розетки\"}"

# 3b. Размещение (чистый NN-3)
curl -X POST http://localhost:8000/design_nn3 -H "Content-Type: application/json" -d "{\"projectId\":\"plan002\",\"preferencesText\":\"\"}"

# 4. Экспорт
curl -X POST http://localhost:8000/export -H "Content-Type: application/json" -d "{\"projectId\":\"plan002\",\"formats\":[\"PNG\"]}"
```

**ВАЖНО:** Каждый тест — новый `projectId` (plan002, plan003...), иначе отдаётся кэш из SQLite.

### 8.4 Переобучение NN-3
```powershell
# Генерация датасета
docker exec proj-planner-1 python -m app.nn3.dataset_gen --out /data/nn3_dataset --count 8000

# Обучение
docker exec -it proj-planner-1 python -m app.nn3.train --data /data/nn3_dataset --out /app/models/nn3 --epochs 80
```

---

## 9. BEST PRACTICES

### 9.1 Формат отчёта о проблемах
При обнаружении проблемы скидывать:
1. **Скриншот плана** (PNG с устройствами)
2. **JSON фрагмент проблемной комнаты:**
```json
{
  "roomId": "room_009",
  "roomType": "toilet",
  "deviceIds": ["room_009_ceiling_lights_0", "room_009_ceiling_lights_1"],
  "violations": []
}
```
3. **Одной строкой:** "room_009 toilet: 2× SVT вместо 1"

### 9.2 Изменение кода
1. **Сначала placement.py** — изменить логику постпроцессинга
2. **Проверить main.py** — нет ли дублированной копии функции
3. **Деплой:** `.\deploy.ps1` или `docker cp`
4. **Тест:** новый projectId

### 9.3 Дебаг
- `GET /debug/db/{project_id}` — смотреть что в БД
- `GET /projects` — список всех проектов
- Логи: `docker logs proj-planner-1`

---

## 10. ROADMAP

### 10.1 Краткосрочные задачи
- [x] Исправить фильтр санузлов (max 1 SVT)
- [x] Дедупликация DYM
- [x] Создать эндпоинт `/design_nn3` (чистый NN-3)
- [ ] Улучшить margin для SVT в больших комнатах
- [ ] Добавить тесты для постпроцессинга

### 10.2 Среднесрочные задачи
- [ ] Переобучить NN-3 с regression head для координат
- [ ] Генерировать датасет с правильными координатами
- [ ] Убрать `_wall_point()` — NN-3 должна сама предсказывать позиции
- [ ] Минимизировать постпроцессинг до валидации

### 10.3 Долгосрочные задачи
- [ ] Улучшить NN-1 для разделения L-комнат
- [ ] Добавить поддержку нестандартных планировок
- [ ] Web UI для интерактивной корректировки
- [ ] Автоматический расчёт BOM (спецификация материалов)

---

## 11. ГЛОССАРИЙ

**NN-1** — DeepLabV3 модель сегментации плана на комнаты  
**NN-2** — BiLSTM+CRF модель парсинга текстовых пожеланий  
**NN-3** — GraphSAGE модель размещения устройств  
**PlanGraph** — структура данных с комнатами и полигонами  
**PreferencesGraph** — структура данных с пожеланиями клиента  
**DesignGraph** — структура данных с размещёнными устройствами  
**SVT** — Светильник (ceiling_lights)  
**RZT** — Розетка (power_socket)  
**DYM** — Датчик дыма (smoke_detector)  
**CO2** — Датчик CO2 (co2_detector)  
**LAN** — Интернет-розетка (internet_sockets)  
**SWI** — Выключатель (switch)  
**Zone-grid** — Сетка зон освещения (автоматическая расстановка SVT)  
**Forced-устройства** — Устройства из явных пожеланий пользователя  
**Постпроцессинг** — Применение нормативных правил после NN-3  
**Ray casting** — Алгоритм проверки точки внутри полигона  
**Centroid (Green)** — Геометрический центр полигона по формуле Грина  
**Bbox** — Bounding box (ограничивающий прямоугольник)  
**L-комната** — Комната неправильной формы (например, гостиная + прихожая)

---

## 12. СПРАВОЧНАЯ ИНФОРМАЦИЯ

### 12.1 Нормативные документы
- **ГОСТ Р 50571.11-96** — Требования к электроустановкам жилых зданий
- **ПУЭ (Правила устройства электроустановок)** — Нормы размещения розеток
- **СП 256.1325800.2016** — Электроустановки жилых и общественных зданий
- **СП 484.1311500.2020** — Системы противопожарной защиты

### 12.2 Константы системы
```python
# Норматив освещённости
SVT_PER_M2 = 1 / 16.0  # 1 светильник на 16m²

# Норматив розеток
RZT_PER_M2 = 1 / 12.0  # 1 розетка на 12m²
MAX_RZT_PER_ROOM = 6

# Отступы от стен (пиксели)
MARGIN_SMALL = 50   # для комнат < 100m²
MARGIN_LARGE = 80   # для комнат ≥ 100m²

# Высоты монтажа (мм)
HEIGHT_SOCKET = 300
HEIGHT_SWITCH = 900
HEIGHT_CEILING = 0

# Лимиты NN-3
MAX_ROOMS = 12
MAX_DEVICE = 12
```

### 12.3 Типы комнат
```python
ROOM_TYPES = [
    "living_room",   # Гостиная
    "bedroom",       # Спальня
    "kitchen",       # Кухня
    "bathroom",      # Ванная
    "toilet",        # Туалет
    "corridor",      # Коридор/прихожая
    "balcony"        # Балкон/лоджия (определяется по площади < 10m²)
]
```

---

**Конец документа**

*Для вопросов и предложений: используйте формат краткого отчёта (скриншот + JSON + проблема)*