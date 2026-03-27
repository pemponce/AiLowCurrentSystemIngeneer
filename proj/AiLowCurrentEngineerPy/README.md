# AI Low-Current Engineer — Production System

**Система автоматизированного проектирования слаботочных систем**  
Версия: 2.0 (Production Ready)  
Дата обновления: 28 марта 2026

---

## 📋 СОДЕРЖАНИЕ

1. [Описание системы](#описание-системы)
2. [Текущая архитектура](#текущая-архитектура)
3. [API эндпоинты](#api-эндпоинты)
4. [Установка и запуск](#установка-и-запуск)
5. [Workflow использования](#workflow-использования)
6. [Структура данных](#структура-данных)
7. [Известные ограничения](#известные-ограничения)
8. [Roadmap](#roadmap)

---

## ОПИСАНИЕ СИСТЕМЫ

### Что делает система

Автоматически размещает устройства слаботочных систем на плане квартиры согласно нормативам ГОСТ/СП/ПУЭ.

**Входные данные:**
- PNG план квартиры
- Текстовые пожелания клиента (опционально)

**Выходные данные:**
- Чертёж с размещёнными устройствами (PNG/PDF/DXF)
- JSON с координатами всех устройств

### Поддерживаемые устройства

| Код | Тип устройства | Монтаж | Высота | Норматив |
|-----|---------------|--------|--------|----------|
| SVT | Светильники | Потолок | 0mm | 1 на 16m² |
| RZT | Розетки | Стена | 300mm | 1 на 12m², max 6 |
| DYM | Датчик дыма | Потолок | 0mm | Max 1 на комнату |
| CO2 | Датчик CO2 | Потолок | 0mm | Только кухня/гостиная |
| LAN | Интернет-розетка | Стена | 300mm | 1 на квартиру |
| SWI | Выключатель | Стена | 900mm | У каждой двери |

---

## ТЕКУЩАЯ АРХИТЕКТУРА

### Pipeline обработки

```
PNG план → NN-1 (сегментация) → NN-2 (пожелания) → NN-3 (размещение) → Постпроцессинг → Экспорт
```

### Компоненты

#### 1. NN-1: Сегментация плана
**Модель:** DeepLabV3 ResNet-50  
**Точность:** val_iou = 0.94  
**Файл:** `models/structure_rf_v4_bestreal.pt`

**Что делает:**
- Разделяет план на комнаты
- Определяет типы комнат (bedroom, bathroom, kitchen, etc)
- Извлекает полигоны границ

**Известные проблемы:**
- Большие комнаты (>100m²) могут не разделяться

#### 2. NN-2: Парсинг пожеланий
**Модель:** BiLSTM + CRF (NER)  
**Точность:** F1 = 1.0  
**Файл:** `models/nn2/nn2_best.pt`

**Что делает:**
- Парсит текст вида `"1: свет розетки; 2: датчик дыма"`
- Извлекает требования по комнатам
- Поддерживает диапазоны (`"2-4 источника света"`)

#### 3. NN-3: Размещение устройств
**Модель:** GraphSAGE (2 слоя GNN)  
**Точность:** val_acc = 0.892  
**Файл:** `models/nn3/nn3_best.pt`

**Что делает:**
- Предсказывает количество устройств каждого типа
- Генерирует координаты по правилам `_wall_point()`

**ВАЖНО:** NN-3 v1 **НЕ** предсказывает координаты напрямую — только количество. Позиции генерируются правилами.

#### 4. Валидация NN-3
**Файл:** `app/main.py::_validate_nn3_output()`

**Что делает:**
- Удаляет временно отключённые устройства (tv_sockets, night_lights)
- Дедупликация по координатам (устройства в одной точке)
- Санузлы: max 1 SVT
- Max 1 DYM на комнату
- Проверка координат внутри bbox полигона

**Статистика:** Удаляет ~46% некорректных устройств от NN-3

#### 5. Постпроцессинг
**Файл:** `app/placement.py::_apply_hard_rules()`

**Что делает:**
- Заменяет SVT на zone-grid (сетка зон освещения)
- Добавляет RZT по нормативу (1 на 12m²)
- Добавляет SWI у каждого дверного проёма
- Применяет запреты (розетки в санузлах, датчики дыма на кухне)

**Статистика:** Изменяет ~72% результата NN-3

---

## API ЭНДПОИНТЫ

### Основной workflow

#### POST /upload
Загрузка PNG плана в MinIO

**Request:**
```json
{
  "projectId": "plan001",
  "imageBase64": "iVBORw0KGgoAAAANS..."
}
```

**Response:**
```json
{
  "srcKey": "raw_plans/plan001.png"
}
```

---

#### POST /ingest
Сегментация плана → комнаты с полигонами

**Request:**
```json
{
  "projectId": "plan001",
  "srcKey": "raw_plans/plan001.png"
}
```

**Response:** `PlanGraph` (см. Структура данных)

---

#### POST /design
Размещение устройств **с постпроцессингом** (production)

**Request:**
```json
{
  "projectId": "plan001",
  "preferencesText": "1: свет розетки; 2: датчик дыма"
}
```

**Response:** `DesignGraph` (36 устройств)

**Pipeline:**
1. NN-2 парсит пожелания
2. NN-3 предсказывает устройства (20 устройств)
3. Валидация удаляет мусор
4. Постпроцессинг добавляет нормативные устройства (+16)

---

#### POST /design_nn3
Размещение устройств **БЕЗ постпроцессинга** (экспериментальный)

**Request:**
```json
{
  "projectId": "plan001",
  "preferencesText": ""
}
```

**Response:** `DesignGraph` (20 устройств)

**Pipeline:**
1. NN-2 парсит пожелания
2. NN-3 предсказывает устройства
3. Валидация удаляет мусор
4. **НЕТ постпроцессинга**

**Отличия от /design:**
- Меньше устройств (~20 vs ~36)
- Нет zone-grid для SVT
- Нет нормативных RZT
- Нет SWI у дверей

---

#### POST /design_compare
A/B сравнение `/design` vs `/design_nn3`

**Request:**
```json
{
  "projectId": "plan001",
  "preferencesText": ""
}
```

**Response:**
```json
{
  "project_id": "plan001",
  "method_a": {
    "name": "/design (with postprocessing)",
    "total_devices": 36,
    "reason_breakdown": {
      "NN-3": 10,
      "zone-grid": 7,
      "normative": 6,
      "user-forced": 7,
      "rule-based": 6
    },
    "nn3_usage_pct": 27.78,
    "postprocessing_override_pct": 72.22
  },
  "method_b": {
    "name": "/design_nn3 (pure NN-3)",
    "total_devices": 20,
    "nn3_usage_pct": 100
  },
  "recommendation": "⚠️ Postprocessing overrides 72.2% of NN-3 predictions."
}
```

---

#### POST /export
Экспорт чертежа в файлы

**Request:**
```json
{
  "projectId": "plan001",
  "formats": ["PNG", "PDF", "DXF"]
}
```

**Response:**
```json
{
  "PNG": "http://minio:9000/outputs/plan001.png",
  "PDF": "http://minio:9000/outputs/plan001.pdf",
  "DXF": "http://minio:9000/outputs/plan001.dxf"
}
```

---

### Вспомогательные эндпоинты

#### GET /projects
Список всех проектов из SQLite

#### GET /debug/db/{project_id}
Дамп данных проекта из БД

#### GET /health
Статус сервиса

---

## УСТАНОВКА И ЗАПУСК

### Требования
- Docker Desktop
- PowerShell (Windows) или Bash (Linux/Mac)
- 8GB RAM минимум
- 10GB свободного места

### Запуск

```powershell
# Windows PowerShell
cd "C:\path\to\proj"
docker compose up -d
```

```bash
# Linux/Mac
cd /path/to/proj
docker-compose up -d
```

**Проверка:**
```powershell
curl http://localhost:8000/health
```

**Ожидаемый ответ:**
```json
{"status": "ok"}
```

---

## WORKFLOW ИСПОЛЬЗОВАНИЯ

### Полный цикл обработки плана

```powershell
# 1. Загрузить план
$bytes = [System.IO.File]::ReadAllBytes("plan.png")
$b64 = [Convert]::ToBase64String($bytes)
$body = @{projectId="plan001"; imageBase64=$b64} | ConvertTo-Json

curl -X POST http://localhost:8000/upload `
  -H "Content-Type: application/json" `
  -d $body

# 2. Сегментация
curl -X POST http://localhost:8000/ingest `
  -H "Content-Type: application/json" `
  -d '{"projectId":"plan009","srcKey":"raw_plans/plan009.png"}'

# 3. Размещение устройств
curl -X POST http://localhost:8000/design `
  -H "Content-Type: application/json" `
  -d '{
  "projectId": "plan009",
  "preferencesText": "1: свет, розетки, дым; 2: свет, розетки; 3: свет, 2 розетки; 4: свет, 2 розетки"
}'

# 4. Экспорт
curl -X POST http://localhost:8000/export `
  -H "Content-Type: application/json" `
  -d '{"projectId":"plan009","formats":["PNG"]}'
```

### Сравнение методов

```powershell
# Сравнить /design vs /design_nn3
curl -X POST http://localhost:8000/design_compare `
  -H "Content-Type: application/json" `
  -d '{
  "projectId": "plan009",
  "preferencesText": "1: свет, розетки, дым; 2: свет, розетки; 3: свет, 2 розетки; 4: свет, 2 розетки"
}'
```

---

## СТРУКТУРА ДАННЫХ

### PlanGraph (после /ingest)
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
  "topology": {"roomAdjacency": []}
}
```

### DesignGraph (после /design)
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
  "totalDevices": 36
}
```

### Поле `reason` (источник устройства)
- `"NN-3 prediction"` — устройство от нейросети (27.78%)
- `"zone: N SVT for Xm²"` — zone-grid светильники (постпроцессинг)
- `"norm: N RZT for Xm²"` — нормативные розетки (постпроцессинг)
- `"user request"` — явный запрос пользователя
- `"rule: SWI at door"` — выключатели у дверей (постпроцессинг)

---

## ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ

### 1. NN-3 v1 не предсказывает координаты
**Проблема:** Модель предсказывает только количество устройств, позиции генерируются правилами.

**Влияние:** Постпроцессинг заменяет 72% результата.

**Решение:** NN-3 v2 с regression head (в разработке).

---

### 2. Большие комнаты не разделяются
**Проблема:** NN-1 может объединить гостиную + прихожую в один полигон (134m²).

**Влияние:** SVT размещаются как в одной комнате.

**Workaround:** Zone-grid корректно разбивает на зоны освещения.

---

### 3. Временно отключённые устройства
**Отключены:** tv_sockets, night_lights

**Причина:** Требуют доработки логики размещения.

**Статус:** Будут включены в NN-3 v2.

---

### 4. Forced-устройства не обходят ограничения
**Проблема:** `"1: интернет-розетка; 2: интернет-розетка"` → всё равно только 1 LAN на квартиру.

**Причина:** Жёсткое правило в постпроцессинге.

**Статус:** По нормативам это корректно (1 точка входа интернета).

---

## МЕТРИКИ КАЧЕСТВА

### Текущие показатели (Production)

| Метрика | Значение |
|---------|----------|
| **NN-1 точность** | 94% IoU |
| **NN-2 точность** | 100% F1 |
| **NN-3 точность** | 89.2% accuracy (только количество) |
| **Валидация удаляет** | 46% некорректных устройств NN-3 |
| **Постпроцессинг изменяет** | 72% результата |
| **NN-3 используется** | 27.78% итоговых устройств |
| **Время обработки** | 8-12 секунд на план |

### Breakdown устройств (типичный план 134m²)

```
Всего устройств: 36

Источники:
- NN-3:        10 (27.78%)  — Нейросеть
- zone-grid:    7 (19.44%)  — Постпроцессинг (SVT)
- normative:    6 (16.67%)  — Постпроцессинг (RZT)
- user-forced:  7 (19.44%)  — Пользователь
- rule-based:   6 (16.67%)  — Постпроцессинг (SWI)
```

---

## НОРМАТИВЫ

### ГОСТ/СП/ПУЭ правила

#### Светильники (SVT)
- **Норма:** 1 на 16m²
- **Размещение:** Zone-grid (сетка зон освещения)
- **Санузел:** max 1 SVT
- **Отступ от стен:** 50-80px

#### Розетки (RZT)
- **Норма:** 1 на 12m², max 6 на комнату
- **Размещение:** Только внешние стены (периметр bbox)
- **Запрещено:** bathroom, toilet, balcony

#### Датчик дыма (DYM)
- **Норма:** max 1 на комнату
- **Размещение:** Центроид потолка (формула Грина)
- **Запрещено:** kitchen, bathroom, toilet, balcony

#### Датчик CO2
- **Разрешено:** Только living_room и kitchen
- **Размещение:** Центроид потолка

#### Интернет-розетка (LAN)
- **Норма:** Строго 1 на квартиру
- **Приоритет:** corridor > living_room

#### Выключатели (SWI)
- **Норма:** У каждого дверного проёма
- **Размещение:** Offset 25px вдоль стены от проёма
- **Высота:** 900mm

---

## ROADMAP

### ✅ Выполнено (v2.0)

- [x] Исправлены баги постпроцессинга (санузлы max 1 SVT)
- [x] Добавлена дедупликация DYM
- [x] Создан эндпоинт `/design_compare`
- [x] Улучшена валидация NN-3 (удаление мусора)
- [x] Получены метрики качества системы

### 🔄 В разработке (v3.0 — NN-3 v2)

**Цель:** NN-3 сама предсказывает координаты, постпроцессинг только валидирует

**Задачи:**

#### Неделя 1-2: Архитектура модели
- [ ] Спроектировать PlacementNetV2 с dual-head
- [ ] Classification head: количество устройств
- [ ] Regression head: нормализованные координаты [0,1]
- [ ] Combined loss: count + coord + validity

#### Неделя 3-4: Генерация датасета
- [ ] Разработать GroundTruthGenerator
- [ ] Генерировать реалистичные позиции (zone-grid, периметр, центроид)
- [ ] Создать датасет 8000 примеров
- [ ] Валидация первых 100 примеров (human review)

#### Неделя 5-6: Обучение модели
- [ ] Обучить PlacementNetV2 (80 epochs)
- [ ] Целевые метрики:
  - count_acc > 0.90
  - coord_mae < 0.05
  - coord_within_threshold > 0.85
- [ ] Реализовать inference_v2.py (без _wall_point)

#### Неделя 7-8: Интеграция и тестирование
- [ ] Создать эндпоинт `/design_v2`
- [ ] Минимальный постпроцессинг (только валидация)
- [ ] A/B тестирование v1 vs v2
- [ ] Production deployment

**Целевые метрики v3.0:**
- NN-3 используется: >80% (вместо 27.78%)
- Постпроцессинг: <20% (вместо 72.22%)
- Точность координат: >85% в пределах 10% от размера комнаты

### 📋 Backlog (v4.0+)

- [ ] Улучшить NN-1 для разделения L-комнат
- [ ] Добавить поддержку нестандартных планировок
- [ ] Web UI для интерактивной корректировки
- [ ] Автоматический расчёт BOM (спецификация материалов)
- [ ] Включить tv_sockets и night_lights

---

## ФАЙЛОВАЯ СТРУКТУРА

```
proj/
├── app/
│   ├── main.py                    # FastAPI эндпоинты
│   ├── placement.py               # Постпроцессинг нормативных правил
│   ├── geometry.py                # Shapely-утилиты
│   ├── export_overlay_png.py      # Рендер PNG чертежа
│   ├── rules.py                   # Нормативные константы
│   ├── db.py                      # SQLite персистентность
│   ├── minio_client.py            # MinIO (S3-хранилище)
│   │
│   ├── api/
│   │   └── design_compare.py      # A/B сравнение методов
│   │
│   ├── nn1/ (ml/)                 # NN-1: сегментация
│   │   └── structure_infer.py     # DeepLabV3
│   │
│   ├── nn2/                       # NN-2: парсинг пожеланий
│   │   └── infer.py               # BiLSTM+CRF
│   │
│   └── nn3/                       # NN-3: размещение устройств
│       ├── infer.py               # Inference v1
│       ├── model.py               # GraphSAGE архитектура
│       ├── dataset_gen.py         # Генератор датасета
│       └── train.py               # Обучение
│
├── models/
│   ├── structure_rf_v4_bestreal.pt  # NN-1 модель
│   ├── nn2/nn2_best.pt              # NN-2 модель
│   └── nn3/nn3_best.pt              # NN-3 модель
│
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## РАЗРАБОТКА

### Деплой изменений

```powershell
# Скопировать файл в контейнер
docker cp app\placement.py proj-planner-1:/app/app/placement.py

# Перезапустить
docker restart proj-planner-1

# Проверить логи
docker logs proj-planner-1 --tail 50
```

### Тестирование

**ВАЖНО:** Каждый тест — новый `projectId`, иначе отдаётся кэш из SQLite.

```powershell
# Тест с новым ID
curl -X POST http://localhost:8000/design `
  -H "Content-Type: application/json" `
  -d '{"projectId":"test_001","preferencesText":""}'
```

### Переобучение NN-3

```powershell
# Генерация датасета
docker exec proj-planner-1 python -m app.nn3.dataset_gen `
  --out /data/nn3_dataset --count 8000

# Обучение
docker exec -it proj-planner-1 python -m app.nn3.train `
  --data /data/nn3_dataset --epochs 80 --out /app/models/nn3
```

---

## FAQ

### Q: Почему постпроцессинг заменяет 72% результата?
**A:** NN-3 v1 предсказывает только количество устройств, а координаты генерируются правилами. Постпроцессинг применяет нормативные правила ГОСТ/ПУЭ для корректного размещения.

### Q: Можно ли использовать только NN-3 без постпроцессинга?
**A:** Да, эндпоинт `/design_nn3`. Но результат будет неполным: нет zone-grid SVT, нормативных RZT, выключателей у дверей.

### Q: Когда будет NN-3 v2?
**A:** Roadmap: 8 недель разработки. Целевая метрика: NN-3 используется >80% вместо 27.78%.

### Q: Как добавить поддержку новых устройств?
**A:** 
1. Добавить в `DEVICES` в `app/nn3/model.py`
2. Переобучить NN-3
3. Добавить правила размещения в `placement.py`

### Q: Почему проект называется "planner" а не "low-current-engineer"?
**A:** Исторические причины. Внутреннее название проекта — "planner".

---

## КОНТАКТЫ И ПОДДЕРЖКА

**Для вопросов:**
- Создайте issue в формате: скриншот + JSON + проблема одной строкой
- Используйте `/design_compare` для A/B анализа

**Документация:**
- `REFACTORING_PLAN.md` — детальный план улучшений
- `QUICK_START.md` — краткое руководство
- `placement_bugfixes.py` — примеры исправлений

---

## ЛИЦЕНЗИЯ

Proprietary — internal use only

---

**Версия документа:** 2.0  
**Последнее обновление:** 28 марта 2026  
**Статус:** Production Ready (с планом развития до v3.0)