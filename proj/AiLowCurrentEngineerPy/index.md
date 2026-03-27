# AI Low-Current Engineer — README

## Что делает система
Загружаешь PNG план квартиры → получаешь чертёж с расставленными устройствами (SVT, RZT, DYM, CO2, LAN, SWI) по нормативам ГОСТ/СП/ПУЭ.

---

## Запуск

```powershell
cd "C:\Users\azhel\Desktop\Ai ingeneer low-current systems\proj"
docker compose up -d
```

**Деплой изменений:**
```powershell
.\deploy.ps1
```
> deploy.ps1 копирует `app/*.py` и `app/nn3/*.py` в контейнер и рестартует planner.  
> Если файл не применяется — копируй вручную: `docker cp app\файл.py proj-planner-1:/app/app/файл.py`

---

## Эндпоинты

```
POST /upload   — загрузка PNG (base64) → MinIO
POST /ingest   — NN-1 сегментация плана → комнаты с polygonPx
POST /design   — NN-2 парсинг пожеланий + NN-3 размещение + постпроцессинг
POST /export   — PNG/PDF/DXF чертёж
GET  /projects — список проектов
GET  /health   — статус сервиса
```

**Полный цикл:**
```powershell
# Загрузить план
$bytes = [System.IO.File]::ReadAllBytes("C:\путь\план.png")
$b64   = [Convert]::ToBase64String($bytes)
$body  = @{ projectId="plan001"; imageBase64=$b64 } | ConvertTo-Json
curl -X POST http://localhost:8000/upload -H "Content-Type: application/json" -d $body

curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"projectId\":\"plan001\",\"srcKey\":\"raw_plans/input.png\"}"
curl -X POST http://localhost:8000/design -H "Content-Type: application/json" -d "{\"projectId\":\"plan001\",\"preferencesText\":\"1: свет розетки; 2: свет розетки датчик дыма\"}"
curl -X POST http://localhost:8000/export -H "Content-Type: application/json" -d "{\"projectId\":\"plan001\",\"formats\":[\"PNG\"]}"
```

> **Важно:** каждый новый тест запускай с новым `projectId` (plan002, plan003…) — иначе отдаётся кэш из SQLite.

---

## Файловая структура

```
AiLowCurrentEngineerPy/app/
│
├── main.py                  — FastAPI эндпоинты + логика /design (1900 строк)
│                              Содержит дублированную копию _apply_hard_rules —
│                              при изменении placement.py менять и здесь тоже
│
├── placement.py             — Постпроцессинг нормативных правил:
│                              • _apply_hard_rules() — фильтрация и коррекция устройств
│                              • _svt_grid_positions() — сетка зон SVT по полигону
│                              • _point_in_polygon() — ray casting
│
├── geometry.py              — Shapely-утилиты:
│                              • detect_doorways() — поиск дверных проёмов между комнатами
│                              • coerce_polygon(), set_geometry()
│
├── export_overlay_png.py    — Рендер PNG чертежа:
│                              • _build_lighting_zones() — сетка SVT внутри полигона
│                              • _nearest_interior_point() — fallback для L-комнат
│                              • export_zones_preview() — preview зон до /design
│
├── rules.py                 — Нормативные константы (ГОСТ/ПУЭ/СП)
├── db.py                    — SQLite персистентность проектов
├── minio_client.py          — Работа с MinIO (хранение PNG/PDF)
│
├── nn1/ (ml/)               — NN-1: DeepLabV3, сегментация плана → полигоны комнат
│   └── ml/structure_infer.py   val_iou=0.94, модель: structure_rf_v4_bestreal.pt
│
├── nn2/                     — NN-2: BiLSTM+CRF, парсинг текстовых пожеланий
│   └── infer.py                F1=1.0, модель: nn2/nn2_best.pt
│
└── nn3/                     — NN-3: GraphSAGE, предсказание устройств по комнате
    ├── infer.py                val_acc=0.892, модель: nn3/nn3_best.pt
    │                           _wall_point() — геометрия размещения на стенах/потолке
    ├── model.py                MAX_DEVICE=12, MAX_ROOMS=12
    ├── dataset_gen.py          генератор датасета по нормам
    └── train.py
```

---

## Устройства

| Код | kind | Норма |
|-----|------|-------|
| SVT | ceiling_lights | 1 на 16m², сетка зон |
| RZT | power_socket | 1 на 12m², max 6 |
| DYM | smoke_detector | 1 на комнату, не кухня/санузел |
| CO2 | co2_detector | только кухня |
| LAN | internet_sockets | 1 на квартиру |
| SWI | switch | у дверных проёмов |

---

## Ключевые правила постпроцессинга (placement.py)

- **SVT**: заменяет все NN-3 светильники на zone-grid; для L-комнат — fallback `_nearest_interior_point`
- **DYM**: максимум 1 на комнату; позиция — центроид потолка (формула Грина)
- **CO2**: только `living_room` и `kitchen`; mount=ceiling
- **LAN**: строго 1 на квартиру, forced-запрос тоже не обходит ограничение
- **RZT**: только внешние стены (рёбра у периметра bbox); `_is_outer()` в infer.py
- **SWI**: проёмы через `detect_doorways()` (Shapely, tolerance=18px); fallback — кратчайшее ребро полигона
- **bathroom/toilet/balcony**: только 1 SVT, никаких RZT/DYM

---

## Известные ограничения

- **NN-1** иногда не делит квартиру на отдельные комнаты — большая комната (134m²) содержит гостиную + прихожую как один полигон. SVT и RZT в этом случае раскладываются по всему L-полигону
- **room_006** (bathroom) может получать 2 SVT от NN-3 — лечится `placement.py` фильтром `bathroom/toilet → max 1 SVT`; если не применяется — скопировать файл вручную через `docker cp`
- Пунктирная линия между комнатами 1 и 2 на плане — это открытый проём, не стена. NN-1 корректно его игнорирует

---

## Переобучение NN-3

```powershell
docker exec proj-planner-1 python -m app.nn3.dataset_gen --out /data/nn3_dataset --count 8000
docker exec -it proj-planner-1 python -m app.nn3.train --data /data/nn3_dataset --out /app/models/nn3 --epochs 80
```