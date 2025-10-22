## Запуск
```bash
docker compose up --build
Проверка планировщика (использует samples/geojson):
# 1) Ingest (поднимаем комнаты)
curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/
json'
-d '{"project_id":"proj-1", "src_s3_key":"raw-plans/plan1.dxf"}'
# 2) Расстановка устройств
curl -X POST http://localhost:8000/place -H 'Content-Type: application/
json'
-d '{"project_id":"proj-1", "preferences":{}}'
# 3) Маршрутизация
curl -X POST http://localhost:8000/route -H 'Content-Type: application/
json'
-d '{"project_id":"proj-1"}'
# 4) Экспорт DXF+PDF
curl -X POST http://localhost:8000/export -H 'Content-Type: application/
json'
-d '{"project_id":"proj-1", "formats":["DXF","PDF"]}'
Файлы появятся в контейнере planner по пути /tmp/exports/ .
Следующие шаги
• [ ] Реальный парсинг DXF: ezdxf → полилинии → полигоны комнат
16
[ ] Слои и типы помещений (семантика)
[ ] Жёсткие правила (ГОСТ/ПУЭ) из БД вместо захардкоженных
[ ] Точная шкала и листы (PDF с легендой)
[ ] Слив результатов в MinIO, отдача ссылок из backend
[ ] Редактор 2D во фронтенде (перетаскивание устройств) ```
Пояснения
Сейчас planner упрощённо берёт samples/geojson/simple_apartment.geojson вместо
реального DXF — это ускоряет старт. Заменим на парсер DXF, когда проверим пайплайн
end‑to‑end.
Маршрутизация — через граф по вершинам полигонов; на практике нужно строить
«линию штробы» на фиксированных высотах и добавлять штрафы за пересечения дверей/
несущих стен.
Экспорт DXF/PDF — примитивный, но рабочий. Символы можно заменить на DXF-блоки.
