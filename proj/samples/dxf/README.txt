Положите сюда DXF-файл (экспорт из AutoCAD/ArchiCAD/Revit), например plan1.dxf.
В запросе /ingest используйте src_s3_key: "raw-plans/plan1.dxf" — сервис возьмёт /data/plan1.dxf

🧪 Пример полного прогона с DXF
# положите ваш DXF в samples/dxf/plan1.dxf

docker compose up --build

# ingest из DXF
curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' \
-d '{"project_id":"proj-2","src_s3_key":"raw-plans/plan1.dxf"}'

# размещение + валидация
curl -X POST http://localhost:8000/place -H 'Content-Type: application/json' \
-d '{"project_id":"proj-2","preferences":{}}'

# маршрутизация + BOM
curl -X POST http://localhost:8000/route -H 'Content-Type: application/json' \
-d '{"project_id":"proj-2"}'

# экспорт + выгрузка в MinIO
curl -X POST http://localhost:8000/export -H 'Content-Type: application/json' \
-d '{"project_id":"proj-2","formats":["DXF","PDF"]}'