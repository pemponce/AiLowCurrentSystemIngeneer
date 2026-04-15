# app/main.py
"""
AI Low-Current Engineer - FastAPI Application
Рефакторинг v1.0 - main.py компактный (150 строк)
"""

import logging
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Импорты роутеров
from app.api.design_compare import router as compare_router
from app.api_structure import router as structure_router
from app.api_artifacts import router as artifacts_router
from app.api_state import router as state_router

from app.api_endpoints.health import router as health_router
from app.api_endpoints.debug import router as debug_router
from app.api_endpoints.upload import router as upload_router
from app.api_endpoints.ingest import router as ingest_router
from app.api_endpoints.design import router as design_router
from app.api_endpoints.export import router as export_router

# Глобальное хранилище
from app.geometry import DB

# SQLite persistence
try:
    from app.db import (
        init_db,
        save_project,
        save_rooms,
        save_room_map,
        list_projects,
        get_rooms,
    )

    _SQLITE_OK = True
    init_db()
    logging.getLogger("planner").info("SQLite DB ready")
except Exception as e:
    _SQLITE_OK = False
    logging.getLogger("planner").warning(f"SQLite unavailable: {e}")

# NN-2: парсинг пожеланий
try:
    from app.nn2.infer import parse_text as nn2_parse_text

    _NN2_AVAILABLE = True
except Exception as e:
    _NN2_AVAILABLE = False
    nn2_parse_text = None
    logging.getLogger("planner").warning(f"NN-2 недоступна: {e}")

# NN-3: размещение устройств
try:
    from app.nn3.infer import run_placement as nn3_run_placement

    _NN3_AVAILABLE = True
except Exception as e:
    _NN3_AVAILABLE = False
    nn3_run_placement = None
    logging.getLogger("planner").warning(f"NN-3 недоступна: {e}")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("planner")

# Создаём FastAPI приложение
app = FastAPI(
    title="AI Low-Current Engineer",
    version="1.0.0",
    description="Автоматизированное проектирование слаботочных систем",
)

# Подключаем роутеры
app.include_router(health_router)
app.include_router(debug_router)
app.include_router(upload_router)
app.include_router(ingest_router)
app.include_router(design_router)
app.include_router(export_router)

# Старые роутеры (совместимость)
app.include_router(compare_router)
app.include_router(structure_router)
app.include_router(artifacts_router)
app.include_router(state_router)

# Инициализация зависимостей в модулях
from app.api_endpoints import debug, ingest, design, export

debug.set_db(DB)
ingest.set_db(DB, _SQLITE_OK, save_project if _SQLITE_OK else None,
              save_rooms if _SQLITE_OK else None, save_room_map if _SQLITE_OK else None)
design.set_dependencies(DB, nn2_parse_text, nn3_run_placement, _NN2_AVAILABLE, _NN3_AVAILABLE)
export.set_db(DB)

# JSON serialization для custom типов
from app.utils_helpers.json_utils import _json_default


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик ошибок с правильной сериализацией."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Инициализация при старте сервиса."""
    logger.info("=" * 60)
    logger.info("AI Low-Current Engineer v1.0 - Starting")
    logger.info("=" * 60)
    logger.info("SQLite: %s", "OK" if _SQLITE_OK else "Disabled")
    logger.info("NN-1: Available (lazy load)")
    logger.info("NN-2: %s", "OK" if _NN2_AVAILABLE else "Disabled")
    logger.info("NN-3: %s", "OK" if _NN3_AVAILABLE else "Disabled")
    logger.info("=" * 60)

    # Создаём необходимые директории
    os.makedirs("/tmp/downloads", exist_ok=True)
    os.makedirs("/tmp/exports", exist_ok=True)
    os.makedirs("/data", exist_ok=True)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке сервиса."""
    logger.info("Shutting down AI Low-Current Engineer")


# Root endpoint
@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "AI Low-Current Engineer",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload",
            "ingest": "POST /ingest",
            "design": "POST /design",
            "export": "POST /export",
            "docs": "GET /docs",
        },
        "features": {
            "nn1": "segmentation",
            "nn2": _NN2_AVAILABLE,
            "nn3": _NN3_AVAILABLE,
            "sqlite": _SQLITE_OK,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)