"""
app/db.py — SQLite персистентное хранилище вместо in-memory dict.

Таблицы:
  projects  — метаданные проекта
  rooms     — полигоны и типы комнат (JSON)
  designs   — результат /design (JSON)
  exports   — пути к файлам экспорта
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger("planner")

DB_PATH = os.environ.get("SQLITE_PATH", "/data/planner.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    conn = _conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Создаём таблицы если не существуют."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                src_key     TEXT,
                local_path  TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS rooms (
                project_id  TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS designs (
                project_id  TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS room_maps (
                project_id  TEXT PRIMARY KEY,
                data        TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exports (
                project_id  TEXT,
                format      TEXT,
                s3_key      TEXT,
                local_path  TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (project_id, format)
            );
        """)
    logger.info("SQLite DB initialised at %s", DB_PATH)


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def save_project(project_id: str, src_key: str, local_path: str) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO projects (id, src_key, local_path, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                src_key=excluded.src_key,
                local_path=excluded.local_path,
                updated_at=datetime('now')
        """, (project_id, src_key, local_path))


def get_project(project_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        ).fetchone()
        return dict(row) if row else None


def save_rooms(project_id: str, rooms: List[Dict]) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO rooms (project_id, data, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET
                data=excluded.data, updated_at=datetime('now')
        """, (project_id, json.dumps(rooms, ensure_ascii=False)))


def get_rooms(project_id: str) -> List[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT data FROM rooms WHERE project_id=?", (project_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else []


def save_design(project_id: str, design: Dict) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO designs (project_id, data, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET
                data=excluded.data, updated_at=datetime('now')
        """, (project_id, json.dumps(design, ensure_ascii=False)))


def get_design(project_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT data FROM designs WHERE project_id=?", (project_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None


def save_room_map(project_id: str, room_map: Dict) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO room_maps (project_id, data)
            VALUES (?, ?)
            ON CONFLICT(project_id) DO UPDATE SET data=excluded.data
        """, (project_id, json.dumps(room_map, ensure_ascii=False)))


def get_room_map(project_id: str) -> Dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT data FROM room_maps WHERE project_id=?", (project_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else {}


def save_export(project_id: str, fmt: str, s3_key: str, local_path: str) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT INTO exports (project_id, format, s3_key, local_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, format) DO UPDATE SET
                s3_key=excluded.s3_key,
                local_path=excluded.local_path,
                created_at=datetime('now')
        """, (project_id, fmt, s3_key, local_path))


def list_projects() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, src_key, created_at FROM projects ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]