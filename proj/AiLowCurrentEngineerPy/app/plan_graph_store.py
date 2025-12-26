from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, Optional

from app.minio_client import upload_file, EXPORT_BUCKET


PLAN_GRAPH_DIR = os.getenv("PLAN_GRAPH_DIR", "/tmp/plan-graphs")
os.makedirs(PLAN_GRAPH_DIR, exist_ok=True)


def plan_graph_path(project_id: str) -> str:
    return osp.join(PLAN_GRAPH_DIR, f"{project_id}.json")


def save_plan_graph(project_id: str, plan_graph: Dict[str, Any]) -> str:
    path = plan_graph_path(project_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan_graph, f, ensure_ascii=False, indent=2)
    return path


def load_plan_graph(project_id: str) -> Optional[Dict[str, Any]]:
    path = plan_graph_path(project_id)
    if not osp.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_plan_graph(project_id: str, plan_graph_local_path: str) -> str:
    """
    Кладём PlanGraph JSON в exports bucket.
    """
    key = f"plan-graphs/{project_id}.json"
    return upload_file(EXPORT_BUCKET, plan_graph_local_path, key)
