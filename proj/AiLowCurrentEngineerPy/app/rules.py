# app/rules.py
import json
import os

_rules = None

def get_rules():
    global _rules
    if _rules is None:
        # Ищем rules.json рядом с этим файлом
        default_path = os.path.join(os.path.dirname(__file__), "rules.json")
        path = os.getenv("RULES_PATH", default_path)
        with open(path, "r", encoding="utf-8") as f:
            _rules = json.load(f)
    return _rules