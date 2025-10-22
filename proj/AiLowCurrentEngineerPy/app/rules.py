import json
import os

_rules = None


def get_rules():
    global _rules
    if _rules is None:
        path = os.getenv('RULES_PATH', '/app/rules.json')
        with open(path, 'r', encoding='utf-8') as f:
            _rules = json.load(f)
    return _rules
