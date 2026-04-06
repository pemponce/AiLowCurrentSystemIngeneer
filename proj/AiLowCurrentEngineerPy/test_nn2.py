import sys
sys.path.insert(0, '.')
from app.nn2.infer import parse_text
import json

tests = [
    'хочу в гостиной 2 розетки для телевизора и датчик дыма',
    'нужен интернет в спальне и на кухне, домофон у входа',
    'поставьте датчики дыма везде кроме кухни и охранную сигнализацию',
]

for t in tests:
    r = parse_text(t, project_id='test')
    print(f'TEXT:   {t}')
    print(f'ROOMS:  {json.dumps(r["rooms"], ensure_ascii=False)}')
    print(f'GLOBAL: {json.dumps(r["global"], ensure_ascii=False)}')
    print()