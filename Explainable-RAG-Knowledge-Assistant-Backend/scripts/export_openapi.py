"""Write the v1 OpenAPI contract to docs/openapi.json (commit it; the frontend generates types from it).

Usage:  python scripts/export_openapi.py          (write)
        python scripts/export_openapi.py --check  (exit 1 if the committed file is stale)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'docs' / 'openapi.json'


def render() -> str:
    from service.config import Settings
    from service.main import create_app
    return json.dumps(create_app(Settings()).openapi(), indent=2, ensure_ascii=False, sort_keys=True) + '\n'


if __name__ == '__main__':
    text = render()
    if '--check' in sys.argv:
        current = OUT.read_text(encoding='utf-8') if OUT.exists() else ''
        sys.exit(0 if current == text else 'docs/openapi.json is stale: run python scripts/export_openapi.py')
    OUT.write_text(text, encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT)}')
