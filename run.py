"""Start the C3 FastAPI service for local development.

    python run.py                 # http://127.0.0.1:8000  (docs at /docs)
    python run.py --port 8001
    python run.py --no-reload

Adds src/ to the import path (no PYTHONPATH or pip install needed) and loads
.env from this folder if present, so RAG_SERVICE_API_KEY can live there.
Variables already set in the shell take precedence over .env.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE reader (comments and blank lines ignored; shell values win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the C3 retrieval API (development).')
    parser.add_argument('--host', default='127.0.0.1', help='default 127.0.0.1 (localhost only)')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-reload', action='store_true', help='disable auto-reload on code changes')
    args = parser.parse_args()

    load_dotenv(ROOT / '.env')
    sys.path.insert(0, str(SRC))
    # The reloader starts a child process, which needs src/ on its path too.
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [str(SRC), os.environ.get('PYTHONPATH')]))

    if not os.environ.get('RAG_SERVICE_API_KEY') and args.host not in ('127.0.0.1', 'localhost'):
        sys.exit('Refusing to listen on a non-local host without RAG_SERVICE_API_KEY.')

    import uvicorn
    uvicorn.run('rag_service.api.main:app', host=args.host, port=args.port,
                reload=not args.no_reload, reload_dirs=[str(SRC)] if not args.no_reload else None)


if __name__ == '__main__':
    main()
