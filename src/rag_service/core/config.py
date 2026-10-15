"""Service settings, read once from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository layout (src/rag_service/core/config.py -> repo root is parents[3]).
# Requires an editable install (pip install -e .) or running from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get('RAG_DATA_DIR') or PROJECT_ROOT / 'data')
RESULTS_DIR = PROJECT_ROOT / 'research' / 'evaluation' / 'results'


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == '':
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class Settings:
    env: str = 'development'                 # development | test | production
    service_api_key: str | None = None       # shared secret for server-to-server calls
    allow_global_scope: bool = True          # allow video_id=null (whole research corpus)
    cors_origins: tuple[str, ...] = ()       # empty: no browser CORS (server-side calls only)
    # LLM for /v1/ask. Disabled unless explicitly configured; the key never appears in repr/logs.
    llm_provider: str = 'disabled'           # disabled | gemini
    llm_model: str = 'gemini-3.8-flash'
    llm_api_key: str | None = field(default=None, repr=False)
    llm_timeout_s: float = 30.0
    # Corpus V2 (3-domain collection export). None = only the frozen Corpus V1 is served.
    corpus_v2_dir: str | None = None

    @property
    def auth_enabled(self) -> bool:
        return bool(self.service_api_key)

    def validate(self) -> None:
        if self.env == 'production' and not self.auth_enabled:
            raise RuntimeError('RAG_SERVICE_API_KEY must be set when RAG_ENV=production.')


def load_settings() -> Settings:
    origins = os.environ.get('RAG_CORS_ORIGINS', '')
    return Settings(
        env=os.environ.get('RAG_ENV', 'development').strip().lower(),
        service_api_key=os.environ.get('RAG_SERVICE_API_KEY') or None,
        allow_global_scope=_bool(os.environ.get('RAG_ALLOW_GLOBAL_SCOPE'), True),
        cors_origins=tuple(o.strip() for o in origins.split(',') if o.strip()),
        llm_provider=os.environ.get('RAG_LLM_PROVIDER', 'disabled').strip().lower() or 'disabled',
        llm_model=os.environ.get('RAG_LLM_MODEL', 'gemini-3.8-flash').strip() or 'gemini-3.8-flash',
        llm_api_key=os.environ.get('RAG_LLM_API_KEY') or os.environ.get('GEMINI_API_KEY') or None,
        llm_timeout_s=float(os.environ.get('RAG_LLM_TIMEOUT_S', '30')),
        corpus_v2_dir=os.environ.get('RAG_CORPUS_V2_DIR') or None,
    )
