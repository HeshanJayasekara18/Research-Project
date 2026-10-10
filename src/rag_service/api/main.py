"""App factory. Run:  uvicorn rag_service.api.main:app --port 8000   (after pip install -e .)"""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from rag_service import SERVICE_VERSION
from rag_service.api.errors import install_error_handlers
from rag_service.api.routes import v1_router
from rag_service.core.config import Settings, load_settings
from rag_service.services.retrieval_service import RetrievalService, build_engine

log = logging.getLogger('rag_service')
_RID = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def create_app(settings: Settings | None = None, engine=None) -> FastAPI:
    """`engine` may be injected (tests); otherwise the corpus is loaded at startup."""
    settings = settings or load_settings()
    settings.validate()
    if not settings.auth_enabled:
        log.warning('RAG_SERVICE_API_KEY not set: /v1/retrieve is UNAUTHENTICATED (development only).')

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retrieval_service = RetrievalService(engine or build_engine(), settings)
        yield

    app = FastAPI(title='Singlish Retrieval Engine API', version=SERVICE_VERSION, lifespan=lifespan,
                  description='Explainable RAG Knowledge Assistant - retrieval service (v1). '
                              'Server-to-server API: call from Next.js Route Handlers or the team API gateway.')
    app.state.settings = settings
    app.state.retrieval_service = None

    @app.middleware('http')
    async def add_request_id(request: Request, call_next):
        incoming = request.headers.get('X-Request-ID', '')
        request.state.request_id = incoming if _RID.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        return response

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Request-ID'])
    install_error_handlers(app)
    app.include_router(v1_router)
    return app


app = create_app()
