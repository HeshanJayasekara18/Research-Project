"""FastAPI dependencies: settings, the shared RetrievalService, and service-to-service auth.

Auth here authenticates the CALLING SERVER (Next.js Route Handler / API gateway)
with a shared secret. It does not identify end users, and scope.video_id is never
treated as proof that a user owns a video: the caller must authorise that first.
"""
from __future__ import annotations

import hmac

from fastapi import Header, Request

from rag_service.api.errors import ApiError
from rag_service.core.config import Settings
from rag_service.rag.ask_service import AskService
from rag_service.services.retrieval_service import RetrievalService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_retrieval_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, 'retrieval_service', None)
    if service is None:
        raise ApiError(503, 'engine_not_ready', 'Retrieval engine is not loaded yet.')
    return service


def get_ask_service(request: Request) -> AskService:
    service = getattr(request.app.state, 'ask_service', None)
    if service is None:
        raise ApiError(503, 'engine_not_ready', 'Retrieval engine is not loaded yet.')
    return service


def require_service_key(request: Request,
                        x_service_key: str | None = Header(default=None, alias='X-Service-Key')) -> None:
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return
    if not x_service_key or not hmac.compare_digest(x_service_key.encode(), settings.service_api_key.encode()):
        raise ApiError(401, 'unauthorized', 'Missing or invalid X-Service-Key header.')
