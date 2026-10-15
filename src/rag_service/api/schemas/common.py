"""Shared v1 models: health and the uniform error envelope (docs/openapi.json)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SemanticStatus(BaseModel):
    index_present: bool
    loaded: bool


class LLMStatus(BaseModel):
    provider: str = Field(description='"disabled" or the configured provider, e.g. "gemini".')
    model: str
    enabled: bool


class HealthResponse(BaseModel):
    status: Literal['ok', 'degraded']
    service: str
    version: str
    api_version: str
    engine_ready: bool
    corpus_documents: int
    videos: int
    semantic: SemanticStatus
    auth_enabled: bool
    llm: LLMStatus | None = None
    corpora: dict[str, dict] | None = Field(default=None, description='Loaded corpora: v1 always; v2 when configured.')


class ErrorBody(BaseModel):
    code: str = Field(description='Machine-readable, e.g. invalid_request, unauthorized, video_not_found, '
                                  'global_scope_disabled, method_unavailable, engine_not_ready, internal_error')
    message: str
    request_id: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
