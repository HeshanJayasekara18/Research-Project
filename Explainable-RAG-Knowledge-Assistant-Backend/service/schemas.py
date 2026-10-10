"""Pydantic models = the public v1 JSON contract (mirrored in docs/openapi.json).

Contract rules: field names are snake_case; adding optional response fields is
non-breaking; renaming/removing fields or enum values requires /v2.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalMethod(str, Enum):
    """Stable API identifiers. Mapped to engine method names in the service layer."""
    bm25 = 'bm25'
    char_tfidf = 'char_tfidf'
    labse = 'labse'
    normalized_bm25 = 'normalized_bm25'
    normalized_labse = 'normalized_labse'
    multiview_bm25 = 'multiview_bm25'
    multiview_labse = 'multiview_labse'
    multiview_hybrid = 'multiview_hybrid'


ScoreType = Literal['bm25', 'char_tfidf_cosine', 'labse_cosine', 'rrf']


class RetrieveScope(BaseModel):
    model_config = ConfigDict(extra='forbid')
    video_id: str | None = Field(
        default=None, min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$',
        description='Search ONLY this video\'s comments (filtering happens before ranking). '
                    'null = global research corpus. This is a retrieval filter, NOT proof of ownership: '
                    'the caller must authorise the user for this video before forwarding the request.')


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [{
        'query': 'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?',
        'method': 'bm25', 'top_k': 5, 'scope': {'video_id': '-3a4abtQH3E'}}]})

    query: str = Field(min_length=1, max_length=1000, description='Singlish, Sinhala or English question.')
    method: RetrievalMethod = RetrievalMethod.bm25
    top_k: int = Field(default=10, ge=1, le=50)
    scope: RetrieveScope = Field(default_factory=RetrieveScope)

    @field_validator('query')
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('query must contain non-whitespace characters')
        return v.strip()


class ViewContribution(BaseModel):
    view: str = Field(description='e.g. "BM25/original", "LaBSE/normalized"')
    rank: int
    score: float


class RetrievedComment(BaseModel):
    rank: int = Field(ge=1)
    document_id: str = Field(description='Stable surrogate ID (not a YouTube comment ID).')
    video_id: str
    text: str
    likes: int | None
    video_url: str | None = Field(description='Video link; comment-level permalinks are not available.')
    score: float = Field(description='Method-specific; compare only within one response.')
    score_type: ScoreType
    views: list[ViewContribution] | None = Field(default=None, description='Multi-view methods only.')


class NormalizationInfo(BaseModel):
    status: str = Field(description='normalized | partial | failed | unchanged | empty')
    normalized_query: str
    coverage: float
    used: bool = Field(description='Whether the normalised query was actually used for retrieval.')


class ScopeInfo(BaseModel):
    type: Literal['video', 'global']
    video_id: str | None
    documents_searched: int
    ownership_verified: Literal[False] = Field(
        default=False, description='Always false: this service never verifies creator/channel ownership.')


class RetrieveResponse(BaseModel):
    request_id: str
    query: str
    method: RetrievalMethod
    top_k: int
    scope: ScopeInfo
    normalization: NormalizationInfo | None = None
    results: list[RetrievedComment]
    result_count: int
    notice: str = Field(description='Results are retrieved candidates, not verified evidence or generated answers.')


class SemanticStatus(BaseModel):
    index_present: bool
    loaded: bool


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


class ErrorBody(BaseModel):
    code: str = Field(description='Machine-readable, e.g. invalid_request, unauthorized, video_not_found, '
                                  'global_scope_disabled, method_unavailable, engine_not_ready, internal_error')
    message: str
    request_id: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
