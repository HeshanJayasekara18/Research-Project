"""POST /v1/ask models (M6: evidence-grounded Sinhala answer). Part of the v1 contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_service.api.schemas.retrieval import RetrievalMethod, ScopeInfo


class AskScope(BaseModel):
    model_config = ConfigDict(extra='forbid')
    corpus: Literal['v1', 'v2'] = Field(default='v1', description='v1 (default) or v2 when configured.')
    video_id: str = Field(
        min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$',
        description='Required: the answer is grounded only in this video\'s comments (filtered before ranking). '
                    'A retrieval filter, NOT proof of ownership: the caller must authorise the user first.')


class AskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', json_schema_extra={'examples': [{
        'query': 'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?',
        'scope': {'video_id': '-3a4abtQH3E'}}]})

    query: str = Field(min_length=1, max_length=1000, description='Singlish, Sinhala or English question.')
    scope: AskScope
    method: RetrievalMethod | None = Field(
        default=None, description='Retrieval method. Default: multiview_hybrid if the LaBSE index is available, '
                                  'otherwise multiview_bm25.')
    candidates: int = Field(default=20, ge=5, le=50, description='Retrieved candidates considered for evidence.')

    @field_validator('query')
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('query must contain non-whitespace characters')
        return v.strip()


class VideoReference(BaseModel):
    url: str | None
    granularity: Literal['video'] = Field(
        default='video', description='Always "video": the dataset has no comment IDs, so this links the video, '
                                     'not the individual comment.')


class EvidenceItem(BaseModel):
    label: str = Field(description='Label shown to the model, e.g. "E1".')
    document_id: str = Field(description='Internal surrogate document ID (not a YouTube comment ID).')
    video_id: str
    comment_text: str = Field(description='Original comment text, unmodified.')
    retrieval_rank: int
    matched_terms: list[str] = Field(description='Query content terms found in the comment (why it was selected).')
    video_reference: VideoReference


class RetrievalSummary(BaseModel):
    method: RetrievalMethod
    scope: ScopeInfo
    normalized_query: str | None
    query_content_terms: list[str]
    candidates_retrieved: int
    candidates_selected: int
    candidates_rejected_no_content_match: int


class GenerationInfo(BaseModel):
    provider: str
    model: str
    called: bool = Field(description='False when the pipeline stopped before calling the model.')


class AskResponse(BaseModel):
    request_id: str
    query: str
    status: Literal['answered', 'insufficient_evidence']
    answer: str | None = Field(description='Sinhala answer with [E#] labels; null unless status="answered".')
    insufficient_reason: Literal['no_content_term_match', 'model_declined'] | None = None
    citations: list[EvidenceItem] = Field(description='The supplied comments the answer cites (subset of evidence_supplied).')
    evidence_supplied: list[EvidenceItem] = Field(description='Every comment sent to the model, in prompt order.')
    retrieval: RetrievalSummary
    generation: GenerationInfo
    notice: str
