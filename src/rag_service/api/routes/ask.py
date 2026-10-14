"""POST /v1/ask: evidence-grounded Sinhala answer (M6). Thin handler; logic in rag.ask_service."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_service.api.dependencies import get_ask_service, require_service_key
from rag_service.api.errors import request_id
from rag_service.api.schemas import ErrorResponse
from rag_service.api.schemas.ask import AskRequest, AskResponse
from rag_service.rag.ask_service import AskService

router = APIRouter(tags=['rag'])

_ERRORS = {code: {'model': ErrorResponse} for code in (401, 404, 422, 500, 502, 503)}


@router.post('/ask', response_model=AskResponse, operation_id='askGrounded', responses=_ERRORS,
             dependencies=[Depends(require_service_key)])
def ask(body: AskRequest, request: Request, service: AskService = Depends(get_ask_service)) -> AskResponse:
    """Answer in Sinhala from the selected video's comments only, citing the exact comments used.

    Returns status "insufficient_evidence" instead of guessing. 503 llm_unavailable when no LLM is
    configured; 502 llm_error / llm_invalid_output / llm_unsupported_citation when the model fails or
    cites comments it was not given (the answer is then withheld)."""
    return service.ask(body, request_id(request))
