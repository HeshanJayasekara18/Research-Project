"""POST /v1/retrieve. Handlers stay thin: validation in schemas, logic in services."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_service.api.dependencies import require_service_key, retrieval_service_for
from rag_service.api.errors import request_id
from rag_service.api.schemas import ErrorResponse, RetrieveRequest, RetrieveResponse

router = APIRouter(tags=['retrieval'])

_ERRORS = {code: {'model': ErrorResponse} for code in (401, 403, 404, 422, 500, 503)}


@router.post('/retrieve', response_model=RetrieveResponse, response_model_exclude_none=False,
             operation_id='retrieveComments', responses=_ERRORS, dependencies=[Depends(require_service_key)])
def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    """Rank YouTube comments for a question. Scope (corpus, domain, video) is applied before ranking."""
    return retrieval_service_for(request, body.scope.corpus).retrieve(body, request_id(request))
