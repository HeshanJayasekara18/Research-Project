"""POST /v1/retrieve. Handlers stay thin: validation in schemas, logic in services."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_service.api.dependencies import get_retrieval_service, require_service_key
from rag_service.api.errors import request_id
from rag_service.api.schemas import ErrorResponse, RetrieveRequest, RetrieveResponse
from rag_service.services.retrieval_service import RetrievalService

router = APIRouter(tags=['retrieval'])

_ERRORS = {code: {'model': ErrorResponse} for code in (401, 403, 404, 422, 500, 503)}


@router.post('/retrieve', response_model=RetrieveResponse, response_model_exclude_none=False,
             operation_id='retrieveComments', responses=_ERRORS, dependencies=[Depends(require_service_key)])
def retrieve(body: RetrieveRequest, request: Request,
             service: RetrievalService = Depends(get_retrieval_service)) -> RetrieveResponse:
    """Rank YouTube comments for a question, optionally scoped to one video (filter before ranking)."""
    return service.retrieve(body, request_id(request))
