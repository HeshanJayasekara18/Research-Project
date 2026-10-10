"""Versioned routes. Handlers stay thin: validation in schemas, logic in services."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from service.dependencies import get_retrieval_service, require_service_key
from service.errors import request_id
from service.schemas import ErrorResponse, HealthResponse, RetrieveRequest, RetrieveResponse
from service.services.retrieval_service import RetrievalService

router = APIRouter(prefix='/v1')

_ERRORS = {code: {'model': ErrorResponse} for code in (401, 403, 404, 422, 500, 503)}


@router.get('/health', response_model=HealthResponse, tags=['system'], operation_id='getHealth',
            responses={503: {'model': ErrorResponse}})
def health(service: RetrievalService = Depends(get_retrieval_service)) -> HealthResponse:
    """Liveness/readiness. Unauthenticated; exposes no corpus content."""
    return service.health()


@router.post('/retrieve', response_model=RetrieveResponse, response_model_exclude_none=False, tags=['retrieval'],
             operation_id='retrieveComments', responses=_ERRORS, dependencies=[Depends(require_service_key)])
def retrieve(body: RetrieveRequest, request: Request,
             service: RetrievalService = Depends(get_retrieval_service)) -> RetrieveResponse:
    """Rank YouTube comments for a question, optionally scoped to one video (filter before ranking)."""
    return service.retrieve(body, request_id(request))
