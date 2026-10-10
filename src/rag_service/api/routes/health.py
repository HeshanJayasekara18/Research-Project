"""GET /v1/health: liveness/readiness for gateways. Unauthenticated; exposes no corpus content."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_service.api.dependencies import get_retrieval_service
from rag_service.api.schemas import ErrorResponse, HealthResponse
from rag_service.services.retrieval_service import RetrievalService

router = APIRouter(tags=['system'])


@router.get('/health', response_model=HealthResponse, operation_id='getHealth',
            responses={503: {'model': ErrorResponse}})
def health(service: RetrievalService = Depends(get_retrieval_service)) -> HealthResponse:
    """Liveness/readiness. Unauthenticated; exposes no corpus content."""
    return service.health()
