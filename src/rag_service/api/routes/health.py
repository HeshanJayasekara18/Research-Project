"""GET /v1/health: liveness/readiness for gateways. Unauthenticated; exposes no corpus content."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_service.api.dependencies import get_retrieval_service
from rag_service.api.schemas import ErrorResponse, HealthResponse, LLMStatus
from rag_service.rag.providers import DisabledProvider
from rag_service.services.retrieval_service import RetrievalService

router = APIRouter(tags=['system'])


@router.get('/health', response_model=HealthResponse, operation_id='getHealth',
            responses={503: {'model': ErrorResponse}})
def health(request: Request, service: RetrievalService = Depends(get_retrieval_service)) -> HealthResponse:
    """Liveness/readiness. Unauthenticated; exposes no corpus content and never calls the LLM."""
    result = service.health()
    ask = getattr(request.app.state, 'ask_service', None)
    if ask is not None:
        p = ask.provider
        result.llm = LLMStatus(provider=p.name, model=p.model, enabled=not isinstance(p, DisabledProvider))
    return result
