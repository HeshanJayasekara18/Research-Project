"""Versioned API routes, mounted under /v1 by rag_service.api.main."""
from fastapi import APIRouter

from rag_service.api.routes import health, retrieve

v1_router = APIRouter(prefix='/v1')
v1_router.include_router(health.router)
v1_router.include_router(retrieve.router)
