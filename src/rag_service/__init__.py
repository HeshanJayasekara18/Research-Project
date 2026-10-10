"""C3 Explainable RAG Knowledge Assistant & Singlish Retrieval Engine.

Layout:
  retrieval/  research retrieval engine (BM25, char TF-IDF, LaBSE), Singlish normalisation, RRF
  services/   application layer: maps the API contract onto the retrieval engine
  api/        FastAPI app, versioned routes and Pydantic schemas (HTTP adapter only)
  core/       settings and repository paths
"""
API_VERSION = 'v1'
SERVICE_VERSION = '0.1.0'
