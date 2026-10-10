"""FastAPI service layer for the Singlish Retrieval Engine.

The service is a thin HTTP adapter: all retrieval, normalisation and RRF logic
stays in the research modules (retrieval.py, normalization.py, fusion.py).
"""
API_VERSION = 'v1'
SERVICE_VERSION = '0.1.0'
