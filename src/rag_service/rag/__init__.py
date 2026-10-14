"""M6: minimal evidence-grounded Sinhala RAG on top of the existing retrieval engine.

evidence.py   conservative selection of retrieved candidates to send to the LLM
prompt.py     grounding rules + numbered evidence block ([E1], [E2], ...)
providers.py  LLM provider adapters (disabled by default; Gemini via REST)
ask_service.py orchestration: retrieve (scoped) -> select -> generate -> validate citations
"""
