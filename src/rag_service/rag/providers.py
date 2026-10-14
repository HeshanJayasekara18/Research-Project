"""LLM provider adapters.

Only real providers are selectable through configuration. There is deliberately no
"mock"/"echo" provider here: automated tests inject their own fake provider object, so a
simulated answer can never be presented as real model output.
"""
from __future__ import annotations

import json
from typing import Protocol

import httpx

from rag_service.core.config import Settings


class LLMUnavailable(Exception):
    """The LLM is disabled or not configured (no request was made)."""


class LLMError(Exception):
    """The provider was called but failed (network, HTTP error, blocked/empty output)."""


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> str:
        """Return the raw model text for one grounded-answer request."""
        ...


class DisabledProvider:
    name = 'disabled'
    model = ''

    def __init__(self, reason: str):
        self.reason = reason

    def generate(self, system: str, user: str) -> str:
        raise LLMUnavailable(self.reason)


class GeminiProvider:
    """Google Gemini via the public REST API (generateContent), JSON output, low temperature."""

    name = 'gemini'
    URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0, client: httpx.Client | None = None):
        self._key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout_s)

    def generate(self, system: str, user: str) -> str:
        body = {
            'systemInstruction': {'parts': [{'text': system}]},
            'contents': [{'role': 'user', 'parts': [{'text': user}]}],
            'generationConfig': {'temperature': 0.1, 'responseMimeType': 'application/json', 'maxOutputTokens': 1024},
        }
        try:
            resp = self._client.post(self.URL.format(model=self.model), json=body,
                                     headers={'x-goog-api-key': self._key, 'Content-Type': 'application/json'})
        except httpx.HTTPError as exc:
            raise LLMError(f'Gemini request failed: {type(exc).__name__}') from exc
        if resp.status_code != 200:
            # Never echo the request (it carries the key header); report status + provider message only.
            try:
                msg = resp.json().get('error', {}).get('message', '')[:200]
            except (ValueError, AttributeError):
                msg = ''
            raise LLMError(f'Gemini HTTP {resp.status_code}: {msg}'.strip())
        try:
            data = resp.json()
            parts = data['candidates'][0]['content']['parts']
            text = ''.join(p.get('text', '') for p in parts).strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            reason = ''
            try:
                reason = json.dumps(data.get('promptFeedback') or data['candidates'][0].get('finishReason'))
            except Exception:
                pass
            raise LLMError(f'Gemini returned no usable text {reason}'.strip()) from exc
        if not text:
            raise LLMError('Gemini returned empty text.')
        return text


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider in ('', 'disabled', 'none'):
        return DisabledProvider('LLM generation is disabled (set RAG_LLM_PROVIDER=gemini and an API key).')
    if settings.llm_provider == 'gemini':
        if not settings.llm_api_key:
            return DisabledProvider('RAG_LLM_PROVIDER=gemini but no RAG_LLM_API_KEY / GEMINI_API_KEY is set.')
        return GeminiProvider(settings.llm_api_key, settings.llm_model, settings.llm_timeout_s)
    return DisabledProvider(f'Unsupported RAG_LLM_PROVIDER {settings.llm_provider!r} (supported: gemini, disabled).')
