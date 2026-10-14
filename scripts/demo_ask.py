"""LIVE /v1/ask demonstrations with the configured LLM (not a test; makes real model calls).

    $env:RAG_LLM_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "<your key>"
    python scripts/demo_ask.py

Refuses to run when the LLM is disabled, so no simulated answer is ever printed.
Only selected public, author-stripped comments are sent to the provider.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

DEMOS = [
    ('Wetakolu creator question (video-scoped)',
     'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?', '-3a4abtQH3E'),
    ('Cross-script cooking question (Sinhala-script evidence)',
     'Bath eka hadaddi wathura damme nathi eka ganayi, pawiichchi karapu hal wargaya ganayi ahanawada?', '_uhQvDg6VtA'),
    ('Unrelated topic (should NOT produce an answer)',
     'Me video eke iPhone battery eka gana viewers la monawada kiyanne?', '-3a4abtQH3E'),
]


def main() -> int:
    from fastapi.testclient import TestClient

    from rag_service.api.main import create_app
    from rag_service.core.config import load_settings
    from rag_service.rag.providers import DisabledProvider, build_provider

    settings = load_settings()
    provider = build_provider(settings)
    if isinstance(provider, DisabledProvider):
        print(f'LLM not available: {provider.reason}\nNo demo run (no simulated answers).')
        return 2
    print(f'LIVE model: {provider.name} / {provider.model}\n')
    with TestClient(create_app(settings, llm_provider=provider)) as client:
        for title, query, video in DEMOS:
            resp = client.post('/v1/ask', json={'query': query, 'scope': {'video_id': video}},
                               headers={'X-Service-Key': settings.service_api_key or ''})
            body = resp.json()
            print('=' * 80, f'\n{title}\nQ: {query}\nvideo: {video}  HTTP {resp.status_code}')
            if resp.status_code != 200:
                print(json.dumps(body, ensure_ascii=False, indent=2))
                continue
            r = body['retrieval']
            print(f"method={r['method']} searched={r['scope']['documents_searched']} retrieved={r['candidates_retrieved']} "
                  f"selected={r['candidates_selected']} rejected_no_content_match={r['candidates_rejected_no_content_match']}")
            print(f"content terms: {r['query_content_terms']}")
            for e in body['evidence_supplied']:
                print(f"  [{e['label']}] {e['document_id']} (rank {e['retrieval_rank']}, matched {e['matched_terms']}): {e['comment_text']}")
            print(f"status: {body['status']}  reason: {body['insufficient_reason']}  model called: {body['generation']['called']}")
            if body['answer']:
                print(f"ANSWER ({body['generation']['provider']}/{body['generation']['model']}): {body['answer']}")
                print('cited:', [(c['label'], c['document_id']) for c in body['citations']])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
