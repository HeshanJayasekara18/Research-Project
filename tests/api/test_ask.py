"""POST /v1/ask grounding and citation tests.

ALL generation here uses FakeProvider (a MOCK). Nothing in this file calls a real LLM, and no
output here is a real model answer. Live demonstrations are separate (docs/RAG_M6.md).
"""
import json
import unittest

from fastapi.testclient import TestClient

from rag_service.api.main import create_app
from rag_service.core.config import Settings
from rag_service.rag.providers import DisabledProvider, LLMError, LLMUnavailable
from rag_service.retrieval.engine import RetrievalEngine

DEMO_VIDEO = '-3a4abtQH3E'
DEMO_QUERY = 'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?'
OTHER_VIDEO = 'QvkLXe99zEk'
OTHER_VIDEO_COMMENT = 'Slt nam jarawak. patta rental ekak gewanna ona heh'  # exact comment from OTHER_VIDEO


class FakeProvider:
    """MOCK provider for tests only: returns a scripted output and records what it was sent."""

    name = 'fake-test-provider'
    model = 'mock'

    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.output if isinstance(self.output, str) else json.dumps(self.output, ensure_ascii=False)


def answered(text='වීඩියෝවේ අය කියනවා වැටකොළු පොත්ත අයින් කරනවා කියලා [E1]', cites=('E1',)):
    return {'status': 'answered', 'answer': text, 'citations': list(cites)}


class AskTestBase(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        if AskTestBase.engine is None:
            AskTestBase.engine = RetrievalEngine()

    def client(self, provider):
        c = TestClient(create_app(Settings(env='test'), engine=self.engine, llm_provider=provider))
        c.__enter__()
        self.addCleanup(c.__exit__, None, None, None)
        return c

    def ask(self, provider, **body):
        body.setdefault('scope', {'video_id': DEMO_VIDEO})
        body.setdefault('method', 'multiview_bm25')  # deterministic, no LaBSE needed
        return self.client(provider).post('/v1/ask', json=body)

    def assert_error(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.text)
        self.assertEqual(resp.json()['error']['code'], code)


class TestValidation(AskTestBase):
    def test_scope_is_required_and_validated(self):
        p = FakeProvider(answered())
        client = self.client(p)
        for body in ({'query': DEMO_QUERY},                                     # no scope
                     {'query': DEMO_QUERY, 'scope': {'video_id': None}},        # global not allowed for /ask
                     {'query': DEMO_QUERY, 'scope': {}},
                     {'query': DEMO_QUERY, 'scope': {'video_id': '../x'}},
                     {'query': '   ', 'scope': {'video_id': DEMO_VIDEO}},
                     {'query': 'x' * 1001, 'scope': {'video_id': DEMO_VIDEO}},
                     {'query': DEMO_QUERY, 'scope': {'video_id': DEMO_VIDEO}, 'creator_id': 'UC1'},
                     {'query': DEMO_QUERY, 'scope': {'video_id': DEMO_VIDEO}, 'candidates': 2}):
            with self.subTest(body=body):
                self.assert_error(client.post('/v1/ask', json=body), 422, 'invalid_request')
        self.assertEqual(p.calls, [])

    def test_unknown_video(self):
        p = FakeProvider(answered())
        self.assert_error(self.ask(p, query=DEMO_QUERY, scope={'video_id': 'NOT_A_VIDEO'}), 404, 'video_not_found')
        self.assertEqual(p.calls, [])


class TestGrounding(AskTestBase):
    def test_answer_cites_only_supplied_real_comments_with_original_text(self):
        p = FakeProvider(answered())
        resp = self.ask(p, query=DEMO_QUERY)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body['status'], 'answered')
        self.assertEqual(len(p.calls), 1)
        supplied = {e['label']: e for e in body['evidence_supplied']}
        _, prompt = p.calls[0]
        for e in body['evidence_supplied']:
            doc = self.engine.by_id[e['document_id']]                       # real document IDs only
            self.assertEqual(e['comment_text'], doc['comment_text'])        # original text, unmodified
            self.assertEqual(e['video_id'], DEMO_VIDEO)
            self.assertEqual(e['video_reference']['granularity'], 'video')
            self.assertIn(f"[{e['label']}] {json.dumps(doc['comment_text'], ensure_ascii=False)}", prompt)
        self.assertEqual([c['label'] for c in body['citations']], ['E1'])
        self.assertTrue(all(c == supplied[c['label']] for c in body['citations']))
        # The Wetakolu peel/ridge comments are selected; generic praise is not.
        ids = {e['document_id'] for e in body['evidence_supplied']}
        self.assertTrue({'DB8B1DE7D2790', 'DD488D4D88EF8'} <= ids)
        texts = ' '.join(e['comment_text'] for e in body['evidence_supplied']).lower()
        self.assertNotIn('nice video', texts)

    def test_cross_video_isolation(self):
        p = FakeProvider(answered(cites=()))  # citations come from inline labels below
        p.output = answered('ලොකු දෙයක් නැහැ [E1]', ())
        body = self.ask(p, query=OTHER_VIDEO_COMMENT).json()
        self.assertTrue(all(e['video_id'] == DEMO_VIDEO for e in body['evidence_supplied']))
        if p.calls:  # the question is that comment's text, so only the evidence section must not contain it
            self.assertNotIn(OTHER_VIDEO_COMMENT, p.calls[0][1].split('Comments:', 1)[1])
        self.assertNotIn(OTHER_VIDEO, json.dumps(body))

    def test_preserves_original_comment_text_exactly(self):
        p = FakeProvider(answered())
        body = self.ask(p, query=DEMO_QUERY).json()
        original = 'Api uyanakota vatakolu vala dara ain karanava.hethuva nam danne na'
        self.assertIn(original, [e['comment_text'] for e in body['evidence_supplied']])


class TestInsufficientEvidence(AskTestBase):
    def test_unrelated_topic_returns_insufficient_without_calling_model(self):
        p = FakeProvider(answered())
        body = self.ask(p, query='Me video eke iPhone battery eka gana viewers la monawada kiyanne?').json()
        self.assertEqual(body['status'], 'insufficient_evidence')
        self.assertEqual(body['insufficient_reason'], 'no_content_term_match')
        self.assertIsNone(body['answer'])
        self.assertEqual(body['citations'], [])
        self.assertFalse(body['generation']['called'])
        self.assertEqual(p.calls, [])

    def test_adversarial_generic_terms_only(self):
        # Only generic words ("video", "eka", "viewers", "monawada"): they match many comments
        # ("Nice video", ...) but must never become evidence.
        p = FakeProvider(answered())
        body = self.ask(p, query='Me video eka gana viewers la monawada kiyanne?').json()
        self.assertEqual(body['status'], 'insufficient_evidence')
        self.assertEqual(body['retrieval']['query_content_terms'], [])
        self.assertGreater(body['retrieval']['candidates_retrieved'], 0)
        self.assertEqual(body['evidence_supplied'], [])
        self.assertEqual(p.calls, [])

    def test_model_declines(self):
        p = FakeProvider({'status': 'insufficient_evidence', 'answer': '', 'citations': []})
        body = self.ask(p, query=DEMO_QUERY).json()
        self.assertEqual(body['status'], 'insufficient_evidence')
        self.assertEqual(body['insufficient_reason'], 'model_declined')
        self.assertIsNone(body['answer'])
        self.assertTrue(body['generation']['called'])


class TestCitationValidation(AskTestBase):
    def test_unknown_citation_label_rejected(self):
        self.assert_error(self.ask(FakeProvider(answered(cites=('E1', 'E99'))), query=DEMO_QUERY),
                          502, 'llm_unsupported_citation')

    def test_unknown_inline_label_rejected(self):
        p = FakeProvider(answered('එක් අයෙක් කියනවා [E1] සහ [E42]', ('E1',)))
        self.assert_error(self.ask(p, query=DEMO_QUERY), 502, 'llm_unsupported_citation')

    def test_fabricated_document_id_style_citation_rejected(self):
        p = FakeProvider(answered('එක් අයෙක් කියනවා', ('DB8B1DE7D2790',)))
        self.assert_error(self.ask(p, query=DEMO_QUERY), 502, 'llm_unsupported_citation')

    def test_answer_without_citations_rejected(self):
        p = FakeProvider(answered('එක් අයෙක් කියනවා පොත්ත අයින් කරනවා', ()))
        self.assert_error(self.ask(p, query=DEMO_QUERY), 502, 'llm_unsupported_citation')

    def test_non_json_output_rejected(self):
        self.assert_error(self.ask(FakeProvider('Sure! Viewers say...'), query=DEMO_QUERY), 502, 'llm_invalid_output')

    def test_bad_status_rejected(self):
        p = FakeProvider({'status': 'maybe', 'answer': 'x [E1]', 'citations': ['E1']})
        self.assert_error(self.ask(p, query=DEMO_QUERY), 502, 'llm_invalid_output')


class TestProviderFailures(AskTestBase):
    def test_disabled_provider_is_503_and_reported_in_health(self):
        p = DisabledProvider('LLM generation is disabled.')
        client = self.client(p)
        self.assert_error(client.post('/v1/ask', json={'query': DEMO_QUERY, 'scope': {'video_id': DEMO_VIDEO}}),
                          503, 'llm_unavailable')
        llm = client.get('/v1/health').json()['llm']
        self.assertEqual(llm, {'provider': 'disabled', 'model': '', 'enabled': False})

    def test_provider_error_is_502(self):
        self.assert_error(self.ask(FakeProvider(error=LLMError('boom')), query=DEMO_QUERY), 502, 'llm_error')

    def test_provider_unavailable_midway_is_503(self):
        self.assert_error(self.ask(FakeProvider(error=LLMUnavailable('quota')), query=DEMO_QUERY), 503, 'llm_unavailable')


if __name__ == '__main__':
    unittest.main()
