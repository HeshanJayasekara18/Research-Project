"""HTTP contract tests for the FastAPI service (GET /v1/health, POST /v1/retrieve)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
from fastapi.testclient import TestClient

from rag_service.retrieval.engine import MV_BM25_METHOD, RetrievalEngine
from rag_service.core.config import Settings
from rag_service.api.main import create_app

DEMO_VIDEO = '-3a4abtQH3E'
DEMO_QUERY = 'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?'
OTHER_VIDEO_QUERY = 'Slt nam jarawak. patta rental ekak gewanna ona heh'  # exact comment from QvkLXe99zEk
KEY = 'test-service-key'


class ApiTestBase(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        if ApiTestBase.engine is None:
            ApiTestBase.engine = RetrievalEngine()

    def client(self, **settings):
        client = TestClient(create_app(Settings(env='test', **settings), engine=self.engine))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def assert_error(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.text)
        err = resp.json()['error']
        self.assertEqual(err['code'], code)
        self.assertTrue(err['message'])
        self.assertEqual(err['request_id'], resp.headers['X-Request-ID'])


class TestHealth(ApiTestBase):
    def test_health(self):
        body = self.client(service_api_key=KEY).get('/v1/health').json()  # no key needed
        self.assertEqual(body['status'], 'ok')
        self.assertEqual(body['api_version'], 'v1')
        self.assertEqual(body['corpus_documents'], len(self.engine.docs))
        self.assertTrue(body['auth_enabled'])
        self.assertIn('index_present', body['semantic'])

    def test_unknown_route_uses_envelope(self):
        self.assert_error(self.client().get('/v1/nope'), 404, 'not_found')


class TestRetrieve(ApiTestBase):
    def post(self, client=None, **body):
        return (client or self.client()).post('/v1/retrieve', json=body)

    def test_bm25_matches_engine_exactly(self):
        resp = self.post(query=DEMO_QUERY, method='bm25', top_k=10, scope={'video_id': DEMO_VIDEO})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        expected = self.engine.search(DEMO_QUERY, 'BM25 (lexical)', 10, DEMO_VIDEO)
        self.assertEqual([(r['document_id'], r['score']) for r in body['results']],
                         [(d['document_id'], s) for d, s in expected])
        self.assertEqual([r['rank'] for r in body['results']], list(range(1, len(expected) + 1)))
        self.assertEqual({r['document_id'] for r in body['results'][:2]}, {'DB8B1DE7D2790', 'DD488D4D88EF8'})
        # Original V1 scope fields unchanged; corpus/domain were added (additive contract change).
        self.assertEqual(body['scope'], {'type': 'video', 'video_id': DEMO_VIDEO,
                                         'documents_searched': len(self.engine.video_rows[DEMO_VIDEO]),
                                         'ownership_verified': False, 'corpus': 'v1', 'domain': None})
        self.assertEqual(body['result_count'], len(body['results']))
        self.assertIn('not human-verified evidence', body['notice'])

    def test_default_method_is_bm25(self):
        self.assertEqual(self.post(query=DEMO_QUERY).json()['method'], 'bm25')

    def test_video_scope_never_leaks_other_videos(self):
        for method in ('bm25', 'char_tfidf', 'normalized_bm25', 'multiview_bm25'):
            with self.subTest(method=method):
                body = self.post(query=OTHER_VIDEO_QUERY, method=method, top_k=50,
                                 scope={'video_id': DEMO_VIDEO}).json()
                self.assertTrue(all(r['video_id'] == DEMO_VIDEO for r in body['results']))

    def test_global_scope(self):
        body = self.post(query=OTHER_VIDEO_QUERY, top_k=1).json()
        self.assertEqual(body['scope']['type'], 'global')
        self.assertEqual(body['results'][0]['video_id'], 'QvkLXe99zEk')

    def test_global_scope_can_be_disabled(self):
        self.assert_error(self.post(self.client(allow_global_scope=False), query='video'), 403, 'global_scope_disabled')

    def test_multiview_matches_engine_and_exposes_views(self):
        body = self.post(query=DEMO_QUERY, method='multiview_bm25', top_k=5, scope={'video_id': DEMO_VIDEO}).json()
        hits, info = self.engine.multiview(DEMO_QUERY, MV_BM25_METHOD, 5, video_id=DEMO_VIDEO)
        self.assertEqual([r['document_id'] for r in body['results']], [h.doc_id for _, h in hits])
        self.assertTrue(all(r['score_type'] == 'rrf' and r['views'] for r in body['results']))
        self.assertEqual(body['normalization']['used'], info['normalized_view_used'])

    def test_normalized_method_reports_normalization(self):
        body = self.post(query='wetakolu poththa ain karanna oneda', method='normalized_bm25').json()
        self.assertIn(body['normalization']['status'], {'normalized', 'partial', 'failed', 'unchanged', 'empty'})
        self.assertEqual({r['score_type'] for r in body['results']} - {'bm25'}, set())

    def test_no_results_is_200_with_empty_list(self):
        body = self.post(query='qwxzplugh', scope={'video_id': DEMO_VIDEO}).json()
        self.assertEqual((body['results'], body['result_count']), ([], 0))

    def test_unknown_video(self):
        self.assert_error(self.post(query='video', scope={'video_id': 'NOT_A_VIDEO'}), 404, 'video_not_found')

    def test_validation_errors(self):
        bad = [
            {'query': '   '},
            {'query': ''},
            {'query': 'x' * 1001},
            {'query': 'video', 'top_k': 0},
            {'query': 'video', 'top_k': 51},
            {'query': 'video', 'method': 'BM25 (lexical)'},              # engine names are not API values
            {'query': 'video', 'scope': {'video_id': '../etc'}},
            {'query': 'video', 'creator_id': 'UC123'},                    # unknown fields rejected
            {'query': 'video', 'scope': {'video_id': DEMO_VIDEO, 'owner': 'me'}},
        ]
        client = self.client()
        for body in bad:
            with self.subTest(body=body):
                self.assert_error(client.post('/v1/retrieve', json=body), 422, 'invalid_request')

    def test_semantic_method_unavailable_without_index(self):
        with mock.patch.object(self.engine, 'has_semantic_index', return_value=False):
            for method in ('labse', 'normalized_labse', 'multiview_labse', 'multiview_hybrid'):
                with self.subTest(method=method):
                    self.assert_error(self.post(query='video', method=method), 503, 'method_unavailable')
            self.assertEqual(self.post(query='video', method='bm25').status_code, 200)

    def test_request_id_propagated(self):
        resp = self.client().post('/v1/retrieve', json={'query': 'video'}, headers={'X-Request-ID': 'abc-123'})
        self.assertEqual(resp.headers['X-Request-ID'], 'abc-123')
        self.assertEqual(resp.json()['request_id'], 'abc-123')


class TestAuth(ApiTestBase):
    def test_service_key_required_when_configured(self):
        client = self.client(service_api_key=KEY)
        self.assert_error(client.post('/v1/retrieve', json={'query': 'video'}), 401, 'unauthorized')
        self.assert_error(client.post('/v1/retrieve', json={'query': 'video'}, headers={'X-Service-Key': 'wrong'}),
                          401, 'unauthorized')
        ok = client.post('/v1/retrieve', json={'query': 'video'}, headers={'X-Service-Key': KEY})
        self.assertEqual(ok.status_code, 200)

    def test_auth_checked_before_scope(self):
        # An unauthenticated caller must not learn which video IDs exist.
        resp = self.client(service_api_key=KEY).post('/v1/retrieve',
                                                     json={'query': 'v', 'scope': {'video_id': 'NOT_A_VIDEO'}})
        self.assert_error(resp, 401, 'unauthorized')

    def test_production_requires_key(self):
        with self.assertRaises(RuntimeError):
            create_app(Settings(env='production'), engine=self.engine)


class TestContract(unittest.TestCase):
    def test_committed_openapi_is_current(self):
        from scripts.export_openapi import OUT, render
        self.assertTrue(OUT.exists(), 'run: python scripts/export_openapi.py')
        self.assertEqual(OUT.read_text(encoding='utf-8'), render(),
                         'API contract changed: review it, then run python scripts/export_openapi.py')

    def test_paths(self):
        from rag_service.core.config import Settings
        paths = create_app(Settings(), engine=object()).openapi()['paths']
        self.assertEqual(set(paths), {'/v1/health', '/v1/retrieve', '/v1/ask'})


if __name__ == '__main__':
    unittest.main()
