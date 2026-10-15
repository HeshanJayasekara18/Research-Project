"""Corpus V2 integration tests on a SYNTHETIC fixture (invented comments; no real data).

Covers loading, unique IDs, thread metadata, domain/video isolation before ranking,
BM25 + normalized BM25, V1 backward compatibility and invalid scopes.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rag_service.api.main import create_app
from rag_service.core.config import Settings
from rag_service.retrieval.corpus_v2 import CorpusV2Error, build_engine_v2, load_records
from rag_service.retrieval.engine import RetrievalEngine

FIELDS = ['comment_id', 'video_id', 'domain', 'parent_id', 'thread_id', 'is_reply', 'published_at', 'likes',
          'total_reply_count', 'youtube_video_url', 'collected_at', 'source_file', 'text_status', 'indexable', 'comment_text']


def row(cid, vid, domain, text, parent='', indexable=True, status='ok'):
    return {'comment_id': cid, 'video_id': vid, 'domain': domain, 'parent_id': parent, 'thread_id': parent or cid,
            'is_reply': 'true' if parent else 'false', 'published_at': '', 'likes': '1', 'total_reply_count': '0',
            'youtube_video_url': f'https://www.youtube.com/watch?v={vid}', 'collected_at': '2026-10-15',
            'source_file': f'{domain}_comments.csv', 'text_status': status,
            'indexable': 'true' if indexable else 'false', 'comment_text': text if indexable or status == 'no_letters' else ''}


# Every domain mentions "recipe"/"video" so domain filtering is what keeps results apart.
SYNTHETIC = [
    row('Fc1', 'vidFOOD0001', 'food', 'kiri bath recipe eka hari rasai pol kiri wadiya dammama'),
    row('Fc2', 'vidFOOD0001', 'food', '@user kiri bath recipe eka mamath hadala balanawa', parent='Fc1'),
    row('Fc3', 'vidFOOD0001', 'food', '[withheld]', parent='Fc1', indexable=False, status='withheld_privacy'),
    row('Fc4', 'vidFOOD0001', 'food', '❤', indexable=False, status='no_letters'),
    row('Fc5', 'vidFOOD0002', 'food', 'kottu recipe eka video eke lassanata pennuwa'),
    row('Fc6', 'vidFOOD0002', 'food', 'nice video'),
    row('Fc7', 'vidFOOD0002', 'food', 'nice video'),  # same text, distinct comment
    row('Gc1', 'vidGAME0001', 'gaming', 'pubg recipe kiyala gameplay video eka ela'),
    row('Gc2', 'vidGAME0001', 'gaming', 'free fire gameplay eka maru headshot', parent='Gc1'),
    row('Gc3', 'vidGAME0002', 'gaming', 'gta gameplay video eka hari lassanai'),
    row('Ec1', 'vidEDUC0001', 'education', 'physics lesson eka recipe ekak wage pahadili video'),
    row('Ec2', 'vidEDUC0001', 'education', 'physics notes thiyenawada kiyala ahanawa', parent='Ec1'),
    row('Ec3', 'vidEDUC0002', 'education', 'ict lesson video eka hondai'),
    row('Ec4', 'vidEDUC0002', 'education', 'ප්‍රශ්න පත්‍රය ගැන video එකක් කරන්න'),
]


def write_corpus(rows):
    d = Path(tempfile.mkdtemp())
    with (d / 'comments_v2.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return d


class V2Base(unittest.TestCase):
    v1 = None

    @classmethod
    def setUpClass(cls):
        if V2Base.v1 is None:
            V2Base.v1 = RetrievalEngine()
        cls.v2_dir = write_corpus(SYNTHETIC)
        cls.v2 = build_engine_v2(cls.v2_dir)

    def client(self, with_v2=True):
        c = TestClient(create_app(Settings(env='test'), engine=self.v1, engine_v2=self.v2 if with_v2 else None))
        c.__enter__()
        self.addCleanup(c.__exit__, None, None, None)
        return c

    def retrieve(self, client=None, **body):
        return (client or self.client()).post('/v1/retrieve', json=body)


class TestLoading(V2Base):
    def test_indexable_only_but_all_records_kept(self):
        self.assertEqual(len(self.v2.docs), 12)                        # 14 records - 2 non-indexable
        self.assertEqual(len(self.v2.thread_records), 14)
        self.assertNotIn('Fc3', self.v2.by_id)
        self.assertEqual(self.v2.thread_records['Fc3']['parent_id'], 'Fc1')  # withheld reply still in the thread
        self.assertEqual(self.v2.thread_report['replies_with_missing_parent'], 0)
        self.assertEqual(self.v2.thread_report['replies_parent_on_other_video'], 0)

    def test_duplicate_texts_are_distinct_documents(self):
        self.assertEqual({'Fc6', 'Fc7'} <= set(self.v2.by_id), True)

    def test_unique_comment_ids_enforced(self):
        with self.assertRaises(CorpusV2Error):
            load_records(write_corpus(SYNTHETIC + [row('Fc1', 'vidFOOD0001', 'food', 'dup id')]))

    def test_missing_columns_and_bad_domain_rejected(self):
        bad = write_corpus([dict(SYNTHETIC[0], domain='music')])
        with self.assertRaises(CorpusV2Error):
            load_records(bad)
        with self.assertRaises(CorpusV2Error):
            load_records(Path(tempfile.mkdtemp()))


class TestScopedRetrieval(V2Base):
    def test_domain_isolation_before_ranking(self):
        for domain in ('food', 'gaming', 'education'):
            with self.subTest(domain=domain):
                body = self.retrieve(query='recipe video eka', top_k=20, scope={'corpus': 'v2', 'domain': domain}).json()
                self.assertTrue(body['results'])
                self.assertEqual({r['domain'] for r in body['results']}, {domain})
                self.assertEqual(body['scope']['type'], 'domain')
                self.assertEqual(body['scope']['documents_searched'], len(self.v2.domain_rows[domain]))

    def test_video_isolation(self):
        body = self.retrieve(query='recipe video eka', top_k=20, scope={'corpus': 'v2', 'video_id': 'vidFOOD0002'}).json()
        self.assertEqual({r['video_id'] for r in body['results']}, {'vidFOOD0002'})

    def test_domain_and_video_together(self):
        ok = self.retrieve(query='gameplay', scope={'corpus': 'v2', 'domain': 'gaming', 'video_id': 'vidGAME0001'})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual({r['video_id'] for r in ok.json()['results']}, {'vidGAME0001'})
        conflict = self.retrieve(query='gameplay', scope={'corpus': 'v2', 'domain': 'food', 'video_id': 'vidGAME0001'})
        self.assertEqual((conflict.status_code, conflict.json()['error']['code']), (400, 'scope_conflict'))

    def test_bm25_and_normalized_bm25_return_v2_metadata(self):
        for method in ('bm25', 'normalized_bm25', 'multiview_bm25', 'char_tfidf'):
            with self.subTest(method=method):
                resp = self.retrieve(query='kiri bath recipe', method=method, scope={'corpus': 'v2', 'domain': 'food'})
                self.assertEqual(resp.status_code, 200, resp.text)
                results = resp.json()['results']
                self.assertTrue(results)
                top = results[0]
                self.assertEqual(top['comment_id'], top['document_id'])          # real comment ID
                self.assertIn(top['comment_id'], {'Fc1', 'Fc2'})
                reply = [r for r in results if r['comment_id'] == 'Fc2']
                if reply:
                    self.assertEqual((reply[0]['is_reply'], reply[0]['parent_id'], reply[0]['thread_id']), (True, 'Fc1', 'Fc1'))

    def test_non_indexable_never_returned(self):
        body = self.retrieve(query='withheld ❤', top_k=50, scope={'corpus': 'v2', 'domain': 'food'}).json()
        self.assertFalse({'Fc3', 'Fc4'} & {r['comment_id'] for r in body['results']})

    def test_labse_unavailable_on_v2_without_index(self):
        resp = self.retrieve(query='recipe', method='labse', scope={'corpus': 'v2', 'domain': 'food'})
        self.assertEqual((resp.status_code, resp.json()['error']['code']), (503, 'method_unavailable'))


class TestCompatibilityAndInvalidScopes(V2Base):
    def test_v1_default_unchanged(self):
        body = self.retrieve(query='Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?',
                             top_k=2, scope={'video_id': '-3a4abtQH3E'}).json()
        self.assertEqual(body['scope']['corpus'], 'v1')
        self.assertEqual({r['document_id'] for r in body['results']}, {'DB8B1DE7D2790', 'DD488D4D88EF8'})
        self.assertTrue(all(r['comment_id'] is None and r['parent_id'] is None for r in body['results']))

    def test_v2_requested_but_not_loaded(self):
        resp = self.retrieve(self.client(with_v2=False), query='recipe', scope={'corpus': 'v2'})
        self.assertEqual((resp.status_code, resp.json()['error']['code']), (503, 'corpus_unavailable'))

    def test_domain_on_v1_rejected(self):
        resp = self.retrieve(query='recipe', scope={'domain': 'food'})
        self.assertEqual((resp.status_code, resp.json()['error']['code']), (404, 'domain_not_found'))

    def test_invalid_scope_values(self):
        client = self.client()
        for scope in ({'corpus': 'v3'}, {'corpus': 'v2', 'domain': 'music'}, {'corpus': 'v2', 'owner': 'me'}):
            with self.subTest(scope=scope):
                resp = self.retrieve(client, query='recipe', scope=scope)
                self.assertEqual((resp.status_code, resp.json()['error']['code']), (422, 'invalid_request'))
        resp = self.retrieve(client, query='recipe', scope={'corpus': 'v2', 'video_id': 'vidFOOD9999'})
        self.assertEqual((resp.status_code, resp.json()['error']['code']), (404, 'video_not_found'))

    def test_health_reports_both_corpora(self):
        corpora = self.client().get('/v1/health').json()['corpora']
        self.assertEqual(set(corpora), {'v1', 'v2'})
        self.assertEqual(corpora['v2']['domains'], {'food': 5, 'gaming': 3, 'education': 4})  # 7 food records, 2 non-indexable
        self.assertIsNone(corpora['v1']['domains'])


class FakeProvider:
    """MOCK provider (tests only)."""
    name, model = 'fake-test-provider', 'mock'

    def __init__(self):
        self.calls = []

    def generate(self, system, user):
        self.calls.append(user)
        return json.dumps({'status': 'answered', 'answer': 'එක් අයෙක් කියනවා kiri bath hondai [E1]', 'citations': ['E1']},
                          ensure_ascii=False)


class TestAskOnV2(V2Base):
    def test_ask_uses_v2_video_and_real_comment_ids(self):
        p = FakeProvider()
        c = TestClient(create_app(Settings(env='test'), engine=self.v1, engine_v2=self.v2, llm_provider=p))
        with c:
            resp = c.post('/v1/ask', json={'query': 'kiri bath recipe eka gana monawada kiyanne?', 'method': 'bm25',
                                           'scope': {'corpus': 'v2', 'video_id': 'vidFOOD0001'}})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body['retrieval']['scope']['corpus'], 'v2')
        self.assertTrue(all(e['video_id'] == 'vidFOOD0001' for e in body['evidence_supplied']))
        self.assertTrue(all(e['document_id'] in self.v2.by_id for e in body['evidence_supplied']))
        self.assertNotIn('[withheld]', p.calls[0])


if __name__ == '__main__':
    unittest.main()
