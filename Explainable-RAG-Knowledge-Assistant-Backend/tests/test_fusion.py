import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fusion import rrf
from retrieval import MV_BM25_METHOD, RetrievalEngine


class TestRRF(unittest.TestCase):
    def test_scores_match_formula(self):
        fused = {h.doc_id: h for h in rrf({'a': [('d1', 9.0), ('d2', 5.0)], 'b': [('d2', 0.8)]}, k=60)}
        self.assertAlmostEqual(fused['d1'].rrf_score, 1 / 61)
        self.assertAlmostEqual(fused['d2'].rrf_score, 1 / 62 + 1 / 61)
        self.assertEqual(fused['d2'].views, {'a': (2, 5.0), 'b': (1, 0.8)})

    def test_ranking_rewards_agreement(self):
        out = rrf({'a': [('x', 1), ('y', 1)], 'b': [('z', 1), ('y', 1)]})
        self.assertEqual(out[0].doc_id, 'y')  # rank 2 in both beats rank 1 in one

    def test_deduplicates_ids(self):
        out = rrf({'a': [('d1', 3), ('d1', 2), ('d2', 1)], 'b': [('d1', 1)]})
        self.assertEqual([h.doc_id for h in out].count('d1'), 1)
        self.assertEqual(out[0].views['a'], (1, 3.0))   # duplicate keeps its first (best) rank
        self.assertEqual(next(h for h in out if h.doc_id == 'd2').views['a'][0], 2)

    def test_empty_inputs(self):
        self.assertEqual(rrf({}), [])
        self.assertEqual(rrf({'a': [], 'b': []}), [])

    def test_deterministic_tie_break(self):
        out = rrf({'a': [('b', 1)], 'c': [('a', 1)]})
        self.assertEqual([h.doc_id for h in out], ['a', 'b'])  # equal score & rank -> by doc ID

    def test_k_controls_and_validates(self):
        self.assertAlmostEqual(rrf({'a': [('d', 1)]}, k=0)[0].rrf_score, 1.0)
        with self.assertRaises(ValueError):
            rrf({'a': [('d', 1)]}, k=-1)


class TestMultiViewEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RetrievalEngine()

    def test_uses_both_views_when_normalization_succeeds(self):
        hits, info = self.engine.multiview('Kaha bath ekata kanna monawada?', MV_BM25_METHOD, 10)
        self.assertTrue(info['normalized_view_used'])
        self.assertEqual(info['views'], ['BM25/original', 'BM25/normalized'])
        ids = [d['document_id'] for d, _ in hits]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), 10)
        self.assertEqual([h.rrf_score for _, h in hits], sorted((h.rrf_score for _, h in hits), reverse=True))

    def test_falls_back_when_normalization_fails(self):
        hits, info = self.engine.multiview('qwxz plugh', MV_BM25_METHOD, 10)
        self.assertEqual(info['normalization'].status, 'failed')
        self.assertFalse(info['normalized_view_used'])
        self.assertEqual(info['views'], ['BM25/original'])

    def test_empty_query(self):
        self.assertEqual(self.engine.search('   ', MV_BM25_METHOD, 10), [])

    def test_app_multiview_and_comparison(self):
        import app
        table, note = app.run_search('Kaha bath ekata kanna monawada?', MV_BM25_METHOD, 5)
        self.assertTrue(table)
        self.assertIn('BM25/normalized', note)
        _, note2 = app.run_search('qwxz plugh', MV_BM25_METHOD, 5)
        self.assertIn('Normalisation not used', note2)
        head, overlap, upd = app.compare_methods('Kaha bath ekata kanna monawada?', 5,
                                                 ['BM25 (lexical)', MV_BM25_METHOD])
        self.assertIn('Normalisation', head)
        self.assertIn('Unique to this method', overlap)
        self.assertTrue(upd['value'])


if __name__ == '__main__':
    unittest.main()
