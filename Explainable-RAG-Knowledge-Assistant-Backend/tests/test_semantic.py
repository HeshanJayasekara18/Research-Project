import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import retrieval
from retrieval import LABSE_INDEX, SEMANTIC_METHOD, RetrievalEngine
from evaluate_crossscript import rank_metrics, script_of, spelling_variants


class TestSemanticFailureModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RetrievalEngine()

    def test_missing_index_reports_not_crashes(self):
        engine = RetrievalEngine.__new__(RetrievalEngine)
        engine.ids, engine._semantic = ['D1'], None
        with mock.patch.object(retrieval, 'LABSE_INDEX', ROOT / 'data' / 'does_not_exist.npz'):
            ready, msg = engine.semantic_status()
        self.assertFalse(ready)
        self.assertIn('build_labse_index.py', msg)

    def test_mismatched_ids_rejected(self):
        tmp = ROOT / 'tests' / '_tmp_index.npz'
        np.savez(tmp, embeddings=np.ones((1, 4), np.float16), document_ids=np.array(['DX']))
        try:
            engine = RetrievalEngine.__new__(RetrievalEngine)
            engine.ids, engine._semantic = ['D1'], None
            with mock.patch.object(retrieval, 'LABSE_INDEX', tmp):
                ready, msg = engine.semantic_status()
            self.assertFalse(ready)
            self.assertIn('differ', msg)
        finally:
            tmp.unlink()


@unittest.skipUnless(LABSE_INDEX.exists(), 'LaBSE index not built')
class TestSemanticRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RetrievalEngine()
        ready, msg = cls.engine.semantic_status()
        if not ready:
            raise unittest.SkipTest(msg)

    def test_index_aligned_and_normalised(self):
        self.assertEqual(len(self.engine._semantic), len(self.engine.ids))
        norms = np.linalg.norm(self.engine._semantic, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    def test_semantic_returns_ranked_cosine_scores(self):
        rows = self.engine.search('Wetakolu uyaddi dara ain karanna oneda?', SEMANTIC_METHOD, 10)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(-1.0 <= s <= 1.0001 for _, s in rows))  # cosine range
        self.assertEqual([s for _, s in rows], sorted([s for _, s in rows], reverse=True))


class TestCrossScriptHelpers(unittest.TestCase):
    def test_script_detection(self):
        self.assertEqual(script_of('වීඩියෝ හොඳයි'), 'sinhala')
        self.assertEqual(script_of('video eka hodai'), 'latin')

    def test_variants_deterministic_and_different(self):
        v = spelling_variants('Wetakolu thiyenawada?')
        self.assertEqual(v, spelling_variants('Wetakolu thiyenawada?'))
        self.assertIn('vetakolu tiyenavada?', v)
        self.assertNotIn('wetakolu thiyenawada?', v)

    def test_rank_metrics(self):
        self.assertEqual(rank_metrics(['a', 'b'], {'b'}), {'hit1': 0, 'hit10': 1, 'rr10': 0.5})
        self.assertEqual(rank_metrics(['a'], {'z'})['hit10'], 0)


if __name__ == '__main__':
    unittest.main()
