import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import (DENSE_METHODS, MULTIVIEW_METHODS, MV_BM25_METHOD, NORM_BM25_METHOD,
                       SEMANTIC_METHOD, RetrievalEngine)

DEMO_VIDEO = '-3a4abtQH3E'
DEMO_QUERY = 'Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?'
LEXICAL = ['BM25 (lexical)', 'Character TF-IDF', NORM_BM25_METHOD, MV_BM25_METHOD]


class TestVideoScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RetrievalEngine()
        cls.semantic_ready, _ = cls.engine.semantic_status()
        cls.methods = LEXICAL + ([m for m in [SEMANTIC_METHOD, *MULTIVIEW_METHODS] if m not in LEXICAL]
                                 if cls.semantic_ready else [])

    def assert_in_video(self, rows, video_id):
        self.assertTrue(rows)
        bad = [d['document_id'] for d, _ in rows if d['video_id'] != video_id]
        self.assertEqual(bad, [], f'comments from another video returned for {video_id}')

    def test_every_method_respects_scope_on_demo(self):
        for m in self.methods:
            with self.subTest(method=m):
                self.assert_in_video(self.engine.search(DEMO_QUERY, m, 20, DEMO_VIDEO), DEMO_VIDEO)

    def test_every_video_lexical_methods(self):
        for vid in self.engine.video_ids():
            for m in ('BM25 (lexical)', MV_BM25_METHOD):
                rows = self.engine.search('video eka hari lassanai', m, 50, vid)
                self.assertTrue(all(d['video_id'] == vid for d, _ in rows), (vid, m))

    def test_every_video_dense(self):
        if not self.semantic_ready:
            self.skipTest('LaBSE index unavailable')
        for vid in self.engine.video_ids():
            rows = self.engine.search('video eka hari lassanai', SEMANTIC_METHOD, 10, vid)
            self.assert_in_video(rows, vid)

    def test_strong_match_in_other_video_is_not_leaked(self):
        q = 'Slt nam jarawak. patta rental ekak gewanna ona heh'  # exact comment from video QvkLXe99zEk
        glob = self.engine.search(q, 'BM25 (lexical)', 1)
        self.assertEqual(glob[0][0]['video_id'], 'QvkLXe99zEk')
        for m in self.methods:
            rows = self.engine.search(q, m, 50, DEMO_VIDEO)
            self.assertNotIn('QvkLXe99zEk', {d['video_id'] for d, _ in rows}, m)

    def test_filter_before_ranking_returns_full_depth(self):
        # If filtering happened after a global top-k, a scoped dense search would
        # usually return far fewer than k rows. Pre-filtering always fills k.
        if not self.semantic_ready:
            self.skipTest('LaBSE index unavailable')
        rows = self.engine.search('kema hadana widiha', SEMANTIC_METHOD, 50, DEMO_VIDEO)
        self.assertEqual(len(rows), 50)
        self.assert_in_video(rows, DEMO_VIDEO)

    def test_topk_capped_by_video_size(self):
        n = len(self.engine.video_rows[DEMO_VIDEO])
        if self.semantic_ready:
            self.assertEqual(len(self.engine.search('video', SEMANTIC_METHOD, n + 100, DEMO_VIDEO)), n)
        self.assertLessEqual(len(self.engine.search('video', 'BM25 (lexical)', n + 100, DEMO_VIDEO)), n)

    def test_all_scripts_kept_in_scope(self):
        from evaluate_crossscript import script_of
        scripts = {script_of(self.engine.texts[i]) for i in self.engine.video_rows[DEMO_VIDEO]}
        self.assertEqual(scripts, {'latin', 'sinhala'})

    def test_global_mode_unchanged(self):
        q = 'Bandi full magic mix eka kade ganna thiyenawada?'
        a = self.engine.search(q, 'BM25 (lexical)', 10)
        b = self.engine.search(q, 'BM25 (lexical)', 10, None)
        self.assertEqual([d['document_id'] for d, _ in a], [d['document_id'] for d, _ in b])

    def test_unknown_video_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search('video', 'BM25 (lexical)', 5, 'NOT_A_VIDEO')

    def test_demo_observed_bm25_result(self):
        # Regression for the observed demo: the two wetakolu peel/seed comments rank 1-2.
        ids = [d['document_id'] for d, _ in self.engine.search(DEMO_QUERY, 'BM25 (lexical)', 2, DEMO_VIDEO)]
        self.assertEqual(set(ids), {'DB8B1DE7D2790', 'DD488D4D88EF8'})


class TestVideoScopeUI(unittest.TestCase):
    def test_app_scoped_search_and_messages(self):
        import app
        table, note = app.run_search(DEMO_QUERY, 'BM25 (lexical)', 10, DEMO_VIDEO)
        self.assertTrue(table)
        self.assertTrue(all(row[3] == DEMO_VIDEO for row in table))
        self.assertIn('129** comments searched', note)
        self.assertIn('no creator login', note)
        table, note = app.run_search('qwxzplugh', 'BM25 (lexical)', 10, DEMO_VIDEO)
        self.assertEqual(table, [])
        self.assertIn('No matching comments were retrieved', note)
        table, note = app.run_search('qwxzplugh', MV_BM25_METHOD, 10, DEMO_VIDEO)
        self.assertEqual(table, [])
        self.assertIn('No matching comments were retrieved', note)
        _, note = app.run_search('video', 'BM25 (lexical)', 5, 'NOT_A_VIDEO')
        self.assertIn('ValueError', note)
        self.assertEqual(app.scope_choices()[0][1], '')  # global mode still available

    def test_no_result_wording_and_candidate_labels(self):
        import app
        _, note = app.run_search('qwxzplugh', 'BM25 (lexical)', 10, DEMO_VIDEO)
        self.assertIn('using BM25 (lexical)', note)
        self.assertIn('does **not** mean that no relevant comments exist', note)
        _, note = app.run_search('qwxzplugh', 'BM25 (lexical)', 10)
        self.assertIn('global corpus', note)
        table, note = app.run_search(DEMO_QUERY, 'BM25 (lexical)', 10, DEMO_VIDEO)
        self.assertIn('not human-verified evidence', note)
        self.assertIn('does **not** generate answers', note)
        shared = {row[5]: row[6] for row in table}
        self.assertIn('dara', shared['DB8B1DE7D2790'])           # topical match
        self.assertEqual(shared.get('D98AA69878182'), 'video')  # "Nice video" matched only a generic word


class TestComparisonScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app
        ready, _ = app.engine().semantic_status()
        cls.methods = app.ALL_METHODS if ready else [m for m in app.ALL_METHODS if m not in DENSE_METHODS
                                                     and m not in (MULTIVIEW_METHODS.keys() - {MV_BM25_METHOD})]

    def rows(self, query, video_id, k=20):
        head, overlap, upd = self.app.compare_methods(query, k, self.methods, video_id)
        vid_col = upd['headers'].index('Video ID')
        return head, upd['value'], vid_col

    def test_comparison_never_leaks_other_videos(self):
        for q in (DEMO_QUERY, 'Slt nam jarawak. patta rental ekak gewanna ona heh'):
            head, rows, vid_col = self.rows(q, DEMO_VIDEO)
            self.assertTrue(rows)
            self.assertEqual({r[vid_col] for r in rows}, {DEMO_VIDEO}, q)
            self.assertIn('129** comments searched', head)

    def test_comparison_scope_on_other_videos(self):
        for vid in self.app.engine().video_ids()[:5]:
            _, rows, vid_col = self.rows('video eka hari lassanai', vid, 10)
            self.assertTrue(all(r[vid_col] == vid for r in rows), vid)

    def test_comparison_global_mode_kept(self):
        head, rows, vid_col = self.rows(DEMO_QUERY, '')
        self.assertIn('global research corpus', head)
        self.assertGreater(len({r[vid_col] for r in rows}), 1)


if __name__ == '__main__':
    unittest.main()
