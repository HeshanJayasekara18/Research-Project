import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from rag_service.retrieval.normalization import LexiconRuleNormalizer, Normalizer, split_units
from rag_service.retrieval.engine import NORM_BM25_METHOD, RetrievalEngine

# Tiny synthetic lexicon: Sinhala-script "comments" (ad is an English loanword inside them).
TOY = ['කහ බත් කන්න', 'කහ බත් හොඳයි', 'මොනවද කියලා', 'Ad කීයක්ද', 'Ad එක දිගයි', 'මස මස', 'මාස මාස']


class TestLexiconRuleNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.norm = LexiconRuleNormalizer(TOY)

    def test_split_units(self):
        self.assertEqual(split_units('bath'), [('C', 'b'), ('V', 'a'), ('C', 'th')])
        self.assertEqual(split_units('kanna'), [('C', 'k'), ('V', 'a'), ('C', 'n'), ('C', 'n'), ('V', 'a')])
        self.assertIsNone(split_units('ab-c'))

    def test_known_words_transliterated(self):
        r = self.norm.normalize('Kaha bath kanna monawada kiyala')
        self.assertEqual(r.normalized, 'කහ බත් කන්න මොනවද කියලා')
        self.assertEqual(r.status, 'normalized')
        self.assertEqual(r.original, 'Kaha bath kanna monawada kiyala')

    def test_unknown_word_kept_and_flagged(self):
        t = self.norm.normalize_word('xyzzy')
        self.assertEqual((t.output, t.status), ('xyzzy', 'unresolved'))

    def test_failed_status_when_nothing_converts(self):
        r = self.norm.normalize('qwerty plugh')
        self.assertEqual(r.status, 'failed')
        self.assertEqual(r.normalized, 'qwerty plugh')

    def test_numbers_and_sinhala_preserved(self):
        r = self.norm.normalize('20k bath කහ')
        self.assertEqual([t.status for t in r.tokens], ['kept_number', 'transliterated', 'kept_sinhala'])
        self.assertEqual(r.normalized, '20k බත් කහ')

    def test_english_loanword_kept(self):
        t = self.norm.normalize_word('Ad')
        self.assertEqual((t.output, t.status), ('Ad', 'kept_english'))

    def test_ambiguity_flagged(self):
        # 'මස' (2) vs 'මාස' (2): equal frequency -> must be flagged, not silently chosen
        t = self.norm.normalize_word('mas')
        self.assertEqual(t.status, 'ambiguous')
        self.assertEqual(set(t.alternatives), {'මස', 'මාස'})

    def test_empty_and_deterministic(self):
        self.assertEqual(self.norm.normalize('  ?').status, 'empty')
        self.assertEqual(self.norm.normalize('kaha bath').normalized, self.norm.normalize('kaha bath').normalized)

    def test_identity_interface(self):
        r = Normalizer().normalize('kaha bath')
        self.assertEqual((r.normalized, r.status), ('kaha bath', 'unchanged'))


class TestNormalizedRetrieval(unittest.TestCase):
    def test_normalized_bm25_reaches_sinhala_script(self):
        engine = RetrievalEngine()
        q = 'Kaha bath ekata kanna galapena curries monawada?'
        self.assertIn('කහ', engine.normalize(q).normalized)
        rows = engine.search(q, NORM_BM25_METHOD, 5)
        self.assertTrue(rows)
        self.assertTrue(any('කහ' in d['comment_text'] for d, _ in rows))

    def test_app_imports_and_renders_normalization_note(self):
        from prototype import gradio_app as app  # builds the Gradio Blocks without launching a server
        table, note = app.run_search('Kaha bath ekata kanna monawada?', NORM_BM25_METHOD, 5)
        self.assertTrue(table)
        self.assertIn('Normalised query', note)
        self.assertIn('කහ', note)


if __name__ == '__main__':
    unittest.main()
