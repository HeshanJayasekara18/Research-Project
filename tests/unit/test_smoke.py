import csv
import json
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from rag_service.retrieval.engine import RESULTS, RetrievalEngine,tokenize

class TestPP1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine=RetrievalEngine()

    def test_corpus_size(self):
        self.assertEqual(len(self.engine.docs), json.loads((RESULTS / "dataset_profile.json").read_text())["clean_corpus_rows"])

    def test_privacy_removed(self):
        self.assertNotIn('author',self.engine.docs[0])
        self.assertNotIn('Author',self.engine.docs[0])
        self.assertTrue(all('comment_text' in x for x in self.engine.docs))

    def test_tokenize_sinhala(self):
        self.assertTrue(tokenize('වීඩියෝ හොඳයි'))

    def test_bm25_returns_real_docs(self):
        rows=self.engine.search('Bandi full magic mix eka kade ganna thiyenawada?','BM25 (lexical)',5)
        self.assertTrue(rows)
        self.assertTrue(all(x['document_id'].startswith('D') for x,_ in rows))
        self.assertTrue(all(score>0 for _,score in rows))

    def test_character_baseline(self):
        rows=self.engine.search('Wetakolu uyaddi poththa ain karanna oneda?','Character TF-IDF',5)
        self.assertTrue(rows)

    def test_empty_query(self):
        self.assertEqual(self.engine.search('  '),[])

    def test_provenance(self):
        profile=json.loads((RESULTS/'dataset_profile.json').read_text())
        self.assertEqual(profile['seed_pairs'],366)
        self.assertEqual(profile['verified_relevance_labels'],0)

if __name__=='__main__':
    unittest.main()
