"""Annotation V2 pool builder on the SYNTHETIC V2 fixture (no real comments)."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from research.annotation_v2.build_pool_v2 import SHEET_FIELDS, build
from tests.api.test_corpus_v2 import SYNTHETIC, write_corpus

QUERIES = [
    {'query_id': 'F1', 'domain': 'food', 'video_id': 'vidFOOD0001', 'query_text': 'kiri bath recipe eka'},
    {'query_id': 'G1', 'domain': 'gaming', 'video_id': 'vidGAME0001', 'query_text': 'gameplay headshot'},
    {'query_id': 'E1', 'domain': 'education', 'video_id': 'vidEDUC0001', 'query_text': 'physics notes lesson'},
]


def read(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class TestPoolV2(unittest.TestCase):
    def setUp(self):
        self.corpus = write_corpus(SYNTHETIC)
        self.out = Path(tempfile.mkdtemp())
        self.queries = self.out / 'queries.csv'
        with self.queries.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['query_id', 'domain', 'video_id', 'query_text'])
            w.writeheader()
            w.writerows(QUERIES)

    def manifest(self, ref):
        (self.corpus / 'export_manifest.json').write_text(json.dumps({'ethics_reference': ref}), encoding='utf-8')

    def test_refuses_without_export_manifest_or_ethics_reference(self):
        with self.assertRaises(SystemExit):
            build(self.corpus, self.queries, self.out)
        self.manifest('')
        with self.assertRaises(SystemExit):
            build(self.corpus, self.queries, self.out)
        self.assertFalse((self.out / 'annotator_A.csv').exists())

    def test_blank_identical_scoped_sheets_with_hidden_provenance(self):
        self.manifest('SYNTHETIC-TEST-REF')
        m = build(self.corpus, self.queries, self.out)
        a, b = read(self.out / 'annotator_A.csv'), read(self.out / 'annotator_B.csv')
        self.assertEqual(a, b)
        self.assertEqual(list(a[0].keys()), SHEET_FIELDS)
        self.assertTrue(all(r['relevance_grade'] == '' and r['annotation_note'] == '' for r in a))
        by_id = {r['comment_id']: r for r in SYNTHETIC}
        for r in a:
            src = by_id[r['comment_id']]
            self.assertEqual((r['video_id'], r['domain'], r['original_comment_text']),
                             (src['video_id'], src['domain'], src['comment_text']))
            self.assertEqual(src['indexable'], 'true')                       # withheld/emoji rows never pooled
        prov = read(self.out / 'provenance' / 'pool_provenance.csv')
        pooled = {(r['query_id'], r['comment_id']) for r in prov}
        self.assertEqual({(r['query_id'], r['comment_id']) for r in a}, pooled)  # full union, not truncated
        self.assertEqual(m['methods_unavailable'], {'labse': 'no V2 LaBSE index (not built in this milestone)'})
        self.assertNotIn('rank', a[0])
        with self.assertRaises(SystemExit):                                  # never overwrites sheets
            build(self.corpus, self.queries, self.out)

    def test_m7_tools_validate_v2_sheets(self):
        from research.annotation import annotation_tools as at
        self.manifest('SYNTHETIC-TEST-REF')
        build(self.corpus, self.queries, self.out)
        res = at.validate(self.out / 'annotator_A.csv', self.out / 'annotator_B.csv')
        self.assertFalse(res['ok'])                                   # blank: every row is unlabelled
        self.assertEqual(len(res['missing']), len(read(self.out / 'annotator_A.csv')))
        self.assertTrue(all(k[1] in {r['comment_id'] for r in SYNTHETIC} for k in res['missing']))
        runs = at.load_runs(self.out / 'provenance' / 'pool_provenance.csv')
        self.assertIn(('F1', 'bm25'), runs)

    def test_requires_all_three_domains(self):
        self.manifest('SYNTHETIC-TEST-REF')
        with self.queries.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['query_id', 'domain', 'video_id', 'query_text'])
            w.writeheader()
            w.writerow(QUERIES[0])
        with self.assertRaises(SystemExit):
            build(self.corpus, self.queries, self.out)


if __name__ == '__main__':
    unittest.main()
