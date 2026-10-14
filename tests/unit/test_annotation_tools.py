"""Tests for the M7 annotation tools. All labels here are synthetic test fixtures in temp
files; no human annotation is generated or modified."""
import csv
import tempfile
import unittest
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from research.annotation import annotation_tools as at

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / 'research' / 'annotation'


def write(path, rows, fields=at.REQUIRED):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def sheet(grades, qid='Q1'):
    return [{'query_id': qid, 'query_text': 'q', 'video_id': 'v', 'document_id': f'D{i}',
             'original_comment_text': f'c{i}', 'relevance_grade': g, 'annotation_note': ''} for i, g in enumerate(grades)]


class TestBlankMaterials(unittest.TestCase):
    """The committed-to-disk annotation sheets must be blank, identical and free of method/rank info."""

    def test_sheets_blank_identical_and_minimal(self):
        a, b = at.read_rows(ANN / 'annotator_A.csv'), at.read_rows(ANN / 'annotator_B.csv')
        self.assertEqual(a, b)
        self.assertEqual(list(a[0].keys()), at.REQUIRED)
        self.assertTrue(all(r['relevance_grade'] == '' and r['annotation_note'] == '' for r in a))
        self.assertGreater(len(a), 0)

    def test_sheet_matches_pool_and_corpus_text(self):
        from rag_service.retrieval.engine import load_corpus
        corpus = {d['document_id']: d for d in load_corpus()}
        prov = at.read_rows(ANN / 'provenance' / 'pool_provenance.csv')
        pooled = {(r['query_id'], r['document_id']) for r in prov}
        rows = at.read_rows(ANN / 'annotator_A.csv')
        self.assertEqual({(r['query_id'], r['document_id']) for r in rows}, pooled)  # full union, not truncated
        for r in rows:
            self.assertEqual(r['original_comment_text'], corpus[r['document_id']]['comment_text'])
            self.assertEqual(r['video_id'], corpus[r['document_id']]['video_id'])

    def test_no_author_data_in_exports(self):
        for name in ('annotator_A.csv', 'annotator_B.csv', 'queries.csv', 'provenance/pool_provenance.csv'):
            header = at.read_rows(ANN / name)[0].keys()
            self.assertFalse({'Author', 'author', 'author_name', 'channel_id'} & set(header), name)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_missing_invalid_and_duplicates(self):
        rows = sheet(['3', '', '7', 'x', '0'])
        rows.append(dict(rows[0]))
        r = at.validate(write(self.tmp / 'a.csv', rows))
        self.assertFalse(r['ok'])
        self.assertEqual(len(r['missing']), 1)
        self.assertEqual({v for _, v, _ in r['invalid']}, {'7', 'x'})
        self.assertEqual(r['duplicates'], [('Q1', 'D0')])

    def test_template_pair_mismatch(self):
        tpl = write(self.tmp / 't.csv', sheet(['', '', '']))
        r = at.validate(write(self.tmp / 'a.csv', sheet(['1', '2'])), tpl)
        self.assertTrue(any('pairs differ' in e for e in r['errors']))

    def test_complete_file_ok(self):
        self.assertTrue(at.validate(write(self.tmp / 'a.csv', sheet(['0', '1', '2', '3'])))['ok'])


class TestAgreement(unittest.TestCase):
    def test_kappa_matches_sklearn(self):
        a = [0, 1, 2, 3, 3, 2, 1, 0, 2, 3, 1, 1]
        b = [0, 2, 2, 3, 2, 2, 0, 0, 3, 3, 1, 2]
        for w in ('quadratic', 'linear'):
            self.assertAlmostEqual(at.weighted_kappa(a, b, w), cohen_kappa_score(a, b, weights=w, labels=[0, 1, 2, 3]), 10)

    def test_kappa_undefined_when_no_variation(self):
        self.assertIsNone(at.weighted_kappa([2, 2, 2], [2, 2, 2]))

    def test_agreement_and_disagreement_export(self):
        tmp = Path(tempfile.mkdtemp())
        a = write(tmp / 'a.csv', sheet(['0', '1', '2', '3', '']))
        b = write(tmp / 'b.csv', sheet(['0', '2', '2', '1', '3']))
        rep = at.agreement(a, b)
        self.assertEqual(rep['pairs_labelled_by_both'], 4)
        self.assertEqual(rep['overall']['exact_agreement'], 0.5)
        n = at.disagreements(a, b, tmp / 'dis.csv')
        self.assertEqual(n, 2)
        rows = at.read_rows(tmp / 'dis.csv')
        self.assertTrue(all(r['adjudicated_grade'] == '' for r in rows))


class TestFinalize(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.a = write(self.tmp / 'a.csv', sheet(['0', '1', '3']))
        self.b = write(self.tmp / 'b.csv', sheet(['0', '2', '3']))

    def test_refuses_without_approval(self):
        with self.assertRaises(SystemExit):
            at.finalize(self.a, self.b, None, None, self.tmp / 'q.csv')

    def test_refuses_unadjudicated_disagreement(self):
        with self.assertRaises(SystemExit):
            at.finalize(self.a, self.b, None, 'Reviewer', self.tmp / 'q.csv')

    def test_refuses_incomplete_annotations(self):
        incomplete = write(self.tmp / 'i.csv', sheet(['0', '', '3']))
        with self.assertRaises(SystemExit):
            at.finalize(incomplete, self.b, None, 'Reviewer', self.tmp / 'q.csv')

    def test_agreed_plus_adjudicated(self):
        at.disagreements(self.a, self.b, self.tmp / 'd.csv')
        rows = at.read_rows(self.tmp / 'd.csv')
        rows[0]['adjudicated_grade'] = '2'
        write(self.tmp / 'd.csv', rows, list(rows[0].keys()))
        n = at.finalize(self.a, self.b, self.tmp / 'd.csv', 'Reviewer', self.tmp / 'q.csv')
        self.assertEqual(n, 3)
        q = at.load_qrels(self.tmp / 'q.csv')
        self.assertEqual(q[('Q1', 'D1')], 2)


class TestMetrics(unittest.TestCase):
    qrels = {('Q1', 'A'): 3, ('Q1', 'B'): 0, ('Q1', 'C'): 2, ('Q1', 'D'): 1}

    def test_ndcg_and_mrr(self):
        self.assertAlmostEqual(at.ndcg_at_k(['A', 'C', 'D', 'B'], 'Q1', self.qrels), 1.0)
        self.assertLess(at.ndcg_at_k(['B', 'D', 'C', 'A'], 'Q1', self.qrels), 1.0)
        self.assertEqual(at.mrr_at_k(['B', 'D', 'C'], 'Q1', self.qrels), 1 / 3)  # first grade >= 2 at rank 3

    def test_unjudged_not_treated_as_irrelevant(self):
        res = at.evaluate(self.qrels, {('Q1', 'm'): ['A', 'X', 'C']}, min_coverage=1.0)
        entry = res['per_run']['Q1/m']
        self.assertAlmostEqual(entry['coverage_at_k'], 2 / 3, 4)
        self.assertEqual(entry['ndcg_at_k'], 'not computed')
        self.assertEqual(res['per_method_mean'], {})

    def test_full_coverage_computes(self):
        res = at.evaluate(self.qrels, {('Q1', 'm'): ['A', 'C', 'D', 'B']}, min_coverage=1.0)
        self.assertEqual(res['per_run']['Q1/m']['ndcg_at_k'], 1.0)
        # A shorter run that misses a judged relevant doc scores below the ideal.
        short = at.evaluate(self.qrels, {('Q1', 'm'): ['A', 'C']}, min_coverage=1.0)
        self.assertLess(short['per_run']['Q1/m']['ndcg_at_k'], 1.0)
        self.assertNotIn('recall', str(res).lower().replace('no recall', ''))

    def test_no_relevant_doc_is_undefined_not_zero(self):
        res = at.evaluate({('Q5', 'A'): 0}, {('Q5', 'm'): ['A']})
        self.assertIn('undefined', res['per_run']['Q5/m']['ndcg_at_k'])


if __name__ == '__main__':
    unittest.main()
