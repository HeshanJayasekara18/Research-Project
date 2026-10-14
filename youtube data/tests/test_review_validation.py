"""Tests for the manual video review tooling. Uses temporary copies only; the decisions here are
synthetic fixtures, not review decisions."""
import csv
import hashlib
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'review'))
import make_review_sheet  # noqa: E402
from review_rules import SHEET_COLUMNS  # noqa: E402
from validate_review import main as validate_main, validate  # noqa: E402

TODAY = date(2026, 10, 15)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class ReviewTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / 'videos.csv'
        with self.manifest.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video_id', 'domain'])
            w.writerows([['vidFOOD0001', 'food'], ['vidEDU00001', 'education'], ['vidEDU00002', 'education']])

    def sheet(self, rows):
        path = self.tmp / 'sheet.csv'
        base = {c: '' for c in SHEET_COLUMNS}
        full = []
        for vid, dom, level, decision, corrected, notes, by, at in rows:
            r = dict(base, video_id=vid, proposed_domain=dom, assistant_flag_level=level, manual_review_decision=decision,
                     corrected_domain=corrected, reviewer_notes=notes, reviewed_by=by, reviewed_at=at,
                     collected_comment_records='10')
            full.append(r)
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(SHEET_COLUMNS))
            w.writeheader()
            w.writerows(full)
        return path

    def check(self, rows):
        return validate(self.sheet(rows), self.manifest, TODAY)


class TestValidation(ReviewTestBase):
    GOOD = [('vidFOOD0001', 'food', '', 'include', '', '', 'HM', '2026-10-15'),
            ('vidEDU00001', 'education', 'HIGH', 'exclude', '', 'celebrity news', 'HM', '2026-10-15'),
            ('vidEDU00002', 'education', 'MEDIUM', 'reclassify', 'gaming', 'desktop setup for gaming', 'HM', '2026-10-14')]

    def test_complete_valid_sheet(self):
        res = self.check(self.GOOD)
        self.assertTrue(res['ok'], res['errors'])
        self.assertEqual(res['summary']['decisions'], {'include': 1, 'exclude': 1, 'reclassify': 1})
        self.assertEqual(res['summary']['videos_per_domain_after_review'], {'food': 1, 'gaming': 1})

    def test_blank_sheet_is_not_valid(self):
        res = self.check([(v, d, l, '', '', '', '', '') for v, d, l, *_ in self.GOOD])
        self.assertFalse(res['ok'])
        self.assertEqual(res['summary']['undecided'], 3)

    def test_rules(self):
        cases = {
            'bad value': [('vidFOOD0001', 'food', '', 'approved', '', '', 'HM', '2026-10-15')],
            'reclassify without domain': [('vidFOOD0001', 'food', '', 'reclassify', '', 'x', 'HM', '2026-10-15')],
            'reclassify same domain': [('vidFOOD0001', 'food', '', 'reclassify', 'food', 'x', 'HM', '2026-10-15')],
            'corrected domain on include': [('vidFOOD0001', 'food', '', 'include', 'gaming', '', 'HM', '2026-10-15')],
            'exclude without notes': [('vidFOOD0001', 'food', '', 'exclude', '', '', 'HM', '2026-10-15')],
            'HIGH without notes': [('vidEDU00001', 'education', 'HIGH', 'include', '', '', 'HM', '2026-10-15')],
            'no reviewer': [('vidFOOD0001', 'food', '', 'include', '', '', '', '2026-10-15')],
            'bad date': [('vidFOOD0001', 'food', '', 'include', '', '', 'HM', '15/10/2026')],
            'future date': [('vidFOOD0001', 'food', '', 'include', '', '', 'HM', '2026-12-01')],
            'notes without decision': [('vidFOOD0001', 'food', '', '', '', 'thinking', '', '')],
            'proposed domain edited': [('vidFOOD0001', 'gaming', '', 'include', '', '', 'HM', '2026-10-15')],
        }
        rest = self.GOOD[1:]
        for name, first in cases.items():
            with self.subTest(name):
                self.assertFalse(self.check(first + rest)['ok'], name)

    def test_missing_and_extra_videos(self):
        res = self.check(self.GOOD[:2] + [('vidOTHER0001', 'food', '', 'include', '', '', 'HM', '2026-10-15')])
        msgs = ' '.join(res['errors'])
        self.assertIn('vidEDU00002: in manifest but missing', msgs)
        self.assertIn('vidOTHER0001: in the sheet but not in the manifest', msgs)

    def test_validator_is_read_only(self):
        sheet = self.sheet(self.GOOD)
        before = {p: sha(p) for p in (sheet, self.manifest)}
        listing = sorted(p.name for p in self.tmp.iterdir())
        validate(sheet, self.manifest, TODAY)
        self.assertEqual({p: sha(p) for p in before}, before)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), listing)

    def test_real_manifest_untouched_by_cli(self):
        real = ROOT / 'data' / 'videos.csv'
        if not real.exists():
            self.skipTest('no manifest')
        before = sha(real)
        validate_main([str(self.sheet(self.GOOD))])
        self.assertEqual(sha(real), before)


class TestSheetGenerator(unittest.TestCase):
    def test_refuses_to_overwrite_reviewer_input(self):
        tmp = Path(tempfile.mkdtemp()) / 'sheet.csv'
        with tmp.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(SHEET_COLUMNS))
            w.writeheader()
            w.writerow({c: '' for c in SHEET_COLUMNS} | {'video_id': 'x', 'manual_review_decision': 'include'})
        original = make_review_sheet.OUT
        try:
            make_review_sheet.OUT = tmp
            before = sha(tmp)
            self.assertEqual(make_review_sheet.main(), 1)
            self.assertEqual(sha(tmp), before)
        finally:
            make_review_sheet.OUT = original

    def test_generated_sheet_blank_with_links(self):
        if not (ROOT / 'data' / 'videos.csv').exists():
            self.skipTest('no manifest')
        rows = make_review_sheet.build_rows()
        self.assertTrue(all(all(r[c] == '' for c in ('manual_review_decision', 'corrected_domain', 'reviewer_notes',
                                                    'reviewed_by', 'reviewed_at')) for r in rows))
        self.assertTrue(all(r['open_video'].startswith('=HYPERLINK("https://www.youtube.com/watch?v=') for r in rows))


if __name__ == '__main__':
    unittest.main()
