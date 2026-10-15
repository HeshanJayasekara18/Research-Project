"""Record a reviewer's EXPLICIT blanket confirmation as a separate completed review file.

    python review/record_confirmation.py --reviewer HM --date 2026-10-15 \
        --statement "I have personally reviewed all 64 YouTube videos ... relevant to their assigned research domains"

Reads the blank review/video_review_sheet_v2.csv (unchanged) and writes
review/video_review_record_<reviewer>_<date>.csv with decision=include for every video in its
existing domain, plus review_basis/review_statement columns that say this came from the
reviewer's explicit confirmation (not the earlier automated verified_by values).
Also writes a provenance JSON with the source-sheet hash. Never touches data/videos.csv.
Refuses to overwrite an existing record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_rules import SHEET_COLUMNS  # noqa: E402

SOURCE = HERE / 'video_review_sheet_v2.csv'


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--reviewer', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--statement', required=True, help='The reviewer\'s own confirmation, quoted verbatim')
    a = p.parse_args(argv)
    out = HERE / f'video_review_record_{a.reviewer}_{a.date}.csv'
    if out.exists():
        print(f'Refusing to overwrite {out.name}')
        return 1
    with SOURCE.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    basis = f'Explicit blanket confirmation by reviewer {a.reviewer} on {a.date}: all videos included in their assigned domain'
    for r in rows:
        r.update({'manual_review_decision': 'include', 'corrected_domain': '', 'reviewed_by': a.reviewer,
                  'reviewed_at': a.date,
                  # Notes are required for HIGH/MEDIUM rows; they record the basis, not a per-video judgement.
                  'reviewer_notes': 'Included per reviewer\'s explicit confirmation (blanket decision, no per-video note given)',
                  'review_basis': 'human_blanket_confirmation', 'review_statement': a.statement})
    fields = list(SHEET_COLUMNS) + ['review_basis', 'review_statement']
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    prov = {'record_file': out.name, 'source_sheet': SOURCE.name,
            'source_sheet_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'reviewer': a.reviewer, 'review_date': a.date, 'decision': 'include (all videos, existing domains)',
            'basis': basis, 'statement': a.statement,
            'note': 'Distinct from data/videos.csv verified_by/verified_at, which were set automatically by a script.',
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    out.with_suffix('.provenance.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out.name} ({len(rows)} videos) and {out.with_suffix(".provenance.json").name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
