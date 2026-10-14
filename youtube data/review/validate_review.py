"""READ-ONLY check of a completed manual video review sheet.

    cd "youtube data"; python review/validate_review.py                       # default: review/video_review_sheet_v2.csv
    python review/validate_review.py review/video_review_sheet_v2.csv

Never writes any file. It does not modify data/videos.csv or the collected comments and does
not mark anything as verified. It reports problems and, if the sheet is complete, a summary of
what the decisions would mean. Applying the decisions to the manifest is a separate, later step.

Exit code 0 = complete and valid, 1 = problems found.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_rules import DECISIONS, DOMAINS, PRIORITIES_NEEDING_NOTES, REVIEWER_COLUMNS  # noqa: E402

ROOT = HERE.parent
MANIFEST = ROOT / 'data' / 'videos.csv'
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
LOCKED = ('video_id', 'proposed_domain')  # must still match the manifest


def read(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate(sheet_path: Path, manifest_path: Path = MANIFEST, today: date | None = None) -> dict:
    today = today or date.today()
    rows = read(sheet_path)
    manifest = {r['video_id']: r for r in read(manifest_path)}
    errors, warnings = [], []

    cols = set(rows[0].keys()) if rows else set()
    missing_cols = [c for c in (*LOCKED, *REVIEWER_COLUMNS, 'assistant_flag_level') if c not in cols]
    if missing_cols:
        return {'ok': False, 'errors': [f'missing columns: {missing_cols}'], 'warnings': [], 'summary': {}}

    seen = Counter(r['video_id'] for r in rows)
    for vid, n in seen.items():
        if n > 1:
            errors.append(f'{vid}: appears {n} times')
    for vid in sorted(set(manifest) - set(seen)):
        errors.append(f'{vid}: in manifest but missing from the sheet')
    for vid in sorted(set(seen) - set(manifest)):
        errors.append(f'{vid}: in the sheet but not in the manifest')

    decided = Counter()
    final_domain = Counter()
    undecided = []
    for i, r in enumerate(rows, start=2):  # line 1 = header
        vid = r['video_id']
        where = f'line {i} ({vid})'
        decision = (r['manual_review_decision'] or '').strip().lower()
        corrected = (r['corrected_domain'] or '').strip().lower()
        notes = (r['reviewer_notes'] or '').strip()
        by = (r['reviewed_by'] or '').strip()
        at = (r['reviewed_at'] or '').strip()
        level = (r['assistant_flag_level'] or '').strip().upper()

        if vid in manifest and r['proposed_domain'] != manifest[vid]['domain']:
            errors.append(f'{where}: proposed_domain was edited ({r["proposed_domain"]!r} != manifest '
                          f'{manifest[vid]["domain"]!r}); use corrected_domain + reclassify instead')
        if not decision:
            undecided.append(vid)
            if corrected or notes or by or at:
                errors.append(f'{where}: reviewer fields filled but manual_review_decision is empty')
            continue
        if decision not in DECISIONS:
            errors.append(f'{where}: manual_review_decision {decision!r} not one of {DECISIONS}')
            continue
        decided[decision] += 1
        if decision == 'reclassify':
            if corrected not in DOMAINS:
                errors.append(f'{where}: reclassify needs corrected_domain in {DOMAINS}')
            elif corrected == r['proposed_domain']:
                errors.append(f'{where}: corrected_domain equals proposed_domain; use include instead')
            else:
                final_domain[corrected] += 1
        else:
            if corrected:
                errors.append(f'{where}: corrected_domain must be blank unless the decision is reclassify')
            if decision == 'include':
                final_domain[r['proposed_domain']] += 1
        if decision in ('exclude', 'reclassify') and not notes:
            errors.append(f'{where}: reviewer_notes required for {decision}')
        if level in PRIORITIES_NEEDING_NOTES and not notes:
            errors.append(f'{where}: reviewer_notes required for {level} priority videos')
        if not by:
            errors.append(f'{where}: reviewed_by is empty (your own initials, typed by you)')
        elif by.upper() in ('', 'AUTO', 'SCRIPT'):
            errors.append(f'{where}: reviewed_by must be a person')
        if not DATE_RE.match(at):
            errors.append(f'{where}: reviewed_at must be YYYY-MM-DD')
        else:
            try:
                d = date.fromisoformat(at)
                if d > today:
                    errors.append(f'{where}: reviewed_at {at} is in the future')
            except ValueError:
                errors.append(f'{where}: reviewed_at {at} is not a real date')

    if undecided:
        errors.append(f'{len(undecided)} videos have no decision yet')
    reviewers = Counter((r['reviewed_by'] or '').strip() for r in rows if (r['manual_review_decision'] or '').strip())
    if len(reviewers) > 1:
        warnings.append(f'several reviewer values used: {dict(reviewers)}')

    comments_by_domain_after = Counter()
    for r in rows:
        decision = (r['manual_review_decision'] or '').strip().lower()
        if decision == 'include':
            comments_by_domain_after[r['proposed_domain']] += int(r.get('collected_comment_records') or 0)
        elif decision == 'reclassify' and (r['corrected_domain'] or '').strip().lower() in DOMAINS:
            comments_by_domain_after[r['corrected_domain'].strip().lower()] += int(r.get('collected_comment_records') or 0)

    return {'ok': not errors, 'errors': errors, 'warnings': warnings,
            'summary': {'videos': len(rows), 'decided': sum(decided.values()), 'undecided': len(undecided),
                        'decisions': dict(decided), 'videos_per_domain_after_review': dict(final_domain),
                        'comment_records_per_domain_after_review': dict(comments_by_domain_after)}}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sheet = Path(argv[0]) if argv else HERE / 'video_review_sheet_v2.csv'
    res = validate(sheet)
    print(f'Checked {sheet} (read-only; nothing was written).')
    for e in res['errors'][:60]:
        print('ERROR  ', e)
    if len(res['errors']) > 60:
        print(f'... and {len(res["errors"]) - 60} more errors')
    for w in res['warnings']:
        print('WARNING', w)
    print('SUMMARY', res['summary'])
    print('RESULT ', 'VALID: complete and consistent' if res['ok'] else 'NOT VALID YET')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
