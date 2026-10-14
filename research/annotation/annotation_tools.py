"""M7 annotation tools: validate labels, agreement, disagreements, final qrels, coverage, metrics.

    python -m research.annotation.annotation_tools validate  research/annotation/annotator_A.csv
    python -m research.annotation.annotation_tools agreement research/annotation/annotator_A.csv research/annotation/annotator_B.csv
    python -m research.annotation.annotation_tools disagreements A.csv B.csv --out research/annotation/work/disagreements.csv
    python -m research.annotation.annotation_tools finalize A.csv B.csv --adjudication research/annotation/work/disagreements.csv --approved-by "<name>"
    python -m research.annotation.annotation_tools coverage  --qrels research/annotation/work/qrels.csv
    python -m research.annotation.annotation_tools evaluate  --qrels research/annotation/work/qrels.csv [--min-coverage 1.0]

Principles:
- Labels are never generated here. Unlabelled pairs are reported as missing, never filled.
- Unjudged documents are UNKNOWN, not irrelevant: metrics are computed for a ranking only when
  its top-10 judgment coverage reaches --min-coverage; otherwise the result is "not computed".
- No recall metric is reported: the pooled qrels are incomplete for each video.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROVENANCE_CSV = HERE / 'provenance' / 'pool_provenance.csv'
GRADES = (0, 1, 2, 3)
RELEVANT_MIN_GRADE = 2  # for MRR: "relevant" = grade 2 or 3
REQUIRED = ['query_id', 'query_text', 'video_id', 'document_id', 'original_comment_text',
            'relevance_grade', 'annotation_note']

Key = tuple[str, str]  # (query_id, document_id)


# ---------------------------------------------------------------- loading / validation
def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate(path: Path, template: Path | None = None) -> dict:
    """Return {'ok', 'errors', 'missing', 'invalid', 'duplicates', 'labels'}; never modifies the file."""
    rows = read_rows(path)
    errors, missing, invalid, labels = [], [], [], {}
    cols = list(rows[0].keys()) if rows else []
    absent = [c for c in REQUIRED if c not in cols]
    if absent:
        errors.append(f'missing columns: {absent}')
        return {'ok': False, 'errors': errors, 'missing': [], 'invalid': [], 'duplicates': [], 'labels': {}}
    counts = Counter((r['query_id'], r['document_id']) for r in rows)
    duplicates = sorted(k for k, n in counts.items() if n > 1)
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        key = (r['query_id'], r['document_id'])
        raw = (r['relevance_grade'] or '').strip()
        if raw == '':
            missing.append(key)
        elif raw not in {str(g) for g in GRADES}:
            invalid.append((key, raw, i))
        else:
            labels[key] = int(raw)
    if template is not None:
        expected = {(r['query_id'], r['document_id']) for r in read_rows(template)}
        got = set(counts)
        if got != expected:
            errors.append(f'pairs differ from template: {len(expected - got)} missing pairs, {len(got - expected)} unexpected pairs')
    if duplicates:
        errors.append(f'{len(duplicates)} duplicated query-document pairs')
    if invalid:
        errors.append(f'{len(invalid)} invalid grades (allowed: 0,1,2,3)')
    if missing:
        errors.append(f'{len(missing)} unlabelled rows')
    return {'ok': not errors, 'errors': errors, 'missing': missing, 'invalid': invalid,
            'duplicates': duplicates, 'labels': labels}


# ---------------------------------------------------------------- agreement
def weighted_kappa(a: list[int], b: list[int], weights: str = 'quadratic', categories=GRADES) -> float | None:
    """Cohen's kappa with linear or quadratic disagreement weights (ordinal grades).

    Returns None when undefined (e.g. both annotators used a single identical category)."""
    if len(a) != len(b) or not a:
        raise ValueError('need two equal-length, non-empty label lists')
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        d = abs(i - j) / (k - 1)
        return d * d if weights == 'quadratic' else d

    observed = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return None
    return 1 - observed / expected


def agreement(path_a: Path, path_b: Path) -> dict:
    va, vb = validate(path_a), validate(path_b)
    shared = sorted(set(va['labels']) & set(vb['labels']))
    report = {'pairs_labelled_by_both': len(shared), 'a_issues': va['errors'], 'b_issues': vb['errors']}
    if not shared:
        report['note'] = 'no pairs labelled by both annotators yet'
        return report
    a = [va['labels'][k] for k in shared]
    b = [vb['labels'][k] for k in shared]

    def summarise(xs, ys):
        return {'n': len(xs), 'exact_agreement': round(sum(x == y for x, y in zip(xs, ys)) / len(xs), 4),
                'kappa_quadratic': _r(weighted_kappa(xs, ys, 'quadratic')),
                'kappa_linear': _r(weighted_kappa(xs, ys, 'linear')),
                'binary_relevant_agreement': round(sum((x >= RELEVANT_MIN_GRADE) == (y >= RELEVANT_MIN_GRADE)
                                                       for x, y in zip(xs, ys)) / len(xs), 4)}

    report['overall'] = summarise(a, b)
    report['confusion_A_rows_B_cols'] = [[sum(1 for x, y in zip(a, b) if x == i and y == j) for j in GRADES] for i in GRADES]
    by_q = defaultdict(lambda: ([], []))
    for k, x, y in zip(shared, a, b):
        by_q[k[0]][0].append(x)
        by_q[k[0]][1].append(y)
    report['per_query'] = {q: summarise(xs, ys) for q, (xs, ys) in sorted(by_q.items())}
    report['interpretation'] = ('Quadratic-weighted kappa suits ordinal 0-3 grades. Per-query kappa on small n is '
                                'unstable; report it only as a pilot diagnostic.')
    return report


def _r(x):
    return None if x is None else round(x, 4)


def disagreements(path_a: Path, path_b: Path, out: Path) -> int:
    """Write pairs where both annotators labelled but disagree, for human adjudication."""
    ra = {(r['query_id'], r['document_id']): r for r in read_rows(path_a)}
    la, lb = validate(path_a)['labels'], validate(path_b)['labels']
    rows = []
    for key in sorted(set(la) & set(lb)):
        if la[key] != lb[key]:
            r = ra[key]
            rows.append({'query_id': key[0], 'query_text': r['query_text'], 'video_id': r['video_id'],
                         'document_id': key[1], 'original_comment_text': r['original_comment_text'],
                         'grade_A': la[key], 'grade_B': lb[key], 'difference': abs(la[key] - lb[key]),
                         'adjudicated_grade': '', 'adjudication_note': ''})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['query_id', 'query_text', 'video_id', 'document_id', 'original_comment_text',
                                          'grade_A', 'grade_B', 'difference', 'adjudicated_grade', 'adjudication_note'])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def finalize(path_a: Path, path_b: Path, adjudication: Path | None, approved_by: str | None, out: Path) -> int:
    """Final qrels = agreed labels + human-adjudicated disagreements. Refuses unless approved and complete."""
    if not approved_by or not approved_by.strip():
        raise SystemExit('Refusing: final qrels need explicit approval (--approved-by "<name>").')
    va, vb = validate(path_a), validate(path_b)
    if not (va['ok'] and vb['ok']):
        raise SystemExit(f'Refusing: annotator files are not complete/valid. A: {va["errors"]}  B: {vb["errors"]}')
    adj = {}
    if adjudication is not None:
        for r in read_rows(adjudication):
            g = (r.get('adjudicated_grade') or '').strip()
            if g not in {str(x) for x in GRADES}:
                raise SystemExit(f'Refusing: missing/invalid adjudicated_grade for {r["query_id"]}/{r["document_id"]}')
            adj[(r['query_id'], r['document_id'])] = int(g)
    qrels = []
    for key in sorted(va['labels']):
        a, b = va['labels'][key], vb['labels'][key]
        if a == b:
            qrels.append((key, a, 'agreed'))
        elif key in adj:
            qrels.append((key, adj[key], 'adjudicated'))
        else:
            raise SystemExit(f'Refusing: disagreement {key} (A={a}, B={b}) has no adjudicated grade.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_id', 'document_id', 'grade', 'source', 'approved_by'])
        for (q, d), g, src in qrels:
            w.writerow([q, d, g, src, approved_by.strip()])
    return len(qrels)


# ---------------------------------------------------------------- coverage + metrics
def load_qrels(path: Path) -> dict[Key, int]:
    return {(r['query_id'], r['document_id']): int(r['grade']) for r in read_rows(path)}


def load_runs(path: Path = PROVENANCE_CSV) -> dict[tuple[str, str], list[str]]:
    """{(query_id, method): [doc ids in rank order]} from the pooling provenance."""
    runs = defaultdict(list)
    for r in sorted(read_rows(path), key=lambda r: (r['query_id'], r['method'], int(r['rank']))):
        runs[(r['query_id'], r['method'])].append(r['document_id'])
    return dict(runs)


def coverage(run: list[str], qid: str, qrels: dict[Key, int], k: int = 10) -> float | None:
    top = run[:k]
    return None if not top else sum((qid, d) in qrels for d in top) / len(top)


def ndcg_at_k(run: list[str], qid: str, qrels: dict[Key, int], k: int = 10) -> float | None:
    """Graded nDCG@k (gain 2^g - 1). Ideal ranking from all judged docs for the query.
    Returns None if the query has no relevant (grade > 0) judged document."""
    gains = [qrels.get((qid, d)) for d in run[:k]]
    dcg = sum(((2 ** g - 1) if g else 0) / math.log2(i + 2) for i, g in enumerate(gains) if g is not None)
    ideal = sorted((g for (q, _), g in qrels.items() if q == qid), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return None if idcg == 0 else dcg / idcg


def mrr_at_k(run: list[str], qid: str, qrels: dict[Key, int], k: int = 10, min_grade: int = RELEVANT_MIN_GRADE) -> float:
    for i, d in enumerate(run[:k], 1):
        if qrels.get((qid, d), -1) >= min_grade:
            return 1 / i
    return 0.0


def evaluate(qrels: dict[Key, int], runs: dict[tuple[str, str], list[str]], min_coverage: float = 1.0, k: int = 10) -> dict:
    out = {'k': k, 'min_coverage': min_coverage, 'relevant_min_grade_for_mrr': RELEVANT_MIN_GRADE,
           'note': 'Unjudged documents are treated as unknown: metrics are only computed when top-k coverage '
                   '>= min_coverage. No recall is reported because pooled qrels are incomplete.',
           'per_run': {}, 'per_method_mean': {}}
    by_method = defaultdict(list)
    for (qid, method), run in sorted(runs.items()):
        cov = coverage(run, qid, qrels, k)
        entry = {'coverage_at_k': _r(cov), 'retrieved': len(run[:k])}
        if cov is None or cov < min_coverage:
            entry.update({'ndcg_at_k': 'not computed', 'mrr_at_k': 'not computed',
                          'reason': 'empty run' if cov is None else f'coverage {cov:.2f} < {min_coverage}'})
        else:
            nd = ndcg_at_k(run, qid, qrels, k)
            entry.update({'ndcg_at_k': _r(nd) if nd is not None else 'undefined (no relevant judged doc)',
                          'mrr_at_k': _r(mrr_at_k(run, qid, qrels, k))})
            if nd is not None:
                by_method[method].append((nd, entry['mrr_at_k']))
        out['per_run'][f'{qid}/{method}'] = entry
    for method, vals in sorted(by_method.items()):
        out['per_method_mean'][method] = {'queries': len(vals), 'ndcg_at_k': _r(sum(v[0] for v in vals) / len(vals)),
                                          'mrr_at_k': _r(sum(v[1] for v in vals) / len(vals))}
    return out


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('validate'); s.add_argument('file', type=Path); s.add_argument('--template', type=Path, default=HERE / 'annotator_A.csv')
    s = sub.add_parser('agreement'); s.add_argument('a', type=Path); s.add_argument('b', type=Path)
    s = sub.add_parser('disagreements'); s.add_argument('a', type=Path); s.add_argument('b', type=Path)
    s.add_argument('--out', type=Path, default=HERE / 'work' / 'disagreements.csv')
    s = sub.add_parser('finalize'); s.add_argument('a', type=Path); s.add_argument('b', type=Path)
    s.add_argument('--adjudication', type=Path); s.add_argument('--approved-by'); s.add_argument('--out', type=Path, default=HERE / 'work' / 'qrels.csv')
    s = sub.add_parser('coverage'); s.add_argument('--qrels', type=Path, required=True); s.add_argument('--runs', type=Path, default=PROVENANCE_CSV)
    s = sub.add_parser('evaluate'); s.add_argument('--qrels', type=Path, required=True); s.add_argument('--runs', type=Path, default=PROVENANCE_CSV)
    s.add_argument('--min-coverage', type=float, default=1.0)
    a = p.parse_args(argv)

    if a.cmd == 'validate':
        r = validate(a.file, a.template)
        print(json.dumps({'ok': r['ok'], 'errors': r['errors'], 'labelled': len(r['labels']),
                          'missing': [f'{q}/{d}' for q, d in r['missing'][:20]],
                          'invalid': [f'{q}/{d} (line {ln}): {v!r}' for (q, d), v, ln in r['invalid'][:20]]},
                         ensure_ascii=False, indent=2))
        return 0 if r['ok'] else 1
    if a.cmd == 'agreement':
        print(json.dumps(agreement(a.a, a.b), ensure_ascii=False, indent=2))
    elif a.cmd == 'disagreements':
        print(f'{disagreements(a.a, a.b, a.out)} disagreements written to {a.out}')
    elif a.cmd == 'finalize':
        print(f'{finalize(a.a, a.b, a.adjudication, a.approved_by, a.out)} qrels written to {a.out}')
    elif a.cmd == 'coverage':
        qrels, runs = load_qrels(a.qrels), load_runs(a.runs)
        for (qid, m), run in sorted(runs.items()):
            print(f'{qid}  {m:<18} top-10 judged: {coverage(run, qid, qrels):.2f}')
    elif a.cmd == 'evaluate':
        print(json.dumps(evaluate(load_qrels(a.qrels), load_runs(a.runs), a.min_coverage), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
