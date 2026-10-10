"""Milestone 3 evaluation: multi-view RRF retrieval. EXPLORATORY ONLY.

Same 82 exact-text-linked seed queries as evaluate_crossscript.py, scored with
known-positive Hit@1 / Hit@10 / MRR@10, split by the script of each positive.
Two positive sets are reported side by side:

  seed_all       - every linked seed pair (as in Milestones 1-2)
  audit_correct  - only pairs labelled 'correct' in data/seed_pair_audit.csv
                   (assistant draft audit, pending human confirmation)

Unjudged documents are NOT negatives; these are not nDCG/Recall benchmark numbers.
Primary configuration (fixed before running): RRF k=60, depth 50 per view.
Writes results/multiview_eval.json and results/multiview_eval.md.
"""
import csv
import json
from collections import defaultdict

from evaluate_crossscript import load_linked, rank_metrics, script_of, summarise
from retrieval import (DATA, MV_BM25_METHOD, MV_HYBRID_METHOD, MV_LABSE_METHOD, NORM_BM25_METHOD,
                       NORM_LABSE_METHOD, ROOT, SEMANTIC_METHOD, RetrievalEngine)

LEXICAL = ['BM25 (lexical)', NORM_BM25_METHOD, MV_BM25_METHOD]
DENSE = [SEMANTIC_METHOD, NORM_LABSE_METHOD, MV_LABSE_METHOD, MV_HYBRID_METHOD]
SHORT = {'BM25 (lexical)': 'BM25 original', NORM_BM25_METHOD: 'BM25 normalized',
         MV_BM25_METHOD: 'RRF: BM25 orig+norm', SEMANTIC_METHOD: 'LaBSE original',
         NORM_LABSE_METHOD: 'LaBSE normalized', MV_LABSE_METHOD: 'RRF: LaBSE orig+norm',
         MV_HYBRID_METHOD: 'RRF: hybrid BM25+LaBSE orig+norm'}


def audited_correct_ids():
    ok = defaultdict(set)
    with (DATA / 'seed_pair_audit.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['audit_label'] == 'correct':
                ok[r['query_id']].update(r['matched_document_ids'].split(';'))
    return ok


def evaluate(engine, methods, positives, text, doc_text, k_override=None):
    out = {}
    for method in methods:
        cats = defaultdict(list)
        for q, pos in positives.items():
            if k_override is not None:
                hits, _ = engine.multiview(text[q], method, 10, k=k_override)
                ranked = [d['document_id'] for d, _ in hits]
            else:
                ranked = [d['document_id'] for d, _ in engine.search(text[q], method, 10)]
            cats['all_queries'].append(rank_metrics(ranked, pos))
            for script in ('latin', 'sinhala'):
                spos = {d for d in pos if script_of(doc_text[d]) == script}
                if spos:
                    cats['singlish_to_' + script].append(rank_metrics(ranked, spos))
        out[method] = {c: summarise(v) for c, v in sorted(cats.items())}
    return out


def to_markdown(rep):
    L = ['# Milestone 3 - Multi-view RRF evaluation (EXPLORATORY, not a benchmark)', '', rep['warning'], '',
         f"RRF k={rep['config']['k']}, depth={rep['config']['depth']} per view. "
         f"Audit labels: {rep['audit_label_counts']}", '']
    for pos_set, res in rep['results'].items():
        L += [f'## Positive set: {pos_set}', '',
              '| Method | Sinhala-target n | Hit@1 | Hit@10 | MRR@10 | Latin-target n | Hit@1 | Hit@10 | MRR@10 | All n | Hit@10 | MRR@10 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m, c in res.items():
            si, la, al = c.get('singlish_to_sinhala', {}), c.get('singlish_to_latin', {}), c['all_queries']
            f = lambda s, key: f"{s[key]:.3f}" if s.get('n_queries') else '-'
            L.append(f"| {SHORT.get(m, m)} | {si.get('n_queries', 0)} | {f(si, 'known_positive_hit_at_1')} | "
                     f"{f(si, 'known_positive_hit_at_10')} | {f(si, 'known_positive_mrr_at_10')} | "
                     f"{la.get('n_queries', 0)} | {f(la, 'known_positive_hit_at_1')} | {f(la, 'known_positive_hit_at_10')} | "
                     f"{f(la, 'known_positive_mrr_at_10')} | {al['n_queries']} | {f(al, 'known_positive_hit_at_10')} | "
                     f"{f(al, 'known_positive_mrr_at_10')} |")
        L.append('')
    L += ['## Sensitivity: RRF k (seed_all, BM25 orig+norm)', '', '| k | Sinhala Hit@10 | Latin Hit@10 | All Hit@10 |', '|---:|---:|---:|---:|']
    for k, c in rep['k_sensitivity'].items():
        L.append(f"| {k} | {c['singlish_to_sinhala']['known_positive_hit_at_10']:.3f} | "
                 f"{c['singlish_to_latin']['known_positive_hit_at_10']:.3f} | {c['all_queries']['known_positive_hit_at_10']:.3f} |")
    return '\n'.join(L) + '\n'


def main():
    engine = RetrievalEngine()
    doc_text = {d['document_id']: d['comment_text'] for d in engine.docs}
    positives, text, _ = load_linked(doc_text)
    ready, msg = engine.semantic_status()
    methods = LEXICAL + (DENSE if ready else [])
    if not ready:
        print(f'[LaBSE skipped] {msg}')

    ok = audited_correct_ids()
    audited = {q: positives[q] & ok[q] for q in positives if positives[q] & ok[q]}
    labels = defaultdict(int)
    with (DATA / 'seed_pair_audit.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            labels[r['audit_label']] += 1

    rep = {'warning': 'Known-positive Hit@k/MRR@10 on unverified seed pairs (incomplete qrels; queries may '
                      'have been written from the comments). Audit is an assistant draft pending human '
                      'confirmation. No significance testing; small n.',
           'config': {'k': 60, 'depth': 50}, 'audit_label_counts': dict(labels),
           'results': {'seed_all': evaluate(engine, methods, positives, text, doc_text),
                       'audit_correct': evaluate(engine, methods, audited, text, doc_text)},
           'k_sensitivity': {k: evaluate(engine, [MV_BM25_METHOD], positives, text, doc_text, k_override=k)[MV_BM25_METHOD]
                             for k in (10, 60, 100)}}
    (ROOT / 'results' / 'multiview_eval.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    md = to_markdown(rep)
    (ROOT / 'results' / 'multiview_eval.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
