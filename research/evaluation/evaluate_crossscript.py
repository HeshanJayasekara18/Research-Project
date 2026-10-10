"""Exploratory cross-script diagnostic. NOT an independent benchmark.

Uses the same exact-text-linked seed pairs as evaluate_seed.py (known positives,
incomplete and unverified qrels) and breaks results down by query category:

  1. singlish_to_latin     - scored only against positives whose corpus text
                             is mostly Latin script (Singlish/English)
  2. singlish_to_sinhala   - scored only against positives whose corpus text
                             is mostly Sinhala script (a query with positives in
                             both scripts appears in both categories)
  3. spelling_variants     - the same queries rewritten by deterministic,
                             documented Singlish spelling rules (SYNTHETIC
                             variants, not real user queries)

Metrics are known-positive Hit@1, Hit@10 and MRR@10. Unjudged documents are not
treated as negatives in any formal sense; scores are a lower bound.
"""
import csv
import json
import re
from collections import defaultdict

from rag_service.retrieval.engine import DATA, RESULTS, SEMANTIC_METHOD, RetrievalEngine

BASE_METHODS = ('BM25 (lexical)', 'Character TF-IDF')
SINHALA_RE = re.compile(r'[඀-෿]')
LATIN_RE = re.compile(r'[A-Za-z]')

# Common Romanised-Sinhala spelling alternations observed in informal writing.
CONSONANT_RULES = [(r'th', 't'), (r'dh', 'd'), (r'w', 'v')]
VOWEL_RULES = [(r'aa', 'a'), (r'ee', 'i'), (r'oo', 'u'), (r'(?<=[a-z])i\b', 'ee')]


def script_of(text):
    return 'sinhala' if len(SINHALA_RE.findall(text)) > len(LATIN_RE.findall(text)) else 'latin'


def spelling_variants(query):
    """Return deterministic variants that differ from the input (may be empty)."""
    out = []
    for rules in (CONSONANT_RULES, VOWEL_RULES):
        v = query.lower()
        for pat, rep in rules:
            v = re.sub(pat, rep, v)
        if v != query.lower() and v not in out:
            out.append(v)
    return out


def load_linked(doc_text):
    """Known positives per query, split by the script of each positive's corpus text."""
    positives, text = defaultdict(set), {}
    with (DATA / 'seed_pairs_linked.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['matched_document_ids']:
                qid = row['query_id']
                positives[qid].update(row['matched_document_ids'].split(';'))
                text[qid] = row['singlish_query']
    by_script = {q: {s: {d for d in pos if script_of(doc_text[d]) == s} for s in ('latin', 'sinhala')}
                 for q, pos in positives.items()}
    return positives, text, by_script


def rank_metrics(ranked_ids, positives):
    first = next((i for i, d in enumerate(ranked_ids[:10], 1) if d in positives), None)
    return {'hit1': int(first == 1), 'hit10': int(first is not None), 'rr10': 1 / first if first else 0.0}


def summarise(rows):
    n = len(rows)
    if not n:
        return {'n_queries': 0}
    return {'n_queries': n,
            'known_positive_hit_at_1': round(sum(r['hit1'] for r in rows) / n, 4),
            'known_positive_hit_at_10': round(sum(r['hit10'] for r in rows) / n, 4),
            'known_positive_mrr_at_10': round(sum(r['rr10'] for r in rows) / n, 4)}


def diagnostic(engine=None, methods=None):
    engine = engine or RetrievalEngine()
    if methods is None:
        ready, msg = engine.semantic_status()
        methods = BASE_METHODS + ((SEMANTIC_METHOD,) if ready else ())
        if not ready:
            print(f'[LaBSE skipped] {msg}')
    positives, text, by_script = load_linked({d['document_id']: d['comment_text'] for d in engine.docs})

    report = {'warning': 'Exploratory: unverified known-positive seed pairs, incomplete qrels, '
                         'queries may be written from the documents; variants are synthetic. '
                         'NOT nDCG/Recall on an independent benchmark.',
              'variant_rules': {'consonant': CONSONANT_RULES, 'vowel': VOWEL_RULES},
              'methods': {}, 'examples': []}
    for method in methods:
        per_cat = defaultdict(list)
        variant_overlap = []
        for qid, pos in positives.items():
            ranked = [d['document_id'] for d, _ in engine.search(text[qid], method, 10)]
            m = rank_metrics(ranked, pos)
            for script, spos in by_script[qid].items():
                if spos:  # score only against positives written in this script
                    per_cat['singlish_to_' + script].append(rank_metrics(ranked, spos))
            for var in spelling_variants(text[qid]):
                vranked = [d['document_id'] for d, _ in engine.search(var, method, 10)]
                per_cat['spelling_variants'].append(rank_metrics(vranked, pos))
                if ranked or vranked:
                    variant_overlap.append(len(set(ranked) & set(vranked)) / len(set(ranked) | set(vranked)))
            per_cat['all_original_queries'].append(m)
        report['methods'][method] = {cat: summarise(rows) for cat, rows in sorted(per_cat.items())}
        report['methods'][method]['variant_vs_original_top10_jaccard'] = (
            round(sum(variant_overlap) / len(variant_overlap), 4) if variant_overlap else None)

    # A few concrete Singlish->Sinhala examples for the presentation (real corpus text).
    for qid in [q for q in positives if by_script[q]['sinhala']][:5]:
        ex = {'query': text[qid], 'variants': spelling_variants(text[qid]), 'top1': {}}
        for method in methods:
            hits = engine.search(text[qid], method, 1)
            if hits:
                d, score = hits[0]
                ex['top1'][method] = {'document_id': d['document_id'], 'is_known_positive': d['document_id'] in positives[qid],
                                      'script': script_of(d['comment_text']),
                                      'score': score, 'comment': d['comment_text'][:160]}
        report['examples'].append(ex)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'crossscript_diagnostic.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def to_markdown(report):
    cats = ['singlish_to_latin', 'singlish_to_sinhala', 'spelling_variants', 'all_original_queries']
    lines = ['| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for method, res in report['methods'].items():
        for c in cats:
            r = res.get(c, {})
            if r.get('n_queries'):
                lines.append(f"| {method} | {c} | {r['n_queries']} | {r['known_positive_hit_at_1']:.3f} | "
                             f"{r['known_positive_hit_at_10']:.3f} | {r['known_positive_mrr_at_10']:.3f} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    rep = diagnostic()
    print(to_markdown(rep))
    for m, r in rep['methods'].items():
        print(f"{m}: mean top-10 Jaccard(original, variant) = {r['variant_vs_original_top10_jaccard']}")
    print('Saved results/crossscript_diagnostic.json')
