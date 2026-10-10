"""Milestone 2 evaluation: Singlish->Sinhala normalisation. EXPLORATORY ONLY.

Part A - normalisation quality on data/normalization_reference_draft.csv
         (word-level references DRAFTED by an AI assistant, pending native review;
         words were taken from the seed queries, so this is in-domain, not held-out).
Part B - retrieval: original query vs normalised query, for BM25 and LaBSE, on the
         same exact-text-linked seed pairs used by evaluate_crossscript.py
         (unverified known positives, incomplete qrels -> Hit@k / MRR@10 only).

Writes results/normalization_eval.json and results/normalization_eval.md.
"""
import csv
import json
from collections import Counter, defaultdict

from evaluate_crossscript import load_linked, rank_metrics, summarise
from retrieval import (DATA, NORM_BM25_METHOD, NORM_LABSE_METHOD, ROOT, SEMANTIC_METHOD,
                       RetrievalEngine)

PAIRS = [('BM25 (lexical)', NORM_BM25_METHOD), (SEMANTIC_METHOD, NORM_LABSE_METHOD)]


def word_quality(normalizer):
    """Score each reference word; 'KEEP' means leaving the Latin word is correct."""
    rows, by_cat = [], defaultdict(Counter)
    with (DATA / 'normalization_reference_draft.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            accepted = r['accepted_sinhala'].split(';')
            t = normalizer.normalize_word(r['singlish'])
            converted = t.status in ('transliterated', 'ambiguous')
            correct = t.output in accepted or ('KEEP' in accepted and t.output == r['singlish'])
            outcome = ('correct' if correct else
                       'wrong_flagged_ambiguous' if t.status == 'ambiguous' else
                       'wrong_silent' if converted else 'abstained')
            by_cat[r['category']][outcome] += 1
            by_cat['ALL'][outcome] += 1
            rows.append({'singlish': r['singlish'], 'expected': r['accepted_sinhala'], 'output': t.output,
                         'status': t.status, 'outcome': outcome, 'alternatives': t.alternatives})
    summary = {}
    for cat, c in by_cat.items():
        n = sum(c.values())
        summary[cat] = {'n': n, **{k: c[k] for k in ('correct', 'wrong_silent', 'wrong_flagged_ambiguous', 'abstained')},
                        'accuracy': round(c['correct'] / n, 4)}
    return summary, rows


def retrieval_comparison(engine):
    positives, text, by_script = load_linked({d['document_id']: d['comment_text'] for d in engine.docs})
    norm = {q: engine.normalize(text[q]) for q in positives}
    ready, msg = engine.semantic_status()
    pairs = PAIRS if ready else PAIRS[:1]
    if not ready:
        print(f'[LaBSE skipped] {msg}')

    out = {'n_queries': len(positives),
           'query_normalization_status': dict(Counter(n.status for n in norm.values())),
           'mean_query_coverage': round(sum(n.coverage for n in norm.values()) / len(norm), 4),
           'methods': {}, 'per_query_sinhala_target': []}
    ranks = {}
    for pair in pairs:
        for method in pair:
            cats = defaultdict(list)
            for q, pos in positives.items():
                ranked = [d['document_id'] for d, _ in engine.search(text[q], method, 10)]
                cats['all_original_queries'].append(rank_metrics(ranked, pos))
                for script, spos in by_script[q].items():
                    if spos:
                        m = rank_metrics(ranked, spos)
                        cats['singlish_to_' + script].append(m)
                        if script == 'sinhala':
                            cats[f'singlish_to_sinhala|norm_{norm[q].status}'].append(m)
                            ranks[(q, method)] = 1 / m['rr10'] if m['rr10'] else None
            out['methods'][method] = {c: summarise(v) for c, v in sorted(cats.items())}

    # Leakage ablation: lexicon built WITHOUT any known-positive comment, so a word
    # that occurs only in the target comment cannot be produced by normalisation.
    from normalization import from_corpus
    all_pos = set().union(*positives.values())
    held_out = from_corpus([d for d in engine.docs if d['document_id'] not in all_pos])
    ablation = {}
    for base, _ in pairs:
        cats = defaultdict(list)
        for q, pos in positives.items():
            ranked = [d['document_id'] for d, _ in engine.search(held_out.normalize(text[q]).normalized, base, 10)]
            for script, spos in by_script[q].items():
                if spos:
                    cats['singlish_to_' + script].append(rank_metrics(ranked, spos))
        ablation[f'Normalized (lexicon excl. known positives) -> {base}'] = {c: summarise(v) for c, v in sorted(cats.items())}
    out['leakage_ablation'] = ablation

    for q in positives:
        if by_script[q]['sinhala']:
            out['per_query_sinhala_target'].append({
                'query': text[q], 'normalized': norm[q].normalized, 'status': norm[q].status,
                'coverage': round(norm[q].coverage, 3),
                'rank_at_10': {m: ranks.get((q, m)) for pair in pairs for m in pair}})
    return out


def to_markdown(rep):
    L = ['# Milestone 2 - Normalisation evaluation (EXPLORATORY, not a benchmark)', '',
         rep['warning'], '', '## A. Word-level normalisation quality (draft references)', '',
         '| Category | n | Correct | Wrong (silent) | Wrong (flagged ambiguous) | Abstained (kept Latin) | Accuracy |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for cat, s in sorted(rep['word_quality'].items()):
        L.append(f"| {cat} | {s['n']} | {s['correct']} | {s['wrong_silent']} | {s['wrong_flagged_ambiguous']} | {s['abstained']} | {s['accuracy']:.3f} |")
    r = rep['retrieval']
    L += ['', '## B. Retrieval: original vs normalised query', '',
          f"Queries: {r['n_queries']}; query-level normalisation status: {r['query_normalization_status']}; "
          f"mean word coverage {r['mean_query_coverage']:.2f}", '',
          '| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for method, cats in r['methods'].items():
        for c, s in cats.items():
            if s.get('n_queries'):
                L.append(f"| {method} | {c} | {s['n_queries']} | {s['known_positive_hit_at_1']:.3f} | "
                         f"{s['known_positive_hit_at_10']:.3f} | {s['known_positive_mrr_at_10']:.3f} |")
    L += ['', '### Leakage ablation (lexicon excludes all known-positive comments)', '',
          '| Method | Category | n | Hit@1 | Hit@10 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for method, cats in r.get('leakage_ablation', {}).items():
        for c, s in cats.items():
            L.append(f"| {method} | {c} | {s['n_queries']} | {s['known_positive_hit_at_1']:.3f} | "
                     f"{s['known_positive_hit_at_10']:.3f} | {s['known_positive_mrr_at_10']:.3f} |")
    return '\n'.join(L) + '\n'


def main():
    engine = RetrievalEngine()
    summary, rows = word_quality(engine.normalizer)
    rep = {'warning': 'Reference words are AI-drafted and await native-speaker review; retrieval positives '
                      'are unverified seed pairs with incomplete qrels. Hit@k/MRR@10 are exploratory; no '
                      'significance testing; n is small.',
           'normalizer': engine.normalizer.name, 'word_quality': summary, 'word_rows': rows,
           'retrieval': retrieval_comparison(engine)}
    (ROOT / 'results' / 'normalization_eval.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    md = to_markdown(rep)
    (ROOT / 'results' / 'normalization_eval.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
