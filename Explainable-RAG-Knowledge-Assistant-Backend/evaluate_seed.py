"""Exploratory paired-data diagnostic. NOT independent benchmark evaluation.

The manually reviewed queries were often written from comments; labels are not
independently assessed. Results are therefore 'known-positive hit@k' ONLY.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from retrieval import RetrievalEngine, DATA, ROOT, SEMANTIC_METHOD

METHODS = ('BM25 (lexical)', 'Character TF-IDF')

def diagnostic(engine=None, methods=None):
    engine = engine or RetrievalEngine()
    if methods is None:  # include LaBSE only when its index/model are actually usable
        ready, msg = engine.semantic_status()
        methods = METHODS + ((SEMANTIC_METHOD,) if ready else ())
        if not ready:
            print(f'[LaBSE skipped] {msg}')
    groups = defaultdict(set)
    query_text = {}
    with (DATA/'seed_pairs_linked.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['matched_document_ids']:
                groups[row['query_id']].update(row['matched_document_ids'].split(';'))
                query_text[row['query_id']] = row['singlish_query']

    if not groups:
        return {'warning':'No seed examples could be linked to the active corpus.'}

    metrics = {'warning': 'Exploratory, incomplete qrels; queries may be written from documents. NOT held-out nDCG/Recall.',
               'eligible_unique_queries': len(groups), 'methods': {}}
    for method in methods:
        hit_at_1 = hit_at_10 = 0
        ranked_examples = []
        for qid, positives in groups.items():
            hits = engine.search(query_text[qid], method=method, top_k=10)
            ids = [d['document_id'] for d, score in hits]
            h1 = bool(set(ids[:1]) & positives)
            h10 = bool(set(ids) & positives)
            hit_at_1 += int(h1)
            hit_at_10 += int(h10)
            ranked_examples.append({'query':query_text[qid], 'hit_at_10':h10,
                                    'returned_results':len(ids)})
        metrics['methods'][method] = {
            'known_positive_hit_at_1': round(hit_at_1/len(groups),4),
            'known_positive_hit_at_10': round(hit_at_10/len(groups),4),
            'queries_with_hit_10': hit_at_10,
            'queries_evaluated':len(groups),
        }
        if method == 'BM25 (lexical)':
            metrics['sample_bm25_hits'] = [x['query'] for x in ranked_examples if x['hit_at_10']][:8]
            metrics['sample_bm25_misses'] = [x['query'] for x in ranked_examples if not x['hit_at_10']][:8]

    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'seed_diagnostic.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    return metrics

if __name__=='__main__':
    print(json.dumps(diagnostic(),indent=2,ensure_ascii=False))
