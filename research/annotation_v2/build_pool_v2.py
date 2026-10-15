"""Corpus V2 annotation pool builder (domain-balanced, 3 x Food / Gaming / Education).

    $env:PYTHONPATH = "src"
    python -m research.annotation_v2.build_pool_v2 --corpus-dir "<V2 export dir>" [--queries research/annotation_v2/queries.csv]

Refuses to run unless the export's export_manifest.json records an ethics_reference
(research-use authorization) - real comments are not pooled before that.
For each question, every available method retrieves its top-10 within the question's video
(filtered before ranking); the pool is the full deduplicated union (never truncated).
Writes BLANK annotator sheets (A and B, identical, seeded random order, no methods/ranks)
and hidden provenance. Refuses to overwrite existing sheets. No labels are ever produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from rag_service.retrieval.corpus_v2 import build_engine_v2
from rag_service.retrieval.engine import MV_BM25_METHOD, NORM_BM25_METHOD, SEMANTIC_METHOD

HERE = Path(__file__).resolve().parent
TOP_K = 10
SEED = 20261016
LEXICAL = {'bm25': 'BM25 (lexical)', 'normalized_bm25': NORM_BM25_METHOD, 'char_tfidf': 'Character TF-IDF',
           'multiview_bm25': MV_BM25_METHOD}
SHEET_FIELDS = ['query_id', 'domain', 'query_text', 'video_id', 'comment_id', 'original_comment_text',
                'relevance_grade', 'annotation_note']


def read(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(corpus_dir: Path, queries_path: Path, out_dir: Path) -> dict:
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / 'export_manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Refusing: {manifest_path} not found (use the research-safe export).')
    export_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not str(export_manifest.get('ethics_reference', '')).strip():
        raise SystemExit('Refusing: the export has no ethics_reference (research-use authorization not recorded).')
    sheets = [out_dir / 'annotator_A.csv', out_dir / 'annotator_B.csv']
    if any(p.exists() for p in sheets):
        raise SystemExit('Refusing to overwrite existing annotation sheets.')
    queries = read(queries_path)
    domains = [q['domain'] for q in queries]
    if sorted(set(domains)) != ['education', 'food', 'gaming']:
        raise SystemExit(f'Queries must cover food, gaming and education; got {sorted(set(domains))}')

    engine = build_engine_v2(corpus_dir)
    methods = dict(LEXICAL)
    if engine.has_semantic_index():
        methods['labse'] = SEMANTIC_METHOD
    unavailable = {} if 'labse' in methods else {'labse': 'no V2 LaBSE index (not built in this milestone)'}

    prov, rows, per_query = [], [], {}
    for q in queries:
        vid = q['video_id']
        if vid not in engine.video_rows:
            raise SystemExit(f'{q["query_id"]}: video {vid} has no indexable comments in the export')
        if engine.docs[engine.video_rows[vid][0]]['domain'] != q['domain']:
            raise SystemExit(f'{q["query_id"]}: video {vid} is not in domain {q["domain"]}')
        pool = {}
        for mid, name in methods.items():
            for rank, (doc, score) in enumerate(engine.search(q['query_text'], name, TOP_K, vid), 1):
                assert doc['video_id'] == vid
                pool.setdefault(doc['comment_id'], []).append(mid)
                prov.append({'query_id': q['query_id'], 'domain': q['domain'], 'video_id': vid, 'method': mid,
                             'rank': rank, 'score': score, 'comment_id': doc['comment_id']})
        ids = sorted(pool)
        random.Random(f"{SEED}-{q['query_id']}").shuffle(ids)
        for cid in ids:
            rows.append({'query_id': q['query_id'], 'domain': q['domain'], 'query_text': q['query_text'],
                         'video_id': vid, 'comment_id': cid, 'original_comment_text': engine.by_id[cid]['comment_text'],
                         'relevance_grade': '', 'annotation_note': ''})
        per_query[q['query_id']] = {'domain': q['domain'], 'video_id': vid, 'pooled_candidates': len(ids),
                                    'video_indexable_comments': len(engine.video_rows[vid])}

    for p in sheets:
        write(p, SHEET_FIELDS, rows)
    write(out_dir / 'provenance' / 'pool_provenance.csv',
          ['query_id', 'domain', 'video_id', 'method', 'rank', 'score', 'comment_id'], prov)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'corpus': 'v2', 'export_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'ethics_reference': export_manifest['ethics_reference'],
        'comments_v2_sha256': hashlib.sha256((corpus_dir / 'comments_v2.csv').read_bytes()).hexdigest(),
        'queries_sha256': hashlib.sha256(Path(queries_path).read_bytes()).hexdigest(),
        'top_k_per_method': TOP_K, 'methods_evaluated': list(methods), 'methods_unavailable': unavailable,
        'pool_rule': "full deduplicated union of every evaluated method's top-10 within the question's video",
        'sheet_order': f'seeded shuffle per query (seed {SEED}); A and B identical', 'queries': per_query,
    }
    (out_dir / 'provenance' / 'pool_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                               encoding='utf-8')
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-dir', type=Path, required=True)
    p.add_argument('--queries', type=Path, default=HERE / 'queries.csv')
    p.add_argument('--out', type=Path, default=HERE)
    a = p.parse_args(argv)
    m = build(a.corpus_dir, a.queries, a.out)
    print(json.dumps({'methods': m['methods_evaluated'], 'unavailable': m['methods_unavailable'],
                      'pools': {q: v['pooled_candidates'] for q, v in m['queries'].items()}}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
