"""M7 pilot: build pooled relevance candidates and BLANK annotation sheets.

    python -m research.annotation.build_pool

For each question, every evaluated method retrieves its top-10 *within the question's
video* (filtering before ranking). The pool is the full deduplicated union of those
top-10 lists; it is never truncated. Annotation sheets list query-document pairs in a
seeded random order and contain no method, rank or score information. Provenance
(methods, ranks, scores, dataset hash, scope) is written separately.

No relevance labels are produced here: both annotator sheets are blank.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rag_service.retrieval.engine import (DATA, MV_BM25_METHOD, MV_HYBRID_METHOD, NORM_BM25_METHOD, SEMANTIC_METHOD,
                                          RetrievalEngine, tokenize)

HERE = Path(__file__).resolve().parent
PROVENANCE = HERE / 'provenance'
TOP_K = 10
SHUFFLE_SEED = 20261014

# Five independent creator information needs. Written from the creator's perspective
# about each video's topic, NOT by paraphrasing any known seed-positive comment.
QUERIES = [
    {'query_id': 'Q1', 'video_id': '5AJ81b3rqmc',
     'query_text': 'Dry skin ekata hari pigmentation walata hondama lotion eka monawada kiyala viewers la ahanawada?',
     'information_need': 'Which skincare products viewers ask about for dry skin or pigmentation. / '
                         'වියළි සම හෝ පිග්මන්ටේෂන් සඳහා නරඹන්නන් අහන නිෂ්පාදන මොනවාද.',
     'design_note': 'Mostly Singlish comments; product-recommendation need.'},
    {'query_id': 'Q2', 'video_id': 'QL2J3Va-fJM',
     'query_text': 'Clothing business ekak patan ganna supplier kenek hari wholesale thogayak hoyaganne kohomada kiyala kattiya ahanawada?',
     'information_need': 'Whether viewers ask how to find suppliers / wholesale stock to start a clothing business. / '
                         'ඇඳුම් ව්‍යාපාරයක් පටන් ගන්න සැපයුම්කරුවන් හෝ තොග හොයාගන්නේ කොහොමද කියා අහනවාද.',
     'design_note': 'Mixed English, Singlish and Sinhala-script comments.'},
    {'query_id': 'Q3', 'video_id': 'NXDizOPtZ98',
     'query_text': 'Inguru wagawen labena aadayama saha wiyadam gana viewers la sandeha karanawada?',
     'information_need': 'Whether viewers doubt the income or costs claimed for ginger cultivation. / '
                         'ඉඟුරු වගාවේ ආදායම හා වියදම ගැන නරඹන්නන් සැක පළ කරනවාද.',
     'design_note': 'Sinhala-script retrieval need (about half the comments are Sinhala script).'},
    {'query_id': 'Q4', 'video_id': 'wrxMnZEWT7U',
     'query_text': 'Me aluth construction kramaya Lankawe gewal hadanna pawichchi karanna puluwanda kiyala minissu ahanawada?',
     'information_need': 'Whether viewers ask if this construction method can be used for houses in Sri Lanka. / '
                         'මේ ඉදිකිරීම් ක්‍රමය ලංකාවේ ගෙවල් හදන්න පාවිච්චි කරන්න පුළුවන්ද කියා අහනවාද.',
     'design_note': 'Sinhala-script heavy video; cross-script need.'},
    {'query_id': 'Q5', 'video_id': 'oLACzeVFCUk',
     'query_text': 'Bike eke insurance saha license gana viewers la monawada kiyanne?',
     'information_need': 'What viewers say about motorbike insurance and licences. / '
                         'යතුරුපැදි රක්ෂණය සහ බලපත්‍ර ගැන නරඹන්නන් කියන්නේ මොනවාද.',
     'design_note': 'Expected to have little or no relevant evidence (insufficient-evidence case).'},
]

# API-style method ids -> engine method names. LaBSE-based methods are included only if they load.
METHODS = {
    'bm25': 'BM25 (lexical)',
    'normalized_bm25': NORM_BM25_METHOD,
    'char_tfidf': 'Character TF-IDF',
    'multiview_bm25': MV_BM25_METHOD,
    'labse': SEMANTIC_METHOD,
    'multiview_hybrid': MV_HYBRID_METHOD,
}
SEMANTIC_DEPENDENT = {'labse', 'multiview_hybrid'}

SHEET_FIELDS = ['query_id', 'query_text', 'video_id', 'document_id', 'original_comment_text',
                'relevance_grade', 'annotation_note']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def max_seed_overlap(query: str) -> tuple[float, str]:
    """Highest token-Jaccard similarity between the question and any seed query (paraphrase check)."""
    q = set(tokenize(query))
    best, best_q = 0.0, ''
    with (DATA / 'seed_pairs_linked.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            s = set(tokenize(row['singlish_query']))
            j = len(q & s) / len(q | s) if q | s else 0.0
            if j > best:
                best, best_q = j, row['singlish_query']
    return round(best, 3), best_q


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens Sinhala text correctly.
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    engine = RetrievalEngine()
    semantic_ok, semantic_msg = engine.semantic_status()
    evaluated = {m: n for m, n in METHODS.items() if semantic_ok or m not in SEMANTIC_DEPENDENT}
    unavailable = {m: semantic_msg for m in METHODS if m not in evaluated}

    provenance_rows, sheet_rows, per_query = [], [], {}
    for q in QUERIES:
        vid = q['video_id']
        if vid not in engine.video_rows:
            raise SystemExit(f'{q["query_id"]}: video {vid} not in corpus')
        pool: dict[str, list[tuple[str, int]]] = {}
        for method_id, engine_name in evaluated.items():
            hits = engine.search(q['query_text'], engine_name, TOP_K, vid)
            for rank, (doc, score) in enumerate(hits, 1):
                assert doc['video_id'] == vid  # filtering before ranking: never another video
                pool.setdefault(doc['document_id'], []).append((method_id, rank))
                provenance_rows.append({'query_id': q['query_id'], 'video_id': vid, 'method': method_id,
                                        'rank': rank, 'score': score, 'document_id': doc['document_id']})
        doc_ids = sorted(pool)
        random.Random(f"{SHUFFLE_SEED}-{q['query_id']}").shuffle(doc_ids)  # order reveals no rank
        for d in doc_ids:
            sheet_rows.append({'query_id': q['query_id'], 'query_text': q['query_text'], 'video_id': vid,
                               'document_id': d, 'original_comment_text': engine.by_id[d]['comment_text'],
                               'relevance_grade': '', 'annotation_note': ''})
        overlap, closest = max_seed_overlap(q['query_text'])
        per_query[q['query_id']] = {
            'video_id': vid, 'video_comments': len(engine.video_rows[vid]), 'pooled_candidates': len(doc_ids),
            'retrieved_per_method': {m: sum(1 for r in provenance_rows if r['query_id'] == q['query_id'] and r['method'] == m)
                                     for m in evaluated},
            'max_seed_query_jaccard': overlap, 'closest_seed_query': closest, 'design_note': q['design_note']}

    write_csv(HERE / 'queries.csv', ['query_id', 'query_text', 'video_id', 'video_url', 'information_need'],
              [{'query_id': q['query_id'], 'query_text': q['query_text'], 'video_id': q['video_id'],
                'video_url': f"https://www.youtube.com/watch?v={q['video_id']}",
                'information_need': q['information_need']} for q in QUERIES])
    for name in ('annotator_A.csv', 'annotator_B.csv'):
        write_csv(HERE / name, SHEET_FIELDS, sheet_rows)
    write_csv(PROVENANCE / 'pool_provenance.csv', ['query_id', 'video_id', 'method', 'rank', 'score', 'document_id'],
              provenance_rows)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'dataset': {'file': 'data/comments_clean.csv', 'sha256': sha256(DATA / 'comments_clean.csv'),
                    'documents': len(engine.docs)},
        'scope': 'video (each question searches only its own video; filtering before ranking)',
        'top_k_per_method': TOP_K,
        'methods_evaluated': list(evaluated),
        'methods_unavailable': unavailable,
        'semantic_status': semantic_msg,
        'pool_rule': 'full deduplicated union of every evaluated method\'s top-10; not truncated; no extra candidates',
        'sheet_order': f'seeded random shuffle per query (seed {SHUFFLE_SEED}); identical order in A and B',
        'queries': per_query,
    }
    (PROVENANCE / 'pool_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('methods_evaluated', 'methods_unavailable')}, ensure_ascii=False))
    for qid, info in per_query.items():
        print(qid, info['video_id'], 'pool =', info['pooled_candidates'], '| per method', info['retrieved_per_method'],
              '| max seed Jaccard', info['max_seed_query_jaccard'])


if __name__ == '__main__':
    main()
