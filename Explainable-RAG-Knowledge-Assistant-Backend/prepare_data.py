"""Prepare a privacy-minimized corpus and link existing seed query/comment pairs.

No model inference or artificial relevance labels are created here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def normalize_text(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s or "")).split())


def normalize_video_id(text: str) -> str:
    raw = str(text or "").strip()
    if VIDEO_ID_RE.fullmatch(raw):
        return raw
    if raw.startswith(('https://', 'http://')):
        parsed = urlparse(raw)
        cand = parse_qs(parsed.query).get('v', [''])[0]
        if not cand and parsed.hostname in ('youtu.be', 'www.youtu.be'):
            cand = parsed.path.strip('/').split('/')[0]
    else:
        cand = raw.split('&')[0].split('?')[0]
    return cand if VIDEO_ID_RE.fullmatch(cand) else ''


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / 'youtube_comments_multiple.csv'
    pairs_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / 'pilot_seed_pairs.csv'
    DATA.mkdir(exist_ok=True)

    corpus = []
    lookup = defaultdict(set)
    seen = set()
    stats = Counter()
    with raw_path.open('r', encoding='utf-8-sig', newline='') as f:
        for raw in csv.DictReader(f):
            stats['original_rows'] += 1
            comment = normalize_text(raw.get('Comment', ''))
            if not comment:
                stats['empty_comments'] += 1
                continue
            video_id = normalize_video_id(raw.get('Video ID', ''))
            if not video_id:
                stats['rows_without_valid_video_id'] += 1
            key = (video_id, comment)
            if key in seen:
                stats['same_video_text_duplicates'] += 1
                continue
            seen.add(key)
            doc_id = 'D' + hashlib.sha1((video_id + '\x00' + comment).encode('utf-8')).hexdigest()[:12].upper()
            try:
                likes = max(0, int(raw.get('Likes', 0)))
            except (TypeError, ValueError):
                likes = 0
            item = {
                'document_id': doc_id,
                'video_id': video_id,
                'comment_text': comment,
                'likes': likes,
                'youtube_video_url': ('https://www.youtube.com/watch?v=' + video_id) if video_id else '',
                'source_granularity': 'video_only_no_comment_id',
                'parent_id': '',
                'domain': 'unlabeled',
            }
            corpus.append(item)
            lookup[comment].add(doc_id)

    pairs = []
    mapped = 0
    queries = {}
    with pairs_path.open('r', encoding='utf-8-sig', newline='') as f:
        for i, raw in enumerate(csv.DictReader(f), 1):
            q = normalize_text(raw.get('Query', ''))
            comment = normalize_text(raw.get('Document', ''))
            if not q or not comment:
                continue
            qid = 'Q' + hashlib.sha1(q.encode('utf-8')).hexdigest()[:10].upper()
            queries[qid] = q
            ids = sorted(lookup.get(comment, set()))
            if ids:
                mapped += 1
            pairs.append({
                'seed_row': i,
                'query_id': qid,
                'singlish_query': q,
                'seed_document_text': comment,
                'matched_document_ids': ';'.join(ids),
                'source_match_status': 'exact_text' if ids else 'not_in_current_corpus',
                'label_status': 'unverified_seed_positive',
                'query_provenance': 'AI-assisted/manual review; row-level origin unknown',
            })

    corpus_fields = ['document_id','video_id','comment_text','likes','youtube_video_url','source_granularity','parent_id','domain']
    pair_fields = ['seed_row','query_id','singlish_query','seed_document_text','matched_document_ids','source_match_status','label_status','query_provenance']
    write_csv(DATA/'comments_clean.csv', corpus_fields, corpus)
    write_csv(DATA/'seed_pairs_linked.csv',pair_fields,pairs)

    stats.update({
        'clean_corpus_rows': len(corpus),
        'distinct_valid_videos': len({x['video_id'] for x in corpus if x['video_id']}),
        'seed_pairs': len(pairs),
        'seed_pairs_exact_text_matched': mapped,
        'seed_pairs_unmatched': len(pairs) - mapped,
        'unique_seed_query_strings': len(queries),
        'verified_relevance_labels': 0,
    })
    with (ROOT/'results'/'dataset_profile.json').open('w', encoding='utf-8') as f:
        json.dump(dict(stats),f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(stats),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
