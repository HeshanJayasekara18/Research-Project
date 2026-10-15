"""Corpus V2 loader: the privacy-safe export produced by the collection repo
(`python -m collection.export_research`), i.e. a folder containing comments_v2.csv
(+ videos_v2.csv, export_manifest.json, optional labse_index_v2.npz).

Corpus V1 is untouched; V2 is a separate corpus with real YouTube comment IDs, thread
structure and domains. Only rows with indexable == "true" become retrieval documents;
all rows (including withheld / emoji-only ones) are kept in `thread_records` so thread
relationships survive for future thread-aware RAG.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from rag_service.retrieval.engine import RetrievalEngine

REQUIRED_COLUMNS = ('comment_id', 'video_id', 'domain', 'parent_id', 'thread_id', 'is_reply', 'indexable', 'comment_text')
DOMAINS = ('food', 'gaming', 'education')
INDEX_NAME = 'labse_index_v2.npz'


class CorpusV2Error(RuntimeError):
    pass


def _truthy(v: str) -> bool:
    return (v or '').strip().lower() in ('true', '1', 'yes')


def load_records(corpus_dir: Path) -> list[dict]:
    path = Path(corpus_dir) / 'comments_v2.csv'
    if not path.exists():
        raise CorpusV2Error(f'Corpus V2 not found: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CorpusV2Error(f'comments_v2.csv is missing columns: {missing}')
        rows = list(reader)
    ids = Counter(r['comment_id'] for r in rows)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise CorpusV2Error(f'{len(dupes)} duplicated comment_id values in comments_v2.csv')
    bad_domain = {r['domain'] for r in rows} - set(DOMAINS)
    if bad_domain:
        raise CorpusV2Error(f'unknown domains in comments_v2.csv: {sorted(bad_domain)}')
    return rows


def thread_integrity(records: list[dict]) -> dict:
    by_id = {r['comment_id']: r for r in records}
    replies = [r for r in records if _truthy(r['is_reply'])]
    return {
        'records': len(records), 'replies': len(replies),
        'replies_with_missing_parent': sum(r['parent_id'] not in by_id for r in replies),
        'replies_parent_on_other_video': sum(r['parent_id'] in by_id and by_id[r['parent_id']]['video_id'] != r['video_id']
                                             for r in replies),
        'top_level_with_parent_id': sum(bool(r['parent_id'].strip()) for r in records if not _truthy(r['is_reply'])),
    }


def to_documents(records: list[dict]) -> list[dict]:
    """Indexable records -> engine documents (document_id = real YouTube comment_id)."""
    docs = []
    for r in records:
        if not _truthy(r['indexable']):
            continue
        docs.append({
            'document_id': r['comment_id'], 'comment_id': r['comment_id'], 'video_id': r['video_id'],
            'domain': r['domain'], 'comment_text': r['comment_text'], 'likes': r.get('likes', ''),
            'youtube_video_url': r.get('youtube_video_url', ''), 'parent_id': r['parent_id'],
            'thread_id': r['thread_id'], 'is_reply': _truthy(r['is_reply']),
        })
    return docs


def build_engine_v2(corpus_dir: Path) -> RetrievalEngine:
    records = load_records(corpus_dir)
    docs = to_documents(records)
    if not docs:
        raise CorpusV2Error('Corpus V2 has no indexable comments.')
    index = Path(corpus_dir) / INDEX_NAME
    engine = RetrievalEngine(docs=docs, labse_index=index if index.exists() else None, corpus_id='v2')
    engine.thread_records = {r['comment_id']: r for r in records}  # all records, incl. non-indexable
    engine.thread_report = thread_integrity(records)
    return engine
