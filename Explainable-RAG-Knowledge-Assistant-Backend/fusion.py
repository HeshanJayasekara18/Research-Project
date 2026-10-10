"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over views v in which d appears of 1 / (k + rank_v(d))

Only ranks are used, so raw scores from different methods (BM25 vs cosine) are
never compared on a common scale. Documents are deduplicated by their stable
document ID; a document missing from a view simply contributes nothing for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_K = 60


@dataclass
class FusedHit:
    doc_id: str
    rrf_score: float
    # view name -> (1-based rank in that view, method-specific raw score)
    views: dict[str, tuple[int, float]] = field(default_factory=dict)


def rrf(ranked_lists: dict[str, list[tuple[str, float]]], k: int = DEFAULT_K) -> list[FusedHit]:
    """Fuse ranked lists of (doc_id, raw_score), each best-first.

    Duplicate IDs inside one list keep only their best (first) rank. Ties in the
    fused score are broken by best single-view rank, then by doc ID, so output is
    deterministic.
    """
    if k < 0:
        raise ValueError('RRF rank constant k must be >= 0')
    fused: dict[str, FusedHit] = {}
    for view, hits in ranked_lists.items():
        seen = set()
        rank = 0
        for doc_id, score in hits:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            hit = fused.setdefault(doc_id, FusedHit(doc_id, 0.0))
            hit.rrf_score += 1.0 / (k + rank)
            hit.views[view] = (rank, float(score))
    return sorted(fused.values(),
                  key=lambda h: (-round(h.rrf_score, 12), min(r for r, _ in h.views.values()), h.doc_id))
