"""Adapter between the HTTP contract and the existing, tested RetrievalEngine.

No ranking logic lives here: it only maps API identifiers to engine methods,
enforces scope rules, and shapes engine output into response models.
"""
from __future__ import annotations

import threading

from rag_service import API_VERSION, SERVICE_VERSION
from rag_service.api.errors import ApiError
from rag_service.api.schemas import (HealthResponse, NormalizationInfo, RetrievalMethod, RetrievedComment,
                                     RetrieveRequest, RetrieveResponse, ScopeInfo, SemanticStatus, ViewContribution)
from rag_service.core.config import Settings
from rag_service.retrieval import engine as retrieval
from rag_service.retrieval.engine import (DENSE_METHODS, MULTIVIEW_METHODS, MV_BM25_METHOD, MV_HYBRID_METHOD,
                                          MV_LABSE_METHOD, NORM_BM25_METHOD, NORM_LABSE_METHOD, NORMALIZED_METHODS,
                                          SEMANTIC_METHOD, RetrievalEngine)

ENGINE_METHODS = {
    RetrievalMethod.bm25: 'BM25 (lexical)',
    RetrievalMethod.char_tfidf: 'Character TF-IDF',
    RetrievalMethod.labse: SEMANTIC_METHOD,
    RetrievalMethod.normalized_bm25: NORM_BM25_METHOD,
    RetrievalMethod.normalized_labse: NORM_LABSE_METHOD,
    RetrievalMethod.multiview_bm25: MV_BM25_METHOD,
    RetrievalMethod.multiview_labse: MV_LABSE_METHOD,
    RetrievalMethod.multiview_hybrid: MV_HYBRID_METHOD,
}
_BASE_SCORE_TYPE = {'BM25 (lexical)': 'bm25', 'Character TF-IDF': 'char_tfidf_cosine', SEMANTIC_METHOD: 'labse_cosine'}

NOTICE = ('Retrieved candidate comments ranked by similarity, not human-verified evidence. '
          'This service does not generate answers or verify claims.')


def _needs_semantic(engine_method: str) -> bool:
    if engine_method in DENSE_METHODS:
        return True
    return SEMANTIC_METHOD in MULTIVIEW_METHODS.get(engine_method, ())


def _likes(doc) -> int | None:
    try:
        return int(doc.get('likes', ''))
    except (TypeError, ValueError):
        return None


class RetrievalService:
    def __init__(self, engine: RetrievalEngine, settings: Settings):
        self.engine = engine
        self.settings = settings
        # The engine lazily builds caches (scoped BM25, LaBSE, normaliser); serialise access.
        self._lock = threading.Lock()

    def health(self) -> HealthResponse:
        index_present = self.engine.has_semantic_index()
        return HealthResponse(
            status='ok', service='singlish-retrieval', version=SERVICE_VERSION, api_version=API_VERSION,
            engine_ready=True, corpus_documents=len(self.engine.docs), videos=len(self.engine.video_ids()),
            semantic=SemanticStatus(index_present=index_present, loaded=self.engine._semantic is not None),
            auth_enabled=self.settings.auth_enabled)

    def _scope(self, video_id: str | None) -> ScopeInfo:
        if video_id is None:
            if not self.settings.allow_global_scope:
                raise ApiError(403, 'global_scope_disabled', 'A scope.video_id is required on this deployment.')
            return ScopeInfo(type='global', video_id=None, documents_searched=len(self.engine.docs))
        if video_id not in self.engine.video_rows:
            raise ApiError(404, 'video_not_found', f'No indexed comments for video {video_id!r}.')
        return ScopeInfo(type='video', video_id=video_id, documents_searched=len(self.engine.video_rows[video_id]))

    def retrieve(self, req: RetrieveRequest, request_id: str) -> RetrieveResponse:
        method = ENGINE_METHODS[req.method]
        scope = self._scope(req.scope.video_id)
        if _needs_semantic(method) and not self.engine.has_semantic_index():
            raise ApiError(503, 'method_unavailable',
                           f'Method {req.method.value!r} needs the LaBSE index, which is not built on this server.')
        try:
            with self._lock:
                results, normalization = self._run(req, method, scope.video_id)
        except RuntimeError as exc:  # e.g. LaBSE model could not be loaded
            raise ApiError(503, 'method_unavailable', str(exc)) from exc
        return RetrieveResponse(
            request_id=request_id, query=req.query, method=req.method, top_k=req.top_k, scope=scope,
            normalization=normalization, results=results, result_count=len(results), notice=NOTICE)

    def _run(self, req: RetrieveRequest, method: str, video_id: str | None):
        e = self.engine
        if method in MULTIVIEW_METHODS:
            hits, info = e.multiview(req.query, method, req.top_k, video_id=video_id)
            norm = info['normalization']
            results = [self._comment(i, doc, round(h.rrf_score, 6), 'rrf',
                                     [ViewContribution(view=v, rank=r, score=round(s, 6)) for v, (r, s) in h.views.items()])
                       for i, (doc, h) in enumerate(hits, 1)]
            return results, self._norm_info(norm, info['normalized_view_used'])

        normalization = None
        if method in NORMALIZED_METHODS:
            norm = e.normalize(req.query)
            normalization = self._norm_info(norm, bool(norm.normalized))
            score_type = _BASE_SCORE_TYPE[NORMALIZED_METHODS[method]]
        else:
            score_type = _BASE_SCORE_TYPE[method]
        rows = e.search(req.query, method, req.top_k, video_id)
        return [self._comment(i, doc, score, score_type) for i, (doc, score) in enumerate(rows, 1)], normalization

    @staticmethod
    def _norm_info(norm, used: bool) -> NormalizationInfo:
        return NormalizationInfo(status=norm.status, normalized_query=norm.normalized or '',
                                 coverage=round(float(norm.coverage), 4), used=used)

    @staticmethod
    def _comment(rank, doc, score, score_type, views=None) -> RetrievedComment:
        return RetrievedComment(
            rank=rank, document_id=doc['document_id'], video_id=doc['video_id'], text=doc['comment_text'],
            likes=_likes(doc), video_url=doc.get('youtube_video_url') or None,
            score=float(score), score_type=score_type, views=views)


def build_engine() -> RetrievalEngine:
    return retrieval.RetrievalEngine()
