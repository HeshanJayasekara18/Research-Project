"""Transparent baseline IR for Sri Lankan YouTube comments.

Methods: BM25 lexical, character n-gram TF-IDF, and optional pretrained LaBSE.
None is claimed to outperform an external baseline before an independent test.
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_service.core.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

ROOT = PROJECT_ROOT
DATA = DATA_DIR
RESULTS = RESULTS_DIR
LABSE_INDEX = DATA / 'labse_index.npz'
# Override with env var SINGLISH_LABSE_MODEL (e.g. a local folder path) if needed.
LABSE_MODEL = os.environ.get('SINGLISH_LABSE_MODEL', 'sentence-transformers/LaBSE')
SEMANTIC_METHOD = 'LaBSE (semantic)'
NORM_BM25_METHOD = 'Normalized query -> BM25 (experimental)'
NORM_LABSE_METHOD = 'Normalized query -> LaBSE (experimental)'
# Normalised-query methods: (base method run on the Sinhala-normalised query).
NORMALIZED_METHODS = {NORM_BM25_METHOD: 'BM25 (lexical)', NORM_LABSE_METHOD: SEMANTIC_METHOD}
DENSE_METHODS = {SEMANTIC_METHOD, NORM_LABSE_METHOD}
# Multi-view RRF methods: base methods each run on the original AND normalised query.
MV_BM25_METHOD = 'Multi-view Retrieval (Experimental): BM25 original + normalized'
MV_LABSE_METHOD = 'Multi-view Retrieval (Experimental): LaBSE original + normalized'
MV_HYBRID_METHOD = 'Multi-view Retrieval (Experimental): hybrid BM25 + LaBSE, original + normalized'
MULTIVIEW_METHODS = {MV_BM25_METHOD: ('BM25 (lexical)',), MV_LABSE_METHOD: (SEMANTIC_METHOD,),
                     MV_HYBRID_METHOD: ('BM25 (lexical)', SEMANTIC_METHOD)}
MV_DEPTH = 50  # candidates per view before fusion
TOKEN_RE = re.compile(r'[A-Za-z0-9_\u0D80-\u0DFF]+')


def load_labse_model(name: str = LABSE_MODEL):
    """Load LaBSE on CPU, preferring the local HF cache; clear error if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('sentence-transformers is not installed. '
                           'Run: pip install -e ".[semantic]"') from exc
    try:  # offline first: avoids network calls once weights are cached
        return SentenceTransformer(name, device='cpu', local_files_only=True)
    except Exception:
        pass
    try:
        return SentenceTransformer(name, device='cpu')
    except Exception as exc:
        raise RuntimeError(f'Could not load model {name!r} (not cached and download failed: '
                           f'{type(exc).__name__}). Check the internet connection or set '
                           'SINGLISH_LABSE_MODEL to a local model folder.') from exc


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


@lru_cache(maxsize=1)
def load_corpus():
    with (DATA / 'comments_clean.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


class BM25:
    def __init__(self, texts):
        self.n = len(texts)
        self.docs_len = np.zeros(self.n, dtype=np.int32)
        self.postings = defaultdict(list)
        for i, text in enumerate(texts):
            words = tokenize(text)
            self.docs_len[i] = len(words)
            for word, freq in Counter(words).items():
                self.postings[word].append((i, freq))
        self.avgdl = max(1., float(np.mean(self.docs_len)))

    def scores(self, query: str):
        k1, b = 1.5, 0.75
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1. + (self.n-df+0.5)/(df+0.5))
            for doc_i, tf in postings:
                norm = k1 * (1. - b + b * self.docs_len[doc_i] / self.avgdl)
                scores[doc_i] += idf * tf * (k1+1.) / (tf + norm)
        return scores


DOMAIN_SCOPE_PREFIX = 'domain:'  # scope keys: a video ID, or "domain:<name>" (video IDs never contain ':')
_DEFAULT_INDEX = object()


class RetrievalEngine:
    def __init__(self, docs: list[dict] | None = None, labse_index=_DEFAULT_INDEX, corpus_id: str = 'v1'):
        """docs=None loads Corpus V1 (data/comments_clean.csv). Other corpora pass their documents
        (each with document_id, video_id, comment_text, ...) and their own LaBSE index path, or None
        when no dense index exists. Retrieval algorithms are identical for every corpus."""
        self.corpus_id = corpus_id
        self.labse_index = labse_index
        self.docs = load_corpus() if docs is None else docs
        self.ids = [d['document_id'] for d in self.docs]
        self.by_id = {d['document_id']: d for d in self.docs}
        self.texts = [d['comment_text'] for d in self.docs]
        self.bm25 = BM25(self.texts)
        self.vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(2, 4),
            min_df=2, max_features=100000, sublinear_tf=True, dtype=np.float32
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self._semantic = None
        self._semantic_model = None
        self._semantic_model_name = None
        self._normalizer = None
        # Video scoping: row indices of each video's comments (all scripts kept).
        rows = defaultdict(list)
        for i, d in enumerate(self.docs):
            rows[d['video_id']].append(i)
        self.video_rows = dict(rows)  # plain dict: an unknown ID must never become an empty scope
        # Domain scoping (Corpus V2): row indices per domain, used exactly like a video scope.
        drows = defaultdict(list)
        for i, d in enumerate(self.docs):
            if d.get('domain'):
                drows[d['domain']].append(i)
        self.domain_rows = dict(drows)
        self._video_bm25 = {}  # scoped BM25 cache, keyed by scope key (video ID or "domain:<name>")

    def _rows_for(self, scope_key):
        if scope_key.startswith(DOMAIN_SCOPE_PREFIX):
            domain = scope_key[len(DOMAIN_SCOPE_PREFIX):]
            if domain not in self.domain_rows:
                raise ValueError(f'Unknown domain: {domain!r}')
            return self.domain_rows[domain]
        if scope_key not in self.video_rows:
            raise ValueError(f'Unknown video ID: {scope_key!r}')
        return self.video_rows[scope_key]

    def video_ids(self):
        """Videos with at least one comment, most comments first."""
        return sorted((v for v in self.video_rows if v), key=lambda v: (-len(self.video_rows[v]), v))

    def scope_rows(self, video_id):
        """None -> global corpus. Otherwise the row indices of a video, or of "domain:<name>"."""
        if not video_id:
            return None
        return np.asarray(self._rows_for(video_id), dtype=np.int64)

    def _scoped_bm25(self, video_id):
        # BM25 statistics (IDF, average length) computed within the video only.
        if video_id not in self._video_bm25:
            self._video_bm25[video_id] = BM25([self.texts[i] for i in self._rows_for(video_id)])
        return self._video_bm25[video_id]

    @property
    def normalizer(self):
        """Lazily built Singlish->Sinhala normaliser (lexicon = this corpus)."""
        if self._normalizer is None:
            from rag_service.retrieval.normalization import from_corpus
            self._normalizer = from_corpus(self.docs)
        return self._normalizer

    def normalize(self, query: str):
        return self.normalizer.normalize(query)

    def _labse_path(self):
        # Corpus V1 uses the module-level LABSE_INDEX (looked up at call time, so tests can patch it).
        path = getattr(self, 'labse_index', _DEFAULT_INDEX)
        return LABSE_INDEX if path is _DEFAULT_INDEX else path

    def has_semantic_index(self):
        path = self._labse_path()
        return path is not None and Path(path).exists()

    def semantic_status(self) -> tuple[bool, str]:
        """(ready, human-readable message) without raising; safe for the UI."""
        try:
            self._load_semantic()
            return True, f'LaBSE index loaded ({len(self._semantic):,} docs, model {self._semantic_model_name}).'
        except Exception as exc:  # report, never crash the app
            return False, str(exc)

    def _load_semantic(self):
        if self._semantic is not None:
            return
        path = self._labse_path()
        if path is None or not Path(path).exists():
            raise RuntimeError('LaBSE index not built. Run: python scripts/build_labse_index.py')
        archive = np.load(path, allow_pickle=False)
        ids = archive['document_ids'].astype(str).tolist()
        if ids != self.ids:
            raise RuntimeError('LaBSE index document IDs differ from the active corpus '
                               '(corpus changed?). Rebuild: python scripts/build_labse_index.py')
        model_name = str(archive['model_name']) if 'model_name' in archive.files else LABSE_MODEL
        emb = np.asarray(archive['embeddings'], dtype=np.float32)
        # Re-normalise after float16 storage so dot product == cosine similarity.
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        model = load_labse_model(model_name)
        self._semantic, self._semantic_model, self._semantic_model_name = emb, model, model_name

    def scores(self, query: str, method: str, video_id=None):
        """Scores aligned with the searchable rows: the whole corpus, or only the
        selected video's rows (filtering happens BEFORE any ranking)."""
        rows = self.scope_rows(video_id)
        if method in NORMALIZED_METHODS:
            # Unresolved words stay in Latin script inside the normalised string.
            return self.scores(self.normalize(query).normalized or query, NORMALIZED_METHODS[method], video_id)
        if method == 'BM25 (lexical)':
            return self.bm25.scores(query) if rows is None else self._scoped_bm25(video_id).scores(query)
        if method == 'Character TF-IDF':
            matrix = self.matrix if rows is None else self.matrix[rows]
            transformed = self.vectorizer.transform([query])
            return np.asarray((matrix @ transformed.T).toarray()).reshape(-1)
        if method == SEMANTIC_METHOD:
            self._load_semantic()
            v = self._semantic_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            emb = self._semantic if rows is None else self._semantic[rows]
            return emb @ np.asarray(v, dtype=np.float32)
        raise ValueError(f'Unknown retrieval method: {method}')

    def multiview(self, query: str, method: str, top_k=10, depth=MV_DEPTH, k=None, video_id=None):
        """RRF over views {base} x {original, normalised}.

        Returns (hits, info): hits = [(doc, FusedHit)], info describes which views
        ran. If normalisation failed or changed nothing, ONLY original-query views
        are used (no second representation is fabricated).
        """
        from rag_service.retrieval.fusion import DEFAULT_K, rrf
        k = DEFAULT_K if k is None else k
        norm = self.normalize(query)
        use_norm = norm.status in ('normalized', 'partial') and norm.normalized != query.strip()
        lists = {}
        for base in MULTIVIEW_METHODS[method]:
            short = 'BM25' if base.startswith('BM25') else 'LaBSE'
            lists[f'{short}/original'] = [(d['document_id'], s) for d, s in self.search(query, base, depth, video_id)]
            if use_norm:
                lists[f'{short}/normalized'] = [(d['document_id'], s)
                                                for d, s in self.search(norm.normalized, base, depth, video_id)]
        fused = rrf(lists, k=k)[:max(1, int(top_k))]
        info = {'normalization': norm, 'normalized_view_used': use_norm, 'views': list(lists), 'k': k, 'depth': depth,
                'video_id': video_id or None}
        return [(self.by_id[h.doc_id], h) for h in fused], info

    def search(self, query: str, method='BM25 (lexical)', top_k=10, video_id=None):
        """Top-k (doc, score). video_id=None searches the global corpus (research
        mode); otherwise ONLY that video's comments are scored and ranked."""
        if not str(query).strip():
            return []
        if method in MULTIVIEW_METHODS:
            hits, _ = self.multiview(str(query), method, top_k, video_id=video_id)
            return [(d, round(h.rrf_score, 6)) for d, h in hits]
        rows = self.scope_rows(video_id)
        docs = self.docs if rows is None else [self.docs[i] for i in rows]
        scores = self.scores(query, method, video_id)
        n = min(max(1, int(top_k)), len(docs))
        if n >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            top = np.argpartition(scores, -n)[-n:]
            order = top[np.argsort(scores[top])[::-1]]
        results = []
        for idx in order:
            value = float(scores[idx])
            # Lexical and character search: all-zero results do not indicate matches.
            if method not in DENSE_METHODS and value <= 0:
                continue
            results.append((docs[int(idx)], round(value, 4)))
        return results
