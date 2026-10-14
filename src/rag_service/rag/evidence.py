"""Initial evidence-selection strategy (heuristic, NOT yet evaluated).

Retrieval ranks are not relevance judgements, and BM25/cosine scores have no calibrated
cut-off, so we do not use score thresholds. Instead a retrieved candidate is eligible as
evidence only if it shares at least one *content term* with the question:

1. Query terms = tokens of the original query + tokens of its Sinhala-normalised form.
2. Generic terms are removed (GENERIC_TERMS: "video", "eka", "viewers", "monawada", ...),
   so comments that match only generic words (e.g. "Nice video") are never evidence.
3. Terms shorter than 3 characters are ignored.
4. A query term matches a comment token if their light Singlish phonetic keys are equal,
   or (both keys >= 5 chars) they are within edit distance 1, which tolerates spelling
   variants such as "wetakolu" / "vatakolu".
5. Eligible candidates are ordered by the number of distinct matched content terms (more
   shared topic words first), ties broken by retrieval rank; at most MAX_EVIDENCE are kept.

If nothing is eligible, the pipeline returns "insufficient_evidence" WITHOUT calling the LLM.

Known limitations: lexical overlap is a weak proxy for relevance (a shared content word
does not make a comment answer the question); purely semantic matches with no shared term
are excluded; the generic-term list and the edit-distance rule are hand-written and have
not been tuned or evaluated on labelled data. The LLM is a second gate: it is instructed
to answer "insufficient evidence" when the supplied comments do not support an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_service.retrieval.engine import tokenize

MAX_EVIDENCE = 6
MIN_TERM_LEN = 3

# Generic Singlish/Sinhala/English words that say nothing about a specific topic.
GENERIC_TERMS = frozenset({
    # English / Singlish generic content of comment questions
    'video', 'videos', 'viewers', 'viewer', 'comment', 'comments', 'channel', 'youtube', 'creator',
    'the', 'and', 'what', 'about', 'this', 'that', 'with', 'for', 'are', 'how', 'why', 'who',
    'eka', 'eke', 'ekak', 'ekata', 'ekath', 'eken', 'ekama', 'meka', 'mekata', 'mee', 'me',
    'la', 'lata', 'minissu', 'kattiya', 'aya', 'ayata', 'oya', 'oyage', 'oyata', 'api', 'mama', 'mage',
    'monawada', 'mokakda', 'mokadda', 'monawa', 'kohomada', 'kawuda', 'koheda', 'ai', 'aye',
    'kiyanne', 'kiyana', 'kiyala', 'kiyanawa', 'kiyanawada', 'kiyuwa', 'gana', 'gena', 'ganayi', 'genayi',
    'thiyenawada', 'tiyenawada', 'thiyenawa', 'tiyenawa', 'thiyena', 'thibba', 'nadda', 'neda', 'ne', 'na',
    'karana', 'karanna', 'karanawa', 'karanawada', 'karapu', 'kala', 'karala', 'hari', 'wage', 'wenuwen',
    'denawada', 'denawa', 'ahanawada', 'ahanawa', 'ahana', 'upades', 'prashna', 'balanna', 'baluwa',
    'hondai', 'hoda', 'nice', 'good', 'very', 'thanks', 'thank', 'saha', 'ehema', 'oni', 'ona', 'one',
    'nathi', 'natha', 'nathuwa', 'nisa', 'nam', 'namuth', 'wela', 'una', 'unu', 'tiyana', 'thiyana', 'puluwan', 'puluwanda',
    # Sinhala-script equivalents (as produced by normalisation)
    'වීඩියෝ', 'වීඩියෝව', 'එක', 'එකේ', 'එකක්', 'එකට', 'මේ', 'මෙක', 'ගැන', 'ගැනයි', 'මොනවද', 'මොකක්ද',
    'කියන්නේ', 'කියලා', 'කියන', 'කට්ටිය', 'මිනිස්සු', 'තියෙනවද', 'තියෙනවා', 'කරන', 'කරන්න', 'කරපු',
    'අහනවද', 'හරි', 'වගේ', 'සහ', 'ඔයා', 'ඔයාගෙ', 'අපි', 'මම', 'නෑ', 'නැහැ', 'නැති', 'නැත', 'නිසා', 'නම්', 'පුළුවන්',
})

_PHONETIC = [(re.compile(a), b) for a, b in [
    (r'w', 'v'), (r'(?<=[bcdgjkpstvzh])h', ''),  # aspiration markers: th/dh/sh/kh/hh -> t/d/s/k/h
    (r'aa', 'a'), (r'ee', 'i'), (r'ii', 'i'), (r'oo', 'u'), (r'uu', 'u'), (r'(.)\1', r'\1'),
]]


def phonetic_key(token: str) -> str:
    """Light Singlish spelling normaliser (Latin only; Sinhala script is returned unchanged)."""
    key = token.casefold()
    if not re.search(r'[a-z]', key):
        return key
    for pattern, repl in _PHONETIC:
        key = pattern.sub(repl, key)
    return key


def _within_one_edit(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True


def content_terms(*texts: str) -> set[str]:
    """Distinct non-generic tokens (>= MIN_TERM_LEN chars) of the given texts."""
    terms = set()
    for text in texts:
        for tok in tokenize(text or ''):
            if len(tok) >= MIN_TERM_LEN and tok not in GENERIC_TERMS and phonetic_key(tok) not in GENERIC_TERMS:
                terms.add(tok)
    return terms


def matched_terms(query_terms: set[str], comment_text: str) -> list[str]:
    """Query content terms that occur (allowing small spelling variation) in the comment."""
    comment_keys = {phonetic_key(t) for t in tokenize(comment_text) if len(t) >= MIN_TERM_LEN}
    hits = []
    for term in sorted(query_terms):
        k = phonetic_key(term)
        if k in comment_keys or (len(k) >= 5 and any(len(c) >= 5 and _within_one_edit(k, c) for c in comment_keys)):
            hits.append(term)
    return hits


@dataclass
class Candidate:
    document_id: str
    video_id: str
    text: str
    video_url: str | None
    retrieval_rank: int


@dataclass
class Selection:
    selected: list[tuple[Candidate, list[str]]] = field(default_factory=list)  # (candidate, matched terms)
    rejected: list[Candidate] = field(default_factory=list)                   # matched only generic terms / nothing
    query_terms: list[str] = field(default_factory=list)


def select_evidence(query: str, normalized_query: str | None, candidates: list[Candidate],
                    max_evidence: int = MAX_EVIDENCE) -> Selection:
    terms = content_terms(query, normalized_query or '')
    sel = Selection(query_terms=sorted(terms))
    eligible = []
    for cand in candidates:
        hits = matched_terms(terms, cand.text) if terms else []
        (eligible.append((cand, hits)) if hits else sel.rejected.append(cand))
    eligible.sort(key=lambda ch: (-len(ch[1]), ch[0].retrieval_rank))
    sel.selected = eligible[:max_evidence]
    return sel
