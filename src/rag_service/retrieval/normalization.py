"""Experimental Singlish (Romanised Sinhala) -> Sinhala-script query normalisation.

Approach "lexicon-rule-v1" (offline, deterministic, reproducible):

1. Candidate generation: each Latin word is split into phonetic units (consonant
   clusters / vowels) using a general Romanised-Sinhala scheme. Every unit maps to
   ALL its plausible Sinhala spellings (e.g. t -> ට/ත, d -> ද/ඩ, n -> න/ණ, short
   or long vowels, prenasals nd -> ඳ). No word-specific entries are used.
2. Lexicon-guided decoding: a beam search keeps only candidates that are
   prefixes of words in a Sinhala lexicon (by default, the Sinhala-script
   vocabulary of the indexed corpus; no relevance labels are used). The most
   frequent complete lexicon word wins.
3. Abstention: if no candidate is a real lexicon word the Latin word is KEPT and
   flagged 'unresolved' (never silently guessed). Close frequency ties are
   flagged 'ambiguous'. Numbers are kept. Words that occur in Latin script inside
   Sinhala-script comments more often than their best Sinhala reading (English
   loanwords such as 'Ad', 'kg') are kept as English.

Limitations (by design, report these): it can only produce words already present
in the lexicon, so unseen inflections stay unresolved; frequency is a crude
disambiguator; it is NOT a general Sinhala transliteration model. The interface
(`Normalizer.normalize`) is kept small so other normalisers can be swapped in.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

SINHALA_WORD_RE = re.compile(r'[\u0D80-\u0DFF\u200d]+')
LATIN_WORD_RE = re.compile(r'[A-Za-z]+')
WORD_RE = re.compile(r'[0-9]+(?:[.,][0-9]+)?[A-Za-z]*|[A-Za-z]+|[\u0D80-\u0DFF\u200d]+')
HAL = '්'  # virama (al-lakuna)
ZWJ = '‍'

# Consonant units: Latin key -> plausible Sinhala renderings (longest keys matched first).
CONSONANTS = {
    'chh': ['ඡ'], 'ndh': ['ඳ', 'න්ද'], 'ksh': ['ක්ෂ'],
    'kh': ['ඛ', 'ක'], 'gh': ['ඝ', 'ග'], 'ch': ['ච', 'ඡ'], 'jh': ['ඣ', 'ජ'],
    'th': ['ත', 'ථ', 'ට'], 'dh': ['ද', 'ධ', 'ඩ'], 'ph': ['ඵ', 'ෆ'], 'bh': ['භ', 'බ'],
    'sh': ['ශ', 'ෂ'], 'nd': ['ඳ', 'න්ද', 'ණ්ඩ', 'න්ඩ'], 'mb': ['ඹ', 'ම්බ'],
    'ng': ['ඟ', 'ංග', 'න්ග'], 'gn': ['ඥ', 'ග්න'], 'kn': ['ඤ', 'ක්න'],
    'k': ['ක'], 'g': ['ග'], 'j': ['ජ'], 't': ['ට', 'ත'], 'd': ['ද', 'ඩ'],
    'n': ['න', 'ණ'], 'p': ['ප'], 'b': ['බ'], 'm': ['ම'], 'y': ['ය'], 'r': ['ර'],
    'l': ['ල', 'ළ'], 'w': ['ව'], 'v': ['ව'], 's': ['ස', 'ශ', 'ෂ'], 'h': ['හ'],
    'f': ['ෆ'], 'c': ['ක', 'ච'], 'q': ['ක'], 'x': ['ක්ස'], 'z': ['ස'],
}
# Vowel units: Latin key -> (independent forms, dependent signs after a consonant).
VOWELS = {
    'aae': (['ඈ'], ['ෑ']), 'aa': (['ආ'], ['ා']), 'ae': (['ඇ', 'ඈ'], ['ැ', 'ෑ']),
    'ai': (['අයි', 'ඓ'], ['යි', 'ෛ']), 'au': (['ඖ', 'අවු'], ['ෞ', 'වු']),
    'ii': (['ඊ'], ['ී']), 'ee': (['ඊ', 'ඒ'], ['ී', 'ේ']), 'uu': (['ඌ'], ['ූ']),
    'oo': (['ඕ', 'ඌ'], ['ෝ', 'ූ']), 'ei': (['එයි'], ['ෙයි']),
    'a': (['අ', 'ආ'], ['', 'ා', 'ැ']), 'i': (['ඉ', 'ඊ'], ['ි', 'ී']),
    'u': (['උ', 'ඌ'], ['ු', 'ූ']), 'e': (['එ', 'ඇ', 'ඒ'], ['ෙ', 'ැ', 'ේ']),
    'o': (['ඔ', 'ඕ'], ['ො', 'ෝ']),
}
_KEYS = sorted([(k, 'C') for k in CONSONANTS] + [(k, 'V') for k in VOWELS], key=lambda x: -len(x[0]))


def split_units(word: str) -> list[tuple[str, str]] | None:
    """Greedy longest-match split into ('C'|'V', key) units; None if unsplittable."""
    units, i, w = [], 0, word.lower()
    while i < len(w):
        for key, kind in _KEYS:
            if w.startswith(key, i):
                units.append((kind, key))
                i += len(key)
                break
        else:
            return None
        # A doubled consonant ('nn', 'tt') is two units: the first takes al-lakuna.
    return units


def _expansions(units):
    """Yield, per unit, the list of string pieces it can contribute."""
    out = []
    for idx, (kind, key) in enumerate(units):
        prev = units[idx - 1] if idx else None
        nxt = units[idx + 1] if idx + 1 < len(units) else None
        if kind == 'C':
            forms = list(CONSONANTS[key])
            if nxt is None and key in ('n', 'ng'):
                forms.append('ං')          # word-final anusvara
                forms = [f + HAL if f != 'ං' else f for f in forms]
            elif nxt is None or nxt[0] == 'C':
                forms = [f + HAL for f in forms]
                if nxt is None:            # spoken final vowel often dropped/added
                    forms += [f[:-1] for f in forms]
            if prev and prev[0] == 'C' and key in ('r', 'y'):
                # rakaransaya / yansaya: previous consonant + ZWJ form
                forms = forms + [ZWJ + f for f in forms]
            out.append(forms)
        else:
            indep, signs = VOWELS[key]
            if prev and prev[0] == 'C':
                out.append(list(signs))
            elif prev and prev[0] == 'V' and key == 'i':
                out.append(['යි', 'ඉ'])
            else:
                out.append(list(indep))
    return out


@dataclass
class TokenResult:
    source: str
    output: str
    status: str  # transliterated | ambiguous | kept_english | kept_number | kept_sinhala | unresolved
    alternatives: list[str] = field(default_factory=list)
    lexicon_freq: int = 0


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    status: str  # normalized | partial | failed | unchanged | empty
    method: str
    tokens: list[TokenResult]

    @property
    def coverage(self) -> float:
        latin = [t for t in self.tokens if t.status not in ('kept_number', 'kept_sinhala')]
        done = [t for t in latin if t.status in ('transliterated', 'ambiguous')]
        return len(done) / len(latin) if latin else 0.0

    def summary(self) -> str:
        c = Counter(t.status for t in self.tokens)
        return f"{self.status} (coverage {self.coverage:.0%}; " + ', '.join(f'{k}={v}' for k, v in sorted(c.items())) + ')'


class Normalizer:
    """Interface: replace with another implementation to compare approaches."""
    name = 'identity'

    def normalize(self, query: str) -> NormalizationResult:
        toks = [TokenResult(w, w, 'unresolved') for w in WORD_RE.findall(query)]
        return NormalizationResult(query, query, 'unchanged', self.name, toks)


class LexiconRuleNormalizer(Normalizer):
    name = 'lexicon-rule-v1'

    def __init__(self, sinhala_texts, beam_width: int = 64, ambiguity_ratio: float = 2.0,
                 min_english_count: int = 2):
        self.lexicon = Counter()
        self.latin_in_sinhala = Counter()
        for text in sinhala_texts:
            sin = SINHALA_WORD_RE.findall(text)
            self.lexicon.update(w.strip(ZWJ) for w in sin)
            if sum(map(len, sin)) > len(''.join(LATIN_WORD_RE.findall(text))):
                self.latin_in_sinhala.update(w.lower() for w in LATIN_WORD_RE.findall(text))
        self.prefixes = {w[:i] for w in self.lexicon for i in range(1, len(w) + 1)}
        self.beam_width, self.ambiguity_ratio, self.min_english_count = beam_width, ambiguity_ratio, min_english_count

    def candidates(self, word: str) -> list[tuple[str, int]]:
        """Lexicon words reachable from `word`, most frequent first."""
        units = split_units(word)
        if not units:
            return []
        beam = ['']
        for pieces in _expansions(units):
            nxt = {b + p for b in beam for p in pieces}
            beam = sorted((c for c in nxt if c in self.prefixes),
                          key=lambda c: -self.lexicon.get(c, 0))[:self.beam_width]
            if not beam:
                return []
        found = {c: self.lexicon[c] for c in beam if c in self.lexicon}
        return sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))

    def normalize_word(self, word: str) -> TokenResult:
        if word[0].isdigit():
            return TokenResult(word, word, 'kept_number')
        if SINHALA_WORD_RE.fullmatch(word):
            return TokenResult(word, word, 'kept_sinhala')
        cands = self.candidates(word)
        best_freq = cands[0][1] if cands else 0
        eng = self.latin_in_sinhala.get(word.lower(), 0)
        if eng >= self.min_english_count and eng > best_freq:
            return TokenResult(word, word, 'kept_english', [c for c, _ in cands[:3]], best_freq)
        if not cands:
            return TokenResult(word, word, 'unresolved')
        alts = [c for c, _ in cands[:4]]
        ambiguous = len(cands) > 1 and cands[0][1] < self.ambiguity_ratio * cands[1][1]
        return TokenResult(word, cands[0][0], 'ambiguous' if ambiguous else 'transliterated', alts, best_freq)

    def normalize(self, query: str) -> NormalizationResult:
        query = str(query)
        words = WORD_RE.findall(query)
        if not words:
            return NormalizationResult(query, '', 'empty', self.name, [])
        tokens = [self.normalize_word(w) for w in words]
        result = NormalizationResult(query, ' '.join(t.output for t in tokens), '', self.name, tokens)
        cov = result.coverage
        result.status = 'failed' if cov == 0 else 'normalized' if cov >= 0.75 else 'partial'
        return result


def from_corpus(docs) -> LexiconRuleNormalizer:
    return LexiconRuleNormalizer(d['comment_text'] for d in docs)


if __name__ == '__main__':
    import sys
    from retrieval import load_corpus
    norm = from_corpus(load_corpus())
    for q in sys.argv[1:] or ['Kaha bath ekata kanna galapena curries monawada?']:
        r = norm.normalize(q)
        print(q, '->', r.normalized, '|', r.summary())
        for t in r.tokens:
            print(f'   {t.source:>16} -> {t.output:<14} {t.status:<14} {t.alternatives}')
