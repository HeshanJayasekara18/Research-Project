# Demo case notes (qualitative, single cases)

These notes document individual observations from the PP1 demo. They are **not**
evaluation results, and a manual reading of one case does **not** demonstrate a
general improvement or degradation of any method. The research corpus
(`data/comments_clean.csv`) and the normaliser are unchanged by these notes.

## Case W1 — "dara" in the Wetakolu video (`-3a4abtQH3E`)

**Question (Singlish):**
`Me wetakolu video eke poththa saha dara ain karana eka gana viewers la monawada kiyanne?`

**Comment (original text, preserved exactly):**
`Api uyanakota vatakolu vala dara ain karanava.hethuva nam danne na`
(document `DB8B1DE7D2790`)

**Contextual reading (user / native-speaker interpretation):**
In the context of preparing *wetakolu* (ridge gourd), `dara` most probably means
**දාර** — the *ridges/edges* of the vegetable that are peeled off — i.e.
"When we cook it we remove the ridges of the wetakolu; I don't know the reason."
It does **not** mean "seeds", and it does **not** mean **දර** ("firewood").

> Correction: the Milestone 4 (video-scoped) report glossed `dara` as "seeds".
> That gloss was wrong; the reading above supersedes it.

**What the system actually did (observed, lexicon-rule-v1):**

| Input | Output | Status reported | Alternatives considered |
|---|---|---|---|
| `dara` | දර (firewood) | `transliterated` (not flagged) | දර, දරා, දරැ |

Why: the normaliser can only output words that already occur in the corpus
lexicon. **දාර does not occur anywhere in the 10,343-comment corpus** (frequency
0), so it can never be produced; දර occurs twice and won. The error was also not
flagged as ambiguous because the competing candidates were rarer.

**Why this matters (limitations, stated honestly):**
1. *Contextual ambiguity*: Romanised `dara` can map to several Sinhala words
   (දාර ridges/edges, දර firewood, ...). Choosing correctly needs context
   (here: a vegetable-preparation question), which lexicon-rule-v1 does not use.
2. *Lexicon coverage*: a correct form missing from the corpus can never be
   generated, so "not flagged" does not mean "correct".
3. In this video both relevant comments are themselves written in Singlish, so
   plain BM25 on the original query retrieves them (ranks 1–2) without any
   normalisation; normalisation only adds noise for this question.

**Not concluded from this case:** no claim that a context-aware normaliser or a
manual `dara → දාර` mapping would improve retrieval in general. Testing that
would need a larger, human-annotated set of ambiguous words and queries.

## Case W2 — Generic praise is not evidence

For the same question, ranks 3–10 under BM25 include comments such as
"Nice video ❤", "Very useful video" and "Yes very good video". They are
retrieved only because the question contains the words *video* / *viewers*.
They are **retrieved candidates, not evidence** about Wetakolu preparation.
The app's *Matched query words* column makes this visible (e.g. only `video`).
