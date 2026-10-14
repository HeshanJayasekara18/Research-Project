# M6: Minimal evidence-grounded Sinhala RAG (`POST /v1/ask`)

Status: **implemented and tested with a mock provider; a live end-to-end answer has not yet been demonstrated** (see "Live demonstration status").

## Retrieval vs RAG generation

| | `/v1/retrieve` | `/v1/ask` |
|---|---|---|
| Output | Ranked candidate comments | A short Sinhala answer + the exact comments it cites |
| Uses an LLM | No | Yes (configurable; disabled by default) |
| Scope | One video or the global corpus | **One video, required** |
| Claim | "These comments are similar to the question" | "Viewers said X [E1]". This reports comments; it does not verify facts |

## Pipeline (`src/rag_service/rag/`)

1. **Scoped retrieval.** This reuses `RetrievalService` and the existing engine, so filtering happens before ranking. The default method is `multiview_hybrid` (BM25 + LaBSE, original + normalised query, RRF), or `multiview_bm25` when the LaBSE index is missing. 20 candidates are retrieved by default.
2. **Evidence selection** (`evidence.py`). See below. If nothing qualifies, the response is `insufficient_evidence` / `no_content_term_match` and **the LLM is not called**.
3. **Grounded generation** (`prompt.py`). Only the selected comments (at most 6) are sent, labelled `[E1]…[E6]` with their original text.
4. **Strict validation** (`ask_service.py`):
   - The model must return `{"status","answer","citations"}` JSON. Anything else gives `502 llm_invalid_output`.
   - Every cited label, whether in the list or inline `[E#]`, must be one that was supplied, and an answer must cite at least one. Otherwise the **whole answer is withheld** with `502 llm_unsupported_citation`. Bad citations are never silently removed.
   - If the model declines, the response is `insufficient_evidence` / `model_declined`.
5. **Response.**
   - `citations` and `evidence_supplied` carry the internal `document_id`, the **unmodified** `comment_text`, the retrieval rank and the matched terms.
   - `video_reference.granularity` is always `"video"`: the dataset has no comment IDs, so links point to the video, not the comment.

## Evidence-selection strategy (initial, heuristic, not evaluated)

BM25/cosine scores have no calibrated cut-off, so **no score thresholds are used**. A retrieved candidate becomes evidence only if it shares a *content term* with the question:

- **Query terms:** tokens of the original query plus its Sinhala-normalised form, minus a hand-written list of generic words (`video`, `eka`, `viewers`, `monawada`, `kiyanne`, `nathi`, Sinhala equivalents, …) and tokens shorter than 3 characters.
- **Matching:** query and comment tokens are compared with a light Singlish phonetic key (w→v, aspiration h dropped, long vowels and doubled letters collapsed). Tokens of 5 or more characters may differ by one edit, so `wetakolu` matches `vatakolu` and `poththa` matches `pothha`.
- **Ordering:** eligible comments are sorted by the number of distinct matched terms, then by retrieval rank. The top 6 are kept.

**Limitations**
- A shared content word is a weak proxy for relevance: one-word matches such as `bath` in an unrelated biryani comment can still be selected. The model's grounding rules are the second filter.
- Purely semantic matches with no shared term are excluded, so recall is lowered on purpose.
- The generic-word list and the edit-distance rule are hand-written, not tuned or evaluated on labelled data. They need the planned human relevance benchmark.

## Prompt grounding rules (summary of `prompt.py`)

- **Comments only:** use only the numbered comments, with no outside knowledge.
- **Format:** answer in Sinhala, 1–4 sentences, phrased as "viewers said…", not as fact.
- **Citations:** every sentence ends with `[E#]`. Only supplied labels may be cited, never document IDs.
- **What isn't evidence:** generic praise and generic-word matches.
- **When unsure:** if the comments do not answer the question, return `insufficient_evidence`.
- **Output:** JSON only. Temperature 0.1.

## Citation limitations

- Citations prove that a comment was **supplied** and **referenced**. They do not prove that the sentence is **entailed** by the comment. That check is the NLI verification step, which is not implemented.
- Document IDs are internal surrogates, not YouTube comment IDs. Links are video-level.
- Threads and replies are not available (no parent IDs), so each comment is used without its conversation context.

## LLM configuration

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_PROVIDER` | `disabled` | `gemini` to enable. Disabled gives `503 llm_unavailable`. |
| `RAG_LLM_MODEL` | `gemini-3.8-flash` | Any model the key can use |
| `RAG_LLM_API_KEY` / `GEMINI_API_KEY` | none | Sent only in the `x-goog-api-key` header. Never logged, never in `repr` |
| `RAG_LLM_TIMEOUT_S` | `30` | Request timeout |

- Only real providers can be configured. The mock provider exists only inside `tests/api/test_ask.py`.
- `GET /v1/health` reports `llm: {provider, model, enabled}` without calling the model.
- Privacy: only the selected public, author-stripped comments and the question are sent to the provider.

## Live demonstration status

`python scripts/demo_ask.py` runs the three demos against the configured model. It refuses to run when the LLM is disabled. Results from the M6 session:

| Demo | Retrieval and evidence (real) | Generation |
|---|---|---|
| 1. Wetakolu, video `-3a4abtQH3E` | Selected `DD488D4D88EF8` ("Wetakolu pothha … ain karanna") and `DB8B1DE7D2790` ("… vatakolu vala dara ain karanava …"). All "Nice video" comments rejected. | **Not obtained.** Gemini returned HTTP 503 "high demand" for gemini-3.8/3.7/3.5-flash. gemini-2.5-flash returned 404 (retired for new users). |
| 2. Rice/water, video `_uhQvDg6VtA` | Top evidence `D21DBE631C8CB` "බත් එක හැදෙන්න වතුර දැම්මෙ නැහැනේ…" (3 matched terms via the normalised query), plus 5 weaker one-term matches | **Not obtained** (same HTTP 503) |
| 3. Unrelated (iPhone battery) | 20 retrieved, 0 selected, 20 rejected (generic words only) | Correctly `insufficient_evidence`, **model not called** |

Every provider failure was returned as `502 llm_error`. No answer was fabricated. Re-run the script when the provider has capacity, and record the real output before presenting it.

## Remaining work (not in M6)

- **Thread-aware RAG:** needs comment IDs, parent IDs and timestamps from a new data collection.
- **Claim-level NLI verification:** check that each answer sentence is entailed by its cited comment(s), and drop or flag unsupported sentences.
- **Evidence-selection evaluation:** precision and recall of selected evidence on the human-annotated benchmark, plus comparison with score-based and reranker-based selection.
- **Answer-quality evaluation:** groundedness and faithfulness ratings by native speakers.
