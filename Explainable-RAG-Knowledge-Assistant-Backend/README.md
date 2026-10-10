# PP1 — Singlish Retrieval Engine (research prototype)

This is a demonstrable **baseline retrieval system**, built from the uploaded YouTube CSV and handmade query–document pairs for the fourth-year Explainable RAG Knowledge Assistant & Singlish Retrieval Engine project. It is **not** a completed RAG system, a trained Singlish model, or a validated information-retrieval benchmark.

## Quick start (Python 3.10+)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Gradio prints a localhost URL (typically `http://127.0.0.1:7860`) and opens the app in a browser. **No YouTube/Gemini API key is needed** for this offline prototype.

## FastAPI retrieval service (v1)

`service/` wraps the existing `RetrievalEngine` with `GET /v1/health` and `POST /v1/retrieve`. The Gradio app is unaffected. Setup: [docs/SERVICE_SETUP.md](docs/SERVICE_SETUP.md). Contract: [docs/openapi.json](docs/openapi.json). Frontend guide: [docs/NEXTJS_INTEGRATION.md](docs/NEXTJS_INTEGRATION.md).

## Included

- `app.py` — live query interface, ranked results, corpus dashboard, baseline diagnostics, architecture roadmap.
- `retrieval.py` — BM25 word retrieval and character n-gram TF-IDF baselines, plus optional LaBSE search.
- `data/youtube_comments_multiple.csv` — **unaltered upload backup** (includes public author handles; keep private).
- `data/comments_clean.csv` — author-removed, text-deduplicated prototype corpus; stable **surrogate** document IDs (not YouTube comment IDs).
- `data/pilot_seed_pairs.csv` — 366 exported query–document examples from the Excel file.
- `data/seed_pairs_linked.csv` — exact-text link to prototype corpus where possible; positive examples **unverified**.
- `results/dataset_profile.json` — observable dataset characteristics.
- `results/seed_diagnostic.json` — preliminary **known-positive Hit@k** diagnostics; NOT nDCG, Recall, verified model accuracy, or an independent evaluation.
- `prepare_data.py` — rebuild corpus and links from the raw CSV plus seed CSV.
- `evaluate_seed.py` — recompute seed diagnostics.
- `build_labse_index.py` — *optional* pretrained semantic index generation.

## Optional LaBSE demo (model download required)

```bash
pip install -r requirements-semantic.txt
python build_labse_index.py
python app.py
```

LaBSE downloads a significant pretrained model on first use and embeds all ~10k comments. This may take some time on a CPU. Do not run it just before a presentation without testing in advance. LaBSE on Romanized Sinhala must be **evaluated rather than assumed to work**.

## Refresh corpus after new API collection

```bash
python prepare_data.py data/youtube_comments_multiple.csv data/pilot_seed_pairs.csv
python evaluate_seed.py
```

If you change the corpus, **rebuild** `labse_index.npz` as well, otherwise the embeddings are out of date.

## What to say in PP1

**Implemented:** dataset ingestion, original comment preservation, video-only source links, privacy-minimized corpus, BM25 baseline, char TF-IDF baseline, seed-data source matching, working retrieval UI, exploratory retrieval diagnostic.

**Not implemented/validated:** comment-level permalinks (source IDs missing), thread-aware RAG (parent IDs missing), full Singlish normalization, contrastive adaptation, robust semantic retrieval, answer generation, NLI verification, full graded relevance labels and independent IR evaluation.

**Do not claim** any improvement from normalization, multi-view retrieval, or verification unless those modules have been implemented and experimentally compared. The diagnostic positive pairs may be biased because the query was produced from the document itself.

## Data ethics and limitations

The raw file contains public YouTube author handles. Do not publish the raw CSV, screenshots with identifying user information, or public downloads without your university's ethics approval and checking YouTube API/data policies. The app uses the stripped dataset. The corpus deduplicates by (video ID, normalized comment text), which is adequate for a prototype *search index* but may collapse distinct real comments with identical wording; only genuine YouTube IDs can distinguish those. Existing links are **video** links, not individual comment evidence URLs.

Original pair creation was Gemini-assisted with user acceptance/editing, and the per-row distinction cannot be recovered from the provided Excel. Human review is not equivalent to independent relevance judging. A robust benchmark needs annotator guidelines, multiple annotators, candidate pools, irrelevant examples, and query-family-disjoint held-out evaluation.
