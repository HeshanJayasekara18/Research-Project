# C3 — Explainable RAG Knowledge Assistant & Singlish Retrieval Engine

Python service for research component C3. It retrieves YouTube comments (`POST /v1/retrieve`) and produces evidence-grounded Sinhala answers with citations (`POST /v1/ask`, M6, see [docs/RAG_M6.md](docs/RAG_M6.md)). Retrieval covers YouTube comments for Singlish, Sinhala or English questions using a FastAPI service (`GET /v1/health`, `POST /v1/retrieve`) built on the research retrieval engine (BM25, character TF-IDF, optional LaBSE, Singlish → Sinhala normalisation, RRF multi-view fusion).

It is **not** yet a complete RAG system. Minimal single-comment grounded generation exists (M6), but thread-aware RAG and claim (NLI) verification are future milestones, the evidence selection is unevaluated, and no method is claimed to outperform a baseline until independently evaluated.

## Layout

```
src/rag_service/            installable package
  api/                      FastAPI app (main.py), routes/, schemas/, errors, dependencies
  services/                 application layer: API contract -> retrieval engine
  retrieval/                engine.py, normalization.py, fusion.py (research algorithms)
  core/config.py            settings + repository paths (RAG_DATA_DIR override)
prototype/gradio_app.py     Gradio research demo
research/evaluation/        evaluation scripts + results/ (recorded diagnostics)
research/benchmarks/        human-labelled benchmark (planned)
research/experiments/       ablations and exploratory experiments
scripts/                    prepare_data, build_labse_index, export_openapi
tests/unit/  tests/api/     unittest suites
docs/                       openapi.json, setup, Next.js integration, notes
data/                       local data (git-ignored raw export; see Data ethics)
```

## Quick start (Windows PowerShell, Python 3.10+)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,prototype]"          # add ,semantic for LaBSE methods
# (or skip the install and run:  $env:PYTHONPATH = "src")

$env:RAG_SERVICE_API_KEY = "dev-local-key"   # service-to-service key; unset = localhost-only dev mode
python run.py                                           # simplest: API on http://127.0.0.1:8000 (see run.py --help)
# equivalent: python -m uvicorn rag_service.api.main:app --host 127.0.0.1 --port 8000 --reload   # http://127.0.0.1:8000/docs
python -m prototype.gradio_app                          # Gradio demo
python -m unittest discover -s tests -t .               # all tests
```

Run commands from the repository root. Service details: [docs/SERVICE_SETUP.md](docs/SERVICE_SETUP.md). Contract: [docs/openapi.json](docs/openapi.json). Frontend guide: [docs/NEXTJS_INTEGRATION.md](docs/NEXTJS_INTEGRATION.md). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Data

- `data/youtube_comments_multiple.csv` — **unaltered upload backup** with public author handles. Git-ignored; keep it private.
- `data/comments_clean.csv` — author-removed, text-deduplicated corpus with stable **surrogate** document IDs (not YouTube comment IDs).
- `data/pilot_seed_pairs.csv` — 366 exported query–document examples. `data/seed_pairs_linked.csv` links them to the corpus; positives are **unverified**.
- `research/evaluation/results/` — dataset profile and preliminary **known-positive Hit@k** diagnostics. These are not nDCG, Recall, verified accuracy or an independent evaluation.

## Optional LaBSE (model download required)

```powershell
pip install -e ".[semantic]"
python scripts/build_labse_index.py
```

LaBSE downloads a large model on first use and embeds all ~10k comments, which can take a while on CPU. Its behaviour on Romanized Sinhala must be **evaluated rather than assumed**.

## Refresh the corpus and diagnostics

```powershell
python scripts/prepare_data.py data/youtube_comments_multiple.csv data/pilot_seed_pairs.csv
python -m research.evaluation.evaluate_seed
python -m research.evaluation.evaluate_crossscript
python -m research.evaluation.evaluate_normalization
python -m research.evaluation.evaluate_multiview
```

If the corpus changes, **rebuild** `data/labse_index.npz` too, otherwise the embeddings are out of date.

## Claims

**Implemented:** dataset ingestion, a privacy-minimised corpus, BM25 and char TF-IDF baselines, optional LaBSE, experimental Singlish normalisation and multi-view RRF, video-scoped retrieval, the Gradio prototype, and the v1 retrieval API.

**Implemented in M6, not yet validated:** `/v1/ask` grounded Sinhala answers with citation validation (mock-tested; no live answer recorded yet).

**Not implemented or validated:** comment-level permalinks, thread-aware RAG, NLI verification, graded relevance labels, and independent IR evaluation. **Do not claim** improvements from normalisation, multi-view retrieval or verification until they are experimentally compared. Diagnostic positive pairs may be biased because each query was produced from its document.

## Data ethics and limitations

The raw file contains public YouTube author handles. Do not publish the raw CSV, screenshots with identifying information, or public downloads without ethics approval and a check of YouTube API/data policies. The corpus deduplicates by (video ID, normalised text), which may collapse distinct comments with identical wording. Links point to **videos**, not individual comments.

Original pair creation was Gemini-assisted with user acceptance/editing. Human review is not independent relevance judging. A robust benchmark needs annotator guidelines, multiple annotators, candidate pools, irrelevant examples, and query-family-disjoint held-out evaluation.
