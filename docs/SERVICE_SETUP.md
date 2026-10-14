# Retrieval service — local setup (Windows)

FastAPI service around the existing `RetrievalEngine`. The Gradio prototype (`python -m prototype.gradio_app`) is unchanged and still works.

## Setup (PowerShell, run from the repository root)

Option A, recommended (virtual environment and editable install):

```powershell
py -3.12 -m venv .venv            # 3.10+ works
.venv\Scripts\Activate.ps1
pip install -e ".[dev,prototype]"          # editable install of src/rag_service
pip install -e ".[semantic]"               # optional: LaBSE methods
```

Option B, current shortcut (dependencies already installed in your Python): skip the install and set `PYTHONPATH` in each new terminal so Python finds `src/rag_service`:

```powershell
$env:PYTHONPATH = "src"
```

## Run the API

Simplest (no PYTHONPATH or install needed; reads `.env` if present, auto-reloads on code changes):

```powershell
python run.py                    # http://127.0.0.1:8000, docs at /docs
python run.py --port 8001        # other port;  --no-reload to disable reload
```

`run.py` refuses to listen on a non-local `--host` without `RAG_SERVICE_API_KEY`.

Equivalent manual command:

The app object is `app` in `src/rag_service/api/main.py`, so the module path is `rag_service.api.main:app`.

```powershell
$env:PYTHONPATH = "src"                       # Option B only
$env:RAG_SERVICE_API_KEY = "dev-local-key"    # recommended even locally
python -m uvicorn rag_service.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for `Application startup complete.`

- Swagger UI: http://127.0.0.1:8000/docs
- Health: `Invoke-RestMethod http://127.0.0.1:8000/v1/health`
- Retrieve:

```powershell
$body = @{ query = "wetakolu poththa ain karanna oneda"; method = "bm25"; top_k = 5; scope = @{ video_id = "-3a4abtQH3E" } } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/retrieve -Method Post -ContentType "application/json; charset=utf-8" -Headers @{ "X-Service-Key" = "dev-local-key" } -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

`.env` is **not** loaded automatically. Either set variables with `$env:` as above, or copy `.env.example` to `.env` and add `--env-file .env` to the uvicorn command.

If port 8000 is busy (`[winerror 10048]`), another backend is probably already running. Stop it, or use `--port 8001` and update `KNOWLEDGE_ASSISTANT_API_URL` in the frontend.

## Grounded answers: `POST /v1/ask` (M6)

Disabled by default (`503 llm_unavailable`). To enable Gemini for a session:

```powershell
$env:RAG_LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "<your key>"          # or RAG_LLM_API_KEY
python run.py
python scripts/demo_ask.py                  # live demos (real model calls); refuses to run if disabled
```

The design, evidence-selection strategy and limitations are in [RAG_M6.md](RAG_M6.md).

## Authentication and security

- `RAG_SERVICE_API_KEY` is **service-to-service authentication**. The Next.js server (or the team API gateway) sends it as `X-Service-Key` to prove that it is the caller. It is **not** creator login and **not** YouTube channel authorisation. `scope.video_id` is only a filter, and every response reports `ownership_verified: false`. The caller must authorise the user for a video before forwarding a request.
- **API-key-free mode is for localhost development only.** Without the key, `/v1/retrieve` accepts any caller and the service logs a warning. Only run it with `--host 127.0.0.1` on your own machine.
- **Production:** set `RAG_ENV=production` and a strong `RAG_SERVICE_API_KEY` (`python -c "import secrets; print(secrets.token_urlsafe(32))"`). With `RAG_ENV=production` the service **refuses to start** without a key. The guard depends on `RAG_ENV`, so every deployment config must set it.
- The key must stay server-side. In the frontend it lives only in `KNOWLEDGE_ASSISTANT_API_KEY` (never `NEXT_PUBLIC_*`) and is used only in Route Handlers.

## Gradio research demo

```powershell
$env:PYTHONPATH = "src"                       # Option B only
python -m prototype.gradio_app                # http://127.0.0.1:7860, opens a browser tab
```

## Test

```powershell
$env:PYTHONPATH = "src"                       # Option B only
python -m unittest discover -s tests -t . -v  # all tests (unit + API)
python -m unittest tests.api.test_api -v      # API only
python scripts/export_openapi.py              # regenerate docs/openapi.json after a contract change
python scripts/export_openapi.py --check      # CI check
```

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `RAG_ENV` | `development` | `production` refuses to start without an API key |
| `RAG_SERVICE_API_KEY` | unset | Service-to-service secret, sent as `X-Service-Key`. If unset, `/v1/retrieve` is open: **localhost development only**. |
| `RAG_ALLOW_GLOBAL_SCOPE` | `true` | `false` makes `scope.video_id` mandatory |
| `RAG_CORS_ORIGINS` | empty | Leave empty. Browsers should never call this service directly. |
| `RAG_DATA_DIR` | `./data` | Folder holding `comments_clean.csv` and `labse_index.npz` |
| `PYTHONPATH` | unset | `src` when running without `pip install -e .` (Option B) |
| `SINGLISH_LABSE_MODEL` | `sentence-transformers/LaBSE` | LaBSE model name or local path |

## Layout

```
src/rag_service/
  api/main.py                     app factory, lifespan (loads engine once), request-id middleware
  api/routes/health.py, retrieve.py  thin /v1 handlers
  api/schemas/retrieval.py, common.py  Pydantic models = API contract
  api/errors.py, dependencies.py  error envelope; service lookup + X-Service-Key check
  services/retrieval_service.py   adapter: API enums -> RetrievalEngine
  retrieval/                      engine, normalization, fusion (research code)
  core/config.py                  env -> Settings; repository paths
scripts/export_openapi.py         writes docs/openapi.json
tests/api/test_api.py             HTTP contract tests
```
