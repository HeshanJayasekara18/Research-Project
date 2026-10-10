# Retrieval service — local setup (Windows)

FastAPI service around the existing `RetrievalEngine`. The Gradio prototype (`python app.py`) is unchanged and still works.

## Setup (PowerShell, run in this folder)

```powershell
py -3.12 -m venv .venv            # 3.10+ works
.venv\Scripts\Activate.ps1
pip install -r requirements-service.txt
pip install -r requirements-semantic.txt   # optional: LaBSE methods
Copy-Item .env.example .env                # then edit values
```

## Run

```powershell
$env:RAG_SERVICE_API_KEY = "dev-local-key"   # optional in development
uvicorn service.main:app --host 127.0.0.1 --port 8000 --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: `Invoke-RestMethod http://127.0.0.1:8000/v1/health`
- Retrieve:

```powershell
$body = @{ query = "wetakolu poththa ain karanna oneda"; method = "bm25"; top_k = 5; scope = @{ video_id = "-3a4abtQH3E" } } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/retrieve -Method Post -ContentType "application/json; charset=utf-8" -Headers @{ "X-Service-Key" = "dev-local-key" } -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

The `.env` file is **not** loaded automatically. Set the variables in the shell, or run uvicorn with `--env-file .env`.

## Test

```powershell
python -m unittest discover -s tests -v       # all tests (research + API)
python -m unittest tests.test_api -v          # API only
python scripts/export_openapi.py              # regenerate docs/openapi.json after a contract change
python scripts/export_openapi.py --check      # CI check
```

## Environment variables

| Name | Default | Meaning |
|---|---|---|
| `RAG_ENV` | `development` | `production` refuses to start without an API key |
| `RAG_SERVICE_API_KEY` | unset | Shared secret, sent as `X-Service-Key`. If unset, `/v1/retrieve` is open (development only). |
| `RAG_ALLOW_GLOBAL_SCOPE` | `true` | `false` makes `scope.video_id` mandatory |
| `RAG_CORS_ORIGINS` | empty | Leave empty. Browsers should never call this service directly. |
| `SINGLISH_LABSE_MODEL` | `sentence-transformers/LaBSE` | LaBSE model name or local path |

## Layout

```
service/
  main.py                      app factory, lifespan (loads engine once), request-id middleware
  config.py                    env → Settings
  schemas.py                   Pydantic models = API contract
  errors.py                    uniform error envelope
  dependencies.py              service lookup + X-Service-Key check
  routers/v1.py                /v1/health, /v1/retrieve (thin)
  services/retrieval_service.py adapter: API enums → existing RetrievalEngine
scripts/export_openapi.py      writes docs/openapi.json
tests/test_api.py              HTTP contract tests
```
