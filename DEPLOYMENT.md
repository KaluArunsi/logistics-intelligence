# Deployment Quickstart (Free-First)

This project supports a free-first local/VPS stack with Docker Compose.

Current deployment scope:
- `ecommerce`
- `shipping_freight`
- `trucking_delivery` (target industry; enable once training artifacts are ready)

R&D reference scope:
- `aviation` configuration exists for experimentation, but is not part of the default deployment path.

## 1) Prepare env

```bash
cp .env.example .env
```

Keep zero-budget defaults enabled:
- `STRICT_ZERO_BUDGET=true`
- `LLM_PROVIDER_MODE=oss_only`
- all `ALLOW_PAID_*` flags `false`

## 2) Validate policy

```bash
python tools/check_cost_policy.py
```

## 3) Start infra services (optional for future API runtime)

```bash
docker compose --profile infra up -d
```

Services:
- Postgres: `localhost:5432`
- Redis: `localhost:6379`
- MinIO API: `localhost:9000`
- MinIO Console: `localhost:9001`

## 4) Start application runtime (backend + frontend)

```bash
docker compose --profile app up -d --build
```

Runtime endpoints:
- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:8080`

If frontend runs on port `8080`, the pages auto-target backend at `http://localhost:8000`.
To override API base manually:

```js
localStorage.setItem('apiBase', 'https://your-backend-url')
```

## 5) Run industry pipeline in container

```bash
docker compose --profile train run --rm model-runner
```

You can override target industry at runtime:

```bash
INDUSTRY=ecommerce docker compose --profile train run --rm model-runner
```

## 6) Backend regression tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.backend.test_runtime_api -v
```

## 7) Railway/Cloudflare (no custom domain required)

- Railway deploy can use platform URL (`*.up.railway.app`)
- Cloudflare Worker can use free URL (`*.workers.dev`)

Custom domain is optional and can be deferred until production readiness.

## 8) Railway Staging/Demo Strategy (Zero-Budget)

Use Railway as your default staging/demo surface until revenue exists.

Principles:
- no paid domain required
- demo clients can use Railway URL directly
- preserve free-first architecture and avoid paid lock-in

Railway quick steps:
1. Create Railway project and connect this repo.
2. Use root `railway.json` (Dockerfile deploy + backend start command).
3. Set only required env vars:
   - `BACKEND_SESSION_SECRET`
   - `BACKEND_CORS_ORIGINS`
   - optional Ollama vars (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`)
4. Confirm health at `/healthz`.
5. Point frontend to Railway API URL using:
   - `localStorage.setItem('apiBase', 'https://<service>.up.railway.app')`

Reference profile:
- `config/deployment/railway_staging.yaml`
