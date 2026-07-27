# Logistics Intelligence

Logistics Intelligence is an open-source analytics workbench for turning operational logistics data into explainable diagnostics, forecasts, dashboards, and executive summaries. It is designed for local-first analysis of ecommerce, shipping/freight, and trucking/delivery workflows.

The product pairs a FastAPI backend with a static browser UI and an Ollama-backed analysis assistant. It can profile uploaded datasets, infer operational targets, generate dashboard bundles, run forecast and driver analysis, and answer follow-up questions against the current analysis context.

## What It Does

- Upload operational CSV data and infer useful business targets.
- Build schema and semantic matches for logistics metrics.
- Generate dashboard-ready summaries, charts, and narrative reports.
- Run local or Ollama-compatible LLM analysis without requiring a paid LLM provider by default.
- Package analysis artifacts for demos, pilots, and repeatable internal workflows.

## Project Structure

```text
frontend/             Static HTML/CSS/JS application
src/backend/          FastAPI runtime, API contracts, stores, reporting, LLM orchestration
src/core/             Pipeline, registry, cost policy, target selection, dataset utilities
src/features/         Feature engineering and domain-specific feature helpers
tools/                CLI entry points, audits, training, packaging, and verification scripts
config/               Industry router manifests, schemas, and semantic configuration
tests/                Backend and end-to-end regression tests
docs/                 Setup notes and release checklists
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=. python tools/check_cost_policy.py
PYTHONPATH=. python tools/run_backend.py --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
python -m http.server 8080 --directory frontend
```

Open `http://localhost:8080`. The frontend targets `http://localhost:8000` by default when served from port `8080`.

## Docker Compose

```bash
cp .env.example .env
docker compose --profile app up -d --build
```

Runtime endpoints:

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:8080`

Optional infrastructure services are available with:

```bash
docker compose --profile infra up -d
```

## Configuration

The default runtime is Ollama-compatible:

- `LLM_RUNTIME_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://localhost:11434` for local Ollama, or a compatible hosted endpoint.
- `OLLAMA_MODEL=gpt-oss:120b` by default.
- `OLLAMA_API_KEY` is optional and should only be set in `.env` or deployment secrets.

Set `BACKEND_SESSION_SECRET` to a long random value before exposing the backend outside local development. Set `BACKEND_CORS_ORIGINS` to explicit frontend origins in production.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/backend
```

If `pytest` is unavailable:

```bash
PYTHONPATH=. python -m unittest tests.backend.test_runtime_api -v
```

## Open Source Readiness

This repository is intended to be public, but public release requires a sanitized Git history. Deleted credentials, private notes, or customer artifacts can remain recoverable from Git history after removal from the current tree. See [docs/OPEN_SOURCE_CHECKLIST.md](docs/OPEN_SOURCE_CHECKLIST.md).

## License

Logistics Intelligence is licensed under the GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
