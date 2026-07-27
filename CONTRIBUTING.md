# Contributing

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=. python tools/check_cost_policy.py
```

Run the backend:

```bash
PYTHONPATH=. python tools/run_backend.py --host 0.0.0.0 --port 8000
```

Serve the frontend:

```bash
python -m http.server 8080 --directory frontend
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/backend
```

If `pytest` is unavailable, run the standard-library smoke suite:

```bash
PYTHONPATH=. python -m unittest tests.backend.test_runtime_api -v
```

## Contribution Rules

- Keep generated artifacts out of Git unless they are small, deterministic fixtures needed by tests.
- Do not commit credentials, real customer data, private run logs, or model checkpoints.
- Keep reusable product logic under `src/`; scripts in `tools/` should stay thin orchestration layers.
- Prefer deterministic tests around schema matching, runtime contracts, and API behavior for backend changes.
