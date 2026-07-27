# Backend Runtime (Core User Flow Contract)

This backend implements the API surface and state flow from `core_user_flow_spec.md`:

- Session: `GET /session/status`, `POST /session/start`
- Intake: `POST /intake/upload`, `POST /intake/tell-us`, `POST /intake/tell-us/turn`, `POST /intake/tell-us/finalize`, `POST /intake/tell-us/confirm`, `POST /intake/tell-us/augment`
- Intake helper: `POST /intake/tell-us/questions`
- Analysis: `POST /analyze/schema`, `POST /analyze/match-category`
- Predict: `POST /run/predict` (async 202), `GET /task/{task_id}`, `GET /report/{report_id}`, `GET /report/{report_id}/asset/{asset_name}`
- Chat: `POST /chat/message`, `GET /chat/history`, `POST /chat/export-summary`
- Streaming: `GET /events` (SSE)

## Run locally

```bash
PYTHONPATH=. .venv/bin/python tools/run_backend.py --host 0.0.0.0 --port 8000
```

## Notes

- Session model uses IP-hash with 1h active window and 24h cooldown.
- Runtime routing is manifest-driven from `config/router/manifests/*_router_manifest.json`.
- Runtime self-healing aliases are persisted under `config/router/alias_registry/*_column_aliases.json`.
- Alias feedback events are appended to `reports/router_feedback/*_alias_events.jsonl`.
- Target semantics registry is persisted under `config/router/target_semantics/*_target_semantics.json`.
- Target semantics feedback events are appended to `reports/router_feedback/*_target_semantic_events.jsonl`.
- Reports and debug bundles are persisted under `exports/runtime_reports` and `exports/runtime_debug`.
- Chat transcripts are stored per report under `exports/runtime_chats`, with summary exports in `exports/runtime_chats/summaries`.
- Task orchestration is async via threadpool and canonical state transitions.
- Built-in rate limits are enforced for upload, tell-us, predict, and chat endpoints.
- Runtime retention sweeps remove expired artifacts using TTL env controls.

## LLM Runtime

Ollama is the only supported provider:

```bash
export LLM_RUNTIME_PROVIDER=ollama
export OLLAMA_BASE_URL=https://ollama.com
export OLLAMA_MODEL=gpt-oss:120b-cloud
export OLLAMA_API_KEY=...
export OLLAMA_TIMEOUT_SEC=90
export OLLAMA_CHAT_NUM_PREDICT=4096
export OLLAMA_DISABLE_THINKING=1
export OLLAMA_EMPTY_CONTENT_RETRY=1
export TOJI_INTAKE_FAST_MODE=1
export TOJI_OLLAMA_CLOUD_ONLY=1
```

Behavior:
- `tell-us/questions` produces 3 industry-level and 3 category-level prompts.
- `tell-us/confirm` synthesizes 1000 rows for selected worker columns.
- Toji unified flow (default) does not require router/L0/L1 endpoints.
- `analyze/match-category` is legacy-only and disabled by default; enable with `ENABLE_LEGACY_ROUTER=1` only for migration/debug.
- `run/predict` attaches runtime inference + optional LLM summary to report payload.
- `chat/message` is guardrailed against prompt injection and off-scope requests; it redirects back to report/industry analysis.
- `run/predict` also generates report visuals (quality/distribution + timeseries trend/seasonality/forecast when temporal data exists).
- `chat/export-summary` writes a summary artifact and links it into the report payload.

Promotion workflow:

```bash
# Promote candidates by threshold
PYTHONPATH=. .venv/bin/python tools/promote_target_semantics.py --industry ecommerce --min-hit-count 3 --min-confidence 0.9

# Manually approve a specific target semantic
PYTHONPATH=. .venv/bin/python tools/promote_target_semantics.py --industry ecommerce --target value1 --status approved --note "Reviewed by ops"
```

QLoRA dataset build:

```bash
# Build exhaustive Toji fine-tune corpus from runtime traces + manifest bootstrap
PYTHONPATH=. .venv/bin/python tools/build_toji_finetune_dataset.py --out-dir data/toji_finetune --output-prefix toji_v1
```

QLoRA/LoRA training + evaluation:

```bash
# Install optional fine-tuning dependencies
.venv/bin/pip install -r requirements-toji-qlora.txt

# Build Colab upload bundle (recommended for GPU training workflow)
PYTHONPATH=. .venv/bin/python tools/package_toji_colab_bundle.py \
  --dataset-dir data/toji_finetune \
  --prefix toji_v2 \
  --out-dir dist/colab

# Preview the training plan (safe check, no model training)
PYTHONPATH=. .venv/bin/python tools/train_toji_qlora.py \
  --train-file data/toji_finetune/toji_v2_messages_train.jsonl \
  --val-file data/toji_finetune/toji_v2_messages_val.jsonl \
  --run-name toji_qwen25_7b_v1 \
  --quantization auto \
  --plan-only

# Train adapter (auto => QLoRA on CUDA+bitsandbytes, LoRA fallback on Apple Silicon)
PYTHONPATH=. .venv/bin/python tools/train_toji_qlora.py \
  --train-file data/toji_finetune/toji_v2_messages_train.jsonl \
  --val-file data/toji_finetune/toji_v2_messages_val.jsonl \
  --run-name toji_qwen25_7b_v1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --quantization auto \
  --epochs 2 \
  --max-seq-len 2048

# Evaluate with local Ollama (pre/post finetune benchmarking)
PYTHONPATH=. .venv/bin/python tools/eval_toji_qlora.py \
  --runtime ollama \
  --ollama-model qwen2.5:7b \
  --test-file data/toji_finetune/toji_v2_messages_test.jsonl \
  --out outputs/toji_eval/qwen25_base_eval.json \
  --save-predictions

# Optional harness sanity check only (not a real model benchmark)
# PYTHONPATH=. .venv/bin/python tools/eval_toji_qlora.py --runtime reference
```

Local CPU quickstart:

- See `docs/toji_local_cpu_quickstart.md`

Ollama adapter packaging:

```bash
# Write Modelfile from adapter output (no create yet)
PYTHONPATH=. .venv/bin/python tools/package_toji_ollama_adapter.py \
  --base-model-tag qwen2.5:7b \
  --adapter-dir outputs/toji_qlora/toji_qwen25_7b_v1/adapter \
  --toji-tag toji:qwen2.5-7b-v1

# Create deployable Ollama model tag from Modelfile
PYTHONPATH=. .venv/bin/python tools/package_toji_ollama_adapter.py \
  --base-model-tag qwen2.5:7b \
  --adapter-dir outputs/toji_qlora/toji_qwen25_7b_v1/adapter \
  --toji-tag toji:qwen2.5-7b-v1 \
  --create

# Use Toji at runtime
export LLM_RUNTIME_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=toji:qwen2.5-7b-v1
```
