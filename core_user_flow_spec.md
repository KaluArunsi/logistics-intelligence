# Production Architecture & Flow Spec (No-Account Privacy Model)

This document consolidates:
- A single-page architecture diagram
- Extracted API boundaries (frontend/backend/services)
- Frontend/backend coordination state machine (with rehydration)
- Async execution model (task queue + workers)
- Privacy model (no accounts) with error-only debug persistence
- Partial-match behavior for sparse datasets

## Diagrams (FigJam)

- **Single-Page Architecture Diagram:** https://www.figma.com/online-whiteboard/create-diagram/e7ca5ae4-890e-4b90-a0e3-c5da2d03060b?utm_source=chatgpt&utm_content=edit_in_figjam&oai_id=v1%2FmgiMLVgiMKutS9xPYjCYDgCdrSHtzlFLo4tQpTM4B7NC7JkMTRzf8A&request_id=a82fcac1-f66b-45da-9aaa-05c056bf1d00
- **State Machine Diagram:** https://www.figma.com/online-whiteboard/create-diagram/a601e6da-ebc8-4a4a-8c0a-aac9d28b68ba?utm_source=chatgpt&utm_content=edit_in_figjam&oai_id=v1%2FmgiMLVgiMKutS9xPYjCYDgCdrSHtzlFLo4tQpTM4B7NC7JkMTRzf8A&request_id=5f39d06f-a1be-4852-bfe7-7d752751b6ba
- **API Boundary / Sequence Diagram:** https://www.figma.com/online-whiteboard/create-diagram/02945224-59ac-441b-a9f0-b236be2d6fe8?utm_source=chatgpt&utm_content=edit_in_figjam&oai_id=v1%2FmgiMLVgiMKutS9xPYjCYDgCdrSHtzlFLo4tQpTM4B7NC7JkMTRzf8A&request_id=a4e2e1d2-3400-4d3b-9358-1fc8b891a12a

---

## 1) Non-Negotiables

### 1.1 No user accounts
- No usernames, passwords, email addresses, or persistent user identifiers.
- No long-term conversation history keyed to a person.

### 1.2 IP-based sessions (ephemeral)
- The backend derives a session key from IP (see §4).
- **1 hour use window** per IP-derived session.
- **24 hour reset/cooldown** after expiration.
- After cooldown ends, sessions restart from scratch.

### 1.3 Minimal data retention
- Default persistence is strictly bounded to operational needs (artifacts required to produce the report).
- Text transcripts should not be stored by default (see §7 for error-only debug).

---

## 2) System Overview

### 2.1 Primary user flow
1. FE boot → rehydrate from BE using `GET /session/status`.
2. Intake:
   - Upload dataset **or**
   - “Tell us” questionnaire → synthetic dataset generation.
3. Schema profiling + category matching (ACCEPT / PARTIAL_ACCEPT / REJECT).
4. Route to sector worker group(s).
5. Async predict pipeline: L0 → alignment → L1 → report build.
6. FE polls tasks or uses WS/SSE for state updates.
7. User optionally chats about the report; chat answers are grounded in model outputs and report context.
8. User can export chat summary (or transcript tail) into report artifacts.

### 2.2 Core services
- **Backend API**: session enforcement, orchestration endpoints, report/task endpoints
- **LLM Orchestrator**: prompts, synthesis, mapping, explainability
- **Data Profiler**: schema summary + lightweight stats
- **Category Matcher**: tri-state decision with missing-fields output
- **Model Router**: routes to sector worker groups and determines fallback/dual routing
- **Task Queue + Workers**: async execution of heavy pipeline
- **L0 Model Service**: broad prediction / triage model(s)
- **Schema Aligner**: column mapping + normalization
- **L1 Expert Workers**: domain-specific specialist models
- **Report Builder**: compiles outputs into a report artifact
- **Object Store**: datasets + artifacts
- **Metadata DB**: report metadata, manifests
- **Debug Store (TTL)**: error-only bundles for 48h

---

## 3) API Boundaries

### 3.1 Frontend ↔ Backend boundary
The frontend never treats local state as source-of-truth. It always re-syncs from the backend on load.

**Required FE calls**
- `GET /session/status` (boot + refresh)
- `POST /session/start` (if no active session)
- `POST /intake/upload` (upload path)
- `POST /intake/tell-us` + `POST /intake/tell-us/confirm` (questionnaire path)
- `POST /analyze/schema`
- `POST /analyze/match-category`
- `POST /run/predict` (async; returns 202 + task_id)
- `GET /task/{task_id}`
- `GET /report/{report_id}`
- `POST /chat/message`
- `GET /chat/history`
- `POST /chat/export-summary`
- Optional: `GET /events` (WS/SSE)

### 3.2 Backend ↔ Worker boundary
Heavy steps must never execute inside the request/response cycle:
- L0 inference
- alignment
- L1 experts
- report building

The API enqueues and workers execute:
- API → Queue: task creation
- Worker → services: L0/L1 inference and alignment
- Worker → object store + DB: report artifacts + metadata

### 3.3 LLM boundary
All LLM usage is behind the orchestrator, never directly in FE.

---

## 4) Session Model (IP-hash)

### 4.1 Derivation
`ip_hash = HMAC_SHA256(server_secret_vN, ip + user_agent_bucket + coarse_time_epoch)`

Notes:
- Use a **rotating server_secret** (versioned) to reduce correlation over long periods.
- Keep any UA component coarse (bucketed) to avoid fingerprinting.
- Coarse time epoch should be aligned with your 1h/24h windows.

### 4.2 Server-side session record (Session Store)
Minimum fields:
- `ip_hash`
- `created_at`, `expires_at`
- `cooldown_until`
- `state` (authoritative)
- `last_successful_step_uri`
- `active_task_id` (optional)
- `report_id` (optional)
- `artifact_uris` (upload_uri, schema_uri, routing_plan_uri, etc.)

### 4.3 Enforcement rules
- If `now < cooldown_until`: deny new work, return “locked” state.
- If `now > expires_at` and no cooldown set: set cooldown and clear live state pointers.
- If active task exists after expiry: cancel task (best-effort) and lock.

---

## 5) Rehydration (Fixes “Ghost State”)

### 5.1 FE boot algorithm
1. Set FE state = `REHYDRATING`.
2. Call `GET /session/status`.
3. If no session: go to `IDLE` and offer to start.
4. If session active: go to `RESUMING` then jump to the UI route based on:
   - `state`
   - `last_successful_step_uri`
   - `active_task_id` / `report_id`

### 5.2 `GET /session/status` response contract (minimum)
```json
{
  "has_session": true,
  "session_expires_at": "ISO8601",
  "cooldown_until": "ISO8601|null",
  "state": "MODEL_RUNNING_L0",
  "last_successful_step_uri": "/analyze/match-category",
  "active_task_id": "task_123|null",
  "report_id": "rep_456|null",
  "artifacts": {
    "upload_uri": "s3://...|null",
    "schema_uri": "s3://...|null",
    "match_uri": "s3://...|null",
    "routing_uri": "s3://...|null"
  }
}
```

---

## 6) Async Execution Model (Required)

### 6.1 Why async
API gateways commonly time out at 30–60 seconds. L0→Align→L1→Report can exceed this.

### 6.2 Contract: `POST /run/predict` returns 202
- Immediately returns:
  - `task_id`
  - `report_id` (optional if pre-created)
  - current state

Example:
```json
{
  "task_id": "task_123",
  "state": "PREDICT_ENQUEUED"
}
```

### 6.3 Task state progression
- `PREDICT_ENQUEUED`
- `MODEL_RUNNING_L0`
- `ALIGNMENT_RUNNING`
- `MODEL_RUNNING_L1`
- `REPORT_BUILDING`
- `REPORT_READY`
- `ERROR`

### 6.4 Task polling endpoint
`GET /task/{task_id}` returns:
- `state`
- `progress` (0..1)
- `updated_at`
- `error` (if ERROR)

### 6.5 Optional push updates
- `GET /events` via WS or SSE sending state transitions:
  - `{ task_id, state, progress, report_id? }`

---

## 7) Privacy vs Debugging: Error-Only Persistence (48h TTL)

### 7.1 Default behavior
- Do not persist chat transcripts or user-provided text beyond what is needed for current session execution.
- Store only:
  - artifacts necessary to build report (uploads, schemas, report)
  - session state pointers (ephemeral)
 - If user engages chat, allow report-scoped chat persistence to support explainability:
   - transcript log per report
   - exported chat summary (recommended default for downstream sharing)
   - keep no account-level long-term identity linkage

### 7.2 On ERROR: persist a debug bundle for 48h
When a session enters `ERROR`, store a minimal bundle with TTL=48h in Debug Store.

**Allowed debug bundle fields**
- `problem_statement` (if Tell-Us path)
- `schema_summary` (columns, inferred dtypes, missingness, sample stats)
- `routing decision` + match scores + missing_fields
- model version hashes (L0/L1)
- stack trace + error code
- data fingerprint (hash of sample rows; avoid raw data)
- Optional raw sample rows **only** if you add an explicit opt-in

**Prohibited**
- any stable user identifier
- full conversation transcript by default

### 7.3 Prompt-injection and scope guardrails (mandatory)
- Chat assistant must remain task-focused on trained industry workflows and report context.
- Assistant must never reveal:
  - model/provider identity
  - system/developer prompts
  - hidden instructions or security internals
- If user asks out-of-scope questions (non-industry/non-report topics), assistant must redirect to:
  - data analysis
  - prediction interpretation
  - operational recommendations tied to the selected industry/category
- Guardrails must execute before LLM calls, with deterministic fallback/redirect responses.

---

## 8) Category Matching: Replace Hard Gate With Partial Match

### 8.1 Decision outcomes
- `ACCEPT`
- `PARTIAL_ACCEPT`
- `REJECT`

### 8.2 `PARTIAL_ACCEPT` response must include
- `missing_fields` for the top domain
- `fallback_models` runnable with current schema
- `next_actions`:
  - provide missing fields (augment)
  - run fallback

Example:
```json
{
  "decision": "PARTIAL_ACCEPT",
  "top_domain": "trucking_efficiency",
  "match_score": 0.32,
  "missing_fields": ["vehicle_type", "load_weight"],
  "fallback_models": ["basic_volume_forecast"],
  "next_actions": ["provide_missing", "run_fallback"]
}
```

---

## 9) State Machine Spec (Frontend/Backend Coordination)

### 9.1 Authoritative state lives on backend
- FE displays and transitions UI according to BE state.
- FE uses `REHYDRATING`/`RESUMING` locally, but BE must return enough info to resume.

### 9.2 States (canonical)
- `IDLE`
- `REHYDRATING` (FE)
- `RESUMING`
- `SESSION_ACTIVE`
- `INTAKE_MODE_SELECT`
- `UPLOAD_PENDING`
- `UPLOAD_RECEIVED`
- `QUESTIONNAIRE_IN_PROGRESS`
- `QUESTIONNAIRE_CONFIRMED`
- `SYNTH_DATA_GENERATING`
- `SCHEMA_ANALYSIS_RUNNING`
- `SCHEMA_READY`
- `CATEGORY_EVAL_RUNNING`
- `CATEGORY_ACCEPTED`
- `PARTIAL_MATCH`
- `CATEGORY_REJECTED`
- `ROUTING_DECIDED`
- `PREDICT_ENQUEUED`
- `MODEL_RUNNING_L0`
- `ALIGNMENT_RUNNING`
- `MODEL_RUNNING_L1`
- `REPORT_BUILDING`
- `REPORT_READY`
- `CHAT_ACTIVE`
- `WARNING_5_MIN`
- `SESSION_LOCKED_COOLDOWN`
- `ERROR`
- `DEBUG_BUNDLE_PERSISTED` (internal)

### 9.3 Resume mapping
On `GET /session/status`, FE maps:
- `MODEL_RUNNING_*` → show “Processing” with task progress, enable polling
- `REPORT_READY` → show report view
- `PARTIAL_MATCH` → show missing-field resolution UI
- `SESSION_LOCKED_COOLDOWN` → show lock UI with cooldown timer

---

## 10) Production Hardening Checklist

### 10.1 Security & abuse
- Rate-limit by `ip_hash` (burst + sustained)
- Upload constraints: size/type; zip-bomb protection
- Virus scanning in intake path
- Audit logs of system events (no user PII)

### 10.2 Reliability
- Circuit breakers per sector worker group
- `503 Sector Unavailable` for degraded groups (not 500)
- Retries with jitter for transient errors (queue + IO)

### 10.3 Scalability
- Separate worker groups per sector:
  - `L0-ecommerce-workers`, `L0-shipping_freight-workers`, `L0-trucking_delivery-workers`
- Autoscale on queue depth + GPU/CPU utilization
- Backpressure: refuse new tasks when queue is saturated

### 10.4 Observability (privacy-safe)
- Structured logs keyed by `request_id`, `task_id`, and short-lived `ip_hash`
- Metrics: queue latency, task duration by stage, error rates per model version
- Tracing across API → queue → workers

---

## 11) Appendix: Minimal Endpoint List

### Session
- `GET /session/status`
- `POST /session/start`

### Intake
- `POST /intake/upload`
- `POST /intake/tell-us`
- `POST /intake/tell-us/confirm`
- `POST /intake/tell-us/augment` (for partial match)

### Analysis
- `POST /analyze/schema`
- `POST /analyze/match-category`

### Predict + Results
- `POST /run/predict` (202 + task_id)
- `GET /task/{task_id}`
- `GET /report/{report_id}`

### Report Chat
- `POST /chat/message`
- `GET /chat/history`
- `POST /chat/export-summary`

### Optional streaming
- `GET /events` (WS/SSE)

---

## 12) What changed vs POC (delta summary)
- Added frontend **rehydration** (`REHYDRATING`) via `/session/status` to eliminate ghost state.
- Enforced **async predict** with queue/worker and 202 responses.
- Added **error-only debug persistence** with 48h TTL to enable reproducibility without accounts.
- Replaced hard 50% gate with **partial match** remediation and fallback routes.
- Added report-grounded chat loop with exportable chat summary artifacts.
- Added runtime report visuals (trend/seasonality/forecast + data-quality plots).

---

## 13) Router Schema Contract (Project-Specific)

To keep inference routing deterministic and auditable, the backend uses a generated router manifest with this hierarchy:
- `industry`
- `category`
- `workers`
- `columns per worker`

Contract artifacts:
- Schema: `config/router/router_manifest.schema.json`
- Manifests:
  - `config/router/manifests/ecommerce_router_manifest.json`
  - `config/router/manifests/shipping_freight_router_manifest.json`
  - `config/router/manifests/trucking_delivery_router_manifest.json`
- Generator: `tools/build_router_manifest.py`

Guarded self-healing aliases:
- Registry: `config/router/alias_registry/<industry>_column_aliases.json`
- Manager: `tools/router_alias_registry.py`
- Feedback log: `reports/router_feedback/<industry>_alias_events.jsonl`

Alias policy:
- Runtime/LLM suggestions are recorded as `candidate`.
- Only `approved` aliases are injected into manifests.
- Promotion uses confidence + hit-count thresholds; no blind runtime mutation of canonical manifests.

---

## 14) Deployment Cost Constraints (Project Policy)

- Operate in free-first mode until paying traction exists.
- No custom domain is required for staging/testing.
- Baseline paid assumption: VPS for Dockerized model runtime only.
- Use free URLs where possible:
  - Railway `*.up.railway.app`
  - Cloudflare `*.workers.dev`
- Keep paid integrations optional and disabled by default via env flags:
  - `STRICT_ZERO_BUDGET=true`
  - `LLM_PROVIDER_MODE=oss_only`
  - `ALLOW_PAID_* = false`

Deployment scope policy:
- Deploy now: `ecommerce`, `shipping_freight`, `trucking_delivery`
- Keep `aviation` as non-production R&D reference for now

Reference:
- `deployment_cost_policy.md`
- `.env.example`
