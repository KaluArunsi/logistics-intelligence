# Core User Flow (Operational Summary)

This document is the concise product narrative aligned to `core_user_flow_spec.md`.

## Entry Points

Users start with two choices:
- `Upload Data`
- `Tell Us`

Session model:
- no user accounts
- IP-hash based session
- 1 hour active window
- 24 hour cooldown
- 5 minute pre-expiry warning

## Upload Data Path

1. User uploads a dataset.
2. Backend confirms upload received and stores upload artifact.
3. System profiles schema and samples data.
4. Category matching returns one of:
- `ACCEPT`
- `PARTIAL_ACCEPT`
- `REJECT`
5. If `ACCEPT`, router selects eligible industry/category worker set.
6. Column alignment maps user columns to worker contracts (canonical + approved aliases).
7. Async prediction pipeline runs:
- `L0` workers
- schema/feature alignment where needed
- `L1` category expert
- report build
8. User gets:
- `View Report`
- `Discuss Report`
- `Download Report`

Inference guardrails:
- L0 horizon max is 30 days.
- Runtime processing is async (`POST /run/predict` returns `202` + `task_id`).

## Tell Us Path

1. User selects industry + category (includes `I don't know` option).
2. User answers guided questionnaire.
3. LLM synthesizes a concise problem statement.
4. User confirms/edits statement.
5. System generates synthetic dataset candidate.
6. Generated dataset enters the same analysis/routing pipeline as upload path:
- schema analysis
- category match
- async L0/L1 run
- report + chat

## Matching and Routing Rules

- Category matching is tri-state, not a hard single threshold only:
- `ACCEPT` routes directly.
- `PARTIAL_ACCEPT` returns missing fields + fallback options + next actions.
- `REJECT` asks user to revise data/intake.
- Multi-category routing is allowed only when categories remain within the same industry and policy gates pass.

## Report Experience

Report outputs include:
- input data profile highlights
- L0 + L1 predictions and confidence signals
- executive summary and health signal
- actionable next-step recommendations

Chat experience:
- grounded on generated report context
- supports follow-up Q&A and recommendations
- respects same 1 hour session window

## State and Recovery

Frontend always rehydrates from backend:
- FE enters `REHYDRATING` on load
- calls `GET /session/status`
- resumes exact safe state via backend truth (`state`, `task_id`, `report_id`)

This prevents ghost state and supports seamless resume during long async jobs.

## Privacy and Debug Policy

- Default: minimal retention, no persistent user identity.
- On errors only: persist debug bundle for 48h TTL.
- No full transcript persistence by default.
- Keep only artifacts and metadata needed for execution and reproducibility.

## Source of Truth

Detailed contract/state/API specification lives in:
- `core_user_flow_spec.md`
- `deployment_cost_policy.md`
