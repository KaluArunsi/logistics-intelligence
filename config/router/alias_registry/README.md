# Router Alias Registry

Purpose:
- Store candidate/approved/rejected column synonyms inferred during runtime.
- Keep router manifests deterministic by only consuming approved aliases.

Files:
- `aviation_column_aliases.json`
- `ecommerce_column_aliases.json`

Status model:
- `candidate`: observed alias, not yet trusted.
- `approved`: safe to inject into router manifest synonym maps.
- `rejected`: blocked alias.

Workflow:
1. Record inferred alias events:
```bash
python tools/router_alias_registry.py --base . record \
  --industry ecommerce \
  --canonical-column order_date \
  --alias transaction_date \
  --category demand_signal \
  --confidence 0.93 \
  --source llm_inference
```
2. Promote strong candidates:
```bash
python tools/router_alias_registry.py --base . promote \
  --industry ecommerce \
  --min-hit-count 3 \
  --min-confidence 0.90
```
3. Rebuild router manifest:
```bash
python tools/build_router_manifest.py --base . --industry ecommerce
```

Feedback logs:
- Runtime events append to `reports/router_feedback/<industry>_alias_events.jsonl`.
