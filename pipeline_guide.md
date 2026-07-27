# Industry Pipeline Guide (v3.0)

This guide describes the **industry-agnostic** pipeline for all industries
(aviation, retail, ecommerce, hospitality, travel, shipping & freight, delivery services,
restaurants, route optimization).

## Architecture
**Two-level hierarchy**:
- **L0 Workers**: 1 model per **unique dataset** (deduplicated by fingerprint)
- **L1 Experts**: 1 model per category (deep by default, reinforcement optional for regression)

## Gating Rules
- L1 trains **only if all L0 workers pass** the **0.90** minimum benchmark.
- Industry is **done** only when:
  - At least **30 unique L0 workers** exist
  - All L0 pass
  - All L1 pass

## Pipeline Stages

### Stage 1: Dataset Inventory
- Scan `data/processed` and `data/sampled`
- Build fingerprints for deduplication
- Emit `dataset_details.json`

### Stage 2: Target Selection + Feature Prep
- Auto-select target column per dataset
- Encode categoricals with deterministic mappings
- Persist feature metadata for reproducibility

### Stage 3: L0 Training
- Train 1 L0 worker per unique dataset
- Validate against `config/benchmarks.yaml` (minimum tier)
- Save models + metadata

### Stage 4: L1 Training (Category Experts)
- Train category experts using L0 outputs
- Deep learning default; reinforcement optional for regression
- Validate against minimum benchmarks

## Run Artifacts
All outputs are written per run to:
`reports/runs/<run_id>/reports/`

Files:
- `dataset_details.json`
- `l0_results.json`
- `l1_results.json`
- `summary.json`

## Running the Pipeline
Use `IndustryPipeline`:
```python
from pathlib import Path
from logistics_intelligence.src.core.industry_pipeline import IndustryPipeline

pipeline = IndustryPipeline(Path("."), industry="aviation")
summary = pipeline.run(min_workers=30, l1_method="deep")
print(summary)
```
