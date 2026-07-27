"""Generate aviation dataset specs from current processed data."""

import sys
import json
from pathlib import Path

base = Path('/Users/kaluarunsi/Desktop/autonomous-dispatcher/logistics_intelligence')
sys.path.insert(0, str(base))

from src.core.dataset_inventory import DatasetInventory
from src.core.industry_pipeline import IndustryPipeline
from src.core.target_selector import TargetSelector

pipeline = IndustryPipeline(base, industry='aviation')
inv = DatasetInventory(base)
entries = inv.build_inventory([base / 'data' / 'processed'])
worker_map = inv.assign_worker_groups(entries)
selector = TargetSelector()

specs = {"industry": "aviation", "datasets": {}}

for entry in entries:
    dataset_id = entry['dataset_id']
    if worker_map.get(dataset_id, dataset_id) != dataset_id:
        continue

    data_file = Path(entry['file_path'])
    if not data_file.exists():
        continue

    try:
        df = pipeline._load_yaml  # to keep lint away
        if str(data_file).endswith(('.parquet', '.parquet.zstd')):
            import polars as pl
            df = pl.read_parquet(data_file, n_rows=10000)
        else:
            import polars as pl
            df = pl.read_csv(data_file, n_rows=10000, ignore_errors=True, truncate_ragged_lines=True)
    except Exception:
        continue

    categories = pipeline._infer_categories(dataset_id, entry['columns'])
    category = categories[0]
    target_candidates = [c for c in df.columns if c in entry['profile'].has_target_candidates]
    selection = selector.select(df, target_candidates)

    targets = []
    if selection.target_column:
        targets.append({
            "name": selection.target_column,
            "task_type": selection.task_type,
            "primary": True,
        })

    specs['datasets'][dataset_id] = {
        "category": category,
        "categories": categories,
        "targets": targets,
        "features": {
            "include": [],
            "exclude": [],
            "fe": {},
        },
        "sampling": {"max_rows": 200000},
    }

out_path = base / 'config' / 'industries' / 'aviation' / 'aviation_dataset_specs.json'
out_path.write_text(json.dumps(specs, indent=2))
print('wrote', out_path)
