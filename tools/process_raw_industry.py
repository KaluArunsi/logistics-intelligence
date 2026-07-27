#!/usr/bin/env python3
"""Process raw industry datasets into cleaned, categorized parquet files and specs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.registry import DatasetMeta, Registry  # noqa: E402
from src.core.target_selector import TargetSelector  # noqa: E402
from src.data.compressor import DataCompressor  # noqa: E402
from src.data.processor import DataProcessor  # noqa: E402


class RawIndustryProcessor:
    """Process raw datasets and generate first-pass training specs."""

    SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls", ".parquet"}
    TARGET_HINT_PATTERNS = [
        "target",
        "label",
        "class",
        "outcome",
        "delay",
        "delayed",
        "cancelled",
        "canceled",
        "rating",
        "score",
        "sentiment",
        "price",
        "fare",
        "revenue",
        "demand",
        "volume",
        "conversion",
        "converted",
        "purchase",
        "intent",
        "fraud",
        "chargeback",
        "risk",
        "gmv",
        "profit",
        "margin",
        "late",
        "sla",
        "breach",
        "quality",
        "suppression",
    ]


    def __init__(self, base_path: Path, industry: str, max_rows: int):
        self.base_path = Path(base_path)
        self.industry = industry
        self.max_rows = int(max_rows)

        self.raw_dir = self.base_path / "data" / "raw" / industry
        self.processed_dir = self.base_path / "data" / "processed" / industry
        self.spec_path = self.base_path / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
        self.report_path = self.base_path / "reports" / f"{industry}_processing_report.json"

        self.processor = DataProcessor()
        self.compressor = DataCompressor(self.processed_dir)
        self.registry = Registry(self.base_path, industry=industry)
        self.target_selector = TargetSelector()

        self.industry_cfg = self._read_yaml(self.base_path / "config" / "industries" / industry / "industry.yaml")
        self.categories_cfg = self._read_yaml(self.base_path / "config" / "industries" / industry / "categories.yaml")

        self.categories = list((self.categories_cfg.get("categories") or {}).keys())
        self.category_keywords = {
            name: [str(k).lower() for k in (cfg.get("keywords") or [])]
            for name, cfg in (self.categories_cfg.get("categories") or {}).items()
        }
        self.dataset_overrides = self.industry_cfg.get("dataset_category_overrides") or {}
        self.target_per_category = {
            name: int((cfg or {}).get("target_datasets", 5))
            for name, cfg in (self.industry_cfg.get("categories") or {}).items()
        }

    @staticmethod
    def _is_id_like_column(name: str) -> bool:
        col = str(name).strip().lower()
        return bool(
            col in {"id", "row_id", "record_id", "uuid", "uid", "index"}
            or col.endswith("_id")
            or re.match(r".*id\d*$", col)
            or "fips" in col
            or col.endswith("code")
            or col.startswith("code_")
        )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _sanitize(value: str) -> str:
        out = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        out = re.sub(r"_+", "_", out)
        return out or "dataset"

    @staticmethod
    def _is_supported(path: Path) -> bool:
        name = path.name.lower()
        if name.startswith("_"):
            return False
        if name in {"metadata.json", "_download_manifest.json"}:
            return False
        if name.endswith(".parquet.zstd"):
            return True
        return path.suffix.lower() in RawIndustryProcessor.SUPPORTED_SUFFIXES

    @staticmethod
    def _infer_separator(path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(8192)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except Exception:
            return ","

    def _derive_dataset_id(self, file_path: Path) -> str:
        rel = file_path.relative_to(self.raw_dir)
        parts = [self._sanitize(p) for p in rel.parts]
        last = parts[-1]
        if last.endswith("parquet_zstd"):
            last = last[: -len("parquet_zstd")].rstrip("_")
        elif "." in last:
            last = last.rsplit("_", 1)[0]
        parts[-1] = last or "dataset"
        dataset_id = "__".join(p for p in parts if p)
        if len(dataset_id) <= 140:
            return dataset_id
        short_hash = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
        return f"{dataset_id[:125]}_{short_hash}"

    def discover_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        files = [p for p in self.raw_dir.rglob("*") if p.is_file() and self._is_supported(p)]
        return sorted(files)

    def _read_csv_like(self, path: Path) -> pl.DataFrame:
        sep = "\t" if path.suffix.lower() == ".tsv" else self._infer_separator(path)
        try:
            return pl.read_csv(
                path,
                separator=sep,
                infer_schema_length=10000,
                ignore_errors=True,
                truncate_ragged_lines=True,
                null_values=["", "NA", "N/A", "null", "NULL", "\\N"],
                try_parse_dates=True,
            )
        except Exception:
            import pandas as pd

            df = pd.read_csv(path, sep=sep, engine="python", dtype=str, on_bad_lines="skip")
            return pl.from_pandas(df)

    def _read_excel(self, path: Path) -> pl.DataFrame:
        import pandas as pd

        xl = pd.ExcelFile(path)
        best_df = None
        best_score = -1
        for sheet in xl.sheet_names:
            try:
                pdf = pd.read_excel(path, sheet_name=sheet, dtype=str)
            except Exception:
                continue
            score = int(pdf.shape[0]) * max(1, int(pdf.shape[1]))
            if score > best_score:
                best_score = score
                best_df = pdf
        if best_df is None:
            return pl.DataFrame()
        return pl.from_pandas(best_df)

    def _read_file(self, path: Path) -> pl.DataFrame:
        name = path.name.lower()
        if name.endswith(".parquet.zstd") or path.suffix.lower() == ".parquet":
            return pl.read_parquet(path)
        if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            return self._read_csv_like(path)
        if path.suffix.lower() == ".json":
            return pl.read_json(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return self._read_excel(path)
        raise ValueError(f"Unsupported file type: {path}")

    def _fallback_category(self, search_text: str) -> tuple[str, str | None]:
        # Industry-agnostic fallback: reuse configured category keywords.
        best_category = None
        best_score = 0
        for category, tokens in self.category_keywords.items():
            score = sum(1 for token in tokens if token and token in search_text)
            if score > best_score:
                best_score = score
                best_category = category

        if best_category and best_score > 0:
            return best_category, None

        default_category = self.categories[0] if self.categories else "operations"
        return default_category, None

    def _infer_categories(self, dataset_id: str, columns: list[str]) -> tuple[str, list[str], bool, str | None]:
        override = self.dataset_overrides.get(dataset_id)
        if override:
            if isinstance(override, list):
                cats = [str(c) for c in override if str(c)]
                if cats:
                    return cats[0], cats, False, None
            return str(override), [str(override)], False, None

        search_text = f"{dataset_id} {' '.join(columns)}".lower()
        scores: dict[str, int] = {}
        for category, keywords in self.category_keywords.items():
            scores[category] = sum(1 for kw in keywords if kw and kw in search_text)

        if scores:
            best = max(scores.values())
            if best > 0:
                selected = [cat for cat, score in scores.items() if score >= max(1, int(0.8 * best))]
                selected = selected or [max(scores, key=scores.get)]
                selected = sorted(selected, key=lambda c: (-scores.get(c, 0), c))
                return selected[0], selected, False, None

        fallback, suggested_new = self._fallback_category(search_text)
        return fallback, [fallback], True, suggested_new

    def _choose_primary_category(self, ranked_categories: list[str], category_counts: Counter) -> str:
        """Choose a primary category, prioritizing underfilled categories when overlap exists."""
        if not ranked_categories:
            return self.categories[0] if self.categories else "operations"
        if len(ranked_categories) == 1:
            return ranked_categories[0]

        best_category = ranked_categories[0]
        best_deficit = -1
        for category in ranked_categories:
            target = int(self.target_per_category.get(category, 0))
            current = int(category_counts.get(category, 0))
            deficit = max(0, target - current)
            if deficit > best_deficit:
                best_deficit = deficit
                best_category = category
        if best_deficit > 0:
            return best_category
        return ranked_categories[0]

    def _target_candidates(self, columns: list[str]) -> list[str]:
        candidates = []
        for col in columns:
            col_l = col.lower()
            if self._is_id_like_column(col_l):
                continue
            if re.search(r"(^|_)count($|_)", col_l):
                candidates.append(col)
                continue
            if any(pattern in col_l for pattern in self.TARGET_HINT_PATTERNS):
                candidates.append(col)
        return candidates

    def _fallback_target(self, df: pl.DataFrame) -> tuple[str | None, str | None]:
        priority = [
            ("converted", "classification"),
            ("conversion", "classification"),
            ("purchase", "classification"),
            ("fraud", "classification"),
            ("chargeback", "classification"),
            ("risk", "classification"),
            ("late", "classification"),
            ("sla_breach", "classification"),
            ("quality", "classification"),
            ("revenue", "regression"),
            ("sales", "regression"),
            ("amount", "regression"),
            ("price", "regression"),
            ("gmv", "regression"),
            ("margin", "regression"),
            ("profit", "regression"),
        ]
        lower_map = {c.lower(): c for c in df.columns}
        for token, task in priority:
            for key, original in lower_map.items():
                if self._is_id_like_column(original):
                    continue
                if token in key:
                    if task == "classification":
                        try:
                            if df[original].n_unique() > 100:
                                continue
                        except Exception:
                            continue
                    return original, task

        # Generic fallback for metric-heavy operational tables that lack explicit target hints.
        metric_name_tokens = (
            "value",
            "volume",
            "count",
            "rate",
            "ratio",
            "ton",
            "teu",
            "weight",
            "time",
            "delay",
            "dwell",
            "revenue",
            "cost",
            "price",
            "amount",
            "shipment",
            "passenger",
            "freight",
            "index",
            "capacity",
        )
        numeric_cols: list[tuple[int, int, str]] = []
        for col in df.columns:
            if self._is_id_like_column(col):
                continue
            series = df[col]
            if series.dtype not in (
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ):
                continue
            try:
                uniq = int(series.n_unique())
                non_null = int(series.is_not_null().sum())
            except Exception:
                continue
            if uniq < 5 or non_null < max(20, int(df.height * 0.2)):
                continue
            token_bonus = 1 if any(t in col.lower() for t in metric_name_tokens) else 0
            numeric_cols.append((token_bonus, uniq, col))

        if numeric_cols:
            numeric_cols.sort(key=lambda x: (-x[0], -x[1], x[2]))
            chosen = numeric_cols[0][2]
            task = "classification" if numeric_cols[0][1] <= 20 else "regression"
            return chosen, task
        return None, None

    def _build_spec(self, dataset_id: str, df: pl.DataFrame, category: str, categories: list[str]) -> tuple[dict[str, Any], str]:
        sample = df if df.height <= 50000 else df.sample(n=50000, seed=42)
        target_candidates = self._target_candidates(sample.columns)
        selection = self.target_selector.select(sample, target_candidates)

        target_name = selection.target_column
        target_task = selection.task_type
        reason = selection.reason
        if target_name and self._is_id_like_column(target_name):
            target_name = None
            target_task = None
            reason = "discarded_id_like_target"
        if target_name is None:
            target_name, target_task = self._fallback_target(sample)
            if target_name:
                reason = "fallback_target_hint"

        text_cols = [c for c in sample.columns if sample[c].dtype in (pl.Utf8, pl.String)]
        features: dict[str, Any] = {
            "include": [],
            "exclude": [],
            "fe": {},
            "text_features": 128 if len(text_cols) > 5 else 64,
        }
        sampling = {"max_rows": self.max_rows, "batch_size": 50000}
        training = {"gating": bool(target_name)}

        spec: dict[str, Any] = {
            "category": category,
            "categories": categories,
            "targets": [],
            "features": features,
            "sampling": sampling,
            "training": training,
        }
        if target_name:
            spec["targets"] = [{"name": target_name, "task_type": target_task or "classification", "primary": True}]
        else:
            spec["skip_training"] = True
        return spec, reason

    def _reset_processed_state(self) -> None:
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        datasets_file = self.base_path / "data" / "metadata" / self.industry / f"{self.industry}_datasets.json"
        datasets_file.parent.mkdir(parents=True, exist_ok=True)
        datasets_file.write_text("{}", encoding="utf-8")
        self.registry = Registry(self.base_path, industry=self.industry)

    def run(self, clean_processed: bool = False) -> dict[str, Any]:
        if clean_processed:
            self._reset_processed_state()
            existing = self._read_json(self.spec_path)
            specs: dict[str, Any] = existing if isinstance(existing, dict) else {}
            specs["industry"] = self.industry
            specs["version"] = int(specs.get("version", 1) or 1)
            specs["notes"] = (
                specs.get("notes")
                or "Dataset-level training specs. Add one entry per unique dataset worker."
            )
            specs["datasets"] = {}
        else:
            specs = self._read_json(self.spec_path)
            specs.setdefault("industry", self.industry)
            specs.setdefault("version", 1)
            specs.setdefault("notes", "Dataset-level training specs. Add one entry per unique dataset worker.")
            specs.setdefault("datasets", {})

        discovered = self.discover_files()
        per_category = Counter()
        skipped: list[dict[str, Any]] = []
        processed: list[dict[str, Any]] = []
        low_confidence: list[dict[str, Any]] = []
        suggested_new_categories = Counter()

        for file_path in discovered:
            dataset_id = self._derive_dataset_id(file_path)
            source = file_path.relative_to(self.raw_dir).parts[0] if file_path.relative_to(self.raw_dir).parts else "raw"
            try:
                df = self._read_file(file_path)
                if df.height == 0 or df.width == 0:
                    skipped.append({"dataset_id": dataset_id, "file": str(file_path), "reason": "empty_input"})
                    continue

                if df.height > self.max_rows:
                    df = df.sample(n=self.max_rows, seed=42)

                df = self.processor.process(df)
                if df.height == 0 or df.width == 0:
                    skipped.append({"dataset_id": dataset_id, "file": str(file_path), "reason": "empty_after_processing"})
                    continue

                category, categories, low_conf, suggested_new = self._infer_categories(dataset_id, df.columns)
                category = self._choose_primary_category(categories, per_category)
                categories = [category] + [c for c in categories if c != category]
                output_path = self.compressor.save(df, dataset_id, category=category)
                spec, target_reason = self._build_spec(dataset_id, df, category, categories)
                specs["datasets"][dataset_id] = spec

                meta = DatasetMeta(
                    id=dataset_id,
                    name=dataset_id,
                    source=str(source),
                    category=category,
                    categories=categories,
                    n_rows=df.height,
                    n_cols=df.width,
                    columns=df.columns,
                    processed=True,
                    file_path=str(output_path),
                )
                self.registry.register_dataset(meta)

                per_category[category] += 1
                processed.append(
                    {
                        "dataset_id": dataset_id,
                        "source_file": str(file_path.relative_to(self.base_path)),
                        "source": source,
                        "category": category,
                        "categories": categories,
                        "rows": df.height,
                        "cols": df.width,
                        "output_path": str(output_path.relative_to(self.base_path)),
                        "target_reason": target_reason,
                        "target_column": (spec.get("targets") or [{}])[0].get("name"),
                        "target_task_type": (spec.get("targets") or [{}])[0].get("task_type"),
                        "skip_training": bool(spec.get("skip_training", False)),
                    }
                )

                if low_conf:
                    entry = {
                        "dataset_id": dataset_id,
                        "assigned_category": category,
                        "source_file": str(file_path.relative_to(self.base_path)),
                        "suggested_new_category": suggested_new,
                    }
                    low_confidence.append(entry)
                    if suggested_new:
                        suggested_new_categories[suggested_new] += 1
            except Exception as exc:
                skipped.append({"dataset_id": dataset_id, "file": str(file_path), "reason": f"error:{exc}"})

        with open(self.spec_path, "w") as f:
            json.dump(specs, f, indent=2)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "industry": self.industry,
            "raw_file_count": len(discovered),
            "processed_dataset_count": len(processed),
            "skipped_count": len(skipped),
            "category_counts": dict(sorted(per_category.items())),
            "low_confidence_assignments": low_confidence,
            "suggested_new_categories": dict(suggested_new_categories),
            "processed_datasets": processed,
            "skipped_datasets": skipped,
            "paths": {
                "spec_path": str(self.spec_path.relative_to(self.base_path)),
                "dataset_metadata_path": str((self.base_path / "data" / "metadata" / self.industry / f"{self.industry}_datasets.json").relative_to(self.base_path)),
            },
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process raw datasets for an industry.")
    parser.add_argument("--base", default=".", help="Project root path.")
    parser.add_argument("--industry", required=True, help="Industry name, e.g. ecommerce.")
    parser.add_argument("--max-rows", type=int, default=250000, help="Maximum rows to keep per dataset.")
    parser.add_argument(
        "--clean-processed",
        action="store_true",
        help="Delete existing processed datasets and reset dataset metadata for this industry first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    runner = RawIndustryProcessor(base_path=base, industry=args.industry, max_rows=args.max_rows)
    report = runner.run(clean_processed=args.clean_processed)
    print(json.dumps(
        {
            "industry": report["industry"],
            "raw_file_count": report["raw_file_count"],
            "processed_dataset_count": report["processed_dataset_count"],
            "skipped_count": report["skipped_count"],
            "category_counts": report["category_counts"],
            "suggested_new_categories": report["suggested_new_categories"],
            "report_path": str(Path("reports") / f"{args.industry}_processing_report.json"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
