"""
Industry-agnostic pipeline for dataset discovery, L0 training, and L1 experts.
"""

from pathlib import Path
from typing import Optional
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .dataset_inventory import DatasetInventory
from .target_selector import TargetSelector
from .run_manager import RunManager
from .logging import get_logger
from .registry import Registry, DatasetMeta
from .dataset_specs import DatasetSpecStore
from ..models.l0_workers.trainer import L0Trainer
from ..models.l1_experts.trainer import L1Trainer, L1TrainingConfig
from ..data.external_ingestor import ExternalIngestor


class IndustryPipeline:
    """Runs the end-to-end pipeline for any industry."""

    def __init__(self, base_path: Path, industry: str):
        self.base_path = Path(base_path)
        self.industry = industry
        self.registry = Registry(self.base_path, industry=self.industry)
        self.inventory = DatasetInventory(self.base_path)
        self.target_selector = TargetSelector()
        self.run_manager = RunManager(self.base_path)
        self.l0_trainer = L0Trainer(self.base_path, industry=self.industry)
        self.l1_trainer = L1Trainer(self.base_path, industry=self.industry)
        self.external_ingestor = ExternalIngestor(self.base_path, industry=self.industry)
        self.logger = get_logger("core.industry_pipeline")
        self.spec_store = DatasetSpecStore(self.base_path, industry)

        self.industry_config = self._load_yaml(self.base_path / "config" / "industries" / industry / "industry.yaml")
        self.categories_config = self._load_yaml(self.base_path / "config" / "industries" / industry / "categories.yaml")
        self.taxonomy_config = self._load_yaml(self.base_path / "config" / "industries" / industry / "taxonomy.yaml")
        training_cfg = self.industry_config.get("training", {}) or {}
        self.enable_large_category_parallel = self._as_bool(
            training_cfg.get("enable_large_category_parallel", False), default=False
        )
        self.enable_numeric_string_coercion = self._as_bool(
            training_cfg.get("enable_numeric_string_coercion", False), default=False
        )
        self.identifier_columns = {
            "id", "record_id", "row_id", "index", "uuid", "uid", "ident", "icao24",
        }

    def _load_yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        """Parse permissive bool-like config values in a deterministic way."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _resolve_l1_training_config(self, l1_method: str) -> L1TrainingConfig:
        """
        Build L1 training config from industry overrides.

        Supported key path in `industry.yaml`:
        - `training.l1_config.<field>`
        """
        training_cfg = self.industry_config.get("training", {}) or {}
        overrides = dict(training_cfg.get("l1_config", {}) or {})
        allowed = set(L1TrainingConfig.__dataclass_fields__.keys())
        payload = {}
        for key, value in overrides.items():
            if key in allowed:
                payload[key] = value
            else:
                self.logger.warning("Ignoring unknown L1 config key: %s", key)
        payload["method"] = l1_method
        return L1TrainingConfig(**payload)

    @staticmethod
    def _normalize_semantic_token(value: Optional[str]) -> str:
        if value is None:
            return ""
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _resolve_l1_target_family(
        self,
        dataset_id: str,
        category: str,
        target_column: Optional[str],
        task_type: Optional[str],
    ) -> str:
        """
        Resolve semantic target family for L1 category aggregation.

        Priority:
        1. explicit dataset spec override (`training.l1_target_family` / `l1_target_family`)
        2. category-level taxonomy override (`categories.<id>.l1_target_family`)
        3. taxonomy subproblem match by canonical/fallback targets
        4. heuristic for high-signal fraud/chargeback labels
        5. fallback `unknown:<target>`
        """
        spec = self.spec_store.get(dataset_id) or {}
        training_cfg = spec.get("training", {}) or {}
        explicit_family = training_cfg.get("l1_target_family") or spec.get("l1_target_family")
        if explicit_family:
            return str(explicit_family)

        taxonomy_categories = (self.taxonomy_config or {}).get("categories", {}) or {}
        category_cfg = taxonomy_categories.get(category, {}) or {}
        category_family = category_cfg.get("l1_target_family")
        if category_family:
            return str(category_family)

        target_norm = self._normalize_semantic_token(target_column)
        task_norm = self._normalize_semantic_token(task_type)
        subproblems = category_cfg.get("subproblems", []) or []

        matched = []
        task_candidates = []
        for sub in subproblems:
            sub_id = sub.get("id")
            if not sub_id:
                continue
            sub_task_norm = self._normalize_semantic_token(sub.get("task_type"))
            if sub_task_norm and task_norm and sub_task_norm != task_norm:
                continue
            task_candidates.append(str(sub_id))
            vocab = []
            vocab.extend(sub.get("canonical_targets", []) or [])
            vocab.extend(sub.get("fallback_targets", []) or [])
            vocab_norm = {self._normalize_semantic_token(v) for v in vocab if v}
            if target_norm and target_norm in vocab_norm:
                matched.append(str(sub_id))

        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return f"ambiguous:{target_norm or 'none'}"

        heuristic_family = self._heuristic_target_family(
            category=category,
            target_norm=target_norm,
            task_norm=task_norm,
        )
        if heuristic_family:
            return heuristic_family

        if len(task_candidates) == 1:
            return task_candidates[0]

        return f"unknown:{target_norm or 'none'}"

    def _heuristic_target_family(self, category: str, target_norm: str, task_norm: str) -> Optional[str]:
        """
        Category-specific target-family heuristics when exact taxonomy target names are absent.
        """
        if not target_norm:
            return None

        if category == "customer":
            if any(k in target_norm for k in ("sentiment", "polarity")):
                return "sentiment_classification"
            if any(k in target_norm for k in ("rating", "recommend", "satisf", "review")):
                return "satisfaction_score"

        if category == "maintenance":
            if any(k in target_norm for k in ("rul", "remaining", "cycle", "life")):
                return "rul_prediction"
            if any(k in target_norm for k in ("failure", "fault", "maint", "target")):
                return "failure_risk"

        if category == "safety":
            if any(k in target_norm for k in ("incident", "accident", "severity", "risk", "fatal", "damage", "value")):
                return "incident_severity"

        if category == "pricing":
            if any(k in target_norm for k in ("fare", "price", "ticket", "yield", "revenue", "mile")):
                return "fare_prediction"

        if category == "network":
            if any(k in target_norm for k in ("hub", "role", "node")):
                return "hub_spoke_classification"
            if any(k in target_norm for k in ("route", "connect", "stop", "active", "hemi", "continent", "country", "region")):
                return "route_connectivity"

        if category == "operations":
            if any(k in target_norm for k in ("status", "delay", "ontime", "onground", "sched", "vis", "weather", "reliab")):
                return "schedule_reliability"
            return "resource_efficiency"

        if category == "checkout_risk":
            if any(k in target_norm for k in ("chargeback", "dispute", "refund")):
                return "chargeback_risk"
            if any(k in target_norm for k in ("fraud", "risk", "abuse", "amount", "price", "subscription")):
                return "payment_fraud"

        if category == "conversion_optimization":
            if any(k in target_norm for k in ("abandon", "dropoff")):
                return "checkout_abandonment"
            if any(k in target_norm for k in ("convert", "purchase", "quote", "class", "revenue", "unitprice", "target")):
                return "session_conversion"

        if category == "basket_intelligence":
            if any(k in target_norm for k in ("bundle", "department", "co", "affinity", "weekday", "evalset")):
                return "bundle_affinity"
            if any(k in target_norm for k in ("addon", "crosssell", "upsell", "cart")):
                return "cross_sell_acceptance"

        if category == "catalog_quality":
            if any(k in target_norm for k in ("quality", "recommend", "flow", "tier", "suppress", "discover")):
                return "listing_quality_risk"

        if category == "fulfillment_flow":
            if any(k in target_norm for k in ("exception", "overbill", "billing")):
                return "fulfillment_exception"
            if any(k in target_norm for k in ("delay", "late", "sla", "orderstatus", "ship")):
                return "late_delivery"

        if category == "demand_signal":
            if any(k in target_norm for k in ("order", "unit", "quantity", "demand", "volume", "sales", "gmv", "revenue", "price", "state", "country", "zone")):
                return "long_horizon_volume"

        if category == "merchandising":
            if any(k in target_norm for k in ("margin", "gmv", "revenue", "price", "sales", "gross", "qty", "retail", "discount")):
                return "sku_margin_prediction"

        return None

    def _build_l1_semantic_preflight(
        self,
        category_workers: dict[str, list[dict]],
        min_workers_per_category: int,
    ) -> dict:
        """
        Validate semantic consistency and choose one family per category.
        """
        report = {
            "industry": self.industry,
            "categories": {},
        }
        for category, workers in category_workers.items():
            task_counts: dict[str, int] = {}
            for worker in workers:
                t = str(worker.get("task_type") or "unknown")
                task_counts[t] = task_counts.get(t, 0) + 1
            dominant_task_type = None
            if task_counts:
                dominant_task_type = sorted(task_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

            family_map: dict[str, list[dict]] = {}
            expanded_workers = []
            for worker in workers:
                dataset_id = str(worker.get("dataset_id", ""))
                target_column = worker.get("target_column")
                task_type = worker.get("task_type")
                family = self._resolve_l1_target_family(
                    dataset_id=dataset_id,
                    category=category,
                    target_column=target_column,
                    task_type=task_type,
                )
                row = {
                    "dataset_id": dataset_id,
                    "target_column": target_column,
                    "task_type": task_type,
                    "target_family": family,
                }
                expanded_workers.append(row)
                family_map.setdefault(family, []).append(row)

            target_families = sorted([k for k, vals in family_map.items() if vals])
            dominant_family = None
            selected_workers = []
            if family_map:
                family_counts = sorted(
                    ((fam, len(rows)) for fam, rows in family_map.items()),
                    key=lambda x: (-x[1], x[0]),
                )
                dominant_family = family_counts[0][0]
                selected_workers = family_map.get(dominant_family, [])

            semantic_ok = len(selected_workers) >= min_workers_per_category
            blocking_reason = None
            if len(workers) == 0:
                blocking_reason = "no_workers_for_category"
            elif len(selected_workers) < min_workers_per_category:
                blocking_reason = "insufficient_same_family_workers"

            report["categories"][category] = {
                "semantic_ok": semantic_ok,
                "blocking_reason": blocking_reason,
                "dominant_task_type": dominant_task_type,
                "task_type_counts": task_counts,
                "dominant_target_family": dominant_family,
                "target_families": target_families,
                "workers_considered": expanded_workers,
                "workers_selected": selected_workers,
                "selected_dataset_ids": [w["dataset_id"] for w in selected_workers],
            }
        return report

    @staticmethod
    def _numeric_string_to_float_expr(col_expr: pl.Expr) -> pl.Expr:
        """
        Parse numeric-like strings while handling common locale formats.

        Supports:
        - Currency/symbol-wrapped values (e.g. "$1,234.56", "EUR 12.30")
        - Decimal comma values (e.g. "1,65")
        - EU mixed thousands/decimal (e.g. "1.234,56")
        - US mixed thousands/decimal (e.g. "1,234.56")
        """
        cleaned = (
            col_expr
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"[\u00A0\s]", "")
            .str.replace_all(r"[^0-9,.\-]", "")
        )
        has_comma = cleaned.str.contains(",")
        has_dot = cleaned.str.contains(r"\.")
        eu_mixed = cleaned.str.contains(r"^-?\d{1,3}(\.\d{3})+,\d+$")
        us_mixed = cleaned.str.contains(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
        comma_decimal = cleaned.str.contains(r"^-?\d+,\d{1,6}$")

        normalized = (
            pl.when(eu_mixed)
            .then(cleaned.str.replace_all(r"\.", "").str.replace_all(",", "."))
            .when(us_mixed)
            .then(cleaned.str.replace_all(",", ""))
            .when(has_comma & has_dot)
            .then(cleaned.str.replace_all(",", ""))
            .when(has_comma & comma_decimal)
            .then(cleaned.str.replace_all(",", "."))
            .when(has_comma)
            .then(cleaned.str.replace_all(",", ""))
            .otherwise(cleaned)
        )
        return normalized.cast(pl.Float64, strict=False)

    def _infer_categories(self, name: str, columns: list[str]) -> list[str]:
        overrides = self.industry_config.get("dataset_category_overrides", {})
        if name in overrides:
            override = overrides[name]
            if isinstance(override, list):
                return override
            return [override]

        categories = self.categories_config.get("categories", {})
        name_lower = name.lower()
        col_text = " ".join(columns).lower()
        search_text = f"{name_lower} {col_text}"

        scores = {}

        for cat_name, cat_config in categories.items():
            keywords = cat_config.get("keywords", [])
            scores[cat_name] = sum(1 for kw in keywords if kw in search_text)

        if not scores:
            return ["operations"]

        best_score = max(scores.values())
        if best_score == 0:
            return ["operations"]

        # Allow overlapping categories near the best score
        selected = [k for k, v in scores.items() if v >= max(1, int(0.8 * best_score))]
        if not selected:
            selected = ["operations"]
        return selected

    def _apply_worker_alias_overrides(self, entries: list[dict], worker_map: dict[str, str]) -> dict[str, str]:
        """
        Merge known dataset replicas to a single canonical worker.

        Dataset specs may define:
        - `worker_of`: "<canonical_dataset_id>" (top-level), or
        - `training.worker_of`: "<canonical_dataset_id>"
        """
        valid_ids = {entry["dataset_id"] for entry in entries}
        merged = dict(worker_map)
        for entry in entries:
            dataset_id = entry["dataset_id"]
            spec = self.spec_store.get(dataset_id) or {}
            training_cfg = spec.get("training", {}) or {}
            worker_of = training_cfg.get("worker_of") or spec.get("worker_of")
            if not worker_of:
                continue
            worker_of = str(worker_of)
            if worker_of not in valid_ids:
                self.logger.warning(
                    "Ignoring worker_of override for %s -> %s (target not present)",
                    dataset_id,
                    worker_of,
                )
                continue
            merged[dataset_id] = worker_of

        # Resolve chains so every dataset points directly to a stable canonical worker.
        for dataset_id in list(merged.keys()):
            seen = {dataset_id}
            current = merged.get(dataset_id, dataset_id)
            while current in merged and merged[current] != current and current not in seen:
                seen.add(current)
                current = merged[current]
            merged[dataset_id] = current

        return merged

    def _worker_target_name(self, dataset_id: str) -> Optional[str]:
        """Return primary target name from spec when available."""
        spec = self.spec_store.get(dataset_id) or {}
        targets = spec.get("targets", []) or []
        if targets:
            primary = next((t for t in targets if t.get("primary")), None)
            if primary and primary.get("name"):
                return str(primary["name"])
            first = targets[0]
            if first.get("name"):
                return str(first["name"])
        if spec.get("target"):
            return str(spec.get("target"))
        return None

    def _low_business_value_target_reason(
        self,
        target_column: str,
        task_type: str,
        df: pl.DataFrame,
        training_cfg: dict,
        industry_training_cfg: dict,
    ) -> Optional[str]:
        """
        Return a reason when target appears operationally weak/metadata-like.

        This protects against training on columns that are unlikely to encode
        meaningful business outcomes (for example `year`, `row_labels`).
        """
        guard_cfg = industry_training_cfg.get("target_guardrails", {}) or {}
        enabled = self._as_bool(guard_cfg.get("enabled", True), default=True)
        if not enabled:
            return None
        if self._as_bool(training_cfg.get("allow_low_value_targets", False), default=False):
            return None

        target_norm = str(target_column).strip().lower()
        allow = {str(v).strip().lower() for v in (guard_cfg.get("allow_exact") or []) if str(v).strip()}
        if target_norm in allow:
            return None

        default_low_exact = {
            "year", "period", "series_no", "row_labels", "price_base", "unit_mult",
            "decimal", "decimals", "ranking", "rank", "index",
        }
        low_exact = {
            str(v).strip().lower()
            for v in (guard_cfg.get("low_value_exact") or default_low_exact)
            if str(v).strip()
        }
        if target_norm in low_exact:
            return f"low_value_exact:{target_norm}"

        default_patterns = [
            r".*_ranking$",
            r"^rank_.*",
            r"^row_.*",
            r".*_label(s)?$",
            r"^label(s)?$",
        ]
        low_patterns = [str(p) for p in (guard_cfg.get("low_value_patterns") or default_patterns)]
        for pattern in low_patterns:
            try:
                if re.match(pattern, target_norm):
                    return f"low_value_pattern:{pattern}"
            except re.error:
                continue

        # Regression targets with near-zero variance are usually not trainable.
        if task_type == "regression" and target_column in df.columns:
            try:
                series = df[target_column].cast(pl.Float64, strict=False)
                if int(series.n_unique()) <= 2:
                    return "regression_low_unique_target"
            except Exception:
                pass
        return None

    def _candidate_feature_count(self, entry: dict, dataset_id: str) -> int:
        """Estimate non-target feature count from inventory columns + spec include/exclude."""
        columns = entry.get("columns", []) or []
        spec = self.spec_store.get(dataset_id) or {}
        features_cfg = spec.get("features", {}) or {}
        include = features_cfg.get("include") or []
        exclude = set(features_cfg.get("exclude", []) or [])
        target_name = self._worker_target_name(dataset_id)

        if include:
            candidate = [c for c in include if c in columns]
        else:
            candidate = list(columns)

        filtered = []
        for col in candidate:
            if target_name and str(col).lower() == str(target_name).lower():
                continue
            if col in exclude:
                continue
            filtered.append(col)
        return len(filtered)

    def _worker_eligibility(self, entry: dict) -> tuple[bool, str]:
        """
        Decide whether a unique dataset should participate as an L0 worker.

        This gate prevents structurally non-trainable tables from counting toward
        `expected_unique_workers` while keeping reasons auditable in run artifacts.
        """
        dataset_id = entry["dataset_id"]
        spec = self.spec_store.get(dataset_id) or {}
        training_cfg = spec.get("training", {}) or {}

        if not self._as_bool(training_cfg.get("worker_enabled", True), default=True):
            return (False, "worker_disabled_spec")

        industry_training_cfg = self.industry_config.get("training", {}) or {}
        gate_cfg = industry_training_cfg.get("worker_eligibility", {}) or {}
        if not self._as_bool(gate_cfg.get("enabled", True), default=True):
            return (True, "eligible")

        n_rows = int(entry.get("n_rows", 0))
        n_cols = int(entry.get("n_cols", 0))
        min_rows = int(gate_cfg.get("min_rows", 0))
        min_columns = int(gate_cfg.get("min_columns", 0))
        min_feature_columns = int(gate_cfg.get("min_feature_columns", 1))
        block_two_column_lookup = self._as_bool(gate_cfg.get("block_two_column_lookup", True), default=True)
        target_required = self._as_bool(gate_cfg.get("target_required", True), default=True)

        if target_required and self._worker_target_name(dataset_id) is None:
            return (False, "no_target_spec")
        if n_rows < min_rows:
            return (False, f"rows_below_min:{n_rows}<{min_rows}")
        if n_cols < min_columns:
            return (False, f"cols_below_min:{n_cols}<{min_columns}")
        if block_two_column_lookup and n_cols <= 2:
            return (False, "lookup_table_shape")

        candidate_features = self._candidate_feature_count(entry=entry, dataset_id=dataset_id)
        if candidate_features < min_feature_columns:
            return (False, f"feature_count_below_min:{candidate_features}<{min_feature_columns}")

        return (True, "eligible")

    def _prepare_features(self, df: pl.DataFrame, feature_columns: list[str], spec_features: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Basic feature preparation: encode categoricals, fill nulls, hash long text."""
        fe_metadata: dict = {"encodings": {}, "dropped_columns": [], "text_features": {}, "numeric_coercions": {}}
        n_rows = max(df.height, 1)
        if spec_features is None:
            spec_features = {}
        numeric_coercion_enabled = self.enable_numeric_string_coercion
        if "numeric_string_coercion" in spec_features:
            numeric_coercion_enabled = self._as_bool(
                spec_features.get("numeric_string_coercion"), default=numeric_coercion_enabled
            )
        one_hot_max_unique = int(spec_features.get("one_hot_max_unique", 50))
        high_cardinality_encoding = str(spec_features.get("high_cardinality_encoding", "ordinal")).lower()
        cat_cols = []
        low_card_cols = []
        text_matrices = []
        text_feature_count = 0
        force_one_hot = set(spec_features.get("force_one_hot", []))
        text_hint_cols = {
            "name", "keywords", "summary", "content", "description", "title",
            "review", "text", "comments", "metar",
        }
        for col in feature_columns:
            dtype = df[col].dtype
            if col in force_one_hot:
                low_card_cols.append(col)
                continue
            if dtype == pl.Utf8 or dtype == pl.String:
                col_lower = col.lower()
                if numeric_coercion_enabled:
                    # Coerce numeric-like strings (currency, comma-separated numbers, etc.) to floats.
                    candidate_expr = self._numeric_string_to_float_expr(pl.col(col))
                    candidate = df.select(candidate_expr.alias(col)).to_series()
                    non_null_count = max(1, int(df[col].is_not_null().sum()))
                    parsed_count = int(candidate.is_not_null().sum())
                    parse_ratio = parsed_count / float(non_null_count)
                    numeric_hint = any(
                        token in col_lower
                        for token in (
                            "price", "amount", "cost", "rate", "score", "rating",
                            "qty", "quantity", "count", "revenue", "sales",
                            "weight", "distance", "time", "duration", "age",
                            "latitude", "longitude", "lat", "lon",
                        )
                    )
                    if parse_ratio >= 0.80 and (numeric_hint or parsed_count >= 100):
                        df = df.with_columns(candidate_expr.alias(col))
                        fe_metadata["numeric_coercions"][col] = {
                            "method": "string_to_float",
                            "parse_ratio": parse_ratio,
                        }
                        continue

                # Drop extremely high-cardinality text columns
                try:
                    unique_count = df[col].n_unique()
                except Exception:
                    unique_count = n_rows
                try:
                    avg_len = df[col].cast(pl.Utf8).str.len_chars().mean()
                except Exception:
                    avg_len = 0
                is_text = (
                    col_lower in text_hint_cols
                    or unique_count > max(1000, int(0.5 * n_rows))
                    or (avg_len and avg_len > 30)
                )
                if is_text:
                    # Hash text into fixed-width features
                    texts = df[col].cast(pl.Utf8).fill_null("").to_list()
                    n_features = spec_features.get("text_features", 64)
                    vectorizer = HashingVectorizer(
                        n_features=n_features,
                        alternate_sign=False,
                        norm=None,
                    )
                    text_matrices.append(vectorizer.transform(texts))
                    fe_metadata["text_features"][col] = {"method": "hashing", "n_features": n_features}
                    fe_metadata["dropped_columns"].append(col)
                    text_feature_count += n_features
                    continue
                if unique_count <= one_hot_max_unique:
                    low_card_cols.append(col)
                else:
                    cat_cols.append(col)
            elif dtype == pl.Boolean:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif dtype == pl.Datetime:
                df = df.with_columns(pl.col(col).dt.timestamp("ms").alias(col))
            elif dtype == pl.Date:
                df = df.with_columns(pl.col(col).dt.epoch("ms").alias(col))

        for col in cat_cols:
            series = df[col].cast(pl.Utf8).fill_null("unknown")
            uniques = sorted(series.unique().to_list())
            if high_cardinality_encoding == "frequency":
                counts = series.value_counts().with_columns(
                    (pl.col("count") / float(n_rows)).alias("freq")
                )
                freq_map = {row[0]: float(row[2]) for row in counts.iter_rows()}
                fe_metadata["encodings"][col] = {
                    "method": "frequency",
                    "mapping": freq_map,
                }
                df = df.with_columns(series.replace(freq_map, default=0.0).cast(pl.Float64).alias(col))
            else:
                mapping = {val: idx + 1 for idx, val in enumerate(uniques)}
                fe_metadata["encodings"][col] = {
                    "method": "ordinal",
                    "mapping": mapping,
                }
                df = df.with_columns(series.replace(mapping, default=0).alias(col))

        if low_card_cols:
            df = df.with_columns([pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in low_card_cols])
            df = df.to_dummies(columns=low_card_cols)
            fe_metadata["one_hot"] = {c: "auto" for c in low_card_cols}

        if low_card_cols:
            base_cols = set(feature_columns) - set(low_card_cols)
            dummy_prefixes = tuple(f"{col}_" for col in low_card_cols)
            dummy_columns = [c for c in df.columns if c.startswith(dummy_prefixes)]
            kept_columns = [
                c for c in df.columns
                if c not in fe_metadata["dropped_columns"]
                and (c in base_cols or c.startswith(dummy_prefixes))
            ]
            fe_metadata["dummy_columns"] = dummy_columns
        else:
            kept_columns = [c for c in feature_columns if c not in fe_metadata["dropped_columns"]]
        fe_metadata["kept_columns"] = kept_columns
        numeric_df = df.select(kept_columns).fill_null(0) if kept_columns else pl.DataFrame()
        if numeric_df.width:
            numeric_df = numeric_df.with_columns(
                [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_df.columns]
            ).fill_null(0)

        if text_matrices:
            text_matrix = sparse.hstack(text_matrices).tocsr()
            fe_metadata["sparse"] = True
            if numeric_df.width == 0:
                X = text_matrix
            else:
                numeric_matrix = sparse.csr_matrix(numeric_df.to_numpy())
                X = sparse.hstack([numeric_matrix, text_matrix]).tocsr()
            fe_metadata["text_feature_count"] = text_feature_count
            return X, fe_metadata

        return numeric_df.to_numpy(), fe_metadata

    def _apply_spec_feature_engineering(self, df: pl.DataFrame, spec_features: Optional[dict]) -> pl.DataFrame:
        """Apply lightweight, spec-driven feature engineering operations."""
        if not spec_features:
            return df
        operations = spec_features.get("operations", []) or []
        if not operations:
            return df

        for op in operations:
            op_type = (op.get("type") or "").strip().lower()
            if op_type == "combine_str":
                cols = [c for c in (op.get("columns") or []) if c in df.columns]
                name = op.get("name")
                sep = op.get("sep", "_")
                if not cols or not name:
                    continue
                expr = pl.col(cols[0]).cast(pl.Utf8).fill_null("")
                for col in cols[1:]:
                    expr = expr + pl.lit(sep) + pl.col(col).cast(pl.Utf8).fill_null("")
                df = df.with_columns(expr.alias(name))
            elif op_type == "hhmm_to_hour_minute":
                source = op.get("source")
                hour_col = op.get("hour_col", f"{source}_hour")
                minute_col = op.get("minute_col", f"{source}_minute")
                if not source or source not in df.columns:
                    continue
                df = df.with_columns(pl.col(source).cast(pl.Int64, strict=False).fill_null(0).alias(source))
                df = df.with_columns(
                    (pl.col(source) // 100).clip(0, 23).alias(hour_col),
                    (pl.col(source) % 100).clip(0, 59).alias(minute_col),
                )
            elif op_type == "cyclic":
                source = op.get("source")
                period = float(op.get("period", 1.0))
                sin_col = op.get("sin_col", f"{source}_sin")
                cos_col = op.get("cos_col", f"{source}_cos")
                if not source or source not in df.columns or period <= 0:
                    continue
                source_expr = pl.col(source).cast(pl.Float64, strict=False).fill_null(0.0)
                df = df.with_columns(
                    (source_expr * (2.0 * np.pi / period)).sin().alias(sin_col),
                    (source_expr * (2.0 * np.pi / period)).cos().alias(cos_col),
                )
            elif op_type == "lookup_join":
                file_path = op.get("file")
                left_on = op.get("left_on")
                right_on = op.get("right_on")
                select = op.get("select", [])
                prefix = op.get("prefix", "")
                if (
                    not file_path
                    or not left_on
                    or not right_on
                    or left_on not in df.columns
                ):
                    continue
                lookup_path = (self.base_path / str(file_path)).resolve()
                if not str(lookup_path).startswith(str(self.base_path.resolve())):
                    self.logger.warning("Blocked path traversal in lookup_join: %s", file_path)
                    continue
                if not lookup_path.exists():
                    continue
                try:
                    lookup_df = pl.read_parquet(lookup_path) if str(lookup_path).endswith((".parquet", ".parquet.zstd")) else pl.read_csv(lookup_path)
                except Exception:
                    continue
                if right_on not in lookup_df.columns:
                    continue
                keep_cols = [right_on] + [c for c in select if c in lookup_df.columns and c != right_on]
                if len(keep_cols) <= 1:
                    continue
                lookup_df = lookup_df.select(keep_cols).unique(subset=[right_on], keep="first")
                rename_map = {c: f"{prefix}{c}" for c in keep_cols if c != right_on and prefix}
                if rename_map:
                    lookup_df = lookup_df.rename(rename_map)
                df = df.join(lookup_df, left_on=left_on, right_on=right_on, how="left")
            elif op_type == "haversine_km":
                lat1_col = op.get("lat1")
                lon1_col = op.get("lon1")
                lat2_col = op.get("lat2")
                lon2_col = op.get("lon2")
                out_col = op.get("name", "distance_km")
                if not all(c in df.columns for c in [lat1_col, lon1_col, lat2_col, lon2_col]):
                    continue
                # Great-circle distance based on joined airport coordinates.
                lat1 = pl.col(lat1_col).cast(pl.Float64, strict=False) * (np.pi / 180.0)
                lon1 = pl.col(lon1_col).cast(pl.Float64, strict=False) * (np.pi / 180.0)
                lat2 = pl.col(lat2_col).cast(pl.Float64, strict=False) * (np.pi / 180.0)
                lon2 = pl.col(lon2_col).cast(pl.Float64, strict=False) * (np.pi / 180.0)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = (dlat / 2.0).sin().pow(2.0) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2.0)
                c = 2.0 * a.sqrt().arcsin()
                earth_km = 6371.0
                df = df.with_columns((earth_km * c).alias(out_col))
            elif op_type == "binary_flag":
                source = op.get("source")
                out_col = op.get("name")
                values = set(str(v) for v in (op.get("values") or []))
                if not source or not out_col or source not in df.columns:
                    continue
                df = df.with_columns(
                    pl.col(source).cast(pl.Utf8).map_elements(
                        lambda v: 1 if (v is not None and str(v) in values) else 0,
                        return_dtype=pl.Int8,
                    ).alias(out_col)
                )
        return df

    def _remove_leaky_features(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
        target_column: str,
        spec: Optional[dict],
        category: Optional[str] = None,
    ) -> tuple[list[str], list[str]]:
        """
        Remove columns that can leak labels:
        - derived target source columns
        - ID-like columns with near-unique values
        - taxonomy-defined leakage guardrails
        """
        dropped: list[str] = []
        if not feature_columns:
            return [], dropped

        taxonomy_forbidden_exact: set[str] = set()
        taxonomy_forbidden_patterns: list[str] = []
        if category:
            taxonomy_categories = (self.taxonomy_config or {}).get("categories", {}) or {}
            category_cfg = taxonomy_categories.get(category, {}) or {}
            leakage_cfg = category_cfg.get("leakage_guardrails", {}) or {}
            taxonomy_forbidden_exact = {
                str(v).strip().lower()
                for v in (leakage_cfg.get("forbidden_columns") or [])
                if str(v).strip()
            }
            taxonomy_forbidden_patterns = [
                str(v).strip()
                for v in (leakage_cfg.get("conditional_forbidden_patterns") or [])
                if str(v).strip()
            ]

        # Exclude source column used to derive target.
        derived_source = None
        if spec is not None:
            derived = spec.get("derived_target") or {}
            derived_source = derived.get("source")
            if derived_source and derived_source not in df.columns:
                lookup = {c.lower(): c for c in df.columns}
                derived_source = lookup.get(str(derived_source).lower())

        cleaned = []
        n_rows = max(df.height, 1)
        for col in feature_columns:
            if col == target_column:
                dropped.append(col)
                continue
            if derived_source and col == derived_source:
                dropped.append(col)
                continue

            col_l = col.lower()
            if col_l in taxonomy_forbidden_exact:
                dropped.append(col)
                continue
            blocked_by_pattern = False
            for pattern in taxonomy_forbidden_patterns:
                try:
                    if re.match(pattern, col_l):
                        blocked_by_pattern = True
                        break
                except re.error:
                    continue
            if blocked_by_pattern:
                dropped.append(col)
                continue

            is_identifier_name = (
                col_l in self.identifier_columns
                or col_l.endswith("_id")
                or bool(re.match(r".*id\d*$", col_l))
            )
            if is_identifier_name and col in df.columns:
                try:
                    uniq_ratio = df[col].n_unique() / n_rows
                except Exception:
                    uniq_ratio = 1.0
                if uniq_ratio >= 0.95:
                    dropped.append(col)
                    continue

            cleaned.append(col)

        return cleaned, dropped

    def _batch_sample(
        self,
        df: pl.DataFrame,
        sample_rows: int,
        batch_size: int,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Sample large datasets in batches to avoid single-shot heavy sampling."""
        if sample_rows <= 0 or df.height <= sample_rows:
            return df
        if batch_size <= 0:
            return df.sample(n=sample_rows, seed=seed)

        n_rows = df.height
        n_batches = int(np.ceil(n_rows / float(batch_size)))
        per_batch = max(1, sample_rows // max(1, n_batches))
        sampled = []
        for i in range(n_batches):
            start = i * batch_size
            chunk = df.slice(start, batch_size)
            if chunk.height == 0:
                continue
            n_take = min(per_batch, chunk.height)
            sampled.append(chunk.sample(n=n_take, seed=seed + i))

        if not sampled:
            return df.sample(n=sample_rows, seed=seed)

        out = pl.concat(sampled, how="vertical")
        if out.height > sample_rows:
            out = out.sample(n=sample_rows, seed=seed + 1000)
        elif out.height < sample_rows:
            remaining = sample_rows - out.height
            if remaining > 0:
                extra = df.sample(n=min(remaining, df.height), seed=seed + 2000)
                out = pl.concat([out, extra], how="vertical")
                if out.height > sample_rows:
                    out = out.sample(n=sample_rows, seed=seed + 3000)
        return out

    def _apply_target_encoding(
        self,
        df: pl.DataFrame,
        target_column: str,
        te_cfg: Optional[dict],
        seed: int = 42,
    ) -> pl.DataFrame:
        """
        Add out-of-fold target-encoding features for high-cardinality categoricals.
        This is leakage-safe per row because each encoded value is computed from
        folds that exclude that row.
        """
        if not te_cfg:
            return df
        columns = [c for c in te_cfg.get("columns", []) if c in df.columns and c != target_column]
        if not columns:
            return df

        n_rows = df.height
        if n_rows < 100:
            return df

        y_series = df[target_column]
        try:
            y_unique = y_series.n_unique()
        except Exception:
            return df
        if y_unique != 2:
            return df

        # Binary target encoding value in [0,1]
        y_text = y_series.cast(pl.Utf8).to_list()
        labels = sorted(set(y_text))
        pos_label = te_cfg.get("positive_label")
        if pos_label is None or str(pos_label) not in labels:
            pos_label = labels[-1]
        y = np.array([1.0 if v == str(pos_label) else 0.0 for v in y_text], dtype=float)
        global_mean = float(np.mean(y))

        n_splits = int(te_cfg.get("n_splits", 5))
        n_splits = max(2, min(n_splits, 10))
        smoothing = float(te_cfg.get("smoothing", 20.0))
        min_count = int(te_cfg.get("min_count", 1))

        rng = np.random.default_rng(seed)
        order = np.arange(n_rows)
        rng.shuffle(order)
        folds = np.array_split(order, n_splits)

        def _encode_column(col_name: str) -> np.ndarray:
            values = df[col_name].cast(pl.Utf8).fill_null("unknown").to_numpy()
            encoded = np.full(n_rows, global_mean, dtype=float)
            for fold_idx in range(n_splits):
                valid_idx = folds[fold_idx]
                if len(valid_idx) == 0:
                    continue
                train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])
                if len(train_idx) == 0:
                    continue

                train_vals = values[train_idx]
                train_y = y[train_idx]
                sum_map: dict[str, float] = {}
                count_map: dict[str, int] = {}
                for key, target_val in zip(train_vals, train_y):
                    k = str(key)
                    sum_map[k] = sum_map.get(k, 0.0) + float(target_val)
                    count_map[k] = count_map.get(k, 0) + 1

                for idx in valid_idx:
                    k = str(values[idx])
                    cnt = count_map.get(k, 0)
                    if cnt < min_count:
                        encoded[idx] = global_mean
                    else:
                        smoothed = (sum_map[k] + smoothing * global_mean) / (cnt + smoothing)
                        encoded[idx] = float(smoothed)
            return encoded

        for col_name in columns:
            encoded = _encode_column(col_name)
            new_col = f"{col_name}_te"
            df = df.with_columns(pl.Series(name=new_col, values=encoded))

        return df

    def _train_l0_worker_entry(self, entry: dict, max_rows: int) -> dict:
        """Train a single L0 worker and return a normalized training outcome."""
        dataset_id = entry["dataset_id"]
        data_file = Path(entry["file_path"])
        if not data_file.exists():
            return {
                "dataset_id": dataset_id,
                "result": {"status": "missing_file"},
                "trained": False,
                "category": None,
                "passed": False,
            }

        try:
            df = (
                pl.read_parquet(data_file)
                if str(data_file).endswith(".parquet") or str(data_file).endswith(".parquet.zstd")
                else pl.read_csv(data_file)
            )
        except Exception as exc:
            self.logger.warning("Failed to load %s: %s", data_file, exc)
            return {
                "dataset_id": dataset_id,
                "result": {"status": "load_failed", "error": str(exc)},
                "trained": False,
                "category": None,
                "passed": False,
            }

        spec = self.spec_store.get(dataset_id)
        training_cfg = (spec or {}).get("training", {}) if spec is not None else {}
        industry_training_cfg = self.industry_config.get("training", {}) or {}
        is_gating_worker = bool(training_cfg.get("gating", True))
        dataset_max_rows = max_rows
        batch_size = 0
        if spec is not None:
            sampling_cfg = spec.get("sampling", {}) or {}
            spec_max_rows = sampling_cfg.get("max_rows")
            if spec_max_rows is not None:
                try:
                    dataset_max_rows = min(dataset_max_rows, int(spec_max_rows))
                except Exception:
                    pass
            try:
                batch_size = int(sampling_cfg.get("batch_size", 0))
            except Exception:
                batch_size = 0
        if df.height > dataset_max_rows:
            df = self._batch_sample(df=df, sample_rows=dataset_max_rows, batch_size=batch_size, seed=42)

        if spec is not None:
            df = self._apply_spec_feature_engineering(df, spec.get("features", {}))
        if spec is not None and spec.get("derived_target"):
            derived = spec["derived_target"]
            source = derived.get("source")
            name = derived.get("name")
            dtype = derived.get("type")
            derived_failed = False
            if source and name:
                if source not in df.columns:
                    lookup = {c.lower(): c for c in df.columns}
                    source = lookup.get(source.lower())
            if source in df.columns and name:
                if dtype == "median_split":
                    df = df.with_columns(self._numeric_string_to_float_expr(pl.col(source)).alias(source))
                    df = df.filter(pl.col(source).is_not_null())
                    if df.height > 0:
                        median = df[source].median()
                        df = df.with_columns(
                            pl.when(pl.col(source) >= median)
                            .then(pl.lit("high"))
                            .otherwise(pl.lit("low"))
                            .alias(name)
                        )
                    else:
                        derived_failed = True
                elif dtype == "length_median":
                    df = df.with_columns(pl.col(source).cast(pl.Utf8).alias(source))
                    df = df.filter(pl.col(source).is_not_null())
                    if df.height > 0:
                        lengths = df[source].str.len_chars()
                        median = lengths.median()
                        df = df.with_columns(
                            pl.when(lengths >= median)
                            .then(pl.lit("long"))
                            .otherwise(pl.lit("short"))
                            .alias(name)
                        )
                    else:
                        derived_failed = True
                elif dtype == "categorical_map":
                    mapping = derived.get("map", {})
                    drop_nulls = derived.get("drop_nulls", True)
                    if mapping:
                        df = df.with_columns(pl.col(source).cast(pl.Utf8).alias(source))
                        try:
                            mapped = pl.col(source).replace(mapping, default=None)
                        except Exception:
                            mapped = pl.col(source).map_elements(lambda v: mapping.get(v), return_dtype=pl.Utf8)
                        df = df.with_columns(mapped.alias(name))
                        if drop_nulls:
                            df = df.filter(pl.col(name).is_not_null())
                        if df.height == 0:
                            derived_failed = True
                    else:
                        derived_failed = True
                elif dtype == "threshold_split":
                    threshold = derived.get("threshold")
                    high_label = derived.get("high_label", "high")
                    low_label = derived.get("low_label", "low")
                    df = df.with_columns(self._numeric_string_to_float_expr(pl.col(source)).alias(source))
                    df = df.filter(pl.col(source).is_not_null())
                    if df.height > 0:
                        if threshold is None:
                            threshold = df[source].median()
                        df = df.with_columns(
                            pl.when(pl.col(source) >= float(threshold))
                            .then(pl.lit(high_label))
                            .otherwise(pl.lit(low_label))
                            .alias(name)
                        )
                    else:
                        derived_failed = True
                elif dtype == "keyword_polarity":
                    positive_keywords = [str(x).lower() for x in derived.get("positive_keywords", [])]
                    negative_keywords = [str(x).lower() for x in derived.get("negative_keywords", [])]
                    positive_label = str(derived.get("positive_label", "positive"))
                    negative_label = str(derived.get("negative_label", "negative"))
                    neutral_label = str(derived.get("neutral_label", "neutral"))
                    drop_neutral = bool(derived.get("drop_neutral", True))

                    if not positive_keywords and not negative_keywords:
                        derived_failed = True
                    else:
                        def _keyword_label(value):
                            text = "" if value is None else str(value).lower()
                            pos = sum(1 for kw in positive_keywords if kw and kw in text)
                            neg = sum(1 for kw in negative_keywords if kw and kw in text)
                            if pos > neg:
                                return positive_label
                            if neg > pos:
                                return negative_label
                            return neutral_label

                        df = df.with_columns(pl.col(source).map_elements(_keyword_label, return_dtype=pl.Utf8).alias(name))
                        if drop_neutral:
                            df = df.filter(pl.col(name) != neutral_label)
                        if df.height == 0:
                            derived_failed = True
            else:
                derived_failed = True

            if derived_failed:
                return {
                    "dataset_id": dataset_id,
                    "result": {"status": "derived_target_failed"},
                    "trained": False,
                    "category": None,
                    "passed": False,
                }
        if spec is not None and spec.get("skip_training"):
            return {
                "dataset_id": dataset_id,
                "result": {"status": "skipped_spec"},
                "trained": False,
                "category": None,
                "passed": False,
            }

        categories = self._infer_categories(dataset_id, df.columns) if spec is None else spec.get("categories", [spec.get("category", "operations")])
        category = categories[0]
        target_candidates = [c for c in df.columns if c in entry["profile"].has_target_candidates]
        selection = self.target_selector.select(df, target_candidates)

        # Support ordered multi-target specs (primary + alternates) per dataset.
        target_plan: list[dict] = []
        if spec is not None:
            spec_targets = spec.get("targets", []) or []
            if spec_targets:
                primary_targets = [t for t in spec_targets if t.get("primary")]
                alternate_targets = [t for t in spec_targets if not t.get("primary")]
                ordered_targets = primary_targets + alternate_targets
                for target_cfg in ordered_targets:
                    target_name = target_cfg.get("name")
                    if not target_name:
                        continue
                    target_plan.append(
                        {
                            "target_column": str(target_name),
                            "task_type": str(target_cfg.get("task_type", selection.task_type)),
                            "reason": "spec_target",
                        }
                    )

        if not target_plan and selection.target_column is not None:
            target_plan.append(
                {
                    "target_column": selection.target_column,
                    "task_type": selection.task_type,
                    "reason": "auto_target_selector",
                }
            )

        if not target_plan:
            return {
                "dataset_id": dataset_id,
                "result": {"status": "no_target"},
                "trained": False,
                "category": category,
                "passed": False,
            }

        allow_target_fallback = self._as_bool(
            training_cfg.get(
                "allow_target_fallback",
                industry_training_cfg.get("allow_target_fallback", len(target_plan) > 1),
            ),
            default=len(target_plan) > 1,
        )
        fallback_on_fail = self._as_bool(
            training_cfg.get(
                "fallback_on_fail",
                industry_training_cfg.get("fallback_on_fail", False),
            ),
            default=False,
        )
        if not allow_target_fallback and len(target_plan) > 1:
            target_plan = target_plan[:1]
        try:
            max_target_trials = int(
                training_cfg.get(
                    "max_target_trials",
                    industry_training_cfg.get("max_target_trials", len(target_plan)),
                )
            )
        except Exception:
            max_target_trials = len(target_plan)
        max_target_trials = max(1, min(max_target_trials, len(target_plan)))
        target_plan = target_plan[:max_target_trials]

        spec_features = spec.get("features", {}) if spec is not None else {}
        include = spec_features.get("include") if isinstance(spec_features, dict) else None
        exclude = (spec_features.get("exclude", []) if isinstance(spec_features, dict) else []) or []
        model_override = training_cfg.get("model_name")
        attempt_log: list[dict] = []
        last_failure: Optional[dict] = None

        for target_cfg in target_plan:
            target_column = target_cfg["target_column"]
            task_type = target_cfg["task_type"]
            df_target = df.clone()

            if target_column not in df_target.columns:
                lookup = {c.lower(): c for c in df_target.columns}
                resolved = lookup.get(str(target_column).lower())
                if resolved is None:
                    attempt_log.append(
                        {
                            "target_column": target_column,
                            "task_type": task_type,
                            "status": "missing_target_column",
                        }
                    )
                    last_failure = {
                        "dataset_id": dataset_id,
                        "result": {"status": "missing_target_column"},
                        "trained": False,
                        "category": category,
                        "passed": False,
                    }
                    continue
                target_column = resolved

            low_value_reason = self._low_business_value_target_reason(
                target_column=target_column,
                task_type=task_type,
                df=df_target,
                training_cfg=training_cfg,
                industry_training_cfg=industry_training_cfg,
            )
            if low_value_reason:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "rejected_low_business_value_target",
                        "reason": low_value_reason,
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {
                        "status": "rejected_low_business_value_target",
                        "target_column": target_column,
                        "task_type": task_type,
                        "reason": low_value_reason,
                    },
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue

            if task_type == "regression":
                target_dtype = df_target[target_column].dtype
                if target_dtype == pl.Utf8 or target_dtype == pl.String:
                    df_target = df_target.with_columns(
                        self._numeric_string_to_float_expr(pl.col(target_column)).alias(target_column)
                    )
                else:
                    df_target = df_target.with_columns(
                        pl.col(target_column).cast(pl.Float64, strict=False).alias(target_column)
                    )
                df_target = df_target.filter(pl.col(target_column).is_not_null())
            else:
                df_target = df_target.with_columns(pl.col(target_column).cast(pl.Utf8).alias(target_column))
                df_target = df_target.filter(
                    pl.col(target_column).is_not_null()
                    & (pl.col(target_column).str.strip_chars() != "")
                )

            if df_target.height == 0:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "no_rows_after_target_clean",
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {"status": "no_rows_after_target_clean"},
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue

            te_cfg = spec_features.get("target_encoding", {}) if isinstance(spec_features, dict) else {}
            if te_cfg:
                df_target = self._apply_target_encoding(
                    df=df_target,
                    target_column=target_column,
                    te_cfg=te_cfg,
                    seed=42,
                )

            if include:
                feature_columns = [c for c in include if c in df_target.columns]
            else:
                feature_columns = [c for c in df_target.columns if c != target_column]
            feature_columns = [c for c in feature_columns if c not in exclude and c != target_column]

            if not feature_columns:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "no_features",
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {"status": "no_features"},
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue

            feature_columns, dropped_leaky_columns = self._remove_leaky_features(
                df=df_target,
                feature_columns=feature_columns,
                target_column=target_column,
                spec=spec,
                category=category,
            )
            if target_column in feature_columns:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "target_in_features_blocked",
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {"status": "target_in_features_blocked"},
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue
            if not feature_columns:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "no_features_after_leak_filter",
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {"status": "no_features_after_leak_filter"},
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue

            X, fe_metadata = self._prepare_features(df_target, feature_columns, spec_features)
            if dropped_leaky_columns:
                fe_metadata["dropped_leaky_columns"] = dropped_leaky_columns
            if getattr(X, "shape", (0, 0))[0] == 0 or getattr(X, "shape", (0, 0))[1] == 0:
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "no_features",
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {"status": "no_features"},
                    "trained": False,
                    "category": category,
                    "passed": False,
                }
                continue
            y = df_target[target_column].to_numpy()

            try:
                result = self.l0_trainer.train(
                    dataset_id=dataset_id,
                    X=X,
                    y=y,
                    task_type=task_type,
                    category=category,
                    categories=categories,
                    model_name=model_override,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    data_file=str(data_file),
                    fe_metadata=fe_metadata,
                    training_options=training_cfg,
                )
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "trained",
                        "passed": bool(result.passed_benchmark),
                    }
                )
                final_payload = {
                    "dataset_id": dataset_id,
                    "result": {
                        "status": "trained",
                        "passed": result.passed_benchmark,
                        "gating_worker": is_gating_worker,
                        "metrics": result.metrics,
                        "model_id": result.model_id,
                        "target_column": target_column,
                        "task_type": task_type,
                        "target_attempts": attempt_log,
                    },
                    "trained": True,
                    "category": category,
                    "passed": bool(result.passed_benchmark),
                }
                if result.passed_benchmark:
                    return final_payload
                last_failure = final_payload
                if not fallback_on_fail:
                    return final_payload
            except Exception as exc:
                self.logger.exception("L0 training failed for %s (target=%s)", dataset_id, target_column)
                attempt_log.append(
                    {
                        "target_column": target_column,
                        "task_type": task_type,
                        "status": "train_failed",
                        "error": str(exc),
                    }
                )
                last_failure = {
                    "dataset_id": dataset_id,
                    "result": {
                        "status": "train_failed",
                        "error": str(exc),
                        "target_column": target_column,
                        "task_type": task_type,
                        "target_attempts": attempt_log,
                    },
                    "trained": False,
                    "category": category,
                    "passed": False,
                }

        if last_failure is not None:
            # Include all attempts in final failure payload for auditability.
            if isinstance(last_failure.get("result"), dict):
                last_failure["result"]["target_attempts"] = attempt_log
            return last_failure
        return {
            "dataset_id": dataset_id,
            "result": {"status": "no_target"},
            "trained": False,
            "category": category,
            "passed": False,
        }

    def run(
        self,
        min_workers: int = 30,
        min_workers_per_category: int = 3,
        min_l1_categories: int = 5,
        l1_method: str = "deep",
        max_rows: int = 200000,
        category_parallelism: Optional[dict[str, int]] = None,
    ) -> dict:
        """Run full industry pipeline and return run summary."""
        run_ctx = self.run_manager.start_run(self.industry)
        training_cfg = self.industry_config.get("training", {}) or {}

        external_datasets = self.industry_config.get("external_datasets", [])
        if external_datasets:
            self.external_ingestor.ingest_from_config(external_datasets)
        data_dirs = [self.base_path / "data" / "processed" / self.industry, self.base_path / "data" / "sampled" / self.industry]
        entries = self.inventory.build_inventory(data_dirs)
        worker_map = self.inventory.assign_worker_groups(entries)
        worker_map = self._apply_worker_alias_overrides(entries, worker_map)

        unique_worker_eligibility: dict[str, tuple[bool, str]] = {}
        ineligible_workers: list[dict] = []
        for entry in entries:
            dataset_id = entry["dataset_id"]
            if worker_map.get(dataset_id, dataset_id) != dataset_id:
                continue
            eligible, reason = self._worker_eligibility(entry)
            unique_worker_eligibility[dataset_id] = (eligible, reason)
            if not eligible:
                spec = self.spec_store.get(dataset_id) or {}
                categories = spec.get("categories")
                if not categories and spec.get("category"):
                    categories = [spec.get("category")]
                if not categories:
                    categories = self._infer_categories(dataset_id, entry["columns"])
                ineligible_workers.append(
                    {
                        "dataset_id": dataset_id,
                        "reason": reason,
                        "rows": int(entry.get("n_rows", 0)),
                        "cols": int(entry.get("n_cols", 0)),
                        "category": categories[0] if categories else "operations",
                    }
                )

        dataset_details = []
        for entry in entries:
            dataset_id = entry["dataset_id"]
            worker_dataset_id = worker_map.get(dataset_id, dataset_id)
            duplicate_of = None if dataset_id == worker_dataset_id else worker_dataset_id

            categories = self._infer_categories(dataset_id, entry["columns"])
            category = categories[0]

            meta = DatasetMeta(
                id=dataset_id,
                name=dataset_id,
                source="local",
                category=category,
                categories=categories,
                n_rows=entry["n_rows"],
                n_cols=entry["n_cols"],
                columns=entry["columns"],
                processed=True,
                file_path=entry["file_path"],
                fingerprint=entry["fingerprint"],
                duplicate_of=duplicate_of,
                worker_dataset_id=worker_dataset_id,
            )
            self.registry.register_dataset(meta)

            dataset_details.append({
                "id": dataset_id,
                "category": category,
                "categories": categories,
                "rows": entry["n_rows"],
                "cols": entry["n_cols"],
                "file_path": entry["file_path"],
                "worker_dataset_id": worker_dataset_id,
                "duplicate_of": duplicate_of,
                "worker_eligible": unique_worker_eligibility.get(worker_dataset_id, (True, "eligible"))[0],
                "worker_eligibility_reason": unique_worker_eligibility.get(worker_dataset_id, (True, "eligible"))[1],
            })

        self.run_manager.write_json(run_ctx.reports_dir / "dataset_details.json", {
            "industry": self.industry,
            "datasets": dataset_details,
        })
        self.run_manager.write_json(
            run_ctx.reports_dir / "worker_eligibility.json",
            {
                "industry": self.industry,
                "ineligible_workers": ineligible_workers,
                "ineligible_count": len(ineligible_workers),
            },
        )

        # L0 training
        l0_results = {}
        trained_workers = 0
        l0_category_counts = {}
        l0_category_passed_counts = {}
        l0_category_gating_counts = {}
        l0_category_gating_passed_counts = {}
        l0_category_passed_workers_all: dict[str, list[dict]] = {}
        l0_category_passed_workers_gating: dict[str, list[dict]] = {}
        categories_below_min_workers = []
        parallel_cfg = category_parallelism if category_parallelism is not None else (training_cfg.get("category_parallelism", {}) or {})
        if self.enable_large_category_parallel:
            try:
                large_category_parallel_threshold = int(training_cfg.get("parallel_threshold_workers", 25))
            except Exception:
                large_category_parallel_threshold = 25
            try:
                large_category_parallel_workers = int(training_cfg.get("parallel_max_workers", 4))
            except Exception:
                large_category_parallel_workers = 4
            if large_category_parallel_threshold < 1:
                large_category_parallel_threshold = 25
            if large_category_parallel_workers < 1:
                large_category_parallel_workers = 4
        else:
            large_category_parallel_threshold = 0
            large_category_parallel_workers = 1

        unique_worker_entries = []
        for entry in entries:
            dataset_id = entry["dataset_id"]
            if worker_map.get(dataset_id, dataset_id) == dataset_id:
                eligible, _reason = unique_worker_eligibility.get(dataset_id, (True, "eligible"))
                if eligible:
                    unique_worker_entries.append(entry)

        category_groups: dict[str, list[dict]] = {}
        category_order: list[str] = []
        for entry in unique_worker_entries:
            dataset_id = entry["dataset_id"]
            spec = self.spec_store.get(dataset_id) or {}
            categories = spec.get("categories")
            if not categories and spec.get("category"):
                categories = [spec.get("category")]
            if not categories:
                categories = self._infer_categories(dataset_id, entry["columns"])
            primary_category = categories[0] if categories else "operations"
            if primary_category not in category_groups:
                category_groups[primary_category] = []
                category_order.append(primary_category)
            category_groups[primary_category].append(entry)

        category_workers: dict[str, int] = {}
        for category in category_order:
            count = len(category_groups[category])
            worker_override = 0
            if isinstance(parallel_cfg, dict) and category in parallel_cfg:
                try:
                    worker_override = int(parallel_cfg.get(category, 0))
                except Exception:
                    worker_override = 0
            workers = worker_override if worker_override > 0 else 1
            if worker_override <= 0 and self.enable_large_category_parallel and count > large_category_parallel_threshold:
                workers = large_category_parallel_workers
            workers = max(1, min(workers, count))
            category_workers[category] = workers

        scheduling_parts = [
            f"{category}:count={len(category_groups[category])},workers={category_workers[category]}"
            for category in category_order
        ]
        self.logger.info("L0 scheduling | %s", "; ".join(scheduling_parts))

        def _apply_outcome(outcome: dict) -> None:
            nonlocal trained_workers
            dataset_id = outcome["dataset_id"]
            result = outcome["result"]
            category = outcome.get("category")
            gating_worker = bool(result.get("gating_worker", True))
            l0_results[dataset_id] = result
            if outcome.get("trained"):
                trained_workers += 1
                if category is not None:
                    l0_category_counts[category] = l0_category_counts.get(category, 0) + 1
                    if outcome.get("passed"):
                        l0_category_passed_counts[category] = l0_category_passed_counts.get(category, 0) + 1
                        worker_row = {
                            "dataset_id": dataset_id,
                            "target_column": result.get("target_column"),
                            "task_type": result.get("task_type"),
                            "gating_worker": gating_worker,
                        }
                        l0_category_passed_workers_all.setdefault(category, []).append(worker_row)
                    if gating_worker:
                        l0_category_gating_counts[category] = l0_category_gating_counts.get(category, 0) + 1
                        if outcome.get("passed"):
                            l0_category_gating_passed_counts[category] = l0_category_gating_passed_counts.get(category, 0) + 1
                            l0_category_passed_workers_gating.setdefault(category, []).append(worker_row)

        for category in category_order:
            category_entries = category_groups[category]
            workers = category_workers[category]
            if workers <= 1 or len(category_entries) <= 1:
                for entry in category_entries:
                    outcome = self._train_l0_worker_entry(entry=entry, max_rows=max_rows)
                    _apply_outcome(outcome)
                continue

            self.logger.info(
                "Training %s workers in parallel with max_workers=%s",
                category,
                workers,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(self._train_l0_worker_entry, entry, max_rows): entry["dataset_id"]
                    for entry in category_entries
                }
                for future in as_completed(future_map):
                    dataset_id = future_map[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        self.logger.exception("Parallel L0 training failed for %s", dataset_id)
                        outcome = {
                            "dataset_id": dataset_id,
                            "result": {"status": "train_failed", "error": str(exc)},
                            "trained": False,
                            "category": category,
                            "passed": False,
                        }
                    _apply_outcome(outcome)

        l0_passed = [r for r in l0_results.values() if r.get("status") == "trained" and r.get("passed")]
        l0_failed = [r for r in l0_results.values() if r.get("status") == "trained" and not r.get("passed")]
        l0_failed_gating = [r for r in l0_failed if r.get("gating_worker", True)]
        require_all_l0_pass_for_l1 = self._as_bool(
            training_cfg.get("l1_requires_all_l0_pass", False),
            default=False,
        )
        expected_workers = len(unique_worker_entries)
        untrained_workers = max(0, expected_workers - trained_workers)
        effective_failed_workers = l0_failed if require_all_l0_pass_for_l1 else l0_failed_gating

        l1_results = {}
        l1_passed = []
        l1_failed = []
        effective_category_passed_counts = (
            l0_category_passed_counts if require_all_l0_pass_for_l1 else l0_category_gating_passed_counts
        )
        effective_category_total_counts = (
            l0_category_counts if require_all_l0_pass_for_l1 else l0_category_gating_counts
        )
        l1_categories_eligible_pre_semantic = sorted(
            [
                category
                for category, count in effective_category_passed_counts.items()
                if count >= min_workers_per_category
            ]
        )
        categories_below_min_workers = sorted(
            [
                category
                for category in sorted(set(l0_category_counts.keys()) | set(effective_category_total_counts.keys()))
                if effective_category_passed_counts.get(category, 0) < min_workers_per_category
            ]
        )

        semantic_source_workers = (
            l0_category_passed_workers_all if require_all_l0_pass_for_l1 else l0_category_passed_workers_gating
        )
        l1_semantic_preflight = self._build_l1_semantic_preflight(
            semantic_source_workers,
            min_workers_per_category=min_workers_per_category,
        )
        semantic_blocked_categories = sorted(
            [
                category
                for category in l1_categories_eligible_pre_semantic
                if not (l1_semantic_preflight.get("categories", {}).get(category, {}) or {}).get("semantic_ok", False)
            ]
        )
        l1_categories_eligible = sorted(
            [category for category in l1_categories_eligible_pre_semantic if category not in set(semantic_blocked_categories)]
        )
        l1_category_worker_filters = {
            category: list((l1_semantic_preflight.get("categories", {}).get(category, {}) or {}).get("selected_dataset_ids", []))
            for category in l1_categories_eligible
        }
        if semantic_blocked_categories:
            self.logger.info(
                "Skipping L1 for semantic-mixed categories: %s",
                semantic_blocked_categories,
            )

        l0_gate_satisfied = (
            len(effective_failed_workers) == 0
            and trained_workers > 0
            and (not require_all_l0_pass_for_l1 or untrained_workers == 0)
        )

        if l0_gate_satisfied:
            l1_config = self._resolve_l1_training_config(l1_method=l1_method)
            l1_results = self.l1_trainer.train_all_experts(
                categories=l1_categories_eligible,
                config=l1_config,
                min_workers_per_category=min_workers_per_category,
                category_worker_filters=l1_category_worker_filters,
            )
            l1_passed = [r for r in l1_results.values() if r.passed_benchmark]
            l1_failed = [r for r in l1_results.values() if not r.passed_benchmark]
        else:
            self.logger.info("Skipping L1 training: L0 gate not satisfied")

        l1_category_trained = set(l1_results.keys())
        coverage_categories = sorted(
            set(l0_category_counts.keys())
            | set(l1_categories_eligible_pre_semantic)
            | l1_category_trained
        )
        category_coverage = {
            "industry": self.industry,
            "requirements": {
                "min_workers_per_category": min_workers_per_category,
            },
            "categories": {},
        }
        for category in coverage_categories:
            l0_total = l0_category_counts.get(category, 0)
            l0_pass = l0_category_passed_counts.get(category, 0)
            l0_gating_total = l0_category_gating_counts.get(category, 0)
            l0_gating_pass = l0_category_gating_passed_counts.get(category, 0)
            l1_result = l1_results.get(category)
            semantic_row = (l1_semantic_preflight.get("categories", {}).get(category, {}) or {})
            category_coverage["categories"][category] = {
                "l0_workers": l0_total,
                "l0_passed_workers": l0_pass,
                "l0_gating_workers": l0_gating_total,
                "l0_gating_passed_workers": l0_gating_pass,
                "l0_meets_min_workers": effective_category_passed_counts.get(category, 0) >= min_workers_per_category,
                "l1_eligible_pre_semantic": category in l1_categories_eligible_pre_semantic,
                "l1_eligible": category in l1_categories_eligible,
                "l1_trained": category in l1_category_trained,
                "l1_passed": bool(l1_result.passed_benchmark) if l1_result else False,
                "l1_semantic_ok": bool(semantic_row.get("semantic_ok", False)),
                "l1_semantic_blocking_reason": semantic_row.get("blocking_reason"),
                "l1_semantic_dominant_task_type": semantic_row.get("dominant_task_type"),
                "l1_target_family": semantic_row.get("dominant_target_family"),
                "l1_target_families": semantic_row.get("target_families", []),
                "l1_selected_workers": len(semantic_row.get("workers_selected", [])),
            }

        l0_worker_target_met = trained_workers >= min_workers
        l0_category_target_met = len(categories_below_min_workers) == 0 and len(effective_category_passed_counts) > 0
        l1_category_target_met = len(l1_categories_eligible) >= min_l1_categories
        l1_passed_categories = len(l1_passed)

        summary = {
            "industry": self.industry,
            "run_id": run_ctx.run_id,
            "unique_workers_trained": trained_workers,
            "expected_unique_workers": expected_workers,
            "untrained_workers": untrained_workers,
            "ineligible_workers": len(ineligible_workers),
            "min_workers_required": min_workers,
            "min_workers_per_category_required": min_workers_per_category,
            "min_l1_categories_required": min_l1_categories,
            "l0_passed": len(l0_passed),
            "l0_failed": len(effective_failed_workers),
            "l0_failed_total": len(l0_failed),
            "l1_passed": len(l1_passed),
            "l1_failed": len(l1_failed),
            "l1_categories_eligible_pre_semantic": len(l1_categories_eligible_pre_semantic),
            "l1_categories_eligible": len(l1_categories_eligible),
            "l1_categories_semantic_blocked": len(semantic_blocked_categories),
            "l1_semantic_blocked_categories": semantic_blocked_categories,
            "l1_categories_trained": len(l1_results),
            "l1_passed_categories": l1_passed_categories,
            "l1_requires_all_l0_pass": require_all_l0_pass_for_l1,
            "l0_all_passed": l0_gate_satisfied,
            "l1_all_passed": len(l1_failed) == 0 and len(l1_results) > 0,
            "l0_worker_target_met": l0_worker_target_met,
            "l0_category_target_met": l0_category_target_met,
            "l1_category_target_met": l1_category_target_met,
            "categories_below_min_workers": categories_below_min_workers,
            "industry_done": False,
        }

        summary["industry_done"] = (
            l0_worker_target_met and
            l0_category_target_met and
            summary["l0_all_passed"] and
            l1_category_target_met and
            summary["l1_all_passed"]
        )

        self.run_manager.write_json(run_ctx.reports_dir / "l0_results.json", l0_results)
        self.run_manager.write_json(run_ctx.reports_dir / "l1_results.json", {
            k: {
                "category": v.category,
                "task_type": v.task_type,
                "metrics": v.metrics,
                "passed": v.passed_benchmark,
                "model_path": v.model_path,
            } for k, v in l1_results.items()
        })
        self.run_manager.write_json(run_ctx.reports_dir / "l1_semantic_preflight.json", l1_semantic_preflight)
        self.run_manager.write_json(run_ctx.reports_dir / "category_coverage.json", category_coverage)
        self.run_manager.write_json(run_ctx.reports_dir / "summary.json", summary)

        return summary
