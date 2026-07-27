from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_VIEW_TYPES = {"vega-lite", "kpi", "table", "markdown"}
ALLOWED_MARK_TYPES = {"line", "area", "bar", "point", "rect", "rule", "text"}
ALLOWED_TRANSFORMS = {
    "filter",
    "aggregate",
    "window",
    "calculate",
    "timeUnit",
    "bin",
    "joinaggregate",
    "fold",
    "pivot",
    "stack",
}
DEFAULT_BUNDLE_VERSION = "1.0.0"
ID_RE = re.compile(r"^[a-z][a-z0-9_\-]*$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_primitive(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat().replace("+00:00", "Z")
    text = str(value)
    return text if len(text) <= 1000 else text[:1000]


def _uniform_downsample(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    if max_rows <= 1:
        return rows[:1]
    step = (len(rows) - 1) / float(max_rows - 1)
    idxs = [int(round(i * step)) for i in range(max_rows)]
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx in idxs:
        j = max(0, min(len(rows) - 1, idx))
        if j in seen:
            continue
        seen.add(j)
        out.append(rows[j])
    return out


def _infer_field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    name = str(series.name or "").lower()
    if any(tok in name for tok in ("date", "time", "timestamp", "ts", "datetime")):
        return "datetime"
    return "string"


def _coerce_datetime_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        valid = parsed.notna().mean() if len(parsed) else 0.0
        if valid >= 0.6:
            return parsed
    except Exception:
        pass
    return series


def _first_matching_column(df: pd.DataFrame, predicates: list[tuple[str, Any]]) -> str | None:
    for col in df.columns:
        s = df[col]
        name = str(col).lower()
        for ptype, pred in predicates:
            if ptype == "name" and pred(name):
                return str(col)
            if ptype == "dtype" and pred(s):
                return str(col)
    return None


def build_default_tables(df: pd.DataFrame, *, max_rows: int = 1000) -> dict[str, Any]:
    if df is None or df.empty:
        rows = [
            {"t": _now_iso(), "metric": 0.0, "category": "Baseline", "id": "row_1"},
            {"t": _now_iso(), "metric": 1.0, "category": "Baseline", "id": "row_2"},
        ]
        return {
            "main": {
                "schema": [
                    {"name": "t", "type": "datetime", "role": "time", "label": "Time"},
                    {"name": "metric", "type": "number", "role": "measure", "label": "Metric"},
                    {"name": "category", "type": "category", "role": "dimension", "label": "Category"},
                    {"name": "id", "type": "string", "role": "id", "label": "ID"},
                ],
                "rows": rows,
            },
            "kpis": {
                "schema": [
                    {"name": "baseline", "type": "number", "role": "measure"},
                    {"name": "current", "type": "number", "role": "measure"},
                    {"name": "delta", "type": "number", "role": "measure"},
                    {"name": "risk", "type": "number", "role": "probability"},
                ],
                "rows": [{"baseline": 0.0, "current": 0.0, "delta": 0.0, "risk": 0.2}],
            },
        }

    pdf = df.copy()
    for col in pdf.columns:
        pdf[col] = pdf[col].replace([np.inf, -np.inf], np.nan)

    time_col = _first_matching_column(
        pdf,
        [
            ("name", lambda n: any(tok in n for tok in ("date", "time", "timestamp", "datetime", "event"))),
            ("dtype", pd.api.types.is_datetime64_any_dtype),
        ],
    )
    if time_col is None:
        for col in pdf.columns:
            coerced = _coerce_datetime_column(pdf[col])
            if pd.api.types.is_datetime64_any_dtype(coerced):
                pdf[col] = coerced
                time_col = str(col)
                break

    numeric_cols = [str(c) for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]
    metric_col = None
    if numeric_cols:
        priority = [
            "demand", "volume", "orders", "revenue", "sales", "count", "qty", "churn", "risk", "rate", "score",
        ]
        for p in priority:
            hit = next((c for c in numeric_cols if p in c.lower()), None)
            if hit:
                metric_col = hit
                break
        if metric_col is None:
            metric_col = numeric_cols[0]

    category_col = _first_matching_column(
        pdf,
        [
            ("name", lambda n: any(tok in n for tok in ("category", "segment", "channel", "region", "type", "group"))),
            ("dtype", lambda s: pd.api.types.is_object_dtype(s) or isinstance(getattr(s, "dtype", None), pd.CategoricalDtype)),
        ],
    )

    id_col = _first_matching_column(
        pdf,
        [
            ("name", lambda n: n == "id" or n.endswith("_id") or "identifier" in n),
        ],
    )

    if time_col is None:
        pdf["__t"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(pdf), freq="D")
        time_col = "__t"
    else:
        pdf[time_col] = _coerce_datetime_column(pdf[time_col])
        if not pd.api.types.is_datetime64_any_dtype(pdf[time_col]):
            parsed = pd.to_datetime(pdf[time_col], errors="coerce", utc=True)
            pdf[time_col] = parsed.fillna(pd.Timestamp.utcnow())

    if metric_col is None:
        pdf["__metric"] = np.arange(1, len(pdf) + 1, dtype=float)
        metric_col = "__metric"
    else:
        pdf[metric_col] = pd.to_numeric(pdf[metric_col], errors="coerce").fillna(0.0)

    if category_col is None:
        pdf["__category"] = "All"
        category_col = "__category"
    else:
        pdf[category_col] = pdf[category_col].astype(str).replace({"nan": "Unknown", "None": "Unknown"})

    if id_col is None:
        pdf["__id"] = [f"row_{i+1}" for i in range(len(pdf))]
        id_col = "__id"
    else:
        pdf[id_col] = pdf[id_col].astype(str)

    main_rows: list[dict[str, Any]] = []
    for idx, row in pdf.iterrows():
        main_rows.append(
            {
                "t": _to_primitive(row[time_col]),
                "metric": float(pd.to_numeric(row[metric_col], errors="coerce") or 0.0),
                "category": str(row[category_col] if pd.notna(row[category_col]) else "Unknown"),
                "id": str(row[id_col] if pd.notna(row[id_col]) else f"row_{idx+1}"),
            }
        )

    max_rows = max(100, int(max_rows or 1000))
    main_rows = _uniform_downsample(main_rows, max_rows=max_rows)

    metric_values = np.array([float(r.get("metric") or 0.0) for r in main_rows], dtype=float)
    split = max(1, int(len(metric_values) * 0.2))
    baseline = float(np.nanmean(metric_values[:split])) if len(metric_values) else 0.0
    current = float(np.nanmean(metric_values[-split:])) if len(metric_values) else 0.0
    delta = ((current - baseline) / baseline) if baseline else 0.0
    risk = float(np.nanstd(metric_values) / (abs(np.nanmean(metric_values)) + 1e-9)) if len(metric_values) else 0.0
    risk = float(max(0.01, min(0.99, risk)))

    return {
        "main": {
            "schema": [
                {"name": "t", "type": "datetime", "role": "time", "label": "Time"},
                {"name": "metric", "type": "number", "role": "measure", "label": "Metric"},
                {"name": "category", "type": "category", "role": "dimension", "label": "Category"},
                {"name": "id", "type": "string", "role": "id", "label": "ID"},
            ],
            "rows": main_rows,
        },
        "kpis": {
            "schema": [
                {"name": "baseline", "type": "number", "role": "measure"},
                {"name": "current", "type": "number", "role": "measure"},
                {"name": "delta", "type": "number", "role": "measure"},
                {"name": "risk", "type": "number", "role": "probability"},
            ],
            "rows": [{"baseline": baseline, "current": current, "delta": delta, "risk": risk}],
        },
    }


def _default_views() -> list[dict[str, Any]]:
    return [
        {
            "id": "main_trend",
            "type": "vega-lite",
            "title": "Trend / Forecast",
            "subtitle": "Time movement of your primary metric",
            "source_table": "main",
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": "t", "type": "temporal", "title": "Time"},
                    "y": {"field": "metric", "type": "quantitative", "title": "Metric"},
                    "color": {"field": "category", "type": "nominal"},
                    "tooltip": [
                        {"field": "t", "type": "temporal"},
                        {"field": "category", "type": "nominal"},
                        {"field": "metric", "type": "quantitative"},
                    ],
                },
            },
            "layout": {"grid": {"xs": {"x": 0, "y": 0, "w": 12, "h": 8}}},
        },
        {
            "id": "breakdown",
            "type": "vega-lite",
            "title": "Category Breakdown",
            "source_table": "main",
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "transform": [{"aggregate": [{"op": "sum", "field": "metric", "as": "sum_metric"}], "groupby": ["category"]}],
                "mark": "bar",
                "encoding": {
                    "y": {"field": "category", "type": "nominal", "sort": "-x"},
                    "x": {"field": "sum_metric", "type": "quantitative", "title": "Total Metric"},
                    "tooltip": [
                        {"field": "category", "type": "nominal"},
                        {"field": "sum_metric", "type": "quantitative"},
                    ],
                },
            },
            "layout": {"grid": {"xs": {"x": 12, "y": 0, "w": 12, "h": 8}}},
        },
        {
            "id": "distribution",
            "type": "vega-lite",
            "title": "Distribution",
            "source_table": "main",
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "point",
                "encoding": {
                    "x": {"field": "metric", "type": "quantitative", "title": "Metric"},
                    "y": {"field": "category", "type": "nominal", "title": "Category"},
                    "tooltip": [
                        {"field": "id", "type": "nominal"},
                        {"field": "metric", "type": "quantitative"},
                    ],
                },
            },
            "layout": {"grid": {"xs": {"x": 0, "y": 8, "w": 12, "h": 8}}},
        },
        {
            "id": "benchmark_position",
            "type": "vega-lite",
            "title": "Benchmark Position",
            "source_table": "main",
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "transform": [
                    {"aggregate": [{"op": "mean", "field": "metric", "as": "avg_metric"}], "groupby": ["category"]}
                ],
                "mark": {"type": "bar", "cornerRadiusEnd": 2},
                "encoding": {
                    "y": {"field": "category", "type": "nominal", "sort": "-x"},
                    "x": {"field": "avg_metric", "type": "quantitative", "title": "Average Metric"},
                    "tooltip": [
                        {"field": "category", "type": "nominal"},
                        {"field": "avg_metric", "type": "quantitative"},
                    ],
                },
            },
            "layout": {"grid": {"xs": {"x": 12, "y": 8, "w": 12, "h": 8}}},
        },
    ]


def _sanitize_layout(layout: Any) -> dict[str, Any]:
    data = layout if isinstance(layout, dict) else {}
    grid = data.get("grid") if isinstance(data.get("grid"), dict) else {}
    xs = grid.get("xs") if isinstance(grid.get("xs"), dict) else {"x": 0, "y": 0, "w": 12, "h": 8}

    def _clamp_int(v: Any, low: int, high: int, default: int) -> int:
        try:
            val = int(v)
        except Exception:
            return default
        return max(low, min(high, val))

    out = {
        "grid": {
            "xs": {
                "x": _clamp_int(xs.get("x"), 0, 24, 0),
                "y": _clamp_int(xs.get("y"), 0, 500, 0),
                "w": _clamp_int(xs.get("w"), 1, 24, 12),
                "h": _clamp_int(xs.get("h"), 1, 40, 8),
            }
        },
        "min_height_px": _clamp_int(data.get("min_height_px"), 120, 1600, 280),
    }
    for bp in ("md", "lg"):
        bpv = grid.get(bp)
        if isinstance(bpv, dict):
            out["grid"][bp] = {
                "x": _clamp_int(bpv.get("x"), 0, 24, out["grid"]["xs"]["x"]),
                "y": _clamp_int(bpv.get("y"), 0, 500, out["grid"]["xs"]["y"]),
                "w": _clamp_int(bpv.get("w"), 1, 24, out["grid"]["xs"]["w"]),
                "h": _clamp_int(bpv.get("h"), 1, 40, out["grid"]["xs"]["h"]),
            }
    return out


def _is_external_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://") or text.startswith("//")


def _sanitize_vega_spec(spec: Any, *, source_table: str, allowed_fields: set[str]) -> dict[str, Any]:
    out = deepcopy(spec) if isinstance(spec, dict) else {}
    out["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"

    # Remove any external URL references recursively.
    def _strip_urls(node: Any) -> Any:
        if isinstance(node, dict):
            clean: dict[str, Any] = {}
            for key, val in node.items():
                if key == "url" and _is_external_url(val):
                    continue
                clean[key] = _strip_urls(val)
            return clean
        if isinstance(node, list):
            return [_strip_urls(v) for v in node]
        return node

    out = _strip_urls(out)
    out["data"] = {"name": source_table}
    out["width"] = "container"
    out["autosize"] = {"type": "fit", "contains": "padding"}

    mark = out.get("mark")
    if isinstance(mark, str):
        if mark not in ALLOWED_MARK_TYPES:
            out["mark"] = "line"
    elif isinstance(mark, dict):
        mtype = str(mark.get("type") or "line")
        if mtype not in ALLOWED_MARK_TYPES:
            mark["type"] = "line"
        out["mark"] = mark
    else:
        out["mark"] = "line"

    # Transform allowlist.
    transforms = out.get("transform")
    if isinstance(transforms, list):
        filtered: list[dict[str, Any]] = []
        for t in transforms:
            if not isinstance(t, dict):
                continue
            if any(k in t for k in ("lookup",)):
                continue
            if any(isinstance(t.get(k), dict) and _is_external_url((t.get(k) or {}).get("url")) for k in ("from", "data")):
                continue
            keys = [k for k in t.keys() if k in ALLOWED_TRANSFORMS]
            if not keys:
                continue
            filtered.append(t)
        out["transform"] = filtered
    elif "transform" in out:
        out.pop("transform", None)

    # Encoding field clamp.
    encoding = out.get("encoding")
    if isinstance(encoding, dict):
        clean_encoding: dict[str, Any] = {}
        for channel, cfg in encoding.items():
            if isinstance(cfg, dict):
                field = cfg.get("field")
                if field is not None and str(field) not in allowed_fields:
                    continue
                clean_encoding[channel] = cfg
            elif channel == "tooltip" and isinstance(cfg, list):
                tips = []
                for tip in cfg:
                    if not isinstance(tip, dict):
                        continue
                    field = tip.get("field")
                    if field is not None and str(field) not in allowed_fields:
                        continue
                    tips.append(tip)
                if tips:
                    clean_encoding[channel] = tips
        out["encoding"] = clean_encoding
    return out


def sanitize_dashboard_bundle(
    raw_bundle: dict[str, Any] | None,
    *,
    default_tables: dict[str, Any],
    max_tables: int = 20,
    max_rows_per_table: int = 50_000,
    max_bundle_bytes: int = 2_500_000,
) -> dict[str, Any]:
    src = raw_bundle if isinstance(raw_bundle, dict) else {}
    bundle: dict[str, Any] = {
        "bundle_version": DEFAULT_BUNDLE_VERSION,
        "generated_at": _now_iso(),
        "title": str(src.get("title") or "Interactive Analytics Dashboard")[:140],
        "description": str(src.get("description") or "")[:2000],
        "assumptions": [str(x)[:500] for x in (src.get("assumptions") or []) if str(x).strip()][:50],
        "tables": deepcopy(default_tables),
        "views": [],
        "controls": [],
        "layout": {
            "grid_columns": 12,
            "breakpoints": {"xs_max_px": 640, "md_max_px": 1024},
            "row_height_px": 32,
            "gap_px": 12,
        },
        "theme": {
            "mode": "light",
            "accent": "#155E40",
            "font_family": "Inter",
        },
    }

    # Sanitize layout if provided.
    if isinstance(src.get("layout"), dict):
        layout = src["layout"]
        try:
            bundle["layout"]["grid_columns"] = max(4, min(24, int(layout.get("grid_columns", 12))))
            bps = layout.get("breakpoints") if isinstance(layout.get("breakpoints"), dict) else {}
            bundle["layout"]["breakpoints"] = {
                "xs_max_px": max(320, min(900, int(bps.get("xs_max_px", 640)))),
                "md_max_px": max(641, min(1400, int(bps.get("md_max_px", 1024)))),
            }
            bundle["layout"]["row_height_px"] = max(20, min(60, int(layout.get("row_height_px", 32))))
            bundle["layout"]["gap_px"] = max(0, min(24, int(layout.get("gap_px", 12))))
        except Exception:
            pass

    if isinstance(src.get("theme"), dict):
        theme = src["theme"]
        mode = str(theme.get("mode") or "light").lower()
        if mode in {"light", "dark"}:
            bundle["theme"]["mode"] = mode
        accent = str(theme.get("accent") or "").strip()
        if re.match(r"^#[0-9A-Fa-f]{6}$", accent):
            bundle["theme"]["accent"] = accent
        font = str(theme.get("font_family") or "").strip()
        if font:
            bundle["theme"]["font_family"] = font[:80]

    main_schema = ((bundle.get("tables") or {}).get("main") or {}).get("schema") or []
    allowed_fields = {str(row.get("name")) for row in main_schema if isinstance(row, dict) and row.get("name")}

    raw_views = src.get("views") if isinstance(src.get("views"), list) else []
    if not raw_views:
        raw_views = _default_views()

    seen_ids: set[str] = set()
    for row in raw_views:
        if not isinstance(row, dict):
            continue
        view_type = str(row.get("type") or "").strip().lower()
        if view_type not in ALLOWED_VIEW_TYPES:
            continue
        vid = str(row.get("id") or "").strip().lower()
        if not ID_RE.match(vid):
            vid = f"view_{len(bundle['views'])+1}"
        if vid in seen_ids:
            vid = f"{vid}_{len(bundle['views'])+1}"
        seen_ids.add(vid)

        source_table = str(row.get("source_table") or "main").strip()
        if source_table not in bundle["tables"] and view_type in {"vega-lite", "kpi", "table"}:
            source_table = "main"

        clean_view: dict[str, Any] = {
            "id": vid,
            "type": view_type,
            "title": str(row.get("title") or vid.replace("_", " ").title())[:140],
            "layout": _sanitize_layout(row.get("layout")),
        }
        subtitle = str(row.get("subtitle") or "").strip()
        if subtitle:
            clean_view["subtitle"] = subtitle[:240]

        if view_type == "vega-lite":
            clean_view["source_table"] = source_table
            clean_view["spec"] = _sanitize_vega_spec(
                row.get("spec"),
                source_table=source_table,
                allowed_fields=allowed_fields,
            )
            interactions = row.get("interactions") if isinstance(row.get("interactions"), dict) else {}
            if interactions:
                out_interactions: dict[str, Any] = {}
                grp = str(interactions.get("crossfilter_group") or "").strip()
                if grp:
                    out_interactions["crossfilter_group"] = grp[:64]
                params = interactions.get("selection_params")
                if isinstance(params, list):
                    out_interactions["selection_params"] = [str(p)[:64] for p in params if str(p).strip()][:10]
                if out_interactions:
                    clean_view["interactions"] = out_interactions
        elif view_type == "kpi":
            clean_view["source_table"] = source_table
            kpi = row.get("kpi") if isinstance(row.get("kpi"), dict) else {}
            fields = kpi.get("fields") if isinstance(kpi.get("fields"), list) else []
            clean_fields: list[dict[str, Any]] = []
            for fld in fields:
                if not isinstance(fld, dict):
                    continue
                field = str(fld.get("field") or "").strip()
                if not field:
                    continue
                entry = {"field": field[:64]}
                label = str(fld.get("label") or "").strip()
                if label:
                    entry["label"] = label[:80]
                fmt = str(fld.get("format") or "").strip()
                if fmt:
                    entry["format"] = fmt[:32]
                delta = str(fld.get("delta_field") or "").strip()
                if delta:
                    entry["delta_field"] = delta[:64]
                clean_fields.append(entry)
            if not clean_fields:
                clean_fields = [
                    {"field": "current", "label": "Current", "format": ",.2f", "delta_field": "delta"},
                    {"field": "baseline", "label": "Baseline", "format": ",.2f"},
                    {"field": "risk", "label": "Risk", "format": ".2%"},
                ]
            clean_view["kpi"] = {"fields": clean_fields[:12]}
        elif view_type == "table":
            clean_view["source_table"] = source_table
            table_cfg = row.get("table") if isinstance(row.get("table"), dict) else {}
            cols = table_cfg.get("columns") if isinstance(table_cfg.get("columns"), list) else []
            cols = [str(c)[:64] for c in cols if str(c).strip()][:30]
            if not cols:
                cols = [str(c.get("name")) for c in (bundle["tables"].get(source_table, {}).get("schema") or []) if isinstance(c, dict) and c.get("name")][:10]
            table_out: dict[str, Any] = {"columns": cols}
            try:
                table_out["page_size"] = max(5, min(200, int(table_cfg.get("page_size", 25))))
            except Exception:
                table_out["page_size"] = 25
            sort = table_cfg.get("sort") if isinstance(table_cfg.get("sort"), dict) else {}
            by = str(sort.get("by") or "").strip()
            if by:
                table_out["sort"] = {
                    "by": by[:64],
                    "dir": "desc" if str(sort.get("dir") or "desc").lower() == "desc" else "asc",
                }
            clean_view["table"] = table_out
        else:  # markdown
            clean_view["markdown"] = str(row.get("markdown") or "")[:8000]

        bundle["views"].append(clean_view)
        if len(bundle["views"]) >= 30:
            break

    if not bundle["views"]:
        bundle["views"] = _default_views()

    # Controls sanitization.
    raw_controls = src.get("controls") if isinstance(src.get("controls"), list) else []
    for ctrl in raw_controls[:30]:
        if not isinstance(ctrl, dict):
            continue
        cid = str(ctrl.get("id") or "").strip().lower()
        ctype = str(ctrl.get("type") or "").strip()
        if not cid or not ID_RE.match(cid):
            continue
        if ctype not in {"slider", "select", "toggle", "dateRange"}:
            continue
        bind = ctrl.get("bind") if isinstance(ctrl.get("bind"), dict) else {}
        view_id = str(bind.get("view_id") or "").strip()
        param = str(bind.get("param") or "").strip()
        if not view_id or not param:
            continue
        clean_ctrl: dict[str, Any] = {
            "id": cid,
            "type": ctype,
            "label": str(ctrl.get("label") or cid).strip()[:100],
            "bind": {"view_id": view_id[:64], "param": param[:64]},
        }
        help_txt = str(ctrl.get("help") or "").strip()
        if help_txt:
            clean_ctrl["help"] = help_txt[:200]
        for key in ("default", "options", "min", "max", "step"):
            if key in ctrl:
                clean_ctrl[key] = ctrl[key]
        bundle["controls"].append(clean_ctrl)

    # Enforce table limits and primitive coercion.
    tables = bundle.get("tables") if isinstance(bundle.get("tables"), dict) else {}
    if len(tables) > max_tables:
        kept = list(tables.items())[:max_tables]
        tables = dict(kept)
        bundle["tables"] = tables
    for table_name, table in list(tables.items()):
        if not isinstance(table, dict):
            bundle["tables"].pop(table_name, None)
            continue
        schema = table.get("schema") if isinstance(table.get("schema"), list) else []
        table["schema"] = [s for s in schema if isinstance(s, dict)][:200]
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            clean_rows.append({str(k): _to_primitive(v) for k, v in row.items()})
        max_rows = max(100, min(200_000, int(max_rows_per_table)))
        if len(clean_rows) > max_rows:
            clean_rows = _uniform_downsample(clean_rows, max_rows=max_rows)
            bundle.setdefault("assumptions", []).append(
                f"Downsampled table '{table_name}' to {max_rows} rows for client performance."
            )
        table["rows"] = clean_rows

    # Enforce bundle max bytes by reducing main table rows.
    def _bundle_size(b: dict[str, Any]) -> int:
        try:
            return len(json.dumps(b, ensure_ascii=False).encode("utf-8"))
        except Exception:
            return 0

    current_size = _bundle_size(bundle)
    if current_size > max_bundle_bytes:
        main = ((bundle.get("tables") or {}).get("main") or {}).get("rows")
        if isinstance(main, list) and len(main) > 200:
            target = max(100, int(len(main) * 0.5))
            while current_size > max_bundle_bytes and len(main) > 100:
                main = _uniform_downsample(main, max_rows=target)
                bundle["tables"]["main"]["rows"] = main
                target = max(100, int(target * 0.75))
                current_size = _bundle_size(bundle)
            bundle.setdefault("assumptions", []).append("Main table rows were downsampled to fit payload limits.")

    return bundle
