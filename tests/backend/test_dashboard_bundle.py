from __future__ import annotations

import pandas as pd

from src.backend.dashboard_bundle import build_default_tables, sanitize_dashboard_bundle


def test_build_default_tables_caps_rows_and_outputs_required_shapes():
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=5000, freq="D"),
            "orders_count": list(range(5000)),
            "region": ["north", "south", "west", "east", "central"] * 1000,
        }
    )
    tables = build_default_tables(df, max_rows=1000)
    assert "main" in tables
    assert "kpis" in tables
    assert len((tables["main"] or {}).get("rows") or []) <= 1000
    assert len((tables["kpis"] or {}).get("rows") or []) == 1


def test_sanitize_dashboard_bundle_removes_external_url_and_clamps_fields():
    tables = build_default_tables(pd.DataFrame({"metric": [1, 2, 3]}), max_rows=500)
    raw = {
        "views": [
            {
                "id": "trend",
                "type": "vega-lite",
                "title": "Trend",
                "source_table": "main",
                "spec": {
                    "data": {"url": "https://bad.example/x.json"},
                    "mark": "line",
                    "encoding": {
                        "x": {"field": "t", "type": "temporal"},
                        "y": {"field": "metric", "type": "quantitative"},
                    },
                },
                "layout": {"grid": {"xs": {"x": 0, "y": 0, "w": 12, "h": 8}}},
            }
        ]
    }
    out = sanitize_dashboard_bundle(raw, default_tables=tables)
    spec = out["views"][0]["spec"]
    assert spec.get("data") == {"name": "main"}
    assert "url" not in (spec.get("data") or {})


def test_sanitize_dashboard_bundle_uses_defaults_when_views_missing():
    tables = build_default_tables(pd.DataFrame({"metric": [1, 2, 3]}), max_rows=500)
    out = sanitize_dashboard_bundle({}, default_tables=tables)
    assert len(out.get("views") or []) >= 1
