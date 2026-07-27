"""
Report visualization helpers for runtime backend reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


matplotlib.use("Agg")


# ── Modern Treasury design system ──────────────────────────────────────────
_MT = {
    "bg":       "#FAFAF8",   # warm white figure background
    "axes_bg":  "#FFFFFF",   # chart plot area
    "text":     "#0A0A0A",   # primary labels / titles
    "muted":    "#9A9A97",   # axis tick labels, secondary text
    "border":   "#E0DDD7",   # grid lines and spines
    "accent":   "#155E40",   # primary line / bar (forest green)
    "accent2":  "#0E7A4F",   # secondary series
    "accent3":  "#A16207",   # amber / 30-day forecast
    "accent4":  "#92400E",   # 90-day forecast / tertiary
    "blue":     "#1D4ED8",   # additional palette
    "violet":   "#7C3AED",   # additional palette
    "rose":     "#E11D48",   # additional palette
    "palette":  ["#155E40", "#0E7A4F", "#A16207", "#1D4ED8", "#7C3AED", "#E11D48", "#92400E"],
}


def _apply_mt_style(fig: Any, axes: Any) -> None:
    """Apply Modern Treasury design tokens to a matplotlib figure and one or more axes."""
    fig.patch.set_facecolor(_MT["bg"])
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in ax_list:
        if ax is None:
            continue
        ax.set_facecolor(_MT["bg"])
        ax.title.set_color(_MT["text"])
        ax.title.set_fontsize(13)
        ax.title.set_fontweight("semibold")
        ax.xaxis.label.set_color(_MT["muted"])
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_color(_MT["muted"])
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(colors=_MT["muted"], labelsize=9, length=0)
        # Clean look — only bottom and left spines, no box
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor(_MT["border"])
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_edgecolor(_MT["border"])
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(axis="y", color=_MT["border"], linewidth=0.4, alpha=0.5)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)


def _save_figure(fig: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _pretty_col(col: str) -> str:
    """Humanize a column name for chart labels."""
    if col == "__row_count__":
        return "Event Count"
    return col.replace("_", " ").title()


def _pretty_category(category: str) -> str:
    """Humanize a category name for chart titles."""
    return category.replace("_", " ").title()


def _chart_bullets(kind: str, category: str, value_col: str) -> list[str]:
    """Generate 2-3 contextual bullet points explaining a chart."""
    cat = _pretty_category(category)
    col = _pretty_col(value_col)

    if kind == "trend":
        return [
            f"Shows how {col} changed month-over-month for {cat}.",
            "Upward trends signal growth; downward trends flag potential issues.",
            "Compare peaks and troughs against known events or seasonality.",
        ]
    if kind == "seasonal":
        return [
            f"Reveals recurring patterns in {col} across months and quarters.",
            "Use this to plan for predictable demand swings or resource allocation.",
            "Flat bars suggest stable operations; high variance signals opportunity.",
        ]
    if kind == "forecast":
        return [
            f"Shows the directional outlook for {col} based on historical patterns.",
            "Dashed line = 30-day outlook; dotted = 90-day outlook. These are directional, not predictions.",
            "Actual outcomes depend on the actions you take — use this for planning.",
        ]
    if kind == "distribution":
        return [
            f"Shows how {col} values are spread across your dataset.",
            "A tight cluster means consistent operations; wide spread signals variability.",
            "Outliers on either tail may warrant investigation.",
        ]
    if kind == "quality":
        return [
            f"Shows data completeness for {cat} — longer bars mean more missing data.",
            "Columns with >50% missing data may reduce model accuracy.",
            "Consider providing these values via Tell Us or Toji Chat.",
        ]
    return [f"Visual analysis for {cat}."]


@dataclass
class VisualArtifact:
    name: str
    title: str
    kind: str
    uri: str
    meta: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "kind": self.kind,
            "uri": self.uri,
            "meta": self.meta,
        }


def _is_numeric_dtype(dtype: Any) -> bool:
    d = str(dtype)
    return any(token in d for token in ("Int", "UInt", "Float", "Decimal"))


def _date_like_column(df: pl.DataFrame) -> Optional[str]:
    for name, dtype in zip(df.columns, df.dtypes):
        d = str(dtype)
        if "Date" in d or "Datetime" in d:
            return name
    for name in df.columns:
        lower = name.lower()
        if any(token in lower for token in ("date", "time", "timestamp", "week", "month", "year")):
            return name
    return None


def _numeric_column(df: pl.DataFrame, exclude_columns: Optional[set[str]] = None) -> Optional[str]:
    preferred = []
    fallback = []
    excluded = set(exclude_columns or set())
    for name, dtype in zip(df.columns, df.dtypes):
        if not _is_numeric_dtype(dtype):
            continue
        if str(name or "").strip() == "":
            continue
        if name in excluded:
            continue
        lower = name.lower()
        if "path_step" in lower:
            continue
        if any(tok in lower for tok in ("id", "index", "code", "path_step", "step")):
            fallback.append(name)
        else:
            preferred.append(name)
    return (preferred or fallback or [None])[0]


def _series_frame(df: pl.DataFrame, time_col: str, value_col: str) -> Optional[pd.DataFrame]:
    local = df.select([pl.col(time_col), pl.col(value_col)]).rename({time_col: "t", value_col: "y"})
    try:
        # Parse date-like strings robustly, including timezone-bearing values.
        # Some Polars versions reject implicit timezone parsing with no time_zone.
        if "Datetime" not in str(local["t"].dtype) and "Date" not in str(local["t"].dtype):
            txt = pl.col("t").cast(pl.Utf8)
            local = local.with_columns(
                pl.coalesce(
                    [
                        txt.str.to_datetime(strict=False, time_zone="UTC"),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%Y-%m-%dT%H:%M:%S%z", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%Y-%m-%d %H:%M:%S%z", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%Y-%m-%dT%H:%M:%S", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%Y-%m-%d %H:%M:%S", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%Y-%m-%d", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%m/%d/%Y", strict=False),
                        txt.str.strptime(pl.Datetime(time_zone="UTC"), "%d/%m/%Y", strict=False),
                        txt.str.to_datetime(strict=False),
                        txt.str.to_date(strict=False).cast(pl.Datetime),
                    ]
                ).alias("t")
            )
        local = local.with_columns(pl.col("y").cast(pl.Float64, strict=False))
        local = local.drop_nulls(["t", "y"]).sort("t")
        if local.height < 6:
            return None
        monthly = (
            local.with_columns(pl.col("t").dt.truncate("1mo").alias("tm"))
            .group_by("tm")
            .agg(pl.col("y").mean().alias("y"))
            .sort("tm")
            .rename({"tm": "t"})
        )
        monthly = monthly.tail(12)
        if monthly.height < 6:
            return None
        return monthly.to_pandas()
    except Exception:
        # Caller can fallback to synthetic monthly axis instead of failing visuals.
        return None


def _series_frame_rowcount_fallback(df: pl.DataFrame, *, time_context: Optional[dict[str, Any]] = None) -> Optional[pd.DataFrame]:
    rows = int(df.height)
    if rows < 6:
        return None
    start, end, lookback_days = _parse_time_context(time_context)
    points = 24 if lookback_days <= 120 else 12
    points = max(6, min(points, rows))
    chunks = np.array_split(np.ones(rows, dtype=np.float64), points)
    y = np.array([float(np.sum(chunk)) for chunk in chunks if len(chunk)], dtype=np.float64)
    if len(y) < 6:
        return None
    idx = pd.date_range(start=start, end=end, periods=len(y))
    return pd.DataFrame({"t": idx, "y": y})


def _series_frame_numeric_fallback(df: pl.DataFrame, value_col: str) -> Optional[pd.DataFrame]:
    return _series_frame_numeric_fallback_with_context(df, value_col, time_context=None)


def _parse_time_context(time_context: Optional[dict[str, Any]]) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    default_lookback = 365
    if not isinstance(time_context, dict):
        start = now - pd.Timedelta(days=default_lookback)
        return start, now, default_lookback

    lookback = time_context.get("lookback_days")
    try:
        lookback_days = int(float(lookback)) if lookback is not None else default_lookback
    except Exception:
        lookback_days = default_lookback
    lookback_days = int(max(30, min(730, lookback_days)))

    start = now - pd.Timedelta(days=lookback_days)
    end = now

    start_raw = str(time_context.get("analysis_start_date") or "").strip()
    end_raw = str(time_context.get("analysis_end_date") or "").strip()
    if start_raw:
        try:
            start = pd.to_datetime(start_raw, utc=True).tz_convert(None)
        except Exception:
            pass
    if end_raw:
        try:
            end = pd.to_datetime(end_raw, utc=True).tz_convert(None)
        except Exception:
            pass
    if end <= start:
        end = now
        start = now - pd.Timedelta(days=lookback_days)
    return start, end, lookback_days


def _series_frame_numeric_fallback_with_context(
    df: pl.DataFrame,
    value_col: str,
    *,
    time_context: Optional[dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    values = (
        df.select(pl.col(value_col).cast(pl.Float64, strict=False).alias("v"))
        .drop_nulls("v")
        .tail(30000)["v"]
        .to_numpy()
    )
    if len(values) < 6:
        return None

    # Build up to 12 monthly points from numeric signal when no usable date axis exists.
    start, end, lookback_days = _parse_time_context(time_context)
    # Use monthly points for longer history, weekly points for shorter windows.
    if lookback_days <= 120:
        points = int(min(24, max(8, len(values))))
        freq = "W"
    else:
        points = int(min(12, max(6, len(values))))
        freq = "MS"
    chunks = np.array_split(values, points)
    means = np.array([float(np.mean(chunk)) for chunk in chunks if len(chunk)], dtype=np.float64)
    if len(means) < 6:
        return None
    periods = len(means)
    try:
        idx = pd.date_range(start=start, end=end, periods=periods) if freq == "W" else pd.date_range(end=end, periods=periods, freq="MS")
    except Exception:
        idx = pd.date_range(end=end, periods=periods, freq="D")
    # Ensure arrays are equal length — trim to shorter if mismatch
    n = min(len(idx), len(means))
    return pd.DataFrame({"t": idx[:n], "y": means[:n]})


def _plot_trend(pdf: pd.DataFrame, out_path: Path, value_col: str = "Value", category: str = "") -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_mt_style(fig, ax)
    ax.plot(pdf["t"], pdf["y"], color=_MT["accent"], linewidth=2.0, label="Observed", solid_capstyle="round")
    # Subtle area fill with secondary color
    ax.fill_between(pdf["t"], pdf["y"], alpha=0.10, color=_MT["accent2"])
    cat = _pretty_category(category) if category else "Operations"
    ax.set_title(f"{cat} — Monthly Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel(_pretty_col(value_col))
    legend = ax.legend(loc="best", frameon=False)
    for text in legend.get_texts():
        text.set_color(_MT["muted"])
    _save_figure(fig, out_path)


def _plot_seasonal(pdf: pd.DataFrame, out_path: Path, value_col: str = "Value", category: str = "") -> None:
    cat = _pretty_category(category) if category else "Operations"
    col_label = f"Avg {_pretty_col(value_col)}"
    try:
        month = pdf.copy()
        month["month"] = month["t"].dt.month
        month_agg = month.groupby("month", as_index=False)["y"].mean()

        quarter = pdf.copy()
        quarter["quarter"] = quarter["t"].dt.quarter
        quarter_agg = quarter.groupby("quarter", as_index=False)["y"].mean()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        _apply_mt_style(fig, axes)
        month_colors = [_MT["palette"][i % len(_MT["palette"])] for i in range(len(month_agg))]
        axes[0].bar(month_agg["month"].values, month_agg["y"].values, color=month_colors, width=0.6)
        axes[0].set_title(f"{cat} — Monthly")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel(col_label)

        quarter_colors = [_MT["palette"][i % len(_MT["palette"])] for i in range(len(quarter_agg))]
        axes[1].bar(quarter_agg["quarter"].values, quarter_agg["y"].values, color=quarter_colors, width=0.5)
        axes[1].set_title(f"{cat} — Quarterly")
        axes[1].set_xlabel("Quarter")
        axes[1].set_ylabel(col_label)
        _save_figure(fig, out_path)
    except Exception:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        _apply_mt_style(fig, ax)
        ax.bar(range(len(pdf)), pdf["y"].values, color=_MT["accent"], width=0.6)
        ax.set_title(f"{cat} — Seasonality")
        ax.set_xlabel("Period")
        ax.set_ylabel(col_label)
        _save_figure(fig, out_path)


def _plot_forecast(pdf: pd.DataFrame, out_path: Path, value_col: str = "Value", category: str = "") -> dict[str, Any]:
    cat = _pretty_category(category) if category else "Operations"
    n = len(pdf)
    horizon_30 = 30
    horizon_90 = 90
    x = np.arange(n, dtype=np.float64)
    y = pdf["y"].to_numpy(dtype=np.float64)

    if n >= 8:
        slope, intercept = np.polyfit(x, y, 1)
        xf_90 = np.arange(n + horizon_90, dtype=np.float64)
        yf_90 = slope * xf_90 + intercept
    else:
        xf_90 = np.arange(n + horizon_90, dtype=np.float64)
        yf_90 = np.full_like(xf_90, fill_value=float(np.mean(y)) if len(y) else 0.0)

    last_t = pd.to_datetime(pdf["t"].iloc[-1])
    future_dates_30 = pd.date_range(last_t + pd.Timedelta(days=1), periods=horizon_30, freq="D")
    future_dates_90 = pd.date_range(last_t + pd.Timedelta(days=1), periods=horizon_90, freq="D")

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_mt_style(fig, ax)
    ax.plot(pdf["t"], y, color=_MT["accent"], linewidth=2.0, label="Historical", solid_capstyle="round")
    ax.fill_between(pdf["t"], y, alpha=0.07, color=_MT["accent"])
    ax.plot(future_dates_30, yf_90[n:n + horizon_30], color=_MT["accent3"], linewidth=1.4, linestyle="--", label="30-day outlook", alpha=0.7)
    ax.plot(future_dates_90[horizon_30:], yf_90[n + horizon_30:], color=_MT["accent4"], linewidth=1.0, linestyle=":", alpha=0.5, label="90-day outlook")
    ax.set_title(f"{cat} — Directional Outlook")
    ax.set_xlabel("Date")
    ax.set_ylabel(_pretty_col(value_col))
    legend = ax.legend(loc="best", frameon=False)
    for text in legend.get_texts():
        text.set_color(_MT["muted"])
    _save_figure(fig, out_path)
    return {
        "horizon_days": horizon_90,
        "forecast_start": str(future_dates_30[0].date()) if len(future_dates_30) else None,
        "forecast_end": str(future_dates_90[-1].date()) if len(future_dates_90) else None,
    }


def _plot_missingness(df: pl.DataFrame, out_path: Path, category: str = "") -> None:
    cat = _pretty_category(category) if category else "Dataset"
    rows = max(1, int(df.height))
    names = []
    ratios = []
    for col in df.columns[:30]:
        names.append(_pretty_col(col))
        ratios.append(float(df[col].null_count()) / rows)
    order = np.argsort(ratios)[::-1]
    top_names = [names[i] for i in order[:20]]
    top_ratios = [ratios[i] for i in order[:20]]

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_mt_style(fig, ax)
    bar_colors = [_MT["accent"] if r < 0.5 else _MT["accent4"] for r in top_ratios[::-1]]
    ax.barh(top_names[::-1], top_ratios[::-1], color=bar_colors, height=0.6)
    ax.set_title(f"{cat} — Data Completeness")
    ax.set_xlabel("Missing Ratio  (0 = complete, 1 = all missing)")
    ax.set_ylabel("Column")
    ax.set_xlim(0, 1)
    _save_figure(fig, out_path)


def _plot_distribution(df: pl.DataFrame, value_col: str, out_path: Path, category: str = "") -> None:
    cat = _pretty_category(category) if category else "Dataset"
    col_label = _pretty_col(value_col)
    values = (
        df.select(pl.col(value_col).cast(pl.Float64, strict=False).alias("v"))
        .drop_nulls("v")
        .head(30000)["v"]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_mt_style(fig, ax)
    if len(values):
        from matplotlib.colors import LinearSegmentedColormap
        _grad_cmap = LinearSegmentedColormap.from_list("mt_grad", [_MT["accent"], _MT["accent2"]])
        n_bins = 30
        counts, bin_edges, patches = ax.hist(values, bins=n_bins, color=_MT["accent"], alpha=0.75, edgecolor=_MT["bg"], linewidth=0.8)
        for i, patch in enumerate(patches):
            patch.set_facecolor(_grad_cmap(i / max(1, len(patches) - 1)))
    ax.set_title(f"{cat} — Distribution of {col_label}")
    ax.set_xlabel(col_label)
    ax.set_ylabel("Frequency")
    _save_figure(fig, out_path)


def _plot_opportunity_analysis(
    opportunity: dict[str, Any],
    out_path: Path,
    *,
    category: str = "",
) -> Optional[dict[str, Any]]:
    if not isinstance(opportunity, dict):
        return None
    levers = opportunity.get("top_3_levers") or []
    if not isinstance(levers, list) or len(levers) == 0:
        return None

    lever_colors = [_MT["accent"], _MT["accent2"], _MT["blue"]]
    effort_markers = {"low": "▸", "medium": "▸▸", "high": "▸▸▸"}
    labels: list[str] = []
    impacts: list[str] = []
    efforts: list[str] = []
    for lev in levers[:3]:
        if not isinstance(lev, dict):
            continue
        labels.append(str(lev.get("lever") or f"Lever {len(labels)+1}")[:50])
        impacts.append(str(lev.get("potential_impact") or "").strip())
        efforts.append(str(lev.get("effort") or "medium").strip().lower())
    if not labels:
        return None

    # Parse numeric impact values for bar lengths (fallback to equal bars)
    import re as _re
    bar_values: list[float] = []
    for imp in impacts:
        nums = _re.findall(r'(\d+(?:\.\d+)?)', imp)
        bar_values.append(float(nums[-1]) if nums else 10.0)

    cat = _pretty_category(category) if category else "Operations"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_mt_style(fig, ax)
    y_pos = list(range(len(labels)))
    colors = [lever_colors[i % len(lever_colors)] for i in range(len(labels))]
    bars = ax.barh(y_pos, bar_values, color=colors, height=0.5, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(f"{cat} — Opportunity Analysis")
    ax.set_xlabel("Potential Impact")

    for i, (bar, impact, effort) in enumerate(zip(bars, impacts, efforts)):
        marker = effort_markers.get(effort, "▸▸")
        ax.text(
            bar.get_width() + max(bar_values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{impact}  [{marker} {effort}]",
            ha="left",
            va="center",
            fontsize=9,
            color=_MT["text"],
        )
    # Add breathing room on the right for labels
    ax.set_xlim(0, max(bar_values) * 1.6)
    _save_figure(fig, out_path)
    return {
        "metric": str(opportunity.get("metric") or "core_kpi"),
        "levers": len(labels),
    }


def generate_report_visuals(
    *,
    base_path: Path,
    report_id: str,
    industry: str,
    category: str,
    df: pl.DataFrame,
    time_context: Optional[dict[str, Any]] = None,
    benchmark_comparison: Optional[dict[str, Any]] = None,
    opportunity_analysis: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    out_dir = Path(base_path) / "exports" / "runtime_reports" / "assets" / report_id
    out_dir.mkdir(parents=True, exist_ok=True)

    visuals: list[VisualArtifact] = []

    # Always include data quality and distribution visuals.
    missing_path = out_dir / "missingness.png"
    _plot_missingness(df, missing_path, category=category)
    visuals.append(
        VisualArtifact(
            name="missingness",
            title=f"{_pretty_category(category)} — Data Completeness",
            kind="quality",
            uri=str(missing_path.relative_to(base_path)),
            meta={"rows": int(df.height), "cols": int(df.width), "bullets": _chart_bullets("quality", category, "")},
        )
    )

    time_col = _date_like_column(df)
    exclude = {time_col} if time_col else set()
    value_col = _numeric_column(df, exclude_columns=exclude)
    if value_col and value_col != "__row_count__":
        dist_path = out_dir / "distribution.png"
        _plot_distribution(df, value_col, dist_path, category=category)
        visuals.append(
            VisualArtifact(
                name="distribution",
                title=f"{_pretty_category(category)} — Distribution of {_pretty_col(value_col)}",
                kind="distribution",
                uri=str(dist_path.relative_to(base_path)),
                meta={"value_column": value_col, "bullets": _chart_bullets("distribution", category, value_col)},
            )
        )

    opp_data = opportunity_analysis or benchmark_comparison
    if opp_data and isinstance(opp_data, dict) and opp_data.get("top_3_levers"):
        opp_path = out_dir / "opportunity_analysis.png"
        opp_meta = _plot_opportunity_analysis(
            opp_data,
            opp_path,
            category=category,
        )
        if opp_meta:
            visuals.append(
                VisualArtifact(
                    name="opportunity_analysis",
                    title=f"{_pretty_category(category)} — Opportunity Analysis",
                    kind="opportunity",
                    uri=str(opp_path.relative_to(base_path)),
                    meta={
                        **opp_meta,
                        "bullets": [
                            "Shows the top improvement levers ranked by potential impact.",
                            "Effort indicators (low/medium/high) help prioritize quick wins vs. larger initiatives.",
                        ],
                    },
                )
            )

    series = None
    synthetic_time_axis = False
    if time_col and value_col:
        series = _series_frame(df, time_col, value_col)
    if series is None and time_col and not value_col:
        # Fallback to event-count trend when only date/time signal exists.
        temp = df.with_columns(pl.lit(1.0).alias("__row_count__"))
        series = _series_frame(temp, time_col, "__row_count__")
        if series is not None:
            value_col = "__row_count__"
    if series is None and value_col:
        series = _series_frame_numeric_fallback_with_context(df, value_col, time_context=time_context)
        synthetic_time_axis = series is not None
    if series is None:
        # Ultimate fallback: synthetic row-count trend from report context window.
        series = _series_frame_rowcount_fallback(df, time_context=time_context)
        synthetic_time_axis = series is not None
        if value_col is None:
            value_col = "__row_count__"

    if series is not None and len(series) >= 6 and value_col:
        trend_path = out_dir / "trend.png"
        _plot_trend(series, trend_path, value_col=value_col, category=category)
        visuals.append(
            VisualArtifact(
                name="trend",
                title=f"{_pretty_category(category)} — Monthly Trend",
                kind="timeseries",
                uri=str(trend_path.relative_to(base_path)),
                meta={
                    "time_column": time_col or "__synthetic_month_axis__",
                    "value_column": value_col,
                    "points": int(len(series)),
                    "synthetic_time_axis": bool(synthetic_time_axis),
                    "bullets": _chart_bullets("trend", category, value_col),
                },
            )
        )

        seasonal_path = out_dir / "seasonality.png"
        _plot_seasonal(series, seasonal_path, value_col=value_col, category=category)
        visuals.append(
            VisualArtifact(
                name="seasonality",
                title=f"{_pretty_category(category)} — Seasonality",
                kind="seasonal",
                uri=str(seasonal_path.relative_to(base_path)),
                meta={
                    "time_column": time_col or "__synthetic_month_axis__",
                    "value_column": value_col,
                    "points": int(len(series)),
                    "synthetic_time_axis": bool(synthetic_time_axis),
                    "bullets": _chart_bullets("seasonal", category, value_col),
                },
            )
        )

        forecast_path = out_dir / "forecast.png"
        forecast_meta = _plot_forecast(series, forecast_path, value_col=value_col, category=category)
        visuals.append(
            VisualArtifact(
                name="forecast",
                title=f"{_pretty_category(category)} — 30/90 Day Forecast",
                kind="forecast",
                uri=str(forecast_path.relative_to(base_path)),
                meta={
                    "time_column": time_col or "__synthetic_month_axis__",
                    "value_column": value_col,
                    "points": int(len(series)),
                    "synthetic_time_axis": bool(synthetic_time_axis),
                    "bullets": _chart_bullets("forecast", category, value_col),
                    **forecast_meta,
                },
            )
        )

    payload = [row.to_payload() for row in visuals]
    return payload
