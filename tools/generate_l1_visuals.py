#!/usr/bin/env python3
"""Generate L1 model visuals for selected industries."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    r2_score,
)
from sklearn.preprocessing import label_binarize

from src.models.l1_experts.trainer import L1Trainer
from src.models.l1_experts.ensemble import CategoryDataPreparer

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def latest_run_dir(base: Path, industry: str) -> Path:
    runs_root = base / "reports" / "runs" / industry
    runs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        summary_path = run / "reports" / "summary.json"
        if summary_path.exists():
            return run
    raise FileNotFoundError(f"No completed runs found for {industry}")


def benchmark_thresholds(base: Path, industry: str) -> Dict[str, Any]:
    cfg_path = base / "config" / "industries" / industry / "industry.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("benchmark_policy", {}) or {}


def required_threshold(policy: Dict[str, Any], category: str, task_type: str) -> float:
    default_f1 = float(policy.get("default_classification_min_f1", 0.80))
    default_r2 = float(policy.get("default_regression_min_r2", 0.80))
    category_overrides = policy.get("category_overrides", {}) or {}
    cat_cfg = category_overrides.get(category, {}) or {}
    if task_type == "classification":
        return float(cat_cfg.get("classification_min_f1", default_f1))
    return float(cat_cfg.get("regression_min_r2", default_r2))


def metric_value(entry: Dict[str, Any]) -> float:
    task = entry.get("task_type")
    metrics = entry.get("metrics") or {}
    if task == "classification":
        return float(metrics.get("f1_score", 0.0))
    return float(metrics.get("r2", 0.0))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ema(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def plot_overview(industry: str, l1_results: Dict[str, Any], policy: Dict[str, Any], out_dir: Path) -> None:
    categories = sorted(l1_results.keys())
    values = []
    thresholds = []
    for c in categories:
        e = l1_results[c]
        values.append(metric_value(e))
        thresholds.append(required_threshold(policy, c, e.get("task_type")))

    x = np.arange(len(categories))

    plt.figure(figsize=(12, 5))
    plt.bar(x, values, color="#2b8a3e", alpha=0.85, label="Observed")
    plt.plot(x, thresholds, color="#d9480f", marker="o", linewidth=2, label="Required threshold")
    plt.xticks(x, categories, rotation=35, ha="right")
    plt.ylabel("Primary metric (f1 or r2)")
    plt.title(f"{industry}: L1 category metrics vs thresholds")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "l1_metrics_vs_thresholds.png", dpi=160)
    plt.close()

    margins = [v - t for v, t in zip(values, thresholds)]
    colors = ["#2f9e44" if m >= 0 else "#e03131" for m in margins]
    plt.figure(figsize=(12, 5))
    plt.bar(x, margins, color=colors, alpha=0.9)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, categories, rotation=35, ha="right")
    plt.ylabel("Margin above threshold")
    plt.title(f"{industry}: L1 benchmark margin by category")
    plt.tight_layout()
    plt.savefig(out_dir / "l1_benchmark_margins.png", dpi=160)
    plt.close()


def sample_data(X: np.ndarray, y: np.ndarray, max_points: int, classification: bool, rng: np.random.Generator):
    n = len(y)
    if n <= max_points:
        return X, y
    if classification:
        idx_all = np.arange(n)
        keep = []
        labels, counts = np.unique(y, return_counts=True)
        for lbl, cnt in zip(labels, counts):
            lbl_idx = idx_all[y == lbl]
            take = max(1, int(round(max_points * (cnt / n))))
            take = min(take, len(lbl_idx))
            choose = rng.choice(lbl_idx, size=take, replace=False)
            keep.append(choose)
        keep_idx = np.concatenate(keep)
        if len(keep_idx) > max_points:
            keep_idx = rng.choice(keep_idx, size=max_points, replace=False)
        return X[keep_idx], y[keep_idx]
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], y[idx]


def predict_classification(model, X: np.ndarray, device: torch.device, batch_size: int = 8192):
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i + batch_size]).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_all.append(probs)
    probs = np.vstack(probs_all)
    preds = probs.argmax(axis=1)
    return probs, preds


def predict_regression(model, X: np.ndarray, device: torch.device, batch_size: int = 8192):
    preds_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i + batch_size]).to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds_all.append(pred)
    return np.concatenate(preds_all)


def plot_classification_diagnostics(
    industry: str,
    category: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / category
    ensure_dir(cat_dir)

    n_classes = int(probs.shape[1])
    classes_for_eval = np.arange(n_classes)
    y_bin = label_binarize(y_true, classes=classes_for_eval)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{industry}/{category}: ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(cat_dir / "roc_curve.png", dpi=160)
        plt.close()
    else:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_micro, tpr_micro, label=f"micro ROC AUC={roc_auc_micro:.4f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{industry}/{category}: ROC (micro-average)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(cat_dir / "roc_curve_micro.png", dpi=160)
        plt.close()

    counts = np.bincount(y_true)
    top_classes = np.argsort(counts)[-min(20, len(counts)) :]
    mask = np.isin(y_true, top_classes)
    y_t = y_true[mask]
    y_p = preds[mask]
    labels = sorted(np.unique(np.concatenate([y_t, y_p])))
    cm = confusion_matrix(y_t, y_p, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Row-normalized")
    plt.title(f"{industry}/{category}: Confusion Matrix (top classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    tick_labels = [str(x) for x in labels]
    plt.xticks(np.arange(len(labels)), tick_labels, rotation=90)
    plt.yticks(np.arange(len(labels)), tick_labels)
    plt.tight_layout()
    plt.savefig(cat_dir / "confusion_matrix_top_classes.png", dpi=160)
    plt.close()


def plot_regression_diagnostics(
    industry: str,
    category: str,
    y_true: np.ndarray,
    preds: np.ndarray,
    out_dir: Path,
) -> None:
    cat_dir = out_dir / category
    ensure_dir(cat_dir)

    r2 = r2_score(y_true, preds)
    minv = float(min(y_true.min(), preds.min()))
    maxv = float(max(y_true.max(), preds.max()))

    if len(y_true) > 20000:
        idx = np.random.default_rng(42).choice(len(y_true), size=20000, replace=False)
        y_plot = y_true[idx]
        p_plot = preds[idx]
    else:
        y_plot = y_true
        p_plot = preds

    plt.figure(figsize=(6, 6))
    plt.scatter(y_plot, p_plot, s=5, alpha=0.2)
    plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=1.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{industry}/{category}: Parity Plot (R2={r2:.4f})")
    plt.tight_layout()
    plt.savefig(cat_dir / "parity_plot.png", dpi=160)
    plt.close()

    residuals = y_true - preds
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=60, alpha=0.85, color="#1971c2")
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.title(f"{industry}/{category}: Residual Distribution")
    plt.tight_layout()
    plt.savefig(cat_dir / "residuals_hist.png", dpi=160)
    plt.close()


def plot_learning_curve(industry: str, category: str, history_path: Path, out_dir: Path) -> None:
    if not history_path.exists():
        return
    payload = load_json(history_path)
    history = payload.get("history") or []
    if not history:
        return

    epochs = np.array([int(h.get("epoch", i)) for i, h in enumerate(history)], dtype=int)
    train_loss = np.array([float(h.get("train_loss", np.nan)) for h in history], dtype=float)
    val_loss = np.array([float(h.get("val_loss", np.nan)) for h in history], dtype=float)
    lr = np.array([float(h.get("lr", np.nan)) for h in history], dtype=float)
    train_smooth = ema(train_loss)
    val_smooth = ema(val_loss)
    best_epoch = int(payload.get("best_epoch", int(np.nanargmin(val_loss))))

    cat_dir = out_dir / category
    ensure_dir(cat_dir)

    fig = plt.figure(figsize=(9.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(epochs, train_loss, linewidth=1.0, alpha=0.22, color="#1f77b4", label="train_loss_raw")
    ax1.plot(epochs, val_loss, linewidth=1.0, alpha=0.22, color="#ff7f0e", label="val_loss_raw")
    ax1.plot(epochs, train_smooth, linewidth=2.2, color="#1f77b4", label="train_loss_ema")
    ax1.plot(epochs, val_smooth, linewidth=2.2, color="#ff7f0e", label="val_loss_ema")
    ax1.axvline(best_epoch, linestyle="--", linewidth=1.2, color="black", label=f"best_epoch={best_epoch}")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{industry}/{category}: Epoch-by-epoch learning curve")
    ax1.grid(alpha=0.18)
    ax1.legend(loc="upper right", ncol=2, fontsize=9)

    ax2.plot(epochs, lr, linewidth=1.8, color="#2b8a3e", label="learning_rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper right", fontsize=9)

    plt.savefig(cat_dir / "learning_curve.png", dpi=170)
    plt.close()


def write_summary_md(out_dir: Path, industry: str, run_id: str, l1_results: Dict[str, Any], policy: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# {industry.title()} L1 Visual Summary")
    lines.append("")
    lines.append(f"- Run: `{run_id}`")
    lines.append(f"- Categories: `{len(l1_results)}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| category | task | metric | threshold | margin | passed |")
    lines.append("|---|---|---:|---:|---:|---|")
    for c in sorted(l1_results.keys()):
        e = l1_results[c]
        task = e.get("task_type")
        mv = metric_value(e)
        th = required_threshold(policy, c, task)
        lines.append(f"| {c} | {task} | {mv:.4f} | {th:.4f} | {mv-th:.4f} | {e.get('passed')} |")
    lines.append("")
    lines.append("## Generated visuals")
    lines.append("")
    lines.append("- `l1_metrics_vs_thresholds.png`")
    lines.append("- `l1_benchmark_margins.png`")
    lines.append("- `learning_curve.png` per category (when history is available)")
    lines.append("- per-category diagnostics under subfolders")
    (out_dir / "README.md").write_text("\n".join(lines))


def generate_for_industry(
    base: Path,
    industry: str,
    run_id: str | None,
    max_samples_per_worker: int,
    max_points_per_category: int,
    worker_prediction_batch_size: int,
    only_categories: list[str] | None = None,
) -> None:
    if run_id is None:
        run_dir = latest_run_dir(base, industry)
    else:
        run_dir = base / "reports" / "runs" / industry / run_id

    run_id = run_dir.name
    reports_dir = run_dir / "reports"
    l1_results = load_json(reports_dir / "l1_results.json")
    preflight_path = reports_dir / "l1_semantic_preflight.json"
    preflight = load_json(preflight_path) if preflight_path.exists() else {}
    preflight_categories = (preflight or {}).get("categories", {}) if isinstance(preflight, dict) else {}
    policy = benchmark_thresholds(base, industry)

    out_dir = base / "reports" / "visuals" / "l1" / industry / run_id
    ensure_dir(out_dir)

    plot_overview(industry, l1_results, policy, out_dir)

    trainer = L1Trainer(base, industry=industry)
    prep = CategoryDataPreparer(base, industry=industry)
    rng = np.random.default_rng(42)

    categories = sorted(l1_results.keys())
    if only_categories:
        allowed = set(only_categories)
        categories = [c for c in categories if c in allowed]

    for category in categories:
        entry = l1_results[category]
        task = entry.get("task_type")
        is_cls = task == "classification"
        allowed_dataset_ids = None
        cat_pf = preflight_categories.get(category) if isinstance(preflight_categories, dict) else None
        if isinstance(cat_pf, dict):
            selected = cat_pf.get("selected_dataset_ids") or []
            if selected:
                allowed_dataset_ids = set(selected)

        X, y, _meta = prep.prepare_training_data(
            category,
            max_samples_per_worker=max_samples_per_worker,
            worker_prediction_batch_size=worker_prediction_batch_size,
            allowed_dataset_ids=allowed_dataset_ids,
        )

        X, y = sample_data(X, y, max_points=max_points_per_category, classification=is_cls, rng=rng)
        X_train, X_val, y_train, y_val = trainer._split_data(
            X,
            y,
            task_type=task,
            val_split=0.2,
            random_state=42,
        )

        model, _metadata = trainer.load_expert(category)
        device = trainer.device
        history_path = base / "models" / industry / "L1" / f"l1_{category}_expert_history.json"
        plot_learning_curve(industry=industry, category=category, history_path=history_path, out_dir=out_dir)

        if is_cls:
            probs, preds = predict_classification(model, X_val, device=device)
            plot_classification_diagnostics(industry, category, y_val, probs, preds, out_dir)
        else:
            preds = predict_regression(model, X_val, device=device)
            plot_regression_diagnostics(industry, category, y_val, preds, out_dir)

    write_summary_md(out_dir, industry, run_id, l1_results, policy)


def main():
    parser = argparse.ArgumentParser(description="Generate L1 visuals for industries.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--industries", nargs="+", default=["aviation", "ecommerce"])
    parser.add_argument("--run-id", action="append", default=[], help="industry=run_id")
    parser.add_argument("--max-samples-per-worker", type=int, default=8000)
    parser.add_argument("--max-points-per-category", type=int, default=50000)
    parser.add_argument("--worker-prediction-batch-size", type=int, default=12000)
    parser.add_argument("--only-category", action="append", default=[], help="Repeatable category filter")
    args = parser.parse_args()

    run_map: Dict[str, str] = {}
    for item in args.run_id:
        if "=" not in item:
            continue
        ind, rid = item.split("=", 1)
        run_map[ind.strip()] = rid.strip()

    base = Path(args.base).resolve()
    for industry in args.industries:
        generate_for_industry(
            base=base,
            industry=industry,
            run_id=run_map.get(industry),
            max_samples_per_worker=args.max_samples_per_worker,
            max_points_per_category=args.max_points_per_category,
            worker_prediction_batch_size=args.worker_prediction_batch_size,
            only_categories=args.only_category or None,
        )
        print(f"Generated visuals for {industry}")


if __name__ == "__main__":
    main()
