#!/usr/bin/env python3
"""Plot epoch-by-epoch learning curves from saved L1 history JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def ema(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def plot_curve(history_path: Path, out_path: Path, title: str) -> None:
    payload = load_json(history_path)
    hist = payload.get('history') or []
    if not hist:
        return

    epochs = np.array([int(h.get('epoch', i)) for i, h in enumerate(hist)], dtype=int)
    train_loss = np.array([float(h.get('train_loss', np.nan)) for h in hist], dtype=float)
    val_loss = np.array([float(h.get('val_loss', np.nan)) for h in hist], dtype=float)
    lr = np.array([float(h.get('lr', np.nan)) for h in hist], dtype=float)
    best_epoch = int(payload.get('best_epoch', int(np.nanargmin(val_loss))))

    train_smooth = ema(train_loss)
    val_smooth = ema(val_loss)

    fig = plt.figure(figsize=(9.2, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(epochs, train_loss, linewidth=1.0, alpha=0.22, color='#1f77b4', label='train_loss_raw')
    ax1.plot(epochs, val_loss, linewidth=1.0, alpha=0.22, color='#ff7f0e', label='val_loss_raw')
    ax1.plot(epochs, train_smooth, linewidth=2.2, color='#1f77b4', label='train_loss_ema')
    ax1.plot(epochs, val_smooth, linewidth=2.2, color='#ff7f0e', label='val_loss_ema')
    ax1.axvline(best_epoch, linestyle='--', color='black', linewidth=1.2, label=f'best_epoch={best_epoch}')
    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.grid(alpha=0.18)
    ax1.legend(loc='upper right', ncol=2, fontsize=9)

    ax2.plot(epochs, lr, color='#2b8a3e', linewidth=1.8, label='learning_rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LR')
    ax2.grid(alpha=0.2)
    ax2.legend(loc='upper right', fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='.')
    parser.add_argument('--industry', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    industry = args.industry
    run_id = args.run_id

    histories = sorted((base / 'models' / industry / 'L1').glob('l1_*_expert_history.json'))
    out_root = base / 'reports' / 'visuals' / 'l1' / industry / run_id

    for hp in histories:
        name = hp.name
        category = name.removeprefix('l1_').removesuffix('_expert_history.json')
        out = out_root / category / 'learning_curve.png'
        plot_curve(hp, out, f"{industry}/{category}: Epoch-by-epoch learning curve")

    print(f'Wrote curves: {len(histories)} for {industry}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
