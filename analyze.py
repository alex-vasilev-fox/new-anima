"""
Analyze and plot results.

Reads the JSON files in results/ and produces:
  - results/summary.txt    -- human-readable summary tables
  - results/fig_pmatch.png  -- boxplot of pmatch open vs closed
  - results/fig_lyapunov.png
  - results/fig_baseline.png
  - results/fig_sweep.png

Usage:
    python analyze.py

This runs whatever JSON files happen to be in results/. If only some
experiments are done, only those plots are produced.

Author: Alexander Vasiliev
"""

import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
    print("matplotlib not installed, plotting will be skipped")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stats(values):
    if not values:
        return float('nan'), float('nan'), 0
    a = np.array(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float('nan'), float('nan'), 0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)


def extract_three_mode(results: dict, mode: str, metric: str):
    """results = {seed: {modes: {baseline:{...}, open:{...}, closed:{...}}}}"""
    return [r['modes'][mode][metric] for r in results.values()]


def analyze_main(report_lines):
    main = load('results/main.json')
    if main is None:
        report_lines.append("[main] no results/main.json found")
        return

    pm_open  = extract_three_mode(main, 'open',   'pmatch_mean')
    pm_closed = extract_three_mode(main, 'closed', 'pmatch_mean')
    lat_open  = extract_three_mode(main, 'open',   'latent_acc')
    lat_closed = extract_three_mode(main, 'closed', 'latent_acc')
    lyap_baseline = extract_three_mode(main, 'baseline', 'lyapunov')
    lyap_open     = extract_three_mode(main, 'open',     'lyapunov')
    lyap_closed   = extract_three_mode(main, 'closed',   'lyapunov')

    diffs = [c - o for c, o in zip(pm_closed, pm_open)]
    n_positive = sum(1 for d in diffs if d > 0)

    report_lines.append("\n=== MAIN RESULT (10 seeds, both loops active) ===")
    report_lines.append(f"  pmatch open:    {stats(pm_open)[0]:.4f} ± {stats(pm_open)[1]:.4f}")
    report_lines.append(f"  pmatch closed:  {stats(pm_closed)[0]:.4f} ± {stats(pm_closed)[1]:.4f}")
    report_lines.append(f"  diff (closed - open): mean {np.mean(diffs):+.4f}, "
                        f"{n_positive}/{len(diffs)} seeds positive")
    report_lines.append(f"  latent_acc open:    {stats(lat_open)[0]:.4f}")
    report_lines.append(f"  latent_acc closed:  {stats(lat_closed)[0]:.4f}")
    report_lines.append(f"  lyapunov baseline:  {stats(lyap_baseline)[0]:.4f}")
    report_lines.append(f"  lyapunov open:      {stats(lyap_open)[0]:.4f}")
    report_lines.append(f"  lyapunov closed:    {stats(lyap_closed)[0]:.4f}")

    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([pm_open, pm_closed], labels=['Open loop', 'Closed loop'])
        ax.set_ylabel('Prediction match (cosine similarity)')
        ax.set_title(f'Prediction match across {len(pm_open)} seeds')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig('results/fig_pmatch.png', dpi=150)
        plt.close(fig)
        report_lines.append("  saved results/fig_pmatch.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot([lyap_baseline, lyap_open, lyap_closed],
                   labels=['Baseline', 'Open loop', 'Closed loop'])
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Estimated Lyapunov exponent')
        ax.set_title('Chaos preservation across modes')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig('results/fig_lyapunov.png', dpi=150)
        plt.close(fig)
        report_lines.append("  saved results/fig_lyapunov.png")


def analyze_ablation(report_lines, which):
    path = f'results/ablation_{which}.json'
    abl = load(path)
    if abl is None:
        report_lines.append(f"[ablation] no {path} found")
        return

    pm_open  = [r['modes']['open']['pmatch_mean']  for r in abl.values()]
    pm_closed = [r['modes']['closed']['pmatch_mean'] for r in abl.values()]
    diffs = [c - o for c, o in zip(pm_closed, pm_open)]
    n_positive = sum(1 for d in diffs if d > 0)

    report_lines.append(f"\n=== ABLATION: {which} ({len(abl)} seeds) ===")
    report_lines.append(f"  pmatch open:    {stats(pm_open)[0]:.4f} ± {stats(pm_open)[1]:.4f}")
    report_lines.append(f"  pmatch closed:  {stats(pm_closed)[0]:.4f} ± {stats(pm_closed)[1]:.4f}")
    report_lines.append(f"  diff (closed - open): mean {np.mean(diffs):+.4f}, "
                        f"{n_positive}/{len(diffs)} seeds positive")


def analyze_sweep(report_lines):
    sw = load('results/sweep.json')
    if sw is None:
        report_lines.append("[sweep] no results/sweep.json found")
        return

    report_lines.append(f"\n=== SWEEP (seed {sw['seed']}) ===")
    report_lines.append(f"  {'alpha':>6s} {'beta':>6s} {'pmatch':>9s} {'lat_acc':>9s} {'lyap':>9s}")
    alphas, pms, lats, lyps = [], [], [], []
    for r in sw['sweep']:
        report_lines.append(f"  {r['alpha']:>6.2f} {r['beta']:>6.2f} "
                            f"{r['pmatch_mean']:>9.4f} {r['latent_acc']:>9.3f} "
                            f"{r['lyapunov']:>9.4f}")
        alphas.append(r['alpha'])
        pms.append(r['pmatch_mean'])
        lats.append(r['latent_acc'])
        lyps.append(r['lyapunov'])

    if HAVE_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(alphas, pms, marker='o')
        axes[0, 0].set_title('(a) Prediction match')
        axes[0, 0].set_ylabel('pmatch')
        axes[0, 1].plot(alphas, lats, marker='o', color='C1')
        axes[0, 1].set_title('(b) Latent discrimination')
        axes[0, 1].set_ylabel('latent_acc')
        axes[1, 0].plot(alphas, lyps, marker='o', color='C4')
        axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('(c) Lyapunov')
        axes[1, 0].set_xlabel('alpha')
        axes[1, 0].set_ylabel('lyapunov')
        axes[1, 1].axis('off')
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        fig.suptitle('Spectrum of states under closed-loop sweep')
        fig.tight_layout()
        fig.savefig('results/fig_sweep.png', dpi=150)
        plt.close(fig)
        report_lines.append("  saved results/fig_sweep.png")


def analyze_baseline(report_lines):
    bl = load('results/baseline.json')
    if bl is None:
        report_lines.append("[baseline] no results/baseline.json found")
        return

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in bl:
        grouped[r['config']].append(r)

    report_lines.append("\n=== VANILLA ESN BASELINE ===")
    report_lines.append(f"  {'Config':<15s} {'N':>3s} {'Acc':>16s} {'Lyap':>10s}")
    for name, rows in sorted(grouped.items()):
        accs = [r['latent_acc'] for r in rows]
        lyps = [r['lyapunov'] for r in rows]
        report_lines.append(f"  {name:<15s} {len(rows):>3d} "
                            f"{stats(accs)[0]:.3f} ± {stats(accs)[1]:.3f}    "
                            f"{stats(lyps)[0]:>+.4f}")


def main():
    os.makedirs('results', exist_ok=True)
    lines = []
    analyze_main(lines)
    analyze_ablation(lines, 'parametric_only')
    analyze_ablation(lines, 'content_only')
    analyze_sweep(lines)
    analyze_baseline(lines)

    report = '\n'.join(lines)
    print(report)
    with open('results/summary.txt', 'w') as f:
        f.write(report + '\n')
    print(f"\nSaved results/summary.txt")


if __name__ == '__main__':
    main()
