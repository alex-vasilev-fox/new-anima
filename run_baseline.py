"""
Vanilla ESN baselines: three single-reservoir configurations, no self-model.

  ESN-chaos:   rho=1.8,  leak=0.30, n=1000  (same as our chaos reservoir)
  ESN-memory:  rho=1.05, leak=0.05, n=1000  (same as our memory reservoir)
  ESN-large:   rho=1.4,  leak=0.15, n=2000  (same parameter budget total)

For each (config, seed) we drive the reservoir with the five test signals,
snapshot the reservoir state 100 steps after each signal onset, then
KMeans-cluster the snapshots and measure latent discrimination accuracy
(same protocol as new anima).

Output: results/baseline.json

Usage:
    python run_baseline.py

Runtime: ~10 minutes on a CPU.

Author: Alexander Vasiliev
"""

import argparse
import json
import os
import time
from itertools import permutations

import numpy as np
from sklearn.cluster import KMeans

from new_anima import Config, build_reservoir, make_signals, step, measure_lyapunov


SEEDS = [42, 137, 271, 314, 577, 1024, 2718, 3141, 4096, 6174]

CONFIGS = {
    'ESN-chaos':  dict(n=1000, rho=1.8,  leak=0.30, density=0.10),
    'ESN-memory': dict(n=1000, rho=1.05, leak=0.05, density=0.05),
    'ESN-large':  dict(n=2000, rho=1.4,  leak=0.15, density=0.05),
}


def run_one(name: str, params: dict, seed: int, cfg: Config):
    """Drive a single reservoir with the test signals, return metrics."""
    np.random.seed(seed)
    n = params['n']
    W, b = build_reservoir(n, params['rho'], params['density'],
                           cfg.bias_scale, seed)
    np.random.seed(seed + 50)
    W_in = np.random.randn(n) * cfg.input_scale

    signals = make_signals(cfg.signal_length)
    sig_keys = list(signals.keys())
    sig_list = list(signals.items())

    np.random.seed(seed + 1000)
    state = np.random.randn(n) * 0.01
    for _ in range(cfg.warmup_steps):
        ns = np.random.randn(n)
        state = step(state, W, b, 0.0, None, params['leak'], cfg.noise_base, ns)

    next_signal_step = 0
    current = None
    sig_step = -1
    pending_snapshot_step = None
    pending_snapshot_label = None
    snapshots, labels = [], []

    for t in range(cfg.n_steps):
        if t >= next_signal_step and current is None:
            sig_name, current = sig_list[np.random.randint(len(sig_list))]
            sig_step = 0
            next_signal_step = t + cfg.signal_interval
            pending_snapshot_step = t + cfg.signal_length + 100
            pending_snapshot_label = sig_keys.index(sig_name)
        inp = 0.0
        if current is not None:
            inp = current[sig_step]
            sig_step += 1
            if sig_step >= len(current):
                current = None
        ns = np.random.randn(n)
        state = step(state, W, b, inp, W_in,
                     params['leak'], cfg.noise_base, ns)
        if pending_snapshot_step is not None and t == pending_snapshot_step:
            snapshots.append(state.copy())
            labels.append(pending_snapshot_label)
            pending_snapshot_step = None

    # KMeans + permutation matching
    snapshots = np.array(snapshots)
    labels = np.array(labels)
    n_classes = len(set(labels))
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    cl = km.fit_predict(snapshots)
    label_set = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])
    best_acc = 0.0
    for perm in permutations(range(n_classes)):
        mapped = np.array([perm[c] for c in cl])
        acc = float(np.mean(mapped == labels_remapped))
        if acc > best_acc:
            best_acc = acc

    # Lyapunov on this reservoir's final state
    cfg_use = Config(**{**cfg.__dict__, 'n_neurons': n})
    lyap = measure_lyapunov(cfg_use, W, b, params['leak'], state, seed)

    return {
        'config': name,
        'seed': seed,
        'n_neurons': n,
        'spectral_radius': params['rho'],
        'leak': params['leak'],
        'latent_acc': best_acc,
        'lyapunov': lyap,
        'n_samples': len(snapshots),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--configs', type=str, nargs='+', default=list(CONFIGS.keys()))
    args = parser.parse_args()

    cfg = Config()
    output = 'results/baseline.json'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    rows = []
    t0 = time.time()

    for name in args.configs:
        params = CONFIGS[name]
        print(f"\n=== {name} (rho={params['rho']}, leak={params['leak']}, n={params['n']}) ===")
        for seed in args.seeds:
            r = run_one(name, params, seed, cfg)
            print(f"  seed {seed}: acc={r['latent_acc']:.3f}, lyap={r['lyapunov']:.4f}")
            rows.append(r)
        with open(output, 'w') as f:
            json.dump(rows, f, indent=2, default=str)

    print(f"\nDone in {(time.time() - t0)/60:.1f} min. {len(rows)} runs in {output}")


if __name__ == '__main__':
    main()
