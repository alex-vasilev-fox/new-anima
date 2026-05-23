"""
Alpha/beta sweep on a single trained model.

Trains one self-model on seed 42, then runs closed-loop simulations with
11 values of alpha from 0 to 1 (beta = 1 - alpha).

Output: results/sweep.json

Usage:
    python run_sweep.py
    python run_sweep.py --seed 42 --n-points 11

Runtime: ~30 minutes (one training + 11 short simulations).

Author: Alexander Vasiliev
"""

import argparse
import json
import os
import time

import numpy as np

from new_anima import Config, train_self_model, simulate, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-points', type=int, default=11)
    args = parser.parse_args()

    cfg = Config()
    print(f"Training model on seed {args.seed}...")
    model = train_self_model(cfg, args.seed, verbose=True)

    alphas = np.linspace(0.0, 1.0, args.n_points)
    sweep_results = []
    t0 = time.time()

    for i, a in enumerate(alphas):
        # Avoid exact zero to keep predictor's input non-degenerate
        a_eff = max(float(a), 1e-6)
        b_eff = max(1.0 - float(a), 1e-6)
        cfg.alpha = a_eff
        cfg.beta = b_eff
        # Mutate model's config too (alpha/beta are read from cfg via the model)
        model.cfg = cfg
        print(f"\n[{i+1}/{len(alphas)}] alpha={a:.2f}, beta={1-a:.2f}")
        history = simulate(cfg, model, args.seed, mode='closed', verbose=False)
        s = summarize(history, cfg, model, args.seed)
        s['alpha'] = float(a)
        s['beta'] = float(1 - a)
        sweep_results.append(s)
        print(f"    pmatch={s['pmatch_mean']:.4f}, lat_acc={s['latent_acc']:.3f}, "
              f"lyap={s['lyapunov']:.4f}")

    output = 'results/sweep.json'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump({
            'seed': args.seed,
            'sweep': sweep_results,
        }, f, indent=2, default=str)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Saved {output}")


if __name__ == '__main__':
    main()
