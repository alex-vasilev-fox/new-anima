"""
Ablation experiments:
  parametric_only -- only Loop 1 active (content_weight = 0)
  content_only    -- only Loop 2 active (parametric_enabled = False)

Each runs on the same 10 fixed seeds. Trains a fresh model per seed.

Output: results/ablation_<which>.json

Usage:
    python run_ablation.py parametric_only
    python run_ablation.py content_only

Runtime: ~3 hours per ablation on a CPU.

Author: Alexander Vasiliev
"""

import argparse
import json
import os
import time

from new_anima import Config, train_self_model, simulate, summarize, set_global_seeds


SEEDS = [42, 137, 271, 314, 577, 1024, 2718, 3141, 4096, 6174]


def run_ablation_for_seed(which: str, seed: int, cfg: Config, verbose=True):
    if verbose:
        print(f"=== seed {seed} ({which}) ===")
    set_global_seeds(seed)
    model = train_self_model(cfg, seed, verbose=verbose)

    if which == 'parametric_only':
        # Loop 1 on, Loop 2 off
        params = dict(parametric_enabled=True, content_weight=0.0)
    elif which == 'content_only':
        # Loop 1 off, Loop 2 on
        params = dict(parametric_enabled=False, content_weight=0.05)
    else:
        raise ValueError(f"unknown ablation: {which}")

    results = {}
    for mode in ('open', 'closed'):
        # open mode: feedback inactive regardless, used as baseline
        history = simulate(cfg, model, seed, mode=mode, **params)
        results[mode] = summarize(history, cfg, model, seed)

    return {'seed': seed, 'ablation': which, 'modes': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('which', choices=['parametric_only', 'content_only'])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()

    output = f'results/ablation_{args.which}.json'
    existing = {}
    if os.path.exists(output):
        with open(output) as f:
            existing = json.load(f)

    cfg = Config()
    t0 = time.time()

    for seed in args.seeds:
        key = str(seed)
        if key in existing:
            print(f"=== seed {seed} already done, skipping ===")
            continue
        result = run_ablation_for_seed(args.which, seed, cfg, verbose=True)
        existing[key] = result
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"  saved ({(time.time() - t0) / 60:.1f} min total)")

    print(f"\nDone. {len(existing)} seeds in {output}")


if __name__ == '__main__':
    main()
