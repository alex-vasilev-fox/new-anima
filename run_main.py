"""
Main experiment: train and evaluate on 10 fixed seeds.

For each seed:
  - train the self-model from scratch
  - run baseline / open / closed
  - save the summary metrics

Output: results/main.json

Usage:
    python run_main.py
    python run_main.py --seeds 42 137   # just two seeds (quick check)

Runtime: about 3 hours for all 10 seeds on a CPU. Each seed is independent,
so you can stop and resume — already-completed seeds are skipped.

Author: Alexander Vasiliev
"""

import argparse
import json
import os
import time

from new_anima import Config, run_seed_three_modes


SEEDS = [42, 137, 271, 314, 577, 1024, 2718, 3141, 4096, 6174]
OUTPUT = 'results/main.json'


def load_existing():
    if not os.path.exists(OUTPUT):
        return {}
    with open(OUTPUT, 'r') as f:
        return json.load(f)


def save(results):
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()

    cfg = Config()
    results = load_existing()
    t0 = time.time()

    for seed in args.seeds:
        key = str(seed)
        if key in results:
            print(f"=== seed {seed} already done, skipping ===")
            continue
        result = run_seed_three_modes(seed, cfg, verbose=True)
        results[key] = result
        save(results)
        print(f"  saved (total runtime so far: {(time.time() - t0) / 60:.1f} min)")

    print(f"\nDone. {len(results)} seeds in {OUTPUT}")


if __name__ == '__main__':
    main()
