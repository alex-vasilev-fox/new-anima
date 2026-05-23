"""
Smoke test: quick sanity check that the architecture runs end to end.

Uses tiny hyperparameters (1 epoch, short simulation) just to verify
imports, shapes, and basic flow. Runs in ~1-2 minutes on a CPU.

Does NOT reproduce paper numbers; full runs need run_main.py.

Usage:
    python smoke_test.py
"""

import time

from new_anima import (
    Config, train_self_model, simulate, summarize, set_global_seeds,
)


def main():
    print("=" * 60)
    print("new anima — smoke test")
    print("=" * 60)
    t0 = time.time()

    # Quick config: 1 epoch, 500 steps, just to verify the flow
    cfg = Config(
        train_epochs=1,
        train_steps=2000,
        n_steps=500,
        warmup_steps=100,
        signal_interval=80,
        lyapunov_steps=100,
    )

    seed = 42
    set_global_seeds(seed)

    print("\n[1/4] training tiny self-model...")
    model = train_self_model(cfg, seed, verbose=True)

    for mode in ('baseline', 'open', 'closed'):
        print(f"\n[run] mode={mode}")
        m = None if mode == 'baseline' else model
        h = simulate(cfg, m, seed, mode=mode, verbose=False)
        s = summarize(h, cfg, model, seed)
        print(f"  pmatch={s['pmatch_mean']:.4f}, "
              f"latent_acc={s['latent_acc']:.3f}, "
              f"lyap={s['lyapunov']:.4f}")

    print(f"\nSmoke test passed in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
