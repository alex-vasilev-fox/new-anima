# new anima

Self-referential feedback in chaotic reservoirs with preserved dynamics.

This repository contains the code, the experiments, and the analysis
for the paper *"Self-referential feedback in chaotic reservoirs with
preserved dynamics: a minimal architecture and the role of feedback type"*
(Vasiliev 2026).

The architecture combines two specialized reservoirs (chaos and memory)
with a dual-head self-model that has two distinct feedback paths back
into the reservoirs. The main empirical finding: a content-based feedback
loop (decoded predicted latent fed back as a weak input) raises a
self-consistency metric — prediction match — from 0.33 to 0.66 across
10 random seeds, while the chaotic dynamics are preserved (Lyapunov
exponent stays at 0.020 throughout).

## Repository layout

All the architecture is in a single file:

```
new_anima.py        all code: reservoirs, self-model, training, simulation, metrics
run_main.py         main experiment: 10 seeds × 3 modes
run_ablation.py     ablations: parametric_only and content_only
run_sweep.py        alpha/beta sweep on a single trained model
run_baseline.py     vanilla ESN baselines (no self-model)
analyze.py          read JSON results and produce summary + plots
smoke_test.py       1-minute end-to-end sanity check
requirements.txt
LICENSE             MIT
```

No package structure, no submodules, no setup.py. Everything is plain
Python files you can read top to bottom.

## Installation

```bash
git clone https://github.com/<USER>/new-anima
cd new-anima
pip install -r requirements.txt
```

Tested on CPU only (Ryzen 7 5800H). GPU works too but is not required.
Python 3.9+ recommended.

## Quick start (1 minute)

To verify that the code runs at all:

```bash
python smoke_test.py
```

This trains a tiny self-model (1 epoch on 2000 samples) and runs all
three modes. It does NOT reproduce the paper numbers — it just shows
that nothing is broken.

## Reproducing the paper (about 7 hours total on CPU)

```bash
# Main result: 10 seeds × 3 modes (~3 hours)
python run_main.py

# Ablation 1: parametric loop only (~3 hours)
python run_ablation.py parametric_only

# Ablation 2: content loop only (~3 hours)
python run_ablation.py content_only

# Alpha/beta sweep on a single trained model (~30 minutes)
python run_sweep.py

# Vanilla ESN baselines, no self-model (~10 minutes)
python run_baseline.py

# Build summary table and plots
python analyze.py
```

Each script writes JSON files into `results/`. The runs are independent
and resumable — already-completed seeds are skipped on restart. You can
start with just a few seeds:

```bash
python run_main.py --seeds 42 137 271
```

## What the experiments measure

**Prediction match.** Cosine similarity between the latent the
self-model predicted 30 steps ago and the actual memory latent now.
This is an inference-time metric, not a training loss. The trained model
is frozen during evaluation; only the feedback loops differ across modes.

**Lyapunov exponent.** Paired-noise method: two trajectories with the
same noise stream, initial perturbation 1e-6, renormalized at each step.
Positive value means chaos is preserved. We check this in all three
modes to verify that adding feedback does not collapse the dynamics.

**Latent discrimination accuracy.** KMeans on memory-latent snapshots
taken 100 steps after each test-signal onset, with best-permutation
matching against the 5 signal types. Chance level = 0.20.

## What you should see

After running `run_main.py` and `analyze.py`, the summary should look
something like:

```
=== MAIN RESULT (10 seeds, both loops active) ===
  pmatch open:    ~0.33
  pmatch closed:  ~0.66
  diff: positive on all 10 seeds
  lyapunov:       ~0.020 in all three modes
```

After both ablations:

```
=== ABLATION: parametric_only ===
  pmatch open:    ~0.34
  pmatch closed:  ~0.30   (slightly degrades)
  9 of 10 seeds get worse, not better

=== ABLATION: content_only ===
  pmatch open:    ~0.33
  pmatch closed:  ~0.65   (essentially matches the full system)
```

This is the central empirical decomposition reported in the paper:
the content loop alone is sufficient, the parametric loop alone is
not (and on this metric is slightly harmful).

## Configuration

All hyperparameters live in the `Config` dataclass at the top of
`new_anima.py`. To change something, edit the default or pass a
`Config` instance to the relevant function.

## Citation

If you use this code, please cite:

```
@misc{vasiliev2026_newanima,
  author = {Alexander Vasiliev},
  title = {Self-referential feedback in chaotic reservoirs with preserved dynamics},
  year = {2026},
  note = {Project new anima. Independent research.},
  url = {https://github.com/<USER>/new-anima}
}
```

## Acknowledgments

AI-native project. The architecture design, code, experiments, and
writing were carried out using Claude and other AI tools. All scientific
decisions and final responsibility are mine.

## License

MIT. See [LICENSE](LICENSE).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20354713.svg)](https://doi.org/10.5281/zenodo.20354713)
## Contact

Open an issue, or write directly. Replication attempts, bug reports,
and ports to other languages are welcome.
alexander.new.anima@gmail.com
