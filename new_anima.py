"""
new anima: minimal architecture for chaos-preserving self-referential feedback
in reservoir computing.

This single file contains:
  - Two specialized reservoirs (chaos and memory)
  - Dual-head self-model with asymmetric compression
  - Two feedback loops (parametric and content-based)
  - Training routine
  - Simulation routine (three modes: baseline / open / closed)
  - Metrics (prediction match, Lyapunov, latent discrimination)

Public API:
  Config              dataclass holding all hyperparameters
  build_reservoir     construct a single reservoir
  SelfModel           dual-head self-model (PyTorch nn.Module)
  train_self_model    train the self-model on a driving signal
  simulate            run the system in baseline/open/closed mode
  measure_lyapunov    paired-noise Lyapunov estimator
  measure_pmatch_etc  metric aggregation
  make_signals        the 5 test signals used in experiments

Repository: https://github.com/<USER>/new-anima
Author: Alexander Vasiliev, Independent Researcher, Israel
Project: new anima
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """All hyperparameters live here. Defaults reproduce the paper exactly."""

    # Reservoirs
    n_neurons: int = 1000
    chaos_spectral_radius: float = 1.8
    chaos_leak: float = 0.30
    chaos_density: float = 0.10
    memory_spectral_radius: float = 1.05
    memory_leak: float = 0.05
    memory_density: float = 0.05
    bias_scale: float = 0.10
    input_scale: float = 0.30
    noise_base: float = 1e-4

    # Self-model
    latent_dim: int = 32          # per head; total joint latent = 2 * latent_dim
    hidden_1: int = 512
    hidden_2: int = 128
    predictor_hidden: int = 128
    predict_horizon: int = 30     # how far ahead the predictor forecasts
    alpha: float = 0.5            # weight of chaos latent in predictor input
    beta: float = 0.5             # weight of memory latent

    # Training
    train_steps: int = 20000
    train_epochs: int = 200
    train_batch_size: int = 64
    train_lr: float = 1e-3
    pred_loss_weight: float = 0.5
    vicreg_var_weight: float = 0.1   # variance term from VICReg (anti-collapse)

    # Strange-loop feedback
    parametric_strength: float = 0.5  # +/- 50% modulation of noise and leak
    content_weight: float = 0.05      # 0 disables Loop 2

    # Simulation
    n_steps: int = 5000
    warmup_steps: int = 1000
    signal_interval: int = 200
    signal_length: int = 50

    # Lyapunov
    lyapunov_steps: int = 500
    lyapunov_perturbation: float = 1e-6


# =============================================================================
# Reservoir construction and dynamics
# =============================================================================

def build_reservoir(n_neurons: int, spectral_radius: float, density: float,
                    bias_scale: float, seed: int):
    """Build a sparse recurrent matrix scaled to the target spectral radius."""
    np.random.seed(seed)
    W_sparse = sparse_random(
        n_neurons, n_neurons, density=density, format='csr',
        data_rvs=np.random.randn, random_state=seed,
    )
    W = W_sparse.toarray()
    try:
        eigenvalues, _ = eigs(W_sparse, k=1, which='LM', maxiter=2000)
        max_eig = abs(eigenvalues[0])
    except Exception:
        max_eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / max_eig)

    np.random.seed(seed + 10)
    b = (np.random.rand(n_neurons) * 2 - 1) * bias_scale
    return W, b


def step(state, W, b, input_value, W_in, leak, noise_level, noise_sample,
         pred_input=None, pred_weight=0.0):
    """One leaky-integrator step. Accepts an optional content-feedback term."""
    pre = W @ state + b
    if W_in is not None and input_value != 0:
        pre = pre + W_in * input_value
    pre = pre + noise_level * noise_sample
    if pred_input is not None and pred_weight > 0:
        pre = pre + pred_weight * pred_input
    return (1 - leak) * state + leak * np.tanh(pre)


def make_signals(signal_length: int):
    """Five distinct signal shapes used in all experiments."""
    L = signal_length
    t = np.arange(L)
    signals = {
        'pulse':        np.zeros(L),
        'low_freq':     np.sin(2 * np.pi * t / L * 1.5),
        'high_freq':    np.sin(2 * np.pi * t / L * 8),
        'ramp':         t / L,
        'double_pulse': np.zeros(L),
    }
    signals['pulse'][L // 4] = 1.0
    signals['double_pulse'][L // 4] = 1.0
    signals['double_pulse'][3 * L // 4] = 1.0
    return signals


# =============================================================================
# Self-model
# =============================================================================

class SelfModel(nn.Module):
    """Two encoder-decoder pairs plus a joint predictor."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_in = cfg.n_neurons
        h1, h2 = cfg.hidden_1, cfg.hidden_2
        d = cfg.latent_dim
        ph = cfg.predictor_hidden

        def encoder():
            return nn.Sequential(
                nn.Linear(n_in, h1), nn.Tanh(),
                nn.Linear(h1, h2),  nn.Tanh(),
                nn.Linear(h2, d),
            )
        def decoder():
            return nn.Sequential(
                nn.Linear(d, h2), nn.Tanh(),
                nn.Linear(h2, h1), nn.Tanh(),
                nn.Linear(h1, n_in), nn.Tanh(),
            )

        self.encoder_chaos  = encoder()
        self.decoder_chaos  = decoder()
        self.encoder_memory = encoder()
        self.decoder_memory = decoder()
        self.predictor = nn.Sequential(
            nn.Linear(2 * d, ph), nn.Tanh(),
            nn.Linear(ph, ph),    nn.Tanh(),
            nn.Linear(ph, 2 * d),
        )

    def encode(self, x_c, x_m):
        return self.encoder_chaos(x_c), self.encoder_memory(x_m)

    def decode_memory(self, z_m):
        return self.decoder_memory(z_m)

    def predict_joint(self, z_c, z_m):
        """Predict joint latent predict_horizon steps ahead.

        alpha and beta weight the inputs to the predictor — this is the
        knob the alpha/beta sweep varies.
        """
        a = self.cfg.alpha
        b = self.cfg.beta
        z_in = torch.cat([a * z_c, b * z_m], dim=-1)
        return self.predictor(z_in)


# =============================================================================
# Training
# =============================================================================

def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_training_trajectories(cfg: Config, seed: int, verbose=True):
    """Drive both reservoirs with sin+noise; return aligned trajectories."""
    if verbose:
        print(f"  Generating {cfg.train_steps} training steps (seed={seed})...")

    W_c, b_c = build_reservoir(cfg.n_neurons, cfg.chaos_spectral_radius,
                               cfg.chaos_density, cfg.bias_scale, seed)
    W_m, b_m = build_reservoir(cfg.n_neurons, cfg.memory_spectral_radius,
                               cfg.memory_density, cfg.bias_scale, seed + 100)
    np.random.seed(seed + 50); W_in_c = np.random.randn(cfg.n_neurons) * cfg.input_scale
    np.random.seed(seed + 60); W_in_m = np.random.randn(cfg.n_neurons) * cfg.input_scale

    np.random.seed(seed)
    state_c = np.random.randn(cfg.n_neurons) * 0.01
    state_m = np.random.randn(cfg.n_neurons) * 0.01

    Xc = np.empty((cfg.train_steps, cfg.n_neurons), dtype=np.float32)
    Xm = np.empty((cfg.train_steps, cfg.n_neurons), dtype=np.float32)

    for t in range(cfg.train_steps):
        sig = 0.5 * np.sin(2 * np.pi * t / 200) + 0.3 * np.random.randn()
        ns_c = np.random.randn(cfg.n_neurons)
        ns_m = np.random.randn(cfg.n_neurons)
        state_c = step(state_c, W_c, b_c, sig, W_in_c,
                       cfg.chaos_leak, cfg.noise_base, ns_c)
        state_m = step(state_m, W_m, b_m, sig, W_in_m,
                       cfg.memory_leak, cfg.noise_base, ns_m)
        Xc[t] = state_c.astype(np.float32)
        Xm[t] = state_m.astype(np.float32)

    return Xc, Xm


def train_self_model(cfg: Config, seed: int, verbose=True) -> SelfModel:
    """Full training procedure. Returns a frozen model in eval mode."""
    set_global_seeds(seed)
    Xc, Xm = generate_training_trajectories(cfg, seed, verbose=verbose)

    model = SelfModel(cfg)
    opt = optim.Adam(model.parameters(), lr=cfg.train_lr)
    mse = nn.MSELoss()

    Xc_t = torch.from_numpy(Xc)
    Xm_t = torch.from_numpy(Xm)
    horizon = cfg.predict_horizon
    n = cfg.train_steps - horizon

    if verbose:
        print(f"  Training {cfg.train_epochs} epochs "
              f"(batch={cfg.train_batch_size}, lr={cfg.train_lr})...")

    for epoch in range(cfg.train_epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for s in range(0, n, cfg.train_batch_size):
            batch = idx[s:s + cfg.train_batch_size]
            t_idx = torch.from_numpy(batch).long()
            t_idx_h = torch.from_numpy(batch + horizon).long()

            xc, xm = Xc_t[t_idx], Xm_t[t_idx]
            xc_f, xm_f = Xc_t[t_idx_h], Xm_t[t_idx_h]

            zc, zm = model.encode(xc, xm)
            xc_hat = model.decoder_chaos(zc)
            xm_hat = model.decoder_memory(zm)
            recon = mse(xc_hat, xc) + mse(xm_hat, xm)

            zc_f_true, zm_f_true = model.encode(xc_f, xm_f)
            z_future_true = torch.cat([zc_f_true, zm_f_true], dim=-1).detach()
            z_future_pred = model.predict_joint(zc, zm)
            pred_loss = mse(z_future_pred, z_future_true)

            # VICReg variance term only (we use the std/hinge part, not the
            # full variance+invariance+covariance objective)
            z_joint = torch.cat([zc, zm], dim=-1)
            std = torch.sqrt(z_joint.var(dim=0) + 1e-4)
            var_loss = torch.mean(torch.clamp(1.0 - std, min=0))

            loss = recon + cfg.pred_loss_weight * pred_loss + cfg.vicreg_var_weight * var_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch % 20 == 0 or epoch == cfg.train_epochs - 1):
            print(f"    Epoch {epoch:3d}: loss={epoch_loss / n_batches:.5f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =============================================================================
# Simulation
# =============================================================================

def _calibrate_tension_scale(W_c, b_c, W_m, b_m, W_in_c, W_in_m, signals,
                             model: SelfModel, cfg: Config,
                             probe_steps: int = 1500, seed: int = 0):
    """Run a brief probe; pick tension_scale so median tension hits tanh~=0.5."""
    np.random.seed(seed)
    state_c = np.random.randn(cfg.n_neurons) * 0.01
    state_m = np.random.randn(cfg.n_neurons) * 0.01

    sig_list = list(signals.items())
    next_signal_step = 0
    current = None
    sig_step = -1

    horizon = cfg.predict_horizon
    pred_buffer = []
    tensions_c, tensions_m = [], []

    for t in range(probe_steps):
        if t >= next_signal_step and current is None:
            _, current = sig_list[np.random.randint(len(sig_list))]
            sig_step = 0
            next_signal_step = t + cfg.signal_interval
        inp = 0.0
        if current is not None:
            inp = current[sig_step]
            sig_step += 1
            if sig_step >= len(current):
                current = None

        ns_c = np.random.randn(cfg.n_neurons)
        ns_m = np.random.randn(cfg.n_neurons)
        state_c = step(state_c, W_c, b_c, inp, W_in_c,
                       cfg.chaos_leak, cfg.noise_base, ns_c)
        state_m = step(state_m, W_m, b_m, inp, W_in_m,
                       cfg.memory_leak, cfg.noise_base, ns_m)

        with torch.no_grad():
            xc_t = torch.from_numpy(state_c.astype(np.float32)).unsqueeze(0)
            xm_t = torch.from_numpy(state_m.astype(np.float32)).unsqueeze(0)
            zc, zm = model.encode(xc_t, xm_t)
            z_future = model.predict_joint(zc, zm).numpy()[0]
            pred_buffer.append((zc.numpy()[0], zm.numpy()[0], z_future))

        if len(pred_buffer) > horizon:
            past_zc, past_zm, past_pred = pred_buffer.pop(0)
            zc_pred = past_pred[:cfg.latent_dim]
            zm_pred = past_pred[cfg.latent_dim:]
            with torch.no_grad():
                zc_now, zm_now = model.encode(xc_t, xm_t)
            t_c = np.linalg.norm(zc_pred - zc_now.numpy()[0])
            t_m = np.linalg.norm(zm_pred - zm_now.numpy()[0])
            tensions_c.append(t_c)
            tensions_m.append(t_m)

    if not tensions_c:
        return 1.0, 1.0

    # tanh(median * scale) ~= 0.5  =>  scale = atanh(0.5) / median
    atanh_05 = 0.5493
    median_c = np.median(tensions_c)
    median_m = np.median(tensions_m)
    scale_c = atanh_05 / max(median_c, 1e-6)
    scale_m = atanh_05 / max(median_m, 1e-6)
    return scale_c, scale_m


def simulate(cfg: Config, model: Optional[SelfModel], seed: int,
             mode: str = 'closed',
             parametric_enabled: bool = True,
             content_weight: Optional[float] = None,
             verbose: bool = False):
    """
    Run the full system for cfg.n_steps and return per-step traces.

    mode:
      'baseline' -- reservoirs run alone, model is ignored
      'open'     -- model observes (encodes, predicts, computes tension)
                    but feedback loops are inactive
      'closed'   -- parametric loop on (if parametric_enabled) and
                    content loop on (if content_weight > 0)

    content_weight: overrides cfg.content_weight if set
    parametric_enabled: master switch for Loop 1
    """
    if content_weight is None:
        content_weight = cfg.content_weight

    # Build reservoirs and inputs
    W_c, b_c = build_reservoir(cfg.n_neurons, cfg.chaos_spectral_radius,
                               cfg.chaos_density, cfg.bias_scale, seed)
    W_m, b_m = build_reservoir(cfg.n_neurons, cfg.memory_spectral_radius,
                               cfg.memory_density, cfg.bias_scale, seed + 100)
    np.random.seed(seed + 50); W_in_c = np.random.randn(cfg.n_neurons) * cfg.input_scale
    np.random.seed(seed + 60); W_in_m = np.random.randn(cfg.n_neurons) * cfg.input_scale

    signals = make_signals(cfg.signal_length)
    sig_keys = list(signals.keys())
    sig_list = list(signals.items())

    # Tension scale calibration (only needed if loops active)
    if mode == 'closed' and parametric_enabled and model is not None:
        scale_c, scale_m = _calibrate_tension_scale(
            W_c, b_c, W_m, b_m, W_in_c, W_in_m, signals, model, cfg,
            seed=seed + 200,
        )
        if verbose:
            print(f"    tension scales: c={scale_c:.4f}, m={scale_m:.4f}")
    else:
        scale_c = scale_m = 1.0

    # Warmup
    np.random.seed(seed + 1000)
    state_c = np.random.randn(cfg.n_neurons) * 0.01
    state_m = np.random.randn(cfg.n_neurons) * 0.01
    for _ in range(cfg.warmup_steps):
        ns_c = np.random.randn(cfg.n_neurons)
        ns_m = np.random.randn(cfg.n_neurons)
        state_c = step(state_c, W_c, b_c, 0.0, None,
                       cfg.chaos_leak, cfg.noise_base, ns_c)
        state_m = step(state_m, W_m, b_m, 0.0, None,
                       cfg.memory_leak, cfg.noise_base, ns_m)

    # Logging buffers
    horizon = cfg.predict_horizon
    pred_buffer = []         # holds (z_c, z_m, z_future_pred, decoded_zm_pred)
    history = {
        'chaos_state':    np.empty((cfg.n_steps, cfg.n_neurons), dtype=np.float32),
        'memory_state':   np.empty((cfg.n_steps, cfg.n_neurons), dtype=np.float32),
        'latent_chaos':   [],
        'latent_memory':  [],
        'pmatch':         [],
        'tension_c':      [],
        'tension_m':      [],
        'signal_markers': [],   # (step, label_idx) for latent-disc snapshots
    }

    # Schedule for snapshot 100 steps after each signal starts
    pending_snapshot_step = None
    pending_snapshot_label = None

    next_signal_step = 0
    current = None
    sig_step = -1

    for t in range(cfg.n_steps):
        # Signal scheduling
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

        # Self-model: encode current state
        z_c = z_m = z_future_pred = None
        decoded_zm_pred = None
        if model is not None and mode in ('open', 'closed'):
            with torch.no_grad():
                xc_t = torch.from_numpy(state_c.astype(np.float32)).unsqueeze(0)
                xm_t = torch.from_numpy(state_m.astype(np.float32)).unsqueeze(0)
                z_c_t, z_m_t = model.encode(xc_t, xm_t)
                z_future_t = model.predict_joint(z_c_t, z_m_t)
                z_c = z_c_t.numpy()[0]
                z_m = z_m_t.numpy()[0]
                z_future_pred = z_future_t.numpy()[0]
                # Decode the predicted memory part for the content loop
                zm_future_pred_t = z_future_t[:, cfg.latent_dim:]
                decoded_zm_pred = model.decoder_memory(zm_future_pred_t).numpy()[0]

        # Tension and the prediction match
        tension_c = tension_m = 0.0
        pmatch = float('nan')
        prev_decoded_pred = None
        if model is not None and len(pred_buffer) > horizon:
            past = pred_buffer.pop(0)
            past_zc, past_zm, past_future, past_decoded = past
            zc_pred = past_future[:cfg.latent_dim]
            zm_pred = past_future[cfg.latent_dim:]
            tension_c = float(np.linalg.norm(zc_pred - z_c))
            tension_m = float(np.linalg.norm(zm_pred - z_m))
            # Prediction match: cosine between predicted memory latent
            # (made horizon steps ago) and the actual memory latent now.
            denom = (np.linalg.norm(zm_pred) * np.linalg.norm(z_m) + 1e-8)
            pmatch = float(np.dot(zm_pred, z_m) / denom)
            prev_decoded_pred = past_decoded

        if z_c is not None:
            pred_buffer.append((z_c, z_m, z_future_pred, decoded_zm_pred))

        # Parametric modulation (Loop 1)
        if mode == 'closed' and parametric_enabled and model is not None and len(pred_buffer) > 1:
            noise_level = cfg.noise_base * (
                1.0 + cfg.parametric_strength * np.tanh(tension_c * scale_c)
            )
            leak_m_eff = cfg.memory_leak * (
                1.0 - cfg.parametric_strength * np.tanh(tension_m * scale_m)
            )
        else:
            noise_level = cfg.noise_base
            leak_m_eff = cfg.memory_leak

        # Content feedback (Loop 2) — only in closed mode, only if weight > 0
        pred_input_to_memory = None
        pred_weight_eff = 0.0
        if (mode == 'closed' and content_weight > 0
                and prev_decoded_pred is not None):
            pred_input_to_memory = prev_decoded_pred
            pred_weight_eff = content_weight

        # One reservoir step
        ns_c = np.random.randn(cfg.n_neurons)
        ns_m = np.random.randn(cfg.n_neurons)
        state_c = step(state_c, W_c, b_c, inp, W_in_c,
                       cfg.chaos_leak, noise_level, ns_c)
        state_m = step(state_m, W_m, b_m, inp, W_in_m,
                       leak_m_eff, cfg.noise_base, ns_m,
                       pred_input=pred_input_to_memory,
                       pred_weight=pred_weight_eff)

        # Log
        history['chaos_state'][t] = state_c.astype(np.float32)
        history['memory_state'][t] = state_m.astype(np.float32)
        if z_c is not None:
            history['latent_chaos'].append(z_c)
            history['latent_memory'].append(z_m)
        history['pmatch'].append(pmatch)
        history['tension_c'].append(tension_c)
        history['tension_m'].append(tension_m)

        if pending_snapshot_step is not None and t == pending_snapshot_step:
            history['signal_markers'].append((t, pending_snapshot_label))
            pending_snapshot_step = None

    history['final_chaos_state']  = state_c
    history['final_memory_state'] = state_m
    return history


# =============================================================================
# Metrics
# =============================================================================

def measure_lyapunov(cfg: Config, W, b, leak, initial_state, seed: int):
    """Paired-noise Lyapunov estimator (Benettin renormalization)."""
    n = cfg.n_neurons
    np.random.seed(seed + 9999)
    pert = np.random.randn(n)
    pert = pert / np.linalg.norm(pert) * cfg.lyapunov_perturbation
    s1 = initial_state.copy()
    s2 = s1 + pert

    np.random.seed(seed + 8888)
    noise = np.random.randn(cfg.lyapunov_steps, n)
    log_divs = []
    for i in range(cfg.lyapunov_steps):
        ns = noise[i]
        pre1 = W @ s1 + b + cfg.noise_base * ns
        pre2 = W @ s2 + b + cfg.noise_base * ns
        s1 = (1 - leak) * s1 + leak * np.tanh(pre1)
        s2 = (1 - leak) * s2 + leak * np.tanh(pre2)
        diff = s2 - s1
        d = np.linalg.norm(diff)
        if d > 0:
            log_divs.append(np.log(d / cfg.lyapunov_perturbation))
            s2 = s1 + diff / d * cfg.lyapunov_perturbation
    return float(np.mean(log_divs)) if log_divs else 0.0


def measure_latent_discrimination(history, cfg, seed: int):
    """KMeans accuracy with best-permutation matching, on memory latents
    snapshotted ~100 steps after each signal onset.
    """
    markers = history['signal_markers']
    if not markers or len(history['latent_memory']) == 0:
        return {'accuracy': 0.0, 'silhouette': 0.0, 'n_samples': 0}

    # latent_memory and history step t are aligned only if the model was
    # active from step 0; if not, len(latent_memory) < n_steps. We use the
    # index where it is available.
    n_avail = len(history['latent_memory'])
    snapshots, labels = [], []
    for step_idx, label in markers:
        if step_idx < n_avail:
            snapshots.append(history['latent_memory'][step_idx])
            labels.append(label)

    if len(snapshots) < 5:
        return {'accuracy': 0.0, 'silhouette': 0.0, 'n_samples': len(snapshots)}

    snapshots = np.array(snapshots)
    labels = np.array(labels)
    n_classes = len(set(labels))
    if n_classes < 2:
        return {'accuracy': 0.0, 'silhouette': 0.0, 'n_samples': len(snapshots)}

    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(snapshots)

    label_set = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_set)}
    labels_remapped = np.array([label_to_idx[l] for l in labels])

    best_acc = 0.0
    for perm in permutations(range(n_classes)):
        mapped = np.array([perm[c] for c in cluster_labels])
        acc = float(np.mean(mapped == labels_remapped))
        if acc > best_acc:
            best_acc = acc

    try:
        sil = float(silhouette_score(snapshots, labels))
    except Exception:
        sil = 0.0
    return {'accuracy': best_acc, 'silhouette': sil, 'n_samples': len(snapshots)}


def summarize(history, cfg, model, seed: int):
    """Aggregate metrics into a single dict per run."""
    pmatch = [v for v in history['pmatch'] if not np.isnan(v)]
    tc = [v for v in history['tension_c'] if v > 0]
    tm = [v for v in history['tension_m'] if v > 0]

    # Lyapunov on BOTH reservoirs. The paper reports the chaos one as the
    # primary chaos-preservation metric. The memory one is included for
    # completeness — memory has a small spectral radius and sits at or
    # below the edge of chaos by design, so a near-zero or slightly
    # negative value is expected.
    W_c, b_c = build_reservoir(cfg.n_neurons, cfg.chaos_spectral_radius,
                               cfg.chaos_density, cfg.bias_scale, seed)
    W_m, b_m = build_reservoir(cfg.n_neurons, cfg.memory_spectral_radius,
                               cfg.memory_density, cfg.bias_scale, seed + 100)
    lyap_chaos = measure_lyapunov(
        cfg, W_c, b_c, cfg.chaos_leak,
        history['final_chaos_state'], seed,
    )
    lyap_memory = measure_lyapunov(
        cfg, W_m, b_m, cfg.memory_leak,
        history['final_memory_state'], seed,
    )

    lat = measure_latent_discrimination(history, cfg, seed)

    return {
        'pmatch_mean':       float(np.mean(pmatch)) if pmatch else float('nan'),
        'pmatch_std':        float(np.std(pmatch))  if pmatch else float('nan'),
        'tension_c_mean':    float(np.mean(tc)) if tc else float('nan'),
        'tension_m_mean':    float(np.mean(tm)) if tm else float('nan'),
        'lyapunov':          lyap_chaos,           # primary (chaos reservoir)
        'lyapunov_memory':   lyap_memory,           # auxiliary
        'latent_acc':        lat['accuracy'],
        'latent_silhouette': lat['silhouette'],
        'latent_n_samples':  lat['n_samples'],
    }


# =============================================================================
# Convenience: run all three modes for a single seed
# =============================================================================

def run_seed_three_modes(seed: int, cfg: Optional[Config] = None,
                         verbose: bool = True):
    """Train one model on `seed`, then run baseline / open / closed.
    Returns a dict with metrics for each mode."""
    cfg = cfg or Config()
    set_global_seeds(seed)

    if verbose:
        print(f"=== seed {seed} ===")
        print("  [1/4] Training self-model...")
    model = train_self_model(cfg, seed, verbose=verbose)

    results = {}
    for mode in ('baseline', 'open', 'closed'):
        if verbose:
            print(f"  [run] mode={mode}")
        m = None if mode == 'baseline' else model
        history = simulate(cfg, m, seed, mode=mode, verbose=False)
        results[mode] = summarize(history, cfg, model, seed)

    if verbose:
        print(f"  baseline lyap: {results['baseline']['lyapunov']:.4f}")
        print(f"  open  pmatch:  {results['open']['pmatch_mean']:.4f}")
        print(f"  closed pmatch: {results['closed']['pmatch_mean']:.4f}")
    return {'seed': seed, 'config': cfg.__dict__, 'modes': results}
