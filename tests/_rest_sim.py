"""Synthetic rest-at-a-fixed-spot sessions for calibrating ephys.rest_distance_coding.

A session is a few contiguous rest segments. Latents (all unit-variance AR(1) unless noted):
    d_fast   partner log-distance, fast (tau ``tau_d`` s)
    d_slow   partner log-distance, slow (tau 600 s), scaled by ``slow_sd``
    A        arousal (tau 10 s); ``rho`` of its variance is shared with d_fast, so nuisance and distance co-vary
Observed nuisance = A and a monotone transform of A, each with independent measurement noise (``nuis_noise`` = 0 means arousal is measured exactly).
Cells: log rate = b0 + slow drift + ``arousal_gain`` * g(A) + ``beta`` * d_fast_std, Poisson counts.
``beta`` is per SD of the fast distance component, so it is the quantity the test is meant to detect.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

DT = 0.5


def _ar1(n: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.exp(-DT / tau)
    x = np.empty(n)
    x[0] = rng.normal()
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


@dataclass
class RestSpec:
    n_seg: int = 3
    seg_sec: float = 300.0
    gap_bins: int = 20
    tau_d: float = 30.0
    slow_sd: float = 1.0          # slow distance component (relative to the fast one)
    rho: float = 0.0              # arousal-distance coupling
    nuis_noise: float = 0.0       # measurement noise on the nuisance covariates (SD of A units)
    n_cells: int = 12
    base_rate_hz: float = 4.0
    drift_sd: float = 0.5         # slow cell drift (log units)
    arousal_gain: float = 0.0     # cells' dependence on arousal (log units per SD)
    beta: float = 0.0             # effect of fast distance (log units per SD) on every cell
    frac_cells_affected: float = 1.0
    flicker_rate: float = 0.0     # bridged speed flickers: expected gaps per second inside rest (each 1-6 bins)
    move_gain: float = 0.0        # cells' log-rate change during a flicker
    move_d_coupling: float = 0.0  # flickers are more likely when the partner is close (d_fast low), logit units per SD
    move_noise: float = 0.3       # measurement noise on the observed movement nuisance


def make_rest_session(seed: int, spec: RestSpec) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    seg_n = int(spec.seg_sec / DT)
    n = spec.n_seg * (seg_n + spec.gap_bins)
    rest = np.zeros(n, bool)
    for k in range(spec.n_seg):
        rest[k * (seg_n + spec.gap_bins): k * (seg_n + spec.gap_bins) + seg_n] = True
    d_fast = _ar1(n, spec.tau_d, rng)
    d_slow = _ar1(n, 600.0, rng)
    d = 5.5 + 0.7 * (d_fast + spec.slow_sd * d_slow)
    A = np.sqrt(spec.rho) * d_fast + np.sqrt(1 - spec.rho) * _ar1(n, 10.0, rng)
    nuis = np.column_stack([A + spec.nuis_noise * rng.normal(size=n),
                            np.exp(0.5 * (A + spec.nuis_noise * rng.normal(size=n)))])
    M = np.zeros(n)
    if spec.flicker_rate > 0:
        p = spec.flicker_rate * DT * 1 / (1 + np.exp(spec.move_d_coupling * d_fast))     # more flickers when d_fast low
        p = p / p.mean() * spec.flicker_rate * DT if spec.move_d_coupling else p
        for i in np.flatnonzero(rest & (rng.random(n) < p)):
            M[i:i + int(rng.integers(1, 7))] = 1.0
        M[~rest] = 0.0
        nuis = np.column_stack([nuis, M + spec.move_noise * rng.normal(size=n)])
    counts = np.empty((n, spec.n_cells))
    affected = rng.random(spec.n_cells) < spec.frac_cells_affected
    d_std = d_fast / d_fast[rest].std()
    for c in range(spec.n_cells):
        drift = spec.drift_sd * _ar1(n, 300.0, rng)
        eta = (np.log(spec.base_rate_hz) + drift + spec.arousal_gain * np.tanh(A) + spec.move_gain * M
               + (spec.beta * d_std if affected[c] else 0.0))
        counts[:, c] = rng.poisson(np.exp(eta) * DT)
    return dict(counts=counts, d=d, nuis=nuis, rest=rest, A=A, affected=affected)
