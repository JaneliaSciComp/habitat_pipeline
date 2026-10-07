"""Synthetic sessions for territory-encoding calibration and power tests.

A focal rat random-walks (Ornstein-Uhlenbeck) around its home; two neighbours do
the same around theirs. Territory labels come from the real
``video.territory`` module, built leave-one-chunk-out from independent chunks, so
the simulator exercises the same labelling code the real analysis will use.

Cells are inhomogeneous Poisson with

    rate = base * exp(field_gain * gaussian_bump(x, y))      # place tuning
                 * exp(step_log * own + grad * signed_dist)  # territory effect
                 * exp(drift(t))                             # slow gain drift

so a cell can have (i) place tuning only, (ii) place tuning that *straddles the
territory boundary*, (iii) a genuine territory step or gradient, and (iv) slow
rate drift unrelated to territory. (ii) and (iv) are the two ways a naive test
manufactures false positives: the territory covariate is spatially smooth and
temporally autocorrelated, so it is collinear with anything slow or smooth.

``naive_poisson_lrt`` is a deliberately simple reference tester (plain Poisson
LRT against chi2_1). It is used only to show the harness *can* detect
mis-calibration; it is not the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from video.territory import build_territory_map_loo, label_positions

ARENA = 100.0
DT = 0.5
HOMES = {"focal": (25.0, 50.0), "nbr_b": (75.0, 25.0), "nbr_c": (75.0, 75.0)}
MAP_SPREAD = 15.0        # how far animals roam when the territory map is estimated
FOCAL_SPREAD = 25.0      # focal roams further in the test chunk, so it crosses borders
OU_TAU_SEC = 20.0


def _ou(rng, n, dt, home, spread, tau=OU_TAU_SEC):
    a = np.exp(-dt / tau)
    noise = rng.standard_normal((n, 2)) * spread * np.sqrt(1 - a * a)
    out = np.empty((n, 2))
    cur = np.asarray(home, float) + rng.standard_normal(2) * spread
    for k in range(n):
        cur = np.asarray(home) + a * (cur - np.asarray(home)) + noise[k]
        cur = np.clip(cur, 0.5, ARENA - 0.5)
        out[k] = cur
    return out


def _chunk(rng, n, spreads, t0=0.0):
    t = t0 + np.arange(n) * DT
    return {name: pd.DataFrame({"t": t, **dict(zip("xy", _ou(rng, n, DT, HOMES[name],
                                                             spreads[name]).T))})
            for name in HOMES}


def make_session(seed: int, duration_s: float = 1800.0, n_map_chunks: int = 3,
                 nbr_spread: float = MAP_SPREAD) -> Dict:
    """One focal-rat session with territory labels from independent chunks."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / DT)
    chunks = {f"map{i}": _chunk(rng, n, {k: MAP_SPREAD for k in HOMES}, t0=i * 1e4)
              for i in range(n_map_chunks)}
    test = _chunk(rng, n, {"focal": FOCAL_SPREAD, "nbr_b": nbr_spread, "nbr_c": nbr_spread},
                  t0=1e5)
    chunks["test"] = test
    tmap = build_territory_map_loo(chunks, "test", bounds=(0, ARENA, 0, ARENA), bins=20,
                                   min_tracked_sec=60.0)

    f = test["focal"]
    x, y = f["x"].to_numpy(), f["y"].to_numpy()
    lab = label_positions(tmap, "focal", x, y)
    speed = np.r_[0.0, np.hypot(np.diff(x), np.diff(y)) / DT]
    partner = np.minimum(np.hypot(x - test["nbr_b"]["x"].to_numpy(), y - test["nbr_b"]["y"].to_numpy()),
                         np.hypot(x - test["nbr_c"]["x"].to_numpy(), y - test["nbr_c"]["y"].to_numpy()))
    # who owns the bin the focal is in, is that neighbour home, and how far away is it
    owner = np.array([o if o in ("nbr_b", "nbr_c") else None for o in lab["owner"]], dtype=object)
    owner_present = np.full(len(x), np.nan)
    owner_dist = np.full(len(x), np.nan)
    for nb in ("nbr_b", "nbr_c"):
        nx_, ny_ = test[nb]["x"].to_numpy(), test[nb]["y"].to_numpy()
        home = label_positions(tmap, nb, nx_, ny_)["own"].to_numpy()
        sel = owner == nb
        owner_present[sel] = home[sel].astype(float)
        owner_dist[sel] = np.hypot(x - nx_, y - ny_)[sel]
    return dict(
        t=f["t"].to_numpy() - f["t"].iloc[0], dt=DT, x=x, y=y, speed=speed,
        owner=owner, owner_present=owner_present, owner_dist=owner_dist,
        partner_dist=partner,
        own=lab["own"].to_numpy(), signed_dist=np.nan_to_num(lab["signed_dist"].to_numpy(), nan=-50.0),
        exclusivity=lab["exclusivity"].to_numpy(),
        territory_map=tmap, seed=seed,
    )


def covariates(session: Dict) -> Dict[str, np.ndarray]:
    return {k: session[k] for k in ("x", "y", "speed", "partner_dist")}


def territory_inputs(session: Dict) -> Dict[str, np.ndarray]:
    return {"own": session["own"], "signed_dist": session["signed_dist"],
            "exclusivity": session["exclusivity"]}


@dataclass
class CellSpec:
    base_hz: float = 2.0
    field_center: tuple = (25.0, 50.0)
    field_sigma: float = 18.0
    field_gain: float = 1.2          # log-gain at the field peak
    step_log: float = 0.0            # log rate ratio own vs not-own
    grad_per_unit: float = 0.0       # log-rate per unit signed boundary distance
    excl_slope: float = 0.0          # log-rate per unit exclusivity (0..1)
    drift_sd: float = 0.0            # sd of slow log-gain drift
    drift_tau_sec: float = 300.0


def boundary_point(session: Dict, rng: Optional[np.random.Generator] = None) -> tuple:
    """A position on the territory boundary that the focal actually visits."""
    d = np.abs(session["signed_dist"])
    near = np.where(d < 4.0)[0]
    if near.size == 0:
        near = np.argsort(d)[:50]
    i = near[0] if rng is None else rng.choice(near)
    return float(session["x"][i]), float(session["y"][i])


def cell_rate(session: Dict, spec: CellSpec, rng: np.random.Generator) -> np.ndarray:
    cx, cy = spec.field_center
    r2 = (session["x"] - cx) ** 2 + (session["y"] - cy) ** 2
    bump = np.exp(-r2 / (2 * spec.field_sigma ** 2))
    log_rate = (np.log(spec.base_hz) + spec.field_gain * bump
                + spec.step_log * session["own"].astype(float)
                + spec.grad_per_unit * np.clip(session["signed_dist"], -30.0, 30.0)
                + spec.excl_slope * np.nan_to_num(session["exclusivity"],
                                                  nan=float(np.nanmean(session["exclusivity"]))))
    if spec.drift_sd > 0:
        n = session["x"].size
        a = np.exp(-session["dt"] / spec.drift_tau_sec)
        z = np.empty(n)
        z[0] = rng.standard_normal() * spec.drift_sd
        eps = rng.standard_normal(n) * spec.drift_sd * np.sqrt(1 - a * a)
        for k in range(1, n):
            z[k] = a * z[k - 1] + eps[k]
        log_rate = log_rate + z
    return np.exp(log_rate)


def simulate_counts(session: Dict, spec: CellSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.poisson(cell_rate(session, spec, rng) * session["dt"])


def simulate_population(session: Dict, rng: np.random.Generator, n_cells: int = 40, *,
                        frac_step: float = 0.0, step_ratio: float = 1.5, drift_sd: float = 0.0,
                        density: str = "uniform", frac_presence: float = 0.0,
                        presence_ratio: float = 1.5, dist_gain: float = 0.0,
                        base_hz: float = 2.0) -> np.ndarray:
    """(n_bins, n_cells) spike counts for a population.

    density      'uniform' place-field centres anywhere; 'boundary_dense' half the
                 cells have narrow fields ON the territory boundary (a place-coding
                 confound that has nothing to do with territory).
    frac_step    share of cells with a genuine own-territory step.
    frac_presence share of cells whose rate rises when the owner of the territory
                 the focal is in is home.
    dist_gain    all cells' log-rate slope on (capped) focal-owner distance / 50:
                 neural information about distance that is *not* about presence.
    """
    n_t = session["x"].size
    od = np.nan_to_num(np.minimum(session["owner_dist"], 100.0), nan=0.0) / 50.0
    pres = np.nan_to_num(session["owner_present"], nan=0.0)
    out = np.empty((n_t, n_cells), dtype=np.int64)
    for c in range(n_cells):
        near_boundary = density == "boundary_dense" and c < n_cells // 2
        spec = CellSpec(
            base_hz=base_hz,
            field_center=boundary_point(session, rng) if near_boundary
            else (rng.uniform(5, 95), rng.uniform(5, 95)),
            field_sigma=8.0 if near_boundary else rng.uniform(12, 20),
            field_gain=1.8 if near_boundary else 1.2,
            step_log=np.log(step_ratio) if c < int(round(frac_step * n_cells)) else 0.0,
            drift_sd=drift_sd, drift_tau_sec=400.0)
        rate = cell_rate(session, spec, rng) * np.exp(dist_gain * od)
        if c >= n_cells - int(round(frac_presence * n_cells)):
            rate = rate * np.exp(np.log(presence_ratio) * pres)
        out[:, c] = rng.poisson(rate * session["dt"])
    return out


# ---------------------------------------------------------------------------
# Reference (naive) tester - for harness sensitivity only
# ---------------------------------------------------------------------------

def _rbf_design(x, y, speed, partner, grid=6):
    cs = np.linspace(0, ARENA, grid)
    w = ARENA / (grid - 1)
    cols = [np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * w ** 2)) for cx in cs for cy in cs]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    return np.column_stack([np.ones_like(x), *cols, z(speed), z(partner)])


def _poisson_irls(X, y, offset, ridge=1e-4, iters=60):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3)) - offset.mean()
    prev = -np.inf
    for _ in range(iters):
        eta = np.clip(X @ beta + offset, -20, 20)
        mu = np.exp(eta)
        ll = np.sum(y * eta - mu) - 0.5 * ridge * beta @ beta
        if abs(ll - prev) < 1e-8:
            break
        prev = ll
        W = mu
        grad = X.T @ (y - mu) - ridge * beta
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        for _ in range(8):                               # step halving
            nb = beta + step
            eta2 = np.clip(X @ nb + offset, -20, 20)
            if np.sum(y * eta2 - np.exp(eta2)) - 0.5 * ridge * nb @ nb >= ll - 1e-9:
                beta = nb
                break
            step = step / 2
    eta = np.clip(X @ beta + offset, -20, 20)
    return beta, float(np.sum(y * eta - np.exp(eta)))


def naive_poisson_lrt(counts: np.ndarray, session: Dict, effect: str = "step") -> float:
    """Plain nested-Poisson LRT p-value against chi2_1 (no bootstrap, no blocking)."""
    base = _rbf_design(session["x"], session["y"], session["speed"], session["partner_dist"])
    cov = session["own"].astype(float) if effect == "step" else np.clip(session["signed_dist"], -30, 30)
    cov = (cov - cov.mean()) / (cov.std() + 1e-9)
    full = np.column_stack([base, cov])
    off = np.full(counts.shape, np.log(session["dt"]))
    _, ll0 = _poisson_irls(base, counts.astype(float), off)
    _, ll1 = _poisson_irls(full, counts.astype(float), off)
    return float(stats.chi2.sf(max(2 * (ll1 - ll0), 0.0), df=1))
