"""Mock-data tests for the binning/alignment helpers in ephys.run_territory."""

import numpy as np
import pandas as pd

from ephys import run_territory as rt


def _track(t, x, y):
    return pd.DataFrame({"t": t, "x": x, "y": y, "speed": np.zeros(len(t))})


def test_interp_blanks_bins_far_from_any_sample():
    t = np.r_[np.arange(0, 10, 0.1), np.arange(20, 30, 0.1)]       # 10 s tracking gap
    df = _track(t, np.linspace(0, 1, len(t)), np.zeros(len(t)))
    centers = np.array([5.0, 14.9, 15.1, 25.0])
    out = rt.interp_on_grid(df, centers, max_gap=1.0)
    assert np.isfinite(out["x"][0]) and np.isfinite(out["x"][3])
    assert np.isnan(out["x"][1]) and np.isnan(out["x"][2])


def test_interp_is_nan_outside_tracking_range():
    df = _track(np.arange(10, 20, 0.1), np.ones(100), np.ones(100))
    assert np.isnan(rt.interp_on_grid(df, np.array([0.0, 100.0]))["x"]).all()


def test_bin_counts_matches_hand_count():
    st = [np.array([0.1, 0.2, 0.7, 1.4]), np.array([1.9])]
    c = rt.bin_counts(st, t0=0.0, n_bins=4, dt=0.5)
    assert c.shape == (4, 2)
    assert c[:, 0].tolist() == [2, 1, 1, 0] and c[:, 1].tolist() == [0, 0, 0, 1]


def test_chunk_inputs_window_and_partner_distance():
    t = np.arange(0, 100, 0.1)
    tracks = {"rat1": _track(t, np.zeros(len(t)), np.zeros(len(t))),
              "rat2": _track(t, np.full(len(t), 30.0), np.full(len(t), 40.0)),
              "rat3": _track(t, np.full(len(t), 300.0), np.zeros(len(t)))}
    inp = rt.chunk_inputs(tracks, "rat1", (10.0, 20.0))
    assert inp["n_bins"] == 20 and inp["t0"] == 10.0
    assert np.allclose(inp["partner_dist"], 50.0)                    # nearest of the others
    assert set(inp["others"]) == {"rat2", "rat3"}


def test_chunk_inputs_never_extends_past_the_focal_tracking():
    t = np.arange(0, 10, 0.1)
    tracks = {"rat1": _track(t, np.zeros(len(t)), np.zeros(len(t))),
              "rat2": _track(t, np.ones(len(t)), np.ones(len(t)))}
    inp = rt.chunk_inputs(tracks, "rat1", (-50.0, 500.0))
    assert inp["t0"] >= 0.0 and inp["n_bins"] <= 20


# ---- exclusivity spread gate ------------------------------------------------

def _spread_chunks(focal_spread, seed=0, n_chunks=3, n=6000):
    rng = np.random.default_rng(seed)
    homes = {"rat1": (25.0, 50.0), "rat2": (75.0, 25.0), "rat3": (75.0, 75.0)}
    out = {}
    for c in range(n_chunks):
        t = c * 1000.0 + np.arange(n) * 0.1
        chunk = {}
        for name, h in homes.items():
            sp = focal_spread if name == "rat1" else 8.0
            xy = np.clip(rng.normal(h, sp, size=(n, 2)), 0.5, 99.5)
            chunk[name] = _track(t, xy[:, 0], xy[:, 1])
        out[f"c{c}"] = chunk
    return out


def _gate(chunks, **kw):
    return rt.exclusivity_spread_gate(
        chunks, "rat1", (-1e9, 1e9), bounds=(0.0, 100.0, 0.0, 100.0), bins=20,
        map_kw=dict(smoothing_sigma_bins=1.0, min_dwell_sec=1.0, min_tracked_sec=10.0, min_spread=0.0),
        min_eff_bins=kw.get("min_eff_bins", 10.0), min_range=kw.get("min_range", 0.2))


def test_spread_gate_fails_for_a_rat_resting_in_one_spot():
    g = _gate(_spread_chunks(focal_spread=1.0))
    assert not g["passed"]
    assert all(r["eff_bins"] < 10 for r in g["per_chunk"].values())


def test_spread_gate_passes_for_a_rat_that_roams_into_neighbours_areas():
    # a rat confined to its own area sees constant exclusivity (range ~0, nothing to separate from
    # position); the covariate only varies once it visits shared or foreign ground
    g = _gate(_spread_chunks(focal_spread=30.0))
    assert g["passed"], g["per_chunk"]


def test_spread_gate_fails_for_a_rat_confined_to_its_own_area():
    g = _gate(_spread_chunks(focal_spread=8.0))
    assert not g["passed"]
    assert all(r["eff_bins"] >= 10 and r["range"] < 0.2 for r in g["per_chunk"].values())
