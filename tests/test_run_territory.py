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
