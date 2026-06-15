"""
Smoke tests for ephys/social_spatial_plots.py.

Builds a small synthetic SocialFieldResults and asserts each plot function
returns a matplotlib Figure without error (Agg backend, no display).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from ephys.social_spatial_fields import compute_social_place_fields
from ephys import social_spatial_plots as sp
from video.tracking_import import VideoTrackingData

ARENA = ((0.0, 80.0), (0.0, 80.0))
DT = 0.04


def _xy(n, seed):
    rng = np.random.default_rng(seed)
    x = np.clip(np.cumsum(rng.normal(0, 3, n)) + 40, 0, 80)
    y = np.clip(np.cumsum(rng.normal(0, 3, n)) + 40, 0, 80)
    t = np.arange(n) * DT
    speed = np.sqrt(np.gradient(x, t) ** 2 + np.gradient(y, t) ** 2)
    return pd.DataFrame({"t": t, "x": x, "y": y, "speed": speed})


def _spikes(xy, center, seed, peak=25.0):
    rng = np.random.default_rng(seed)
    x = xy["x"].to_numpy(); y = xy["y"].to_numpy(); t = xy["t"].to_numpy()
    rate = 0.2 + peak * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * 8.0 ** 2))
    counts = rng.poisson(rate * DT)
    out = [ti + rng.uniform(0, DT, c) for ti, c in zip(t, counts) if c]
    return np.sort(np.concatenate(out)) if out else np.array([])


class _StubSync:
    def convert_behavior_to_ephys(self, behav_seconds):
        return np.asarray(behav_seconds, dtype=np.float64)


def _video_tracking(tracking, session_id="S") -> VideoTrackingData:
    parsed = {}
    timestamps_ns = None
    for aid, df in tracking.items():
        n = df.shape[0]
        parsed[aid] = pd.DataFrame({
            "frame": np.arange(n),
            "center_x": df["x"].to_numpy(dtype=np.float64),
            "center_y": df["y"].to_numpy(dtype=np.float64),
        })
        if timestamps_ns is None:
            timestamps_ns = (df["t"].to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    return VideoTrackingData(
        animal_id=list(tracking.keys())[0],
        session_id=session_id,
        parsed_data=parsed,
        timestamps=timestamps_ns,
    )


@pytest.fixture(scope="module")
def results():
    tr = {"A": _xy(9000, 1), "B": _xy(9000, 2), "C": _xy(9000, 3)}
    ks = SimpleNamespace(
        ks_ids=[0, 1],
        spike_times_by_cell=[
            _spikes(tr["A"], (40, 40), 10),   # self-tuned
            _spikes(tr["B"], (40, 40), 11),   # partner-tuned
        ],
    )
    return compute_social_place_fields(
        ks, _video_tracking(tr), _StubSync(), focal_animal="A",
        target_animals=["A", "B", "C"], pixels_per_cm=None,
        bin_size_cm=5.0, smoothing_sigma_cm=5.0,
        speed_filter_subject="none", n_shuffles=30, min_n_spikes=20,
        use_quality_cells=False, arena_bounds=ARENA, seed=0,
    )


def test_rate_maps_grid(results):
    fig = sp.plot_rate_maps_grid(results, cluster_id=0)
    assert fig is not None
    plt.close(fig)


def test_field_similarity_grid(results):
    fig = sp.plot_field_similarity_grid(results, cluster_id=0)
    assert fig is not None
    plt.close(fig)


def test_classification_summary(results):
    fig = sp.plot_cell_classification_summary(results)
    assert fig is not None
    plt.close(fig)


def test_population_similarity(results):
    key = next(iter(results.population_field_similarity))
    fig = sp.plot_population_field_similarity(results, key)
    assert fig is not None
    plt.close(fig)


def test_skaggs_vs_shuffle(results):
    fig = sp.plot_skaggs_vs_shuffle(results, target_animal="A", top_k=2)
    assert fig is not None
    plt.close(fig)


def test_field_stability(results):
    fig = sp.plot_field_stability(results)
    assert fig is not None
    plt.close(fig)


def test_summary_dashboard(results):
    fig = sp.plot_social_place_summary(results)
    assert fig is not None
    plt.close(fig)
