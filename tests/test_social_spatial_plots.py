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


# ---------------------------------------------------------------------------
# bits/spike vs firing rate (effect-size bias panel)
# ---------------------------------------------------------------------------

def _stub_results(stats, focal="A", targets=("A", "B")):
    # rate_maps is needed even though target_animals is set: _targets passes
    # `list(results.rate_maps.keys())` as dict.get's default, which Python
    # evaluates eagerly.
    return SimpleNamespace(
        parameters={"focal_animal": focal, "target_animals": list(targets)},
        stats=stats,
        rate_maps={},
    )


def _fs(rate, bits):
    return SimpleNamespace(mean_rate_hz=rate, skaggs_bits_per_spike=bits)


def test_bits_vs_rate_draws_one_series_per_target(results):
    fig, ax = plt.subplots()
    sp._draw_bits_vs_rate(ax, results)
    assert len(ax.collections) == 3           # A, B, C
    assert ax.get_xlabel() == "mean rate (Hz)"
    assert ax.get_ylabel() == "Skaggs bits/spike"
    # Colour-matched per-target rho annotations double as the legend.
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 3
    assert any("(self)" in t for t in texts)
    assert all(("rho" in t) or ("n/a" in t) for t in texts)
    plt.close(fig)


def test_bits_vs_rate_reports_rho_per_target_not_pooled():
    """Pooling across targets would average a real effect with null ones."""
    stats = {
        # Rising with rate.
        "A": {i: _fs(float(i + 1), 0.1 * (i + 1)) for i in range(6)},
        # Falling with rate: the low-spike inflation signature.
        "B": {i: _fs(float(i + 1), 1.0 / (i + 1)) for i in range(6)},
    }
    fig, ax = plt.subplots()
    sp._draw_bits_vs_rate(ax, _stub_results(stats))
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 2
    assert "+1.00" in texts[0]        # A perfectly increasing
    assert "-1.00" in texts[1]        # B perfectly decreasing
    plt.close(fig)


def test_bits_vs_rate_skips_targets_without_a_positive_rate():
    """A target whose maps are empty (zero occupancy) contributes no points."""
    stats = {"A": {0: _fs(1.0, 0.5), 1: _fs(2.0, 0.3)},
             "B": {0: _fs(0.0, 0.0), 1: _fs(np.nan, np.nan)}}
    fig, ax = plt.subplots()
    sp._draw_bits_vs_rate(ax, _stub_results(stats))
    assert len(ax.collections) == 1
    plt.close(fig)


def test_bits_vs_rate_survives_having_no_data():
    fig, ax = plt.subplots()
    sp._draw_bits_vs_rate(ax, _stub_results({"A": {}, "B": {}}))
    assert not ax.collections
    assert ax.get_xscale() == "linear"
    plt.close(fig)


def test_bits_vs_rate_uses_log_x_only_for_a_wide_range():
    wide = {"A": {i: _fs(10.0 ** i, 0.5) for i in range(4)}}
    narrow = {"A": {i: _fs(1.0 + 0.1 * i, 0.5) for i in range(4)}}
    for stats, expected in ((wide, "log"), (narrow, "linear")):
        fig, ax = plt.subplots()
        sp._draw_bits_vs_rate(ax, _stub_results(stats, targets=("A",)))
        assert ax.get_xscale() == expected
        plt.close(fig)


def test_summary_dashboard_includes_the_rate_panel(results):
    fig = sp.plot_social_place_summary(results)
    # 3 target maps on top + 4 population panels below.
    assert len(fig.axes) == 7
    assert "bits/spike vs rate" in [a.get_title() for a in fig.axes]
    plt.close(fig)


def test_summary_dashboard_rows_are_sized_independently():
    """4 targets must not squeeze or gap the 4 fixed bottom panels."""
    tr = {k: _xy(4000, s) for k, s in zip("ABCD", (1, 2, 3, 4))}
    ks = SimpleNamespace(ks_ids=[0], spike_times_by_cell=[_spikes(tr["B"], (40, 40), 9)])
    res = compute_social_place_fields(
        ks, _video_tracking(tr), _StubSync(), focal_animal="A",
        target_animals=list("ABCD"), pixels_per_cm=None, bin_size_cm=5.0,
        smoothing_sigma_cm=5.0, speed_filter_subject="none", n_shuffles=5,
        min_n_spikes=10, use_quality_cells=False, arena_bounds=ARENA, seed=0,
    )
    fig = sp.plot_social_place_summary(res)
    assert len(fig.axes) == 8                 # 4 maps + 4 panels
    assert "bits/spike vs rate" in [a.get_title() for a in fig.axes]
    plt.close(fig)


# ---------------------------------------------------------------------------
# Focal-position overlay
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stratified_results():
    """Same session, but every map restricted to one focal-position disc."""
    tr = {"A": _xy(9000, 1), "B": _xy(9000, 2), "C": _xy(9000, 3)}
    ks = SimpleNamespace(
        ks_ids=[0, 1],
        spike_times_by_cell=[_spikes(tr["A"], (40, 40), 10),
                             _spikes(tr["B"], (40, 40), 11)],
    )
    return compute_social_place_fields(
        ks, _video_tracking(tr), _StubSync(), focal_animal="A",
        target_animals=["A", "B", "C"], pixels_per_cm=None,
        bin_size_cm=5.0, smoothing_sigma_cm=5.0,
        speed_filter_subject="none", n_shuffles=10, min_n_spikes=10,
        use_quality_cells=False, arena_bounds=ARENA, seed=0,
        null_method="position_shuffle",
        self_stratum_radius_cm=15.0, self_stratum_center=(40.0, 40.0),
    )


def test_overlay_draws_stratum_disc(stratified_results):
    fig, ax = plt.subplots()
    assert sp.overlay_focal_position(ax, stratified_results) is True
    # A circle patch at the stratum centre, with the stratum radius.
    circles = [p for p in ax.patches if isinstance(p, plt.Circle)]
    assert len(circles) == 1
    assert circles[0].get_radius() == 15.0
    assert circles[0].get_center() == (40.0, 40.0)
    plt.close(fig)


def test_overlay_falls_back_to_occupancy_contours(results):
    """Without a stratum there is no single focal location, so draw contours."""
    fig, ax = plt.subplots()
    assert sp.overlay_focal_position(ax, results) is True
    assert not [p for p in ax.patches if isinstance(p, plt.Circle)]
    assert ax.collections or ax.lines      # contours and/or the modal marker
    plt.close(fig)


def test_overlay_returns_false_without_focal_tracking(results):
    """Focal absent from the rate maps ⇒ nothing to draw, and no exception."""
    stripped = SimpleNamespace(
        rate_maps={k: v for k, v in results.rate_maps.items() if k != "A"},
        parameters={**results.parameters, "self_stratum_radius_cm": None,
                    "self_stratum_center": None},
    )
    fig, ax = plt.subplots()
    assert sp.overlay_focal_position(ax, stripped) is False
    plt.close(fig)


def test_overlay_does_not_rescale_the_map_axes(stratified_results):
    """A disc near the arena edge must not stretch the panel past the arena."""
    fig = sp.plot_rate_maps_grid(stratified_results, cluster_id=0)
    for ax in fig.axes:
        if ax.get_xlabel() == "x (cm)":
            assert ax.get_xlim() == pytest.approx(ARENA[0], abs=5.0)
    plt.close(fig)


def test_grid_and_summary_accept_show_focal_false(stratified_results):
    fig = sp.plot_rate_maps_grid(stratified_results, cluster_id=0, show_focal=False)
    assert not [p for p in fig.axes[0].patches if isinstance(p, plt.Circle)]
    plt.close(fig)
    fig = sp.plot_social_place_summary(stratified_results, show_focal=False)
    assert fig is not None
    plt.close(fig)


def test_overlay_note_describes_the_condition(results, stratified_results):
    assert "unconditioned" in sp._focal_overlay_note(results)
    note = sp._focal_overlay_note(stratified_results)
    assert "within 15 cm" in note and "(40, 40)" in note
