"""
Tests for ephys/decode_location.py.

Two things are guarded here, since this module previously had zero test
coverage and shipped a real bug (leaky, shuffled cross-validation folds on
autocorrelated position data — see the module's "Statistical notes"):

1. Cross-validation folds must be contiguous (``KFold(shuffle=False)``),
   matching the established convention in ``decode_partner_distance.py`` /
   ``inter_brain_dynamics._fit_r2``. A regression back to shuffled folds
   leaks adjacent (near-identical) time bins between train and test.
2. A population with real, strong place tuning must actually decode its own
   position better than a shuffle-null — the basic sanity check any spatial
   decoder needs to pass before a partner-decoding claim means anything.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection

from ephys.decode_location import decode_location, _cv_decode
from video.tracking_import import VideoTrackingData

DT = 0.05  # 20 Hz tracking


def _random_walk(n, bounds, step_sd, seed):
    """Reflecting random walk inside ``bounds`` ((xmin,xmax),(ymin,ymax))."""
    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = bounds
    x = np.empty(n)
    y = np.empty(n)
    x[0] = 0.5 * (xmin + xmax)
    y[0] = 0.5 * (ymin + ymax)
    for i in range(1, n):
        x[i] = np.clip(x[i - 1] + rng.normal(0, step_sd), xmin, xmax)
        y[i] = np.clip(y[i - 1] + rng.normal(0, step_sd), ymin, ymax)
    return x, y


def _poisson_spikes_from_place_field(t, x, y, center, sigma, peak_hz, base_hz, seed):
    """Spikes whose instantaneous rate is a Gaussian bump over (x, y)."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    rate = base_hz + peak_hz * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    counts = rng.poisson(rate * DT)
    spikes = [ti + rng.uniform(0, DT, size=c) for ti, c in zip(t, counts) if c]
    return np.sort(np.concatenate(spikes)) if spikes else np.array([])


class _StubSync:
    """Identity behavior->ephys clock map (ephys seconds == behavior seconds)."""

    def convert_behavior_to_ephys(self, behav_seconds):
        return np.asarray(behav_seconds, dtype=np.float64)


def _tracking_and_ks(n=6000, bounds=((0.0, 100.0), (0.0, 100.0)), n_cells=12,
                      n_field_centers=6, peak_hz=15.0, base_hz=0.5, sigma=12.0,
                      seed=0):
    """A self-only VideoTrackingData plus a population with real place tuning
    to that same trajectory (a mix of tuned and untuned cells)."""
    x, y = _random_walk(n, bounds, step_sd=3.0, seed=seed)
    t = np.arange(n) * DT

    rng = np.random.default_rng(seed)
    (xmin, xmax), (ymin, ymax) = bounds
    centers = [(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)) for _ in range(n_field_centers)]

    spike_times_by_cell = []
    for i in range(n_cells):
        if i < n_field_centers:
            st = _poisson_spikes_from_place_field(
                t, x, y, centers[i], sigma, peak_hz, base_hz, seed=seed + i + 1)
        else:
            # Untuned cell: flat-rate Poisson, no spatial information.
            rng_i = np.random.default_rng(seed + i + 1)
            counts = rng_i.poisson(base_hz * DT, size=n)
            spikes = [ti + rng_i.uniform(0, DT, size=c) for ti, c in zip(t, counts) if c]
            st = np.sort(np.concatenate(spikes)) if spikes else np.array([])
        spike_times_by_cell.append(st)

    from types import SimpleNamespace
    ks_data = SimpleNamespace(
        ks_ids=list(range(n_cells)),
        spike_times_by_cell=spike_times_by_cell,
        get_filtered_cells_spike_times=lambda **kw: (list(range(n_cells)), spike_times_by_cell),
    )

    tracking = VideoTrackingData(
        animal_id="focal",
        session_id="synthetic",
        parsed_data={"focal": pd.DataFrame({
            "frame": np.arange(n), "center_x": x, "center_y": y,
        })},
        timestamps=(t * 1e9).astype(np.int64),
    )
    tracking.synchronize_with_ephys(_StubSync())
    return ks_data, tracking


class TestContiguousFolds:
    def test_cv_decode_uses_shuffle_false(self):
        """Regression guard: KFold must be constructed with shuffle=False.

        Shuffled folds leak adjacent (near-identical) time bins between train
        and test for autocorrelated position data, inflating both the real
        decode and the shuffle-null comparably and hiding real effects.
        """
        rng = np.random.default_rng(0)
        n, n_cells = 40, 3
        X = rng.poisson(1.0, size=(n, n_cells)).astype(np.float64)
        Y = np.column_stack([np.linspace(0, 10, n), np.linspace(0, 10, n)])
        x_edges = np.linspace(0, 10, 6)
        y_edges = np.linspace(0, 10, 6)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        with patch("sklearn.model_selection.KFold",
                   wraps=sklearn.model_selection.KFold) as spy:
            _cv_decode(X, Y, x_edges, y_edges, x_centers, y_centers,
                       bin_size=0.5, smoothing_sigma=1.0,
                       use_occupancy_prior=True, estimate="expected", cv_folds=5)

        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs.get("shuffle") is False

    def test_folds_are_contiguous_blocks(self):
        """KFold(shuffle=False) assigns contiguous index blocks to each fold."""
        kf = sklearn.model_selection.KFold(n_splits=4, shuffle=False)
        X = np.arange(20).reshape(-1, 1)
        for _, test_idx in kf.split(X):
            assert np.array_equal(test_idx, np.arange(test_idx[0], test_idx[-1] + 1))


class TestSelfDecodingSanityCheck:
    def test_self_position_beats_shuffle_null(self):
        """A population with real place tuning must decode its own position
        better than a shuffle-null — the sanity check that failed on real
        data before the contiguous-fold fix (see HANDOFF.md, 2026-08-19)."""
        ks_data, tracking = _tracking_and_ks()

        res = decode_location(
            ks_data, tracking, "focal",
            bin_size=0.5, n_spatial_bins=15, smoothing_sigma=1.0,
            rate_smoothing_sigma=1.0, cv_folds=5,
            use_quality_cells=False, null="shuffle", n_shuffles=30,
        )

        assert res["status"] == "success"
        # A clear margin, not just nominally lower -- guards against a
        # regression that makes both real and null comparably (and wrongly)
        # tight, which is exactly what shuffled folds did on real data.
        assert res["median_error"] < 0.7 * res["null_median_error"]
