"""
Smoke tests for the Inter-Brain GUI surface.

We can't render Streamlit widgets without a running session, so these
tests focus on:

* The new ``InterBrainParams`` dataclass: immutability, hashability,
  ``as_dict`` round-trip.
* The new tab module ``gui.tabs.inter_brain`` imports cleanly.
* ``gui.app`` imports cleanly (it now references the new tab).
* The refactored pure-compute ``ephys.run_inter_brain._analyze`` returns
  the same payload shape that ``_analyze_and_save`` does, modulo I/O.
"""

import importlib
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

class TestInterBrainParams:
    def test_construct_and_as_dict(self):
        from gui.state import InterBrainParams
        p = InterBrainParams(
            partner_animal_ids=("632",),
            bin_size=0.5,
            smoothing_sigma_sec=0.25,
            n_components=5,
            n_shuffles=50,
            t_window=None,
            method="regularized",
            reg=1e-3,
            cv_folds=5,
            max_lag_bins=10,
            alpha=1.0,
            event_window=1.0,
            behavior_type=None,
        )
        d = p.as_dict()
        assert d["partner_animal_ids"] == ("632",)
        assert d["bin_size"] == 0.5
        assert d["n_components"] == 5
        assert d["method"] == "regularized"

    def test_is_frozen_and_hashable(self):
        from gui.state import InterBrainParams
        p = InterBrainParams(
            partner_animal_ids=("632",),
            bin_size=0.5, smoothing_sigma_sec=0.0,
            n_components=3, n_shuffles=10, t_window=None,
            method="regularized", reg=1e-3, cv_folds=5,
            max_lag_bins=5, alpha=1.0, event_window=1.0,
            behavior_type=None,
        )
        with pytest.raises((AttributeError, Exception)):
            p.bin_size = 0.25  # type: ignore[misc]
        # Hashable for set/dict membership.
        assert hash(p) == hash(p)

    def test_different_params_differ_in_cache_key(self):
        from gui.cache import _make_key
        from gui.state import InterBrainParams
        base = dict(
            partner_animal_ids=("632",),
            bin_size=0.5, smoothing_sigma_sec=0.0,
            n_components=3, n_shuffles=10, t_window=None,
            method="regularized", reg=1e-3, cv_folds=5,
            max_lag_bins=5, alpha=1.0, event_window=1.0,
            behavior_type=None,
        )
        p1 = InterBrainParams(**base)
        p2 = InterBrainParams(**{**base, "bin_size": 1.0})
        p3 = InterBrainParams(**{**base, "partner_animal_ids": ("633",)})
        k1 = _make_key("631", "20251216", None, p1.as_dict())
        k2 = _make_key("631", "20251216", None, p2.as_dict())
        k3 = _make_key("631", "20251216", None, p3.as_dict())
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_gui_tabs_inter_brain_imports(self):
        # Streamlit needs to be importable for the module to load.
        pytest.importorskip("streamlit")
        # Drop any prior import so we get a fresh load.
        for mod in [
            "gui.tabs.inter_brain", "gui.tabs", "gui.runners", "gui.state",
        ]:
            sys.modules.pop(mod, None)
        m = importlib.import_module("gui.tabs.inter_brain")
        assert hasattr(m, "render")
        assert callable(m.render)
        assert "Summary" in m.VIEWS
        assert "Canonical Correlations" in m.VIEWS

    def test_gui_app_imports(self):
        pytest.importorskip("streamlit")
        # gui.app transitively imports the Tracking tab, which depends on
        # `networkx` (optional). Skip if any transitive optional dep is
        # missing — that's not a regression in our wiring.
        pytest.importorskip("networkx")
        sys.modules.pop("gui.app", None)
        m = importlib.import_module("gui.app")
        assert m is not None


# ---------------------------------------------------------------------------
# Pure-compute pipeline (used by the GUI tab)
# ---------------------------------------------------------------------------

def _synthetic_rates(T=300, N_A=10, N_B=10, K_true=2, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, K_true))
    W_A = rng.standard_normal((K_true, N_A))
    W_B = rng.standard_normal((K_true, N_B))
    X_A = Z @ W_A + noise * rng.standard_normal((T, N_A))
    X_B = Z @ W_B + noise * rng.standard_normal((T, N_B))
    return X_A, X_B


class TestAnalyzePure:
    def test_pure_analyze_returns_full_payload(self):
        from ephys.inter_brain_dynamics import SharedSubspaceFit
        from ephys.run_inter_brain import _analyze
        X_A, X_B = _synthetic_rates(T=300, N_A=10, N_B=10, K_true=2)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        payload = _analyze(
            X_A, X_B, bin_centers,
            animal_ids=("631", "632"),
            bin_size=0.5, smoothing=None, t_window=None,
            n_components=2, max_K=10,
            method="regularized", reg=1e-3,
            cv_folds=3, n_shuffles=5, max_lag_bins=2,
            alpha=1.0, behavior_by_animal=None, seed=0,
        )
        assert set(payload.keys()) >= {
            "fit", "shuffle_null", "time_lagged", "cross_corr",
            "regression_results", "bin_centers", "animal_ids", "parameters",
        }
        assert isinstance(payload["fit"], SharedSubspaceFit)
        assert payload["shuffle_null"].shape == (5, 2)
        assert payload["cross_corr"].shape == (10, 10)
        lags, ccs = payload["time_lagged"]
        assert ccs.shape == (5, 2)  # 2*max_lag_bins + 1
        assert payload["regression_results"] is None
        assert payload["animal_ids"] == ("631", "632")
        assert payload["parameters"]["n_components"] == 2

    def test_pure_analyze_does_not_write_to_disk(self, tmp_path, monkeypatch):
        """_analyze is I/O-free; the only side-effects are logging."""
        from ephys.run_inter_brain import _analyze
        X_A, X_B = _synthetic_rates(T=300, N_A=8, N_B=8, K_true=2)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        monkeypatch.chdir(tmp_path)
        _analyze(
            X_A, X_B, bin_centers,
            animal_ids=("A", "B"),
            bin_size=0.5, smoothing=None, t_window=None,
            n_components=2, max_K=10,
            method="regularized", reg=1e-3,
            cv_folds=2, n_shuffles=3, max_lag_bins=2,
            alpha=1.0, behavior_by_animal=None, seed=0,
        )
        # No files in cwd (other than possibly .gui_cache from earlier work,
        # but _analyze itself must not have created any).
        cwd_files = list(tmp_path.iterdir())
        assert cwd_files == [], f"_analyze wrote files: {cwd_files}"
