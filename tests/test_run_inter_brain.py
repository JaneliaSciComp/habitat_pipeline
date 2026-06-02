"""
Tests for ephys/run_inter_brain.py.

Covers:
* argparse contract (the example invocation from the prompt parses)
* the inner pipeline ``_analyze_and_save`` runs end-to-end on synthetic
  binned rates and writes ``results.pkl`` + ``summary.png`` with the
  expected payload shape (no disk-backed MultiAnimalSession required)
"""

import matplotlib

matplotlib.use("Agg")

import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ephys.inter_brain_dynamics import SharedSubspaceFit  # noqa: E402
from ephys.run_inter_brain import _analyze_and_save, _build_parser  # noqa: E402


# ---------------------------------------------------------------------------
# Argparse contract
# ---------------------------------------------------------------------------

class TestArgParser:
    def test_prompt_example_parses(self):
        parser = _build_parser()
        argv = [
            "--session_id", "20251216",
            "--animal_ids", "631", "632",
            "--bin_size", "0.5",
            "--smoothing", "0.25",
            "--max_K", "20",
            "--n_shuffles", "200",
            "--behavior_type", "EC",
            "--output_dir", "./results",
        ]
        args = parser.parse_args(argv)
        assert args.session_id == "20251216"
        assert args.animal_ids == ["631", "632"]
        assert args.bin_size == 0.5
        assert args.smoothing == 0.25
        assert args.max_K == 20
        assert args.n_shuffles == 200
        assert args.behavior_type == "EC"
        assert args.output_dir == "./results"
        # Defaults preserved.
        assert args.method == "regularized"
        assert args.cv_folds == 5
        assert args.alpha == 1.0
        assert args.skip_regression is False

    def test_required_args_enforced(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing required

    def test_choices_enforced(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--session_id", "S", "--animal_ids", "A", "B",
                "--output_dir", "/tmp", "--method", "bogus",
            ])


# ---------------------------------------------------------------------------
# Inner pipeline
# ---------------------------------------------------------------------------

def _synthetic_rates(T=400, N_A=12, N_B=10, K_true=3, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, K_true))
    W_A = rng.standard_normal((K_true, N_A))
    W_B = rng.standard_normal((K_true, N_B))
    X_A = Z @ W_A + noise * rng.standard_normal((T, N_A))
    X_B = Z @ W_B + noise * rng.standard_normal((T, N_B))
    return X_A, X_B


class TestAnalyzeAndSave:
    def test_runs_and_writes_artifacts_without_regression(self, tmp_path):
        X_A, X_B = _synthetic_rates(T=400, N_A=12, N_B=10, K_true=3)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        out_dir = tmp_path / "run1"
        payload = _analyze_and_save(
            X_A, X_B, bin_centers,
            animal_ids=("631", "632"),
            output_dir=out_dir,
            bin_size=0.5, smoothing=None,
            t_window=(0.0, float(bin_centers[-1])),
            n_components=3, max_K=20,
            method="regularized", reg=1e-3,
            cv_folds=3, n_shuffles=10, max_lag_bins=3,
            alpha=1.0,
            behavior_by_animal=None,
            seed=0,
        )
        assert isinstance(payload["fit"], SharedSubspaceFit)
        assert payload["shuffle_null"].shape == (10, 3)
        assert payload["regression_results"] is None
        assert (out_dir / "results.pkl").exists()
        assert (out_dir / "summary.png").exists()
        # Pickle round-trip preserves the payload.
        with open(out_dir / "results.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded["fit"], SharedSubspaceFit)
        assert loaded["animal_ids"] == ("631", "632")
        assert loaded["parameters"]["n_components"] == 3
        assert loaded["parameters"]["n_shuffles"] == 10

    def test_runs_with_regression(self, tmp_path):
        X_A, X_B = _synthetic_rates(T=400, N_A=10, N_B=10, K_true=3)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        T = X_A.shape[0]
        rng = np.random.default_rng(7)
        beh_A = pd.DataFrame({
            "speed": rng.standard_normal(T),
            "distance": rng.standard_normal(T),
        })
        beh_B = pd.DataFrame({
            "speed": rng.standard_normal(T),
            "distance": rng.standard_normal(T),
        })
        out_dir = tmp_path / "run2"
        payload = _analyze_and_save(
            X_A, X_B, bin_centers,
            animal_ids=("631", "632"),
            output_dir=out_dir,
            bin_size=0.5, smoothing=None,
            t_window=(0.0, float(bin_centers[-1])),
            n_components=3, max_K=20,
            method="regularized", reg=1e-3,
            cv_folds=3, n_shuffles=10, max_lag_bins=3,
            alpha=1.0,
            behavior_by_animal={"631": beh_A, "632": beh_B},
            seed=0,
        )
        assert payload["regression_results"] is not None
        assert "631" in payload["regression_results"]
        assert "632" in payload["regression_results"]
        assert (out_dir / "results.pkl").exists()
        assert (out_dir / "summary.png").exists()

    def test_n_components_default_clipped_by_max_K(self, tmp_path):
        X_A, X_B = _synthetic_rates(T=400, N_A=50, N_B=50, K_true=3)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        out_dir = tmp_path / "run3"
        payload = _analyze_and_save(
            X_A, X_B, bin_centers,
            animal_ids=("631", "632"),
            output_dir=out_dir,
            bin_size=0.5, smoothing=None,
            t_window=None,
            n_components=None, max_K=4,
            method="regularized", reg=1e-3,
            cv_folds=3, n_shuffles=5, max_lag_bins=2,
            alpha=1.0,
            behavior_by_animal=None,
            seed=0,
        )
        assert payload["fit"].n_components == 4
        assert payload["parameters"]["n_components"] == 4

    def test_creates_parent_directories(self, tmp_path):
        X_A, X_B = _synthetic_rates(T=300, N_A=8, N_B=8)
        bin_centers = np.arange(X_A.shape[0]) * 0.5
        out_dir = tmp_path / "deep" / "nested" / "run"
        assert not out_dir.exists()
        _analyze_and_save(
            X_A, X_B, bin_centers,
            animal_ids=("A", "B"),
            output_dir=out_dir,
            bin_size=0.5, smoothing=None, t_window=None,
            n_components=2, max_K=10,
            method="regularized", reg=1e-3,
            cv_folds=2, n_shuffles=5, max_lag_bins=2,
            alpha=1.0,
            behavior_by_animal=None,
            seed=0,
        )
        assert out_dir.exists()
        assert (out_dir / "results.pkl").exists()
        assert (out_dir / "summary.png").exists()
