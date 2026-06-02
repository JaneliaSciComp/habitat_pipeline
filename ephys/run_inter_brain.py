#!/usr/bin/env python3
"""
Inter-brain shared-subspace analysis — command-line entry point.

Loads two simultaneously-recorded animals via :class:`MultiAnimalSession`,
bins their spike trains onto a common ephys-second grid, fits the shared
subspace (regularized CCA by default), runs the circular-shift shuffle
null, time-lagged CCA, cross-animal correlation matrix, and (optionally,
when tracking is available) a ridge regression of each shared dim onto
self vs partner behavior. Writes a results pickle and a six-panel
summary PNG into the output directory.

Mirrors the CLI structure of ``ephys/decode_opponent_identity.py``.

Example
-------
::

    python -m ephys.run_inter_brain \\
        --session_id 20251216 --animal_ids 631 632 \\
        --bin_size 0.5 --smoothing 0.25 \\
        --max_K 20 --n_shuffles 200 \\
        --behavior_type EC --output_dir ./results
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ephys.inter_brain_dynamics import (
    cross_animal_correlation_matrix,
    fit_shared_subspace,
    regress_shared_on_behavior,
    shuffle_null_subspace,
    time_lagged_cca,
)
from ephys.inter_brain_plots import plot_inter_brain_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit an inter-brain shared subspace between two animals.",
    )
    p.add_argument("--session_id", type=str, required=True,
                   help="Session identifier shared by both animals.")
    p.add_argument("--animal_ids", type=str, nargs="+", required=True,
                   help="Animal identifiers (exactly two for the current "
                        "two-brain implementation).")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to a cohort config JSON (default: "
                        "config/default_paths.json under the repo).")
    p.add_argument("--bin_size", type=float, default=0.5,
                   help="Spike-bin width in seconds.")
    p.add_argument("--smoothing", type=float, default=None,
                   help="Optional Gaussian smoothing sigma in seconds "
                        "(applied per cell along time after binning).")
    p.add_argument("--t_start", type=float, default=None,
                   help="Start of the analysis window in ephys seconds "
                        "(default: 0).")
    p.add_argument("--t_end", type=float, default=None,
                   help="End of the analysis window in ephys seconds "
                        "(default: min of per-animal durations).")
    p.add_argument("--n_components", type=int, default=None,
                   help="Shared-subspace dimensionality K (default: "
                        "min(N_A, N_B, T//4, 10)).")
    p.add_argument("--max_K", type=int, default=20,
                   help="Cap on n_components when None is used (a no-op "
                        "if n_components is set explicitly).")
    p.add_argument("--method", type=str, default="regularized",
                   choices=["regularized", "cca", "pls"],
                   help="Fit method. 'regularized' (default) handles N≳T.")
    p.add_argument("--reg", type=float, default=1e-3,
                   help="Ridge added to per-animal cell covariance before "
                        "whitening (regularized method only).")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Contiguous-block CV folds (no shuffling).")
    p.add_argument("--n_shuffles", type=int, default=200,
                   help="Circular-shift shuffles for the null distribution.")
    p.add_argument("--max_lag_bins", type=int, default=10,
                   help="Maximum lag (bins) for the time-lagged CCA sweep.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge alpha for behavior regression.")
    p.add_argument("--behavior_type", type=str, default=None,
                   help="Restrict event indicator columns to this single "
                        "type (default: all event types).")
    p.add_argument("--event_window", type=float, default=1.0,
                   help="± window (s) around each event_start for indicator "
                        "columns.")
    p.add_argument("--skip_regression", action="store_true",
                   help="Skip behavior loading + regression even if "
                        "tracking is available.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the shuffle null.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory; per-run sub-directory will be "
                        "created inside it.")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ---------------------------------------------------------------------------
# Inner pipeline (factored so it is testable without disk-backed sessions)
# ---------------------------------------------------------------------------

def _analyze(
    X_A: np.ndarray,
    X_B: np.ndarray,
    bin_centers: np.ndarray,
    animal_ids: Tuple[str, str],
    *,
    bin_size: float,
    smoothing: Optional[float],
    t_window: Optional[Tuple[float, float]],
    n_components: Optional[int],
    max_K: int,
    method: str,
    reg: float,
    cv_folds: int,
    n_shuffles: int,
    max_lag_bins: int,
    alpha: float,
    behavior_by_animal: Optional[Dict[str, "object"]] = None,
    seed: int = 0,
) -> Dict:
    """Pure-compute analysis pipeline (no I/O). Returns the result payload.

    Used directly by the GUI tab; the CLI's :func:`_analyze_and_save`
    wraps this with pickle + PNG output.
    """
    if n_components is None:
        n_components = max(
            1, min(X_A.shape[1], X_B.shape[1], X_A.shape[0] // 4, int(max_K)),
        )

    logger.info("Fitting shared subspace (K=%d, method=%s)", n_components, method)
    fit = fit_shared_subspace(
        X_A, X_B,
        n_components=n_components, reg=reg, method=method, cv_folds=cv_folds,
        animal_ids=animal_ids,
        t_window=t_window,
        bin_size_sec=bin_size,
        smoothing_sigma_sec=smoothing,
    )

    logger.info("Running circular-shift shuffle null (n_shuffles=%d)", n_shuffles)
    null = shuffle_null_subspace(
        X_A, X_B,
        n_components=n_components, n_shuffles=n_shuffles,
        reg=reg, method=method, seed=seed,
    )

    logger.info("Running time-lagged CCA (max_lag_bins=±%d)", max_lag_bins)
    lags, ccs = time_lagged_cca(
        X_A, X_B,
        max_lag_bins=max_lag_bins, n_components=n_components,
        reg=reg, method=method,
    )

    logger.info("Computing cross-animal cell-pair correlation matrix")
    cross_corr = cross_animal_correlation_matrix(X_A, X_B)

    regression_results = None
    if behavior_by_animal:
        try:
            logger.info("Running behavior regression (alpha=%.3g)", alpha)
            regression_results = regress_shared_on_behavior(
                fit, behavior_by_animal,
                alpha=alpha, cv_folds=cv_folds,
            )
        except Exception as e:
            logger.warning("Behavior regression failed: %s", e)

    return {
        "fit": fit,
        "shuffle_null": null,
        "time_lagged": (lags, ccs),
        "cross_corr": cross_corr,
        "regression_results": regression_results,
        "bin_centers": np.asarray(bin_centers),
        "animal_ids": tuple(animal_ids),
        "parameters": {
            "bin_size": bin_size,
            "smoothing": smoothing,
            "t_window": t_window,
            "n_components": int(n_components),
            "method": method,
            "reg": reg,
            "cv_folds": int(cv_folds),
            "n_shuffles": int(n_shuffles),
            "max_lag_bins": int(max_lag_bins),
            "alpha": alpha,
            "seed": int(seed),
        },
    }


def _analyze_and_save(
    X_A: np.ndarray,
    X_B: np.ndarray,
    bin_centers: np.ndarray,
    animal_ids: Tuple[str, str],
    output_dir: Path,
    *,
    bin_size: float,
    smoothing: Optional[float],
    t_window: Optional[Tuple[float, float]],
    n_components: Optional[int],
    max_K: int,
    method: str,
    reg: float,
    cv_folds: int,
    n_shuffles: int,
    max_lag_bins: int,
    alpha: float,
    behavior_by_animal: Optional[Dict[str, "object"]] = None,
    seed: int = 0,
) -> Dict:
    """Run the full inter-brain analysis and write pickle + PNG.

    Thin wrapper around :func:`_analyze` that additionally pickles the
    payload and saves the six-panel summary PNG.
    """
    payload = _analyze(
        X_A, X_B, bin_centers, animal_ids,
        bin_size=bin_size, smoothing=smoothing, t_window=t_window,
        n_components=n_components, max_K=max_K,
        method=method, reg=reg, cv_folds=cv_folds,
        n_shuffles=n_shuffles, max_lag_bins=max_lag_bins, alpha=alpha,
        behavior_by_animal=behavior_by_animal, seed=seed,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %s", pkl_path)

    fit = payload["fit"]
    t_bins = (
        np.asarray(bin_centers)[fit.valid_mask] if bin_centers is not None else None
    )
    fig = plot_inter_brain_summary(
        fit,
        shuffle_null=payload["shuffle_null"],
        t_bins=t_bins,
        cross_corr=payload["cross_corr"],
        time_lagged=payload["time_lagged"],
        regression_results=payload["regression_results"],
        bin_size_sec=bin_size,
    )
    png_path = output_dir / "summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)

    return payload


# ---------------------------------------------------------------------------
# CLI orchestrator
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(args.animal_ids) != 2:
        logger.error("This CLI currently supports exactly two animals; got %d.",
                     len(args.animal_ids))
        return 1
    a_id, b_id = args.animal_ids

    sys.path.append(str(Path(__file__).parent.parent))

    try:
        from ingestion.multi_animal_session import MultiAnimalSession
        from video.behavior_features import build_behavior_feature_matrix
        from video.tracking_import import load_tracking_data
    except ImportError as e:  # pragma: no cover — defensive
        logger.error("Import error: %s", e)
        return 1

    logger.info("Building MultiAnimalSession for %s / animals %s",
                args.session_id, args.animal_ids)
    try:
        session = MultiAnimalSession(
            session_id=args.session_id,
            animal_ids=list(args.animal_ids),
            config_path=args.config_path,
        )
    except Exception as e:
        logger.error("Failed to build MultiAnimalSession: %s", e)
        return 1

    try:
        bin_centers, rates_by_animal = session.get_common_binned_rates(
            bin_size_sec=args.bin_size,
            t_start_ephys=args.t_start,
            t_end_ephys=args.t_end,
            smoothing_sigma_sec=args.smoothing,
            use_cache=True,
        )
    except Exception as e:
        logger.error("Failed to build common binned rates: %s", e)
        return 1

    X_A = rates_by_animal[a_id].T  # (T, N_A)
    X_B = rates_by_animal[b_id].T  # (T, N_B)
    logger.info("Binned rates: X_A=%s, X_B=%s", X_A.shape, X_B.shape)

    behavior_by_animal: Optional[Dict[str, "object"]] = None
    if not args.skip_regression:
        try:
            logger.info("Loading tracking + building behavior features")
            tracking = load_tracking_data(
                session.dsm_by_animal[session.sync_from_animal],
            )
            try:
                tracking.synchronize_with_ephys(session.sync)
            except Exception as e:
                logger.warning("Tracking sync failed (will rely on sync arg): %s", e)
            event_types = [args.behavior_type] if args.behavior_type else None
            beh_A = build_behavior_feature_matrix(
                tracking, session.events, session.sync, bin_centers,
                focal=a_id, partner=b_id,
                event_window_sec=args.event_window, event_types=event_types,
            )
            beh_B = build_behavior_feature_matrix(
                tracking, session.events, session.sync, bin_centers,
                focal=b_id, partner=a_id,
                event_window_sec=args.event_window, event_types=event_types,
            )
            behavior_by_animal = {a_id: beh_A, b_id: beh_B}
        except Exception as e:
            logger.warning("Skipping behavior regression: %s", e)

    out_dir = Path(args.output_dir) / (
        f"inter_brain_{args.session_id}_" + "_".join(sorted(args.animal_ids))
    )

    try:
        _analyze_and_save(
            X_A, X_B, bin_centers,
            animal_ids=(a_id, b_id),
            output_dir=out_dir,
            bin_size=args.bin_size,
            smoothing=args.smoothing,
            t_window=(
                float(bin_centers[0]) - args.bin_size / 2,
                float(bin_centers[-1]) + args.bin_size / 2,
            ),
            n_components=args.n_components,
            max_K=args.max_K,
            method=args.method,
            reg=args.reg,
            cv_folds=args.cv_folds,
            n_shuffles=args.n_shuffles,
            max_lag_bins=args.max_lag_bins,
            alpha=args.alpha,
            behavior_by_animal=behavior_by_animal,
            seed=args.seed,
        )
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
