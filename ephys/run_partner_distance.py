#!/usr/bin/env python3
"""
Partner-distance decoding — command-line entry point.

Loads the focal (implanted) animal's ``KilosortData`` plus the session
tracking (which already contains the partner), bins the focal's spike trains
and the focal↔partner distance onto a common ephys-second grid, and runs the
single-cell + population regression decode (with focal self-motion partialled
out and a circular-shift null). Only the focal animal needs ephys. Writes a
results pickle and a summary PNG.

Mirrors the CLI structure of ``ephys/run_inter_brain.py``.

Example
-------
::

    python -m ephys.run_partner_distance \\
        --session_id 20251216 --animal_ids 631 632 \\
        --bin_size 0.5 --smoothing 0.25 --alpha 1.0 --cv_folds 5 \\
        --null shuffle --n_shuffles 100 --output_dir ./results
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ephys.decode_partner_distance import (
    _analyze,
    build_distance_binned_data,
    load_partner_distance_inputs,
)
from ephys.decode_partner_distance_plots import plot_partner_distance_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode focal-partner distance from neural activity "
                    "(regression).",
    )
    p.add_argument("--session_id", type=str, required=True,
                   help="Session identifier shared by both animals.")
    p.add_argument("--animal_ids", type=str, nargs=2, required=True,
                   metavar=("FOCAL", "PARTNER"),
                   help="Focal (implanted) animal then the specific partner.")
    p.add_argument("--config_path", type=str, default=None,
                   help="Path to a cohort config JSON (default: "
                        "config/default_paths.json).")
    p.add_argument("--dio_channel", type=int, default=1,
                   help="DIO channel for the focal animal's ephys↔behavior sync.")
    p.add_argument("--bin_size", type=float, default=0.5,
                   help="Spike-bin width in seconds.")
    p.add_argument("--smoothing", type=float, default=None,
                   help="Optional Gaussian smoothing sigma (s) applied per "
                        "cell along time after binning.")
    p.add_argument("--t_start", type=float, default=None,
                   help="Start of the analysis window in ephys seconds.")
    p.add_argument("--t_end", type=float, default=None,
                   help="End of the analysis window in ephys seconds.")
    p.add_argument("--n_distance_bins", type=int, default=15,
                   help="Number of bins for the 1-D distance tuning curves.")
    p.add_argument("--tuning_smoothing", type=float, default=1.0,
                   help="Gaussian smoothing of tuning curves (distance-bin "
                        "units; 0 = none).")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge alpha for the regression.")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Contiguous-block CV folds (no shuffling).")
    p.add_argument("--null", type=str, default="shuffle",
                   choices=["shuffle", "reverse", "none"],
                   help="Null baseline from a broken rate-distance pairing.")
    p.add_argument("--n_shuffles", type=int, default=100,
                   help="Circular shifts when --null shuffle.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the shuffle null.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory; a per-run sub-directory is created "
                        "inside it.")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ---------------------------------------------------------------------------
# Save wrapper around the pure-compute _analyze
# ---------------------------------------------------------------------------

def _analyze_and_save(data: dict, output_dir: Path, *,
                      alpha: float, cv_folds: int, n_distance_bins: int,
                      tuning_smoothing_sigma: float,
                      null: Optional[str], n_shuffles: int, seed: int) -> dict:
    """Run ``_analyze`` on pre-binned data and write pickle + summary PNG."""
    result = _analyze(
        data["firing_rates"], data["distance"], data["nuisance"],
        data["bin_centers"], data["units"], data["focal"], data["partner"],
        alpha=alpha, cv_folds=cv_folds,
        n_distance_bins=n_distance_bins,
        tuning_smoothing_sigma=tuning_smoothing_sigma,
        null=(None if null == "none" else null), n_shuffles=n_shuffles,
        nuisance_names=data["nuisance_names"], seed=seed,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %s", pkl_path)

    if result.get("status") == "success":
        fig = plot_partner_distance_summary(result)
        if fig is not None:
            png_path = output_dir / "summary.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Wrote %s", png_path)
        logger.info("CV R2=%.3f  RMSE=%.2f %s  null R2=%.3f",
                    result["cv_r2"], result["rmse"], result["units"],
                    result.get("null_r2", float("nan")))
    else:
        logger.warning("Decode status: %s", result.get("status"))

    return result


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
    focal, partner = args.animal_ids

    sys.path.append(str(Path(__file__).parent.parent))

    logger.info("Loading focal %s ephys + session tracking for %s (partner %s)",
                focal, args.session_id, partner)
    try:
        inputs = load_partner_distance_inputs(
            args.session_id, focal,
            config_path=args.config_path, dio_channel=args.dio_channel,
        )
    except Exception as e:
        logger.error("Failed to load inputs: %s", e)
        return 1

    try:
        data = build_distance_binned_data(
            inputs.ks_focal, inputs.tracking, inputs.sync, focal, partner,
            pixels_per_cm=inputs.pixels_per_cm,
            bin_size=args.bin_size, smoothing_sigma_sec=args.smoothing,
            t_start=args.t_start, t_end=args.t_end,
        )
    except Exception as e:
        logger.error("Failed to build binned data: %s", e)
        return 1
    logger.info("Binned: %d bins x %d cells (units=%s)",
                len(data["distance"]), data["n_cells"], data["units"])

    out_dir = Path(args.output_dir) / (
        f"partner_distance_{args.session_id}_{focal}_{partner}"
    )
    try:
        _analyze_and_save(
            data, out_dir,
            alpha=args.alpha, cv_folds=args.cv_folds,
            n_distance_bins=args.n_distance_bins,
            tuning_smoothing_sigma=args.tuning_smoothing,
            null=args.null, n_shuffles=args.n_shuffles, seed=args.seed,
        )
    except Exception as e:
        logger.exception("Analysis failed: %s", e)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
