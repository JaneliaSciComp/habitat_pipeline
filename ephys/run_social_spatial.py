#!/usr/bin/env python3
"""
Allocentric social place fields — command-line entry point.

Loads all requested animals from one session via :class:`MultiAnimalSession`,
builds each animal's ``(t, x, y, speed)`` on the shared ephys clock, computes
occupancy-normalized rate maps of every focal cell over every (self and partner)
animal's position, classifies cells by which conspecific(s) they encode, and
writes a results pickle, a six-panel summary PNG, and a multi-page PDF with the
per-cluster rate-map grids for the top-N cells by Skaggs information.

Mirrors the CLI structure of ``ephys/run_inter_brain.py``.

Example
-------
::

    python -m ephys.run_social_spatial \\
        --session_id 20251216 --animal_ids 631 632 633 \\
        --focal 631 \\
        --bin_size 5 --smoothing 5 --speed_threshold 5 \\
        --speed_filter_subject target \\
        --n_shuffles 500 --output_dir ./results
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute allocentric social place fields for a focal animal."
    )
    p.add_argument("--session_id", type=str, required=True,
                   help="Session identifier shared by all animals.")
    p.add_argument("--animal_ids", type=str, nargs="+", required=True,
                   help="All simultaneously-recorded animals to use as targets "
                        "(self + partners). The focal animal is added if absent.")
    p.add_argument("--focal", type=str, required=True,
                   help="Focal animal whose spikes generate the rate maps.")
    p.add_argument("--config_path", type=str, default=None,
                   help="Cohort config JSON (default: config/default_paths.json).")
    p.add_argument("--bin_size", type=float, default=5.0,
                   help="Spatial bin size in cm (or pixels if uncalibrated).")
    p.add_argument("--smoothing", type=float, default=5.0,
                   help="Gaussian smoothing sigma in cm.")
    p.add_argument("--speed_threshold", type=float, default=5.0,
                   help="Speed gate threshold in cm/s.")
    p.add_argument("--speed_filter_subject", type=str, default="target",
                   choices=["focal", "target", "none"],
                   help="Which animal's speed gates the time samples.")
    p.add_argument("--n_shuffles", type=int, default=500,
                   help="Number of shuffles for the significance null.")
    p.add_argument("--null_method", type=str, default="circular_shift",
                   choices=["circular_shift", "position_shuffle"])
    p.add_argument("--min_n_spikes", type=int, default=50,
                   help="Cells with fewer in-window spikes skip significance.")
    p.add_argument("--use_quality_cells", action="store_true",
                   help="Restrict to quality cells (firing-pattern filter).")
    p.add_argument("--t_start", type=float, default=None,
                   help="Window start in ephys seconds.")
    p.add_argument("--t_end", type=float, default=None,
                   help="Window end in ephys seconds.")
    p.add_argument("--top_n", type=int, default=20,
                   help="Cells (by max Skaggs across targets) for the PDF grids.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _max_skaggs_by_cluster(results) -> List[tuple]:
    """[(cluster_id, max_bits_per_spike_across_targets), ...] descending."""
    df = results.cell_classification
    targets = results.parameters.get("target_animals", [])
    bits_cols = [f"bits_per_spike_{t}" for t in targets
                 if f"bits_per_spike_{t}" in df.columns]
    if df.empty or not bits_cols:
        return []
    maxbits = df[bits_cols].max(axis=1)
    ranked = sorted(zip(df["cluster_id"].tolist(), maxbits.tolist()),
                    key=lambda kv: (np.nan_to_num(kv[1], nan=-np.inf)), reverse=True)
    return ranked


def _write_outputs(results, output_dir: Path, top_n: int) -> None:
    from ephys.social_spatial_plots import (
        plot_rate_maps_grid,
        plot_social_place_summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %s", pkl_path)

    fig = plot_social_place_summary(results)
    png_path = output_dir / "summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", png_path)

    ranked = _max_skaggs_by_cluster(results)[:top_n]
    pdf_path = output_dir / "rate_map_grids.pdf"
    with PdfPages(pdf_path) as pdf:
        if not ranked:
            fig = plt.figure(figsize=(6, 4))
            fig.text(0.5, 0.5, "No cells to plot", ha="center", va="center")
            pdf.savefig(fig)
            plt.close(fig)
        for cid, _ in ranked:
            try:
                fig = plot_rate_maps_grid(results, cluster_id=int(cid))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                logger.warning("Could not plot cluster %s: %s", cid, e)
    logger.info("Wrote %s (%d cells)", pdf_path, len(ranked))


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

    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from ingestion.multi_animal_session import MultiAnimalSession
        from ephys.social_spatial_fields import compute_social_place_fields
    except ImportError as e:
        logger.error("Import error: %s", e)
        return 1

    animal_ids = list(args.animal_ids)
    if args.focal not in animal_ids:
        animal_ids.append(args.focal)

    logger.info("Building MultiAnimalSession for %s / animals %s (focal %s)",
                args.session_id, animal_ids, args.focal)
    try:
        session = MultiAnimalSession(
            session_id=args.session_id,
            animal_ids=animal_ids,
            config_path=args.config_path,
        )
    except Exception as e:
        logger.error("Failed to build MultiAnimalSession: %s", e)
        return 1

    t_window = None
    if args.t_start is not None or args.t_end is not None:
        session_window = session.get_common_time_window()
        t0 = args.t_start if args.t_start is not None else session_window[0]
        t1 = args.t_end if args.t_end is not None else session_window[1]
        t_window = (t0, t1)

    try:
        ks = session.get_ks(args.focal)
    except Exception as e:
        logger.error("Failed to load focal KilosortData: %s", e)
        return 1

    logger.info("Computing social place fields...")
    try:
        results = compute_social_place_fields(
            ks, session, focal_animal=args.focal, target_animals=animal_ids,
            bin_size_cm=args.bin_size, smoothing_sigma_cm=args.smoothing,
            speed_threshold_cms=args.speed_threshold,
            speed_filter_subject=args.speed_filter_subject,
            n_shuffles=args.n_shuffles, null_method=args.null_method,
            min_n_spikes=args.min_n_spikes, use_quality_cells=args.use_quality_cells,
            t_window_ephys=t_window, seed=args.seed,
        )
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        return 1

    df = results.cell_classification
    if not df.empty:
        counts = df["category"].value_counts().to_dict()
        logger.info("Classified %d cells: %s", len(df), counts)

    _write_outputs(results, Path(args.output_dir), args.top_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
