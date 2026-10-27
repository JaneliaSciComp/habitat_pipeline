"""Rest-distance coding runner: does the focal rat's activity track partner distance while it rests at its own home base?

    python -m ephys.run_rest_distance --session_id 20251216_145034 --animal_id 630 --gates_only
    python -m ephys.run_rest_distance --session_id 20251216_145034 --animal_id 630 --smoke
    python -m ephys.run_rest_distance --session_id 20251216_145034 --animal_id 630 --purpose exploratory

Statistics live in ``ephys/rest_distance_coding.py`` (calibrated on synthetic sessions in
``tests/test_rest_distance_calibration.py``); this module only builds its inputs from real chunks.

Per APT chunk that lies inside the focal's quality-cell spike window:
    * home base = argmax of the smoothed low-speed dwell map of *that chunk* (``video.territory_behavior``); rest
      bins = focal within ``--home_frac_diag`` of the arena diagonal of it and slower than ``--still_px_s``
      (selection uses only the focal's own position and speed, never the partner or the neural data);
    * target ``d`` = log pixel distance to the nearest other tracked rat, on 0.5 s bins;
    * nuisance = log1p of the focal's centre speed, mean keypoint speed and mean posture (keypoint minus centre)
      speed, i.e. movement that is not the partner;
    * cells with fewer than ``--min_rest_spikes`` spikes in rest bins are dropped *before* testing (outcome-independent);
    * a chunk with fewer than ``--min_rest_min`` minutes of usable rest (after the >= ``--min_seg_sec`` segment rule
      and high-passing) is reported as ``insufficient_rest`` and not tested.

What a positive result means: population/cell activity covaries with fast changes in partner distance at a fixed
spot, after removing slow drift and the movement nuisance. It is not territory coding, and it is not separated from
arousal, partner approach sensory input, or any movement the keypoints do not capture (the calibration shows noisy
arousal proxies leak into the test).

Units are pixels (``pixels_per_cm=None``; the configured calibration is wrong for APT, see CLAUDE.md). Defaults marked
provisional are guesses, not validated thresholds.

This runner does **not** write to the lab notebook: families, holdouts and frozen predictions are the scientist's acts.
It refuses held-out targets (``assert_not_held_out``, ``multi_animal=True``) before touching any data, and reports
the FDR resolution of the budget it used (per chunk; the declared family's denominator governs any claim).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ephys import rest_distance_coding as rd
from ephys._stats_utils import fdr_resolution
from ephys.run_territory import (DT, apt_chunk_dirs, bin_counts, chunk_ephys_window, chunk_inputs, git_commit,
                                 interp_on_grid, save)
from video.territory_behavior import arena_bounds, chunk_summary, speed_px_s

logger = logging.getLogger(__name__)

MIN_FRAME_DT = 1e-3
MAX_FRAME_DT = 0.25       # frame-to-frame differences across longer gaps are not motion


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def keypoint_motion(obj: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-frame keypoint motion of one animal: ``t`` (ephys s), ``kp_speed`` and ``posture_speed`` (px/s).

    ``kp_speed`` is the mean over keypoints of the keypoint's speed (includes translation of the whole body);
    ``posture_speed`` is the same after subtracting the centre, i.e. limb/head movement at a fixed body position.
    Returns ``None`` when the animal has no keypoint columns or no ephys timestamps.
    """
    if "ephys_timestamps" not in obj.columns:
        return None
    ids = sorted(int(m.group(1)) for c in obj.columns if (m := re.fullmatch(r"kp(\d+)_x", c))
                 and f"kp{m.group(1)}_y" in obj.columns)
    if not ids:
        return None
    o = obj.sort_values("ephys_timestamps")
    t = o["ephys_timestamps"].to_numpy(float)
    cx, cy = o["center_x"].to_numpy(float), o["center_y"].to_numpy(float)
    X = np.column_stack([o[f"kp{i}_x"].to_numpy(float) for i in ids])
    Y = np.column_stack([o[f"kp{i}_y"].to_numpy(float) for i in ids])
    dt = np.diff(t)
    ok = (dt > MIN_FRAME_DT) & (dt < MAX_FRAME_DT)
    with np.errstate(invalid="ignore", divide="ignore"):
        kp = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)).mean(axis=1) / dt
        post = np.hypot(np.diff(X - cx[:, None], axis=0), np.diff(Y - cy[:, None], axis=0)).mean(axis=1) / dt
    return pd.DataFrame(dict(t=0.5 * (t[1:] + t[:-1]), kp_speed=np.where(ok, kp, np.nan),
                             posture_speed=np.where(ok, post, np.nan)))


def bin_mean(t: np.ndarray, v: np.ndarray, centers: np.ndarray, dt: float = DT) -> np.ndarray:
    """Mean of ``v`` in each ``dt`` bin centred on ``centers`` (NaN where no finite sample falls in the bin)."""
    edges = centers[0] - dt / 2 + np.arange(len(centers) + 1) * dt
    ok = np.isfinite(v) & np.isfinite(t)
    s, _ = np.histogram(t[ok], bins=edges, weights=v[ok])
    n, _ = np.histogram(t[ok], bins=edges)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def bridge_gaps(mask: np.ndarray, max_gap_bins: int) -> np.ndarray:
    """Fill runs of ``False`` no longer than ``max_gap_bins`` that sit between two ``True`` runs.

    Tracking noise makes a rat that is resting flicker over the speed gate (on rat630/20251216_145034 the median
    rest run was 2.5-3 s with ~200 runs per chunk); without bridging, almost all rest falls under the segment rule.
    """
    mask = np.asarray(mask, bool)
    out = mask.copy()
    if max_gap_bins < 1 or not mask.any():
        return out
    edges = np.flatnonzero(np.diff(np.r_[1, mask.astype(int), 1]))     # alternating starts of False/True runs
    for a, b in zip(edges[::2], edges[1::2]):                           # False runs [a, b)
        if a > 0 and b < len(mask) and (b - a) <= max_gap_bins and mask[a - 1] and mask[b]:
            out[a:b] = True
    return out


def rest_inputs(tracks: Dict[str, pd.DataFrame], motion: Optional[pd.DataFrame], focal: str, window: Tuple[float, float],
                *, home, radius: float, still_px_s: float, bridge_sec: float = 0.0) -> Dict[str, object]:
    """Bin grid, log nearest-partner distance, nuisance matrix and rest mask for one chunk."""
    inp = chunk_inputs(tracks, focal, window)
    f = tracks[focal]
    slow = interp_on_grid(f.assign(speed=speed_px_s(f)), inp["centers"])["speed"]
    with np.errstate(invalid="ignore"):
        at_home = np.hypot(inp["x"] - home[0], inp["y"] - home[1]) < radius
        rest = at_home & (slow < still_px_s)
    # bridge brief speed flickers, but never across a bin where the rat left the home radius or was untracked
    rest = bridge_gaps(rest, int(round(bridge_sec / DT))) & np.isfinite(slow) & at_home
    d = np.log(np.maximum(inp["partner_dist"], 1.0))
    cols = [np.log1p(inp["speed"])]
    names = ["log1p_speed"]
    if motion is not None:
        for c in ("kp_speed", "posture_speed"):
            cols.append(np.log1p(bin_mean(motion["t"].to_numpy(), motion[c].to_numpy(), inp["centers"])))
            names.append(f"log1p_{c}")
    return dict(inp=inp, d=d, nuis=np.column_stack(cols), nuis_names=names, rest=rest, slow_speed=slow)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chunk_with_motion(path: str, sync, focal: str):
    """Tracks on the ephys clock (pixels) for every animal plus the focal's keypoint motion (or ``None``)."""
    from video.tracking_import import load_tracking_data, resolve_tracking_on_ephys_clock
    tr = load_tracking_data(path)
    names = list(tr.parsed_data.keys())
    tracks = resolve_tracking_on_ephys_clock(tr, sync, names, pixels_per_cm=None)
    obj = tr.get_object_data(focal)
    motion = keypoint_motion(obj) if obj is not None else None
    return tracks, motion


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------

def prepare_chunk(name: str, tracks, motion, focal: str, window, cluster_ids, spike_times, *, bounds, args):
    """Everything up to the statistics: ``(res, P, keep)``; ``P`` is ``None`` when the chunk cannot be prepared."""
    if focal not in tracks:
        return dict(chunk=name, status="focal_not_tracked"), None, None
    sm = chunk_summary({focal: tracks[focal]}, bounds, min_tracked_sec=args.min_tracked_sec)
    if focal not in sm:
        return dict(chunk=name, status="focal_barely_tracked"), None, None
    home = sm[focal]["home"]
    radius = args.home_frac_diag * float(np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]))
    R = rest_inputs(tracks, motion, focal, window, home=home, radius=radius, still_px_s=args.still_px_s,
                    bridge_sec=args.bridge_sec)
    inp = R["inp"]
    res: Dict[str, object] = dict(chunk=name, home_px=[round(float(home[0])), round(float(home[1]))],
                                  radius_px=round(radius), nuisance=R["nuis_names"],
                                  raw_rest_min=round(float(R["rest"].sum() * DT / 60), 2),
                                  has_keypoints=motion is not None)
    if motion is None:
        res["warning"] = "no keypoint motion: nuisance is centre speed only (weaker control)"
    counts = bin_counts(spike_times, inp["t0"], inp["n_bins"])
    rest_spikes = counts[R["rest"]].sum(axis=0)
    keep = rest_spikes >= args.min_rest_spikes
    res["n_cells_total"], res["n_cells_tested"] = int(len(keep)), int(keep.sum())
    if args.smoke and args.max_cells:
        keep &= np.cumsum(keep) <= args.max_cells
        res["n_cells_tested"] = int(keep.sum())
    if keep.sum() < 3:
        res["status"] = "too_few_cells"
        return res, None, keep
    P = rd.prepare(counts[:, keep], R["d"], R["nuis"], R["rest"], dt=DT, window_sec=args.window_sec,
                   min_seg_sec=args.min_seg_sec)
    res.update(rest_min=round(P["rest_min"], 2), n_segments=P["n_segments"], d_sd=round(P["d_sd"], 3))
    return res, P, keep


def analyse_chunk(name: str, tracks, motion, focal: str, window, cluster_ids, spike_times, *, bounds, args) -> Dict:
    t_start = time.time()
    res, P, keep = prepare_chunk(name, tracks, motion, focal, window, cluster_ids, spike_times, bounds=bounds,
                                 args=args)
    if P is None:
        return res
    if P["rest_min"] < args.min_rest_min:
        res["status"] = "insufficient_rest"
        return res
    if args.gates_only:
        res["status"] = "gates_only"
        return res
    cells = rd.single_cell_tests(P, n_shifts=args.n_shifts, seed=args.seed)
    cells.insert(0, "cluster_id", np.asarray(cluster_ids)[keep])
    pop = rd.population_test(P, n_shifts=args.n_shifts, seed=args.seed)
    res.update(status="ok", population=pop, cells=cells,
               n_p05=int((cells.p_value <= 0.05).sum()), n_q05=int((cells.q_value <= 0.05).sum()),
               fdr_resolution=fdr_resolution(int(keep.sum()), args.n_shifts, args.alpha),
               seconds=round(time.time() - t_start, 1))
    return res


def replication(results: List[Dict]) -> Dict[str, object]:
    """Cross-chunk agreement of the per-cell partial correlations (same cells, different chunks)."""
    ok = [r for r in results if r.get("status") == "ok"]
    if len(ok) < 2:
        return dict(status="needs_2_tested_chunks")
    R = pd.concat([r["cells"].set_index("cluster_id")["r"].rename(r["chunk"]) for r in ok], axis=1)
    pairs = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = R.iloc[:, i], R.iloc[:, j]
            m = a.notna() & b.notna()
            if m.sum() >= 5:
                pairs.append(dict(a=R.columns[i], b=R.columns[j], n_cells=int(m.sum()),
                                  r_between_chunks=float(np.corrcoef(a[m], b[m])[0, 1])))
    return dict(status="ok", pairs=pairs,
                mean_r_between_chunks=float(np.mean([p["r_between_chunks"] for p in pairs])) if pairs else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> Dict[str, object]:
    from database.lab_notebook import LabNotebook
    LabNotebook().assert_not_held_out(args.session_id, args.animal_id, purpose=args.purpose,
                                      hypothesis_id=args.hypothesis_id, multi_animal=True)

    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    from ingestion.kilosort_data_import import load_kilosort_data

    focal = f"rat{args.animal_id}"
    dsm = DataStorageManager(args.animal_id, args.session_id, config_path=args.config_path)
    ks_path = dsm.get_kilosort_path()
    if ks_path is None:
        return dict(session_id=args.session_id, animal=focal, status="no_ephys_for_this_animal_and_recording",
                    recording_id=dsm.recording_id, purpose=args.purpose, smoke=args.smoke)
    ks = load_kilosort_data(ks_path)
    sync = DataSyncManager(dsm, dio_channel=args.dio_channel)
    cluster_ids, spike_times = ks.get_filtered_cells_spike_times()
    win = ks.quality_ephys_window()

    cfg = json.load(open(args.config_path))
    roots = cfg["tracking"] if isinstance(cfg["tracking"], list) else [cfg["tracking"]]
    apt = next((r for r in roots if "APT" in str(r)), None)
    if apt is None:
        raise RuntimeError("no APT tracking root in config")

    feasibility, keep = [], {}
    for name, path in apt_chunk_dirs(apt, args.session_id[:8]):
        lo, hi = chunk_ephys_window(path, sync)
        overlap = max(0.0, min(hi, win[1]) - max(lo, win[0])) / max(hi - lo, 1e-9)
        use = overlap >= args.min_chunk_overlap
        feasibility.append(dict(chunk=name, ephys_window=[round(lo, 1), round(hi, 1)],
                                overlap_with_quality_window=round(overlap, 3), used=use))
        if use:
            keep[name] = path
    out = dict(session_id=args.session_id, recording_id=dsm.recording_id, animal=focal, purpose=args.purpose,
               hypothesis_id=args.hypothesis_id, git_commit=git_commit(), n_quality_cells=len(cluster_ids),
               spike_window=list(win), chunks=feasibility, parameters=vars(args).copy(), smoke=args.smoke,
               units="pixels", pixels_per_cm=None)
    if not keep:
        out["status"] = "no_chunk_overlapping_recording"
        return out
    if args.max_chunks:
        keep = dict(list(keep.items())[: args.max_chunks])

    loaded = {n: load_chunk_with_motion(p, sync, focal) for n, p in keep.items()}
    bounds = arena_bounds([{a: df for a, df in tr.items()} for tr, _ in loaded.values()])
    out["bounds_px"] = list(map(float, bounds))

    results = [analyse_chunk(n, tr, mo, focal, win, cluster_ids, spike_times, bounds=bounds, args=args)
               for n, (tr, mo) in loaded.items()]
    out["results"] = results
    out["replication"] = replication(results)
    out["status"] = "gates_only" if args.gates_only else "ok"
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_id", required=True, help="recording id, e.g. 20251216_145034")
    p.add_argument("--animal_id", required=True, help="focal rat number, e.g. 630")
    p.add_argument("--config_path", default="config/default_paths.json")
    p.add_argument("--dio_channel", type=int, default=1)
    p.add_argument("--purpose", choices=["exploratory", "confirmatory"], default="exploratory")
    p.add_argument("--hypothesis_id", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--gates_only", action="store_true", help="build inputs and report rest budget; run no tests")
    p.add_argument("--smoke", action="store_true", help="tiny plumbing run; results are not findings")
    p.add_argument("--max_cells", type=int, default=None, help="with --smoke")
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--min_chunk_overlap", type=float, default=0.8)
    p.add_argument("--n_shifts", type=int, default=999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--window_sec", type=float, default=60.0, help="high-pass window (calibrated at 60 s)")
    p.add_argument("--min_seg_sec", type=float, default=60.0)
    p.add_argument("--home_frac_diag", type=float, default=0.04, help="rest radius as a fraction of the arena diagonal")
    p.add_argument("--still_px_s", type=float, default=20.0)
    p.add_argument("--bridge_sec", type=float, default=3.0,
                   help="fill speed-gate flickers up to this long inside the home radius (provisional; chosen from "
                        "rest budgets on rat630/20251216_145034, 3 s recovers 16-20 min/chunk from 1-6, 5-10 s add nothing)")
    p.add_argument("--min_tracked_sec", type=float, default=300.0)
    p.add_argument("--min_rest_min", type=float, default=10.0, help="usable rest per chunk (provisional)")
    p.add_argument("--min_rest_spikes", type=int, default=50, help="per-cell spikes in rest bins (provisional)")
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    out = run(args)
    out_dir = args.output_dir or f"results/rest_distance/{args.session_id}_{args.animal_id}_{args.purpose}" \
                                 f"{'_smoke' if args.smoke else ''}{'_gates' if args.gates_only else ''}"
    d = save(out, out_dir)
    print(f"status: {out['status']}  -> {d}")
    for r in out.get("results", []):
        print({k: v for k, v in r.items() if k in ("chunk", "status", "raw_rest_min", "rest_min", "n_segments",
                                                  "n_cells_tested", "n_p05", "n_q05", "has_keypoints")},
              (r.get("population") or {}).get("p_value"))
    print("replication:", out.get("replication"))


if __name__ == "__main__":
    main()
