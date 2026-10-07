"""Territory-encoding runner: one focal rat, one recording, APT tracking chunks.

    python -m ephys.run_territory --session_id 20251216 --animal_id 631 --gates_only
    python -m ephys.run_territory --session_id 20251216 --animal_id 631 --smoke
    python -m ephys.run_territory --session_id 20251216 --animal_id 631 --purpose exploratory

What it does
------------
1. Refuses held-out targets (``LabNotebook.assert_not_held_out``, ``multi_animal=True``
   because partners' tracking is read) before any data is touched.
2. Loads the focal rat's ephys, builds the clock sync, finds every APT 30-minute
   chunk for the date, maps each through the sync and keeps those that lie inside
   the focal's quality-cell spike window.
3. **Gates** (reported; the run stops on a failed gate unless ``--ignore_gates``):
   territory stability between chunks, and a position-coding positive control.
4. For each kept chunk, builds the territory map from the *other* chunks
   (leave-one-chunk-out) and runs the single-cell GLM (step and gradient) and the
   three population tests.
5. Writes ``summary.json``, per-chunk per-cell CSVs and a pickle under
   ``--output_dir``.

Units: tracking is left in **pixels** (``pixels_per_cm=None``): the config's
calibration was made on the manual-tracking video and is wrong for APT (see
CLAUDE.md). All distances, bands and clips are pixels and are recorded in
``parameters``. Defaults marked *provisional* are guesses to be revisited with data,
not validated thresholds.

This runner does **not** write to the lab notebook: families, holdouts and frozen
predictions are the scientist's acts (``scripts/notebook_cli.py``), and the
multiple-comparison denominator must come from a declared family. It does report
the FDR resolution of the budget it used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ephys._stats_utils import benjamini_hochberg, fdr_resolution
from ephys import territory_encoding as te
from ephys import territory_population as tp
from video.territory import (UNCLAIMED, build_territory_map_loo, compute_territory_map, exclusivity_map,
                             label_positions, territory_stability)

logger = logging.getLogger(__name__)

DT = 0.5
MAX_GAP_SEC = 1.0          # interpolate tracking onto bins only across gaps shorter than this
APT_SUBDIR = ("solution", "TQT_named.csv")


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def apt_chunk_dirs(apt_root: str, date: str) -> List[Tuple[str, str]]:
    """``[(chunk_id, TQT_named.csv path)]`` for every APT chunk of ``date``."""
    out = []
    for name in sorted(os.listdir(apt_root)):
        if name.startswith(f"cohort7_{date}_") and name.count("_") == 2:
            p = os.path.join(apt_root, name, *APT_SUBDIR)
            if os.path.exists(p):
                out.append((name, p))
    return out


def chunk_ephys_window(path: str, sync) -> Tuple[float, float]:
    ts = pd.read_csv(path, usecols=["timestamp"], encoding="utf-8-sig")["timestamp"].to_numpy()
    e = sync.convert_behavior_to_ephys(np.array([ts.min(), ts.max()]) / 1e9)
    return float(e[0]), float(e[1])


def load_chunk(path: str, sync) -> Dict[str, pd.DataFrame]:
    """All tracked animals in a chunk on the ephys clock, in pixels: ``{name: DataFrame[t,x,y,speed]}``."""
    from video.tracking_import import load_tracking_data, resolve_tracking_on_ephys_clock
    tr = load_tracking_data(path)
    return resolve_tracking_on_ephys_clock(tr, sync, list(tr.parsed_data.keys()),
                                           pixels_per_cm=None)


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def interp_on_grid(df: pd.DataFrame, t_centers: np.ndarray, max_gap: float = MAX_GAP_SEC) -> Dict[str, np.ndarray]:
    """x, y, speed at bin centres; NaN where the nearest tracked sample is > ``max_gap`` away."""
    t = df["t"].to_numpy()
    out = {k: np.interp(t_centers, t, df[k].to_numpy(), left=np.nan, right=np.nan) for k in ("x", "y", "speed")}
    idx = np.clip(np.searchsorted(t, t_centers), 1, len(t) - 1)
    gap = np.minimum(np.abs(t_centers - t[idx - 1]), np.abs(t[idx] - t_centers))
    for k in out:
        out[k] = np.where(gap <= max_gap, out[k], np.nan)
    return out


def bin_counts(spike_times: List[np.ndarray], t0: float, n_bins: int, dt: float = DT) -> np.ndarray:
    edges = t0 + np.arange(n_bins + 1) * dt
    return np.column_stack([np.histogram(st, bins=edges)[0] for st in spike_times]).astype(float)


def chunk_inputs(tracks: Dict[str, pd.DataFrame], focal: str, window: Tuple[float, float],
                 dt: float = DT) -> Dict[str, object]:
    """Time grid, focal covariates and every other animal's position on that grid."""
    f = tracks[focal]
    lo = max(window[0], float(f["t"].iloc[0]))
    hi = min(window[1], float(f["t"].iloc[-1]))
    n = int((hi - lo) // dt)
    centers = lo + (np.arange(n) + 0.5) * dt
    foc = interp_on_grid(f, centers)
    others = {k: interp_on_grid(v, centers) for k, v in tracks.items() if k != focal and len(v) > 2}
    if others:
        d = np.column_stack([np.hypot(foc["x"] - o["x"], foc["y"] - o["y"]) for o in others.values()])
        partner = np.where(np.isfinite(d).any(axis=1), np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1), np.nan)
        partner = np.where(np.isfinite(partner), partner, np.nan)
    else:
        partner = np.full(n, np.nan)
    return dict(t0=lo, n_bins=n, centers=centers, x=foc["x"], y=foc["y"], speed=foc["speed"],
                partner_dist=partner, others=others)


def owner_presence_inputs(inp: Dict[str, object], tmap, focal: str):
    """Who owns the territory the focal is in, is that owner home, and how far away is it."""
    lab = label_positions(tmap, focal, inp["x"], inp["y"])
    n = len(inp["x"])
    owner = np.array([o if (o is not None and o != focal) else None for o in lab["owner"]], dtype=object)
    present = np.full(n, np.nan)
    dist = np.full(n, np.nan)
    for name, pos in inp["others"].items():
        sel = owner == name
        if not sel.any() or name not in tmap.animal_ids:
            continue
        home = label_positions(tmap, name, pos["x"], pos["y"])
        valid = np.isfinite(pos["x"]) & np.isfinite(pos["y"])
        present[sel] = np.where(valid, home["own"].to_numpy().astype(float), np.nan)[sel]
        dist[sel] = np.hypot(inp["x"] - pos["x"], inp["y"] - pos["y"])[sel]
    return owner, present, dist


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def stability_gate(chunks: Dict[str, Dict[str, pd.DataFrame]], focal: str, *, bounds, bins: int,
                   map_kw: Dict, min_exclusivity_corr: float, min_owner_agreement: float,
                   min_focal_jaccard: float) -> Dict[str, object]:
    """Compare single-chunk territory maps pairwise (each built from independent data).

    The gate that decides whether the run proceeds is the stability of the focal's
    **exclusivity** map (continuous; correlation over bins used by someone in both chunks).
    The binary owner-map agreement is reported alongside and decides only whether the
    boundary-based population tests are run: on 20251216 it did not replicate between chunks.
    """
    maps = {}
    for name, tr in chunks.items():
        try:
            maps[name] = compute_territory_map(tr, bounds=bounds, bins=bins, **map_kw)
        except ValueError as e:
            logger.warning("no territory map for chunk %s: %s", name, e)
    pairs = []
    names = sorted(maps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            s = territory_stability(maps[a], maps[b])
            corr = None
            if focal in maps[a].animal_ids and focal in maps[b].animal_ids:
                ea, eb = exclusivity_map(maps[a], focal), exclusivity_map(maps[b], focal)
                ok = np.isfinite(ea) & np.isfinite(eb)
                if ok.sum() >= 10 and np.std(ea[ok]) > 0 and np.std(eb[ok]) > 0:
                    corr = float(np.corrcoef(ea[ok], eb[ok])[0, 1])
            pairs.append(dict(a=a, b=b, exclusivity_corr=corr, owner_agreement=s["owner_agreement"],
                              adjusted_rand_index=s["adjusted_rand_index"],
                              focal_jaccard=s["jaccard_by_animal"].get(focal),
                              n_jointly_claimed_bins=s["n_jointly_claimed_bins"]))

    def med(key):
        v = [p[key] for p in pairs if p[key] is not None]
        return float(np.median(v)) if v else None

    mc, ma, mj = med("exclusivity_corr"), med("owner_agreement"), med("focal_jaccard")
    return dict(passed=bool(mc is not None and mc >= min_exclusivity_corr), pairs=pairs,
                median_exclusivity_corr=mc, median_owner_agreement=ma, median_focal_jaccard=mj,
                binary_owner_stable=bool(ma is not None and mj is not None and ma >= min_owner_agreement
                                         and mj >= min_focal_jaccard),
                thresholds=dict(min_exclusivity_corr=min_exclusivity_corr, min_owner_agreement=min_owner_agreement,
                                min_focal_jaccard=min_focal_jaccard, note="provisional"),
                focal_in_maps=[n for n, m in maps.items() if focal in m.animal_ids])


def position_control_gate(counts: np.ndarray, cov: Dict[str, np.ndarray], *, min_gain: float,
                          min_fraction: float, max_cells: Optional[int]) -> Dict[str, object]:
    cols = range(counts.shape[1]) if max_cells is None else range(min(max_cells, counts.shape[1]))
    gains = np.array([te.position_coding_gain(counts[:, c], cov, dt=DT)["delta_ll_cv_per_spike"] for c in cols])
    ok = np.isfinite(gains)
    frac = float(np.mean(gains[ok] > min_gain)) if ok.any() else 0.0
    return dict(passed=bool(frac >= min_fraction), fraction_with_gain=frac, n_cells=int(ok.sum()),
                median_gain=float(np.median(gains[ok])) if ok.any() else None,
                thresholds=dict(min_gain_per_spike=min_gain, min_fraction=min_fraction, note="provisional"))


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------

def analyse_chunk(chunk: str, chunks: Dict[str, Dict[str, pd.DataFrame]], focal: str, window, cluster_ids,
                  spike_times, *, bounds, bins, map_kw, args, binary_stable: bool = False) -> Dict[str, object]:
    t_start = time.time()
    tmap = build_territory_map_loo(chunks, chunk, bounds=bounds, bins=bins, **map_kw)
    if focal not in tmap.animal_ids:
        return dict(chunk=chunk, status="focal_has_no_territory", excluded=tmap.excluded)
    inp = chunk_inputs(chunks[chunk], focal, window)
    counts = bin_counts(spike_times, inp["t0"], inp["n_bins"])
    cov = dict(x=inp["x"], y=inp["y"], speed=inp["speed"], partner_dist=inp["partner_dist"])
    lab = label_positions(tmap, focal, inp["x"], inp["y"])
    terr = dict(own=lab["own"].to_numpy(), signed_dist=lab["signed_dist"].to_numpy(),
                exclusivity=lab["exclusivity"].to_numpy())
    res: Dict[str, object] = dict(
        chunk=chunk, status="ok", n_bins=inp["n_bins"], t0=inp["t0"], train_chunks=tmap.parameters["train_chunks"],
        territory_fraction=tmap.territory_fraction(), excluded_animals=tmap.excluded,
        own_fraction_of_bins=float(np.nanmean(lab["own"])), foreign_fraction=float(np.nanmean(lab["foreign"])),
        tracked_fraction=float(np.mean(np.isfinite(inp["x"]) & np.isfinite(inp["partner_dist"]))),
    )

    # ---- single cells -----------------------------------------------------
    n_cells = counts.shape[1]
    cell_idx = list(range(n_cells)) if args.max_cells is None else list(range(min(args.max_cells, n_cells)))
    testable = [c for c in cell_idx if counts[:, c].sum() >= te.MIN_SPIKES]
    res_fdr = fdr_resolution(max(len(testable), 1), args.n_boot, args.alpha)
    n_boot = args.n_boot
    if not res_fdr["resolvable"] and not args.smoke:
        n_boot = min(max(args.n_boot, res_fdr["recommended_n_shuffles"]), args.max_n_boot)
        res_fdr = fdr_resolution(max(len(testable), 1), n_boot, args.alpha)
        logger.warning("chunk %s: raised n_boot %d -> %d for FDR resolution (resolvable=%s)",
                       chunk, args.n_boot, n_boot, res_fdr["resolvable"])
    res["fdr_resolution"] = dict(res_fdr, n_boot_used=n_boot)
    rows = []
    effects = [e for e in args.effects.split(",") if e]
    for effect in effects:
        for c in testable:
            r = te.fit_territory_effect(counts[:, c], cov, terr, effect=effect, dt=DT, n_boot=n_boot,
                                        grad_clip=args.grad_clip_px, seed=int(cluster_ids[c]))
            rows.append(dict(chunk=chunk, effect=effect, cluster_id=cluster_ids[c],
                             **{k: r[k] for k in ("status", "p_value", "lrt_stat", "delta_ll_cv_per_spike",
                                                  "effect_estimate", "n_spikes", "dispersion") if k in r}))
    cells = pd.DataFrame(rows)
    if len(cells):
        for effect, g in cells.groupby("effect"):
            q = benjamini_hochberg(g["p_value"].to_numpy())
            cells.loc[g.index, "q_value"] = q
    res["cells"] = cells
    res["n_cells_tested"] = len(testable)
    res["n_significant_q05"] = {e: int(((cells["effect"] == e) & (cells["q_value"] < 0.05)).sum())
                                for e in effects} if len(cells) else {}

    # ---- population ---------------------------------------------------------
    rates = counts / DT
    pop_cols = [c for c in cell_idx]
    rates = rates[:, pop_cols]
    keep = np.isfinite(inp["x"]) & np.isfinite(inp["y"])
    res["binary_owner_stable"] = bool(binary_stable)
    if not (binary_stable or args.population_anyway):
        res["population_status"] = ("skipped: the binary territory map did not replicate between chunks "
                                    "(see stability_gate); pass --population_anyway to run the boundary-based tests")
    elif keep.sum() > 0.5 * len(keep):
        x = np.where(keep, inp["x"], np.nan)
        y = np.where(keep, inp["y"], np.nan)
        res["straddle"] = _strip(tp.boundary_straddle_test(rates, x, y, tmap, focal, dt=DT,
                                                           band=args.band_px, n_displace_max=args.n_displace))
        spd = np.nan_to_num(inp["speed"], nan=0.0)
        res["decoder_gain"] = _strip(tp.territory_decoder_gain(
            rates[keep], inp["x"][keep], inp["y"][keep], spd[keep], tmap, focal, dt=DT,
            band=args.band_px, n_displace_max=min(args.n_displace, 100)))
        owner, present, dist = owner_presence_inputs(inp, tmap, focal)
        res["owner_presence"] = _strip(tp.owner_presence_decoding(
            rates, np.nan_to_num(inp["x"]), np.nan_to_num(inp["y"]), owner, present, dist, dt=DT,
            n_shifts=args.n_shifts))
    else:
        res["population_status"] = "insufficient_tracking"
    res["seconds"] = round(time.time() - t_start, 1)
    return res


def _strip(d: Dict) -> Dict:
    """Keep scalars/short fields for the JSON summary; the full null is stored in the pickle."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) and v.size <= 5 else v)
            for k, v in d.items() if not (isinstance(v, np.ndarray) and v.size > 5)} | \
           {"_null_len": int(len(d.get("null", [])))}


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
    ks = load_kilosort_data(dsm.get_kilosort_path())
    sync = DataSyncManager(dsm, dio_channel=args.dio_channel)
    cluster_ids, spike_times = ks.get_filtered_cells_spike_times()
    win = ks.quality_ephys_window()
    logger.info("%s/%s: %d quality cells, spike window [%.1f, %.1f]", args.animal_id, dsm.recording_id,
                len(cluster_ids), *win)

    apt_root = Path(args.config_path or "config/default_paths.json")
    import json as _json
    cfg = _json.load(open(apt_root))
    roots = cfg["tracking"] if isinstance(cfg["tracking"], list) else [cfg["tracking"]]
    apt = next((r for r in roots if "APT" in str(r)), None)
    if apt is None:
        raise RuntimeError("no APT tracking root in config")
    date = args.session_id[:8]

    feasibility = []
    keep: Dict[str, str] = {}
    for name, path in apt_chunk_dirs(apt, date):
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
    if len(keep) < 2:
        out["status"] = "needs_at_least_2_chunks_overlapping_recording"
        return out
    if args.max_chunks:
        keep = dict(list(keep.items())[: args.max_chunks])

    chunks = {n: load_chunk(p, sync) for n, p in keep.items()}
    bounds = (min(df["x"].min() for t in chunks.values() for df in t.values()),
              max(df["x"].max() for t in chunks.values() for df in t.values()),
              min(df["y"].min() for t in chunks.values() for df in t.values()),
              max(df["y"].max() for t in chunks.values() for df in t.values()))
    map_kw = dict(smoothing_sigma_bins=1.0, min_dwell_sec=args.min_dwell_sec,
                  min_tracked_sec=args.min_tracked_sec, min_spread=args.min_spread_px)
    out["bounds_px"] = list(map(float, bounds))
    out["presence"] = {n: {a: round(len(df) / max(len(next(iter(t.values()))), 1), 2) for a, df in t.items()}
                       for n, t in chunks.items()}

    # ---- gates --------------------------------------------------------------
    out["stability_gate"] = stability_gate(chunks, focal, bounds=bounds, bins=args.bins, map_kw=map_kw,
                                           min_exclusivity_corr=args.min_exclusivity_corr,
                                           min_owner_agreement=args.min_owner_agreement,
                                           min_focal_jaccard=args.min_focal_jaccard)
    first = next(n for n in keep if focal in chunks[n])
    inp0 = chunk_inputs(chunks[first], focal, (win[0], win[1]))
    counts0 = bin_counts(spike_times, inp0["t0"], inp0["n_bins"])
    out["position_control_gate"] = position_control_gate(
        counts0, dict(x=inp0["x"], y=inp0["y"], speed=inp0["speed"], partner_dist=inp0["partner_dist"]),
        min_gain=args.min_position_gain, min_fraction=args.min_position_fraction,
        max_cells=args.max_cells if args.smoke else None)
    gates_ok = out["stability_gate"]["passed"] and out["position_control_gate"]["passed"]
    out["gates_passed"] = bool(gates_ok)
    if args.gates_only or (not gates_ok and not args.ignore_gates):
        out["status"] = "gates_only" if args.gates_only else "stopped_at_failed_gate"
        return out

    results = []
    for name in keep:
        if focal not in chunks[name]:
            results.append(dict(chunk=name, status="focal_not_tracked"))
            continue
        results.append(analyse_chunk(name, chunks, focal, win, cluster_ids, spike_times, bounds=bounds,
                                     bins=args.bins, map_kw=map_kw, args=args,
                                     binary_stable=out["stability_gate"]["binary_owner_stable"]))
    out["status"] = "ok"
    out["results"] = results
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save(out: Dict[str, object], output_dir: str) -> Path:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps({k: v for k, v in out.items() if k != "results"}, default=_json_default))
    slim["results"] = []
    for r in out.get("results", []):
        r2 = {k: v for k, v in r.items() if k != "cells"}
        slim["results"].append(json.loads(json.dumps(r2, default=_json_default)))
        if "cells" in r:
            r["cells"].to_csv(d / f"cells_{r['chunk']}.csv", index=False)
    (d / "summary.json").write_text(json.dumps(slim, indent=2))
    with open(d / "full_results.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return d


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session_id", required=True, help="date or full recording id, e.g. 20251216 or 20251216_144334")
    p.add_argument("--animal_id", required=True, help="focal rat number, e.g. 631")
    p.add_argument("--config_path", default="config/default_paths.json")
    p.add_argument("--dio_channel", type=int, default=1)
    p.add_argument("--purpose", choices=["exploratory", "confirmatory"], default="exploratory")
    p.add_argument("--hypothesis_id", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--gates_only", action="store_true", help="feasibility + gates, no analysis")
    p.add_argument("--ignore_gates", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny plumbing run (few cells, few draws); results are not findings")
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--min_chunk_overlap", type=float, default=0.8,
                   help="keep a chunk if this fraction lies inside the focal's quality-cell spike window; "
                        "the analysis window is truncated to the recording")
    p.add_argument("--n_boot", type=int, default=999)
    p.add_argument("--max_n_boot", type=int, default=6000)
    p.add_argument("--n_displace", type=int, default=300)
    p.add_argument("--n_shifts", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--bins", type=int, default=20,
                   help="territory grid per axis; 20 not 40: exclusivity replicated between chunks at "
                        "r=0.57 on a 20x20 grid vs 0.42 on 40x40 (rat631, 20251216)")
    p.add_argument("--effects", default="exclusivity",
                   help="comma list of GLM effects: exclusivity (default), step, gradient; the binary ones "
                        "rest on an owner map that did not replicate on 20251216, and each extra effect "
                        "adds to the multiple-comparison family")
    p.add_argument("--population_anyway", action="store_true",
                   help="run the boundary-based population tests even if the binary map is unstable")
    p.add_argument("--band_px", type=float, default=300.0, help="band around the boundary (pixels; provisional)")
    p.add_argument("--grad_clip_px", type=float, default=600.0, help="gradient covariate clip (pixels; provisional)")
    p.add_argument("--min_dwell_sec", type=float, default=2.0)
    p.add_argument("--min_tracked_sec", type=float, default=300.0)
    p.add_argument("--min_spread_px", type=float, default=0.0,
                   help="drop near-stationary animals from the territory map; 0 = keep (a stationary rat is a real "
                        "territory; the near-stationary hazard HZ-STAT-007 concerns decoding targets, not mapping)")
    p.add_argument("--min_exclusivity_corr", type=float, default=0.5,
                   help="stability gate on the focal's exclusivity map (provisional)")
    p.add_argument("--min_owner_agreement", type=float, default=0.5, help="binary-map stability (provisional)")
    p.add_argument("--min_focal_jaccard", type=float, default=0.3, help="stability gate (provisional)")
    p.add_argument("--min_position_gain", type=float, default=0.005, help="position-control gate, ll/spike (provisional)")
    p.add_argument("--min_position_fraction", type=float, default=0.2, help="position-control gate (provisional)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.max_cells = args.max_cells or 12
        args.n_boot = min(args.n_boot, 99)
        args.n_displace = min(args.n_displace, 30)
        args.n_shifts = min(args.n_shifts, 19)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run(args)
    tag = "_smoke" if args.smoke else ("_gates" if args.gates_only else "")
    d = save(out, args.output_dir or f"results/territory/{args.session_id}_{args.animal_id}{tag}")
    print(json.dumps({k: out[k] for k in ("status", "gates_passed", "stability_gate", "position_control_gate")
                      if k in out}, indent=2, default=_json_default))
    print(f"saved to {d}")


if __name__ == "__main__":
    main()
