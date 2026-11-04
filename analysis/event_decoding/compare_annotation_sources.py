#!/usr/bin/env python3
"""
Compare opponent-identity decoding on two annotation sources for the same recording
(manual scoring vs. FERAL automated annotations).

Three subcommands:

``agreement``
    Event-level agreement between the two files for one date: events matched on
    the unordered animal pair plus time IoU. Reports precision / recall / F1 of
    FERAL against manual, and how often a matched pair agrees on who initiated.

``controls``
    Writes the event files for the two controls that separate "different
    annotator" from "different sample":

    * ``intersection`` -- FERAL rows that match a manual row (same events,
      FERAL's timing and labels).
    * ``manual_matchedN`` -- manual EC rows for the focal animal subsampled to
      FERAL's in-window EC count (several seeds). FERAL has *fewer* events than
      the manual file in the 20251216 morning block, so it is the manual set that
      is cut down, not FERAL.

    Decode each file with ``plot_decoding_accuracy_violins.py --events_file``
    (separate ``--cache_dir``) and ``refine_permutation_fdr.py``.

``compare``
    Paired per-cell comparison from two ``refined_fdr_*.csv`` tables.

Event windows come from the capability manifest's stored per-recording sync and
each animal's ephys window, with no DIO reads, the same placement as
``gui/session_index.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion._conventions import normalize_object_name  # noqa: E402
from analysis._paths import results_dir  # noqa: E402

_ID_COLS = ("initiator", "victim", "winner", "loser")


def load_events(path, date: Optional[str] = None) -> pd.DataFrame:
    """Read an event CSV with ids normalised (``rat4613`` -> ``rat613``), optionally one date."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in _ID_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: normalize_object_name(v) if pd.notna(v) else v)
    if date and "session" in df.columns:
        df = df[df["session"].astype(str).str.contains(f"_{date}_", regex=False)]
    return df.reset_index(drop=True)


def _intervals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return df["ts_start"].to_numpy(float) / 1e9, df["ts_end"].to_numpy(float) / 1e9


def match_events(manual: pd.DataFrame, feral: pd.DataFrame, *, min_iou: float = 0.3,
                 event_type: str = "EC") -> pd.DataFrame:
    """Greedy one-to-one match, highest IoU first, on the unordered animal pair.

    Returns one row per matched pair: ``manual_idx``, ``feral_idx``, ``iou``,
    ``same_initiator``.
    """
    m = manual[manual["type"] == event_type]
    f = feral[feral["type"] == event_type]
    ms, me = _intervals(m)
    fs, fe = _intervals(f)
    mpair = [frozenset((a, b)) for a, b in zip(m["initiator"], m["victim"])]
    fpair = [frozenset((a, b)) for a, b in zip(f["initiator"], f["victim"])]
    cand = []
    for j in range(len(f)):
        lo = np.maximum(ms, fs[j])
        hi = np.minimum(me, fe[j])
        inter = np.clip(hi - lo, 0, None)
        union = (me - ms) + (fe[j] - fs[j]) - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        for i in np.nonzero(iou >= min_iou)[0]:
            if mpair[i] == fpair[j]:
                cand.append((iou[i], i, j))
    cand.sort(reverse=True)
    used_m, used_f, rows = set(), set(), []
    for iou, i, j in cand:
        if i in used_m or j in used_f:
            continue
        used_m.add(i)
        used_f.add(j)
        rows.append({"manual_idx": m.index[i], "feral_idx": f.index[j], "iou": float(iou),
                     "same_initiator": m["initiator"].iloc[i] == f["initiator"].iloc[j]})
    return pd.DataFrame(rows, columns=["manual_idx", "feral_idx", "iou", "same_initiator"])


def agreement(manual: pd.DataFrame, feral: pd.DataFrame, *, min_iou: float = 0.3,
              event_type: str = "EC") -> Dict:
    pairs = match_events(manual, feral, min_iou=min_iou, event_type=event_type)
    n_m = int((manual["type"] == event_type).sum())
    n_f = int((feral["type"] == event_type).sum())
    tp = len(pairs)
    precision = tp / n_f if n_f else float("nan")
    recall = tp / n_m if n_m else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if tp else 0.0
    return {"n_manual": n_m, "n_feral": n_f, "n_matched": tp, "precision": precision,
            "recall": recall, "f1": f1, "min_iou": min_iou,
            "initiator_agreement": float(pairs["same_initiator"].mean()) if tp else float("nan"),
            "median_iou": float(pairs["iou"].median()) if tp else float("nan")}


def in_window_mask(df: pd.DataFrame, animal: str, recording: str,
                   manifest: Optional[dict] = None) -> pd.Series:
    """Rows inside ``animal``'s ephys window for ``recording``, via the manifest sync."""
    manifest = manifest or json.load(open(REPO_ROOT / "discovery" / "capability_manifest.json"))
    info = manifest["sessions"][recording]["ephys"]["per_animal"][animal]
    sync, window = info["sync"], info["ephys_window"]
    hi = min(window[1], window[0] + (info.get("duration_seconds") or np.inf))
    t = (df["ts_start"].astype(float) / 1e9 - sync["intercept"]) / sync["slope"]
    return (t >= window[0]) & (t <= hi)


def _involving(df: pd.DataFrame, animal: str) -> pd.Series:
    return (df["initiator"] == animal) | (df["victim"] == animal)


def write_controls(manual_csv, feral_csv, date: str, recording: str, animals,
                   out_dir: Path, *, seeds=(0, 1, 2), min_iou: float = 0.3) -> Dict:
    """Write the control event files; return a manifest of what was written."""
    manual = load_events(manual_csv, date)
    feral = load_events(feral_csv, date)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(REPO_ROOT / "discovery" / "capability_manifest.json"))
    written: Dict = {}

    pairs = match_events(manual, feral, min_iou=min_iou)
    inter = feral.loc[pairs["feral_idx"]]
    p = out_dir / f"feral_intersection_{date}.csv"
    inter.to_csv(p, index=False)
    written["intersection"] = {"file": str(p), "n_rows": len(inter), "min_iou": min_iou}

    for animal in animals:
        f_ec = feral[(feral["type"] == "EC") & _involving(feral, animal)]
        n_feral = int(in_window_mask(f_ec, animal, recording, manifest).sum())
        m_ec = manual[(manual["type"] == "EC") & _involving(manual, animal)]
        m_in = m_ec[in_window_mask(m_ec, animal, recording, manifest)]
        written[animal] = {"n_feral_in_window": n_feral, "n_manual_in_window": len(m_in)}
        for seed in seeds:
            keep = m_in.sample(n=min(n_feral, len(m_in)), random_state=seed).index
            drop = set(m_ec.index) - set(keep)
            sub = manual.drop(index=list(drop))
            p = out_dir / f"manual_matchedN_{animal}_{date}_s{seed}.csv"
            sub.to_csv(p, index=False)
            written[animal][f"seed{seed}"] = str(p)
    return written


def compare_tables(feral_csv, manual_csv, animal: str, session_id) -> Dict:
    """Paired per-cell comparison of two refined_fdr tables for one animal."""
    from scipy.stats import spearmanr, wilcoxon
    f = pd.read_csv(feral_csv)
    m = pd.read_csv(manual_csv)
    f = f[f["animal_id"] == animal]
    m = m[(m["animal_id"] == animal) & (m["session_id"].astype(str) == str(session_id))]
    j = f.merge(m, on="cluster_id", suffixes=("_feral", "_manual"))
    sig_f = set(j.loc[j["significant_feral"], "cluster_id"])
    sig_m = set(j.loc[j["significant_manual"], "cluster_id"])
    diff = j["accuracy_feral"] - j["accuracy_manual"]
    out = {"animal": animal, "n_cells": len(j), "n_sig_feral": len(sig_f),
           "n_sig_manual": len(sig_m), "n_sig_both": len(sig_f & sig_m),
           "jaccard_sig": len(sig_f & sig_m) / len(sig_f | sig_m) if (sig_f | sig_m) else float("nan"),
           "spearman_accuracy": float(spearmanr(j["accuracy_feral"], j["accuracy_manual"])[0]),
           "mean_accuracy_diff": float(diff.mean())}
    out["wilcoxon_p"] = float(wilcoxon(diff)[1]) if (diff != 0).any() else float("nan")
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agreement")
    a.add_argument("--manual", required=True)
    a.add_argument("--feral", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--min_iou", type=float, default=0.3)
    c = sub.add_parser("controls")
    c.add_argument("--manual", required=True)
    c.add_argument("--feral", required=True)
    c.add_argument("--date", required=True)
    c.add_argument("--recording", required=True)
    c.add_argument("--animals", nargs="+", required=True)
    c.add_argument("--out_dir", default=None)
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    k = sub.add_parser("compare")
    k.add_argument("--feral_csv", required=True)
    k.add_argument("--manual_csv", required=True)
    k.add_argument("--animal", required=True)
    k.add_argument("--session_id", required=True)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "agreement":
        res = agreement(load_events(args.manual, args.date), load_events(args.feral, args.date),
                        min_iou=args.min_iou)
    elif args.cmd == "controls":
        out_dir = Path(args.out_dir) if args.out_dir else \
            results_dir("event_decoding") / "feral" / "controls"
        res = write_controls(args.manual, args.feral, args.date, args.recording,
                             args.animals, out_dir, seeds=tuple(args.seeds))
    else:
        res = compare_tables(args.feral_csv, args.manual_csv, args.animal, args.session_id)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
