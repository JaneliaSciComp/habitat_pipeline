#!/usr/bin/env python3
"""Render every SfN-poster panel at its exact poster size, and write facts.json.

Each panel is drawn at the (width, height) in inches that ``poster_spec.layout()``
gives its slot, so fonts in the panels print at their nominal size.

Numbers printed anywhere on the poster come from ``facts.json``. Each fact is
read from its source (manifest, result CSVs, decode caches, a live sync fit) and
asserted against the value the poster text was written for, so the text cannot
drift from the data silently.

Inputs that need the //nearline share (live sync, one cell's spikes, event-file
row counts) are cached under ``results/poster/data/`` on first run; pass
``--refresh`` to re-read them.

    python scripts/poster/make_panels.py            # all panels
    python scripts/poster/make_panels.py --only sync decode_by_rat
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import poster_spec as spec  # noqa: E402
from poster_style import (ACCENT, CAPTION_PT, COHORT, DPI, FIG_LABEL_PT, FIG_TICK_PT,  # noqa: E402
                          FIG_TITLE_PT, INK, MUTED, NAVY, NULL, OPP, PANEL_BG, RULE, SIG,
                          SMALL_PT, STREAM, apply_mpl_style)

apply_mpl_style()
SIZES = spec.figure_sizes()
spec.PANELS.mkdir(parents=True, exist_ok=True)
spec.DATA.mkdir(parents=True, exist_ok=True)

TARGETS = [("rat650", "20250819", "cohort5"), ("rat631", "20251216", "cohort7"),
           ("rat630", "20251216", "cohort7"), ("rat4678", "20260929", "cohort9"),
           ("rat4679", "20260929", "cohort9"), ("rat5675", "20260929", "cohort9")]
EXAMPLE_RAT = "rat631"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fig(name):
    w, h = SIZES[name]
    return plt.figure(figsize=(w, h)), w, h


def _save(fig, name):
    base = spec.PANELS / name
    fig.savefig(base.with_suffix(".png"), dpi=DPI)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {base.with_suffix('.png').relative_to(REPO)}")


def _cached(name, fn, refresh=False):
    f = spec.DATA / f"{name}.pkl"
    if f.exists() and not refresh:
        with open(f, "rb") as fh:
            return pickle.load(fh)
    obj = fn()
    with open(f, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def _expect(label, got, want, tol=0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    if not ok:
        raise AssertionError(f"poster fact '{label}': source gives {got!r}, poster text "
                             f"was written for {want!r}. Update the text, then the expectation.")


def _box(ax, x, y, w, h, text, fc, ec=None, tc="white", size=FIG_TICK_PT, bold=True,
         rad=0.12, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc=fc, ec=ec or fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=size,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=3.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=34,
                                 color=color, lw=lw, zorder=1, shrinkA=0, shrinkB=0))


def _inch_axes(fig, w, h):
    """An axes filling the figure whose data units are inches."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


# --------------------------------------------------------------------------
# data / facts
# --------------------------------------------------------------------------
def load_decode_summary():
    from analysis.event_decoding import plot_decoding_accuracy_violins as violins
    from analysis.event_decoding.refine_permutation_fdr import decode_params
    params = decode_params("opponent")
    out = {}
    for animal, session, _ in TARGETS:
        f = REPO / ".gui_cache" / "accuracy_violins" / violins._cache_key(animal, session, params)
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        bs = r["behavioral_summary"]
        out[animal] = {"n_events": int(bs["n_events"]),
                       "class_counts": {str(k): int(v) for k, v in bs["class_counts"].items()},
                       "baseline": float(r["population_baseline_accuracy"]),
                       "event_times": np.asarray(r["event_times"], float),
                       "labels": np.asarray(r["labels"]).astype(str),
                       "n_cells": int(r["n_successful_cells"])}
    return out


def load_sync():
    from ingestion.data_paths import DataStorageManager
    from ingestion.ephys_sync import DataSyncManager
    dsm = DataStorageManager("631", "20251216", auto_load=True)
    m = DataSyncManager(dsm, dio_channel=1).mapping
    return {k: (np.asarray(v, float) if k.startswith("matched") else v) for k, v in m.items()}


def load_example_spikes(cluster_id):
    from ingestion.data_paths import DataStorageManager
    from ingestion.kilosort_data_import import load_kilosort_data
    ks = load_kilosort_data(DataStorageManager("631", "20251216", auto_load=True).get_kilosort_path())
    i = list(ks.ks_ids).index(cluster_id)
    return np.asarray(ks.spike_times_by_cell[i], float)


def load_event_rows():
    from ingestion.data_paths import DataStorageManager
    rows = {}
    for cohort, animal, session, cfg in (("cohort7", "631", "20251216", None),
                                          ("cohort9", "4679", "20260929", "cohort9_paths.json"),
                                          ("cohort5", "650", "20250819", "cohort5_paths.json")):
        files = DataStorageManager(animal, session, config_path=cfg).get_behavioral_event_files()
        rows[cohort] = {"file": str(files[0]), "rows": int(len(pd.read_csv(files[0])))}
    return rows


def manifest_facts():
    m = json.loads((REPO / "discovery" / "capability_manifest.json").read_text())
    recs = [p for s in m["sessions"].values() for p in s["ephys"]["per_animal"].values()]
    loadable = [p for p in recs if p.get("load_error") is None and p.get("files_verified")]
    animals = {a for s in m["sessions"].values() for a in s["ephys"]["animals"]}
    tracked = set()
    for s in m["sessions"].values():
        tracked |= set((s.get("tracking") or {}).get("identity_resolved_animals") or [])
        tracked |= set(((s.get("tracking") or {}).get("objects") or {}).keys())
    return {"generated_at": m["generated_at"][:10],
            "n_sessions": len(m["sessions"]),
            "n_animal_recordings": len(recs),
            "n_spike_files": sum(bool(p.get("files_verified")) for p in recs),
            "n_loadable": len(loadable),
            "n_quality_c57": int(sum(p["n_quality_cells"] or 0 for p in loadable)),
            "hours_c57": float(sum(p["duration_seconds"] or 0 for p in loadable) / 3600),
            "animals_c57": sorted(animals),
            "c7_days": len({v["session_date"] for v in m["sessions"].values()
                            if v.get("cohort") == "cohort7"}),
            "c7_max_sim": max(len(v["ephys"]["animals"]) for v in m["sessions"].values()
                              if v.get("cohort") == "cohort7"),
            "tracked_manifest": sorted(tracked)}


def load_apt_tracked():
    """Rats with a named APT track on 2025-12-16 (the manifest predates the APT exports)."""
    from ingestion.data_paths import DataStorageManager
    from video.tracking_import import normalize_object_name
    names = set()
    for f in DataStorageManager("631", "20251216").get_tracking_files():
        if "TQT_named" in str(f):
            cols = pd.read_csv(f, nrows=0, encoding="utf-8-sig").columns
            names |= {normalize_object_name(c[:-len("_center_x")]) for c in cols
                      if c.endswith("_center_x")}
    return sorted(names)


def count_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
                         cwd=REPO, capture_output=True, text=True).stdout
    line = [ln for ln in out.splitlines() if "tests collected" in ln or " collected" in ln][-1]
    return int(line.split()[0])


def ai_loop_facts():
    """Numbers for the AI-in-the-loop section, read from the lab notebook (read-only)."""
    from analysis._common._stats_utils import benjamini_hochberg, fdr_resolution
    from database.lab_notebook import Hypothesis, Iteration, LabNotebook, TestFamily
    hz = json.loads((REPO / "discovery" / "hazards.json").read_text(encoding="utf-8"))
    hz = hz["hazards"] if isinstance(hz, dict) else hz
    nb = LabNotebook()
    with nb.get_db_session() as s:
        hyp_ids = [h.id for h in s.query(Hypothesis).all()]
        n_fam = s.query(TestFamily).count()
        n_it = s.query(Iteration).count()
    tiers = [nb.evidence_tier(h).tier for h in hyp_ids]
    res = fdr_resolution(149, 200)
    # Family 4: decode_location sweep on 20251210 (tracking-based; no event file involved).
    fam = nb.family_fdr(4, n_shuffles=180)
    scored = fam["per_test"]
    keys = sorted(scored)
    q_logged = dict(zip(keys, benjamini_hochberg(np.array([scored[k]["p_value"] for k in keys]))))
    f = {"n_hazards": len(hz), "n_hypotheses": len(hyp_ids), "n_families": n_fam,
         "n_iterations": n_it, "q_floor_200_149": res["best_achievable_q"],
         "fam4_m_logged": len(keys), "fam4_n_declared": fam["denominator"]["n_declared"], "fam4_m_declared": fam["n_tests_for_correction"],
         "fam4_q_logged": float(q_logged["object=613"]),
         "fam4_q_declared": scored["object=613"]["q_value"]}
    _expect("hazard entries", f["n_hazards"], 34)
    _expect("hypotheses", f["n_hypotheses"], 5)
    _expect("all hypotheses exploratory", set(tiers), {"exploratory"})
    _expect("best q at 200 shuffles x 149 cells", f["q_floor_200_149"], 0.74, tol=0.005)
    _expect("family 4 m as logged", f["fam4_m_logged"], 7)
    _expect("family 4 declared m", f["fam4_m_declared"], 11)
    _expect("family 4 declared tests", f["fam4_n_declared"], 12)
    _expect("family 4 rat613 q as logged", f["fam4_q_logged"], 0.0387, tol=5e-4)
    _expect("family 4 rat613 q at declared m", f["fam4_q_declared"], 0.0608, tol=5e-4)
    _expect("family 4 resolvable", fam["fdr_resolution"]["resolvable"], False)
    return f


def build_facts(refresh=False):
    print("collecting facts ...")
    mf = manifest_facts()
    _expect("animal-recordings", mf["n_animal_recordings"], 96)
    _expect("recordings with spike files", mf["n_spike_files"], 94)
    _expect("loadable recordings", mf["n_loadable"], 63)
    _expect("quality cells cohorts 5+7", mf["n_quality_c57"], 12227)
    _expect("cohort 7 recording days", mf["c7_days"], 14)
    _expect("cohort 7 rats recorded at once", mf["c7_max_sim"], 4)

    fdr = pd.read_csv(REPO / "results" / "event_decoding" / "refined_fdr_opponent.csv")
    dec = _cached("decode_summary", load_decode_summary, refresh)
    per_rat = {}
    for animal, session, cohort in TARGETS:
        d = fdr[(fdr.animal_id == animal) & (fdr.session_id.astype(str) == session)]
        per_rat[animal] = {"session": session, "cohort": cohort, "n_cells": int(len(d)),
                           "n_sig": int(d.significant.sum()),
                           "max_shuffles": int(d.n_shuffles.max()),
                           **{k: v for k, v in dec[animal].items()
                              if k not in ("event_times", "labels")}}
        _expect(f"{animal} cells (CSV vs cache)", per_rat[animal]["n_cells"], dec[animal]["n_cells"])
    for animal, want in (("rat631", 21), ("rat630", 5), ("rat4679", 36), ("rat650", 0),
                         ("rat4678", 0), ("rat5675", 0)):
        _expect(f"{animal} FDR-significant cells", per_rat[animal]["n_sig"], want)
    c9_cells = sum(per_rat[a]["n_cells"] for a in ("rat4678", "rat4679", "rat5675"))

    ev = _cached("event_rows", load_event_rows, refresh)
    _expect("cohort 7 event rows (20251216)", ev["cohort7"]["rows"], 1141)
    _expect("cohort 9 event rows (20260929)", ev["cohort9"]["rows"], 409)

    sync = _cached("sync_rat631_20251216", load_sync, refresh)
    e, b = sync["matched_ephys"], sync["matched_behavior"]
    resid_ms = (b - (sync["slope"] * e + sync["intercept"])) * 1e3
    _expect("sync matched pulses", int(sync["n_matches"]), 158)

    bc = pd.read_csv(REPO / "results" / "opponent_barcode" / "barcode_opponent.csv")
    single = bc[(bc.metric == "d") & (bc.threshold == 0.5) & (bc.item == "single_opponent_cells")]
    bcd = {}
    for animal, _, cohort in TARGETS:
        r = single[single.animal_id == animal].set_index("null")
        bcd[animal] = {"cohort": cohort, "obs": float(r.loc["free", "observed"]),
                       "null_mean": float(r.loc["free", "null_mean"]),
                       "null_lo": float(r.loc["free", "null_lo"]),
                       "null_hi": float(r.loc["free", "null_hi"]),
                       "p_free": float(r.loc["free", "p"]), "p_block": float(r.loc["block", "p"]),
                       "n_cells": int(r.loc["free", "n_cells"])}
    rep = _cached("barcode_rat631_replication", barcode_replication, refresh)
    _expect("rat631 single-opponent fraction (CSV vs rerun)", rep["single_frac"],
            bcd["rat631"]["obs"], tol=1e-9)

    apt = _cached("apt_tracked_20251216", load_apt_tracked, refresh)
    _expect("APT-tracked rats 20251216", len(apt), 10)
    n_tests = count_tests()
    ai = {**ai_loop_facts(), **connectivity_facts()}
    facts = {
        **ai,
        "manifest_date": mf["generated_at"],
        "n_animal_recordings": mf["n_animal_recordings"], "n_spike_files": mf["n_spike_files"],
        "n_loadable": mf["n_loadable"], "n_missing_ts": mf["n_spike_files"] - mf["n_loadable"],
        "n_quality_c57": mf["n_quality_c57"], "n_quality_c9": c9_cells,
        "n_quality_total": mf["n_quality_c57"] + c9_cells,
        "hours_c57": mf["hours_c57"],
        "n_implanted": len(set(mf["animals_c57"]) | {"rat4678", "rat4679", "rat5675"}),
        "n_tracked": len(apt),
        "n_events_c7": ev["cohort7"]["rows"], "n_events_c9": ev["cohort9"]["rows"],
        "n_events_c5": ev["cohort5"]["rows"],
        "n_events_total": sum(v["rows"] for v in ev.values()),
        "n_tests": n_tests,
        "sync_n": int(sync["n_matches"]), "sync_ppm": (sync["slope"] - 1) * 1e6,
        "sync_resid_us": float(np.std(resid_ms) * 1e3), "sync_r2": float(sync["r_squared"]),
        "max_shuffles": max(v["max_shuffles"] for v in per_rat.values()),
        "n_q_631": per_rat["rat631"]["n_cells"],
        "n_frames_631": 74985,
        "c7_days": mf["c7_days"], "c7_max_sim": mf["c7_max_sim"],
        "n_sig_all": sum(v["n_sig"] for v in per_rat.values()),
        "n_cells_all": sum(v["n_cells"] for v in per_rat.values()),
        "per_rat": per_rat, "barcode": bcd,
        "bc631_obs": 100 * bcd["rat631"]["obs"], "bc631_null": 100 * bcd["rat631"]["null_mean"],
        "bc631_p_free": bcd["rat631"]["p_free"], "bc631_p_block": bcd["rat631"]["p_block"],
        "bc631_rep": 100 * rep["rep"], "bc631_rep_chance": 100 / rep["K"],
        "bc631_rep_p": rep["rep_p"],
        "event_files": ev,
    }
    _expect("pooled cells", facts["n_cells_all"], 2101)
    _expect("pooled FDR-significant cells", facts["n_sig_all"], 62)
    sig = [f"{a} {v['n_sig']}/{v['n_cells']}" for a, v in per_rat.items() if v["n_sig"]]
    none = [a for a, v in per_rat.items() if not v["n_sig"]]
    facts["decode_sig_sentence"] = (
        "FDR-significant cells (q < 0.05): " + "; ".join(sig) + ". None in "
        + ", ".join(none[:-1]) + " or " + none[-1] + ".")
    (spec.DATA / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print("  facts.json written")
    return facts, dec, sync, rep


def barcode_replication():
    from analysis.opponent_barcode import opponent_barcode as ob
    data = ob.load_events_and_responses("rat631", "20251216", "opponent", "evoked",
                                        REPO / ".gui_cache" / "opponent_barcode")
    a = ob.analyze(data, 1000, 15.0, 0)
    _, hist, _ = a["obs"]["d"][ob.HEADLINE]
    rep_null = np.array(a["null"]["free"]["rep"], float)
    return {"rep": float(a["rep_obs"][0]), "rep_n": int(a["rep_obs"][1]),
            "rep_p": float(ob._p(rep_null, a["rep_obs"][0])), "K": int(a["K"]),
            "single_frac": float(hist[1] / a["n_cells"]),
            "names": list(map(str, a["names"])), "counts": a["counts"].tolist(),
            "d_odd": a["d_odd"], "d_even": a["d_even"]}


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def panel_qr(facts):
    import qrcode
    fig, w, h = _fig("qr")
    ax = _inch_axes(fig, w, h)
    ax.add_patch(FancyBboxPatch((0, 0), w, h, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="white", ec="white"))
    q = qrcode.QRCode(border=1, box_size=10)
    q.add_data("https://github.com/JaneliaSciComp/habitat_pipeline")
    img = np.asarray(q.make_image(fill_color=NAVY, back_color="white").convert("RGB"))
    s = h - 1.25
    ax.imshow(img, extent=((w - s) / 2, (w + s) / 2, 1.05, 1.05 + s), interpolation="nearest", zorder=3)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.text(w / 2, 0.55, "Code on GitHub", ha="center", va="center", fontsize=CAPTION_PT,
            color=NAVY, fontweight="bold")
    _save(fig, "qr")


def panel_clocks(facts):
    fig, w, h = _fig("clocks")
    ax = _inch_axes(fig, w, 8.35)       # drawn on an 8.35 in design grid, stretched to fit
    rng = np.random.default_rng(3)
    x0, x1 = 3.9, w - 0.25
    lanes = [("Electrophysiology", "30 kHz sample counter\n(Kilosort 4, per animal)", STREAM["ephys"], 6.55),
             ("Video tracking", "PC wall clock, ~10 Hz\n(manual or APT)", STREAM["video"], 4.45),
             ("Scored events", "Linux ns timestamps\n(human scoring)", STREAM["events"], 2.35)]
    for name, sub, col, y in lanes:
        ax.text(0.05, y + 0.32, name, fontsize=FIG_LABEL_PT, fontweight="bold", color=col, va="center")
        ax.text(0.05, y - 0.45, sub, fontsize=SMALL_PT, color=MUTED, va="center", linespacing=1.1)
        ax.plot([x0, x1], [y - 0.65, y - 0.65], color=RULE, lw=2)
    # ephys: four animals' spike ticks
    for k in range(4):
        t = np.sort(rng.uniform(x0, x1, 70))
        yy = 6.55 + 0.75 - k * 0.38
        ax.vlines(t, yy - 0.14, yy + 0.14, color=STREAM["ephys"], lw=1.6, alpha=0.85)
    # video: frames + a moving dot trace
    tf = np.arange(x0, x1, 0.32)
    ax.vlines(tf, 4.45 - 0.35, 4.45 + 0.85, color=STREAM["video"], lw=1.0, alpha=0.35)
    ax.plot(tf, 4.45 + 0.25 + 0.45 * np.sin(tf * 0.9) * np.cos(tf * 0.31), "o-",
            color=STREAM["video"], ms=6, lw=1.8)
    # events: intervals
    for (a, b, lab) in ((4.6, 5.6, "EC"), (6.4, 6.9, "C"), (7.9, 9.4, "F"), (10.2, 10.9, "EC"),
                        (11.9, 13.0, "PS"), (13.6, 14.3, "EC")):
        ax.add_patch(Rectangle((a, 2.35 - 0.25), b - a, 0.75, fc=STREAM["events"], ec="none", alpha=0.85))
        ax.text((a + b) / 2, 2.35 + 0.12, lab, ha="center", va="center", color="white",
                fontsize=SMALL_PT, fontweight="bold")
    # sync pulses: same irregular train on two clocks with a small offset/drift
    pulses = x0 + 0.3 + np.cumsum(rng.uniform(0.35, 1.6, 12))
    pulses = pulses[pulses < x1 - 0.2]
    ye, yb = 0.95, 0.15
    ax.text(0.05, 0.75, "Sync pulses", fontsize=FIG_LABEL_PT, fontweight="bold", color=STREAM["sync"], va="center")
    ax.text(0.05, 0.12, "DIO ch. 1 ↔ pulse log", fontsize=SMALL_PT, color=MUTED, va="center")
    for p in pulses:
        q = p + 0.18 + 0.012 * (p - x0)
        ax.vlines(p, ye, ye + 0.4, color=STREAM["sync"], lw=3.2)
        ax.vlines(q, yb, yb + 0.4, color=STREAM["sync"], lw=3.2)
        ax.plot([p, q], [ye, yb + 0.4], color=STREAM["sync"], lw=1.0, ls=(0, (3, 3)), alpha=0.7)
    ax.text(x1, 7.75, "t_behavior = a · t_ephys + b", ha="right", va="center",
            fontsize=FIG_LABEL_PT, color=INK, fontstyle="italic",
            bbox=dict(fc=PANEL_BG, ec=RULE, boxstyle="round,pad=0.3"))
    _save(fig, "clocks")


def panel_tiles(facts):
    fig, w, h = _fig("tiles")
    ax = _inch_axes(fig, w, h)
    tiles = [(f"{3}", "cohorts"),
             (f"{facts['n_implanted']}", "implanted rats"),
             (f"{facts['n_animal_recordings']}", "animal-recordings\n(cohorts 5, 7)"),
             (f"{facts['n_quality_total']:,}", "quality single units"),
             (f"{facts['hours_c57']:.0f} h", "loadable ephys\n(cohorts 5, 7)"),
             (f"{facts['n_tracked']}", "rats tracked at once\n(APT, 2025-12-16)"),
             (f"{facts['n_events_total']:,}", "scored social events\n(3 annotated days)"),
             (f"{facts['n_tests']:,}", "automated tests")]
    cols, rows, gap = 4, 2, 0.22
    tw, th = (w - gap * (cols - 1)) / cols, (h - 0.5 - gap * (rows - 1)) / rows
    for i, (num, lab) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = c * (tw + gap), h - (r + 1) * th - r * gap
        ax.add_patch(FancyBboxPatch((x, y), tw, th, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=PANEL_BG, ec="none"))
        ax.text(x + tw / 2, y + th * 0.66, num, ha="center", va="center", fontsize=56,
                fontweight="bold", color=NAVY)
        ax.text(x + tw / 2, y + th * 0.25, lab, ha="center", va="center", fontsize=SMALL_PT,
                color=MUTED, linespacing=1.05)
    ax.text(0, 0.18, f"Cohorts 5 and 7 from the capability manifest ({facts['manifest_date']}); "
            "cohort 9 units from its decoding run.", fontsize=18, color=MUTED, va="center")
    _save(fig, "tiles")


def panel_sync(facts, sync):
    fig, w, h = _fig("sync")
    e, b = sync["matched_ephys"], sync["matched_behavior"]
    resid_ms = (b - (sync["slope"] * e + sync["intercept"])) * 1e3
    ipi_e, ipi_b = np.diff(e), np.diff(b)
    ax1 = fig.add_axes([0.11, 0.2, 0.36, 0.66])
    ax1.scatter(ipi_e, ipi_b, s=70, color=STREAM["sync"], edgecolor="white", lw=1, zorder=3)
    lo, hi = min(ipi_e.min(), ipi_b.min()), max(ipi_e.max(), ipi_b.max())
    ax1.plot([lo, hi], [lo, hi], color=MUTED, lw=2, ls="--", zorder=2)
    ax1.set_xlabel("Inter-pulse interval, ephys (s)")
    ax1.set_ylabel("IPI, behavior PC (s)")
    ax1.set_title("Pulse patterns match", loc="left")
    ax2 = fig.add_axes([0.6, 0.2, 0.37, 0.66])
    ax2.axhline(0, color=MUTED, lw=2)
    ax2.plot((e - e.min()) / 60, resid_ms * 1e3, "o", ms=8, color=STREAM["sync"], mec="white", mew=1)
    ax2.set_xlabel("Ephys time (min)")
    ax2.set_ylabel("Fit residual (µs)")
    ax2.set_title(f"Linear fit: {facts['sync_n']} pulses, R² = {facts['sync_r2']:.6f}", loc="left")
    _save(fig, "sync")


def _shot(ax, path, crop=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if crop:
        W, H = im.size
        im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
    ax.imshow(np.asarray(im), interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(RULE)
        s.set_linewidth(2)
    return im.size


def _callouts(ax, size, pts):
    W, H = size
    r = 0.022 * W
    for n, (fx, fy) in enumerate(pts, 1):
        ax.add_patch(Circle((fx * W, fy * H), r, fc=SIG, ec="white", lw=4, zorder=5))
        ax.text(fx * W, fy * H, str(n), ha="center", va="center", color="white",
                fontsize=30, fontweight="bold", zorder=6)


def panel_gui_browser(facts):
    fig, w, h = _fig("gui_browser")
    ax = fig.add_axes([0, 0, 1, 1])
    size = _shot(ax, spec.GUI / "panel_browser_c2.png")
    _callouts(ax, size, [(0.36, 0.30), (0.75, 0.40), (0.855, 0.765), (0.95, 0.80)])
    _save(fig, "gui_browser")


def panel_gui_explore(facts):
    fig, w, h = _fig("gui_explore")
    ax = fig.add_axes([0, 0, 1, 1])
    size = _shot(ax, spec.GUI / "panel_explore_clean_2.png")
    _callouts(ax, size, [(0.213, 0.046), (0.213, 0.241), (0.21, 0.57), (0.598, 0.575)])
    _save(fig, "gui_explore")


def panel_gui_streamlit(facts):
    fig, w, h = _fig("gui_streamlit")
    left = fig.add_axes([0.0, 0.0, 0.53, 1.0])
    _shot(left, spec.GUI / "st_tracking_tall.png", crop=(0.2, 0.225, 0.96, 0.655))
    right = fig.add_axes([0.55, 0.0, 0.45, 1.0])
    _shot(right, spec.GUI / "st_behavioral_tall.png", crop=(0.205, 0.226, 0.785, 0.84))
    for a in (left, right):
        a.set_aspect("auto")
    _save(fig, "gui_streamlit")


def panel_decode_method(facts):
    fig, w, h = _fig("decode_method")
    ax = _inch_axes(fig, w, h)
    steps = [("Encounter onsets", "−1 … +2 s, 0.5 s bins"),
             ("Per-cell LDA", "5-fold cross-validated"),
             ("Label-shuffle null", f"staged, ≤ {facts['max_shuffles']:,} perms"),
             ("BH-FDR", "within each rat, q < 0.05")]
    gap = 0.55
    bw = (w - gap * (len(steps) - 1)) / len(steps)
    for i, (a, b) in enumerate(steps):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.15), bw, h - 0.3, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=PANEL_BG, ec=ACCENT, lw=3))
        ax.text(x + bw / 2, h * 0.62, a, ha="center", va="center", fontsize=FIG_TICK_PT,
                fontweight="bold", color=NAVY)
        ax.text(x + bw / 2, h * 0.33, b, ha="center", va="center", fontsize=SMALL_PT - 2, color=MUTED)
        if i < len(steps) - 1:
            _arrow(ax, x + bw + 0.06, h / 2, x + bw + gap - 0.06, h / 2, color=ACCENT)
    _save(fig, "decode_method")


def _cohort_bands(ax, rats, cohorts, y):
    prev, start = None, 0
    for i, c in enumerate(cohorts + [None]):
        if c != prev and prev is not None:
            ax.annotate("", xy=(start - 0.4, y), xytext=(i - 1 + 0.4, y), xycoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="-", color=COHORT[prev], lw=6))
            ax.text((start + i - 1) / 2, y + 0.035, prev.replace("cohort", "cohort "), ha="center",
                    va="bottom", transform=ax.get_xaxis_transform(), fontsize=FIG_TICK_PT,
                    color=COHORT[prev], fontweight="bold")
            start = i
        prev = c


def panel_decode_by_rat(facts):
    fig, w, h = _fig("decode_by_rat")
    fdr = pd.read_csv(REPO / "results" / "event_decoding" / "refined_fdr_opponent.csv")
    ax = fig.add_axes([0.1, 0.235, 0.88, 0.63])
    rng = np.random.default_rng(0)
    rats = [t[0] for t in TARGETS]
    for i, (animal, session, cohort) in enumerate(TARGETS):
        d = fdr[(fdr.animal_id == animal) & (fdr.session_id.astype(str) == session)]
        acc = d.accuracy.to_numpy()
        sig = d.significant.to_numpy(bool)
        vp = ax.violinplot(acc, positions=[i], widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(COHORT[cohort])
            b.set_alpha(0.16)
            b.set_edgecolor("none")
        jit = rng.uniform(-0.28, 0.28, acc.size)
        ax.scatter(i + jit[~sig], acc[~sig], s=14, color=NULL, alpha=0.7, lw=0, zorder=2)
        ax.scatter(i + jit[sig], acc[sig], s=60, color=SIG, edgecolor="white", lw=0.8, zorder=3)
        pr = facts["per_rat"][animal]
        ax.plot([i - 0.38, i + 0.38], [pr["baseline"]] * 2, color=INK, lw=4, zorder=4,
                solid_capstyle="butt")
        cc = list(pr["class_counts"].values())
        ax.text(i, -0.115, f"{animal}", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=FIG_TICK_PT, fontweight="bold", color=INK)
        ax.text(i, -0.235, f"{pr['n_sig']}/{pr['n_cells']} sig.\n{pr['n_events']} EC ({cc[0]}/{cc[1]})",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=SMALL_PT - 2,
                color=SIG if pr["n_sig"] else MUTED, linespacing=1.1)
    ax.axhline(0.5, color=MUTED, lw=1.8, ls="--", zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(rats) - 0.4)
    ax.set_ylim(0.2, 0.92)
    ax.set_ylabel("Per-cell accuracy\n(5-fold CV)")
    _cohort_bands(ax, rats, [t[2] for t in TARGETS], 1.02)
    # legend
    ax.scatter([], [], s=60, color=SIG, label="q < 0.05 (BH within rat)")
    ax.scatter([], [], s=24, color=NULL, label="not significant")
    ax.plot([], [], color=INK, lw=4, label="majority-class baseline")
    ax.plot([], [], color=MUTED, lw=1.8, ls="--", label="0.5")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=SMALL_PT - 2,
              handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    _save(fig, "decode_by_rat")


# FDR-significant rat631 cells that do not separate the partners before onset (|d| < 0.3 on
# the [-1, 0] s rate) but do after it (|d| > 0.6 on [0, 2] s); see _onset_d.
EXAMPLE_CELLS = (1119, 626)
NAME_TAG_CELL, NAME_TAG_OPP = 1016, "rat635"   # passes rat635 only (d 1.19; next best 0.07)


def _onset_d(spikes, t, lab, names, lo, hi):
    """Partner-vs-partner d' of the mean rate in [lo, hi) s around onset."""
    s = np.sort(spikes)
    x, y = [(np.searchsorted(s, t[lab == n] + hi) - np.searchsorted(s, t[lab == n] + lo))
            / (hi - lo) for n in names[:2]]
    return float((x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))
HIST_ALL, HIST_SIG = "#8fb3e3", "#d73027"   # as in results/event_decoding/*_hist_fdr_*.png


def panel_decode_examples(facts, dec):
    fdr = pd.read_csv(REPO / "results" / "event_decoding" / "refined_fdr_opponent.csv")
    fig, w, h = _fig("decode_examples")
    t = dec[EXAMPLE_RAT]["event_times"]
    lab = dec[EXAMPLE_RAT]["labels"]
    names = list(facts["per_rat"][EXAMPLE_RAT]["class_counts"])
    lo, hi, bw = -1.0, 2.0, 0.1
    edges = np.arange(lo, hi + 1e-9, bw)
    cen = edges[:-1] + bw / 2
    row = fdr[(fdr.animal_id == EXAMPLE_RAT)].set_index("cluster_id")
    for j, cid in enumerate(EXAMPLE_CELLS):
        spikes = _cached(f"spikes_{EXAMPLE_RAT}_{cid}", lambda c=cid: load_example_spikes(c))
        _expect(f"cell {cid} FDR-significant", bool(row.loc[cid, "significant"]), True)
        _expect(f"cell {cid} silent pre-onset", abs(_onset_d(spikes, t, lab, names, -1, 0)) < 0.3, True)
        _expect(f"cell {cid} splits post-onset", abs(_onset_d(spikes, t, lab, names, 0, 2)) > 0.6, True)
        ax = fig.add_axes([0.075 + j * 0.285, 0.2, 0.22, 0.62])
        for k, name in enumerate(names):
            tt = np.sort(t[lab == name])
            c = np.array([np.histogram(spikes[(spikes >= t0 + lo) & (spikes < t0 + hi)] - t0,
                                       edges)[0] / bw for t0 in tt])
            m, se = c.mean(0), c.std(0) / np.sqrt(len(c))
            ax.fill_between(cen, m - se, m + se, color=OPP[k], alpha=0.25, lw=0)
            ax.plot(cen, m, color=OPP[k], lw=3.2, label=f"vs {name} (n = {len(tt)})")
        ax.axvline(0, color=MUTED, lw=1.8, ls="--")
        ax.set_xlim(lo, hi)
        if j == 0:          # one shared label under both PSTHs (two would collide)
            fig.text(0.075 + (0.285 + 0.22) / 2, 0.075, "Time from encounter onset (s)",
                     ha="center", va="center", fontsize=FIG_LABEL_PT)
            ax.set_ylabel("Rate (Hz)")
        r = row.loc[cid]
        ax.set_title(f"Cell {cid}: acc {r.accuracy:.2f}, q = {r.q_value:.3f}", loc="left",
                     fontsize=FIG_TICK_PT - 2)
        ax.legend(loc="upper left", fontsize=SMALL_PT - 6, handlelength=1.0,
                  borderaxespad=0.1, frameon=True, facecolor="white", edgecolor="none",
                  framealpha=1)
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.35)
    # pooled distribution, all cells vs FDR-significant
    axh = fig.add_axes([0.70, 0.2, 0.285, 0.62])
    bins = np.arange(0.3, 0.875, 0.025)
    axh.hist(fdr.accuracy, bins=bins, color=HIST_ALL, edgecolor="white", lw=1.2, label="All cells")
    axh.hist(fdr.accuracy[fdr.significant], bins=bins, color=HIST_SIG, edgecolor="white", lw=1.2,
             label=f"q < 0.05 (BH), {100 * facts['n_sig_all'] / facts['n_cells_all']:.0f}%")
    axh.axvline(0.5, color=INK, lw=1.6, ls=":")
    axh.set_xlabel("Accuracy (opponent)")
    axh.set_ylabel("Cells")
    axh.set_title(f"{len(TARGETS)} rats pooled, n = {facts['n_cells_all']:,}", loc="left",
                  fontsize=FIG_TICK_PT - 2)
    axh.legend(loc="upper right", fontsize=SMALL_PT - 6, frameon=False, handlelength=1.0,
               borderaxespad=0.1)
    _save(fig, "decode_examples")


def panel_example_cell(facts, dec):
    fdr = pd.read_csv(REPO / "results" / "event_decoding" / "refined_fdr_opponent.csv")
    d = fdr[(fdr.animal_id == EXAMPLE_RAT) & fdr.significant].sort_values("accuracy", ascending=False)
    row = d.iloc[0]
    cid = int(row.cluster_id)
    spikes = _cached(f"spikes_{EXAMPLE_RAT}_{cid}", lambda: load_example_spikes(cid))
    t = dec[EXAMPLE_RAT]["event_times"]
    lab = dec[EXAMPLE_RAT]["labels"]
    names = list(facts["per_rat"][EXAMPLE_RAT]["class_counts"])
    lo, hi = -1.0, 2.0
    fig, w, h = _fig("example_cell")
    ax_r = fig.add_axes([0.1, 0.2, 0.4, 0.63])
    ax_p = fig.add_axes([0.6, 0.2, 0.37, 0.63])
    yrow = 0
    edges = np.arange(lo, hi + 1e-9, 0.1)
    for k, name in enumerate(names):
        tt = np.sort(t[lab == name])
        counts = []
        for t0 in tt:
            s = spikes[(spikes >= t0 + lo) & (spikes < t0 + hi)] - t0
            ax_r.vlines(s, yrow, yrow + 0.9, color=OPP[k], lw=1.4)
            counts.append(np.histogram(s, edges)[0] / 0.1)
            yrow += 1
        counts = np.array(counts)
        m, se = counts.mean(0), counts.std(0) / np.sqrt(len(counts))
        cen = edges[:-1] + 0.05
        ax_p.fill_between(cen, m - se, m + se, color=OPP[k], alpha=0.25, lw=0)
        ax_p.plot(cen, m, color=OPP[k], lw=3.5, label=f"vs {name} (n={len(tt)})")
        yrow += 2
    for a in (ax_r, ax_p):
        a.axvline(0, color=MUTED, lw=2, ls="--")
        a.set_xlim(lo, hi)
        a.set_xlabel("Time from encounter onset (s)")
    ax_r.set_ylabel("Encounters")
    ax_r.set_yticks([])
    ax_p.set_ylabel("Rate (Hz)")
    ax_p.legend(loc="upper right", fontsize=SMALL_PT - 2, handlelength=1.2)
    fig.text(0.1, 0.93, f"Example cell: {EXAMPLE_RAT} cluster {cid}, accuracy {row.accuracy:.2f}, "
             f"q = {row.q_value:.3f}", fontsize=FIG_TICK_PT, fontweight="bold", va="center")
    facts["example"] = {"cluster": cid, "accuracy": float(row.accuracy), "q": float(row.q_value)}
    _save(fig, "example_cell")



def _coupling_schematic(fig, ax, w, h, ccg_rect):
    """Two cells, A excites B; their cross-correlogram peaks 1-4 ms after A spikes."""
    A, B = "#0072B2", "#009E73"
    # neurons
    for (cx, col, name) in ((0.9, A, "A"), (3.1, B, "B")):
        ax.add_patch(plt.Polygon([[cx - 0.42, h - 1.75], [cx + 0.42, h - 1.75], [cx, h - 0.95]],
                                 closed=True, fc=col, ec="none", zorder=3))
        ax.text(cx, h - 1.5, name, ha="center", va="center", color="white",
                fontsize=FIG_TICK_PT, fontweight="bold", zorder=4)
    ax.add_patch(FancyArrowPatch((1.4, h - 1.38), (2.6, h - 1.38), arrowstyle="-|>",
                                 mutation_scale=34, color=INK, lw=3, zorder=2))
    ax.text(2.0, h - 1.2, "excites", ha="center", va="bottom", fontsize=SMALL_PT - 4, color=INK)
    ax.text(0, h - 0.35, "If co-tuned cells are coupled ...", fontsize=FIG_TICK_PT - 2,
            fontweight="bold", color=NAVY, va="center")
    # spike trains: B follows A by ~2 ms
    rng = np.random.default_rng(3)
    ta = np.sort(rng.uniform(0.6, 3.9, 9))
    tb = np.sort(np.r_[ta[rng.random(9) < 0.7] + 0.09, rng.uniform(0.6, 3.9, 3)])
    for y0, tt, col, name in ((h - 2.35, ta, A, "A"), (h - 2.85, tb, B, "B")):
        ax.vlines(tt, y0 - 0.17, y0 + 0.17, color=col, lw=3)
        ax.text(0.05, y0, name, ha="left", va="center", fontsize=SMALL_PT - 4, color=col,
                fontweight="bold")
    for x in ta[:3]:
        ax.plot([x, x + 0.09], [h - 2.52, h - 2.68], color=MUTED, lw=1.2)
    # cross-correlogram (schematic, not data)
    axc = fig.add_axes(ccg_rect)
    lags = np.arange(-10, 10.5, 1.0)
    pred = 20 + 3 * np.exp(-(lags / 7) ** 2)
    obs = pred + rng.normal(0, 1.6, lags.size)
    pk = (lags >= 1) & (lags <= 4)
    obs[pk] += np.array([9, 13, 10, 6])
    axc.bar(lags[~pk], obs[~pk], width=0.9, color=NULL)
    axc.bar(lags[pk], obs[pk], width=0.9, color=SIG)
    axc.plot(lags, pred, color=INK, lw=2.4, ls="--", label="predictor")
    axc.set_xlabel("Lag of B after A (ms)")
    axc.set_ylabel("Count")
    axc.set_yticks([])
    axc.set_xlim(-10.6, 10.6)
    axc.set_ylim(0, 40)
    axc.annotate("peak at +1–4 ms", xy=(2.5, obs[pk].max()), xytext=(4.6, 37),
                 fontsize=SMALL_PT - 5, color=SIG, va="center",
                 arrowprops=dict(arrowstyle="-", color=SIG, lw=1.5))
    axc.text(-10.2, 37, "z = (observed − predicted) / s.d.", fontsize=SMALL_PT - 5,
             color=INK, va="center")
    axc.legend(loc="lower left", fontsize=SMALL_PT - 6, frameon=False, borderaxespad=0.1,
               handlelength=1.4)
    axc.set_title("... their cross-correlogram (schematic)", loc="left", fontsize=FIG_TICK_PT - 4)


def _load_nametag_events():
    from analysis.opponent_barcode import opponent_barcode as ob
    d = ob.load_events_and_responses(EXAMPLE_RAT, "20251216", "opponent", "evoked",
                                     REPO / ".gui_cache" / "opponent_barcode")
    return {"t": np.asarray(d["event_times"], float), "labels": np.asarray(d["labels"]).astype(str)}


def _name_tag_psth(ax):
    """Peri-encounter rate of one name-tag cell: every opponent, the preferred one highlighted."""
    cells = pd.read_csv(REPO / "results" / "opponent_barcode" / "barcode_opponent_cells.csv")
    r = cells[(cells.animal_id == EXAMPLE_RAT) & (cells.cluster_id == NAME_TAG_CELL)].iloc[0]
    dv = r[[c for c in cells.columns if c.startswith("d_")]].dropna().astype(float)
    _expect(f"cell {NAME_TAG_CELL} passes only {NAME_TAG_OPP}",
            [c[2:] for c in dv.index[dv > 0.5]], [NAME_TAG_OPP])
    ev = _cached(f"nametag_events_{EXAMPLE_RAT}", _load_nametag_events)
    spikes = _cached(f"spikes_{EXAMPLE_RAT}_{NAME_TAG_CELL}",
                     lambda: load_example_spikes(NAME_TAG_CELL))
    lo, hi, bw = -1.0, 2.0, 0.1
    edges = np.arange(lo, hi + 1e-9, bw)
    cen = edges[:-1] + bw / 2

    def psth(tt):
        c = np.array([np.histogram(spikes[(spikes >= t0 + lo) & (spikes < t0 + hi)] - t0,
                                   edges)[0] / bw for t0 in tt])
        return c.mean(0), c.std(0) / np.sqrt(len(c))
    names = sorted(set(ev["labels"]))
    for n in names:
        if n != NAME_TAG_OPP:
            ax.plot(cen, psth(ev["t"][ev["labels"] == n])[0], color=NULL, lw=1.6, alpha=0.9)
    m, se = psth(ev["t"][ev["labels"] != NAME_TAG_OPP])
    ax.plot(cen, m, color=INK, lw=3, ls="--", label=f"other {len(names) - 1} opponents (mean)")
    sel = ev["labels"] == NAME_TAG_OPP
    m, se = psth(ev["t"][sel])
    ax.fill_between(cen, m - se, m + se, color=SIG, alpha=0.25, lw=0)
    ax.plot(cen, m, color=SIG, lw=3.6, label=f"vs {NAME_TAG_OPP} (n = {sel.sum()}), d = {dv['d_' + NAME_TAG_OPP]:.1f}")
    ax.plot([], [], color=NULL, lw=1.6, label="each other opponent")
    ax.axvline(0, color=MUTED, lw=1.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlabel("Time from encounter onset (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(f"Name-tag cell {NAME_TAG_CELL} ({EXAMPLE_RAT}): one opponent only", loc="left",
                 fontsize=FIG_TICK_PT - 2)
    ax.legend(loc="upper left", fontsize=SMALL_PT - 6, handlelength=1.2, borderaxespad=0.1,
              frameon=True, facecolor="white", edgecolor="none", framealpha=1)


def panel_barcode(facts, rep):
    fig, w, h = _fig("barcode")
    _name_tag_psth(fig.add_axes([0.085, 0.2, 0.43, 0.62]))
    ax = fig.add_axes([0.64, 0.2, 0.34, 0.62])     # same baseline/height as the PSTH
    for i, (animal, _, cohort) in enumerate(TARGETS):
        b = facts["barcode"][animal]
        ax.plot([i, i], [100 * b["null_lo"], 100 * b["null_hi"]], color=NULL, lw=12,
                solid_capstyle="butt", zorder=1)
        ax.plot([i - 0.22, i + 0.22], [100 * b["null_mean"]] * 2, color=MUTED, lw=3, zorder=2)
        both = b["p_free"] < 0.05 and b["p_block"] < 0.05
        free = b["p_free"] < 0.05
        ax.scatter(i, 100 * b["obs"], s=200, zorder=3, color=SIG if free else "white",
                   edgecolor=SIG if free else INK, lw=3, marker="o" if both else ("D" if free else "o"))
    ax.set_xticks(range(len(TARGETS)), [a.replace("rat", "") for a, _, _ in TARGETS],
                  fontsize=FIG_TICK_PT - 5, fontweight="bold")
    for lab, (_, _, cohort) in zip(ax.get_xticklabels(), TARGETS):
        lab.set_color(COHORT[cohort])
    ax.set_xlim(-0.6, len(TARGETS) - 0.4)
    ax.set_xlabel("Animal ID")
    ax.set_ylabel("Single-opponent\ncells (%)")
    ax.set_title("Name-tag cells vs shuffle null", loc="left", fontsize=FIG_TICK_PT - 2)
    ax.scatter([], [], s=150, color=SIG, label="p < .05, both nulls")
    ax.scatter([], [], s=120, color=SIG, marker="D", label="free null only")
    ax.legend(loc="upper left", fontsize=SMALL_PT - 5, handletextpad=0.2, borderaxespad=0,
              frameon=False)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    _save(fig, "barcode")


def _icon(ax, kind, cx, cy, s, col):
    """Simple line icons, s = size in inches."""
    from matplotlib.patches import Arc, Polygon
    lw = 3.2
    if kind == "run":
        ax.add_patch(Circle((cx, cy), s * 0.5, fc="none", ec=col, lw=lw, zorder=3))
        ax.add_patch(Polygon([[cx - s * 0.15, cy - s * 0.22], [cx - s * 0.15, cy + s * 0.22],
                              [cx + s * 0.24, cy]], closed=True, fc=col, ec=col, zorder=3))
    elif kind == "chart":
        x0, y0 = cx - s * 0.45, cy - s * 0.4
        ax.plot([x0, x0, x0 + s * 0.9], [y0 + s * 0.85, y0, y0], color=col, lw=lw, zorder=3)
        for i, hh in enumerate((0.3, 0.55, 0.42, 0.75)):
            ax.add_patch(Rectangle((x0 + s * (0.1 + 0.2 * i), y0), s * 0.13, s * hh,
                                   fc=col, ec="none", zorder=3))
    elif kind == "doc":
        for dx, dy in ((0.1, -0.08), (0, 0)):
            ax.add_patch(Rectangle((cx - s * 0.32 + s * dx, cy - s * 0.45 + s * dy), s * 0.6,
                                   s * 0.85, fc="white", ec=col, lw=lw, zorder=3))
        for i in range(4):
            yy = cy + s * (0.25 - 0.17 * i)
            ax.plot([cx - s * 0.2, cx + s * 0.18], [yy, yy], color=col, lw=2.2, zorder=4)
    elif kind == "bulb":
        ax.add_patch(Circle((cx, cy + s * 0.08), s * 0.3, fc="none", ec=col, lw=lw, zorder=3))
        ax.add_patch(Rectangle((cx - s * 0.13, cy - s * 0.42), s * 0.26, s * 0.2, fc=col,
                               ec="none", zorder=3))
        for ang in (90, 30, 150, 0, 180):
            a = np.deg2rad(ang)
            ax.plot([cx + np.cos(a) * s * 0.4, cx + np.cos(a) * s * 0.52],
                    [cy + s * 0.08 + np.sin(a) * s * 0.4, cy + s * 0.08 + np.sin(a) * s * 0.52],
                    color=col, lw=2.4, zorder=3)
    elif kind == "code":
        ax.text(cx, cy, "</>", ha="center", va="center", fontsize=40, family="Consolas",
                fontweight="bold", color=col, zorder=3)
    elif kind == "rerun":
        ax.add_patch(Arc((cx, cy), s * 0.85, s * 0.85, theta1=40, theta2=330, color=col, lw=lw,
                         zorder=3))
        a = np.deg2rad(40)
        tx, ty = cx + np.cos(a) * s * 0.425, cy + np.sin(a) * s * 0.425
        ax.add_patch(Polygon([[tx - s * 0.16, ty + s * 0.02], [tx + s * 0.12, ty + s * 0.1],
                              [tx + s * 0.02, ty - s * 0.2]], closed=True, fc=col, ec=col, zorder=3))


def panel_ai_loop(facts):
    fig, w, h = _fig("ai_loop")
    ax = _inch_axes(fig, w, h)
    S = SMALL_PT
    # Layer 0 (top): what every stage consults before acting
    _box(ax, 0, h - 0.62, w, 0.62,
         f"Layer 0: capability manifest (can this run here?)  +  hazard registry "
         f"({facts['n_hazards']} known traps, executable detectors)",
         "#ffffff", ec=ACCENT, tc=ACCENT, size=S - 3, lw=3)
    stages = [("Run", "run-analysis"), ("Interpret", "interpreter\n(numbers first)"),
              ("Literature", "PubMed ·\nbioRxiv"), ("Hypothesize", "propose-\nhypotheses"),
              ("Implement", "implement-\nmodule · coder"), ("Re-run", "run-analysis")]
    gap = 0.36
    bw = (w - 5 * gap) / 6
    bh = 1.62 + max(0.0, h - 4.3)                  # taller slot -> taller stage boxes
    by = h - 0.62 - 0.55 - bh
    gates = {1: "real?", 3: "which?", 4: "code OK?"}      # gate after stage i
    icons = ["run", "chart", "doc", "bulb", "code", "rerun"]
    for i, (t, tag) in enumerate(stages):
        x = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="white", ec=ACCENT, lw=3.5, zorder=2))
        ax.text(x + bw / 2, by + bh - 0.36, f"{i + 1}  {t}", ha="center", va="center",
                color=NAVY, fontsize=S - 1, fontweight="bold", zorder=3)
        _icon(ax, icons[i], x + bw / 2, by + bh / 2 + 0.05, min(0.95, bh - 1.7), ACCENT)
        ax.text(x + bw / 2, by + 0.45, tag, ha="center", va="center", color=MUTED,
                fontsize=S - 6, family="Consolas", zorder=3, linespacing=1.05)
        if i < 5:
            xa, ya = x + bw, by + bh / 2
            _arrow(ax, xa + 0.02, ya, xa + gap - 0.02, ya, lw=2.6)
            if i in gates:
                cx = xa + gap / 2
                ax.scatter([cx], [ya + 0.02], marker="D", s=420, color=SIG, zorder=4,
                           edgecolors="white", linewidths=2)
                ax.text(cx, by + bh + 0.08, gates[i], ha="center", va="bottom", color=SIG,
                        fontsize=S - 6, fontweight="bold", zorder=4)
    # loop-back arrow, Re-run -> Run, beneath the boxes
    yl = by - 0.42
    x6, x1 = 5 * (bw + gap) + bw / 2, bw / 2
    ax.plot([x6, x6, x1], [by, yl, yl], color=MUTED, lw=2.6, zorder=1,
            solid_capstyle="butt")
    _arrow(ax, x1, yl, x1, by - 0.01, lw=2.6)
    ax.text((x1 + x6) / 2, yl + 0.04, "iterate", ha="center", va="bottom", color=MUTED,
            fontsize=S - 6, style="italic")
    # holdout gate on the way back
    hx = x6 - bw * 0.9
    ax.scatter([hx], [yl], marker="D", s=420, color=SIG, zorder=4, edgecolors="white",
               linewidths=2)
    ax.text(hx + 0.25, yl + 0.04, "holdout?", ha="left", va="bottom", color=SIG,
            fontsize=S - 6, fontweight="bold")
    # lab notebook (bottom)
    nh = 0.95
    _box(ax, 0, 0, w, nh,
         f"Lab notebook (SQLite): every run logged with git commit, parameters, figures and its "
         f"test family\n{facts['n_hypotheses']} hypotheses · {facts['n_families']} "
         f"test families · {facts['n_iterations']} logged iterations",
         "#2f6f4e", size=S - 4)
    ax.scatter([w - 0.05], [h - 0.62 - 0.17], s=0)                 # keep limits
    _save(fig, "ai_loop")


FIGS = spec.OUT / "figures"
TEAL = "#0e8a8a"


def _img(ax, path, x, y, w, h, align="center"):
    """Place an image file inside the box (x, y, w, h) of an inch axes, keeping its aspect."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    iw, ih = im.size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    x0 = x + (w - dw) / 2 if align == "center" else x
    y0 = y + (h - dh) / 2
    ax.imshow(np.asarray(im), extent=(x0, x0 + dw, y0, y0 + dh), interpolation="lanczos",
              zorder=2)
    return x0, y0, dw, dh


def _card(ax, x, y, w, h, name, title, desc, color, size=SMALL_PT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=RULE, lw=2, zorder=2))
    ax.add_patch(Rectangle((x, y), 0.09, h, fc=color, ec="none", zorder=3))
    ax.text(x + 0.25, y + h - 0.12, name, fontsize=size - 3, family="Consolas",
            fontweight="bold", color=color, va="top", zorder=3)
    ax.text(x + 0.25, y + h - 0.52, title, fontsize=size - 5, fontweight="bold", color=INK,
            va="top", zorder=3)
    ax.text(x + 0.25, y + h - 0.88, desc, fontsize=size - 7, color=MUTED, va="top", zorder=3,
            linespacing=1.15)


def panel_logo(facts):
    fig, w, h = _fig("logo")
    ax = _inch_axes(fig, w, h)
    _img(ax, FIGS / "janelia_logo.png", 0, 0, w, h)
    _save(fig, "logo")


def panel_arena(facts):
    fig, w, h = _fig("arena")
    ax = _inch_axes(fig, w, h)
    _img(ax, FIGS / "Transparent rat Behavioral Arena schematic.png", 0, 0, w, h)
    _save(fig, "arena")



def panel_data_formats(facts):
    from PIL import Image
    fig, w, h = _fig("data_formats")
    ax = _inch_axes(fig, w, h)
    S = SMALL_PT
    vid = FIGS / "video_frame.jpg"
    rows = [
        ("Electrophysiology", STREAM["ephys"],
         (FIGS / "deck_raw_signals.jpg", "multi-channel voltage, 30 kHz", None),
         (FIGS / "deck_kilosort.jpg", "Kilosort 4: spikes → clusters", None),
         [("spike_times.npy", "sample index per spike"),
          ("spike_clusters.npy", "cluster id per spike"),
          ("cluster_info.tsv", "quality label · channel · depth"),
          ("*.timestamps.dat", "sample → day clock")]),
        ("Video tracking", STREAM["video"],
         (vid, "overhead camera, timestamped", None),
         (FIGS / "rat_trajectories.png", "multi-animal tracks (APT)", None),
         [("TQT_named.csv", "APT, wide: timestamp (ns),\n<rat>_center_x/y, kp0–3"),
          ("*_mask_metrics.csv", "manual, long: frame ·\nobject_name · center_x/y")]),
        ("Behavior", STREAM["events"],
         (FIGS / "video_frame_behavior.jpg", "video reviewed by scorers", None),
         (FIGS / "event_types_2.png", "scored social events", None),
         [("<date>_behavior_event_df.csv",
           "one row per event: type · initiator ·\nvictim · winner · loser · location ·\n"
           "duration · ts_start / ts_end (ns)")]),
    ]
    colw = [4.35, 4.35]
    arrow_w = 0.5
    xs = [0.0, colw[0] + arrow_w]
    xc = xs[1] + colw[1] + arrow_w
    card_w = w - xc
    head_h, lab_h, gap_r, cap_h, ch_gap = 1.05, 0.5, 0.25, 0.42, 0.1

    def card_h(desc):
        return 0.5 + 0.29 * (desc.count(chr(10)) + 1) + 0.12

    def load(path, crop):
        im = Image.open(path).convert("RGBA")
        if crop:
            W, H = im.size
            im = im.crop((int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H)))
        # flatten onto white: transparent pixels can carry dark RGB that bleeds when resampled
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im)

    def row_h(r, cap=2.6):
        need_img = max(min(colw[k] * load(p_, c_).size[1] / load(p_, c_).size[0], cap)
                       for k, (p_, _, c_) in enumerate((r[2], r[3]))) + cap_h
        need_cards = sum(card_h(d) for _, d in r[4]) + ch_gap * (len(r[4]) - 1)
        return max(need_img, need_cards)

    rhs = [row_h(r) for r in rows]
    natural = head_h + sum(rhs) + 3 * lab_h + 2 * gap_r
    cap = 2.6
    while natural < h - 0.05 and cap < 8:       # grow the images into a taller slot
        cap += 0.05
        rhs = [row_h(r, cap) for r in rows]
        natural = head_h + sum(rhs) + 3 * lab_h + 2 * gap_r
    if natural > h:
        cap -= 0.05
        rhs = [row_h(r, cap) for r in rows]
        natural = head_h + sum(rhs) + 3 * lab_h + 2 * gap_r
    print(f"  data_formats natural height {natural:.2f} in (slot {h:.2f})")
    top = h - max(0.0, (h - natural) / 2)
    # conceptual columns: shaded bands with a heading each
    heads = [("1  RAW DATA", "as acquired"), ("2  PROCESSED", "sorted · tracked · scored"),
             ("3  DIGITAL ARTIFACTS", "files the pipeline reads")]
    for k, (x0, cw) in enumerate(((xs[0], colw[0]), (xs[1], colw[1]), (xc, card_w))):
        ax.add_patch(FancyBboxPatch((x0 - 0.08, top - natural), cw + 0.16, natural,
                                    boxstyle="round,pad=0,rounding_size=0.12", fc="#f2f5f9",
                                    ec="none", zorder=0))
        ax.text(x0 + cw / 2, top - 0.33, heads[k][0], ha="center", va="center",
                fontsize=S - 1, fontweight="bold", color=NAVY)
        ax.text(x0 + cw / 2, top - 0.75, heads[k][1], ha="center", va="center",
                fontsize=S - 5, color=MUTED, style="italic")
        if k < 2:
            ax.add_patch(FancyArrowPatch((x0 + cw + 0.1, top - 0.33),
                                         (x0 + cw + arrow_w - 0.1, top - 0.33),
                                         arrowstyle="-|>", mutation_scale=30, color=NAVY, lw=3))
    y = top - head_h
    for (title, col, raw, proc, cards), rh in zip(rows, rhs):
        ax.text(0.05, y - lab_h / 2, title, fontsize=S - 1, fontweight="bold", color=col,
                va="center")
        y -= lab_h + rh
        for k, (path, cap, crop) in enumerate((raw, proc)):
            im = np.asarray(load(path, crop))
            iw, ih = im.shape[1], im.shape[0]
            sc = min(colw[k] / iw, (rh - cap_h) / ih)
            dw, dh = iw * sc, ih * sc
            x0, y0 = xs[k] + (colw[k] - dw) / 2, y + cap_h + (rh - cap_h - dh) / 2
            ax.imshow(im, extent=(x0, x0 + dw, y0, y0 + dh), interpolation="lanczos", zorder=2)
            ax.add_patch(Rectangle((x0, y0), dw, dh, fc="none", ec=RULE, lw=1.5, zorder=3))
            ax.text(xs[k] + colw[k] / 2, y0 - 0.06, cap, fontsize=S - 6, color=MUTED,
                    ha="center", va="top")
            ya = y + cap_h + (rh - cap_h) / 2
            _arrow(ax, xs[k] + colw[k] + 0.06, ya, xs[k] + colw[k] + arrow_w - 0.06, ya,
                   color=col, lw=2.6)
        need_cards = sum(card_h(d) for _, d in cards) + ch_gap * (len(cards) - 1)
        cy = y + rh - (rh - need_cards) / 2
        for name, desc in cards:
            ch = card_h(desc)
            cy -= ch
            ax.add_patch(FancyBboxPatch((xc, cy), card_w, ch,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="white", ec=RULE, lw=2, zorder=2))
            ax.add_patch(Rectangle((xc, cy), 0.09, ch, fc=col, ec="none", zorder=3))
            ax.text(xc + 0.25, cy + ch - 0.1, name, fontsize=S - 3,
                    family="Consolas", fontweight="bold", color=col, va="top", zorder=3)
            ax.text(xc + 0.25, cy + ch - 0.5, desc, fontsize=S - 4,
                    color=INK, va="top", zorder=3, linespacing=1.12)
            cy -= ch_gap
        y -= gap_r
    _save(fig, "data_formats")


TIER_NOTE = {"FILES": "as written by Trodes, Kilosort, APT and scorers",
             "PATH DISCOVERY": "no paths in code",
             "DATA CLASSES": "one aligned session model",
             "ANALYSIS MODULES": "analysis/<name>/, each with a CLI",
             "VISUALIZATION MODULES": "shared plotting",
             "GUI & NOTEBOOKS": "same loaders, same caches"}


def panel_architecture(facts):
    fig, w, h = _fig("architecture")
    ax = _inch_axes(fig, w, h)
    S = SMALL_PT
    tiers = [
        ("FILES", [("Kilosort 4\nspikes", STREAM["ephys"]), ("DIO +\npulse log", STREAM["sync"]),
                   ("Tracking\nmanual · APT", STREAM["video"]), ("Scored\nevents", STREAM["events"]),
                   ("NWB\n(DANDI)", "#7f7f7f")], 1.15),
        ("PATH DISCOVERY", [("DataStorageManager · per-cohort config/*.json\n"
                             "recordings, blocks and companion files", NAVY)], 1.05),
        ("DATA CLASSES", [("KilosortData", STREAM["ephys"]), ("DataSyncManager", STREAM["sync"]),
                          ("VideoTrackingData", STREAM["video"]),
                          ("BehavioralEventsData", STREAM["events"])], 0.85),
        ("ANALYSIS MODULES", None, 0),
        ("VISUALIZATION MODULES", [("decoding_plots", TEAL), ("rastermap_viz", TEAL),
                                   ("behavioral_\nvisualization", TEAL),
                                   ("inter_brain_plots", TEAL)], 1.05),
        ("GUI & NOTEBOOKS", [("Panel\nsession browser · Explore", "#2f6f4e"),
                             ("Streamlit\none tab per analysis", "#2f6f4e"),
                             ("Jupyter\nnotebooks", "#2f6f4e"),
                             ("Lab-notebook\nHTML reports", "#2f6f4e")], 1.15),
    ]
    analyses = [["decode_opponent_identity", "decode_event_outcome", "opponent_barcode"],
                ["social_spatial_fields", "inter_brain_dynamics", "partner_distance"],
                ["decode_location", "population_geometry", "territory"]]
    k = 1.0
    lab_h, tgap = 0.45, 0.32
    an_box, an_gap, core_h = 0.75, 0.12, 0.95
    an_h = core_h + 0.22 + 3 * an_box + 2 * an_gap
    total = sum(t[2] for t in tiers) + an_h + len(tiers) * lab_h + (len(tiers) - 1) * tgap
    f = min(1.1, 0.98 * h / total)                 # fit the slot; stretch at most 10%
    tiers = [(n_, b_, bh_ * f) for n_, b_, bh_ in tiers]
    tgap, an_box, an_gap, core_h = tgap * f, an_box * f, an_gap * f, core_h * f
    an_h = core_h + 0.22 * f + 3 * an_box + 2 * an_gap
    lab_h = lab_h * f
    total = sum(t[2] for t in tiers) + an_h + len(tiers) * lab_h + (len(tiers) - 1) * tgap
    y = h - (h - total) / 2
    prev_bottom = None
    for name, boxes, bh in tiers:
        ax.text(0, y - lab_h / 2, name, fontsize=S - 1, fontweight="bold", color=MUTED,
                va="center")
        ax.text(w, y - lab_h / 2, TIER_NOTE.get(name, ""), fontsize=S - 5, color=MUTED,
                va="center", ha="right", style="italic")
        top = y - lab_h
        if prev_bottom is not None:
            _arrow(ax, w / 2, prev_bottom - 0.02, w / 2, y - 0.08, color=TEAL, lw=3)
        if boxes is None:                                # analysis tier
            _box(ax, 0, top - core_h, w * 0.4, core_h, "_lda_decoding core\nlabel-agnostic LDA + nulls",
                 "white", ec=TEAL, tc=TEAL, size=S - 2, lw=3)
            _box(ax, w * 0.42, top - core_h, w * 0.58, core_h,
                 "_stats_utils: add-one p, BH-FDR,\nfdr_resolution, majority baseline",
                 "white", ec=TEAL, tc=TEAL, size=S - 2, lw=3)
            bw3 = (w - 2 * 0.18) / 3
            for r, row in enumerate(analyses):
                for c, t in enumerate(row):
                    by = top - core_h - 0.22 * f - (r + 1) * an_box - r * an_gap
                    _box(ax, c * (bw3 + 0.18), by, bw3, an_box, t, PANEL_BG, tc=INK,
                         size=S - 3, bold=False)
            for c in range(2):
                _arrow(ax, (c + 0.5) * (bw3 + 0.18) - 0.09, top - core_h,
                       (c + 0.5) * (bw3 + 0.18) - 0.09, top - core_h - 0.2 * f, color=TEAL, lw=2.5)
            bottom = top - an_h
        else:
            n = len(boxes)
            bw = (w - (n - 1) * 0.2) / n
            for i, (t, col) in enumerate(boxes):
                filled = name in ("FILES", "PATH DISCOVERY", "GUI & NOTEBOOKS")
                _box(ax, i * (bw + 0.2), top - bh, bw, bh, t, col if filled else "white",
                     ec=col, tc="white" if filled else col, size=S - (1 if n > 1 else 0),
                     lw=0 if filled else 4)
            bottom = top - bh
        prev_bottom = bottom
        y = bottom - tgap
    _save(fig, "architecture")


def connectivity_facts():
    d = REPO / "results" / "opponent_barcode"
    sm = pd.read_csv(d / "barcode_connectivity_rat631_summary.csv")
    sm = sm[sm.variant == "all"].set_index(["measure", "stat"])
    k = "co-tuned minus competing (pref-label perm)"
    z, zr = sm.loc[("z", k)], sm.loc[("z_resid", k)]
    pairs = pd.read_csv(d / "barcode_connectivity_rat631_pairs.csv")
    n_co = int((pairs["class"] == "co-tuned").sum())
    n_comp = int((pairs["class"] == "competing").sum())
    f = {"conn_diff": float(z.observed), "conn_lo": float(z.null_lo), "conn_hi": float(z.null_hi),
         "conn_p": float(z.p), "conn_diff_resid": float(zr.observed),
         "conn_lo_resid": float(zr.null_lo), "conn_hi_resid": float(zr.null_hi),
         "conn_p_resid": float(zr.p),
         "conn_exc_co": 100 * float(sm.loc[("z", "mean_co-tuned")].frac_exc),
         "conn_exc_comp": 100 * float(sm.loc[("z", "mean_competing")].frac_exc),
         "conn_fisher": float(sm.loc[("conn", "fisher_exc")].p),
         "conn_n_co": n_co, "conn_n_comp": n_comp, "conn_n_pairs": int(len(pairs))}
    _expect("connectivity p (label perm)", f["conn_p"], 0.0904, tol=5e-4)
    _expect("connectivity p (residualised)", f["conn_p_resid"], 0.0714, tol=5e-4)
    _expect("co-tuned pairs", n_co, 144)
    _expect("competing pairs", n_comp, 362)
    return f


def panel_connectivity(facts):
    fig, w, h = _fig("connectivity")
    pairs = pd.read_csv(REPO / "results" / "opponent_barcode" / "barcode_connectivity_rat631_pairs.csv")
    classes = [("co-tuned", "co-tuned", "#009E73"),
               ("competing", "competing", "#D55E00"),
               ("barcode x non", "name-tag\n× non", "#7f7f7f"),
               ("non x non", "non × non", "#b9bec8")]
    ax0 = fig.add_axes([0, 0, 0.5, 1])
    ax0.set_xlim(0, w * 0.5)
    ax0.set_ylim(0, h)
    ax0.axis("off")
    _coupling_schematic(fig, ax0, w * 0.5, h, ccg_rect=[0.075, 0.17, 0.39, 0.34])
    axl = fig.add_axes([0.6, 0.2, 0.385, 0.62])
    rng = np.random.default_rng(0)
    for i, (k, lab, col) in enumerate(classes):
        v = pairs.loc[pairs["class"] == k, "z"].to_numpy()
        vp = axl.violinplot(v[(v > -4) & (v < 6.2)], positions=[i], widths=0.85,
                            showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col)
            b.set_alpha(0.22)
            b.set_edgecolor(col)
            b.set_linewidth(1.5)
        show = v if len(v) <= 600 else rng.choice(v, 600, replace=False)
        axl.scatter(i + rng.uniform(-0.28, 0.28, len(show)), show, s=10, color=col, alpha=0.45,
                    lw=0, rasterized=True)
        axl.plot([i - 0.34, i + 0.34], [v.mean()] * 2, color=INK, lw=4)
        axl.text(i, 6.4, f"n = {len(v):,}", ha="center", fontsize=SMALL_PT - 4, color=MUTED)
    axl.axhline(0, color=MUTED, lw=1.2)
    axl.set_ylim(-4, 7.3)
    axl.set_xticks(range(4), [c[1] for c in classes])
    axl.set_ylabel("Excess z, +1–4 ms")
    axl.set_title(f"... measured (rat631): co-tuned − competing p = {facts['conn_p']:.2f}",
                  fontsize=FIG_TICK_PT - 4, loc="left")
    axl.tick_params(axis="x", labelsize=SMALL_PT - 4)
    _save(fig, "connectivity")


PANEL_FUNCS = ["qr", "logo", "arena", "data_formats", "architecture", "gui_browser",
               "gui_explore", "ai_loop", "decode_method", "decode_examples", "barcode",
               "connectivity"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", default=None, choices=PANEL_FUNCS)
    ap.add_argument("--refresh", action="store_true", help="re-read share-backed inputs")
    args = ap.parse_args(argv)
    facts, dec, sync, rep = build_facts(args.refresh)
    todo = args.only or PANEL_FUNCS
    print("rendering panels ...")
    for name in todo:
        fn = globals()[f"panel_{name}"]
        extra = {"sync": (sync,), "example_cell": (dec,), "decode_examples": (dec,),
                 "barcode": (rep,)}.get(name, ())
        fn(facts, *extra)
    (spec.DATA / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
