"""
FERAL import: per-animal predicted bouts -> paired, non-overlapping dyadic events
==================================================================================

FERAL (automatic behaviour classifier) annotated every cohort-7 movie (Dec 9-22
2025, 09:00-21:00, one 30-minute movie per half hour). Its export
``RatCityBehavior/cohort7/FERAL/feral_events.csv`` cannot be used as a
pipeline event file as it stands, for two reasons:

* **One row per animal per bout, no partner.** The pipeline's event model is
  dyadic (``initiator``/``victim``, see
  :meth:`~video.behavioral_events.BehavioralEventsData.extract_opponent_labels`),
  and one interaction appears once for *each* animal involved.
* **Each behaviour is thresholded independently**, so one animal's bouts of
  different types overlap (25% of its EC bouts overlap its own SN bouts). In
  the manual annotations, labels inside one dyad essentially never overlap.

The partner *is* known upstream. The movie browser's per-movie FERAL file
``movie_browser/behavior_feral/<movie>.json`` holds the same bouts as
``best: [[tracklet, behavior, role, f0, f1, partner_tracklet], ...]``, with
``role`` = ``initiator``/``victim`` for directed behaviours and ``""`` for
encounters. It also holds ``gt``, the manual labels in the same format, for
the annotated movies. The CSV is the same predictions with partner and role
dropped. Verified 2026-11-24: rank-joined within (movie, tracklet, behaviour),
every group agrees on all 322 movies apart from two JSON-only bouts of a rat
already removed from the arena. Frames map to the CSV times exactly where the
APT frame timestamps exist.

Pipeline
--------
1. :func:`load_feral_bouts` joins the JSON (frames, role, partner) to the CSV
   (wall-clock times, animal ids) and converts the times to UTC Linux
   nanoseconds, the clock of the manual event files.
2. :func:`pair_bouts` merges reciprocal bouts (A names B, B names A, same
   behaviour, overlapping) into one dyadic event over the union of the two
   intervals. Every bout is kept and graded by ``support``:

   * ``reciprocal``: the partner has the matching bout back;
   * ``cross_supported``: the partner has some *other* bout pointing back at
     this animal at the same time (e.g. F answered by D);
   * ``one_sided``: only one animal's classifier fired.

3. :func:`resolve_overlaps` keeps one label per frame in each dyad, choosing
   by ``priority`` (default PS > F > C > D > EC > SN). It then re-segments,
   drops runs shorter than 8 frames and bridges same-label gaps of up to 8
   frames, which are FERAL's own bout rules. Suppressed labels are recorded
   in ``co_labels``. Overlaps *across* dyads (A fights B while C sniffs A) are
   real and kept; ``n_concurrent_partners`` flags them.

The output CSV uses the canonical column names (``type``, ``initiator``,
``victim``, ``ts_start``, ``ts_end`` in Linux ns, ``session`` = movie name), so
:func:`~video.behavioral_events.load_behavioral_events`,
``synchronize_with_ephys`` and the ``--events_file`` override of the decoding
scripts take it unchanged. FERAL has no winner/loser, so opponent decoding
applies and outcome decoding does not.

Caveats
-------
* Predictions are uncorrected for false positives (FERAL README). SN and D
  are predicted far more often than they were annotated.
* EC is symmetric. Its ``initiator`` is the animal whose bout began first and
  carries no meaning (``directed=False``).
* Frame times are the APT frame timestamps. Where both exist they lie within
  75 ms (median ~7 ms) of the raw video ``_ts.npy``.

Entry points
------------
    python -m ingestion.feral_import --validate          # build + validate
    from ingestion.feral_import import load_feral_events
    events = load_feral_events(path, date='20251216', min_support='reciprocal')
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # ``python ingestion/feral_import.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion._conventions import normalize_object_name, uses_name_normalization  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Facts about the FERAL export
# ---------------------------------------------------------------------------

#: Movie frame rate. JSON frames are inclusive ``[f0, f1]``.
FPS = 40

#: Wall-clock zone of ``start_time``/``end_time`` in ``feral_events.csv``.
LOCAL_TZ = "America/New_York"

#: Behaviours FERAL labels without a direction (role ``""`` in the JSON).
SYMMETRIC_BEHAVIORS = frozenset({"EC"})

#: Default per-frame priority inside a dyad. Specific agonistic labels beat
#: the generic encounter, and EC beats SN because FERAL predicts SN far more
#: often than it was annotated. PS is ranked above F on evidence: on FERAL's
#: two validation movies, where F and PS are both predicted in one dyad, the
#: GT says PS on 1356 frames and F on 14, so F-first cost PS a third of its
#: recall. The training movies have no such conflicts and score identically
#: under either order. Because the validation movies informed this choice,
#: their F/PS agreement is no longer a strictly held-out estimate.
DEFAULT_PRIORITY: Tuple[str, ...] = ("PS", "F", "C", "D", "EC", "SN")

#: FERAL's own bout rules (README): bouts under 8 frames dropped, gaps of up to
#: 8 frames bridged.
DEFAULT_MIN_FRAMES = 8
DEFAULT_BRIDGE_FRAMES = 8

#: Best first.
SUPPORT_LEVELS: Tuple[str, ...] = ("reciprocal", "cross_supported", "one_sided")
_SUPPORT_RANK = {s: i for i, s in enumerate(SUPPORT_LEVELS)}

#: Encounter and sniff were annotated only for these (helmet) tracklets, so a
#: prediction between two other animals cannot be scored against the GT.
HELMET_TRACKLETS = frozenset({2, 4, 7, 8, 9})
_HELMET_ONLY_BEHAVIORS = frozenset({"EC", "SN"})

#: FERAL's threshold-tuning (validation) movies; the other GT movies were
#: training data, so agreement on them flatters the model.
GT_VALIDATION_MOVIES = frozenset({"cohort7_20251216_0929", "cohort7_20251216_1159"})

OUTPUT_FILENAME = "feral_dyadic_events.csv"
REPORT_FILENAME = "feral_import_report.json"

#: Column order of the output CSV. The first block matches the manual event
#: files so existing loaders and filters apply.
OUTPUT_COLUMNS = [
    "event_id", "date", "session", "type", "initiator", "victim", "winner", "loser",
    "ts_start", "ts_end", "start_time_local", "end_time_local", "duration",
    "tracklet_initiator", "tracklet_victim", "frame_start", "frame_end",
    "support", "role_basis", "directed", "iou_pair", "n_source_bouts",
    "co_labels", "mutual", "trimmed", "n_concurrent_partners", "at_movie_edge",
    "source",
]

_JSON_COLUMNS = ["movie", "tracklet", "behavior", "role", "f0", "f1", "partner_tracklet"]


class FeralImport(NamedTuple):
    """Everything :func:`import_feral` produces."""
    events: pd.DataFrame       # final dyadic events (OUTPUT_COLUMNS)
    candidates: pd.DataFrame   # paired events before overlap resolution
    bouts: pd.DataFrame        # per-animal bouts joined to times
    gt: pd.DataFrame           # manual GT bouts from the browser JSON (may be empty)
    report: Dict


# ---------------------------------------------------------------------------
# Default locations
# ---------------------------------------------------------------------------

def default_dirs(config_path: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
    """``(feral_dir, browser_dir)`` under the cohort config's ``events`` root (read only)."""
    config_path = Path(config_path) if config_path else REPO_ROOT / "config" / "default_paths.json"
    with open(config_path) as fh:
        events_root = Path(json.load(fh)["events"])
    return events_root / "FERAL", events_root / "movie_browser"


# ---------------------------------------------------------------------------
# Stage 0: read and join
# ---------------------------------------------------------------------------

def read_browser_json(browser_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[int, int]]]:
    """Read ``behavior_feral/*.json`` -> ``(predicted bouts, GT bouts, frame windows)``.

    ``browser_dir`` is the movie-browser directory (or its ``behavior_feral``
    subdirectory). Both tables have :data:`_JSON_COLUMNS`.
    """
    browser_dir = Path(browser_dir)
    jdir = browser_dir / "behavior_feral" if (browser_dir / "behavior_feral").is_dir() else browser_dir
    files = sorted(jdir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No FERAL movie JSON files in {jdir}")
    best, gt, windows = [], [], {}
    for fp in files:
        with open(fp) as fh:
            doc = json.load(fh)
        movie = doc.get("movie", fp.stem)
        if doc.get("window"):
            windows[movie] = (int(doc["window"][0]), int(doc["window"][1]))
        best.extend([movie, *row] for row in doc.get("best") or [])
        if doc.get("gt"):
            gt.extend([movie, *row] for row in doc["gt"])
    bouts = pd.DataFrame(best, columns=_JSON_COLUMNS)
    gt_df = pd.DataFrame(gt, columns=_JSON_COLUMNS)
    for df in (bouts, gt_df):
        for col in ("tracklet", "f0", "f1", "partner_tracklet"):
            df[col] = df[col].astype(int)
        df["role"] = df["role"].fillna("").astype(str)
    return bouts, gt_df, windows


def _movie_prefix(movies: Iterable[str]) -> str:
    prefixes = {m.rsplit("_", 2)[0] for m in movies}
    if len(prefixes) != 1:
        raise ValueError(f"Movies do not share one name prefix: {sorted(prefixes)}")
    return prefixes.pop()


def _movie_of(start_local: pd.Series, prefix: str) -> pd.Series:
    """Movie name for a local start time. Movies start on the half hour and
    are named one minute earlier (``..._0959`` starts at 10:00:00)."""
    named = start_local.dt.floor("30min") - pd.Timedelta(minutes=1)
    return prefix + "_" + named.dt.strftime("%Y%m%d_%H%M")


def _check_roles(bouts: pd.DataFrame) -> None:
    sym = bouts["behavior"].isin(SYMMETRIC_BEHAVIORS)
    bad = (sym & (bouts["role"] != "")) | (~sym & ~bouts["role"].isin(["initiator", "victim"]))
    if bad.any():
        sample = bouts.loc[bad, ["movie", "tracklet", "behavior", "role"]].head().to_dict("records")
        raise ValueError(f"{int(bad.sum())} FERAL bouts have an unexpected role "
                         f"(symmetric behaviours need '', directed ones initiator/victim): {sample}")


def load_feral_bouts(feral_csv: Union[str, Path], browser_dir: Union[str, Path]
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[int, int]], Dict]:
    """Join the browser JSON to ``feral_events.csv``.

    Returns ``(bouts, gt, windows, report)``. ``bouts`` has one row per animal
    per bout with frames, role, partner, animal ids and ``ts_start``/``ts_end``
    in UTC Linux ns (the time of the first and last frame).

    The join is by rank within (movie, tracklet, behaviour): same-animal
    same-behaviour bouts never overlap, so start order is unambiguous. Groups
    present only in the JSON are dropped and reported (bouts of a rat after
    its removal from the arena, which the CSV export excluded). Any other
    count mismatch raises, because the ranks would no longer line up.
    """
    feral_csv = Path(feral_csv)
    json_bouts, gt, windows = read_browser_json(browser_dir)
    _check_roles(json_bouts)

    csv = pd.read_csv(feral_csv, encoding="utf-8-sig")
    missing = {"tracklet_id", "rat_id", "behavior", "start_time", "end_time"} - set(csv.columns)
    if missing:
        raise ValueError(f"{feral_csv.name} lacks columns {sorted(missing)}")
    start_local = pd.to_datetime(csv["start_time"])
    end_local = pd.to_datetime(csv["end_time"])
    csv = csv.assign(
        movie=_movie_of(start_local, _movie_prefix(json_bouts["movie"].unique())),
        tracklet=csv["tracklet_id"].astype(int),
        ts_start=start_local.dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC").astype("int64"),
        ts_end=end_local.dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC").astype("int64"),
        start_time_local=csv["start_time"].astype(str),
        end_time_local=csv["end_time"].astype(str),
    )

    # Tracklet -> animal id, which must be fixed for the whole export.
    ids = csv.groupby("tracklet")["rat_id"].unique()
    multi = ids[ids.map(len) > 1]
    if len(multi):
        raise ValueError(f"Tracklets map to more than one rat_id: {multi.to_dict()}")
    normalise = uses_name_normalization(feral_csv)
    animal_of = {int(t): (normalize_object_name(v[0]) if normalise else str(v[0])) for t, v in ids.items()}

    key = ["movie", "tracklet", "behavior"]
    csv = csv.sort_values(key + ["ts_start", "ts_end"], kind="stable")
    csv["rank"] = csv.groupby(key).cumcount()
    json_bouts = json_bouts.sort_values(key + ["f0", "f1"], kind="stable")
    json_bouts["rank"] = json_bouts.groupby(key).cumcount()

    n_csv = csv.groupby(key).size().rename("n_csv")
    n_json = json_bouts.groupby(key).size().rename("n_json")
    counts = pd.concat([n_csv, n_json], axis=1).fillna(0).astype(int)
    json_only = counts[(counts.n_csv == 0) & (counts.n_json > 0)]
    unequal = counts[(counts.n_csv != counts.n_json) & (counts.n_csv > 0)]
    if len(unequal):
        raise ValueError("FERAL CSV and browser JSON disagree on bout counts for "
                         f"{len(unequal)} (movie, tracklet, behavior) groups, e.g. "
                         f"{unequal.head().reset_index().to_dict('records')}")
    if len(json_only):
        logger.warning("Dropping %d JSON-only bouts in %d groups (not in the CSV export)",
                       int(json_only.n_json.sum()), len(json_only))

    bouts = json_bouts.merge(
        csv[key + ["rank", "ts_start", "ts_end", "start_time_local", "end_time_local"]],
        on=key + ["rank"], how="inner", validate="one_to_one")

    # Duration check: CSV seconds vs JSON frames (one frame of slack for the ms truncation).
    frames_s = (bouts["f1"] - bouts["f0"]) / FPS
    csv_s = (bouts["ts_end"] - bouts["ts_start"]) / 1e9
    bouts["duration_mismatch"] = (csv_s - frames_s).abs() > (1.0 / FPS + 1e-3)

    bouts["animal"] = bouts["tracklet"].map(animal_of)
    bouts["partner"] = bouts["partner_tracklet"].map(animal_of)
    unmapped = bouts["partner"].isna()
    if unmapped.any():
        logger.warning("Dropping %d bouts whose partner tracklet has no rat id", int(unmapped.sum()))
        bouts = bouts[~unmapped]
    self_partner = bouts["partner_tracklet"] == bouts["tracklet"]
    if self_partner.any():
        logger.warning("Dropping %d bouts naming themselves as partner", int(self_partner.sum()))
        bouts = bouts[~self_partner]

    bouts = bouts.drop(columns="rank").sort_values(["movie", "f0", "tracklet", "behavior"], kind="stable")
    bouts = bouts.reset_index(drop=True)
    bouts["bout_id"] = np.arange(len(bouts))

    report = {
        "n_csv_rows": int(len(csv)),
        "n_json_bouts": int(len(json_bouts)),
        "n_joined": int(len(bouts)),
        "n_movies": int(bouts["movie"].nunique()),
        "json_only_dropped": [
            {"movie": m, "tracklet": int(t), "behavior": b, "n": int(r.n_json)}
            for (m, t, b), r in json_only.iterrows()],
        "n_duration_mismatch": int(bouts["duration_mismatch"].sum()),
        "n_partner_unmapped": int(unmapped.sum()),
        "n_self_partner": int(self_partner.sum()),
        "tracklet_to_animal": {str(k): v for k, v in sorted(animal_of.items())},
        "name_normalization": bool(normalise),
        "n_gt_bouts": int(len(gt)),
        "gt_movies": sorted(gt["movie"].unique().tolist()),
    }
    return bouts, gt, windows, report


# ---------------------------------------------------------------------------
# Frame clock
# ---------------------------------------------------------------------------

class FrameClock:
    """Per-movie frame -> UTC ns map, anchored on the joined bouts' boundaries.

    Every bout boundary is an exact (frame, time) pair, and boundaries are
    dense (a few seconds apart), so interpolating between them is accurate to
    within a frame even where frames were dropped. Outside the anchors, time
    extrapolates at :data:`FPS`.
    """

    def __init__(self, bouts: pd.DataFrame):
        self._anchors: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
        frames = pd.concat([
            bouts[["movie", "f0", "ts_start"]].set_axis(["movie", "frame", "ns"], axis=1),
            bouts[["movie", "f1", "ts_end"]].set_axis(["movie", "frame", "ns"], axis=1),
        ])
        for movie, g in frames.groupby("movie"):
            g = g.drop_duplicates("frame").sort_values("frame")
            base = int(g["ns"].iloc[0])
            self._anchors[movie] = (g["frame"].to_numpy(float), (g["ns"] - base).to_numpy(float), base)

    def __call__(self, movie: str, frames) -> np.ndarray:
        fx, fy, base = self._anchors[movie]
        f = np.asarray(frames, dtype=float)
        out = np.interp(f, fx, fy)
        step = 1e9 / FPS
        out = np.where(f < fx[0], fy[0] + (f - fx[0]) * step, out)
        out = np.where(f > fx[-1], fy[-1] + (f - fx[-1]) * step, out)
        return (np.rint(out).astype(np.int64) + base)


# ---------------------------------------------------------------------------
# Stage 1: pair
# ---------------------------------------------------------------------------

def _overlap_merge(bouts: pd.DataFrame, on_behavior: bool) -> pd.DataFrame:
    """Bout pairs (a, b) in one movie that name each other and overlap in frames."""
    cols = ["movie", "tracklet", "partner_tracklet", "behavior", "f0", "f1", "bout_id"]
    left_on = ["movie", "partner_tracklet", "tracklet"] + (["behavior"] if on_behavior else [])
    right_on = ["movie", "tracklet", "partner_tracklet"] + (["behavior"] if on_behavior else [])
    m = bouts[cols].merge(bouts[cols], left_on=left_on, right_on=right_on, suffixes=("_a", "_b"))
    return m[(m["f0_a"] <= m["f1_b"]) & (m["f0_b"] <= m["f1_a"])]


def pair_bouts(bouts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Per-animal bouts -> dyadic candidate events, one per reciprocal pair or lone bout.

    Reciprocal bouts are matched one-to-one, highest frame IoU first. A
    matched pair becomes one event over the union of the two intervals
    (``core_f0``/``core_f1`` keep the intersection). Direction:

    * directed behaviours follow FERAL's roles (``role_basis='feral'``). If a
      reciprocal pair both claim the same role, the longer bout's claim wins
      (ties go to the earlier onset) and ``role_basis='conflict'``;
    * symmetric behaviours (EC) get the earlier-onset animal as initiator
      (``role_basis='symmetric'``, ``directed=False``), a naming convention
      only.
    """
    cand = _overlap_merge(bouts, on_behavior=True)
    cand = cand[cand["bout_id_a"] < cand["bout_id_b"]].copy()
    inter = np.minimum(cand["f1_a"], cand["f1_b"]) - np.maximum(cand["f0_a"], cand["f0_b"]) + 1
    union = np.maximum(cand["f1_a"], cand["f1_b"]) - np.minimum(cand["f0_a"], cand["f0_b"]) + 1
    cand["iou"] = inter / union
    cand = cand.sort_values(["iou", "bout_id_a", "bout_id_b"], ascending=[False, True, True])

    used: set = set()
    pairs: List[Tuple[int, int, float]] = []
    for a, b, iou in zip(cand["bout_id_a"].to_numpy(), cand["bout_id_b"].to_numpy(), cand["iou"].to_numpy()):
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pairs.append((a, b, iou))

    # Partner active toward this animal at the same time, in any behaviour.
    any_back = set(_overlap_merge(bouts, on_behavior=False)["bout_id_a"].to_numpy())

    # bout_id is the row position (load_feral_bouts resets it), so arrays index by id.
    if not np.array_equal(bouts["bout_id"].to_numpy(), np.arange(len(bouts))):
        raise ValueError("bouts['bout_id'] must equal the row position")
    col = {c: bouts[c].to_numpy() for c in
           ("movie", "behavior", "role", "tracklet", "partner_tracklet", "animal", "partner", "f0", "f1")}
    A = np.array([p[0] for p in pairs], dtype=int)
    Bi = np.array([p[1] for p in pairs], dtype=int)
    sym = np.isin(col["behavior"][A], list(SYMMETRIC_BEHAVIORS))
    comp = ~sym & (col["role"][A] != col["role"][Bi])
    conflict = ~sym & ~comp
    f0, f1, tr, role = col["f0"], col["f1"], col["tracklet"], col["role"]
    first_a = (f0[A] < f0[Bi]) | ((f0[A] == f0[Bi]) & (tr[A] <= tr[Bi]))
    dur = f1 - f0
    a_longer = (dur[A] > dur[Bi]) | ((dur[A] == dur[Bi]) & (f0[A] <= f0[Bi]))
    longer_claims_init = np.where(a_longer, role[A], role[Bi]) == "initiator"
    init_is_a = np.where(sym, first_a,
                         np.where(comp, role[A] == "initiator", a_longer == longer_claims_init))
    I, V = np.where(init_is_a, A, Bi), np.where(init_is_a, Bi, A)
    paired = pd.DataFrame({
        "movie": col["movie"][A], "type": col["behavior"][A],
        "initiator": col["animal"][I], "victim": col["animal"][V],
        "tracklet_initiator": tr[I], "tracklet_victim": tr[V],
        "f0": np.minimum(f0[A], f0[Bi]), "f1": np.maximum(f1[A], f1[Bi]),
        "core_f0": np.maximum(f0[A], f0[Bi]), "core_f1": np.minimum(f1[A], f1[Bi]),
        "support": "reciprocal",
        "role_basis": np.where(sym, "symmetric", np.where(conflict, "conflict", "feral")),
        "directed": ~sym,
        "iou_pair": np.array([p[2] for p in pairs], dtype=float),
        "n_source_bouts": 2,
    })

    lone = bouts[~bouts["bout_id"].isin(used)]
    lsym = lone["behavior"].isin(SYMMETRIC_BEHAVIORS).to_numpy()
    self_first = lone["role"].isin(["initiator", ""]).to_numpy()
    single = pd.DataFrame({
        "movie": lone["movie"].to_numpy(), "type": lone["behavior"].to_numpy(),
        "initiator": np.where(self_first, lone["animal"], lone["partner"]),
        "victim": np.where(self_first, lone["partner"], lone["animal"]),
        "tracklet_initiator": np.where(self_first, lone["tracklet"], lone["partner_tracklet"]),
        "tracklet_victim": np.where(self_first, lone["partner_tracklet"], lone["tracklet"]),
        "f0": lone["f0"].to_numpy(), "f1": lone["f1"].to_numpy(),
        "core_f0": lone["f0"].to_numpy(), "core_f1": lone["f1"].to_numpy(),
        "support": np.where(lone["bout_id"].isin(any_back), "cross_supported", "one_sided"),
        "role_basis": np.where(lsym, "symmetric", "feral"),
        "directed": ~lsym,
        "iou_pair": np.nan,
        "n_source_bouts": 1,
    })

    out = pd.concat([paired, single], ignore_index=True)
    for c in ("tracklet_initiator", "tracklet_victim", "f0", "f1", "core_f0", "core_f1", "n_source_bouts"):
        out[c] = out[c].astype(int)
    out["directed"] = out["directed"].astype(bool)
    out["dyad_lo"] = out[["tracklet_initiator", "tracklet_victim"]].min(axis=1)
    out["dyad_hi"] = out[["tracklet_initiator", "tracklet_victim"]].max(axis=1)
    out = out.sort_values(["movie", "dyad_lo", "dyad_hi", "f0", "type"], kind="stable").reset_index(drop=True)

    report = {
        "n_candidates": int(len(out)),
        "n_reciprocal_pairs": int(len(pairs)),
        "support_by_type": {t: g["support"].value_counts().to_dict() for t, g in out.groupby("type")},
        "role_basis": out["role_basis"].value_counts().to_dict(),
    }
    return out, report


# ---------------------------------------------------------------------------
# Stage 2: resolve overlaps within a dyad
# ---------------------------------------------------------------------------

def _runs(keys: np.ndarray) -> List[Tuple[int, int, int]]:
    """``[(start, end_inclusive, key), ...]`` for each run of equal values."""
    if len(keys) == 0:
        return []
    change = np.flatnonzero(np.diff(keys)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, len(keys) - 1]
    return [(int(s), int(e), int(keys[s])) for s, e in zip(starts, ends)]


def _segment_cluster(c: pd.DataFrame, prio: Dict[str, int], min_frames: int,
                     bridge_frames: int) -> List[Dict]:
    """One label per frame for one cluster of overlapping candidates in a dyad."""
    c0, c1 = int(c["f0"].min()), int(c["f1"].max())
    n = c1 - c0 + 1
    f0 = c["f0"].to_numpy() - c0
    f1 = c["f1"].to_numpy() - c0
    types = c["type"].to_numpy()
    inits = c["tracklet_initiator"].to_numpy()
    support = c["support"].map(_SUPPORT_RANK).to_numpy()

    order = sorted(range(len(c)), key=lambda k: (prio[types[k]], support[k], -(f1[k] - f0[k]), f0[k], k))
    owner = np.full(n, -1, dtype=int)
    for k in order:
        sl = owner[f0[k]:f1[k] + 1]
        sl[sl == -1] = k

    label_ids: Dict[Tuple[str, int], int] = {}
    cand_label = np.array([label_ids.setdefault((types[k], int(inits[k])), len(label_ids))
                           for k in range(len(c))])
    keys = np.where(owner >= 0, cand_label[np.maximum(owner, 0)], -1)

    # Drop short label runs, bridge short gaps between equal labels, drop again.
    for s, e, k in _runs(keys):
        if k >= 0 and e - s + 1 < min_frames:
            keys[s:e + 1] = -1
    runs = _runs(keys)
    for i in range(1, len(runs) - 1):
        s, e, k = runs[i]
        if k == -1 and e - s + 1 <= bridge_frames and runs[i - 1][2] == runs[i + 1][2] != -1:
            keys[s:e + 1] = runs[i - 1][2]

    rows = []
    for s, e, k in _runs(keys):
        if k < 0 or e - s + 1 < min_frames:
            continue
        own = owner[s:e + 1]
        own = own[(own >= 0) & (cand_label[np.maximum(own, 0)] == k)]
        members, counts = np.unique(own, return_counts=True)
        rep = int(members[np.argmax(counts)])
        overlapping = np.flatnonzero((f0 <= e) & (f1 >= s))
        others = [j for j in overlapping if cand_label[j] != k]
        r = c.iloc[rep]
        rows.append({
            "movie": r["movie"], "type": r["type"],
            "initiator": r["initiator"], "victim": r["victim"],
            "tracklet_initiator": int(r["tracklet_initiator"]), "tracklet_victim": int(r["tracklet_victim"]),
            "frame_start": c0 + s, "frame_end": c0 + e,
            "support": SUPPORT_LEVELS[int(support[members].min())],
            "role_basis": r["role_basis"], "directed": bool(r["directed"]),
            "iou_pair": r["iou_pair"],
            "n_source_bouts": int(c["n_source_bouts"].to_numpy()[members].sum()),
            "co_labels": "+".join(sorted({types[j] for j in others if types[j] != r["type"]})),
            "mutual": any(types[j] == r["type"] for j in others),
            "trimmed": (c0 + s, c0 + e) != (int(r["f0"]), int(r["f1"])),
            "_members": [int(c.index[m]) for m in members],
        })
    return rows


def resolve_overlaps(candidates: pd.DataFrame,
                     priority: Sequence[str] = DEFAULT_PRIORITY,
                     min_frames: int = DEFAULT_MIN_FRAMES,
                     bridge_frames: int = DEFAULT_BRIDGE_FRAMES) -> Tuple[pd.DataFrame, Dict]:
    """Keep one label per frame within each (movie, dyad) and re-segment.

    Candidates that overlap or lie within ``bridge_frames`` of each other form
    a cluster. Inside a cluster each frame goes to the candidate with the best
    ``(priority, support, longer duration)``. The resulting runs of equal
    (behaviour, initiator) are cleaned with FERAL's bout rules. A candidate
    with no overlap passes through unchanged.

    Output frames are ``frame_start``/``frame_end`` (inclusive). Times are
    added by :func:`import_feral`.
    """
    unknown = sorted(set(candidates["type"]) - set(priority))
    if unknown:
        raise ValueError(f"Behaviours missing from priority {list(priority)}: {unknown}")
    prio = {b: i for i, b in enumerate(priority)}

    rows: List[Dict] = []
    for (movie, lo, hi), g in candidates.groupby(["movie", "dyad_lo", "dyad_hi"], sort=False):
        g = g.sort_values("f0", kind="stable")
        f0 = g["f0"].to_numpy()
        f1 = g["f1"].to_numpy()
        reach = np.maximum.accumulate(f1)
        new = np.r_[True, f0[1:] > reach[:-1] + bridge_frames + 1]
        cluster = np.cumsum(new)
        for cid in np.unique(cluster):
            c = g[cluster == cid]
            if len(c) == 1:
                r = c.iloc[0]
                rows.append({
                    "movie": movie, "type": r["type"], "initiator": r["initiator"], "victim": r["victim"],
                    "tracklet_initiator": int(r["tracklet_initiator"]),
                    "tracklet_victim": int(r["tracklet_victim"]),
                    "frame_start": int(r["f0"]), "frame_end": int(r["f1"]),
                    "support": r["support"], "role_basis": r["role_basis"], "directed": bool(r["directed"]),
                    "iou_pair": r["iou_pair"], "n_source_bouts": int(r["n_source_bouts"]),
                    "co_labels": "", "mutual": False, "trimmed": False, "_members": [int(c.index[0])],
                })
            else:
                rows.extend(_segment_cluster(c, prio, min_frames, bridge_frames))

    events = pd.DataFrame(rows)
    used = {m for ms in events["_members"] for m in ms}
    suppressed = candidates.loc[~candidates.index.isin(used)]
    events = events.drop(columns="_members")
    report = {
        "priority": list(priority), "min_frames": min_frames, "bridge_frames": bridge_frames,
        "n_events": int(len(events)),
        "n_candidates_fully_suppressed": int(len(suppressed)),
        "suppressed_by_type": suppressed["type"].value_counts().to_dict(),
        "n_trimmed": int(events["trimmed"].sum()),
        "n_mutual": int(events["mutual"].sum()),
        "events_by_type": events["type"].value_counts().to_dict(),
        "co_labels": events.loc[events["co_labels"] != "", "co_labels"].value_counts().head(20).to_dict(),
    }
    return events, report


def _concurrent_partners(events: pd.DataFrame) -> np.ndarray:
    """For each event, the number of *other* dyads sharing one of its animals
    that have an event overlapping it in time (frames, same movie)."""
    out = np.zeros(len(events), dtype=int)
    ev = events.reset_index(drop=True)
    sides = pd.concat([
        pd.DataFrame({"i": ev.index, "movie": ev["movie"], "animal": ev["tracklet_initiator"],
                      "other": ev["tracklet_victim"], "f0": ev["frame_start"], "f1": ev["frame_end"]}),
        pd.DataFrame({"i": ev.index, "movie": ev["movie"], "animal": ev["tracklet_victim"],
                      "other": ev["tracklet_initiator"], "f0": ev["frame_start"], "f1": ev["frame_end"]}),
    ])
    for _, s in sides.groupby("movie"):
        m = s.merge(s, on="animal", suffixes=("", "_o"))
        m = m[(m["other"] != m["other_o"]) & (m["f0"] <= m["f1_o"]) & (m["f0_o"] <= m["f1"])]
        if len(m):
            n = m.drop_duplicates(["i", "animal", "other_o"]).groupby("i").size()
            out[n.index.to_numpy()] = n.to_numpy()
    return out


# ---------------------------------------------------------------------------
# Orchestration and output
# ---------------------------------------------------------------------------

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_feral(feral_csv: Union[str, Path], browser_dir: Union[str, Path],
                 priority: Sequence[str] = DEFAULT_PRIORITY,
                 min_frames: int = DEFAULT_MIN_FRAMES,
                 bridge_frames: int = DEFAULT_BRIDGE_FRAMES) -> FeralImport:
    """Run all stages and return the final dyadic events plus intermediates."""
    feral_csv = Path(feral_csv)
    bouts, gt, windows, r0 = load_feral_bouts(feral_csv, browser_dir)
    candidates, r1 = pair_bouts(bouts)
    events, r2 = resolve_overlaps(candidates, priority, min_frames, bridge_frames)

    clock = FrameClock(bouts)
    ts_start = np.empty(len(events), dtype=np.int64)
    ts_end = np.empty(len(events), dtype=np.int64)
    for movie, idx in events.groupby("movie").indices.items():
        ts_start[idx] = clock(movie, events["frame_start"].to_numpy()[idx])
        ts_end[idx] = clock(movie, events["frame_end"].to_numpy()[idx])
    events["ts_start"] = ts_start
    events["ts_end"] = ts_end

    to_local = lambda ns: (pd.to_datetime(ns, utc=True).dt.tz_convert(LOCAL_TZ)  # noqa: E731
                           .dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3])
    events["start_time_local"] = to_local(events["ts_start"])
    events["end_time_local"] = to_local(events["ts_end"])
    events["duration"] = (events["ts_end"] - events["ts_start"]) / 1e9
    events["session"] = events["movie"]
    events["date"] = events["movie"].str.rsplit("_", n=2).str[1]
    events["winner"] = np.nan
    events["loser"] = np.nan
    events["n_concurrent_partners"] = _concurrent_partners(events)
    win = events["movie"].map(windows)
    events["at_movie_edge"] = [
        bool(w is not None and isinstance(w, tuple) and (fs <= w[0] or fe >= w[1]))
        for w, fs, fe in zip(win, events["frame_start"], events["frame_end"])]
    events["source"] = "feral"
    events = events.sort_values(["ts_start", "ts_end", "tracklet_initiator", "type"], kind="stable")
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    events = events[OUTPUT_COLUMNS]

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"feral_csv": str(feral_csv), "feral_csv_md5": _md5(feral_csv),
                   "browser_dir": str(browser_dir)},
        "stage0_join": r0,
        "stage1_pairing": r1,
        "stage2_overlaps": r2,
        "output": {
            "n_events": int(len(events)),
            "by_type_and_support": {t: g["support"].value_counts().to_dict()
                                    for t, g in events.groupby("type")},
            "frac_with_concurrent_partner": float((events["n_concurrent_partners"] > 0).mean()),
            "n_at_movie_edge": int(events["at_movie_edge"].sum()),
            "dates": sorted(events["date"].unique().tolist()),
        },
    }
    return FeralImport(events, candidates, bouts, gt, report)


def write_feral_events(result: FeralImport, out_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Write the events CSV (no BOM) and the JSON report into ``out_dir``.

    ``out_dir`` must not be a dated events directory: the event resolver globs
    ``<date>_behavior_event_df*.csv`` there, and the file is named so it can
    never match that pattern anyway (HZ-DATA-007).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_FILENAME
    report_path = out_dir / REPORT_FILENAME
    result.events.to_csv(csv_path, index=False)
    report = dict(result.report)
    report["output"] = dict(report["output"], file=str(csv_path), md5=_md5(csv_path))
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    return csv_path, report_path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------------------
# Loading into the pipeline
# ---------------------------------------------------------------------------

def load_feral_events(path: Union[str, Path], date: Optional[str] = None,
                      min_support: str = "one_sided",
                      behaviors: Optional[Union[str, Iterable[str]]] = None,
                      session_id: Optional[str] = None):
    """Load the importer's output as :class:`~video.behavioral_events.BehavioralEventsData`.

    Parameters
    ----------
    path : the ``feral_dyadic_events.csv`` written by :func:`write_feral_events`.
    date : ``'YYYYMMDD'`` (or any session id starting with it) to keep one day.
    min_support : weakest support level kept: ``'one_sided'`` (everything),
        ``'cross_supported'`` or ``'reciprocal'``.
    behaviors : one abbreviation or several to keep; ``None`` keeps all.
    session_id : stored on the returned object (defaults to ``date`` or ``'feral'``).

    The events are **not** synchronised. Call
    ``events.synchronize_with_ephys(sync)`` as for the manual file: both are
    on the video's Linux-ns clock.
    """
    from video.behavioral_events import load_behavioral_events

    if min_support not in _SUPPORT_RANK:
        raise ValueError(f"min_support must be one of {SUPPORT_LEVELS}, got {min_support!r}")
    data = load_behavioral_events(Path(path), session_id=session_id or (date[:8] if date else "feral"))
    df = data.events_data
    keep = df["support"].map(_SUPPORT_RANK) <= _SUPPORT_RANK[min_support]
    if date:
        keep &= df["session"].astype(str).str.contains(f"_{date[:8]}_", regex=False)
    if behaviors is not None:
        keep &= df["type"].isin([behaviors] if isinstance(behaviors, str) else list(behaviors))
    data.events_data = df[keep].reset_index(drop=True)
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def gt_dyadic_events(gt: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (fully reciprocal) per-animal GT bouts to one row per dyadic event."""
    sym = gt["behavior"].isin(SYMMETRIC_BEHAVIORS)
    keep = (sym & (gt["tracklet"] < gt["partner_tracklet"])) | (~sym & (gt["role"] == "initiator"))
    g = gt[keep]
    return pd.DataFrame({
        "movie": g["movie"].to_numpy(), "type": g["behavior"].to_numpy(),
        "tracklet_initiator": g["tracklet"].to_numpy(), "tracklet_victim": g["partner_tracklet"].to_numpy(),
        "frame_start": g["f0"].to_numpy(), "frame_end": g["f1"].to_numpy(),
    })


def _with_movie(ev: pd.DataFrame) -> pd.DataFrame:
    """Output events name the movie in ``session``; candidates and GT in ``movie``."""
    return ev if "movie" in ev.columns else ev.assign(movie=ev["session"])


def _frame_table(ev: pd.DataFrame) -> pd.DataFrame:
    """Explode dyadic events to one row per (movie, dyad, frame, type)."""
    if ev.empty:
        return pd.DataFrame(columns=["movie", "lo", "hi", "frame", "type", "init"])
    n = (ev["frame_end"] - ev["frame_start"] + 1).to_numpy()
    rep = lambda col: np.repeat(ev[col].to_numpy(), n)  # noqa: E731
    starts = np.repeat(ev["frame_start"].to_numpy(), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    ti, tv = rep("tracklet_initiator"), rep("tracklet_victim")
    t = pd.DataFrame({"movie": rep("movie"), "lo": np.minimum(ti, tv), "hi": np.maximum(ti, tv),
                      "frame": starts + offs, "type": rep("type"), "init": ti})
    return t.drop_duplicates(["movie", "lo", "hi", "frame", "type"])


def _helmet_ok(t: pd.DataFrame) -> pd.Series:
    helmet = t["lo"].isin(HELMET_TRACKLETS) | t["hi"].isin(HELMET_TRACKLETS)
    return ~t["type"].isin(_HELMET_ONLY_BEHAVIORS) | helmet


def frame_agreement(gt_events: pd.DataFrame, pred: pd.DataFrame,
                    movies: Optional[Iterable[str]] = None) -> Dict[str, Dict]:
    """Frame-level precision/recall per behaviour, per (movie, unordered dyad).

    EC and SN are scored only on dyads with a helmet animal (the only ones
    the GT annotated for them). For directed behaviours,
    ``direction_agreement`` is the share of true-positive frames where the
    initiator also agrees.
    """
    pred = _with_movie(pred)
    movies = set(movies) if movies is not None else set(gt_events["movie"])
    g = _frame_table(gt_events[gt_events["movie"].isin(movies)])
    p = _frame_table(pred[pred["movie"].isin(movies)])
    g, p = g[_helmet_ok(g)], p[_helmet_ok(p)]
    k = ["movie", "lo", "hi", "frame", "type"]
    tp = g.merge(p, on=k, suffixes=("_g", "_p"))
    out = {}
    for t in sorted(set(g["type"]) | set(p["type"])):
        n_g, n_p = int((g["type"] == t).sum()), int((p["type"] == t).sum())
        tpt = tp[tp["type"] == t]
        prec = len(tpt) / n_p if n_p else float("nan")
        rec = len(tpt) / n_g if n_g else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if len(tpt) else 0.0
        row = {"gt_frames": n_g, "pred_frames": n_p, "tp_frames": int(len(tpt)),
               "precision": prec, "recall": rec, "f1": f1}
        if t not in SYMMETRIC_BEHAVIORS and len(tpt):
            row["direction_agreement"] = float((tpt["init_g"] == tpt["init_p"]).mean())
        out[t] = row
    return out


def _as_ns_events(ev: pd.DataFrame, movie_index: Dict[str, int]) -> pd.DataFrame:
    """Frames -> fake ns with movies spaced apart, for ``compare_annotation_sources``."""
    base = ev["movie"].map(movie_index).to_numpy() * 10_000_000
    to_ns = lambda f: ((base + f.to_numpy()) * (1e9 / FPS)).astype(np.int64)  # noqa: E731
    return pd.DataFrame({"type": ev["type"].to_numpy(),
                         "initiator": ev["tracklet_initiator"].astype(str).to_numpy(),
                         "victim": ev["tracklet_victim"].astype(str).to_numpy(),
                         "ts_start": to_ns(ev["frame_start"]), "ts_end": to_ns(ev["frame_end"])})


def event_agreement(gt_events: pd.DataFrame, pred: pd.DataFrame, movies: Iterable[str],
                    min_iou: float = 0.3) -> Dict[str, Dict]:
    """Event-level P/R/F1 per behaviour (unordered pair + IoU), via ``compare_annotation_sources``."""
    from analysis.event_decoding.compare_annotation_sources import agreement

    pred = _with_movie(pred)
    movies = sorted(set(movies))
    idx = {m: i for i, m in enumerate(movies)}
    g = gt_events[gt_events["movie"].isin(movies)]
    p = pred[pred["movie"].isin(movies)]
    out = {}
    for t in sorted(set(g["type"]) | set(p["type"])):
        gt_t, p_t = g[g["type"] == t], p[p["type"] == t]
        if t in _HELMET_ONLY_BEHAVIORS:
            hel = lambda d: d[d["tracklet_initiator"].isin(HELMET_TRACKLETS)  # noqa: E731
                              | d["tracklet_victim"].isin(HELMET_TRACKLETS)]
            gt_t, p_t = hel(gt_t), hel(p_t)
        out[t] = agreement(_as_ns_events(gt_t, idx), _as_ns_events(p_t, idx), min_iou=min_iou, event_type=t)
    return out


def _candidates_as_events(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.rename(columns={"f0": "frame_start", "f1": "frame_end"})


def validate_against_gt(result: FeralImport, min_iou: float = 0.3) -> Dict:
    """Raw paired candidates vs cleaned events, scored against the browser GT."""
    if result.gt.empty:
        return {"note": "no GT in the browser JSON"}
    gt_ev = gt_dyadic_events(result.gt)
    movies = sorted(gt_ev["movie"].unique())
    splits = {"validation": sorted(GT_VALIDATION_MOVIES & set(movies)),
              "training": sorted(set(movies) - GT_VALIDATION_MOVIES), "all": movies}
    raw = _candidates_as_events(result.candidates)
    out = {"n_gt_events": int(len(gt_ev)), "movies": splits}
    for name, ms in splits.items():
        if not ms:
            continue
        out[name] = {
            "frame_raw_candidates": frame_agreement(gt_ev, raw, ms),
            "frame_cleaned": frame_agreement(gt_ev, result.events, ms),
            "event_raw_candidates": event_agreement(gt_ev, raw, ms, min_iou),
            "event_cleaned": event_agreement(gt_ev, result.events, ms, min_iou),
            "event_cleaned_reciprocal": event_agreement(
                gt_ev, result.events[result.events["support"] == "reciprocal"], ms, min_iou),
        }
    return out


def validate_against_manual(result: FeralImport, date: str = "20251216",
                            config_path: Optional[Union[str, Path]] = None,
                            min_iou: float = 0.3) -> Dict:
    """Event agreement with the canonical manual file for ``date`` (HZ-DATA-007 resolver).

    FERAL events are restricted to the span the manual file covers, so the
    manual file's unscored hours do not count as false positives.
    """
    from analysis.event_decoding.compare_annotation_sources import agreement, load_events
    from ingestion.data_paths import get_event_files_by_date

    files = get_event_files_by_date(date, config_path=str(config_path) if config_path else None)
    if len(files) != 1:
        return {"note": f"no canonical event file for {date}"}
    manual = load_events(files[0])
    lo, hi = manual["ts_start"].min(), manual["ts_end"].max()
    feral = result.events[(result.events["date"] == date)
                          & (result.events["ts_start"] >= lo) & (result.events["ts_end"] <= hi)]
    out = {"manual_file": str(files[0]), "manual_rows": int(len(manual)),
           "feral_events_in_span": int(len(feral)), "min_iou": min_iou, "by_type": {}}
    for t in DEFAULT_PRIORITY:
        out["by_type"][t] = {
            "all": agreement(manual, feral, min_iou=min_iou, event_type=t),
            "reciprocal": agreement(manual, feral[feral["support"] == "reciprocal"],
                                    min_iou=min_iou, event_type=t),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import FERAL predictions as paired, non-overlapping dyadic events.")
    p.add_argument("--config", default=None, help="cohort config (default config/default_paths.json)")
    p.add_argument("--feral_csv", default=None, help="feral_events.csv (default <events>/FERAL/feral_events.csv)")
    p.add_argument("--browser_dir", default=None, help="movie browser dir (default <events>/movie_browser)")
    p.add_argument("--out_dir", default=None, help="output dir (default results/feral_import)")
    p.add_argument("--priority", nargs="+", default=list(DEFAULT_PRIORITY))
    p.add_argument("--min_frames", type=int, default=DEFAULT_MIN_FRAMES)
    p.add_argument("--bridge_frames", type=int, default=DEFAULT_BRIDGE_FRAMES)
    p.add_argument("--validate", action="store_true",
                   help="score against the browser GT and the canonical 20251216 manual file")
    p.add_argument("--min_iou", type=float, default=0.3)
    return p


def _print_summary(report: Dict) -> None:
    r0, r1, r2, ro = (report["stage0_join"], report["stage1_pairing"],
                      report["stage2_overlaps"], report["output"])
    print(f"bouts       {r0['n_json_bouts']} JSON / {r0['n_csv_rows']} CSV -> {r0['n_joined']} joined "
          f"({sum(d['n'] for d in r0['json_only_dropped'])} JSON-only dropped, "
          f"{r0['n_duration_mismatch']} duration mismatches)")
    print(f"pairing     {r1['n_reciprocal_pairs']} reciprocal pairs -> {r1['n_candidates']} candidates")
    print(f"overlaps    {r2['n_events']} events ({r2['n_candidates_fully_suppressed']} candidates "
          f"fully suppressed, {r2['n_trimmed']} trimmed)")
    print(f"{'type':<4} {'events':>7} {'recip':>7} {'cross':>7} {'one':>7}")
    for t, s in sorted(ro["by_type_and_support"].items()):
        tot = sum(s.values())
        print(f"{t:<4} {tot:>7} {s.get('reciprocal', 0):>7} {s.get('cross_supported', 0):>7} "
              f"{s.get('one_sided', 0):>7}")


def _print_gt(v: Dict) -> None:
    for split in ("validation", "training"):
        if split not in v:
            continue
        print(f"\nGT frame agreement ({split}: {', '.join(m[-4:] for m in v['movies'][split])})")
        print(f"{'type':<4} {'P raw':>6} {'R raw':>6} {'F1 raw':>6} | {'P cln':>6} {'R cln':>6} {'F1 cln':>6}")
        raw, cln = v[split]["frame_raw_candidates"], v[split]["frame_cleaned"]
        for t in DEFAULT_PRIORITY:
            a, b = raw.get(t), cln.get(t)
            if a and b:
                print(f"{t:<4} {a['precision']:>6.2f} {a['recall']:>6.2f} {a['f1']:>6.2f} | "
                      f"{b['precision']:>6.2f} {b['recall']:>6.2f} {b['f1']:>6.2f}")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    feral_dir, browser_default = default_dirs(args.config)
    feral_csv = Path(args.feral_csv) if args.feral_csv else feral_dir / "feral_events.csv"
    browser_dir = Path(args.browser_dir) if args.browser_dir else browser_default
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from analysis._paths import results_dir
        out_dir = results_dir("feral_import")

    result = import_feral(feral_csv, browser_dir, args.priority, args.min_frames, args.bridge_frames)
    if args.validate:
        result.report["validation_gt"] = validate_against_gt(result, args.min_iou)
        result.report["validation_manual_20251216"] = validate_against_manual(
            result, "20251216", args.config, args.min_iou)
    csv_path, report_path = write_feral_events(result, out_dir)
    _print_summary(result.report)
    if args.validate:
        _print_gt(result.report["validation_gt"])
    print(f"\nwrote {csv_path}\n      {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
