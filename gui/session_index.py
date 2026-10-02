"""Session index for the Panel session browser: one row per recording block.

Everything the browser's table and detail panel show comes from here, so the
browser itself never touches Kilosort, DIO or a tracking CSV on a click.

Sources, cheapest first:

- ``discovery/capability_manifest.json``: per-animal ``n_quality_cells``,
  ``n_clusters``, ``ephys_window``, ``load_error`` and the per-recording sync
  ``slope``/``intercept``. The stored sync is what lets us place a video chunk
  or an event on a block **without reading any DIO**. Its *tracking* section is
  stale (it predates multi-root APT resolution), so tracking is not read from it.
- A filesystem scan of the share, done once per index build:
  ``<video>/social_videos/RatCity_<date>_<HHMM>[_40Hz].mp4`` + ``_ts.npy``
  (30-minute chunks), tracking files via :func:`get_tracking_files_by_date`, and
  the canonical event CSV via :func:`get_event_files_by_date`, restricted to a
  date-named directory (HZ-DATA-007).

Definitions the table uses:

- a video chunk is **tracked** when an APT ``TQT_named.csv`` exists for the
  same ``<date>_<HHMM>``, or a mask-metrics file's ``_ts.npy`` span overlaps it;
- a chunk is **annotated** when at least one scored event's ``ts_start`` falls
  inside its span;
- a chunk belongs to the block whose ephys window (mapped to wall-clock through
  that block's sync) it overlaps most. A chunk that overlaps no block is kept
  on the day timeline but counted in no row.

The built index is pickled to ``.gui_cache/session_browser/``. It is rebuilt
when the manifest or the cohort config changes, or on ``refresh=True``. The
per-day tracking coverage grid (slow: reads every APT CSV of the day) is cached
separately by :func:`get_day_tracking_grid`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ingestion.data_paths import (
    _load_config,
    get_animals_and_sessions,
    get_event_files_by_date,
    get_tracking_files_by_date,
)

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
TIMELINE_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".gui_cache" / "session_browser"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "discovery" / "capability_manifest.json"

#: Cohort name (as the manifest spells it) -> config file.
COHORT_CONFIGS = {
    "cohort7": "config/default_paths.json",
    "cohort5": "config/cohort5_paths.json",
}

#: Video chunks are nominally 30 min; used only when a chunk's _ts.npy is unusable.
NOMINAL_CHUNK_SECONDS = 30 * 60
#: Below this an .mp4 is a stub (262-byte files exist on the share).
MIN_VIDEO_BYTES = 1_000_000
#: Fewer quality cells than this and the decoders are not viable (requirements layer).
MIN_GOOD_CELLS = 10

_VIDEO_RE = re.compile(r"^RatCity_(20\d{6})_(\d{4})(_40Hz)?\.mp4$", re.IGNORECASE)
_DIR_CHUNK_RE = re.compile(r"(20\d{6})_(\d{4})(?!\d)")
_RECORDING_RE = re.compile(r"^(20\d{6})_(\d{6})$")

#: analysis_readiness key -> short tag shown in the table.
READY_TAGS = {
    "ephys.decode_event_outcome": "outcome",
    "ephys.decode_opponent_identity": "opponent",
    "ephys.inter_brain_dynamics": "inter-brain",
    "ephys.decode_location": "location",
    "ephys.decode_partner_distance": "partner-dist",
    "ephys.social_spatial_fields": "social-fields",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@dataclass
class LinearSync:
    """Stand-in for :class:`DataSyncManager` built from the manifest's stored fit.

    ``behavior_epoch_seconds = slope * ephys_seconds + intercept`` — the same
    convention as :meth:`DataSyncManager.convert_ephys_to_behavior`.
    """
    slope: float
    intercept: float

    def convert_ephys_to_behavior(self, ephys_timestamps):
        return self.slope * np.asarray(ephys_timestamps, dtype=np.float64) + self.intercept

    def convert_behavior_to_ephys(self, behavior_timestamps):
        return (np.asarray(behavior_timestamps, dtype=np.float64) - self.intercept) / self.slope


def local_midnight_epoch(date: str) -> float:
    """Epoch seconds of local midnight starting ``date`` (YYYYMMDD)."""
    return datetime.strptime(date, "%Y%m%d").timestamp()


def _stamp_epoch(date: str, hhmm_or_hhmmss: str) -> float:
    """Local wall-clock stamp from a file/dir name -> epoch seconds."""
    fmt = "%Y%m%d%H%M%S" if len(hhmm_or_hhmmss) == 6 else "%Y%m%d%H%M"
    return datetime.strptime(date + hhmm_or_hhmmss, fmt).timestamp()


def _file_sig(path: Path) -> Tuple[str, int, float]:
    try:
        st = path.stat()
        return (str(path), int(st.st_size), float(st.st_mtime))
    except OSError:
        return (str(path), -1, 0.0)


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def npy_first_last(path: Path) -> Tuple[Optional[float], Optional[float], int]:
    """First and last element of a 1-D ``.npy`` without reading the whole file.

    The video ``_ts.npy`` files sit on SMB; reading two values instead of
    ~576 KB each is the difference between seconds and minutes for a cohort.
    Returns ``(None, None, 0)`` for an empty or unreadable file.
    """
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            n = int(np.prod(shape)) if shape else 0
            if n == 0 or dtype.hasobject:
                return None, None, n
            offset = fh.tell()
            first = np.frombuffer(fh.read(dtype.itemsize), dtype=dtype)[0]
            fh.seek(offset + (n - 1) * dtype.itemsize)
            last = np.frombuffer(fh.read(dtype.itemsize), dtype=dtype)[0]
            return float(first), float(last), n
    except Exception as exc:
        logger.debug("unreadable npy %s: %s", path, exc)
        return None, None, 0


def _to_epoch_seconds(value: float) -> float:
    """Linux ns (the share's convention) -> s; already-seconds passes through."""
    return value / 1e9 if value > 1e12 else value


# ---------------------------------------------------------------------------
# Video chunks
# ---------------------------------------------------------------------------

@dataclass
class VideoChunk:
    key: str                # '<date>_<HHMM>'
    date: str
    path: str
    size: int
    t0: float               # epoch seconds
    t1: float
    n_frames: int
    broken: bool
    reason: str = ""
    tracked: bool = False
    tracked_by: str = ""    # 'APT' | 'manual' | ''
    n_events: int = 0
    block: Optional[str] = None


def scan_video_chunks(config: Mapping[str, Any],
                      span_cache: Optional[Dict[str, Any]] = None) -> List[VideoChunk]:
    """Every raw-video chunk under ``<video>/social_videos`` (one directory listing).

    ``span_cache`` maps ``ts_path -> [size, mtime, t0, t1, n]`` and is updated in
    place, so a rescan only opens ``_ts.npy`` files that are new or changed.
    """
    span_cache = span_cache if span_cache is not None else {}
    video_dir = Path(config.get("video", "")) / "social_videos"
    if not video_dir.exists():
        logger.warning("no social_videos directory at %s", video_dir)
        return []

    entries: Dict[str, os.DirEntry] = {}
    with os.scandir(video_dir) as it:
        for entry in it:
            if entry.is_file():
                entries[entry.name] = entry

    by_key: Dict[str, VideoChunk] = {}
    for name, entry in sorted(entries.items()):
        m = _VIDEO_RE.match(name)
        if not m:
            continue
        date, hhmm = m.group(1), m.group(2)
        key = f"{date}_{hhmm}"
        size = entry.stat().st_size
        ts_name = name[:-4] + "_ts.npy"
        ts_entry = entries.get(ts_name)

        t0 = t1 = None
        n = 0
        if ts_entry is not None:
            ts_stat = ts_entry.stat()
            cached = span_cache.get(ts_entry.path)
            if cached and cached[0] == ts_stat.st_size and cached[1] == ts_stat.st_mtime:
                t0, t1, n = cached[2], cached[3], cached[4]
            else:
                t0, t1, n = npy_first_last(Path(ts_entry.path))
                span_cache[ts_entry.path] = [ts_stat.st_size, ts_stat.st_mtime, t0, t1, n]

        reasons = []
        if size < MIN_VIDEO_BYTES:
            reasons.append(f"stub mp4 ({size} B)")
        if ts_entry is None:
            reasons.append("no _ts.npy")
        elif t0 is None or n < 2:
            reasons.append("empty _ts.npy")
        if t0 is None or t1 is None or n < 2:
            t0 = _stamp_epoch(date, hhmm)
            t1 = t0 + NOMINAL_CHUNK_SECONDS
        else:
            t0, t1 = _to_epoch_seconds(t0), _to_epoch_seconds(t1)

        chunk = VideoChunk(key=key, date=date, path=entry.path, size=size, t0=t0, t1=t1,
                           n_frames=n, broken=bool(reasons), reason="; ".join(reasons))
        # Cohort 5 keeps both 'RatCity_<d>_<t>.mp4' and '..._40Hz.mp4' for one
        # chunk; keep the healthier one so a chunk is counted once.
        prev = by_key.get(key)
        if prev is None or (prev.broken and not chunk.broken) or \
                (prev.broken == chunk.broken and chunk.n_frames > prev.n_frames):
            by_key[key] = chunk
    return sorted(by_key.values(), key=lambda c: c.key)


# ---------------------------------------------------------------------------
# Tracking + events per date
# ---------------------------------------------------------------------------

def tracking_spans(files: Sequence[Path],
                   span_cache: Dict[str, Any]) -> Tuple[set, List[Tuple[float, float]]]:
    """``(apt_chunk_keys, manual_epoch_spans)`` for one date's tracking files.

    APT/TQT exports are matched to a video chunk by their directory's
    ``<date>_<HHMM>`` (reading their timestamps would mean reading a ~100 MB
    CSV). Mask-metrics files are matched by their sibling ``_ts.npy`` span.
    """
    apt_keys = set()
    manual = []
    for path in files:
        path = Path(path)
        if path.name.lower() == "tqt_named.csv":
            for part in (path.parent.name, path.parent.parent.name):
                m = _DIR_CHUNK_RE.search(part)
                if m:
                    apt_keys.add(f"{m.group(1)}_{m.group(2)}")
                    break
            continue
        if "_mask_metrics" in path.stem:
            ts_path = path.parent / f"{path.stem.replace('_mask_metrics', '_ts')}.npy"
            sig = _file_sig(ts_path)
            cached = span_cache.get(str(ts_path))
            if cached and cached[0] == sig[1] and cached[1] == sig[2]:
                t0, t1, n = cached[2], cached[3], cached[4]
            else:
                t0, t1, n = npy_first_last(ts_path)
                span_cache[str(ts_path)] = [sig[1], sig[2], t0, t1, n]
            if t0 is not None and t1 is not None:
                manual.append((_to_epoch_seconds(t0), _to_epoch_seconds(t1)))
    return apt_keys, manual


def _event_dates(config: Mapping[str, Any]) -> set:
    """Dates that have a date-named directory under the events root (one listing)."""
    root = Path(config.get("events", ""))
    try:
        return {d.name[:8] for d in root.iterdir() if d.is_dir() and d.name[:8].isdigit()}
    except OSError:
        return set()


def canonical_event_files(date: str, config: Mapping[str, Any]) -> List[Path]:
    """Event CSVs for ``date``, only from a date-named directory (HZ-DATA-007)."""
    try:
        files = get_event_files_by_date(date, _config=dict(config))
    except Exception as exc:
        logger.debug("no event files for %s: %s", date, exc)
        return []
    return [Path(f) for f in files if Path(f).parent.name[:8] == date]


def read_event_times(files: Sequence[Path]) -> pd.DataFrame:
    """``(t_epoch, type)`` for every scored event; ts_start is Linux ns on disk."""
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=lambda c: c in ("ts_start", "type"))
        except Exception as exc:
            logger.warning("could not read events %s: %s", fp, exc)
            continue
        if "ts_start" not in df.columns:
            continue
        t = pd.to_numeric(df["ts_start"], errors="coerce").to_numpy(dtype=np.float64)
        t = np.where(t > 1e12, t / 1e9, t)
        frames.append(pd.DataFrame({
            "t_epoch": t,
            "type": df["type"].astype(str) if "type" in df.columns else "?",
        }))
    if not frames:
        return pd.DataFrame({"t_epoch": np.array([], dtype=float), "type": []})
    out = pd.concat(frames, ignore_index=True).dropna(subset=["t_epoch"])
    return out.sort_values("t_epoch").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Manifest → per-block facts
# ---------------------------------------------------------------------------

def _manifest_sessions(manifest_path: Optional[Path]) -> Tuple[Dict[str, Any], Optional[str]]:
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        return {}, None
    try:
        from discovery.capability_manifest import load_manifest
        manifest = load_manifest(path)
    except Exception as exc:
        logger.warning("capability manifest unreadable (%s); index will lack ephys facts", exc)
        return {}, None
    return manifest.get("sessions", {}), manifest.get("generated_at")


def block_sync(record: Optional[Mapping[str, Any]]) -> Optional[LinearSync]:
    """The first animal's stored sync fit for this recording, if any succeeded."""
    if not record:
        return None
    for _, info in sorted((record.get("ephys", {}).get("per_animal") or {}).items()):
        s = info.get("sync") or {}
        if s.get("ok") and s.get("slope") and s.get("intercept") is not None:
            return LinearSync(float(s["slope"]), float(s["intercept"]))
    return None


def per_animal_facts(record: Optional[Mapping[str, Any]],
                     animals: Sequence[str]) -> List[Dict[str, Any]]:
    """One dict per animal of the block: neurons, window, loadability."""
    per = ((record or {}).get("ephys") or {}).get("per_animal") or {}
    out = []
    for animal in sorted(set(animals) | set(per)):
        info = per.get(animal) or {}
        window = info.get("ephys_window")
        load_error = info.get("load_error")
        if record is None:
            status = "not in manifest"
        elif load_error:
            status = "not loadable"
        elif info.get("n_quality_cells") is None:
            status = "unknown"
        else:
            status = "ok"
        out.append({
            "animal": animal,
            "n_clusters": info.get("n_clusters"),
            "n_quality_cells": info.get("n_quality_cells"),
            "ephys_window": list(window) if window else None,
            "window_hours": (window[1] - window[0]) / 3600.0 if window else None,
            "load_error": load_error,
            "window_suspect": bool(info.get("duration_disagrees_with_window")),
            "sync_ok": bool((info.get("sync") or {}).get("ok")),
            "status": status,
            # Loadable unless the manifest positively says it isn't. An animal
            # the manifest never probed is offered, since the load will tell.
            "loadable": status in ("ok", "unknown", "not in manifest"),
        })
    return out


def block_epoch_window(recording_id: str, record: Optional[Mapping[str, Any]],
                       next_stamp: Optional[float]) -> Tuple[float, float, str]:
    """Wall-clock ``[t0, t1]`` (epoch s) of a block, and how it was derived.

    Prefers each animal's measured ephys window through the stored sync. A block
    the manifest lacks falls back to its directory stamp, running to the next
    block's stamp (or +6 h): coarse, and flagged as such.
    """
    sync = block_sync(record)
    windows = [a["ephys_window"] for a in per_animal_facts(record, []) if a["ephys_window"]]
    if sync is not None and windows:
        lo = min(w[0] for w in windows)
        hi = max(w[1] for w in windows)
        t = sync.convert_ephys_to_behavior([lo, hi])
        return float(t[0]), float(t[1]), "sync"
    m = _RECORDING_RE.match(recording_id)
    if m:
        t0 = _stamp_epoch(m.group(1), m.group(2))
        return t0, (next_stamp if next_stamp else t0 + 6 * 3600), "stamp"
    return float("nan"), float("nan"), "none"


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def _index_path(cohort: str) -> Path:
    return CACHE_DIR / f"index_{cohort}.pkl"


def _config_file(cohort: str) -> Path:
    return REPO_ROOT / COHORT_CONFIGS[cohort]


def _expected_meta(cohort: str, manifest_path: Optional[Path]) -> Dict[str, Any]:
    mpath = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    return {
        "index_version": INDEX_VERSION,
        "cohort": cohort,
        "config_sha256": _sha256(_config_file(cohort)),
        "manifest_mtime": mpath.stat().st_mtime if mpath.exists() else None,
    }


def load_session_index(cohort: str, manifest_path: Optional[Path] = None,
                       refresh: bool = False, progress=None) -> Dict[str, Any]:
    """The cached index for ``cohort``, rebuilding only when it is invalid.

    Returns a dict with ``table`` (DataFrame, one row per recording),
    ``details`` (recording_id -> detail dict), ``days`` (date -> day dict with
    chunks, events and block windows) and ``meta``.
    """
    path = _index_path(cohort)
    expected = _expected_meta(cohort, manifest_path)
    previous = None
    if path.exists():
        try:
            with open(path, "rb") as fh:
                previous = pickle.load(fh)
        except Exception as exc:
            logger.warning("discarding unreadable session index %s: %s", path, exc)
    if previous is not None and not refresh:
        meta = previous.get("meta", {})
        if all(meta.get(k) == v for k, v in expected.items()):
            return previous
        logger.info("session index for %s is stale; rebuilding", cohort)

    span_cache = (previous or {}).get("span_cache", {})
    index = build_session_index(cohort, manifest_path=manifest_path,
                                span_cache=span_cache, progress=progress)
    index["meta"] = {**expected, "built_at": time.time()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(index, fh)
    os.replace(tmp, path)
    return index


def build_session_index(cohort: str, manifest_path: Optional[Path] = None,
                        span_cache: Optional[Dict[str, Any]] = None,
                        progress=None) -> Dict[str, Any]:
    """Scan the share and the manifest for ``cohort``. Minutes cold, seconds warm."""
    def say(msg):
        logger.info(msg)
        if progress is not None:
            progress(msg)

    span_cache = span_cache if span_cache is not None else {}
    config = _load_config(str(_config_file(cohort)))
    sessions, generated_at = _manifest_sessions(manifest_path)

    say("listing recordings…")
    try:
        recs = get_animals_and_sessions(_config=config)
    except Exception as exc:
        logger.warning("could not list recordings for %s: %s", cohort, exc)
        recs = pd.DataFrame(columns=["session", "animal", "is_primary", "session_dir"])

    say("scanning raw video chunks…")
    chunks = scan_video_chunks(config, span_cache)

    animals_by_rec: Dict[str, List[str]] = {}
    primary_by_rec: Dict[str, bool] = {}
    for rid, grp in recs.groupby("session") if not recs.empty else []:
        animals_by_rec[rid] = sorted(grp["animal"].unique().tolist())
        primary_by_rec[rid] = bool(grp["is_primary"].any())
    # A manifest-only recording (dir since moved) still gets a row.
    for rid, rec in sessions.items():
        if rec.get("cohort") == cohort and rid not in animals_by_rec:
            animals_by_rec[rid] = sorted((rec.get("ephys") or {}).get("animals") or [])
            primary_by_rec[rid] = bool((rec.get("recording") or {}).get("is_primary"))

    dates = sorted({rid[:8] for rid in animals_by_rec} | {c.date for c in chunks})
    event_dates = _event_dates(config)

    days: Dict[str, Dict[str, Any]] = {}
    details: Dict[str, Dict[str, Any]] = {}
    rows = []

    for i, date in enumerate(dates):
        say(f"[{i + 1}/{len(dates)}] {date}: tracking + events")
        try:
            tracking_files = get_tracking_files_by_date(date, _config=config)
        except Exception as exc:
            logger.debug("no tracking for %s: %s", date, exc)
            tracking_files = []
        apt_keys, manual_spans = tracking_spans(tracking_files, span_cache)
        event_files = canonical_event_files(date, config) if date in event_dates else []
        events = read_event_times(event_files)
        ev_t = events["t_epoch"].to_numpy()

        day_chunks = [c for c in chunks if c.date == date]
        for c in day_chunks:
            if c.key in apt_keys:
                c.tracked, c.tracked_by = True, "APT"
            elif any(min(c.t1, b) - max(c.t0, a) > 60 for a, b in manual_spans):
                c.tracked, c.tracked_by = True, "manual"
            c.n_events = int(np.count_nonzero((ev_t >= c.t0) & (ev_t < c.t1)))

        rids = sorted(r for r in animals_by_rec if r[:8] == date)
        stamps = []
        for rid in rids:
            m = _RECORDING_RE.match(rid)
            stamps.append(_stamp_epoch(m.group(1), m.group(2)) if m else None)
        windows = {}
        for j, rid in enumerate(rids):
            nxt = stamps[j + 1] if j + 1 < len(stamps) else None
            windows[rid] = block_epoch_window(rid, sessions.get(rid), nxt)

        for c in day_chunks:
            best, best_ov = None, 0.0
            for rid, (a, b, _) in windows.items():
                ov = min(c.t1, b) - max(c.t0, a)
                if ov > best_ov:
                    best, best_ov = rid, ov
            c.block = best

        days[date] = {
            "date": date,
            "chunks": [asdict(c) for c in day_chunks],
            "events": events,
            "event_files": [str(f) for f in event_files],
            "tracking_files": [str(f) for f in tracking_files],
            "tracking_signature": [_file_sig(Path(f)) for f in tracking_files],
            "manual_spans": manual_spans,
            "block_windows": {rid: list(w) for rid, w in windows.items()},
            "recordings": rids,
        }

        for j, rid in enumerate(rids):
            record = sessions.get(rid)
            animals = per_animal_facts(record, animals_by_rec.get(rid, []))
            mine = [c for c in day_chunks if c.block == rid]
            a, b, how = windows[rid]
            in_block = (ev_t >= a) & (ev_t <= b)
            types = events.loc[in_block, "type"].value_counts()
            loaded = [x for x in animals if x["n_quality_cells"] is not None]
            good = [x["n_quality_cells"] for x in loaded]
            ready = ((record or {}).get("analysis_readiness") or {})
            m = _RECORDING_RE.match(rid)
            rows.append({
                "recording": rid,
                "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "start": f"{m.group(2)[:2]}:{m.group(2)[2:4]}" if m else "",
                "block": f"{j + 1}/{len(rids)}" + (" ★" if primary_by_rec.get(rid) else ""),
                "ephys_animals": len(animals_by_rec.get(rid, [])),
                "loadable": sum(x["status"] == "ok" for x in animals) if record else None,
                "good_neurons": int(sum(good)) if good else None,
                "min_good": int(min(good)) if good else None,
                "ephys_h": round(max((x["window_hours"] or 0) for x in animals), 2)
                if any(x["window_hours"] for x in animals) else None,
                "video": len(mine),
                "tracked": sum(c.tracked for c in mine),
                "annotated": sum(c.n_events > 0 for c in mine),
                "broken": sum(c.broken for c in mine),
                "events": int(in_block.sum()),
                "top_events": ", ".join(f"{k} {v}" for k, v in types.head(3).items()),
                "sync": ("ok" if block_sync(record) else "—") if record else "?",
                "ready": ", ".join(tag for key, tag in READY_TAGS.items()
                                   if (ready.get(key) or {}).get("testable")),
                "animals": " ".join(animals_by_rec.get(rid, [])),
                "manifest": (record or {}).get("provenance", {}).get("probe_level", "missing")
                if record else "missing",
            })
            details[rid] = {
                "recording": rid,
                "date": date,
                "block_index": j,
                "n_blocks": len(rids),
                "is_primary": primary_by_rec.get(rid, False),
                "animals": animals,
                "block_window": [a, b],
                "block_window_source": how,
                "event_types": types.to_dict(),
                "sync": asdict(block_sync(record)) if block_sync(record) else None,
            }

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("recording", ascending=False).reset_index(drop=True)
        for col in ("loadable", "good_neurons", "min_good"):
            table[col] = table[col].astype("Int64")   # NaN would otherwise force floats
    return {
        "cohort": cohort,
        "table": table,
        "details": details,
        "days": days,
        "manifest_generated_at": generated_at,
        "span_cache": span_cache,
    }


# ---------------------------------------------------------------------------
# Day tracking grid (slow; cached separately)
# ---------------------------------------------------------------------------

def _timeline_path(cohort: str, date: str) -> Path:
    return CACHE_DIR / f"timeline_{cohort}_{date}.pkl"


def _day_signature(day: Mapping[str, Any], bin_seconds: float) -> str:
    payload = json.dumps({"v": TIMELINE_VERSION, "bin": bin_seconds,
                          "files": day.get("tracking_signature", [])}, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cached_day_tracking_grid(cohort: str, day: Mapping[str, Any],
                             bin_seconds: float = 60.0) -> Optional[Dict[str, Any]]:
    """The cached per-animal tracking grid for a day, or None if absent/stale.

    Staleness is judged against the file signatures the index recorded, so this
    never touches the share.
    """
    path = _timeline_path(cohort, day["date"])
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            grid = pickle.load(fh)
    except Exception:
        return None
    if grid.get("signature") != _day_signature(day, bin_seconds):
        return None
    return grid


def get_day_tracking_grid(cohort: str, day: Mapping[str, Any], sync: LinearSync,
                          required_animals: Sequence[str] = (),
                          bin_seconds: float = 60.0) -> Dict[str, Any]:
    """Per-animal fraction of expected frames detected, on a wall-clock grid.

    Reads every tracking file of the day (the slow part), via
    :func:`video.tracking_ephys_timeline.tracking_grid`, then caches the result.
    Bin edges are stored in **epoch seconds** so the browser can draw them
    on the same axis as video chunks and events.
    """
    cached = cached_day_tracking_grid(cohort, day, bin_seconds)
    if cached is not None:
        return cached

    from video.tracking_ephys_timeline import tracking_grid

    date = day["date"]
    files = [Path(f) for f in day.get("tracking_files", [])]
    midnight = local_midnight_epoch(date)
    # Grid spans the whole local day (+6 h for blocks that run past midnight).
    edges_epoch = midnight + np.arange(0, 30 * 3600 + bin_seconds, bin_seconds)
    edges_ephys = sync.convert_behavior_to_ephys(edges_epoch)
    animals, grid, covered = tracking_grid(None, sync, edges_ephys,
                                           required_animals=required_animals, files=files)
    result = {
        "signature": _day_signature(day, bin_seconds),
        "date": date,
        "edges_epoch": edges_epoch,
        "animals": animals,
        "grid": grid,
        "covered": covered,
        "built_at": time.time(),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_timeline_path(cohort, date), "wb") as fh:
        pickle.dump(result, fh)
    return result


def day_sync(index: Mapping[str, Any], date: str) -> Optional[LinearSync]:
    """Any block's stored sync on ``date`` — all blocks of a day share one ephys clock."""
    for rid in index["days"].get(date, {}).get("recordings", []):
        s = (index["details"].get(rid) or {}).get("sync")
        if s:
            return LinearSync(**s)
    return None
