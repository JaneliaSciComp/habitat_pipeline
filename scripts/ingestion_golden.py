"""Fingerprint the ingestion layer's outputs on real data, to prove a refactor changed nothing.

    python scripts/ingestion_golden.py --save baseline      # before touching ingestion code
    python scripts/ingestion_golden.py --compare baseline   # after every step

Each target (resolved paths, raw Kilosort load, sync fit, events, tracking) is
reduced to hashes of its arrays/frames plus a few scalars, and written to
``.cache/ingestion_golden/<name>.json``. ``--compare`` re-runs everything and
prints each key whose fingerprint differs. Kilosort loads use
``force_reload=True`` so the raw-file path is what is exercised, not a pkl.
Needs the ``//nearline`` share; takes several minutes (each raw load reads a
~2 GB ``timestamps.dat``). ``--only`` restricts to target names containing a
substring, e.g. ``--only paths``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion.data_paths import DataStorageManager, get_animals_and_sessions  # noqa: E402
from ingestion.ephys_sync import DataSyncManager  # noqa: E402
from ingestion.kilosort_data_import import load_kilosort_data  # noqa: E402
from video.behavioral_events import load_behavioral_events  # noqa: E402
from video.tracking_import import load_tracking_data, resolve_tracking_on_ephys_clock  # noqa: E402

OUT_DIR = ROOT / ".cache" / "ingestion_golden"

COHORT5 = "cohort5_paths.json"
COHORT9 = "cohort9_paths.json"

PATH_TARGETS = [
    ("631", "20251216", None),
    ("613", "20251216_144334", None),
    ("613", "20251216_194334", None),
    ("615", "20251210", None),
    ("631", "20251217", None),
    ("rat651", "20250812_115448", COHORT5),
    ("rat650", "20250812_115448", COHORT5),
    ("rat4678", "20260929_095542", COHORT9),
]
KS_TARGETS = [
    ("631", "20251216", None),
    ("613", "20251216_144334", None),
    ("rat4678", "20260929_095542", COHORT9),
]
SYNC_TARGETS = [
    ("631", "20251216", None),
    ("rat4678", "20260929_095542", COHORT9),
]


def _h(obj) -> str:
    """Stable short hash of an array, list of arrays, DataFrame or JSON-able value."""
    digest = hashlib.sha256()
    if isinstance(obj, pd.DataFrame):
        digest.update(repr(list(obj.columns)).encode())
        digest.update(repr([str(t) for t in obj.dtypes]).encode())
        digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        digest.update(f"{arr.dtype}{arr.shape}".encode())
        digest.update(arr.tobytes() if arr.dtype != object else repr(arr.tolist()).encode())
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(x, np.ndarray) for x in obj):
        for x in obj:
            digest.update(_h(x).encode())
    else:
        digest.update(json.dumps(obj, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def _dsm(animal, session, config, **kw):
    return DataStorageManager(animal, session, config_path=config, **kw)


def fp_paths(animal, session, config):
    dsm = _dsm(animal, session, config, use_cache=False)
    d = dsm._to_cache_dict()
    d.pop("version", None)
    return d


def fp_inventory(config):
    df = get_animals_and_sessions(config_path=config)
    return {"n_rows": len(df), "hash": _h(df.astype(str))}


def fp_kilosort(animal, session, config):
    dsm = _dsm(animal, session, config)
    ks = load_kilosort_data(dsm.get_kilosort_path(), force_reload=True)
    out = {
        "animal_id": ks.animal_id, "session_id": ks.session_id,
        "n_clusters": len(ks.ks_ids), "n_spikes": int(len(ks.spike_times)),
        "duration_seconds": ks.duration_seconds, "ephys_window": list(ks.ephys_window),
        "metadata": {k: v for k, v in ks.metadata.items()},
    }
    for name in ("ks_ids", "spike_times", "spike_clusters", "spike_times_by_cell", "channel",
                 "amplitude", "fr", "amp", "DV", "XX", "cell_numbers", "to_load",
                 "curated_cells", "channel_map", "cluster_info", "ks_labels"):
        value = getattr(ks, name)
        if isinstance(value, list) and value and not isinstance(value[0], np.ndarray):
            value = np.asarray(value)
        out[name] = None if value is None else _h(value)
    passed = ks.filter_cells_by_firing_patterns()["passed_clusters"]
    out["passed_default"] = _h(np.asarray(passed))
    ids, _ = ks.get_filtered_cells_spike_times(min_firing_rate=0.5, min_presence_ratio=0.8,
                                               max_cv_isi=5.0)
    out["passed_decoder_defaults"] = _h(np.asarray(ids))
    mat, centers = ks.bin_spike_times(bin_size_sec=1.0)
    out["binned"] = _h(mat)
    out["bin_centers"] = _h(centers)
    return out


def fp_sync(animal, session, config):
    sync = DataSyncManager(_dsm(animal, session, config))
    m = sync.mapping
    out = {"method": sync.method, "slope": repr(float(sync.slope)),
           "intercept": repr(float(sync.intercept)), "n_matches": m.get("n_matches"),
           "r_squared": None if m.get("r_squared") is None else repr(float(m["r_squared"]))}
    for name in ("matched_ephys", "matched_behavior"):
        if name in m:
            out[name] = _h(np.asarray(m[name]))
    if sync.ephys_sync is not None:
        out["ephys_sync"] = _h(np.asarray(sync.ephys_sync))
        out["behavior_sync"] = _h(np.asarray(sync.behavior_sync))
        out["system_time"] = repr(sync.system_time)
    if "pulse_check" in m:
        out["pulse_check"] = {k: v for k, v in m["pulse_check"].items()}
    return out


def fp_events(animal, session, config):
    dsm = _dsm(animal, session, config)
    files = dsm.get_behavioral_event_files()
    ev = load_behavioral_events(files, session_id=session)
    out = {"files": [str(f) for f in files], "n_rows": len(ev.events_data),
           "raw": _h(ev.events_data.drop(columns=["file_index"], errors="ignore"))}
    sync = DataSyncManager(dsm)
    ev.synchronize_with_ephys(sync)
    out["synced"] = _h(ev.events_data)
    rat = animal if animal.startswith("rat") else f"rat{animal}"
    for name, fn in (("opponent", ev.extract_opponent_labels),
                     ("outcome", ev.extract_outcome_labels),
                     ("group", ev.extract_group_labels)):
        try:
            out[name] = [_h(np.asarray(a)) for a in fn(rat)]
        except Exception as e:  # noqa: BLE001 - fingerprint the failure too
            out[name] = f"{type(e).__name__}: {e}"
    out["available_rats"] = ev.get_available_rats()
    out["types"] = ev.get_available_event_types()
    return out


def fp_tracking(animal, session, config, file_index):
    dsm = _dsm(animal, session, config)
    tr = load_tracking_data(dsm, file_index=file_index)
    out = {"file": str(tr.tracking_file), "format": tr.tracking_format,
           "objects": tr.get_object_names(),
           "timestamps": None if tr.timestamps is None else _h(np.asarray(tr.timestamps))}
    for name, df in sorted(tr.parsed_data.items()):
        out[f"obj:{name}"] = _h(df)
    sync = DataSyncManager(dsm)
    resolved = resolve_tracking_on_ephys_clock(
        tr, sync, ["rat613", "rat615", "rat630", "rat631"], pixels_per_cm=dsm.get_pixels_per_cm())
    for name, df in sorted(resolved.items()):
        out[f"ephys:{name}"] = _h(df)
    return out


def targets():
    for a, s, c in PATH_TARGETS:
        yield f"paths:{a}:{s}", lambda a=a, s=s, c=c: fp_paths(a, s, c)
    for c in (None, COHORT5, COHORT9):
        yield f"inventory:{c or 'cohort7'}", lambda c=c: fp_inventory(c)
    for a, s, c in SYNC_TARGETS:
        yield f"sync:{a}:{s}", lambda a=a, s=s, c=c: fp_sync(a, s, c)
    yield "events:631:20251216", lambda: fp_events("631", "20251216", None)
    yield "events:rat4678:20260929_095542", lambda: fp_events("rat4678", "20260929_095542", COHORT9)
    for i in (0, 1):
        yield f"tracking:631:20251216:{i}", lambda i=i: fp_tracking("631", "20251216", None, i)
    for a, s, c in KS_TARGETS:
        yield f"kilosort:{a}:{s}", lambda a=a, s=s, c=c: fp_kilosort(a, s, c)


def run(only):
    results = {}
    for name, fn in targets():
        if only and only not in name:
            continue
        t0 = time.time()
        try:
            results[name] = fn()
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)
    # Round-trip through JSON so saved and fresh results compare like with like.
    return json.loads(json.dumps(results, sort_keys=True, default=str))


def _diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from _diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{prefix}{k}.")
    elif a != b:
        yield f"{prefix.rstrip('.')}: {a!r} -> {b!r}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="NAME")
    g.add_argument("--compare", metavar="NAME")
    p.add_argument("--only", default="")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    results = run(args.only)
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{args.save}.json"
        path.write_text(json.dumps(results, indent=1, sort_keys=True))
        print(f"saved {len(results)} targets to {path}")
        return 0

    saved = json.loads((OUT_DIR / f"{args.compare}.json").read_text())
    if args.only:
        saved = {k: v for k, v in saved.items() if args.only in k}
    diffs = list(_diff(saved, results))
    for line in diffs:
        print("DIFF", line)
    print(f"{len(results)} targets, {len(diffs)} differences")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
