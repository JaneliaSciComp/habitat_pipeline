"""Tests for gui/session_index.py — the session browser's data layer.

A temporary share mirrors the real layouts: two recording blocks on one day
(``20251216_094334`` and ``20251216_144334``), 30-minute raw video chunks with
``_ts.npy`` siblings, one APT chunk directory, one manual mask-metrics
export, the canonical event CSV in its date directory, and a loose event CSV
one level up that must be ignored (HZ-DATA-007). Nothing touches //nearline.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import gui.session_index as si

DATE = "20251216"


def _epoch(hhmm):
    return datetime.strptime(DATE + hhmm, "%Y%m%d%H%M").timestamp()


#: Ephys second 0 is 09:43:34 local, as for the real primary block.
T0 = datetime.strptime(DATE + "094334", "%Y%m%d%H%M%S").timestamp()


def _write_chunk(video_dir, hhmm, n_frames=1800, stub=False, empty_ts=False):
    mp4 = video_dir / f"RatCity_{DATE}_{hhmm}_40Hz.mp4"
    with open(mp4, "wb") as fh:
        fh.truncate(1000 if stub else 2_000_000)
    ts = np.array([], dtype=np.int64) if empty_ts else \
        ((_epoch(hhmm) + np.arange(n_frames)) * 1e9).astype(np.int64)
    np.save(video_dir / f"RatCity_{DATE}_{hhmm}_40Hz_ts.npy", ts)


def _record(rid, windows, load_errors=()):
    per = {}
    for animal, window in windows.items():
        per[animal] = {
            "n_clusters": 150, "n_quality_cells": None if animal in load_errors else 100,
            "ephys_window": None if animal in load_errors else window,
            "load_error": "FileNotFoundError: No '*.timestamps.dat' found"
            if animal in load_errors else None,
            "sync": {"ok": True, "slope": 1.0, "intercept": T0},
        }
    return {
        "cohort": "cohortT", "session_date": DATE, "session_id": rid,
        "recording": {"recording_id": rid, "is_primary": rid.endswith("094334")},
        "ephys": {"animals": sorted(windows), "per_animal": per},
        "analysis_readiness": {"ephys.decode_event_outcome": {"testable": True}},
        "provenance": {"probe_level": "full"},
    }


@pytest.fixture
def share(tmp_path, monkeypatch):
    root = tmp_path / "share"
    ephys = root / "ephys"
    for rec_dir, animal, stems in [
        ("20251216_094334", "rat613", ["20251216_094334_merged", "20251216_144334_merged"]),
        ("20251216_094334", "rat615", ["20251216_094334_merged"]),
    ]:
        for stem in stems:
            (ephys / f"{rec_dir}.rec" / animal / f"{stem}.kilosort").mkdir(parents=True)

    video = root / "video" / "social_videos"
    video.mkdir(parents=True)
    _write_chunk(video, "0229")                         # before any block
    _write_chunk(video, "0959")                         # block A, APT-tracked
    _write_chunk(video, "1059")                         # block A, manually tracked
    _write_chunk(video, "1459")                         # block B, untracked
    _write_chunk(video, "1529", stub=True, empty_ts=True)  # block B, broken

    apt = root / "apt" / f"cohortT_{DATE}_0959" / "solution"
    apt.mkdir(parents=True)
    (apt / "TQT_named.csv").write_text("timestamp,rat613_center_x,rat613_center_y\n")

    manual = root / "manual" / f"RatCity_{DATE}_1059_40Hz"
    manual.mkdir(parents=True)
    n = 1800
    ts = ((_epoch("1059") + np.arange(n)) * 1e9).astype(np.int64)
    np.save(manual / f"RatCity_{DATE}_1059_40Hz_ts.npy", ts)
    frames = np.arange(n)
    pd.concat([
        pd.DataFrame({"frame": frames, "object_id": 1, "object_name": "613",
                      "center_x": 1.0, "center_y": 1.0}),
        pd.DataFrame({"frame": frames[: n // 2], "object_id": 2, "object_name": "630",
                      "center_x": 1.0, "center_y": 1.0}),
    ]).to_csv(manual / f"RatCity_{DATE}_1059_40Hz_mask_metrics.csv", index=False)

    events = root / "events" / DATE
    events.mkdir(parents=True)
    pd.DataFrame({
        "type": ["F", "EC", "EC"],
        "initiator": ["rat613"] * 3, "victim": ["rat630"] * 3,
        "ts_start": [int(_epoch("1005") * 1e9), int(_epoch("1006") * 1e9),
                     int(_epoch("1500") * 1e9)],
        "ts_end": [0, 0, 0],
    }).to_csv(events / f"{DATE}_behavior_event_df.csv", index=False)
    # Loose export one level up: newer, and must never be read.
    pd.DataFrame({"type": ["F"] * 5, "ts_start": [int(_epoch("1100") * 1e9)] * 5}) \
        .to_csv(root / "events" / f"{DATE}_behavior_event_df_update.csv", index=False)

    config = tmp_path / "cohortT_paths.json"
    config.write_text(json.dumps({
        "video": str(root / "video"), "ephys": str(ephys),
        "tracking": [str(root / "manual"), str(root / "apt")],
        "events": str(root / "events"), "pixels_per_cm": None,
    }))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "generated_at": "2026-10-01T00:00:00Z", "cohorts": [],
        "sessions": {
            "20251216_094334": _record("20251216_094334",
                                       {"rat613": [0.0, 3 * 3600.0], "rat615": None},
                                       load_errors=("rat615",)),
            "20251216_144334": _record("20251216_144334",
                                       {"rat613": [5 * 3600.0, 7 * 3600.0]}),
        },
    }))

    monkeypatch.setitem(si.COHORT_CONFIGS, "cohortT", str(config))
    monkeypatch.setattr(si, "_config_file", lambda cohort: config)
    monkeypatch.setattr(si, "CACHE_DIR", tmp_path / "cache")
    return {"root": root, "config": config, "manifest": manifest, "tmp": tmp_path}


def _rows(index):
    return index["table"].set_index("recording")


class TestCounts:
    def test_one_row_per_recording_block(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        assert sorted(index["table"]["recording"]) == ["20251216_094334", "20251216_144334"]

    def test_block_a_video_tracked_annotated(self, share):
        row = _rows(si.load_session_index("cohortT", manifest_path=share["manifest"])) \
            .loc["20251216_094334"]
        assert row["video"] == 2
        assert row["tracked"] == 2          # one APT chunk, one manual chunk
        assert row["annotated"] == 1        # both events fall in the 09:59 chunk
        assert row["broken"] == 0
        assert row["events"] == 2

    def test_block_b_counts_broken_chunk_once(self, share):
        row = _rows(si.load_session_index("cohortT", manifest_path=share["manifest"])) \
            .loc["20251216_144334"]
        assert row["video"] == 2
        assert row["tracked"] == 0
        assert row["broken"] == 1
        assert row["annotated"] == 1
        assert row["events"] == 1

    def test_loose_event_export_is_ignored(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        day = index["days"][DATE]
        assert len(day["events"]) == 3
        assert all(os.path.basename(os.path.dirname(f)) == DATE for f in day["event_files"])

    def test_chunk_outside_every_block_is_unassigned(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        blocks = {c["key"]: c["block"] for c in index["days"][DATE]["chunks"]}
        assert blocks[f"{DATE}_0229"] is None
        assert blocks[f"{DATE}_0959"] == "20251216_094334"
        assert blocks[f"{DATE}_1529"] == "20251216_144334"

    def test_neurons_and_loadability(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        row = _rows(index).loc["20251216_094334"]
        assert row["good_neurons"] == 100
        assert row["loadable"] == 1
        animals = {a["animal"]: a for a in index["details"]["20251216_094334"]["animals"]}
        assert animals["rat613"]["loadable"]
        assert not animals["rat615"]["loadable"]
        assert row["ready"] == "outcome"


class TestCache:
    def test_reused_when_unchanged(self, share):
        first = si.load_session_index("cohortT", manifest_path=share["manifest"])
        second = si.load_session_index("cohortT", manifest_path=share["manifest"])
        assert first["meta"]["built_at"] == second["meta"]["built_at"]

    def test_rebuilt_when_manifest_changes(self, share):
        first = si.load_session_index("cohortT", manifest_path=share["manifest"])
        st = share["manifest"].stat()
        os.utime(share["manifest"], (st.st_atime, st.st_mtime + 10))
        second = si.load_session_index("cohortT", manifest_path=share["manifest"])
        assert second["meta"]["built_at"] > first["meta"]["built_at"] or \
            second["meta"]["manifest_mtime"] != first["meta"]["manifest_mtime"]

    def test_refresh_picks_up_new_chunk(self, share):
        si.load_session_index("cohortT", manifest_path=share["manifest"])
        _write_chunk(share["root"] / "video" / "social_videos", "1129")
        stale = si.load_session_index("cohortT", manifest_path=share["manifest"])
        fresh = si.load_session_index("cohortT", manifest_path=share["manifest"],
                                      refresh=True)
        assert _rows(stale).loc["20251216_094334", "video"] == 2
        assert _rows(fresh).loc["20251216_094334", "video"] == 3


class TestNpyFirstLast:
    def test_reads_ends_without_full_load(self, tmp_path):
        path = tmp_path / "ts.npy"
        np.save(path, np.arange(10, 20, dtype=np.int64))
        assert si.npy_first_last(path) == (10.0, 19.0, 10)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "ts.npy"
        np.save(path, np.array([], dtype=np.int64))
        assert si.npy_first_last(path) == (None, None, 0)

    def test_zero_byte_file(self, tmp_path):
        path = tmp_path / "ts.npy"
        path.write_bytes(b"")
        assert si.npy_first_last(path) == (None, None, 0)


class TestDayTrackingGrid:
    def test_grid_cached_and_reused(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        day = dict(index["days"][DATE])
        day["tracking_files"] = [f for f in day["tracking_files"] if "mask_metrics" in f]
        sync = si.day_sync(index, DATE)
        assert si.cached_day_tracking_grid("cohortT", day) is None

        grid = si.get_day_tracking_grid("cohortT", day, sync, required_animals=["rat615"])
        assert grid["animals"] == ["rat613", "rat615", "rat630"]
        means = dict(zip(grid["animals"], np.nanmean(grid["grid"], axis=1)))
        assert means["rat613"] == pytest.approx(1.0, abs=0.05)
        assert means["rat630"] == pytest.approx(0.5, abs=0.05)
        assert means["rat615"] == 0.0
        # Only the 30 min the manual file spans are "covered".
        assert grid["covered"].sum() == pytest.approx(30, abs=1)

        again = si.cached_day_tracking_grid("cohortT", day)
        assert again is not None and again["built_at"] == grid["built_at"]

    def test_grid_invalidated_when_files_change(self, share):
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        day = dict(index["days"][DATE])
        si.get_day_tracking_grid("cohortT", day, si.day_sync(index, DATE))
        day["tracking_signature"] = day["tracking_signature"] + [("new.csv", 1, 1.0)]
        assert si.cached_day_tracking_grid("cohortT", day) is None


class TestManifestFailure:
    def test_unreadable_manifest_raises_and_keeps_previous_index(self, share):
        good = si.load_session_index("cohortT", manifest_path=share["manifest"])
        share["manifest"].write_text("{not json")
        with pytest.raises(si.ManifestUnavailable):
            si.load_session_index("cohortT", manifest_path=share["manifest"], refresh=True)
        with open(si._index_path("cohortT"), "rb") as fh:
            kept = __import__("pickle").load(fh)
        assert kept["meta"]["built_at"] == good["meta"]["built_at"]
        assert _rows(kept).loc["20251216_094334", "good_neurons"] == 100

    def test_missing_manifest_still_builds(self, share):
        share["manifest"].unlink()
        index = si.load_session_index("cohortT", manifest_path=share["manifest"])
        assert set(index["table"]["manifest"]) == {"missing"}
