"""Tests for the APT keypoints export (``solution_keypoints.csv``) and its discovery.

All mock data in ``tmp_path``; nothing touches //nearline. The fixture copies
what the 2026-11 export does differently from ``TQT_named.csv``:

- space-separated although named ``.csv`` (pandas would read it as one column);
- long, one row per ``(t, tracklet_id)``, numeric tracklets, no names;
- no timestamps (they come from a ``<movie>_ts.npy`` or the FERAL clock cache);
- an undetected frame is a placeholder row (all keypoints ``-1``,
  ``original_id == -1``, position carried forward), not a blank.

As for the wide format, ``TestFrameNumbering`` is the one that matters:
``timestamps`` is indexed by frame, so ``frame`` must be the file's ``t``
unchanged even after placeholders are dropped.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest

import ingestion.data_paths as dp
import video.tracking_import as ti
from ingestion.errors import MissingCompanionFile
from ingestion.nwb_import import IdentitySyncManager
from video.tracking_import import (
    TRACKING_FORMAT_APT_KEYPOINTS,
    TRACKING_FORMAT_APT_TQT,
    TRACKING_FORMAT_MASK_METRICS,
    detect_tracking_format,
    is_apt_format,
    load_apt_frame_timestamps,
    load_timestamps_with_source,
    load_tracking_data,
    parse_apt_keypoints,
    read_tracking_centers,
    read_tracking_header,
    tracking_source,
)

MOVIE = "cohort7_20251216_0959"
_FRAME_NS = 25_000_000
_T0 = 1_765_897_200_041_500_040
NAMES = {2: "rat4631", 4: "rat4613", 11: "rat4635"}
_COLUMNS = ("group id t y x parent_id tracklet_id kp0_y kp0_x kp1_y kp1_x "
            "kp2_y kp2_x kp3_y kp3_x original_id original_tracklet_id").split()


def _keypoints_frame(n_frames=10, tracklets=(2, 4, 11), missing=None):
    """Long keypoints rows; *missing* maps tracklet -> frames written as placeholders."""
    missing = missing or {}
    rows = []
    for tr in tracklets:
        last = (0.0, 0.0)
        for t in range(n_frames):
            if t in missing.get(tr, ()):
                x, y = last                       # carried forward, as APT does
                kps = [-1.0] * 8
                oid = -1
            else:
                x, y = 100.0 * tr + t, 10.0 * tr + t
                kps = [x + k for k in range(8)]
                oid = 1000 * tr + t
                last = (x, y)
            rows.append([MOVIE, 1000 * tr + t, t, y, x, 0, tr,
                         kps[0], kps[1], kps[2], kps[3], kps[4], kps[5], kps[6], kps[7],
                         oid, tr + 20])
    return pd.DataFrame(rows, columns=_COLUMNS)


def _write_keypoints(tmp_path, *, cohort_dir="cohort7", movie=MOVIE, ts_frames=10,
                     ts_beside=True, **kw):
    chunk = tmp_path / cohort_dir / movie
    chunk.mkdir(parents=True, exist_ok=True)
    path = chunk / "solution_keypoints.csv"
    _keypoints_frame(**kw).to_csv(path, sep=" ", index=False)
    if ts_beside:
        np.save(chunk / f"{movie}_ts.npy", _T0 + np.arange(ts_frames, dtype=np.int64) * _FRAME_NS)
    return path


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """No real config or clock cache leaks in: an empty config dir and cache."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr(dp, "_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(ti, "APT_FRAME_TS_DIR", tmp_path / "clock_cache")
    return cfg_dir


def _write_config(cfg_dir, tmp_path, *, names=NAMES, video=None, name="default_paths.json"):
    cfg = {"tracking": [str(tmp_path / "manual"), str(tmp_path / "cohort7")],
           "video": str(video or tmp_path / "video"),
           "apt_tracklet_names": {str(k): v for k, v in names.items()}}
    (cfg_dir / name).write_text(json.dumps(cfg))
    return cfg


class TestFormatDetection:
    def test_space_separated_csv_is_read_as_columns(self, tmp_path):
        header = read_tracking_header(_write_keypoints(tmp_path))
        assert header == _COLUMNS
        assert detect_tracking_format(header) == TRACKING_FORMAT_APT_KEYPOINTS

    def test_apt_family(self):
        assert is_apt_format(TRACKING_FORMAT_APT_KEYPOINTS) and is_apt_format(TRACKING_FORMAT_APT_TQT)
        assert not is_apt_format(TRACKING_FORMAT_MASK_METRICS)
        assert tracking_source(TRACKING_FORMAT_APT_KEYPOINTS) == tracking_source(TRACKING_FORMAT_APT_TQT) == "apt"
        assert tracking_source(TRACKING_FORMAT_MASK_METRICS) == "manual"


class TestFrameNumbering:
    def test_placeholders_dropped_per_animal_without_renumbering(self, tmp_path):
        path = _write_keypoints(tmp_path, missing={2: [0, 3, 4], 11: [9]})
        tr = load_tracking_data(path, tracklet_names=NAMES)
        a = tr.parsed_data["rat631"]
        assert a["frame"].tolist() == [1, 2, 5, 6, 7, 8, 9]
        assert tr.parsed_data["rat635"]["frame"].tolist() == list(range(9))
        assert len(tr.parsed_data["rat613"]) == 10
        # positions stay with their own frame, timestamps index by frame
        assert np.allclose(a["center_x"], 200.0 + a["frame"])
        assert (a["timestamp"].to_numpy() == _T0 + a["frame"].to_numpy() * _FRAME_NS).all()
        traj = tr.get_object_trajectory("631")
        assert (traj["timestamps"].to_numpy() == (_T0 + traj["frame"] * _FRAME_NS)).all()

    def test_original_id_alone_marks_a_placeholder(self, tmp_path):
        df = _keypoints_frame()
        df.loc[(df.tracklet_id == 4) & (df.t == 5), "original_id"] = -1
        out = parse_apt_keypoints(df, NAMES)
        assert 5 not in out["rat613"]["frame"].tolist()

    def test_parsed_column_order_matches_tqt(self, tmp_path):
        tr = load_tracking_data(_write_keypoints(tmp_path), tracklet_names=NAMES)
        assert list(tr.parsed_data["rat613"].columns) == [
            "frame", "timestamp", "center_x", "center_y",
            "kp0_x", "kp0_y", "kp1_x", "kp1_y", "kp2_x", "kp2_y", "kp3_x", "kp3_y"]
        assert tr.tracking_format == TRACKING_FORMAT_APT_KEYPOINTS
        assert tr.timestamps.dtype == np.int64


class TestNames:
    def test_tracklets_named_and_normalised(self, tmp_path):
        tr = load_tracking_data(_write_keypoints(tmp_path), tracklet_names=NAMES)
        assert sorted(tr.get_object_names()) == ["rat613", "rat631", "rat635"]
        assert tr.get_object_data("635") is not None

    def test_cohort9_keeps_four_digit_names(self, tmp_path):
        path = _write_keypoints(tmp_path, cohort_dir="cohort9", movie="cohort9_20260929_0959")
        tr = load_tracking_data(path, tracklet_names=NAMES)
        assert "rat4635" in tr.parsed_data

    def test_unnamed_tracklet_raises(self, tmp_path):
        path = _write_keypoints(tmp_path, tracklets=(2, 4, 7))
        with pytest.raises(ValueError, match=r"\[7\]"):
            load_tracking_data(path, tracklet_names=NAMES)

    def test_no_map_raises(self, tmp_path):
        with pytest.raises(ValueError, match="apt_tracklet_names"):
            load_tracking_data(_write_keypoints(tmp_path))

    def test_map_comes_from_the_config_owning_the_path(self, tmp_path, _isolated_config):
        _write_config(_isolated_config, tmp_path)
        tr = load_tracking_data(_write_keypoints(tmp_path))
        assert sorted(tr.get_object_names()) == ["rat613", "rat631", "rat635"]
        assert dp.config_for_tracking_file(tmp_path / "elsewhere" / "x.csv") is None


class TestTimestamps:
    def test_sibling_ts_npy_first(self, tmp_path):
        tr = load_tracking_data(_write_keypoints(tmp_path), tracklet_names=NAMES)
        assert tr.timestamp_source == "apt_ts_npy"
        assert len(tr.timestamps) == 10

    def test_social_videos4apt_via_config_video_root(self, tmp_path, _isolated_config):
        _write_config(_isolated_config, tmp_path)
        path = _write_keypoints(tmp_path, ts_beside=False)
        apt_videos = tmp_path / "video" / "social_videos4APT"
        apt_videos.mkdir(parents=True)
        np.save(apt_videos / f"{MOVIE}_ts.npy", _T0 + np.arange(10, dtype=np.int64) * _FRAME_NS)
        ts, source = load_timestamps_with_source(path)
        assert source == "apt_ts_npy" and ts[3] == _T0 + 3 * _FRAME_NS

    def test_feral_clock_cache_last_and_trimmed(self, tmp_path):
        path = _write_keypoints(tmp_path, ts_beside=False)
        cache = tmp_path / "clock_cache"
        cache.mkdir()
        np.save(cache / f"{MOVIE}_ts.npy", _T0 + np.arange(72000, dtype=np.int64) * _FRAME_NS)
        tr = load_tracking_data(path, tracklet_names=NAMES)
        assert tr.timestamp_source == "feral_anchor"
        assert len(tr.timestamps) == 10          # trimmed to the file's frames

    def test_none_found(self, tmp_path, caplog):
        path = _write_keypoints(tmp_path, ts_beside=False)
        with pytest.raises(MissingCompanionFile, match="apt_frame_clock"):
            load_apt_frame_timestamps(path)
        with caplog.at_level(logging.WARNING):
            tr = load_tracking_data(path, tracklet_names=NAMES)
        assert tr.timestamps is None and tr.timestamp_source is None
        assert "timestamp" not in tr.parsed_data["rat631"].columns

    def test_short_ts_raises(self, tmp_path):
        path = _write_keypoints(tmp_path, ts_frames=5)
        with pytest.raises(ValueError, match="different video"):
            load_tracking_data(path, tracklet_names=NAMES)

    def test_sync_uses_each_animals_own_frames(self, tmp_path):
        path = _write_keypoints(tmp_path, missing={2: [0, 1]})
        tr = load_tracking_data(path, tracklet_names=NAMES)
        assert tr.synchronize_with_ephys(IdentitySyncManager())
        a = tr.parsed_data["rat631"]
        assert np.allclose(a["ephys_timestamps"], (_T0 + a["frame"] * _FRAME_NS) / 1e9)


class TestReadTrackingCenters:
    def test_long_shape(self, tmp_path):
        path = _write_keypoints(tmp_path, missing={4: [2]})
        c = read_tracking_centers(path, tracklet_names=NAMES)
        assert list(c.columns) == ["frame", "object_name", "center_x", "center_y"]
        assert c.groupby("object_name").size().to_dict() == {"rat613": 9, "rat631": 10, "rat635": 10}


class TestDiscovery:
    def _tree(self, tmp_path, both=True):
        chunk = tmp_path / "apt" / "cohort7_20251216_0959"
        (chunk / "solution").mkdir(parents=True)
        (chunk / "solution_keypoints.csv").write_text("t tracklet_id x y\n")
        if both:
            (chunk / "solution" / "TQT_named.csv").write_text("timestamp,rat613_center_x\n")
        old = tmp_path / "apt" / "cohort7_20251216_1029" / "solution"
        old.mkdir(parents=True)
        (old / "TQT_named.csv").write_text("timestamp,rat613_center_x\n")
        return {"tracking": [str(tmp_path / "apt")]}

    def test_one_file_per_chunk_preferring_keypoints(self, tmp_path):
        files = dp.get_tracking_files_by_date("20251216", _config=self._tree(tmp_path))
        names = [(f.parent.name if f.parent.name != "solution" else f.parent.parent.name, f.name)
                 for f in files]
        assert sorted(names) == [("cohort7_20251216_0959", "solution_keypoints.csv"),
                                 ("cohort7_20251216_1029", "TQT_named.csv")]

    def test_helpers(self, tmp_path):
        self._tree(tmp_path)
        chunk = tmp_path / "apt" / "cohort7_20251216_0959"
        assert dp.apt_chunk_tracking_file(chunk).name == "solution_keypoints.csv"
        assert dp.apt_chunk_tracking_file(tmp_path) is None
        assert dp.is_apt_tracking_file(chunk / "solution_keypoints.csv")
        assert dp.is_apt_tracking_file("x/solution/TQT_named.csv")
        assert not dp.is_apt_tracking_file("x/RatCity_20251216_mask_metrics.csv")

    def test_manager_exposes_tracklet_names(self, tmp_path, _isolated_config):
        _write_config(_isolated_config, tmp_path, name="c.json")
        dsm = dp.DataStorageManager("631", "20251216", config_path=str(_isolated_config / "c.json"),
                                    auto_load=False)
        assert dsm.get_apt_tracklet_names() == NAMES


class TestGuiConsumers:
    def test_session_index_counts_keypoints_chunks(self, tmp_path):
        from gui.session_index import tracking_spans
        f = tmp_path / "cohort7_20251216_0959" / "solution_keypoints.csv"
        keys, manual = tracking_spans([f], {})
        assert keys == {"20251216_0959"} and manual == []

    def test_explore_cache_chunk_span(self, tmp_path):
        from gui import explore_cache as xc
        span = xc._apt_chunk_epoch_span(tmp_path / "cohort7_20251216_0959" / "solution_keypoints.csv")
        assert span is not None and span[1] - span[0] == 1800
