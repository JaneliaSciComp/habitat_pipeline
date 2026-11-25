"""Tests for ``ingestion/apt_frame_clock.py`` (synthetic FERAL inputs, no share)."""
import json

import numpy as np
import pandas as pd

from ingestion import apt_frame_clock as afc
from ingestion import feral_import as fi
from tests.test_feral_import import MOVIE, MOVIE_T0, _write_inputs

BOUTS = [(2, "EC", "", 100, 300, 11), (11, "EC", "", 120, 310, 2),
         (4, "F", "initiator", 4000, 4100, 7), (7, "F", "victim", 4010, 4090, 4)]


def _setup(tmp_path, *, real_ts=False):
    csv, browser = _write_inputs(tmp_path, BOUTS)
    chunk = tmp_path / "apt" / MOVIE
    chunk.mkdir(parents=True)
    (chunk / "solution_keypoints.csv").write_text("t tracklet_id x y\n")
    if real_ts:
        np.save(chunk / f"{MOVIE}_ts.npy", np.arange(10, dtype=np.int64))
    config = {"tracking": [str(tmp_path / "apt")], "video": str(tmp_path / "video")}
    return csv, browser, config


def _local_ns(frame):
    return pd.Timestamp(MOVIE_T0 + pd.Timedelta(seconds=frame / fi.FPS)) \
        .tz_localize(fi.LOCAL_TZ).tz_convert("UTC").value


def test_builds_clock_through_feral_anchors(tmp_path):
    csv, browser, config = _setup(tmp_path)
    out = tmp_path / "clock"
    manifest = afc.build_frame_clocks(csv, browser, out, config=config)
    ts = np.load(out / f"{MOVIE}_ts.npy")
    assert ts.dtype == np.int64 and len(ts) == afc.FULL_CHUNK_FRAMES
    # exact (to the ms truncation) at bout boundaries, 25 ms steps between them
    for frame in (100, 300, 4000, 4100):
        assert abs(ts[frame] - _local_ns(frame)) <= 1_000_000
    assert np.all(np.diff(ts) > 0)
    assert manifest["movies"][MOVIE]["n_anchors"] >= 8
    assert json.loads((out / afc.MANIFEST_NAME).read_text())["source"] == "feral_anchor"


def test_movies_with_real_ts_are_skipped(tmp_path):
    csv, browser, config = _setup(tmp_path, real_ts=True)
    out = tmp_path / "clock"
    manifest = afc.build_frame_clocks(csv, browser, out, config=config)
    assert manifest["skipped_real_ts"] == [MOVIE]
    assert not (out / f"{MOVIE}_ts.npy").exists()
    manifest = afc.build_frame_clocks(csv, browser, out, config=config, skip_real=False)
    assert (out / f"{MOVIE}_ts.npy").exists()


def test_movie_without_feral_bouts_is_reported(tmp_path):
    csv, browser, config = _setup(tmp_path)
    other = tmp_path / "apt" / "cohort7_20251216_1029"
    other.mkdir()
    (other / "solution_keypoints.csv").write_text("t tracklet_id x y\n")
    manifest = afc.build_frame_clocks(csv, browser, tmp_path / "clock", config=config)
    assert manifest["no_feral_anchors"] == ["cohort7_20251216_1029"]
