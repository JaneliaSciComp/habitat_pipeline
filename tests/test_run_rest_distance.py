"""Input-building tests for ephys.run_rest_distance (mock tracks, no share needed)."""

import numpy as np
import pandas as pd

from ephys.run_rest_distance import bin_mean, bridge_gaps, keypoint_motion, replication, rest_inputs


class TestBridgeGaps:
    def test_fills_short_interior_gaps_only(self):
        m = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0], bool)
        out = bridge_gaps(m, 2)
        assert out.tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0]       # 1-bin gap filled; 3-bin gap, edges not

    def test_zero_and_empty_are_noops(self):
        m = np.array([1, 0, 1], bool)
        assert bridge_gaps(m, 0).tolist() == m.tolist()
        assert not bridge_gaps(np.zeros(5, bool), 3).any()

    def test_bridge_never_crosses_leaving_home(self):
        t = np.arange(0, 100, 0.1)
        x = np.where((t > 40) & (t < 41), 500.0, 100.0)                      # steps out for 1 s
        tr = {"rat1": _track(t, x, 100 + 0 * t), "rat2": _track(t, 400 + 0 * t, 100 + 0 * t)}
        R = rest_inputs(tr, None, "rat1", (0, 100), home=(100, 100), radius=50, still_px_s=1e9, bridge_sec=5.0)
        assert not R["rest"][(R["inp"]["centers"] > 40.5) & (R["inp"]["centers"] < 40.9)].any()


def _track(t, x, y):
    return pd.DataFrame(dict(t=t, x=x, y=y, speed=np.r_[0.0, np.hypot(np.diff(x), np.diff(y)) / np.diff(t)]))


def _kp_obj(t, cx, cy, wiggle):
    n = len(t)
    d = dict(ephys_timestamps=t, center_x=cx, center_y=cy)
    for i in range(2):
        d[f"kp{i}_x"] = cx + 10 * (i + 1) + wiggle
        d[f"kp{i}_y"] = cy + 0 * t
    return pd.DataFrame(d)


class TestKeypointMotion:
    def test_translation_vs_posture(self):
        t = np.arange(0, 20, 1 / 30)
        moving = keypoint_motion(_kp_obj(t, 100.0 * t, 0 * t, 0 * t))        # whole body translates at 100 px/s
        assert np.isclose(np.nanmedian(moving.kp_speed), 100.0, rtol=0.02)
        assert np.nanmax(moving.posture_speed) < 1e-6                         # but its posture does not change
        wig = keypoint_motion(_kp_obj(t, 0 * t + 5, 0 * t, 3 * np.sin(2 * np.pi * t)))
        assert np.nanmedian(wig.kp_speed) > 1.0 and np.nanmedian(wig.posture_speed) > 1.0
        assert np.nanmedian(np.abs(wig.kp_speed - wig.posture_speed)) < 1e-6   # centre fixed => same thing

    def test_gap_is_not_motion(self):
        t = np.r_[np.arange(0, 5, 1 / 30), np.arange(10, 15, 1 / 30)]
        m = keypoint_motion(_kp_obj(t, 50.0 * (t > 7), 0 * t, 0 * t))
        assert m.kp_speed.iloc[len(t[t < 5]) - 1] != m.kp_speed.iloc[len(t[t < 5]) - 1]  # NaN across the 5 s hole

    def test_no_keypoints_returns_none(self):
        assert keypoint_motion(pd.DataFrame(dict(ephys_timestamps=[0.0, 1.0], center_x=[0, 1], center_y=[0, 1]))) is None


def test_bin_mean_ignores_empty_and_nan_bins():
    centers = 0.25 + 0.5 * np.arange(4)
    t = np.array([0.1, 0.2, 1.1])
    v = np.array([1.0, 3.0, np.nan])
    out = bin_mean(t, v, centers)
    assert out[0] == 2.0 and np.isnan(out[1:]).all()


class TestRestInputs:
    def _scene(self):
        t = np.arange(0, 600, 0.1)
        focal = _track(t, 100 + 0 * t, 100 + 0 * t)                                # sits at (100, 100) all along
        partner = _track(t, 100 + 300 + 200 * np.sin(2 * np.pi * t / 120), 100 + 0 * t)
        return {"rat1": focal, "rat2": partner}

    def test_rest_mask_and_distance(self):
        tr = self._scene()
        R = rest_inputs(tr, None, "rat1", (0, 600), home=(100, 100), radius=50, still_px_s=20)
        assert R["rest"].mean() > 0.95
        ok = np.isfinite(R["d"])
        assert np.isclose(np.exp(R["d"][ok]).min(), 100, rtol=0.05) and np.exp(R["d"][ok]).max() < 510
        assert R["nuis"].shape[1] == 1 and R["nuis_names"] == ["log1p_speed"]

    def test_rest_requires_being_at_home(self):
        tr = self._scene()
        R = rest_inputs(tr, None, "rat1", (0, 600), home=(900, 900), radius=50, still_px_s=20)
        assert not R["rest"].any()

    def test_motion_adds_nuisance_columns(self):
        tr = self._scene()
        t = np.arange(0, 600, 1 / 30)
        mo = keypoint_motion(_kp_obj(t, 0 * t + 100, 0 * t + 100, 2 * np.sin(t)))
        R = rest_inputs(tr, mo, "rat1", (0, 600), home=(100, 100), radius=50, still_px_s=20)
        assert R["nuis_names"] == ["log1p_speed", "log1p_kp_speed", "log1p_posture_speed"]
        assert np.isfinite(R["nuis"][R["rest"]]).all()


def test_replication_correlates_cells_across_chunks():
    rng = np.random.default_rng(0)
    r = rng.normal(size=30)
    mk = lambda name, v: dict(chunk=name, status="ok", cells=pd.DataFrame(dict(cluster_id=np.arange(30), r=v)))  # noqa: E731
    out = replication([mk("a", r), mk("b", r + 0.1 * rng.normal(size=30)), dict(chunk="c", status="insufficient_rest")])
    assert out["status"] == "ok" and out["mean_r_between_chunks"] > 0.9
    assert replication([mk("a", r)])["status"] == "needs_2_tested_chunks"
