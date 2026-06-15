"""
Tests for ephys/run_social_spatial.py.

Covers the argparse contract (the prompt's example invocation parses) and the
output writer (_write_outputs produces results.pkl, summary.png, and the
multi-page PDF) on a synthetic SocialFieldResults — no disk-backed
MultiAnimalSession required.
"""

import matplotlib
matplotlib.use("Agg")

import pickle  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ephys.social_spatial_fields import compute_social_place_fields  # noqa: E402
from ephys.run_social_spatial import (  # noqa: E402
    _build_parser,
    _write_outputs,
    _max_skaggs_by_cluster,
)
from video.tracking_import import VideoTrackingData  # noqa: E402

ARENA = ((0.0, 80.0), (0.0, 80.0))
DT = 0.04


class TestArgParser:
    def test_prompt_example_parses(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--session_id", "20251216",
            "--animal_ids", "631", "632", "633",
            "--focal", "631",
            "--bin_size", "5", "--smoothing", "5", "--speed_threshold", "5",
            "--speed_filter_subject", "target",
            "--n_shuffles", "500", "--output_dir", "./results",
        ])
        assert args.session_id == "20251216"
        assert args.animal_ids == ["631", "632", "633"]
        assert args.focal == "631"
        assert args.speed_filter_subject == "target"
        assert args.n_shuffles == 500
        assert args.null_method == "circular_shift"  # default

    def test_required_args_enforced(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_choices_enforced(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([
                "--session_id", "S", "--animal_ids", "A",
                "--focal", "A", "--output_dir", "/tmp",
                "--speed_filter_subject", "bogus",
            ])


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _xy(n, seed):
    rng = np.random.default_rng(seed)
    x = np.clip(np.cumsum(rng.normal(0, 3, n)) + 40, 0, 80)
    y = np.clip(np.cumsum(rng.normal(0, 3, n)) + 40, 0, 80)
    t = np.arange(n) * DT
    speed = np.sqrt(np.gradient(x, t) ** 2 + np.gradient(y, t) ** 2)
    return pd.DataFrame({"t": t, "x": x, "y": y, "speed": speed})


def _spikes(xy, center, seed):
    rng = np.random.default_rng(seed)
    x = xy["x"].to_numpy(); y = xy["y"].to_numpy(); t = xy["t"].to_numpy()
    rate = 0.2 + 25.0 * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * 8.0 ** 2))
    counts = rng.poisson(rate * DT)
    out = [ti + rng.uniform(0, DT, c) for ti, c in zip(t, counts) if c]
    return np.sort(np.concatenate(out)) if out else np.array([])


class _StubSync:
    def convert_behavior_to_ephys(self, behav_seconds):
        return np.asarray(behav_seconds, dtype=np.float64)


def _video_tracking(tracking, session_id="S") -> VideoTrackingData:
    parsed = {}
    timestamps_ns = None
    for aid, df in tracking.items():
        n = df.shape[0]
        parsed[aid] = pd.DataFrame({
            "frame": np.arange(n),
            "center_x": df["x"].to_numpy(dtype=np.float64),
            "center_y": df["y"].to_numpy(dtype=np.float64),
        })
        if timestamps_ns is None:
            timestamps_ns = (df["t"].to_numpy(dtype=np.float64) * 1e9).astype(np.int64)
    return VideoTrackingData(
        animal_id=list(tracking.keys())[0],
        session_id=session_id,
        parsed_data=parsed,
        timestamps=timestamps_ns,
    )


@pytest.fixture(scope="module")
def results():
    tr = {"A": _xy(9000, 1), "B": _xy(9000, 2)}
    ks = SimpleNamespace(
        ks_ids=[0, 1],
        spike_times_by_cell=[_spikes(tr["A"], (40, 40), 10),
                             _spikes(tr["B"], (40, 40), 11)],
    )
    return compute_social_place_fields(
        ks, _video_tracking(tr), _StubSync(), focal_animal="A",
        target_animals=["A", "B"], pixels_per_cm=None,
        bin_size_cm=5.0, smoothing_sigma_cm=5.0,
        speed_filter_subject="none", n_shuffles=30, min_n_spikes=20,
        use_quality_cells=False, arena_bounds=ARENA, seed=0,
    )


def test_max_skaggs_ranking(results):
    ranked = _max_skaggs_by_cluster(results)
    assert len(ranked) == 2
    # Descending by max bits/spike.
    assert ranked[0][1] >= ranked[1][1]


def test_write_outputs_creates_artifacts(results, tmp_path):
    out_dir = tmp_path / "spf"
    _write_outputs(results, out_dir, top_n=5)
    assert (out_dir / "results.pkl").exists()
    assert (out_dir / "summary.png").exists()
    assert (out_dir / "rate_map_grids.pdf").exists()
    with open(out_dir / "results.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert loaded.parameters["class_label"] == "target_position"
    assert loaded.parameters["focal_animal"] == "A"
