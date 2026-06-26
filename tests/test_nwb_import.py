"""
Tests for the NWB importer (ingestion/nwb_import.py).

Builds a tiny in-memory NWBFile (units + interval tables + a SpatialSeries) and
checks that the readers emit the pipeline's native dataclasses satisfying their
contracts. Also exercises the on-disk round-trip via ``load_nwb_session`` and
confirms the imported KilosortData feeds the LDA decoding core.

Skipped entirely when pynwb is not installed, so the existing mock-data suite is
unaffected on environments without NWB dependencies.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("pynwb")

from pynwb import NWBFile, NWBHDF5IO  # noqa: E402
from pynwb.behavior import Position, SpatialSeries  # noqa: E402
from pynwb.epoch import TimeIntervals  # noqa: E402
from pynwb.file import Subject  # noqa: E402

from ingestion.nwb_import import (  # noqa: E402
    IdentitySyncManager,
    load_nwb_session,
    nwb_to_behavioral_events,
    nwb_to_kilosort_data,
    nwb_to_tracking_data,
)

N_UNITS = 6
RNG = np.random.default_rng(0)


def _build_nwbfile(with_tracking: bool = True) -> NWBFile:
    """A minimal but structurally faithful reward-competition-style NWB file."""
    nwb = NWBFile(
        session_description="unit-test session",
        identifier="nwb-import-test",
        session_start_time=datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc),
        subject=Subject(subject_id="mouse_test", species="Mus musculus", sex="M"),
    )

    # --- units: cluster_id + quality columns, spike times in seconds ---------
    nwb.add_unit_column("cluster_id", "kilosort cluster id")
    nwb.add_unit_column("quality", "curation label")
    for u in range(N_UNITS):
        n_sp = 50 + 10 * u
        spikes = np.sort(RNG.uniform(0.0, 100.0, size=n_sp))
        quality = "good" if u % 2 == 0 else "unsorted"
        nwb.add_unit(spike_times=spikes, cluster_id=10 + u, quality=quality)

    # --- interval tables: cs_onsets / us_deliveries --------------------------
    cs = TimeIntervals(name="cs_onsets", description="conditioned stimulus onsets")
    for t in [10.0, 25.0, 40.0, 55.0, 70.0, 85.0]:
        cs.add_row(start_time=t, stop_time=t + 5.0)
    nwb.add_time_intervals(cs)

    us = TimeIntervals(name="us_deliveries", description="reward deliveries")
    for t in [12.0, 27.0, 42.0, 57.0, 72.0]:
        us.add_row(start_time=t, stop_time=t)
    nwb.add_time_intervals(us)

    # --- optional position / tracking ----------------------------------------
    if with_tracking:
        n_frames = 200
        ts = np.linspace(0.0, 100.0, n_frames)
        xy = np.column_stack([RNG.uniform(0, 640, n_frames),
                              RNG.uniform(0, 480, n_frames)])
        ss = SpatialSeries(
            name="mouse_center", data=xy, reference_frame="(0,0) top-left",
            timestamps=ts,
        )
        pos = Position(spatial_series=ss)
        mod = nwb.create_processing_module("behavior", "behavioral data")
        mod.add(pos)

    return nwb


@pytest.fixture
def nwbfile():
    return _build_nwbfile(with_tracking=True)


# ---------------------------------------------------------------------------
# KilosortData
# ---------------------------------------------------------------------------

class TestUnitsToKilosort:
    def test_alignment_contract(self, nwbfile):
        ks = nwb_to_kilosort_data(nwbfile, "mouse_test", "20200101")
        n = len(ks.ks_ids)
        assert n == N_UNITS
        assert len(ks.spike_times_by_cell) == n
        assert len(ks.channel) == n
        assert len(ks.to_load) == n
        assert ks.cell_numbers.shape == (n, 2)

    def test_spike_times_in_seconds(self, nwbfile):
        ks = nwb_to_kilosort_data(nwbfile, "mouse_test", "20200101")
        for st in ks.spike_times_by_cell:
            assert st.dtype == np.float64
            assert st.max() <= 100.0  # session is 100 s, not sample indices
        assert ks.duration_seconds > 0
        assert ks.metadata["source"] == "nwb"
        assert ks.metadata["n_clusters"] == N_UNITS

    def test_cluster_ids_from_column(self, nwbfile):
        ks = nwb_to_kilosort_data(nwbfile, "mouse_test", "20200101")
        assert ks.ks_ids == [10, 11, 12, 13, 14, 15]

    def test_quality_filter(self, nwbfile):
        ks = nwb_to_kilosort_data(nwbfile, "m", "s", quality_filter="good")
        assert len(ks.ks_ids) == 3  # units 0,2,4
        assert ks.ks_ids == [10, 12, 14]

    def test_quality_filter_no_match_keeps_all(self, nwbfile):
        ks = nwb_to_kilosort_data(nwbfile, "m", "s", quality_filter="nonexistent")
        assert len(ks.ks_ids) == N_UNITS

    def test_quality_filter_feeds_firing_metrics(self, nwbfile):
        # The decoding stack recomputes metrics from spike times; make sure that
        # path works on imported data.
        ks = nwb_to_kilosort_data(nwbfile, "m", "s")
        ids, sts = ks.get_filtered_cells_spike_times(
            min_firing_rate=0.0, min_presence_ratio=0.0, max_cv_isi=1e9
        )
        assert len(ids) == len(sts) == N_UNITS


# ---------------------------------------------------------------------------
# BehavioralEventsData
# ---------------------------------------------------------------------------

class TestIntervalsToEvents:
    def test_events_collected_and_synced(self, nwbfile):
        ev = nwb_to_behavioral_events(nwbfile, "20200101")
        assert ev.synchronized is True
        assert len(ev.events_data) == 11  # 6 cs + 5 us
        for col in ("type", "ts_start", "ts_end", "ts_start_ephys", "ts_end_ephys"):
            assert col in ev.events_data.columns
        assert set(ev.events_data["type"]) == {"cs_onsets", "us_deliveries"}

    def test_ephys_times_in_seconds_and_ns_consistent(self, nwbfile):
        ev = nwb_to_behavioral_events(nwbfile, "20200101")
        df = ev.events_data
        # ns column is seconds * 1e9
        np.testing.assert_allclose(
            df["ts_start"].to_numpy() / 1e9, df["ts_start_ephys"].to_numpy(), rtol=1e-6
        )
        assert df["ts_start_ephys"].max() <= 100.0

    def test_sorted_by_time(self, nwbfile):
        ev = nwb_to_behavioral_events(nwbfile, "20200101")
        t = ev.events_data["ts_start_ephys"].to_numpy()
        assert np.all(np.diff(t) >= 0)

    def test_empty_when_no_intervals(self):
        nwb = _build_nwbfile(with_tracking=False)
        # remove interval tables
        nwb.intervals.clear()
        ev = nwb_to_behavioral_events(nwb, "s")
        assert len(ev.events_data) == 0
        assert "ts_start_ephys" in ev.events_data.columns


# ---------------------------------------------------------------------------
# VideoTrackingData
# ---------------------------------------------------------------------------

class TestPositionToTracking:
    def test_tracking_parsed(self, nwbfile):
        tr = nwb_to_tracking_data(nwbfile, "mouse_test", "20200101")
        assert "mouse_center" in tr.parsed_data
        df = tr.parsed_data["mouse_center"]
        for col in ("frame", "center_x", "center_y", "ephys_timestamps"):
            assert col in df.columns
        assert tr.synchronized is True
        assert tr.ephys_timestamps is not None

    def test_no_tracking_is_graceful(self):
        nwb = _build_nwbfile(with_tracking=False)
        tr = nwb_to_tracking_data(nwb, "m", "s")
        assert tr.parsed_data == {}
        assert tr.synchronized is False


# ---------------------------------------------------------------------------
# IdentitySyncManager
# ---------------------------------------------------------------------------

class TestIdentitySync:
    def test_identity_conversion(self):
        sync = IdentitySyncManager("s")
        assert sync.slope == 1.0 and sync.intercept == 0.0
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sync.convert_behavior_to_ephys(x), x)
        np.testing.assert_array_equal(sync.convert_ephys_to_behavior(x), x)


# ---------------------------------------------------------------------------
# End-to-end: on-disk round-trip + decode core integration
# ---------------------------------------------------------------------------

class TestLoadNwbSession:
    def test_roundtrip_and_decode(self, tmp_path, nwbfile):
        path = tmp_path / "test.nwb"
        with NWBHDF5IO(str(path), mode="w") as io:
            io.write(nwbfile)

        ks, events, tracking, sync = load_nwb_session(path)

        # ids derived from subject / session_start_time
        assert ks.animal_id == "mouse_test"
        assert len(ks.spike_times_by_cell) == len(ks.ks_ids) == N_UNITS
        assert events.synchronized and "ts_start_ephys" in events.events_data.columns
        assert "mouse_center" in tracking.parsed_data
        assert isinstance(sync, IdentitySyncManager)

        # imported dataclasses feed the LDA decoding core
        from ephys._lda_decoding import run_population_per_cell_decode

        cs = events.events_data.query("type == 'cs_onsets'")["ts_start_ephys"].to_numpy()
        us = events.events_data.query("type == 'us_deliveries'")["ts_start_ephys"].to_numpy()
        event_times = np.concatenate([cs, us])
        labels = np.array(["cs"] * len(cs) + ["us"] * len(us))

        _, ok_ids, accs = run_population_per_cell_decode(
            spike_times_list=list(ks.spike_times_by_cell),
            cluster_ids=list(ks.ks_ids),
            event_times=event_times,
            labels=labels,
            time_window=(-1.0, 1.0),
            time_bin_size=0.5,
            cv_folds=3,
            min_events_per_class=3,
        )
        assert len(ok_ids) > 0
        assert len(accs) == len(ok_ids)
