"""
Tests for the Greyware (SysClock-fit) sync path (ingestion/greyware_sync.py and
the ``sync_method='greyware'`` branch of ``DataSyncManager``).

Mock data only: the saved merge-day fits are checked against the coefficient
table in ingestion/Sync/README.md, and a tiny synthetic ``.rec`` exercises
``greyware_time.fit_recording``.
"""

import numpy as np
import pytest

import ingestion.ephys_sync as ephys_sync
import ingestion.multi_animal_session as mas
from ingestion.ephys_sync import DataSyncManager
from ingestion.greyware_sync import (
    fit_to_linear,
    load_greyware_fit,
    pulse_residuals,
)
from ingestion.Sync.greyware_time import fit_recording, greyware_time

# ingestion/Sync/README.md "Merge day coefficients": T_ref has the 118.5 ms
# latency already subtracted.
README_TABLE = {
    "20260929_095542": (261075, 1790690149.397185, 29999.77321),
    "20260929_150139": (165676, 1790708503.174497, 29999.788746),
    "20260929_201052": (141083, 1790727054.616411, 29999.765906),
}


class _StubDSM:
    def __init__(self, recording_id, sync_method="greyware", latency=0.1185):
        self.recording_id = recording_id
        self.sync_method = sync_method
        self.greyware_latency_s = latency
        self.animal_id = "4678"
        self.session_id = recording_id

    def get_dio_path(self, channel=1):
        return None  # -> load_ephys_sync raises, pulse check unavailable

    def get_pulse_log_path(self):
        return None


# ---------------------------------------------------------------------------
# Saved fits
# ---------------------------------------------------------------------------

class TestSavedFits:
    @pytest.mark.parametrize("rec", sorted(README_TABLE))
    def test_reproduces_readme_table(self, rec):
        n_ref, t_ref_table, rate = README_TABLE[rec]
        fit = load_greyware_fit(rec)
        assert fit["n_ref"] == n_ref
        assert fit["rate_hz"] == pytest.approx(rate, abs=1e-6)
        assert greyware_time(fit, [n_ref])[0] == pytest.approx(t_ref_table, abs=1e-6)

    def test_unknown_recording_raises_clearly(self):
        with pytest.raises(KeyError, match="No Greyware fit"):
            load_greyware_fit("20261006_101726")

    @pytest.mark.parametrize("rec", sorted(README_TABLE))
    def test_linear_form_matches_equation(self, rec):
        fit = load_greyware_fit(rec)
        slope, intercept = fit_to_linear(fit, 0.1185)
        counts = np.random.default_rng(0).integers(fit["n_ref"], 2 ** 31, 50)
        np.testing.assert_allclose(slope * (counts / 30000.0) + intercept,
                                   greyware_time(fit, counts, 0.1185), atol=1e-6, rtol=0)

    def test_latency_shifts_intercept_only(self):
        fit = load_greyware_fit("20260929_095542")
        s1, i1 = fit_to_linear(fit, 0.1185)
        s2, i2 = fit_to_linear(fit, 0.1144)
        assert s1 == s2
        assert i2 - i1 == pytest.approx(0.0041, abs=1e-6)


# ---------------------------------------------------------------------------
# fit_recording on a synthetic .rec
# ---------------------------------------------------------------------------

def _write_fake_rec(path, n_packets, rate, t0_ns, n0, rng):
    header = (
        '<?xml version="1.0"?>\n<Configuration>\n'
        '<GlobalConfiguration systemTimeAtCreation="1790690148404"/>\n'
        '<HardwareConfiguration numChannels="0">\n'
        '<Device name="Controller_DIO" numBytes="1" available="1" packetOrderPreference="10"/>\n'
        '<Device name="Multiplexed" numBytes="8" available="1" packetOrderPreference="20"/>\n'
        '<Device name="SysClock" numBytes="8" available="1" packetOrderPreference="30"/>\n'
        '</HardwareConfiguration>\n</Configuration>\n'
    ).encode("latin1")
    counts = n0 + np.arange(n_packets, dtype=np.int64)
    lateness = 0.1185e9 + rng.exponential(5e5, n_packets)  # late, noisy stamps
    stamps = (t0_ns + (counts - n0) * 1e9 / rate + lateness).astype(np.uint64)
    pkt = np.zeros((n_packets, 22), dtype=np.uint8)
    pkt[:, 0] = 0x55
    pkt[:, 10:18] = stamps.view(np.uint8).reshape(-1, 8)
    pkt[:, 18:22] = counts.astype("<u4").view(np.uint8).reshape(-1, 4)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(pkt.tobytes())


def test_fit_recording_recovers_rate(tmp_path):
    rate, t0_ns, n0 = 29999.78, 1_790_690_149_000_000_000, 261075
    rec = tmp_path / "fake.rec"
    _write_fake_rec(rec, 200_000, rate, t0_ns, n0, np.random.default_rng(1))
    fit = fit_recording(str(rec), n_blocks=40, block_packets=2000)
    assert fit["rate_hz"] == pytest.approx(rate, abs=0.1)
    assert fit["n_ref"] == n0
    # Stamp lateness is 118.5 ms + mean noise (0.5 ms); the latency constant removes the bulk.
    assert greyware_time(fit, [n0])[0] - t0_ns / 1e9 == pytest.approx(0.0005, abs=0.001)


# ---------------------------------------------------------------------------
# DataSyncManager dispatch
# ---------------------------------------------------------------------------

class TestDataSyncManagerGreyware:
    def test_builds_without_pulses(self):
        s = DataSyncManager(_StubDSM("20260929_095542"))
        assert s.method == "greyware"
        assert s.mapping["source"] == "greyware_sysclock"
        assert s.mapping["r_squared"] is None and s.mapping["n_matches"] is None
        assert s.mapping["pulse_check"]["ok"] is False
        assert s.ephys_sync is None
        assert s.slope == pytest.approx(30000 / 29999.77321)
        assert "greyware" in repr(s)

    def test_conversion_round_trip(self):
        s = DataSyncManager(_StubDSM("20260929_150139"))
        e = np.array([10.0, 5000.0, 18000.0])
        np.testing.assert_allclose(s.convert_behavior_to_ephys(s.convert_ephys_to_behavior(e)), e,
                                   atol=1e-6)
        n = e * 30000.0
        np.testing.assert_allclose(s.convert_ephys_to_behavior(e),
                                   greyware_time(load_greyware_fit("20260929_150139"), n),
                                   atol=1e-6, rtol=0)

    def test_unknown_recording_raises(self):
        with pytest.raises(KeyError, match="No Greyware fit"):
            DataSyncManager(_StubDSM("20261006_101726"))

    def test_plot_without_pulse_match_raises(self):
        s = DataSyncManager(_StubDSM("20260929_095542"))
        with pytest.raises(ValueError, match="no pulse match"):
            s.plot_sync_results()

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown sync_method"):
            DataSyncManager(_StubDSM("20260929_095542", sync_method="bogus"))

    def test_pulse_check_runs_when_pulses_load(self, monkeypatch):
        dsm = _StubDSM("20260929_095542")
        slope, intercept = fit_to_linear(load_greyware_fit(dsm.recording_id), 0.1185)
        rng = np.random.default_rng(2)
        tse = np.cumsum(rng.uniform(5, 7, 60)) + 20.0
        tsb = slope * tse + intercept + 0.010  # pulses 10 ms behind the Greyware map
        monkeypatch.setattr(ephys_sync, "load_ephys_sync",
                            lambda dm, ch: (tse, tsb, float(tsb[0])))
        s = DataSyncManager(dsm)
        pc = s.mapping["pulse_check"]
        assert pc["ok"] is True
        assert pc["n_matches"] >= 50
        assert pc["median_offset_s"] == pytest.approx(-0.010, abs=1e-6)
        assert s.ephys_sync is tse
        assert s.slope == slope  # Greyware map wins; pulses are only a check


def test_default_method_is_pulses(monkeypatch):
    class _PulseDSM(_StubDSM):
        pass

    dsm = _PulseDSM("20251216_094334", sync_method="pulses")
    tse = np.cumsum(np.random.default_rng(3).uniform(5, 7, 40))
    monkeypatch.setattr(ephys_sync, "load_ephys_sync",
                        lambda dm, ch: (tse, tse * 1.00001 + 100.0, 100.0 + tse[0]))
    s = DataSyncManager(dsm)
    assert s.method == "pulses"
    assert "source" not in s.mapping
    assert s.slope == pytest.approx(1.00001, abs=1e-9)
    assert s.mapping["r_squared"] > 0.999

    # A DSM without the property at all (older stand-ins) also means pulses.
    class _Bare:
        animal_id = session_id = "x"
    s2 = DataSyncManager(_Bare())
    assert s2.method == "pulses"


# ---------------------------------------------------------------------------
# pulse_residuals
# ---------------------------------------------------------------------------

def test_pulse_residuals_offsets():
    e = np.linspace(10, 1000, 20)
    base = {"matched_ephys": e, "matched_behavior": 1.0000073 * e + 50.0, "slope": 1.0000073}
    assert pulse_residuals(1.0000073, 50.0, base)["median_offset_s"] == pytest.approx(0.0, abs=1e-9)
    shifted = dict(base, matched_behavior=base["matched_behavior"] - 1.0)
    out = pulse_residuals(1.0000073, 50.0, shifted)
    assert out["median_offset_s"] == pytest.approx(1.0, abs=1e-9)
    assert out["max_abs_offset_s"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# MultiAnimalSession.verify_sync_consistency
# ---------------------------------------------------------------------------

def test_verify_sync_consistency_accepts_greyware(monkeypatch):
    monkeypatch.setattr(mas, "DataSyncManager", lambda dsm, dio_channel=1: DataSyncManager(dsm))
    sess = mas.MultiAnimalSession.__new__(mas.MultiAnimalSession)
    sess.animal_ids = ["4678", "4679"]
    sess.dsm_by_animal = {a: _StubDSM("20260929_095542") for a in sess.animal_ids}
    sess.dio_channel = 1
    sess._sync = DataSyncManager(_StubDSM("20260929_095542"))
    report = sess.verify_sync_consistency()
    assert all(r["passed"] for r in report.values())
    assert all(r["r_squared"] is None for r in report.values())


def test_windowed_pulse_match_ignores_far_pulses():
    from ingestion.greyware_sync import windowed_pulse_match
    rng = np.random.default_rng(4)
    tse = np.cumsum(rng.uniform(5, 10, 80)) + 30.0
    slope, intercept = 1.0000073, 1.79e9
    true_b = slope * tse + intercept - 0.020
    # A week of earlier pulses that would be scanned first by an unbounded search,
    # including an exact copy of the true interval pattern 3 days earlier.
    decoy = true_b - 3 * 86400.0
    filler = np.cumsum(rng.uniform(5, 10, 2000)) + true_b[0] - 7 * 86400.0
    tsb = np.sort(np.concatenate([filler, decoy, true_b]))
    mp = windowed_pulse_match(slope, intercept, tse, tsb)
    out = pulse_residuals(slope, intercept, mp)
    assert out["median_offset_s"] == pytest.approx(0.020, abs=1e-6)
    assert out["n_matches"] >= 70
