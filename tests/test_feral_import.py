"""Tests for ``ingestion/feral_import.py`` (synthetic FERAL CSV + browser JSON, no share)."""

import fnmatch
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ingestion import feral_import as fi
from ingestion.nwb_import import IdentitySyncManager

MOVIE = "cohort7_20251216_0959"           # starts 10:00:00 local
MOVIE_T0 = datetime(2025, 12, 16, 10, 0, 0)
RAT = {1: "rat4620", 2: "rat4631", 4: "rat4613", 7: "rat4615", 11: "rat4635"}


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _write_inputs(tmp_path, bouts, *, extra_json=(), drop_csv=(), gt=(), movie=MOVIE, t0=MOVIE_T0):
    """Write a FERAL CSV and one browser JSON from ``(tracklet, beh, role, f0, f1, partner)`` rows.

    ``extra_json`` rows go only into the JSON; ``drop_csv`` indexes are left
    out of the CSV.
    """
    feral_dir = tmp_path / "cohort7" / "FERAL"
    browser = tmp_path / "cohort7" / "movie_browser" / "behavior_feral"
    feral_dir.mkdir(parents=True)
    browser.mkdir(parents=True)
    rows = []
    for i, (tr, beh, role, f0, f1, partner) in enumerate(bouts):
        if i in drop_csv:
            continue
        rows.append({"tracklet_id": tr, "rat_id": RAT[tr], "behavior": beh,
                     "start_time": _fmt(t0 + timedelta(seconds=f0 / fi.FPS)),
                     "end_time": _fmt(t0 + timedelta(seconds=f1 / fi.FPS))})
    # Every tracklet appears at least once so the id map is complete.
    seen = {r["tracklet_id"] for r in rows}
    for tr in sorted(set(RAT) - seen):
        rows.append({"tracklet_id": tr, "rat_id": RAT[tr], "behavior": "SN",
                     "start_time": _fmt(t0 + timedelta(seconds=1500)),
                     "end_time": _fmt(t0 + timedelta(seconds=1501))})
        bouts = list(bouts) + [(tr, "SN", "initiator", 1500 * fi.FPS, 1501 * fi.FPS, 1 if tr != 1 else 11)]
    pd.DataFrame(rows).sort_values("start_time").to_csv(feral_dir / "feral_events.csv", index=False)
    doc = {"movie": movie, "best": [list(b) for b in bouts] + [list(b) for b in extra_json],
           "window": [0, 71999], "gt": [list(g) for g in gt]}
    (browser / f"{movie}.json").write_text(json.dumps(doc))
    return feral_dir / "feral_events.csv", browser.parent


def _run(tmp_path, bouts, **kw):
    csv, browser = _write_inputs(tmp_path, bouts, **kw)
    return fi.import_feral(csv, browser)


def _dyad(events, a, b):
    return events[((events.tracklet_initiator == a) & (events.tracklet_victim == b))
                  | ((events.tracklet_initiator == b) & (events.tracklet_victim == a))]


# --- Stage 0: join -----------------------------------------------------------

class TestJoin:
    def test_times_are_utc_ns_of_first_and_last_frame(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 40, 80, 4)])
        b = r.bouts[(r.bouts.tracklet == 2) & (r.bouts.behavior == "EC")].iloc[0]
        # 10:00:01 EST == 15:00:01 UTC
        assert b.ts_start == pd.Timestamp("2025-12-16 15:00:01", tz="UTC").value
        assert b.ts_end == pd.Timestamp("2025-12-16 15:00:02", tz="UTC").value
        assert not b.duration_mismatch

    def test_animal_ids_are_normalised(self, tmp_path):
        r = _run(tmp_path, [(11, "EC", "", 40, 80, 2)])
        assert r.report["stage0_join"]["tracklet_to_animal"]["11"] == "rat635"
        assert set(r.events["initiator"]) | set(r.events["victim"]) <= {
            "rat620", "rat631", "rat613", "rat615", "rat635"}

    def test_json_only_group_is_dropped_and_reported(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 40, 80, 4)], extra_json=[(7, "F", "initiator", 100, 140, 2)])
        dropped = r.report["stage0_join"]["json_only_dropped"]
        assert dropped == [{"movie": MOVIE, "tracklet": 7, "behavior": "F", "n": 1}]
        assert not ((r.bouts.tracklet == 7) & (r.bouts.behavior == "F")).any()

    def test_unequal_counts_raise(self, tmp_path):
        bouts = [(2, "EC", "", 40, 80, 4), (2, "EC", "", 200, 240, 4)]
        csv, browser = _write_inputs(tmp_path, bouts, drop_csv={1})
        with pytest.raises(ValueError, match="disagree on bout counts"):
            fi.load_feral_bouts(csv, browser)

    def test_bad_role_raises(self, tmp_path):
        csv, browser = _write_inputs(tmp_path, [(2, "F", "", 40, 80, 4)])
        with pytest.raises(ValueError, match="unexpected role"):
            fi.load_feral_bouts(csv, browser)


# --- Stage 1: pairing ----------------------------------------------------------

class TestPairing:
    def test_reciprocal_bouts_become_one_event_over_the_union(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 200, 4), (4, "EC", "", 110, 220, 2)])
        ev = _dyad(r.events, 2, 4)
        ev = ev[ev.type == "EC"]
        assert len(ev) == 1
        e = ev.iloc[0]
        assert (e.frame_start, e.frame_end) == (100, 220)
        assert e.support == "reciprocal" and e.n_source_bouts == 2
        assert e.initiator == "rat631"              # earlier onset
        assert not e.directed and e.role_basis == "symmetric"
        assert e.iou_pair == pytest.approx(91 / 121)

    def test_directed_roles_follow_feral(self, tmp_path):
        r = _run(tmp_path, [(2, "F", "victim", 100, 200, 4), (4, "F", "initiator", 100, 200, 2)])
        e = r.events[r.events.type == "F"].iloc[0]
        assert (e.initiator, e.victim, e.role_basis) == ("rat613", "rat631", "feral")

    def test_role_conflict_goes_to_the_longer_bout(self, tmp_path):
        r = _run(tmp_path, [(2, "C", "initiator", 100, 150, 4), (4, "C", "initiator", 100, 200, 2)])
        e = r.events[r.events.type == "C"].iloc[0]
        assert (e.initiator, e.role_basis) == ("rat613", "conflict")

    def test_one_sided_victim_bout_makes_the_partner_initiator(self, tmp_path):
        r = _run(tmp_path, [(2, "SN", "victim", 100, 200, 7)])
        e = _dyad(r.events, 2, 7).iloc[0]
        assert (e.initiator, e.victim, e.support) == ("rat615", "rat631", "one_sided")

    def test_partner_answering_with_another_behaviour_is_cross_support(self, tmp_path):
        r = _run(tmp_path, [(2, "F", "initiator", 100, 200, 4), (4, "D", "initiator", 120, 180, 2)])
        sup = r.candidates.set_index("type")["support"]
        assert sup["F"] == "cross_supported" and sup["D"] == "cross_supported"

    def test_matching_is_one_to_one_by_iou(self, tmp_path):
        # A overlaps both of B's bouts; it pairs with the higher-IoU one (100-150),
        # and B's other bout is left with A's activity as cross support.
        r = _run(tmp_path, [(2, "EC", "", 100, 200, 4), (4, "EC", "", 100, 150, 2),
                            (4, "EC", "", 160, 200, 2)])
        c = r.candidates[r.candidates.type == "EC"]
        assert sorted(c.support) == ["cross_supported", "reciprocal"]


# --- Stage 2: overlaps ------------------------------------------------------------

class TestOverlaps:
    def test_higher_priority_label_splits_the_lower_one(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 300, 4), (4, "EC", "", 100, 300, 2),
                            (2, "F", "initiator", 150, 200, 4), (4, "F", "victim", 150, 200, 2)])
        ev = _dyad(r.events, 2, 4).sort_values("frame_start")
        got = list(zip(ev.type, ev.frame_start, ev.frame_end))
        assert got == [("EC", 100, 149), ("F", 150, 200), ("EC", 201, 300)]
        # co_labels lists what was suppressed *inside* a segment; trimming is ``trimmed``.
        assert list(ev.co_labels) == ["", "EC", ""]
        assert list(ev.trimmed) == [True, False, True]
        assert (ev.support == "reciprocal").all()

    def test_short_remnant_is_dropped(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 205, 4), (2, "F", "initiator", 100, 200, 4)])
        ev = _dyad(r.events, 2, 4)
        assert list(ev.type) == ["F"]
        assert r.report["stage2_overlaps"]["suppressed_by_type"] == {"EC": 1}

    def test_short_gap_between_equal_labels_is_bridged(self, tmp_path):
        r = _run(tmp_path, [(2, "F", "initiator", 100, 150, 4), (4, "F", "victim", 155, 200, 2)])
        ev = _dyad(r.events, 2, 4)
        assert list(zip(ev.type, ev.frame_start, ev.frame_end)) == [("F", 100, 200)]
        assert ev.iloc[0].n_source_bouts == 2

    def test_long_gap_is_not_bridged(self, tmp_path):
        r = _run(tmp_path, [(2, "F", "initiator", 100, 150, 4), (4, "F", "victim", 160, 200, 2)])
        assert len(_dyad(r.events, 2, 4)) == 2

    def test_reciprocal_role_conflict_stays_one_event(self, tmp_path):
        r = _run(tmp_path, [(2, "SN", "initiator", 100, 200, 4), (4, "SN", "initiator", 100, 150, 2)])
        assert len(_dyad(r.events, 2, 4)) == 1

    def test_overlap_across_dyads_is_kept_and_counted(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 200, 4), (2, "EC", "", 100, 200, 7)])
        ab, ac = _dyad(r.events, 2, 4), _dyad(r.events, 2, 7)
        assert len(ab) == 1 and len(ac) == 1
        assert ab.iloc[0].n_concurrent_partners == 1 and ac.iloc[0].n_concurrent_partners == 1

    def test_priority_must_rank_every_behaviour(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 200, 4)])
        with pytest.raises(ValueError, match="missing from priority"):
            fi.resolve_overlaps(r.candidates, priority=("F",))

    def test_edge_flag(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 0, 50, 4)])
        assert _dyad(r.events, 2, 4).iloc[0].at_movie_edge


# --- Frame clock --------------------------------------------------------------------

def test_frame_clock_interpolates_and_extrapolates():
    bouts = pd.DataFrame({"movie": ["m", "m"], "f0": [0, 100], "f1": [10, 200],
                          "ts_start": [0, 2_525_000_000], "ts_end": [250_000_000, 5_025_000_000]})
    clock = fi.FrameClock(bouts)
    assert clock("m", [0, 10, 100, 200]).tolist() == [0, 250_000_000, 2_525_000_000, 5_025_000_000]
    assert clock("m", [55])[0] == pytest.approx((250_000_000 + 2_525_000_000) / 2, abs=1)
    assert clock("m", [201])[0] == 5_050_000_000
    assert clock("m", [-1])[0] == -25_000_000


# --- Output + pipeline loading ---------------------------------------------------------

class TestOutput:
    def test_output_name_cannot_match_the_canonical_resolver(self):
        assert not fnmatch.fnmatch(fi.OUTPUT_FILENAME, "*_behavior_event_df*.csv")

    def test_write_load_sync_and_extract_opponents(self, tmp_path):
        bouts = []
        for k in range(3):                     # 631 vs 613: reciprocal EC
            f = 1000 + 400 * k
            bouts += [(2, "EC", "", f, f + 80, 4), (4, "EC", "", f, f + 80, 2)]
        for k in range(2):                     # 631 sniffed by 615: one-sided
            f = 3000 + 400 * k
            bouts += [(2, "SN", "victim", f, f + 80, 7)]
        r = _run(tmp_path, bouts)
        csv_path, report_path = fi.write_feral_events(r, tmp_path / "out")
        assert csv_path.read_bytes()[:3] != b"\xef\xbb\xbf"
        assert json.loads(report_path.read_text())["output"]["n_events"] == len(r.events)

        ev = fi.load_feral_events(csv_path, date="20251216")
        assert ev.session_id == "20251216"
        assert ev.synchronize_with_ephys(IdentitySyncManager())
        starts, _, labels = ev.extract_opponent_labels("631", min_events_per_class=1)
        assert sorted(labels.tolist()).count("rat613") == 3
        assert sorted(labels.tolist()).count("rat615") == 2
        assert starts[0] == pytest.approx(
            pd.Timestamp("2025-12-16 15:00:25", tz="UTC").value / 1e9)

        recip = fi.load_feral_events(csv_path, min_support="reciprocal").events_data
        with_631 = recip[(recip.initiator == "rat631") | (recip.victim == "rat631")]
        assert set(with_631["type"]) == {"EC"} and len(with_631) == 3
        assert len(fi.load_feral_events(csv_path, date="20251217").events_data) == 0
        with pytest.raises(ValueError):
            fi.load_feral_events(csv_path, min_support="strong")

    def test_events_file_override_filter_matches_session_column(self, tmp_path):
        r = _run(tmp_path, [(2, "EC", "", 100, 200, 4)])
        assert r.events["session"].str.contains("_20251216_", regex=False).all()


# --- Validation helpers ------------------------------------------------------------------

def test_perfect_prediction_scores_one_against_gt(tmp_path):
    bouts = [(2, "EC", "", 100, 200, 4), (4, "EC", "", 100, 200, 2),
             (2, "PS", "initiator", 300, 400, 4), (4, "PS", "victim", 300, 400, 2)]
    r = _run(tmp_path, bouts, gt=bouts)
    gt = fi.gt_dyadic_events(r.gt)
    assert len(gt) == 2
    agree = fi.frame_agreement(gt, r.events, [MOVIE])
    for t in ("EC", "PS"):
        assert agree[t]["precision"] == 1.0 and agree[t]["recall"] == 1.0
    assert agree["PS"]["direction_agreement"] == 1.0
    ev = fi.event_agreement(gt, r.events, [MOVIE])
    assert ev["EC"]["f1"] == 1.0 and ev["PS"]["f1"] == 1.0
