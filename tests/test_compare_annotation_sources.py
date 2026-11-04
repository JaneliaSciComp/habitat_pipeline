"""Tests for ``analysis/event_decoding/compare_annotation_sources.py`` (mock data, no share)."""

import numpy as np
import pandas as pd

from analysis.event_decoding import compare_annotation_sources as cas

NS = 1_000_000_000


def _events(rows):
    return pd.DataFrame([{"type": t, "initiator": a, "victim": b,
                          "ts_start": s * NS, "ts_end": e * NS} for t, a, b, s, e in rows])


def test_match_requires_same_pair_and_overlap():
    manual = _events([("EC", "rat631", "rat613", 0, 10), ("EC", "rat631", "rat615", 20, 30)])
    feral = _events([("EC", "rat631", "rat613", 1, 9),      # matches event 0
                     ("EC", "rat631", "rat615", 100, 110),  # right pair, no overlap
                     ("EC", "rat631", "rat617", 20, 30)])   # overlaps, wrong pair
    pairs = cas.match_events(manual, feral)
    assert list(pairs["manual_idx"]) == [0] and list(pairs["feral_idx"]) == [0]


def test_pair_is_unordered_and_initiator_agreement_is_reported():
    manual = _events([("EC", "rat631", "rat613", 0, 10)])
    feral = _events([("EC", "rat613", "rat631", 0, 10)])
    pairs = cas.match_events(manual, feral)
    assert len(pairs) == 1 and not pairs["same_initiator"].iloc[0]


def test_matching_is_one_to_one():
    manual = _events([("EC", "rat631", "rat613", 0, 10)])
    feral = _events([("EC", "rat631", "rat613", 0, 10), ("EC", "rat631", "rat613", 1, 10)])
    assert len(cas.match_events(manual, feral)) == 1


def test_agreement_precision_recall():
    manual = _events([("EC", "a", "b", 0, 10), ("EC", "a", "c", 20, 30)])
    feral = _events([("EC", "a", "b", 0, 10), ("EC", "a", "d", 50, 60),
                     ("EC", "a", "e", 70, 80), ("EC", "a", "f", 90, 99)])
    r = cas.agreement(manual, feral)
    assert r["n_matched"] == 1
    assert r["precision"] == 0.25 and r["recall"] == 0.5


def test_other_event_types_are_ignored():
    manual = _events([("F", "a", "b", 0, 10)])
    feral = _events([("F", "a", "b", 0, 10)])
    assert len(cas.match_events(manual, feral, event_type="EC")) == 0


def test_compare_tables_pairs_cells_by_cluster(tmp_path):
    f = pd.DataFrame({"animal_id": "rat631", "session_id": 1, "cluster_id": [1, 2, 3, 4],
                      "accuracy": [.6, .5, .7, .4], "significant": [True, False, True, False]})
    m = pd.DataFrame({"animal_id": "rat631", "session_id": 20251216, "cluster_id": [1, 2, 3, 4],
                      "accuracy": [.5, .5, .6, .5], "significant": [True, False, False, False]})
    fp, mp = tmp_path / "f.csv", tmp_path / "m.csv"
    f.to_csv(fp, index=False)
    m.to_csv(mp, index=False)
    r = cas.compare_tables(fp, mp, "rat631", "20251216")
    assert r["n_cells"] == 4 and r["n_sig_both"] == 1 and r["n_sig_feral"] == 2
    assert np.isclose(r["jaccard_sig"], 0.5)
