"""Mock-data tests for video.territory (no network share needed)."""

import numpy as np
import pandas as pd
import pytest

from video.territory import (
    UNCLAIMED,
    build_territory_map_loo,
    compute_territory_map,
    dwell_time_maps,
    exclusivity_map,
    label_positions,
    signed_boundary_distance,
    territory_stability,
)

BOUNDS = (0.0, 100.0, 0.0, 100.0)
HOMES = {"A": (25.0, 50.0), "B": (75.0, 25.0), "C": (75.0, 75.0)}


def _walk(home, n=6000, dt=0.1, spread=8.0, seed=0, t0=0.0):
    rng = np.random.default_rng(seed)
    xy = rng.normal(home, spread, size=(n, 2))
    xy = np.clip(xy, 0.5, 99.5)
    return pd.DataFrame({"t": t0 + np.arange(n) * dt, "x": xy[:, 0], "y": xy[:, 1]})


def _chunk(seed=0, t0=0.0, n=6000):
    return {name: _walk(h, n=n, seed=seed + i, t0=t0) for i, (name, h) in enumerate(HOMES.items())}


def _tmap(**kw):
    kw.setdefault("bounds", BOUNDS)
    kw.setdefault("bins", 20)
    kw.setdefault("min_tracked_sec", 10.0)
    return compute_territory_map(_chunk(), **kw)


class TestDwell:
    def test_dwell_is_seconds_not_frames(self):
        df = pd.DataFrame({"t": np.arange(100) * 0.5, "x": 10.0, "y": 10.0})
        xe = ye = np.linspace(0, 100, 11)
        h = dwell_time_maps({"A": df}, xe, ye, max_gap_sec=10.0)["A"]
        # 99 intervals of 0.5 s (the last sample has no successor)
        assert h.sum() == pytest.approx(99 * 0.5)

    def test_gap_is_capped(self):
        df = pd.DataFrame({"t": [0.0, 0.1, 500.0, 500.1], "x": 10.0, "y": 10.0})
        xe = ye = np.linspace(0, 100, 11)
        h = dwell_time_maps({"A": df}, xe, ye, max_gap_sec=0.5)["A"]
        assert h.sum() == pytest.approx(0.1 + 0.5 + 0.1)

    def test_nan_positions_are_skipped(self):
        df = pd.DataFrame({"t": np.arange(10) * 0.1, "x": [np.nan] * 5 + [10.0] * 5, "y": 10.0})
        xe = ye = np.linspace(0, 100, 11)
        assert dwell_time_maps({"A": df}, xe, ye)["A"].sum() == pytest.approx(0.4)


class TestOwnerMap:
    def test_each_animal_owns_its_home(self):
        tm = _tmap()
        for name, (hx, hy) in HOMES.items():
            lab = label_positions(tm, name, [hx], [hy])
            assert lab["own"].iloc[0], name
            assert lab["owner"].iloc[0] == name

    def test_unvisited_corner_is_unclaimed(self):
        tm = _tmap(min_dwell_sec=2.0)
        assert tm.owner[0, 0] == UNCLAIMED
        assert not label_positions(tm, "A", [0.1], [0.1])["foreign"].iloc[0]

    def test_normalisation_stops_long_tracking_dominating(self):
        # A is tracked 5x longer than B but both live in the same place: with
        # normalisation neither should win by tracking length alone.
        chunk = {"A": _walk((50, 50), n=30000, seed=1), "B": _walk((50, 50), n=6000, seed=2)}
        tm = compute_territory_map(chunk, bounds=BOUNDS, bins=10, min_tracked_sec=10.0,
                                   normalize=True)
        frac = tm.territory_fraction()
        assert min(frac["A"], frac["B"]) > 0.0
        raw = compute_territory_map(chunk, bounds=BOUNDS, bins=10, min_tracked_sec=10.0,
                                    normalize=False)
        assert raw.territory_fraction()["B"] == 0.0

    def test_excludes_barely_tracked_and_stationary_animals(self):
        chunk = _chunk()
        chunk["D"] = _walk((50, 50), n=20, seed=9)                     # barely tracked
        chunk["E"] = pd.DataFrame({"t": np.arange(6000) * 0.1, "x": 40.0, "y": 40.0})
        tm = compute_territory_map(chunk, bounds=BOUNDS, bins=20, min_tracked_sec=10.0,
                                   min_spread=1.0)
        assert "D" in tm.excluded and "E" in tm.excluded
        assert "E" not in tm.animal_ids
        assert "near-stationary" in tm.excluded["E"]

    def test_needs_two_animals(self):
        with pytest.raises(ValueError, match="at least|>=2"):
            compute_territory_map({"A": _walk((50, 50))}, bounds=BOUNDS, bins=10,
                                  min_tracked_sec=10.0)


class TestSignedDistance:
    def test_sign_and_monotone_into_territory(self):
        tm = _tmap()
        d = signed_boundary_distance(tm, "A")
        assert (d[tm.owner == 0] > 0).all()
        assert (d[tm.owner != 0] < 0).all()
        # the animal's centre is deeper inside than a bin on its edge
        lab = label_positions(tm, "A", [HOMES["A"][0], 5.0], [HOMES["A"][1], 50.0])
        assert lab["signed_dist"].iloc[0] > 0

    def test_labels_use_the_named_animal_not_a_neighbour(self):
        # HZ-API-005 analogue: "own" for B must differ from "own" for A at the same spot.
        tm = _tmap()
        la = label_positions(tm, "A", [HOMES["A"][0]], [HOMES["A"][1]])
        lb = label_positions(tm, "B", [HOMES["A"][0]], [HOMES["A"][1]])
        assert la["own"].iloc[0] and not lb["own"].iloc[0]
        assert lb["foreign"].iloc[0]
        assert la["signed_dist"].iloc[0] > 0 > lb["signed_dist"].iloc[0]

    def test_unmapped_animal_has_no_own_labels(self):
        tm = _tmap()
        lab = label_positions(tm, "Z", [25.0], [50.0])
        assert not lab["own"].iloc[0]
        assert np.isnan(lab["signed_dist"].iloc[0])

    def test_off_grid_points_are_flagged(self):
        tm = _tmap()
        lab = label_positions(tm, "A", [-50.0, 250.0], [50.0, 50.0])
        assert not lab["in_grid"].any()
        assert lab["owner"].isna().all()


class TestLeaveOneChunkOut:
    def _chunks(self):
        return {f"c{i}": _chunk(seed=10 * i, t0=i * 1000.0) for i in range(3)}

    def test_excludes_test_chunk_from_training(self):
        chunks = self._chunks()
        tm = build_territory_map_loo(chunks, "c1", bounds=BOUNDS, bins=20,
                                     min_tracked_sec=10.0)
        assert tm.parameters["test_chunk"] == "c1"
        assert tm.parameters["train_chunks"] == ["c0", "c2"]
        # dwell from two training chunks only: 2 x 600 s per animal
        assert tm.dwell_sec["A"].sum() == pytest.approx(2 * 599.9, rel=0.01)

    def test_refuses_leaky_train_set(self):
        with pytest.raises(ValueError, match="leakage"):
            build_territory_map_loo(self._chunks(), "c1", train_chunks=["c0", "c1"],
                                    bounds=BOUNDS, bins=20, min_tracked_sec=10.0)

    def test_unknown_test_chunk(self):
        with pytest.raises(KeyError):
            build_territory_map_loo(self._chunks(), "nope", bounds=BOUNDS)

    def test_single_chunk_has_nothing_to_train_on(self):
        with pytest.raises(ValueError, match="no training chunks"):
            build_territory_map_loo({"c0": _chunk()}, "c0", bounds=BOUNDS)


class TestStability:
    def test_same_world_is_stable(self):
        a = compute_territory_map(_chunk(seed=1), bounds=BOUNDS, bins=20, min_tracked_sec=10.0)
        b = compute_territory_map(_chunk(seed=50), bounds=BOUNDS, bins=20, min_tracked_sec=10.0)
        s = territory_stability(a, b)
        assert s["owner_agreement"] > 0.9
        assert s["adjusted_rand_index"] > 0.8
        assert all(v > 0.7 for v in s["jaccard_by_animal"].values())

    def test_moved_animals_are_unstable(self):
        a = compute_territory_map(_chunk(seed=1), bounds=BOUNDS, bins=20, min_tracked_sec=10.0)
        swapped = {"A": _walk(HOMES["B"], seed=3), "B": _walk(HOMES["C"], seed=4),
                   "C": _walk(HOMES["A"], seed=5)}
        b = compute_territory_map(swapped, bounds=BOUNDS, bins=20, min_tracked_sec=10.0)
        s = territory_stability(a, b)
        assert s["owner_agreement"] < 0.2

    def test_grid_mismatch_raises(self):
        a = compute_territory_map(_chunk(), bounds=BOUNDS, bins=20, min_tracked_sec=10.0)
        b = compute_territory_map(_chunk(), bounds=BOUNDS, bins=10, min_tracked_sec=10.0)
        with pytest.raises(ValueError, match="same grid"):
            territory_stability(a, b)


class TestExclusivity:
    def test_bounded_and_highest_at_own_home(self):
        tm = _tmap()
        ex = exclusivity_map(tm, "A")
        assert np.nanmin(ex) >= 0 and np.nanmax(ex) <= 1
        lab = label_positions(tm, "A", [HOMES["A"][0], HOMES["B"][0]], [HOMES["A"][1], HOMES["B"][1]])
        assert lab["exclusivity"].iloc[0] > 0.6 > lab["exclusivity"].iloc[1]

    def test_shares_sum_to_one_across_animals(self):
        tm = _tmap()
        tot = sum(np.nan_to_num(exclusivity_map(tm, a)) for a in tm.animal_ids)
        used = ~np.isnan(exclusivity_map(tm, "A"))
        assert np.allclose(tot[used], 1.0)

    def test_unused_bins_are_nan_not_zero(self):
        tm = _tmap()
        assert np.isnan(exclusivity_map(tm, "A")[0, 0])

    def test_does_not_flip_when_two_animals_swap_rank(self):
        # two animals share a home with nearly equal occupancy: the owner map's winner is
        # arbitrary, the exclusivity of each stays near one half
        chunk = {"A": _walk((50, 50), seed=1), "B": _walk((50, 50), seed=2), "C": _walk((10, 10), seed=3)}
        tm = compute_territory_map(chunk, bounds=BOUNDS, bins=10, min_tracked_sec=10.0)
        centre = label_positions(tm, "A", [50.0], [50.0])["exclusivity"].iloc[0]
        assert 0.3 < centre < 0.7

    def test_unmapped_animal_is_nan(self):
        tm = _tmap()
        assert np.isnan(label_positions(tm, "Z", [25.0], [50.0])["exclusivity"].iloc[0])
        with pytest.raises(KeyError):
            exclusivity_map(tm, "Z")
