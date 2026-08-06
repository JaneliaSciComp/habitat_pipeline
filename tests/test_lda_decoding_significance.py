"""Rigor-layer tests for the per-cell LDA decoder.

Two things are guarded here:

1. **The original Phase 0 finding** — a naive "accuracy > chance + margin"
   screen over many cells manufactures false positives
   (`scripts/phase0_probe.py::check_loop`, seed=0: 24 cells, 6 truly tuned,
   naive screen flagged 9). BH-FDR correction must not do worse.
2. **The Phase 1.5 finding** — a permutation budget too small for the number
   of cells makes a null result meaningless. `fdr_resolution` must detect
   that, and `null_mode='pooled'` must fix it at the same compute.
"""
import warnings

import numpy as np
import pytest

from ephys._lda_decoding import compute_population_significance, run_population_per_cell_decode
from ephys._stats_utils import fdr_resolution


def _synthetic_opponent_scenario(seed=0, dur=200.0, n_cells=24, n_tuned=6):
    """Mirror scripts/phase0_probe.py::check_loop's synthetic ground truth."""
    rng = np.random.default_rng(seed)
    event_times = np.sort(rng.uniform(5, dur - 5, 60))
    labels = rng.integers(0, 2, size=len(event_times))

    spike_times_list, cluster_ids, tuned_mask = [], [], []
    for c in range(n_cells):
        base = rng.uniform(2, 8)
        spikes = np.sort(rng.uniform(0, dur, int(base * dur)))
        is_tuned = c < n_tuned
        if is_tuned:
            pref = c % 2
            for et, lab in zip(event_times, labels):
                if lab == pref:
                    spikes = np.concatenate(
                        [spikes, rng.uniform(et, et + 1.0, rng.poisson(6))])
        spike_times_list.append(np.sort(spikes))
        cluster_ids.append(c)
        tuned_mask.append(is_tuned)

    return spike_times_list, cluster_ids, event_times, labels, np.array(tuned_mask)


def _decode(spike_times_list, cluster_ids, event_times, labels):
    return run_population_per_cell_decode(
        spike_times_list, cluster_ids, event_times, labels,
        time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
        min_events_per_class=5, progress_every=0,
    )


def _significance(spike_times_list, cluster_ids, event_times, labels, ok_ids, **kw):
    kw.setdefault('n_shuffles', 100)
    kw.setdefault('seed', 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return compute_population_significance(
            spike_times_list, cluster_ids, event_times, labels,
            successful_cluster_ids=ok_ids,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5, **kw,
        )


class TestPermutationSignificanceGuardrail:
    @pytest.mark.slow
    def test_fdr_does_not_exceed_naive_false_positive_count(self):
        stl, cids, event_times, labels, tuned_mask = _synthetic_opponent_scenario(seed=0)
        _, ok_ids, accs = _decode(stl, cids, event_times, labels)
        accs = np.asarray(accs)
        chance = float(max(np.mean(labels == 0), np.mean(labels == 1)))

        naive_flagged = {cid for cid, acc in zip(ok_ids, accs) if acc > chance + 0.05}
        naive_fp = sum(1 for cid in naive_flagged if not tuned_mask[cid])

        # Pooled null so the screen is actually resolvable at this budget.
        sig = _significance(stl, cids, event_times, labels, ok_ids,
                            n_shuffles=200, null_mode='pooled')
        fdr_flagged = {cid for cid, v in sig['per_cell'].items() if v['significant']}
        fdr_fp = sum(1 for cid in fdr_flagged if not tuned_mask[cid])

        assert len(naive_flagged) > 0, "sanity: naive screen should flag something"
        assert fdr_fp <= naive_fp
        assert fdr_fp <= 1

    def test_structured_return_shape(self):
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(seed=1, n_cells=8)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)
        sig = _significance(stl, cids, event_times, labels, ok_ids, n_shuffles=20)

        assert set(sig) == {'per_cell', 'population', 'resolution'}
        assert set(sig['per_cell']) == set(ok_ids)
        for entry in sig['per_cell'].values():
            assert 0.0 <= entry['p_value'] <= 1.0
            assert 0.0 <= entry['q_value'] <= 1.0
            assert entry['n_shuffles'] == 20

    def test_no_successful_cells_returns_empty_per_cell(self):
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(seed=2, n_cells=4)
        sig = _significance(stl, cids, event_times, labels, [], n_shuffles=20)
        assert sig['per_cell'] == {}
        assert sig['population'] is None


class TestResolutionGuard:
    def test_under_resolved_run_is_flagged_and_warns(self):
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(seed=3, n_cells=8)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sig = compute_population_significance(
                stl, cids, event_times, labels, successful_cluster_ids=ok_ids,
                time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
                n_shuffles=5, seed=0,
            )
        # 8 cells x 5 shuffles -> p-floor 1/6, best q = 8/6 > 1 -> hopeless.
        assert sig['resolution']['resolvable'] is False
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert sig['resolution']['recommended_n_shuffles'] > 5

    @pytest.mark.slow
    def test_pooled_null_resolves_where_per_cell_cannot(self):
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(seed=4, n_cells=12)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)

        per_cell = _significance(stl, cids, event_times, labels, ok_ids,
                                 n_shuffles=100, null_mode='per_cell')
        pooled = _significance(stl, cids, event_times, labels, ok_ids,
                               n_shuffles=100, null_mode='pooled')

        assert per_cell['resolution']['resolvable'] is False
        assert pooled['resolution']['resolvable'] is True
        # Same compute, strictly finer p-value resolution.
        assert pooled['resolution']['p_floor'] < per_cell['resolution']['p_floor']
        assert pooled['resolution']['null_mode'] == 'pooled'

    def test_rejects_bad_null_mode(self):
        with pytest.raises(ValueError):
            compute_population_significance(
                [np.array([1.0])], [0], np.array([0.5]), np.array([0, 1]),
                successful_cluster_ids=[0], null_mode='bogus',
            )


class TestPopulationLevelTest:
    @pytest.mark.slow
    def test_detects_population_signal_and_is_well_resolved(self):
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(seed=5, n_cells=12, n_tuned=6)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)
        sig = _significance(stl, cids, event_times, labels, ok_ids, n_shuffles=100)

        pop = sig['population']
        # A single test needs no FDR correction, so 100 shuffles resolves it
        # even though the 12-cell per-cell screen at the same budget cannot.
        assert pop['p_value'] < 0.05
        assert pop['observed_mean_accuracy'] > pop['null_mean']
        assert pop['n_cells'] == len(ok_ids)
        assert fdr_resolution(1, 100, 0.05)['resolvable'] is True

    @pytest.mark.slow
    def test_null_data_gives_unremarkable_population_p(self):
        # No cell is tuned -> population mean should sit inside its null.
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(
            seed=6, n_cells=10, n_tuned=0)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)
        sig = _significance(stl, cids, event_times, labels, ok_ids, n_shuffles=100)
        assert sig['population']['p_value'] > 0.05

    @pytest.mark.slow
    def test_shared_permutations_keep_population_null_honest(self):
        """All cells must share one permutation per shuffle index.

        With independent per-cell permutations the population-mean null's
        variance shrinks by ~1/n_cells, which would make this test
        anti-conservative. A shared permutation keeps the null spread
        comparable to a single cell's.
        """
        stl, cids, event_times, labels, _ = _synthetic_opponent_scenario(
            seed=7, n_cells=10, n_tuned=0)
        _, ok_ids, _ = _decode(stl, cids, event_times, labels)
        sig = _significance(stl, cids, event_times, labels, ok_ids, n_shuffles=100)
        # Independent permutations would drive this toward ~0.01; shared
        # permutations keep real spread in the population-mean null.
        assert sig['population']['null_std'] > 0.02


class TestBaselineKeys:
    def test_imbalanced_labels_expose_below_baseline_accuracy(self):
        from ephys._lda_decoding import single_cell_lda_decode

        rng = np.random.default_rng(0)
        # 12/7 split, mirroring the real 631/20251216 winner/loser imbalance.
        labels = np.array(['winner'] * 12 + ['loser'] * 7)
        event_times = np.arange(len(labels)) * 10.0
        spikes = np.sort(rng.uniform(0, event_times[-1] + 10, 400))

        res = single_cell_lda_decode(
            spikes, event_times, labels,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
        )
        assert res['baseline_accuracy'] == pytest.approx(12 / 19)
        # 1/n_classes would claim 50%; the honest baseline is 63.2%.
        assert res['baseline_accuracy'] > 1 / res['n_classes']
        assert np.isfinite(res['balanced_accuracy'])

    def test_baseline_present_on_insufficient_data_path(self):
        from ephys._lda_decoding import single_cell_lda_decode

        labels = np.array(['a'] * 10 + ['b'] * 2)
        res = single_cell_lda_decode(
            np.array([1.0, 2.0]), np.arange(len(labels)) * 5.0, labels,
            min_events_per_class=5,
        )
        assert res['status'] == 'insufficient_data'
        assert res['baseline_accuracy'] == pytest.approx(10 / 12)
