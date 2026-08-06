"""Domain guardrail: the rigor layer must correct the false positives a naive
accuracy-vs-chance screen produces when many cells are tested at once.

Recreates the exact synthetic scenario from the Phase 0 loop demo
(``scripts/phase0_probe.py::check_loop``, seed=0): 24 cells, 6 truly tuned to
one of two opponent labels, 18 untuned. That demo found a naive
"accuracy > chance + 0.05" screen flagged 9 cells as significant against the
6 that were actually tuned — three false positives purely from testing 24
cells at once. This test asserts ``compute_population_significance`` (the
label-permutation + BH-FDR rigor layer added in Phase 1) does not do worse
than that naive screen, and materially reduces its false-positive rate.
"""
import numpy as np
import pytest

from ephys._lda_decoding import compute_population_significance, run_population_per_cell_decode


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


class TestPermutationSignificanceGuardrail:
    def test_fdr_does_not_exceed_naive_false_positive_count(self):
        spike_times_list, cluster_ids, event_times, labels, tuned_mask = \
            _synthetic_opponent_scenario(seed=0)

        cell_results, ok_ids, accs = run_population_per_cell_decode(
            spike_times_list, cluster_ids, event_times, labels,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
            min_events_per_class=5, progress_every=0,
        )
        accs = np.asarray(accs)
        chance = float(max(np.mean(labels == 0), np.mean(labels == 1)))

        naive_flagged = {cid for cid, acc in zip(ok_ids, accs) if acc > chance + 0.05}
        naive_false_positives = sum(1 for cid in naive_flagged if not tuned_mask[cid])

        significance = compute_population_significance(
            spike_times_list, cluster_ids, event_times, labels,
            successful_cluster_ids=ok_ids,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
            n_shuffles=200, alpha=0.05, seed=0,
        )
        fdr_flagged = {cid for cid, v in significance.items() if v['significant']}
        fdr_false_positives = sum(1 for cid in fdr_flagged if not tuned_mask[cid])

        assert len(naive_flagged) > 0, "sanity check: naive screen should flag something"
        assert fdr_false_positives <= naive_false_positives
        # The whole point of the guardrail: FDR-correcting 24 simultaneous
        # tests should not manufacture more false "discoveries" than a
        # cruder threshold screen did.
        assert fdr_false_positives <= 1

    def test_significance_keys_match_requested_cluster_ids(self):
        spike_times_list, cluster_ids, event_times, labels, _ = \
            _synthetic_opponent_scenario(seed=1)
        _, ok_ids, _ = run_population_per_cell_decode(
            spike_times_list, cluster_ids, event_times, labels,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
            min_events_per_class=5, progress_every=0,
        )
        significance = compute_population_significance(
            spike_times_list, cluster_ids, event_times, labels,
            successful_cluster_ids=ok_ids,
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
            n_shuffles=20, alpha=0.05, seed=0,
        )
        assert set(significance.keys()) == set(ok_ids)
        for entry in significance.values():
            assert 0.0 <= entry['p_value'] <= 1.0
            assert 0.0 <= entry['q_value'] <= 1.0
            assert entry['n_shuffles'] == 20

    def test_no_successful_cells_returns_empty_dict(self):
        spike_times_list, cluster_ids, event_times, labels, _ = \
            _synthetic_opponent_scenario(seed=2)
        significance = compute_population_significance(
            spike_times_list, cluster_ids, event_times, labels,
            successful_cluster_ids=[],
            time_window=(0.0, 1.0), time_bin_size=0.25, cv_folds=5,
            n_shuffles=20, seed=0,
        )
        assert significance == {}
