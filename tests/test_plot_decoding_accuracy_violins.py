"""Tests for ``scripts/plot_decoding_accuracy_violins.py``.

Mock result dicts only — no share access. The point of the file is the
baseline logic: CLAUDE.md records that reading plain ``'accuracy'`` against
``1/n_classes`` once turned a 60.6% result into a "finding" when the
majority-class rate was 63.2%, so ``summarize`` must pick the majority
baseline for ``accuracy`` and ``1/n_classes`` for ``balanced_accuracy``, and
never the other way round.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "plot_decoding_accuracy_violins",
    REPO_ROOT / "scripts" / "plot_decoding_accuracy_violins.py",
)
violins = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = violins
_spec.loader.exec_module(violins)


def make_results(accuracies, balanced=None, *, classes=("rat613", "rat635"),
                 class_counts=(12, 7), significance=None, resolution=None,
                 n_total_cells=None):
    """A minimal result dict in the unified per-cell decoding schema."""
    cluster_ids = list(range(len(accuracies)))
    if balanced is None:
        balanced = accuracies
    cell_results = {
        cid: {"accuracy": acc, "balanced_accuracy": bal, "status": "success"}
        for cid, acc, bal in zip(cluster_ids, accuracies, balanced)
    }
    counts = dict(zip(classes, class_counts))
    majority = max(class_counts) / sum(class_counts)
    return {
        "cell_results": cell_results,
        "successful_cells": cluster_ids,
        "n_successful_cells": len(cluster_ids),
        "n_total_cells": n_total_cells or len(cluster_ids),
        "population_baseline_accuracy": majority,
        "significance": significance,
        "significance_population": None,
        "significance_resolution": resolution,
        "behavioral_summary": {
            "n_events": sum(class_counts),
            "unique_classes": np.array(classes),
            "class_counts": counts,
        },
        "parameters": {"class_label": "Opponent"},
        "status": "success",
        "_group": {"animal_id": "rat631", "session_id": "20251216",
                   "recording_id": "20251216_094334",
                   "config_path": "config/default_paths.json"},
    }


class TestBaselineChoice:
    """The baseline must follow the metric, not habit."""

    def test_accuracy_uses_majority_class_not_one_over_n(self):
        # 12/7 split -> majority 63.2%, which is well above 1/2.
        out = violins.summarize(make_results([0.606] * 4), "accuracy")
        assert out["baseline"] == pytest.approx(12 / 19)
        assert out["baseline_name"] == "majority class"
        assert out["chance"] == pytest.approx(0.5)
        assert out["baseline"] > out["chance"]

    def test_a_60_percent_result_is_below_the_majority_baseline(self):
        """The exact trap CLAUDE.md records, as an assertion."""
        out = violins.summarize(make_results([0.606] * 4), "accuracy")
        assert np.mean(out["values"]) < out["baseline"]
        # ...and would have looked like a win against 1/n_classes.
        assert np.mean(out["values"]) > out["chance"]

    def test_balanced_accuracy_uses_one_over_n_classes(self):
        out = violins.summarize(
            make_results([0.606] * 4, balanced=[0.55] * 4), "balanced_accuracy")
        assert out["baseline"] == pytest.approx(0.5)
        assert out["baseline_name"].startswith("chance")

    def test_chance_tracks_the_class_count(self):
        out = violins.summarize(
            make_results([0.4] * 3, classes=("a", "b", "c", "d"),
                         class_counts=(5, 5, 5, 5)), "accuracy")
        assert out["n_classes"] == 4
        assert out["chance"] == pytest.approx(0.25)
        assert out["baseline"] == pytest.approx(0.25)  # balanced -> they agree


class TestSummarize:
    def test_reads_the_requested_metric(self):
        out = violins.summarize(
            make_results([0.6, 0.7], balanced=[0.1, 0.2]), "balanced_accuracy")
        assert sorted(out["values"]) == pytest.approx([0.1, 0.2])

    def test_nan_cells_are_dropped_not_propagated(self):
        out = violins.summarize(make_results([0.6, np.nan, 0.7]), "accuracy")
        assert out["n_cells"] == 2
        assert np.all(np.isfinite(out["values"]))

    def test_n_total_cells_is_the_denominator_not_the_survivors(self):
        out = violins.summarize(
            make_results([0.6, 0.7], n_total_cells=149), "accuracy")
        assert out["n_cells"] == 2
        assert out["n_total_cells"] == 149

    def test_significant_count_is_none_when_the_screen_was_off(self):
        out = violins.summarize(make_results([0.6, 0.7]), "accuracy")
        assert out["n_significant"] is None

    def test_significant_count_counts_only_flagged_cells(self):
        sig = {0: {"significant": True, "q_value": 0.01},
               1: {"significant": False, "q_value": 0.9}}
        out = violins.summarize(
            make_results([0.6, 0.7], significance=sig), "accuracy")
        assert out["n_significant"] == 1

    def test_group_identity_comes_through(self):
        out = violins.summarize(make_results([0.6]), "accuracy")
        assert out["animal_id"] == "rat631"
        assert out["session_id"] == "20251216"
        # The recording, not just the date -- a .rec directory is a day, and
        # 20251216 has three blocks.
        assert out["recording_id"] == "20251216_094334"


class TestTargetParsing:
    def test_parses_animal_and_session(self):
        assert violins._target("rat650:20250819") == ("rat650", "20250819")

    @pytest.mark.parametrize("bad", ["rat650", "rat650:", ":20250819",
                                     "rat650:20250819:extra", ""])
    def test_rejects_malformed(self, bad):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            violins._target(bad)


class TestCacheKey:
    def test_differs_when_any_parameter_changes(self):
        base = {"behavior_type": "EC", "max_opponents": 2, "n_shuffles": 0}
        a = violins._cache_key("rat631", "20251216", base)
        b = violins._cache_key("rat631", "20251216",
                               {**base, "max_opponents": 3})
        c = violins._cache_key("rat630", "20251216", base)
        assert a != b
        assert a != c

    def test_is_stable_for_the_same_parameters(self):
        base = {"behavior_type": "EC", "max_opponents": 2}
        assert (violins._cache_key("rat631", "20251216", base)
                == violins._cache_key("rat631", "20251216", dict(base)))

    def test_is_insensitive_to_key_order(self):
        a = violins._cache_key("rat631", "20251216",
                               {"behavior_type": "EC", "max_opponents": 2})
        b = violins._cache_key("rat631", "20251216",
                               {"max_opponents": 2, "behavior_type": "EC"})
        assert a == b


class TestCohortConfig:
    def test_explicit_override_wins(self):
        assert violins.resolve_cohort_config(
            "20251216", "config/cohort5_paths.json") == "config/cohort5_paths.json"

    def test_cohort5_session_maps_to_the_cohort5_config(self):
        # 20250819 is cohort 5 in the shipped manifest; a prefix match is
        # required because the manifest keys on the full recording stamp.
        assert violins.resolve_cohort_config("20250819", None) == \
            "config/cohort5_paths.json"

    def test_cohort7_session_maps_to_the_default_config(self):
        assert violins.resolve_cohort_config("20251216", None) == \
            "config/default_paths.json"

    def test_unknown_session_falls_back_to_none(self):
        assert violins.resolve_cohort_config("19000101", None) is None


class TestFigure:
    def test_builds_without_error_and_labels_every_group(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [
            violins.summarize(make_results([0.5, 0.6, 0.7]), "accuracy"),
            violins.summarize(
                make_results([0.45, 0.55], class_counts=(10, 10)), "accuracy"),
        ]
        groups[1]["animal_id"] = "rat630"
        args = violins._build_parser().parse_args([])
        fig = violins.make_figure(groups, args)
        try:
            ax_raw, ax_delta = fig.axes[0], fig.axes[1]
            assert len(ax_raw.get_xticklabels()) == 2
            for ax in (ax_raw, ax_delta):
                labels = [t.get_text() for t in ax.get_xticklabels()]
                assert any("rat631" in t for t in labels)
                assert any("rat630" in t for t in labels)
        finally:
            plt.close(fig)

    def test_per_group_chance_lines_when_class_counts_differ(self):
        """Two groups with different n_classes must not share one chance line."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [
            violins.summarize(make_results([0.5, 0.6]), "accuracy"),
            violins.summarize(
                make_results([0.3, 0.4], classes=("a", "b", "c", "d"),
                             class_counts=(5, 5, 5, 5)), "accuracy"),
        ]
        assert groups[0]["chance"] != groups[1]["chance"]
        args = violins._build_parser().parse_args([])
        fig = violins.make_figure(groups, args)
        try:
            assert fig.axes  # the mixed-chance branch is exercised
        finally:
            plt.close(fig)
