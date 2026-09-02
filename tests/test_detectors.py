"""Tests for discovery/detectors.py.

Every detector gets three cases: a pass, a fail, and a **missing-input case
that must raise MissingValue** rather than returning either verdict. That
third case is the point of the file. `discovery.hazards.run_detector` turns
MissingValue into `ran=False, passed=None`; a detector that instead guessed
would either hide a real hazard (guessed pass) or train the reader to ignore
the layer (guessed fail).

Where a number appears below without explanation it is a real one from this
project's lab notebook, cited in the test.
"""
import pytest

from discovery._predicates import MissingValue
from discovery.detectors import (
    accuracy_beats_baseline,
    class_selection_is_prespecified,
    date_resolved_files_belong_to_recording,
    kfold_shuffle_audit,
    method_implemented,
    param_equals,
    param_is_explicit,
    param_is_str,
    passthrough,
    pinned_at_p_floor,
    repo_pattern_absent,
    resolved_from_dated_directory,
    target_position_variance_ok,
    tracking_coverage_ok,
)


class TestPassthrough:
    def test_returns_value(self):
        assert passthrough(3) == 3
        assert passthrough(0) == 0        # falsy but present
        assert passthrough(False) is False

    def test_none_is_missing_not_a_value(self):
        with pytest.raises(MissingValue):
            passthrough(None)


class TestAccuracyBeatsBaseline:
    def test_the_real_12_7_outcome_split_fails(self):
        """decode_event_outcome on 631/20251216: 60.6% under a 63.2% baseline."""
        out = accuracy_beats_baseline(0.606, class_counts={'winner': 12, 'loser': 7})
        assert out['baseline_accuracy'] == pytest.approx(0.6315789, abs=1e-6)
        assert out['beats_baseline'] is False
        assert out['margin'] < 0

    def test_the_real_ec_opponent_result_passes(self):
        """decode_opponent_identity EC: 28.8% against a 27.7% baseline."""
        out = accuracy_beats_baseline(0.288, baseline=0.277)
        assert out['beats_baseline'] is True

    def test_uniform_chance_would_have_passed_it_wrongly(self):
        """Why the hazard exists: 1/n_classes makes the same run look fine.

        With two classes, 1/n_classes is 0.5 and 60.6% clears it comfortably.
        Against the actual prevalence it does not.
        """
        assert accuracy_beats_baseline(0.606, baseline=0.5)['beats_baseline'] is True
        assert accuracy_beats_baseline(0.606, baseline=0.6316)['beats_baseline'] is False

    def test_accepts_a_label_sequence(self):
        labels = ['w'] * 12 + ['l'] * 7
        out = accuracy_beats_baseline(0.606, class_counts=labels)
        assert out['baseline_accuracy'] == pytest.approx(12 / 19)

    def test_margin_requirement_is_applied(self):
        assert accuracy_beats_baseline(0.30, baseline=0.28, margin=0.05)['beats_baseline'] is False

    def test_missing_accuracy_is_missing(self):
        with pytest.raises(MissingValue):
            accuracy_beats_baseline(None, baseline=0.5)

    def test_missing_baseline_and_counts_is_missing(self):
        with pytest.raises(MissingValue):
            accuracy_beats_baseline(0.6)

    def test_empty_class_counts_is_missing(self):
        with pytest.raises(MissingValue):
            accuracy_beats_baseline(0.6, class_counts={})

    def test_nan_accuracy_is_missing(self):
        with pytest.raises(MissingValue):
            accuracy_beats_baseline(float('nan'), baseline=0.5)

    @pytest.mark.parametrize('baseline_key', [
        'population_baseline_accuracy',   # the decoder's raw result dict
        'baseline_accuracy',              # the notebook's curated summary
    ])
    def test_accepts_either_spelling_of_the_baseline(self, baseline_key):
        """Reading only one key made this silently unrunnable.

        The raw result dict and the curated summary name the baseline
        differently, and this detector runs against both. Bound to a single key,
        it reported "cannot check" on correctly-logged iterations - including
        rat630 and rat613 of hypothesis 4, which were genuinely below baseline.
        That is the worst possible failure for this particular check.
        """
        out = accuracy_beats_baseline(
            results={'population_accuracy_mean': 0.2184, baseline_key: 0.2358})
        assert out['beats_baseline'] is False
        assert out['margin'] < 0

    def test_results_mapping_detects_an_above_baseline_run(self):
        out = accuracy_beats_baseline(
            results={'population_accuracy_mean': 0.2881,
                     'population_baseline_accuracy': 0.2775})
        assert out['beats_baseline'] is True

    def test_an_empty_results_mapping_is_missing_not_a_pass(self):
        with pytest.raises(MissingValue):
            accuracy_beats_baseline(results={})

    def test_explicit_arguments_still_win_over_the_mapping(self):
        out = accuracy_beats_baseline(0.9, baseline=0.1,
                                      results={'population_accuracy_mean': 0.1,
                                               'baseline_accuracy': 0.9})
        assert out['accuracy'] == 0.9
        assert out['beats_baseline'] is True


class TestPinnedAtPFloor:
    def test_iteration_12_rat613_is_pinned(self):
        """q=0.0387 came from a p sitting exactly on the 180-shuffle floor."""
        out = pinned_at_p_floor(1.0 / 181, 180)
        assert out['pinned'] is True
        assert out['p_floor'] == pytest.approx(0.0055249, abs=1e-7)

    def test_a_p_clear_of_the_floor_is_not_pinned(self):
        out = pinned_at_p_floor(0.03, 180)
        assert out['pinned'] is False
        assert out['not_pinned'] is True

    def test_boundary_is_inclusive(self):
        assert pinned_at_p_floor(0.005, 199)['pinned'] is True

    def test_missing_p_is_missing(self):
        with pytest.raises(MissingValue):
            pinned_at_p_floor(None, 180)

    def test_missing_budget_is_missing(self):
        with pytest.raises(MissingValue):
            pinned_at_p_floor(0.01, None)

    def test_zero_shuffles_is_missing_not_a_verdict(self):
        with pytest.raises(MissingValue):
            pinned_at_p_floor(0.01, 0)


class TestTargetPositionVarianceOk:
    def test_rat630_does_not_move_enough(self):
        """The near-stationary animal that produced a spurious win."""
        out = target_position_variance_ok(11.0, 13.0)
        assert out['moves_enough'] is False
        assert out['smaller_std'] == 11.0

    def test_rat613_moves_enough(self):
        assert target_position_variance_ok(143.2, 121.9)['moves_enough'] is True

    def test_uses_the_smaller_axis(self):
        """A target pacing one axis is still degenerate on the other."""
        assert target_position_variance_ok(200.0, 5.0)['moves_enough'] is False

    def test_missing_std_is_missing(self):
        with pytest.raises(MissingValue):
            target_position_variance_ok(None, 100.0)
        with pytest.raises(MissingValue):
            target_position_variance_ok(100.0, None)


class TestTrackingCoverageOk:
    def test_session_20251216_fails(self):
        """merged_20251216_0950_1200 against a recording starting 09:43:34."""
        out = tracking_coverage_ok(0.63, ephys_window=[412.7, 8231.4])
        assert out['covers_enough'] is False
        assert out['ephys_window'] == [412.7, 8231.4]

    def test_full_coverage_passes(self):
        assert tracking_coverage_ok(0.99)['covers_enough'] is True

    def test_missing_fraction_is_missing(self):
        with pytest.raises(MissingValue):
            tracking_coverage_ok(None)


class TestDateResolvedFilesBelongToRecording:
    """Session 20251216 holds three recordings and one tracking file."""

    THREE = ['20251216_094334', '20251216_144334', '20251216_194334']

    def test_verified_overlap_passes(self):
        out = date_resolved_files_belong_to_recording(
            attachment_status='overlap_verified', is_primary=True,
            recording_ids_on_date=self.THREE)
        assert out['belongs_to_recording'] is True
        assert out['n_recordings_on_date'] == 3

    def test_no_overlap_fails(self):
        """The 14:43 block, offered the 09:50-12:00 tracking file."""
        out = date_resolved_files_belong_to_recording(
            attachment_status='no_overlap', is_primary=False,
            recording_ids_on_date=self.THREE)
        assert out['belongs_to_recording'] is False

    def test_undetermined_on_a_multi_recording_date_fails(self):
        """Unverified is not the same as fine.

        The file may well be the right one. But nothing has checked, and the
        wrong one yields a plausible rate map rather than an error, so the
        unchecked case has to fail.
        """
        out = date_resolved_files_belong_to_recording(
            attachment_status='undetermined', is_primary=False,
            recording_ids_on_date=self.THREE)
        assert out['attachment_verified'] is False
        assert out['belongs_to_recording'] is False

    def test_single_recording_on_the_date_passes_unverified(self):
        """Most sessions: one recording, nothing to confuse it with."""
        out = date_resolved_files_belong_to_recording(
            attachment_status='undetermined', is_primary=True,
            recording_ids_on_date=['20251210_110059'])
        assert out['single_recording_on_date'] is True
        assert out['belongs_to_recording'] is True

    def test_primary_alone_does_not_excuse_an_unverified_attachment(self):
        """Being the morning block is not evidence the video covers it.

        20251210's tracking starts at 13:59 against an 11:00 recording, so
        'primary' says nothing about whether the file lands inside.
        """
        out = date_resolved_files_belong_to_recording(
            attachment_status='undetermined', is_primary=True,
            recording_ids_on_date=self.THREE)
        assert out['belongs_to_recording'] is False

    def test_no_context_at_all_is_missing(self):
        with pytest.raises(MissingValue):
            date_resolved_files_belong_to_recording()


class TestParamDetectors:
    def test_param_equals(self):
        assert param_equals('reverse', 'shuffle')['matches'] is False
        assert param_equals('shuffle', 'shuffle')['matches'] is True

    def test_param_equals_missing_is_missing(self):
        with pytest.raises(MissingValue):
            param_equals(None, 'shuffle')

    def test_param_is_explicit(self):
        assert param_is_explicit(None)['is_explicit'] is False
        assert param_is_explicit('F')['is_explicit'] is True

    def test_param_is_explicit_accepts_falsy_explicit_values(self):
        """0 is a choice; None is the absence of one."""
        assert param_is_explicit(0)['is_explicit'] is True

    def test_param_is_str(self):
        assert param_is_str(631)['is_str'] is False
        assert param_is_str('631')['is_str'] is True
        assert param_is_str('rat631')['is_str'] is True

    def test_param_is_str_missing_is_missing(self):
        with pytest.raises(MissingValue):
            param_is_str(None)


class TestClassSelectionIsPrespecified:
    """The real iteration 14 -> 15 -> 16 sequence on animal 631/20251216."""

    def test_full_class_set_at_the_default_floor_passes(self):
        out = class_selection_is_prespecified(
            {'behavior_type': 'EC', 'min_events_per_class': 5, 'cv_folds': 5})
        assert out['is_prespecified'] is True

    def test_hand_picked_opponent_pair_trips(self):
        out = class_selection_is_prespecified(
            {'behavior_type': 'EC', 'selected_opponents': ['rat613', 'rat635'],
             'min_events_per_class': 5})
        assert out['is_prespecified'] is False
        assert out['narrowed_class_set'] is True
        assert out['n_selected'] == 2
        assert out['subset_key'] == 'selected_opponents'

    def test_relaxed_event_floor_trips(self):
        out = class_selection_is_prespecified({'min_events_per_class': 1})
        assert out['floor_ok'] is False
        assert out['is_prespecified'] is False

    def test_both_together_trips(self):
        out = class_selection_is_prespecified(
            {'selected_opponents': ['rat634', 'rat635'], 'min_events_per_class': 1,
             'cv_folds': 4})
        assert out['is_prespecified'] is False
        assert out['floor_ok'] is False
        assert out['narrowed_class_set'] is True

    def test_absent_floor_is_the_safe_default_not_a_failure(self):
        """Omitting the parameter means the module default applies."""
        out = class_selection_is_prespecified({'behavior_type': 'EC'})
        assert out['is_prespecified'] is True

    def test_empty_subset_list_is_not_a_narrowing(self):
        out = class_selection_is_prespecified({'selected_opponents': []})
        assert out['narrowed_class_set'] is False

    def test_other_subset_key_names_are_recognised(self):
        for key in ('selected_classes', 'selected_objects'):
            out = class_selection_is_prespecified({key: ['a', 'b']})
            assert out['subset_key'] == key
            assert out['is_prespecified'] is False

    def test_missing_params_is_missing(self):
        with pytest.raises(MissingValue):
            class_selection_is_prespecified(None)

    def test_non_mapping_params_is_missing(self):
        with pytest.raises(MissingValue):
            class_selection_is_prespecified(['not', 'a', 'mapping'])


class TestResolvedFromDatedDirectory:
    """The cohort-7 events root holds three versions of one session's scoring."""

    DATED = ('//nearline/karpova/TervoLab/analysis/RatCityBehavior/cohort7/'
             '20251216/20251216_behavior_event_df.csv')
    LOOSE = ('//nearline/karpova/TervoLab/analysis/RatCityBehavior/cohort7/'
             'behavior_event_df_update.csv')

    def test_the_canonical_dated_file_passes(self):
        out = resolved_from_dated_directory([self.DATED])
        assert out['all_from_dated_directory'] is True
        assert out['loose_files'] == []

    def test_a_loose_file_trips(self):
        """The newer 2026-08-13 exports the path resolver cannot see."""
        out = resolved_from_dated_directory([self.LOOSE])
        assert out['all_from_dated_directory'] is False
        assert len(out['loose_files']) == 1

    def test_mixing_conventions_trips(self):
        out = resolved_from_dated_directory([self.DATED, self.LOOSE])
        assert out['all_from_dated_directory'] is False
        assert len(out['dated_files']) == 1
        assert len(out['loose_files']) == 1

    def test_accepts_a_bare_string(self):
        assert resolved_from_dated_directory(self.DATED)['all_from_dated_directory']

    def test_no_files_is_missing_not_a_pass(self):
        with pytest.raises(MissingValue):
            resolved_from_dated_directory([])

    def test_none_is_missing(self):
        with pytest.raises(MissingValue):
            resolved_from_dated_directory(None)


class TestMethodImplemented:
    def test_pairwise_subspace_exists(self):
        out = method_implemented('ephys.inter_brain_dynamics', 'fit_shared_subspace')
        assert out['implemented'] is True

    def test_multiset_cca_does_not(self):
        """N>2 is blocked on missing math, not missing data."""
        out = method_implemented('ephys.inter_brain_dynamics', 'fit_multiset_subspace')
        assert out['implemented'] is False

    def test_unimportable_module_is_missing_not_false(self):
        """Distinguish 'not implemented' from 'could not look'."""
        with pytest.raises(MissingValue):
            method_implemented('ephys.no_such_module_at_all', 'anything')


class TestRepoPatternAbsent:
    def test_stale_kilosortdata_api_is_absent(self):
        out = repo_pattern_absent('KilosortData(data_input')
        assert out['absent'] is True, f"stale constructor reappeared at {out['hits']}"

    def test_finds_a_pattern_that_is_present(self):
        out = repo_pattern_absent('def load_kilosort_data')
        assert out['absent'] is False
        assert any('kilosort_data_import.py' in h for h in out['hits'])

    def test_reports_file_and_line(self):
        out = repo_pattern_absent('def load_kilosort_data')
        assert all(':' in h for h in out['hits'])

    def test_discovery_package_is_not_scanned(self, tmp_path):
        """Otherwise every scanning detector self-matches on its own regexes."""
        out = repo_pattern_absent('KilosortData(data_input')
        assert not any(h.startswith('discovery/') for h in out['hits'])


class TestKfoldShuffleAudit:
    def test_no_unapproved_sites_today(self):
        out = kfold_shuffle_audit()
        assert out['no_unapproved_sites'] is True, (
            f"new shuffle=True CV site(s): {out['unapproved_sites']}. On a "
            "time-ordered axis use KFold(shuffle=False)."
        )

    def test_the_lda_decoding_sites_are_found_and_allowlisted(self):
        """Allowlisted means reviewed, not cleared - see the hazard's known_gaps."""
        out = kfold_shuffle_audit()
        assert 'ephys/_lda_decoding.py' in out['reviewed_sites']

    def test_allowlist_has_no_stale_entries(self):
        """A stale entry means the allowlist stopped guarding what it names."""
        out = kfold_shuffle_audit()
        assert out['stale_allowlist_entries'] == [], (
            "allowlist names files with no shuffle=True CV site: "
            f"{out['stale_allowlist_entries']}. Either the site was fixed "
            "(delete the entry) or the file moved (update it)."
        )

    def test_decode_location_is_not_a_site_anymore(self):
        """The leaky splitter was fixed to contiguous folds on 2026-08-19."""
        out = kfold_shuffle_audit()
        assert 'ephys/decode_location.py' not in out['reviewed_sites']
        assert 'ephys/decode_location.py' not in out['unapproved_sites']

    def test_a_new_unapproved_site_is_flagged(self, tmp_path):
        """The ratchet fires on new code, which is its whole purpose."""
        (tmp_path / 'ephys').mkdir()
        (tmp_path / 'ephys' / 'brand_new.py').write_text(
            "from sklearn.model_selection import KFold\n"
            "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n",
            encoding='utf-8',
        )
        out = kfold_shuffle_audit(repo_root=tmp_path)
        assert out['unapproved_sites'] == ['ephys/brand_new.py']
        assert out['no_unapproved_sites'] is False

    def test_contiguous_folds_are_not_flagged(self, tmp_path):
        (tmp_path / 'ephys').mkdir()
        (tmp_path / 'ephys' / 'good.py').write_text(
            "kf = KFold(n_splits=5, shuffle=False)\n", encoding='utf-8')
        out = kfold_shuffle_audit(repo_root=tmp_path)
        assert out['unapproved_sites'] == []

    def test_multiline_splitter_construction_is_found(self, tmp_path):
        """The real _lda_decoding site spans two lines."""
        (tmp_path / 'ephys').mkdir()
        (tmp_path / 'ephys' / 'wrapped.py').write_text(
            "cv = StratifiedKFold(n_splits=5,\n"
            "                     shuffle=True, random_state=42)\n",
            encoding='utf-8',
        )
        assert kfold_shuffle_audit(repo_root=tmp_path)['unapproved_sites'] == ['ephys/wrapped.py']

    def test_explicit_allow_files_suppresses(self, tmp_path):
        (tmp_path / 'ephys').mkdir()
        (tmp_path / 'ephys' / 'ok.py').write_text(
            "kf = KFold(n_splits=5, shuffle=True)\n", encoding='utf-8')
        out = kfold_shuffle_audit(repo_root=tmp_path, allow_files=['ephys/ok.py'])
        assert out['unapproved_sites'] == []
        assert out['reviewed_sites'] == ['ephys/ok.py']
