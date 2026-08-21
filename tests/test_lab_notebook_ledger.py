"""Tests for the multiple-comparisons ledger.

`TestIteration12GoldenRegression` is the reason this layer exists. It replays
the real per-object p-values from lab notebook iteration 12 and pins both
answers: the honest one at the declared denominator (m=11, `rat613` not
significant) and the one that was actually reported (m=7, significant). Having
both in the suite means the bug cannot come back quietly — if someone
"simplifies" the denominator back to a count of surviving tests, the m=11 case
starts failing.

The subtler half is `test_resolution_guard_is_fooled_by_the_shrunk_family`:
the exclusion did not merely improve q-values, it flipped
`fdr_resolution`'s own verdict from unresolvable to resolvable, because the
guard was handed the number the exclusion produced.
"""
import numpy as np
import pytest

from database.lab_notebook import (
    LabNotebook,
    TestBudgetExhausted,
    UndeclaredTestError,
    canonical_params,
    compute_run_key,
)
from ephys._stats_utils import benjamini_hochberg, fdr_resolution


# Iteration 12: animal 631 / session 20251210, contiguous folds +
# rate_smoothing_sigma=2.0, null='shuffle', n_shuffles=180.
ITER12_P_VALUES = {
    '631': 0.022099447513812154,   # self-decoding sanity check
    '613': 0.0055248618784530384,  # the reported "finding", pinned at the p-floor
    '616': 0.08839779005524862,
    '617': 0.13259668508287292,
    '633': 0.9447513812154696,
    '634': 0.2541436464088398,
    '635': 0.19337016574585636,
}
# Iteration 10 swept every tracked object; that is the family a priori.
ITER10_OBJECTS = ['613', '615', '616', '617', '620', '621', '629',
                  '630', '631', '633', '634', '635']
ITER12_N_SHUFFLES = 180


@pytest.fixture
def nb(tmp_path):
    return LabNotebook(tmp_path / 'notebook.db')


@pytest.fixture
def family(nb):
    fam = nb.get_or_create_test_family('decode_location 631/20251210', alpha=0.05)
    nb.declare_family_tests(fam.id, [f'object={o}' for o in ITER10_OBJECTS],
                            declared_by='test')
    return fam


def _record_iter12(nb, family_id, commit='0703456'):
    for obj, p in ITER12_P_VALUES.items():
        nb.record_family_test(family_id, f'object={obj}', p_value=p, git_commit=commit)


def _abandon_the_five(nb, family_id):
    """Exactly the exclusions iterations 11/12 actually made."""
    nb.abandon_family_test(
        family_id, 'object=630',
        reason='near-stationary tracking (x/y std ~11-13px) -> degenerate error',
        outcome_dependent=False, criterion_available_a_priori=True,
        applied_after_seeing_results=True)
    for obj in ['615', '620', '621', '629']:
        nb.abandon_family_test(
            family_id, f'object={obj}',
            reason='showed no margin / reverse-null anomaly',
            outcome_dependent=True)


class TestIteration12GoldenRegression:
    def test_honest_denominator_is_eleven(self, nb, family):
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        denom = nb.family_denominator(family.id)
        assert denom['n_declared'] == 12
        assert denom['n_run'] == 7
        assert denom['n_excluded_prespecified'] == 1   # only the near-stationary one
        assert denom['n_abandoned'] == 4
        assert denom['n_tests_for_correction'] == 11

    def test_rat613_is_not_significant_at_the_honest_denominator(self, nb, family):
        """The reported finding does not survive its own family size."""
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        result = nb.family_fdr(family.id)
        rat613 = result['per_test']['object=613']
        assert rat613['q_value'] == pytest.approx(0.0608, abs=1e-4)
        assert rat613['significant'] is False

    def test_self_decoding_is_not_significant_either(self, nb, family):
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        assert nb.family_fdr(family.id)['per_test']['object=631']['significant'] is False

    def test_the_shrunk_denominator_reproduces_the_reported_number(self, nb):
        """m=7 gives back exactly the q=0.0387 that was logged.

        Pinned so the two answers stay side by side: the bug is not an
        arithmetic error, it is a choice of denominator.
        """
        fam = nb.get_or_create_test_family('shrunk', alpha=0.05)
        nb.declare_family_tests(fam.id, [f'object={o}' for o in ITER12_P_VALUES],
                                declared_by='test')
        _record_iter12(nb, fam.id)
        result = nb.family_fdr(fam.id)
        assert result['n_tests_for_correction'] == 7
        rat613 = result['per_test']['object=613']
        assert rat613['q_value'] == pytest.approx(0.03867403314917127, abs=1e-9)
        assert rat613['significant'] is True

    def test_resolution_guard_is_fooled_by_the_shrunk_family(self, nb, family):
        """The exclusion bought the *permission* to claim, not just the q-value.

        Fed the post-exclusion count the guard says resolvable; fed the honest
        one it says the design could not have resolved a lone test at all.
        """
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        honest = nb.family_fdr(family.id, n_shuffles=ITER12_N_SHUFFLES)
        assert honest['fdr_resolution']['resolvable'] is False
        assert honest['fdr_resolution']['recommended_n_shuffles'] == 220
        assert fdr_resolution(n_tests=7, n_shuffles=ITER12_N_SHUFFLES)['resolvable'] is True

    def test_outcome_dependent_exclusions_are_named_in_the_status(self, nb, family):
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        denom = nb.family_denominator(family.id)
        assert denom['denominator_status'] == 'outcome_dependent_exclusions'
        dropped = {e['test_key'] for e in denom['outcome_dependent_exclusions']}
        assert dropped == {'object=615', 'object=620', 'object=621', 'object=629'}

    def test_a_clean_family_says_so(self, nb, family):
        _record_iter12(nb, family.id)
        nb.abandon_family_test(family.id, 'object=630', reason='near-stationary',
                               outcome_dependent=False, criterion_available_a_priori=True)
        for obj in ['615', '620', '621', '629']:
            nb.record_family_test(family.id, f'object={obj}', p_value=0.5,
                                  git_commit='0703456')
        assert nb.family_denominator(family.id)['denominator_status'] == 'clean'


class TestPaddingEquivalence:
    def test_padding_reproduces_conservative_bh_at_the_declared_m(self, nb, family):
        """The one piece of arithmetic that lets _stats_utils stay untouched."""
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        result = nb.family_fdr(family.id)
        m = result['n_tests_for_correction']

        pvals = list(ITER12_P_VALUES.values()) + [1.0] * (m - len(ITER12_P_VALUES))
        expected = benjamini_hochberg(np.array(pvals, dtype=np.float64))
        for i, obj in enumerate(ITER12_P_VALUES):
            assert result['per_test'][f'object={obj}']['q_value'] == pytest.approx(
                float(expected[i]), abs=1e-12)

    def test_padding_is_conservative_relative_to_scoring_only_the_run_tests(self, nb, family):
        """Padding can only ever make a q-value larger, never smaller."""
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        padded = nb.family_fdr(family.id)['per_test']
        unpadded = benjamini_hochberg(
            np.array(list(ITER12_P_VALUES.values()), dtype=np.float64))
        for i, obj in enumerate(ITER12_P_VALUES):
            assert padded[f'object={obj}']['q_value'] >= float(unpadded[i]) - 1e-12

    def test_padded_count_is_reported(self, nb, family):
        _record_iter12(nb, family.id)
        _abandon_the_five(nb, family.id)
        result = nb.family_fdr(family.id)
        assert result['n_scored'] == 7
        assert result['n_padded'] == 4

    def test_fully_run_family_pads_nothing(self, nb, family):
        for obj in ITER10_OBJECTS:
            nb.record_family_test(family.id, f'object={obj}', p_value=0.5)
        assert nb.family_fdr(family.id)['n_padded'] == 0

    def test_empty_family_result_is_empty_not_an_error(self, nb, family):
        result = nb.family_fdr(family.id)
        assert result['per_test'] == {}


class TestDeclarationDiscipline:
    def test_recording_an_undeclared_test_is_refused(self, nb, family):
        with pytest.raises(UndeclaredTestError, match='never declared'):
            nb.record_family_test(family.id, 'object=999', p_value=0.01)

    def test_the_refusal_lists_what_was_declared(self, nb, family):
        with pytest.raises(UndeclaredTestError, match='object=613'):
            nb.record_family_test(family.id, 'object=999', p_value=0.01)

    def test_redeclaring_a_family_is_refused(self, nb, family):
        with pytest.raises(ValueError, match='already declares'):
            nb.declare_family_tests(family.id, ['object=999'], declared_by='test')

    def test_extending_is_allowed_but_recorded(self, nb, family):
        nb.declare_family_tests(family.id, ['object=999'], declared_by='test',
                                extend=True, notes='added a late target')
        denom = nb.family_denominator(family.id)
        assert denom['n_declared'] == 13
        tests = {t.test_key for t in nb.family_tests(family.id)}
        assert 'object=999' in tests

    def test_declaring_nothing_is_refused(self, nb):
        fam = nb.get_or_create_test_family('empty')
        with pytest.raises(ValueError):
            nb.declare_family_tests(fam.id, [], declared_by='test')

    def test_duplicate_keys_are_collapsed_at_declaration(self, nb):
        fam = nb.get_or_create_test_family('dupes')
        created = nb.declare_family_tests(fam.id, ['a', 'b', 'a'], declared_by='test')
        assert len(created) == 2
        assert nb.family_denominator(fam.id)['n_declared'] == 2

    def test_unknown_family_raises(self, nb):
        with pytest.raises(ValueError, match='No test family'):
            nb.declare_family_tests(9999, ['a'], declared_by='test')


class TestAbandonRequiresAnAnswer:
    def test_outcome_dependent_has_no_default(self, nb, family):
        """You cannot drop a test without saying why you're dropping it."""
        with pytest.raises(TypeError):
            nb.abandon_family_test(family.id, 'object=630', reason='because')

    def test_non_boolean_outcome_dependent_is_refused(self, nb, family):
        with pytest.raises(TypeError):
            nb.abandon_family_test(family.id, 'object=630', reason='r',
                                   outcome_dependent='yes')

    def test_reason_is_required(self, nb, family):
        with pytest.raises(ValueError):
            nb.abandon_family_test(family.id, 'object=630', reason='   ',
                                   outcome_dependent=False)

    def test_outcome_dependent_exclusion_stays_in_the_denominator(self, nb, family):
        """Deciding to drop it *was* a test."""
        _record_iter12(nb, family.id)
        nb.abandon_family_test(family.id, 'object=615', reason='no margin',
                               outcome_dependent=True)
        denom = nb.family_denominator(family.id)
        assert denom['n_excluded_prespecified'] == 0
        assert denom['n_tests_for_correction'] == 12

    def test_a_priori_exclusion_leaves_the_denominator(self, nb, family):
        _record_iter12(nb, family.id)
        nb.abandon_family_test(family.id, 'object=630', reason='near-stationary',
                               outcome_dependent=False,
                               criterion_available_a_priori=True)
        assert nb.family_denominator(family.id)['n_tests_for_correction'] == 11

    def test_both_flags_are_stored_separately(self, nb, family):
        """The near-stationary case: a priori criterion, applied post hoc."""
        test = nb.abandon_family_test(
            family.id, 'object=630', reason='x', outcome_dependent=False,
            criterion_available_a_priori=True, applied_after_seeing_results=True)
        assert test.criterion_available_a_priori is True
        assert test.applied_after_seeing_results is True
        assert test.exclusion_outcome_dependent is False
        assert test.status == 'excluded_prespecified'

    def test_abandoning_an_undeclared_test_raises(self, nb, family):
        with pytest.raises(UndeclaredTestError):
            nb.abandon_family_test(family.id, 'object=nope', reason='r',
                                   outcome_dependent=False)


class TestBudget:
    def test_budget_defaults_to_the_declared_size(self, nb, family):
        assert nb.family_tests(family.id)
        for obj in ITER10_OBJECTS:
            nb.record_family_test(family.id, f'object={obj}', p_value=0.5)
        # A 13th recording of an already-run key is a rerun and exceeds it.
        with pytest.raises(TestBudgetExhausted):
            nb.record_family_test(family.id, 'object=613', p_value=0.4)

    def test_exhaustion_message_is_actionable(self, nb):
        fam = nb.get_or_create_test_family('tiny')
        nb.declare_family_tests(fam.id, ['a'], declared_by='t')
        nb.record_family_test(fam.id, 'a', p_value=0.1)
        with pytest.raises(TestBudgetExhausted, match='extend_family_budget'):
            nb.record_family_test(fam.id, 'a', p_value=0.2)

    def test_extension_permits_the_rerun_and_leaves_a_trail(self, nb):
        fam = nb.get_or_create_test_family('extendable')
        nb.declare_family_tests(fam.id, ['a'], declared_by='t')
        nb.record_family_test(fam.id, 'a', p_value=0.1)
        nb.extend_family_budget(fam.id, 2, actor='misha', reason='re-run after a code fix')
        rerun = nb.record_family_test(fam.id, 'a', p_value=0.2)
        assert rerun.rerun_of_id is not None
        with nb.get_db_session() as s:
            from database.lab_notebook import TestFamily
            assert 're-run after a code fix' in s.get(TestFamily, fam.id).notes

    def test_extension_requires_a_reason(self, nb, family):
        with pytest.raises(ValueError):
            nb.extend_family_budget(family.id, 99, actor='m', reason='')

    def test_budget_cannot_be_lowered_retroactively(self, nb, family):
        with pytest.raises(ValueError, match='not lowered'):
            nb.extend_family_budget(family.id, 2, actor='m', reason='shrink')

    def test_declaring_beyond_an_explicit_budget_is_refused(self, nb):
        fam = nb.get_or_create_test_family('capped')
        with pytest.raises(TestBudgetExhausted):
            nb.declare_family_tests(fam.id, ['a', 'b', 'c'], declared_by='t',
                                    budget_max_tests=2)

    def test_reruns_do_not_inflate_the_denominator(self, nb):
        fam = nb.get_or_create_test_family('reruns')
        nb.declare_family_tests(fam.id, ['a', 'b'], declared_by='t')
        nb.record_family_test(fam.id, 'a', p_value=0.1)
        nb.record_family_test(fam.id, 'b', p_value=0.2)
        nb.extend_family_budget(fam.id, 3, actor='m', reason='rerun')
        nb.record_family_test(fam.id, 'a', p_value=0.15)
        denom = nb.family_denominator(fam.id)
        assert denom['n_declared'] == 2
        assert denom['n_tests_for_correction'] == 2
        assert denom['n_reruns'] == 1


class TestPipelineDrift:
    def test_more_than_one_commit_flags_the_family(self, nb, family):
        """Iterations 11 and 12 are the same family under two CV splitters."""
        nb.record_family_test(family.id, 'object=613', p_value=0.0055,
                              git_commit='c00a3cb')  # leaky shuffled folds
        nb.record_family_test(family.id, 'object=631', p_value=0.43,
                              git_commit='0703456')  # contiguous folds
        denom = nb.family_denominator(family.id)
        assert denom['denominator_status'] == 'pipeline_changed'
        assert denom['distinct_commits'] == ['0703456', 'c00a3cb']

    def test_one_commit_is_clean(self, nb, family):
        _record_iter12(nb, family.id, commit='0703456')
        denom = nb.family_denominator(family.id)
        assert denom['distinct_commits'] == ['0703456']
        assert denom['denominator_status'] == 'clean'

    def test_reconstructed_status_dominates(self, nb):
        """A denominator recovered after the fact is a lower bound, and the
        report must say so even if everything else looks tidy."""
        fam = nb.get_or_create_test_family('recon')
        nb.declare_family_tests(fam.id, ['a', 'b'], declared_by='backfill',
                                denominator_status='reconstructed')
        nb.record_family_test(fam.id, 'a', p_value=0.01, git_commit='x')
        assert nb.family_denominator(fam.id)['denominator_status'] == 'reconstructed'


class TestRunIdentity:
    def test_same_params_same_code_same_key(self):
        params = {'behavior_type': 'EC', 'n_shuffles': 200}
        a = compute_run_key('ephys.decode_opponent_identity', params, git_commit='abc')
        b = compute_run_key('ephys.decode_opponent_identity', dict(params), git_commit='abc')
        assert a == b

    def test_explicit_default_collapses_with_an_omitted_one(self):
        """Iterations 5 and 7 differ only in which defaults they spelled out."""
        sparse = {'behavior_type': 'EC'}
        verbose = canonical_params('ephys.decode_opponent_identity', sparse)
        assert len(verbose) > len(sparse), "signature defaults were not filled in"
        assert compute_run_key('ephys.decode_opponent_identity', sparse, git_commit='c') == \
               compute_run_key('ephys.decode_opponent_identity', verbose, git_commit='c')

    def test_a_real_param_difference_separates(self):
        """Iterations 3 and 4 differ by null_mode and are two tests."""
        a = compute_run_key('ephys.decode_opponent_identity',
                            {'behavior_type': 'EC', 'null_mode': 'per_cell'}, git_commit='c')
        b = compute_run_key('ephys.decode_opponent_identity',
                            {'behavior_type': 'EC', 'null_mode': 'pooled'}, git_commit='c')
        assert a != b

    def test_a_code_change_separates(self):
        params = {'behavior_type': 'EC'}
        assert compute_run_key('ephys.decode_location', params, git_commit='c00a3cb') != \
               compute_run_key('ephys.decode_location', params, git_commit='0703456')

    def test_a_dataset_change_separates(self):
        params = {'behavior_type': 'EC'}
        assert compute_run_key('ephys.decode_location', params, dataset_fingerprint='a') != \
               compute_run_key('ephys.decode_location', params, dataset_fingerprint='b')

    def test_output_location_does_not_change_the_identity(self):
        base = {'behavior_type': 'EC'}
        assert compute_run_key('ephys.decode_location', base) == \
               compute_run_key('ephys.decode_location', {**base, 'output_dir': '/tmp/x'})

    def test_unknown_module_still_hashes(self):
        assert compute_run_key('ephys.not_a_real_module', {'a': 1})

    def test_numpy_values_do_not_break_hashing(self):
        assert compute_run_key('ephys.decode_location', {'n': np.int64(5),
                                                          'x': np.float64(0.5)})


class TestGetOrCreateTestFamily:
    def test_second_call_returns_the_same_family(self, nb):
        """create_test_family's missing check produced two families 8 minutes
        apart in the real notebook, one empty - which silently splits the
        denominator."""
        first = nb.get_or_create_test_family('phase1.5-verification 631/20251216')
        second = nb.get_or_create_test_family('phase1.5-verification 631/20251216')
        assert first.id == second.id

    def test_create_test_family_still_duplicates(self, nb):
        """Documents the behaviour that motivated get_or_create."""
        a = nb.create_test_family('same name')
        b = nb.create_test_family('same name')
        assert a.id != b.id


class TestExistingApiUnchanged:
    def test_recompute_family_significance_still_works(self, nb):
        """The campaign-level method predates the ledger and must be untouched."""
        fam = nb.create_test_family('campaign')
        nb.log_iteration('ephys.decode_opponent_identity', {'behavior_type': 'EC'},
                         {'status': 'success', 'p_value': 0.005},
                         test_family_id=fam.id, session_id='20251216')
        nb.log_iteration('ephys.decode_event_outcome', {},
                         {'status': 'success', 'p_value': 0.114},
                         test_family_id=fam.id, session_id='20251216')
        qvals = nb.recompute_family_significance(fam.id)
        assert len(qvals) == 2
        assert all(0.0 <= q <= 1.0 for q in qvals.values())
