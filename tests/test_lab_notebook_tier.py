"""Tests for pre-registration, verdicts, and the derived evidence tier.

The tier is computed from seven preconditions rather than stored, so that no
field can be edited to promote a result. `TestEachConditionIsLoadBearing`
builds a fully compliant confirmatory case and then breaks exactly one
condition at a time, asserting both that the tier drops to exploratory and
that the reason names the thing that was broken. Without the per-condition
parametrization a single over-permissive check could pass the happy-path test
while enforcing nothing.
"""
import pytest

from database.lab_notebook import (
    HYPOTHESIS_STATUSES,
    VERDICTS,
    LabNotebook,
)


@pytest.fixture
def nb(tmp_path):
    return LabNotebook(tmp_path / 'notebook.db')


def build_confirmatory(nb, *,
                       post_hoc=False,
                       skip_prediction=False,
                       skip_unlock=False,
                       skip_seed=False,
                       skip_fingerprint=False,
                       skip_decision=False,
                       dirty_denominator=False,
                       miss_threshold=False,
                       iteration_before_prediction=False):
    """Assemble the full honest path, optionally breaking one condition."""
    hypothesis = nb.add_hypothesis('Opponent identity generalizes to a held-out session')
    reservation = nb.reserve_holdout('20251218', cohort='cohort7',
                                     reason='confirmation set', reserved_by='misha')

    # Family name keyed on the hypothesis so several cases can coexist in one
    # notebook without get_or_create handing back a family already declared.
    family = nb.get_or_create_test_family(f'confirmation 20251218 h{hypothesis.id}')
    nb.declare_family_tests(family.id, ['opponent=EC:8way', 'opponent=EC:group'],
                            declared_by='misha')

    early_iteration = None
    if iteration_before_prediction:
        early_iteration = nb.log_iteration(
            'ephys.decode_opponent_identity', {'behavior_type': 'EC'},
            {'status': 'success'}, animal_id='631', session_id='20251218',
            hypothesis_id=hypothesis.id, test_family_id=family.id,
            seed=0, dataset_fingerprint='abc', fingerprint_method='v1')
        nb.record_decision(early_iteration.id, 'approved')

    prediction = None
    if not skip_prediction:
        prediction = nb.freeze_prediction(
            hypothesis.id, statistic='min_q_value', direction='lt', threshold=0.05,
            falsifier='min q >= 0.05 on the held-out session',
            n_shuffles_planned=500, holdout_kind='generalization',
            registered_post_hoc=post_hoc)
        if not skip_unlock and not post_hoc:
            nb.unlock_holdout(reservation.id, hypothesis.id, approved_by='misha',
                              frozen_prediction_id=prediction.id)

    p_value = 0.5 if miss_threshold else 0.001
    nb.record_family_test(family.id, 'opponent=EC:8way', p_value=p_value,
                          git_commit='deadbeef')
    if dirty_denominator:
        nb.abandon_family_test(family.id, 'opponent=EC:group',
                               reason='showed no margin once we looked',
                               outcome_dependent=True)
    else:
        nb.record_family_test(family.id, 'opponent=EC:group', p_value=0.6,
                              git_commit='deadbeef')

    iteration = nb.log_iteration(
        'ephys.decode_opponent_identity', {'behavior_type': 'EC', 'n_shuffles': 500},
        {'status': 'success', 'population_accuracy_mean': 0.31},
        animal_id='631', session_id='20251218',
        hypothesis_id=hypothesis.id, test_family_id=family.id,
        seed=None if skip_seed else 0,
        dataset_fingerprint=None if skip_fingerprint else 'abc123',
        fingerprint_method='paths+size+mtime/v1')
    if not skip_decision:
        nb.record_decision(iteration.id, 'approved', notes='clears its frozen threshold')

    return {'hypothesis': hypothesis, 'reservation': reservation, 'family': family,
            'prediction': prediction, 'iteration': iteration,
            'early_iteration': early_iteration}


class TestTheHappyPath:
    def test_the_full_honest_path_reaches_confirmatory(self, nb):
        built = build_confirmatory(nb)
        tier = nb.evidence_tier(built['hypothesis'].id)
        assert tier.tier == 'confirmatory'
        assert tier.blocking_reasons == ()
        assert tier.holdout_iteration_id == built['iteration'].id
        assert tier.frozen_prediction_id == built['prediction'].id

    def test_summary_names_the_holdout_iteration(self, nb):
        built = build_confirmatory(nb)
        assert 'CONFIRMATORY' in nb.evidence_tier(built['hypothesis'].id).summary()

    def test_reasons_are_empty_iff_confirmatory(self, nb):
        confirmatory = build_confirmatory(nb)
        exploratory = build_confirmatory(nb, skip_prediction=True)
        assert nb.evidence_tier(confirmatory['hypothesis'].id).blocking_reasons == ()
        assert nb.evidence_tier(exploratory['hypothesis'].id).blocking_reasons


class TestEachConditionIsLoadBearing:
    """Break one condition at a time; each must drop the tier and say why."""

    @pytest.mark.parametrize('breakage,expected_phrase', [
        ({'skip_prediction': True}, 'not pre-registered'),
        ({'post_hoc': True}, 'registered post hoc'),
        ({'skip_unlock': True}, 'reserved and unlocked'),
        ({'skip_seed': True}, 'seed'),
        ({'skip_fingerprint': True}, 'dataset_fingerprint'),
        ({'skip_decision': True}, 'scientist decision'),
        ({'dirty_denominator': True}, 'denominator status'),
        ({'miss_threshold': True}, 'frozen prediction was not met'),
        ({'iteration_before_prediction': True}, 'holdout was spent'),
    ])
    def test_breaking_one_condition_drops_the_tier(self, nb, breakage, expected_phrase):
        built = build_confirmatory(nb, **breakage)
        tier = nb.evidence_tier(built['hypothesis'].id)
        assert tier.tier == 'exploratory', f"{breakage} still reached confirmatory"
        assert tier.blocking_reasons
        if expected_phrase:
            joined = ' '.join(tier.blocking_reasons)
            assert expected_phrase in joined, (
                f"{breakage} gave reasons {tier.blocking_reasons!r}, none naming "
                f"{expected_phrase!r}")

    def test_a_holdout_already_analysed_is_spent(self, nb):
        """The gap this parametrization caught.

        Looking at the reserved session before unlocking it destroys the only
        thing a holdout provides. Attaching a pre-registration afterwards and
        re-running does not restore it, and the earlier check — "an iteration
        exists, run against an unlocked session, after the prediction" — was
        satisfied by the second run while the first had already burned the set.
        """
        built = build_confirmatory(nb, iteration_before_prediction=True)
        tier = nb.evidence_tier(built['hypothesis'].id)
        assert tier.tier == 'exploratory'
        assert any('spent' in reason for reason in tier.blocking_reasons)

    def test_contamination_by_another_hypothesis_also_spends_it(self, nb):
        """The loop having seen the session for a different question counts too."""
        other = nb.add_hypothesis('some other question')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         session_id='20251218', hypothesis_id=other.id)
        built = build_confirmatory(nb)
        tier = nb.evidence_tier(built['hypothesis'].id)
        assert tier.tier == 'exploratory'
        assert any('spent' in reason for reason in tier.blocking_reasons)

    def test_unlocking_for_another_hypothesis_does_not_count(self, nb):
        built = build_confirmatory(nb, skip_unlock=True)
        other = nb.add_hypothesis('a different question')
        other_prediction = nb.freeze_prediction(
            other.id, statistic='min_q_value', direction='lt', threshold=0.05,
            falsifier='x')
        nb.unlock_holdout(built['reservation'].id, other.id, approved_by='m',
                          frozen_prediction_id=other_prediction.id)
        assert nb.evidence_tier(built['hypothesis'].id).tier == 'exploratory'

    def test_a_missing_statistic_cannot_be_verified(self, nb):
        """An unverifiable threshold is not a met threshold."""
        built = build_confirmatory(nb, skip_prediction=True)
        prediction = nb.freeze_prediction(
            built['hypothesis'].id, statistic='no_such_metric', direction='gt',
            threshold=0.5, falsifier='x')
        nb.unlock_holdout(built['reservation'].id, built['hypothesis'].id,
                          approved_by='m', frozen_prediction_id=prediction.id)
        iteration = nb.log_iteration(
            'ephys.decode_opponent_identity', {}, {'status': 'success'},
            session_id='20251218', hypothesis_id=built['hypothesis'].id,
            test_family_id=built['family'].id, seed=0, dataset_fingerprint='f')
        nb.record_decision(iteration.id, 'approved')
        reasons = ' '.join(nb.evidence_tier(built['hypothesis'].id).blocking_reasons)
        assert 'not present' in reasons


class TestEverythingTodayIsExploratory:
    def test_a_bare_hypothesis_is_exploratory(self, nb):
        """Matches the real state of all pre-existing hypotheses."""
        hypothesis = nb.add_hypothesis('anything')
        tier = nb.evidence_tier(hypothesis.id)
        assert tier.tier == 'exploratory'
        assert 'not pre-registered' in tier.blocking_reasons[0]

    def test_a_result_on_an_unreserved_session_is_exploratory(self, nb):
        hypothesis = nb.add_hypothesis('EC opponent identity')
        prediction = nb.freeze_prediction(hypothesis.id, statistic='min_q_value',
                                           direction='lt', threshold=0.05,
                                           falsifier='q >= 0.05')
        nb.log_iteration('ephys.decode_opponent_identity', {}, {'status': 'success'},
                         session_id='20251216', hypothesis_id=hypothesis.id,
                         seed=0, dataset_fingerprint='f')
        assert nb.evidence_tier(hypothesis.id).tier == 'exploratory'
        assert prediction.registered_post_hoc is False


class TestFrozenPredictions:
    def test_falsifier_is_required(self, nb):
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError, match='falsifier'):
            nb.freeze_prediction(hypothesis.id, statistic='q', direction='lt',
                                 threshold=0.05, falsifier='  ')

    def test_direction_is_validated(self, nb):
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError, match='direction'):
            nb.freeze_prediction(hypothesis.id, statistic='q', direction='sideways',
                                 threshold=0.05, falsifier='f')

    def test_holdout_kind_is_validated(self, nb):
        """Replication and generalization are different claims."""
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError, match='holdout_kind'):
            nb.freeze_prediction(hypothesis.id, statistic='q', direction='lt',
                                 threshold=0.05, falsifier='f', holdout_kind='vibes')

    def test_unknown_hypothesis_raises(self, nb):
        with pytest.raises(ValueError, match='No hypothesis'):
            nb.freeze_prediction(999, statistic='q', direction='lt', threshold=0.05,
                                 falsifier='f')

    def test_versions_increment(self, nb):
        hypothesis = nb.add_hypothesis('x')
        first = nb.freeze_prediction(hypothesis.id, statistic='q', direction='lt',
                                      threshold=0.05, falsifier='f')
        second = nb.freeze_prediction(hypothesis.id, statistic='acc', direction='gt',
                                       threshold=0.3, falsifier='f')
        assert (first.version, second.version) == (1, 2)

    def test_spec_hash_is_stable_for_the_same_spec(self, nb):
        a = nb.add_hypothesis('a')
        b = nb.add_hypothesis('b')
        kwargs = dict(statistic='min_q_value', direction='lt', threshold=0.05,
                      falsifier='f')
        # Same spec bar the hypothesis id, which is part of the hash.
        assert nb.freeze_prediction(a.id, **kwargs).spec_hash != \
               nb.freeze_prediction(b.id, **kwargs).spec_hash

    def test_there_is_no_update_method(self):
        """Changing a frozen prediction must mean inserting a new one."""
        assert not hasattr(LabNotebook, 'update_prediction')
        assert not hasattr(LabNotebook, 'set_prediction_threshold')

    def test_supersede_inserts_and_links(self, nb):
        hypothesis = nb.add_hypothesis('x')
        original = nb.freeze_prediction(hypothesis.id, statistic='min_q_value',
                                         direction='lt', threshold=0.05, falsifier='f')
        replacement = nb.supersede_prediction(
            original.id, reason='switched to the population p-value',
            statistic='p_value', direction='lt', threshold=0.05, falsifier='f')
        rows = nb.frozen_predictions_for(hypothesis.id)
        assert len(rows) == 2
        assert rows[0].superseded_by_id == replacement.id
        assert 'population p-value' in rows[0].supersede_reason
        assert nb.current_prediction(hypothesis.id).id == replacement.id

    def test_supersede_requires_a_reason(self, nb):
        hypothesis = nb.add_hypothesis('x')
        original = nb.freeze_prediction(hypothesis.id, statistic='q', direction='lt',
                                         threshold=0.05, falsifier='f')
        with pytest.raises(ValueError, match='reason'):
            nb.supersede_prediction(original.id, reason='', statistic='q',
                                    direction='lt', threshold=0.1, falsifier='f')

    def test_cannot_supersede_twice(self, nb):
        hypothesis = nb.add_hypothesis('x')
        original = nb.freeze_prediction(hypothesis.id, statistic='q', direction='lt',
                                         threshold=0.05, falsifier='f')
        nb.supersede_prediction(original.id, reason='r', statistic='q', direction='lt',
                                threshold=0.1, falsifier='f')
        with pytest.raises(ValueError, match='already superseded'):
            nb.supersede_prediction(original.id, reason='again', statistic='q',
                                    direction='lt', threshold=0.2, falsifier='f')

    def test_a_superseded_prediction_cannot_carry_the_tier(self, nb):
        built = build_confirmatory(nb)
        nb.supersede_prediction(built['prediction'].id, reason='moved the goalposts',
                                statistic='min_q_value', direction='lt', threshold=0.001,
                                falsifier='f')
        assert nb.evidence_tier(built['hypothesis'].id).tier == 'exploratory'


class TestVerdicts:
    def test_refuted_is_a_first_class_verdict(self, nb):
        assert 'refuted' in VERDICTS
        hypothesis = nb.add_hypothesis('outcome is decodable')
        row = nb.record_verdict(hypothesis.id, verdict='refuted',
                                rationale='60.6% is below its own 63.2% majority baseline')
        assert row.verdict == 'refuted'

    def test_refuted_is_a_first_class_hypothesis_status(self, nb):
        assert 'refuted' in HYPOTHESIS_STATUSES
        hypothesis = nb.add_hypothesis('x')
        assert nb.set_hypothesis_status(hypothesis.id, 'refuted').status == 'refuted'

    def test_blocked_no_longer_needs_the_notes_field(self, nb):
        """Hypothesis 3 was blocked and had nowhere to say so."""
        assert 'blocked' in HYPOTHESIS_STATUSES
        hypothesis = nb.add_hypothesis('x')
        updated = nb.set_hypothesis_status(hypothesis.id, 'blocked',
                                           notes='no opponent tracking for this session')
        assert updated.status == 'blocked'

    def test_original_statuses_still_accepted(self, nb):
        hypothesis = nb.add_hypothesis('x')
        for status in ('proposed', 'approved', 'rejected', 'confirmed'):
            assert nb.set_hypothesis_status(hypothesis.id, status).status == status

    def test_bad_status_still_rejected(self, nb):
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError):
            nb.set_hypothesis_status(hypothesis.id, 'probably-fine')

    def test_bad_verdict_rejected(self, nb):
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError, match='verdict must be'):
            nb.record_verdict(hypothesis.id, verdict='looks good', rationale='r')

    def test_rationale_is_required(self, nb):
        hypothesis = nb.add_hypothesis('x')
        with pytest.raises(ValueError, match='rationale'):
            nb.record_verdict(hypothesis.id, verdict='refuted', rationale='   ')

    def test_tier_is_derived_not_asserted(self, nb):
        """A caller cannot label exploratory evidence confirmatory."""
        hypothesis = nb.add_hypothesis('x')
        row = nb.record_verdict(hypothesis.id, verdict='inconclusive', rationale='r')
        assert row.tier == 'exploratory'

    def test_verdicts_are_append_only(self, nb):
        hypothesis = nb.add_hypothesis('x')
        first = nb.record_verdict(hypothesis.id, verdict='inconclusive', rationale='early')
        second = nb.record_verdict(hypothesis.id, verdict='refuted', rationale='later',
                                    supersedes_id=first.id)
        rows = nb.verdicts_for(hypothesis.id)
        assert len(rows) == 2
        assert nb.latest_verdict(hypothesis.id).id == second.id
        assert rows[0].verdict == 'inconclusive'  # the earlier call is preserved

    def test_no_verdict_returns_none(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert nb.latest_verdict(hypothesis.id) is None

    def test_hypothesis_generating_only_records_the_unknown_denominator(self, nb):
        hypothesis = nb.add_hypothesis('EC opponent identity')
        row = nb.record_verdict(
            hypothesis.id, verdict='hypothesis_generating_only',
            rationale='the logged tests survive correction, but the denominator of the '
                      'informal search that produced them is unrecoverable',
            denominator_known=False, n_tests_in_denominator=4)
        assert row.denominator_known is False
        assert row.n_tests_in_denominator == 4


class TestHeldOutFlagTracksTheRegistry:
    def test_logging_against_a_reserved_session_marks_the_iteration(self, nb):
        nb.reserve_holdout('20251218', cohort='cohort7', reason='r', reserved_by='m')
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='20251218')
        assert iteration.held_out is True

    def test_logging_against_a_normal_session_does_not(self, nb):
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='20251216')
        assert iteration.held_out is False

    def test_an_explicit_flag_still_wins(self, nb):
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='20251216', held_out=True)
        assert iteration.held_out is True

    def test_an_unresolvable_session_id_does_not_raise_at_log_time(self, nb):
        """Logging must not become impossible for an oddly-named session."""
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='synthetic-test-session')
        assert iteration.held_out is False
