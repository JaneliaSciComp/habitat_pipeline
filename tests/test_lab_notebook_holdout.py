"""Tests for the session-level holdout registry.

`test_never_touched_session_is_blocked` is the regression that motivates the
whole table. The gate it replaces asked `iterations_for_session(...)` and
concluded "not held out" when the answer was empty — so a session the loop had
never run against, i.e. exactly the thing a holdout protects, passed the
check. That gate could only fire after it had already been violated.

Everything here runs against a temp database, so full enforcement coverage
exists with no cohort-7 session actually reserved.
"""
import pytest

from database.lab_notebook import (
    HoldoutIndeterminate,
    HoldoutViolation,
    LabNotebook,
)


@pytest.fixture
def nb(tmp_path):
    return LabNotebook(tmp_path / 'notebook.db')


@pytest.fixture
def reserved(nb):
    """A whole-session reservation on a session with zero iterations."""
    return nb.reserve_holdout(
        'RatCity_20251210_1359_40Hz', cohort='cohort7',
        reason='reserved for confirmation of the EC opponent-identity finding',
        reserved_by='misha',
    )


class TestEmptyRegistryChangesNothing:
    """The mechanism ships built but unarmed, per the scope decision."""

    def test_nothing_is_held_out(self, nb):
        status = nb.holdout_status('20251210')
        assert status.held_out is False
        assert status.registry_is_empty is True

    def test_assert_passes_for_everything(self, nb):
        nb.assert_not_held_out('20251210', '631')
        nb.assert_not_held_out('20251216', '631', multi_animal=True)

    def test_summary_says_the_registry_is_empty(self, nb):
        assert 'registry is empty' in nb.holdout_status('20251210').summary()

    def test_list_is_empty(self, nb):
        assert nb.list_holdout() == []


class TestTheInvertedGateIsFixed:
    def test_never_touched_session_is_blocked(self, nb, reserved):
        """THE regression: zero iterations, and still blocked.

        The old check queried iterations_for_session and passed on an empty
        result, which is the state of every session a holdout is protecting.
        """
        assert nb.iterations_for_session('20251210') == []
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210')

    @pytest.mark.parametrize('query', [
        '20251210',
        'RatCity_20251210_1359_40Hz',
        'RatCity_20251210_1359_40Hz.rec',
        '20251210_094334',
    ])
    def test_blocked_under_every_id_form(self, nb, reserved, query):
        """Reserved by one string, queried by another - must still block."""
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out(query)

    def test_unreserved_session_is_unaffected(self, nb, reserved):
        assert nb.holdout_status('20251216').held_out is False

    def test_status_reports_which_reservation_and_why(self, nb, reserved):
        status = nb.holdout_status('20251210')
        assert status.held_out is True
        assert status.reservation_ids == (reserved.id,)
        assert 'EC opponent-identity' in status.reason
        assert status.scope == 'session'

    def test_violation_message_names_the_reservation(self, nb, reserved):
        with pytest.raises(HoldoutViolation, match='HELD OUT'):
            nb.assert_not_held_out('20251210')


class TestFailsClosed:
    def test_unresolvable_id_raises_rather_than_passing(self, nb, reserved):
        """'Cannot tell' must never be reported as 'not held out'."""
        with pytest.raises(HoldoutIndeterminate):
            nb.holdout_status('2025')

    @pytest.mark.parametrize('bad', ['2025', '', 'rat631', None])
    def test_unresolvable_forms_all_raise(self, nb, bad):
        with pytest.raises(HoldoutIndeterminate):
            nb.holdout_status(bad)

    def test_assert_also_fails_closed(self, nb):
        with pytest.raises(HoldoutIndeterminate):
            nb.assert_not_held_out('nope')

    def test_cannot_reserve_an_unresolvable_session(self, nb):
        with pytest.raises(HoldoutIndeterminate):
            nb.reserve_holdout('no-date', cohort='cohort7', reason='x', reserved_by='y')


class TestReservationHygiene:
    def test_reason_is_required(self, nb):
        with pytest.raises(ValueError):
            nb.reserve_holdout('20251210', cohort='cohort7', reason='  ', reserved_by='misha')

    def test_reserved_by_is_required(self, nb):
        with pytest.raises(ValueError):
            nb.reserve_holdout('20251210', cohort='cohort7', reason='r', reserved_by='')

    def test_records_the_original_id_for_provenance(self, nb, reserved):
        assert reserved.session_dir == 'RatCity_20251210_1359_40Hz'
        assert reserved.session_key == '20251210'


class TestRelease:
    def test_release_stops_blocking(self, nb, reserved):
        nb.release_holdout(reserved.id, reason='spent on confirmation', released_by='misha')
        assert nb.holdout_status('20251210').held_out is False

    def test_release_never_deletes_the_record(self, nb, reserved):
        nb.release_holdout(reserved.id, reason='spent', released_by='misha')
        assert nb.list_holdout(active_only=True) == []
        archived = nb.list_holdout(active_only=False)
        assert len(archived) == 1
        assert archived[0].release_reason == 'spent'
        assert archived[0].released_by == 'misha'
        assert archived[0].released_at is not None

    def test_double_release_is_rejected(self, nb, reserved):
        nb.release_holdout(reserved.id, reason='spent', released_by='misha')
        with pytest.raises(ValueError, match='already released'):
            nb.release_holdout(reserved.id, reason='again', released_by='misha')

    def test_unknown_id_raises(self, nb):
        with pytest.raises(ValueError, match='No holdout reservation'):
            nb.release_holdout(999, reason='x', released_by='y')


class TestAnimalScopeLeaks:
    """Animal-scoped reservations cannot be honoured by multi-animal analyses.

    Most cohort-7 sessions have four simultaneously implanted animals, and
    decode_location / run_inter_brain / social_spatial_fields /
    decode_partner_distance all read a partner's data through a focal animal.
    """

    @pytest.fixture
    def animal_reserved(self, nb):
        return nb.reserve_holdout('20251210', cohort='cohort7', animal_id='rat613',
                                   reason='hold out this animal', reserved_by='misha')

    def test_scope_is_recorded_as_animal(self, nb, animal_reserved):
        assert animal_reserved.scope == 'animal'

    def test_blocks_the_named_animal(self, nb, animal_reserved):
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210', 'rat613')

    def test_blocks_the_named_animal_under_either_id_form(self, nb, animal_reserved):
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210', '613')

    def test_single_animal_analysis_on_another_animal_is_allowed(self, nb, animal_reserved):
        nb.assert_not_held_out('20251210', 'rat631', multi_animal=False)

    def test_multi_animal_analysis_is_blocked_for_the_whole_session(self, nb, animal_reserved):
        """rat631 is not reserved, but a multi-animal analysis reads rat613 too."""
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210', 'rat631', multi_animal=True)

    def test_multi_animal_analysis_with_no_focal_animal_is_blocked(self, nb, animal_reserved):
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210', multi_animal=True)


class TestCohortScoping:
    def test_same_date_in_another_cohort_still_blocks_when_unspecified(self, nb):
        """Over-blocking is the safe direction when the caller can't say."""
        nb.reserve_holdout('20251210', cohort='cohort5', reason='r', reserved_by='m')
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210')

    def test_explicit_other_cohort_does_not_block(self, nb):
        nb.reserve_holdout('20251210', cohort='cohort5', reason='r', reserved_by='m')
        assert nb.holdout_status('20251210', cohort='cohort7').held_out is False


class TestPromotion:
    """An unlock requires a prospective frozen prediction to unlock *against*.

    A confirmation run whose prediction was written after the exploratory
    result was seen confirms nothing, so `unlock_holdout` validates the
    prediction rather than taking an id on trust.
    """

    @pytest.fixture
    def prediction(self, nb):
        hypothesis = nb.add_hypothesis('opponent identity generalizes')
        return nb.freeze_prediction(
            hypothesis.id, statistic='min_q_value', direction='lt', threshold=0.05,
            falsifier='min q >= 0.05 on the held-out session')

    @pytest.fixture
    def other_prediction(self, nb):
        hypothesis = nb.add_hypothesis('a different question')
        return nb.freeze_prediction(
            hypothesis.id, statistic='p_value', direction='lt', threshold=0.05,
            falsifier='p >= 0.05')

    def test_unlock_admits_exactly_one_hypothesis(self, nb, reserved, prediction):
        nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='misha',
                          frozen_prediction_id=prediction.id)
        nb.assert_not_held_out('20251210', purpose='confirmatory',
                               hypothesis_id=prediction.hypothesis_id)

    def test_unlock_does_not_admit_a_different_hypothesis(self, nb, reserved,
                                                          prediction, other_prediction):
        nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='misha',
                          frozen_prediction_id=prediction.id)
        with pytest.raises(HoldoutViolation, match='Not unlocked for hypothesis'):
            nb.assert_not_held_out('20251210', purpose='confirmatory',
                                   hypothesis_id=other_prediction.hypothesis_id)

    def test_unlock_does_not_admit_exploratory_work(self, nb, reserved, prediction):
        """An unlock buys one confirmation, not general access."""
        nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='misha',
                          frozen_prediction_id=prediction.id)
        with pytest.raises(HoldoutViolation):
            nb.assert_not_held_out('20251210', purpose='exploratory')

    def test_confirmatory_without_a_hypothesis_id_is_rejected(self, nb, reserved):
        with pytest.raises(HoldoutViolation, match='needs hypothesis_id'):
            nb.assert_not_held_out('20251210', purpose='confirmatory')

    def test_cannot_unlock_twice(self, nb, reserved, prediction, other_prediction):
        nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='m',
                          frozen_prediction_id=prediction.id)
        with pytest.raises(ValueError, match='already unlocked'):
            nb.unlock_holdout(reserved.id, other_prediction.hypothesis_id,
                              approved_by='m', frozen_prediction_id=other_prediction.id)

    def test_cannot_unlock_a_released_reservation(self, nb, reserved, prediction):
        nb.release_holdout(reserved.id, reason='spent', released_by='m')
        with pytest.raises(ValueError, match='already released'):
            nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='m',
                              frozen_prediction_id=prediction.id)

    def test_a_post_hoc_prediction_cannot_unlock(self, nb, reserved):
        """The point of the holdout is defeated if the prediction came second."""
        hypothesis = nb.add_hypothesis('x')
        post_hoc = nb.freeze_prediction(
            hypothesis.id, statistic='min_q_value', direction='lt', threshold=0.05,
            falsifier='f', registered_post_hoc=True)
        with pytest.raises(ValueError, match='registered post hoc'):
            nb.unlock_holdout(reserved.id, hypothesis.id, approved_by='m',
                              frozen_prediction_id=post_hoc.id)

    def test_a_prediction_for_another_hypothesis_cannot_unlock(self, nb, reserved,
                                                               prediction):
        other = nb.add_hypothesis('unrelated')
        with pytest.raises(ValueError, match='belongs to hypothesis'):
            nb.unlock_holdout(reserved.id, other.id, approved_by='m',
                              frozen_prediction_id=prediction.id)

    def test_a_superseded_prediction_cannot_unlock(self, nb, reserved, prediction):
        nb.supersede_prediction(prediction.id, reason='moved the target',
                                statistic='p_value', direction='lt', threshold=0.05,
                                falsifier='f')
        with pytest.raises(ValueError, match='superseded'):
            nb.unlock_holdout(reserved.id, prediction.hypothesis_id, approved_by='m',
                              frozen_prediction_id=prediction.id)

    def test_a_nonexistent_prediction_cannot_unlock(self, nb, reserved):
        with pytest.raises(ValueError, match='No frozen prediction'):
            nb.unlock_holdout(reserved.id, 1, approved_by='m', frozen_prediction_id=999)

    def test_bad_purpose_is_rejected(self, nb):
        with pytest.raises(ValueError, match='purpose must be'):
            nb.assert_not_held_out('20251210', purpose='whatever')


class TestMultipleReservations:
    def test_all_blocking_reservations_are_reported(self, nb):
        first = nb.reserve_holdout('20251210', cohort='cohort7', reason='first',
                                    reserved_by='m')
        second = nb.reserve_holdout('20251210', cohort='cohort7', reason='second',
                                     reserved_by='m')
        status = nb.holdout_status('20251210')
        assert set(status.reservation_ids) == {first.id, second.id}
        assert 'first' in status.reason and 'second' in status.reason

    def test_releasing_one_of_two_still_blocks(self, nb):
        first = nb.reserve_holdout('20251210', cohort='cohort7', reason='first',
                                    reserved_by='m')
        nb.reserve_holdout('20251210', cohort='cohort7', reason='second', reserved_by='m')
        nb.release_holdout(first.id, reason='spent', released_by='m')
        assert nb.holdout_status('20251210').held_out is True
