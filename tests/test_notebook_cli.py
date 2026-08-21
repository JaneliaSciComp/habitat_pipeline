"""Tests for scripts/notebook_cli.py.

Exit codes are part of the contract here, because the CLI is meant to be usable
from a shell guard: `check-holdout` returns 0 for free, 1 for blocked, and
**2 for indeterminate**. That third case is the one that matters — a session id
that cannot be resolved must not be reported as free, and a caller doing
`if notebook_cli check-holdout "$s"; then run; fi` has to see a non-zero exit.
"""
import json

import pytest

from scripts.notebook_cli import main


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / 'notebook.db')


def run(db_path, *argv):
    return main(['--db', db_path, *argv])


def _seed_hypothesis(db_path, statement='a hypothesis'):
    from database.lab_notebook import LabNotebook
    return LabNotebook(db_path).add_hypothesis(statement).id


class TestHoldoutExitCodes:
    def test_free_session_exits_zero(self, db, capsys):
        assert run(db, 'check-holdout', '20251216') == 0
        assert 'not held out' in capsys.readouterr().out

    def test_reserved_session_exits_one(self, db):
        run(db, 'reserve-holdout', '20251218', '--cohort', 'cohort7',
            '--reason', 'confirmation', '--by', 'misha')
        assert run(db, 'check-holdout', '20251218') == 1

    def test_unresolvable_session_exits_two_not_zero(self, db, capsys):
        """A shell guard must not treat 'cannot tell' as 'go ahead'."""
        assert run(db, 'check-holdout', '2025') == 2
        assert 'INDETERMINATE' in capsys.readouterr().out

    def test_any_id_form_of_a_reserved_session_is_blocked(self, db):
        run(db, 'reserve-holdout', '20251210', '--cohort', 'cohort7',
            '--reason', 'r', '--by', 'm')
        for query in ('20251210', 'RatCity_20251210_1359_40Hz',
                      'RatCity_20251210_1359_40Hz.rec'):
            assert run(db, 'check-holdout', query) == 1

    def test_multi_animal_flag_promotes_an_animal_reservation(self, db):
        run(db, 'reserve-holdout', '20251210', '--cohort', 'cohort7',
            '--reason', 'r', '--by', 'm', '--animal', 'rat613')
        assert run(db, 'check-holdout', '20251210', '--animal', 'rat631') == 0
        assert run(db, 'check-holdout', '20251210', '--animal', 'rat631',
                   '--multi-animal') == 1

    def test_animal_scope_warns_that_it_leaks(self, db, capsys):
        run(db, 'reserve-holdout', '20251210', '--cohort', 'cohort7',
            '--reason', 'r', '--by', 'm', '--animal', 'rat613')
        assert 'multi-animal' in capsys.readouterr().out

    def test_list_holdout_when_empty_says_so(self, db, capsys):
        assert run(db, 'list-holdout') == 0
        assert 'nothing is currently reserved' in capsys.readouterr().out

    def test_release_keeps_the_record(self, db, capsys):
        run(db, 'reserve-holdout', '20251218', '--cohort', 'cohort7',
            '--reason', 'r', '--by', 'm')
        run(db, 'release-holdout', '1', '--reason', 'spent', '--by', 'm')
        capsys.readouterr()
        run(db, 'list-holdout', '--all')
        assert 'released' in capsys.readouterr().out


class TestLedgerCommands:
    def test_declare_then_report_the_denominator(self, db, capsys):
        assert run(db, 'declare-family', 'fam', 'a', 'b', 'c', '--by', 'misha') == 0
        out = capsys.readouterr().out
        assert 'declared 3 new test(s)' in out
        assert '"n_declared": 3' in out

    def test_abandon_requires_stating_outcome_dependence(self, db):
        run(db, 'declare-family', 'fam', 'a', '--by', 'm')
        with pytest.raises(SystemExit):
            run(db, 'abandon-test', '1', 'a', '--reason', 'because')

    def test_outcome_dependent_drop_stays_in_the_denominator(self, db, capsys):
        run(db, 'declare-family', 'fam', 'a', 'b', '--by', 'm')
        capsys.readouterr()
        run(db, 'abandon-test', '1', 'a', '--reason', 'no margin',
            '--outcome-dependent')
        assert 'STAYS in the denominator' in capsys.readouterr().out
        run(db, 'denominator', '1')
        payload = json.loads(capsys.readouterr().out.split('\n\n')[0])
        assert payload['n_tests_for_correction'] == 2

    def test_a_priori_drop_leaves_the_denominator(self, db, capsys):
        run(db, 'declare-family', 'fam', 'a', 'b', '--by', 'm')
        run(db, 'abandon-test', '1', 'a', '--reason', 'near-stationary',
            '--not-outcome-dependent', '--a-priori')
        capsys.readouterr()
        run(db, 'denominator', '1')
        payload = json.loads(capsys.readouterr().out.split('\n\n')[0])
        assert payload['n_tests_for_correction'] == 1

    def test_denominator_warns_when_not_clean(self, db, capsys):
        run(db, 'declare-family', 'fam', 'a', 'b', '--by', 'm')
        run(db, 'abandon-test', '1', 'a', '--reason', 'no margin',
            '--outcome-dependent')
        capsys.readouterr()
        run(db, 'denominator', '1')
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'hand-counted n_tests' in out

    def test_ledger_reports_the_resolution_verdict(self, db, capsys):
        run(db, 'declare-family', 'fam', *[f'c{i}' for i in range(149)], '--by', 'm')
        capsys.readouterr()
        run(db, 'ledger', '1', '--n-shuffles', '200')
        out = capsys.readouterr().out
        assert 'resolvable at 200 shuffles: False' in out
        assert '2980 would be needed' in out

    def test_extend_budget_requires_a_reason(self, db):
        run(db, 'declare-family', 'fam', 'a', '--by', 'm')
        with pytest.raises(SystemExit):
            run(db, 'extend-budget', '1', '5', '--by', 'm')


class TestPreRegistrationCommands:
    def test_freeze_requires_a_falsifier(self, db):
        hypothesis_id = _seed_hypothesis(db)
        with pytest.raises(SystemExit):
            run(db, 'freeze', str(hypothesis_id), '--statistic', 'min_q_value',
                '--direction', 'lt', '--threshold', '0.05')

    def test_freeze_reports_a_post_hoc_registration_as_such(self, db, capsys):
        hypothesis_id = _seed_hypothesis(db)
        run(db, 'freeze', str(hypothesis_id), '--statistic', 'min_q_value',
            '--direction', 'lt', '--threshold', '0.05',
            '--falsifier', 'q >= 0.05 on held-out data', '--post-hoc')
        assert 'POST HOC' in capsys.readouterr().out

    def test_unlock_refuses_a_post_hoc_prediction(self, db, capsys):
        hypothesis_id = _seed_hypothesis(db)
        run(db, 'reserve-holdout', '20251218', '--cohort', 'cohort7',
            '--reason', 'r', '--by', 'm')
        run(db, 'freeze', str(hypothesis_id), '--statistic', 'min_q_value',
            '--direction', 'lt', '--threshold', '0.05', '--falsifier', 'f',
            '--post-hoc')
        assert run(db, 'unlock-holdout', '1', str(hypothesis_id),
                   '--prediction', '1', '--by', 'm') == 2

    def test_verdict_derives_the_tier(self, db, capsys):
        hypothesis_id = _seed_hypothesis(db)
        run(db, 'verdict', str(hypothesis_id), 'refuted',
            '--rationale', 'below its own majority baseline')
        assert 'tier exploratory, derived' in capsys.readouterr().out

    def test_refuted_is_an_accepted_verdict(self, db):
        hypothesis_id = _seed_hypothesis(db)
        assert run(db, 'verdict', str(hypothesis_id), 'refuted',
                   '--rationale', 'r') == 0

    def test_blocked_is_an_accepted_status(self, db):
        hypothesis_id = _seed_hypothesis(db)
        assert run(db, 'status', str(hypothesis_id), 'blocked',
                   '-m', 'no partner tracking') == 0

    def test_tier_explains_itself(self, db, capsys):
        hypothesis_id = _seed_hypothesis(db)
        run(db, 'tier', str(hypothesis_id))
        out = capsys.readouterr().out
        assert 'EXPLORATORY' in out
        assert 'not pre-registered' in out


class TestDecideRemovesTheFriction:
    def test_one_line_records_a_decision(self, db, capsys):
        """The absence of this is why every iteration sat at 'pending'."""
        from database.lab_notebook import LabNotebook
        nb = LabNotebook(db)
        iteration = nb.log_iteration('ephys.decode_location', {},
                                     {'status': 'success'}, session_id='20251210')
        assert run(db, 'decide', str(iteration.id), 'approved', '-m', 'real') == 0
        assert 'approved' in capsys.readouterr().out
        assert nb.iterations_for_session('20251210')[0].scientist_decision == 'approved'

    def test_summary_lists_pending_and_orphan_iterations(self, db, capsys):
        from database.lab_notebook import LabNotebook
        nb = LabNotebook(db)
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         session_id='20251210')
        run(db, 'summary')
        out = capsys.readouterr().out
        assert 'awaiting a scientist decision' in out
        assert 'attached to no hypothesis' in out


class TestReportCommands:
    def test_report_writes_a_file(self, db, tmp_path, capsys):
        hypothesis_id = _seed_hypothesis(db)
        out_dir = tmp_path / 'out'
        assert run(db, 'report', '--hypothesis', str(hypothesis_id),
                   '--out', str(out_dir)) == 0
        assert (out_dir / f'hypothesis_{hypothesis_id}.html').exists()

    def test_index_writes_a_file(self, db, tmp_path):
        _seed_hypothesis(db)
        out_dir = tmp_path / 'out'
        assert run(db, 'index', '--out', str(out_dir)) == 0
        assert (out_dir / 'index.html').exists()

    def test_report_without_a_target_is_an_error(self, db, tmp_path):
        assert run(db, 'report', '--out', str(tmp_path)) == 2

    def test_orphan_report_by_iteration_ids(self, db, tmp_path):
        from database.lab_notebook import LabNotebook
        nb = LabNotebook(db)
        first = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                 session_id='20251210')
        second = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                  session_id='20251210')
        out_dir = tmp_path / 'out'
        assert run(db, 'report', '--iterations', f'{first.id},{second.id}',
                   '--out', str(out_dir)) == 0
        assert list(out_dir.glob('iterations_*.html'))


class TestErrorHandling:
    def test_unknown_family_is_a_clean_error_not_a_traceback(self, db, capsys):
        assert run(db, 'denominator', '999') == 2
        assert 'ERROR' in capsys.readouterr().err

    def test_undeclared_test_is_refused_with_its_own_code(self, db, capsys):
        run(db, 'declare-family', 'fam', 'a', '--by', 'm')
        assert run(db, 'abandon-test', '1', 'nope', '--reason', 'r',
                   '--outcome-dependent') == 4
        assert 'REFUSED' in capsys.readouterr().err
