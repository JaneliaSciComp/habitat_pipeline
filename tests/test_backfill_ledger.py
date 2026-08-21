"""Tests for scripts/backfill_ledger.py.

The load-bearing test is `test_apply_never_mutates_an_iteration_row`. Iteration 12's
`excluded_objects` prose and its q=0.0387 *are* the evidence for the denominator
problem, so the backfill corrects the record by adding to `family_tests`, never by
rewriting what was logged. A full row hash is compared before and after.

`TestDryRunWritesNothing` is the other half: the pass is reviewed before it is
applied, so the dry run has to be genuinely inert.
"""
import hashlib
import json

import pytest

from database.lab_notebook import FamilyTest, Iteration, LabNotebook, TestFamily
from scripts.backfill_ledger import (
    A_PRIORI_EXCLUSION,
    ITER10_OBJECTS,
    OUTCOME_DEPENDENT_EXCLUSIONS,
    _differs_only_in_correction_method,
    denominator_table,
    main,
    survey,
)


# Iteration 12's real per-object p-values.
ITER12_PER_OBJECT = [
    {'object_name': '631', 'is_self': True, 'p_value': 0.022099447513812154,
     'q_value': 0.07734806629834254},
    {'object_name': '613', 'is_self': False, 'p_value': 0.0055248618784530384,
     'q_value': 0.03867403314917127},
    {'object_name': '616', 'is_self': False, 'p_value': 0.08839779005524862},
    {'object_name': '617', 'is_self': False, 'p_value': 0.13259668508287292},
    {'object_name': '633', 'is_self': False, 'p_value': 0.9447513812154696},
    {'object_name': '634', 'is_self': False, 'p_value': 0.2541436464088398},
    {'object_name': '635', 'is_self': False, 'p_value': 0.19337016574585636},
]


@pytest.fixture
def seeded(tmp_path):
    """A notebook shaped like the real one before the backfill."""
    nb = LabNotebook(tmp_path / 'notebook.db')

    nb.add_hypothesis('Fight outcome is decodable for animal 631')
    family_one = nb.create_test_family('phase1-verification, animal 631/20251216')
    # The empty duplicate that create_test_family's missing check produced.
    nb.create_test_family('phase1.5-verification 631/20251216')
    family_three = nb.create_test_family('phase1.5-verification 631/20251216')

    nb.log_iteration('ephys.decode_event_outcome', {'n_shuffles': 200},
                     {'status': 'success', 'population_accuracy_mean': 0.606,
                      'population_baseline_accuracy': 0.6315789473684211,
                      'p_value': 0.114},
                     animal_id='631', session_id='20251216', hypothesis_id=1,
                     test_family_id=family_one.id, git_commit='bd61df56')
    # Iterations 3 and 4: one question, two correction methods.
    for null_mode in ('per_cell', 'pooled'):
        nb.log_iteration('ephys.decode_opponent_identity',
                         {'behavior_type': 'EC', 'null_mode': null_mode},
                         {'status': 'success',
                          'population_accuracy_mean': 0.2881021938976933,
                          'p_value': 0.004975124378109453, 'best_cell_id': 862},
                         animal_id='631', session_id='20251216',
                         test_family_id=family_three.id, git_commit='37723225')
    nb.log_iteration('ephys.decode_location',
                     {'null': 'reverse', 'focal_animal': '631'},
                     {'status': 'success',
                      'per_object': [{'object_name': o} for o in ITER10_OBJECTS]},
                     animal_id='631', session_id='20251210', git_commit='c00a3cbe')
    nb.log_iteration('ephys.decode_location',
                     {'null': 'shuffle', 'n_shuffles': 180, 'focal_animal': '631'},
                     {'status': 'success', 'per_object': ITER12_PER_OBJECT},
                     animal_id='631', session_id='20251210', git_commit='0703456f')
    return nb


#: The one column the backfill may fill, and only when it is NULL.
_LINKABLE_COLUMN = 'test_family_id'


def _iteration_fingerprint(nb, *, exclude=()):
    """Hash every field of every iteration row, for before/after comparison."""
    skip = set(exclude)
    with nb.get_db_session() as db_session:
        rows = db_session.query(Iteration).order_by(Iteration.id).all()
        payload = [
            {c.name: str(getattr(row, c.name))
             for c in Iteration.__table__.columns if c.name not in skip}
            for row in rows
        ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


class TestDryRunWritesNothing:
    def test_row_counts_are_unchanged(self, seeded, capsys):
        with seeded.get_db_session() as s:
            before = (s.query(Iteration).count(), s.query(TestFamily).count(),
                      s.query(FamilyTest).count())
        assert main(['--db', str(seeded.db_path)]) == 0
        with seeded.get_db_session() as s:
            after = (s.query(Iteration).count(), s.query(TestFamily).count(),
                     s.query(FamilyTest).count())
        assert before == after

    def test_it_says_it_wrote_nothing(self, seeded, capsys):
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'DRY RUN' in out
        assert 'nothing written' in out

    def test_it_prints_the_denominator_table(self, seeded, capsys):
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'DENOMINATOR TABLE' in out
        # m=7 significant, m=11 not: the whole point of the pass.
        assert '0.0387' in out
        assert '0.0608' in out

    def test_it_surfaces_the_resolvable_flip(self, seeded, capsys):
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'permission to claim it' in out

    def test_it_lists_the_judgment_calls(self, seeded, capsys):
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'JUDGMENT CALLS' in out
        assert 'will not guess' in out
        for fragment in ('outcome-independent', 'one test or two',
                         'pending scientist decisions', 'falsifiers'):
            assert fragment in out

    def test_it_flags_pending_and_unattached_iterations(self, seeded, capsys):
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'no hypothesis' in out
        assert 'pending' in out


class TestApplyRefusesWithoutConfirmation:
    def test_it_exits_rather_than_guessing(self, seeded):
        with pytest.raises(SystemExit) as excinfo:
            main(['--db', str(seeded.db_path), '--apply', '--no-snapshot'])
        assert excinfo.value.code == 4

    def test_nothing_was_written_by_the_refusal(self, seeded):
        with seeded.get_db_session() as s:
            before = s.query(FamilyTest).count()
        with pytest.raises(SystemExit):
            main(['--db', str(seeded.db_path), '--apply', '--no-snapshot'])
        with seeded.get_db_session() as s:
            assert s.query(FamilyTest).count() == before


class TestApply:
    @pytest.fixture
    def applied(self, seeded):
        main(['--db', str(seeded.db_path), '--apply', '--no-snapshot',
              '--confirm-exclusion-classification'])
        return seeded

    def test_apply_never_rewrites_logged_evidence(self, seeded):
        """Every field except a NULL test_family_id must be byte-identical.

        The logged record is the evidence for the denominator problem, so the
        correction goes into `family_tests`. Filling a NULL foreign key is the
        one sanctioned change: it adds a pointer and alters no measurement.
        """
        before = _iteration_fingerprint(seeded, exclude=(_LINKABLE_COLUMN,))
        main(['--db', str(seeded.db_path), '--apply', '--no-snapshot',
              '--confirm-exclusion-classification'])
        assert _iteration_fingerprint(seeded, exclude=(_LINKABLE_COLUMN,)) == before

    def test_an_existing_test_family_id_is_never_overwritten(self, seeded):
        before = {row.id: row.test_family_id
                  for row in _all_iterations(seeded) if row.test_family_id}
        assert before, 'fixture should have at least one attached iteration'
        main(['--db', str(seeded.db_path), '--apply', '--no-snapshot',
              '--confirm-exclusion-classification'])
        after = {row.id: row.test_family_id for row in _all_iterations(seeded)}
        for iteration_id, family_id in before.items():
            assert after[iteration_id] == family_id

    def test_decode_location_iterations_get_linked(self, applied):
        """Otherwise the reconstruction is invisible from the iteration."""
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        linked = [row for row in _all_iterations(applied)
                  if row.analysis_module == 'ephys.decode_location']
        assert linked
        assert all(row.test_family_id == family.id for row in linked)

    def test_the_report_then_shows_the_declared_denominator(self, applied, tmp_path):
        """End-to-end: the correction has to reach the human-readable output."""
        from reports.hypothesis_report import (collect_orphan_report_data,
                                                render_html)
        ids = [row.id for row in _all_iterations(applied)
               if row.analysis_module == 'ephys.decode_location']
        text = render_html(collect_orphan_report_data(applied, ids, title='t'))
        assert 'No test family is attached' not in text
        assert 'q at declared m=11' in text
        assert '0.0387' in text     # as logged
        assert '0.0608' in text     # at the declared denominator

    def test_result_summary_is_untouched(self, applied):
        """Iteration 12's q=0.0387 stays exactly as logged."""
        rows = applied.iterations_for_session('20251210')
        latest = max(rows, key=lambda r: r.id)
        per_object = {e['object_name']: e for e in
                      latest.result_summary_dict()['per_object']}
        assert per_object['613']['q_value'] == pytest.approx(0.03867403314917127)

    def test_it_declares_all_twelve_objects(self, applied):
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        keys = {t.test_key for t in applied.family_tests(family.id)}
        assert keys == {f'object={o}' for o in ITER10_OBJECTS}

    def test_the_denominator_is_eleven(self, applied):
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        denom = applied.family_denominator(family.id)
        assert denom['n_declared'] == 12
        assert denom['n_excluded_prespecified'] == 1
        assert denom['n_tests_for_correction'] == 11

    def test_rat613_is_not_significant_at_that_denominator(self, applied):
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        result = applied.family_fdr(family.id)
        rat613 = result['per_test']['object=613']
        assert rat613['q_value'] == pytest.approx(0.0608, abs=1e-4)
        assert rat613['significant'] is False

    def test_the_outcome_dependent_drops_are_recorded(self, applied):
        """`reconstructed` wins the status, but the drops are still enumerated.

        The precedence is deliberate: "the denominator is unrecoverable" is a
        strictly weaker claim than "the denominator is known but was selected on
        the outcome", so it dominates the single status string. The
        outcome-dependent exclusions remain listed in the dict (and in the
        report's exclusions table), so nothing is lost.
        """
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        denom = applied.family_denominator(family.id)
        assert denom['denominator_status'] == 'reconstructed'
        dropped = {e['test_key'] for e in denom['outcome_dependent_exclusions']}
        assert dropped == {f'object={o}' for o in OUTCOME_DEPENDENT_EXCLUSIONS}

    def test_a_non_reconstructed_family_reports_the_drops_as_its_status(self, seeded):
        """Without the reconstructed flag, the outcome-dependent status surfaces."""
        family = seeded.get_or_create_test_family('freshly declared')
        seeded.declare_family_tests(family.id, ['a', 'b'], declared_by='t')
        seeded.record_family_test(family.id, 'a', p_value=0.01)
        seeded.abandon_family_test(family.id, 'b', reason='no margin',
                                   outcome_dependent=True)
        assert seeded.family_denominator(family.id)['denominator_status'] == \
            'outcome_dependent_exclusions'

    def test_the_a_priori_exclusion_records_both_flags(self, applied):
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        test = next(t for t in applied.family_tests(family.id)
                    if t.test_key == f'object={A_PRIORI_EXCLUSION}')
        assert test.status == 'excluded_prespecified'
        assert test.criterion_available_a_priori is True
        assert test.applied_after_seeing_results is True
        assert test.exclusion_outcome_dependent is False

    def test_the_reconstructed_status_is_recorded_on_declaration(self, applied):
        """A recovered denominator is a lower bound, and must say so."""
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        with applied.get_db_session() as s:
            assert s.get(TestFamily, family.id).denominator_status in (
                'reconstructed', 'outcome_dependent_exclusions')

    def test_the_empty_duplicate_family_is_invalidated(self, applied):
        invalidated = [f for f in _families(applied) if f.status == 'invalidated']
        assert invalidated
        assert 'phase1.5' in invalidated[0].name

    def test_it_never_invalidates_a_reconstruction_family(self, seeded):
        """Regression: an iterations-only emptiness check ate its own output.

        The reconstruction families deliberately hold `family_tests` and no
        iterations — the iterations still point at the original families — so a
        rule that invalidates "any family with zero iterations" marks exactly
        the rows the pass just created. It only showed up on a second run,
        because the first run's families aren't in the first run's survey.
        """
        args = ['--db', str(seeded.db_path), '--apply', '--no-snapshot',
                '--confirm-exclusion-classification']
        main(args)
        main(args)   # the second pass is where the bug appeared

        for family in _families(seeded):
            if 'reconstructed' in family.name:
                assert family.status != 'invalidated', (
                    f"family {family.id} ({family.name!r}) was invalidated despite "
                    "being a reconstruction family")

    def test_a_family_with_declared_tests_is_never_invalidated(self, seeded):
        """Zero iterations is not emptiness; zero of *both* is."""
        family = seeded.get_or_create_test_family('has tests, no iterations')
        seeded.declare_family_tests(family.id, ['a', 'b'], declared_by='t')
        main(['--db', str(seeded.db_path), '--apply', '--no-snapshot',
              '--confirm-exclusion-classification'])
        with seeded.get_db_session() as s:
            assert s.get(TestFamily, family.id).status != 'invalidated'

    def test_the_ledger_reads_the_honest_denominator_end_to_end(self, applied):
        """The whole point, through the public API."""
        family = next(f for f in _families(applied) if 'decode_location' in f.name)
        result = applied.family_fdr(family.id, n_shuffles=180)
        assert result['n_tests_for_correction'] == 11
        assert result['n_padded'] == 4
        assert result['fdr_resolution']['resolvable'] is False
        assert result['fdr_resolution']['recommended_n_shuffles'] == 220
        assert not any(v['significant'] for v in result['per_test'].values())

    def test_it_is_idempotent(self, seeded):
        """Re-running must not double-declare or raise."""
        args = ['--db', str(seeded.db_path), '--apply', '--no-snapshot',
                '--confirm-exclusion-classification']
        main(args)
        family = next(f for f in _families(seeded) if 'decode_location' in f.name)
        first = seeded.family_denominator(family.id)['n_tests_for_correction']
        main(args)
        assert seeded.family_denominator(family.id)['n_tests_for_correction'] == first

    def test_a_snapshot_is_taken_by_default(self, seeded, capsys):
        main(['--db', str(seeded.db_path), '--apply',
              '--confirm-exclusion-classification'])
        assert 'snapshot written' in capsys.readouterr().out
        assert list(seeded.db_path.parent.glob('*.bak_*'))


def _families(nb):
    with nb.get_db_session() as db_session:
        return db_session.query(TestFamily).order_by(TestFamily.id).all()


def _all_iterations(nb):
    with nb.get_db_session() as db_session:
        return db_session.query(Iteration).order_by(Iteration.id).all()


class TestRunIdentityReporting:
    def test_correction_only_difference_is_recognised(self, seeded):
        state = survey(seeded)
        ids = [r['id'] for r in state['iterations']
               if r['params'].get('null_mode')]
        assert len(ids) == 2
        assert _differs_only_in_correction_method(state, ids) is True

    def test_a_substantive_difference_is_not(self, seeded):
        state = survey(seeded)
        ids = [r['id'] for r in state['iterations']
               if r['module'] == 'ephys.decode_location']
        assert _differs_only_in_correction_method(state, ids) is False

    def test_the_report_explains_rather_than_alarms(self, seeded, capsys):
        """Identical accuracy across null_modes is expected, not a duplicate."""
        main(['--db', str(seeded.db_path)])
        out = capsys.readouterr().out
        assert 'correction method only' in out
        assert 'probable_duplicate' not in out


class TestDenominatorTable:
    def test_it_reports_the_flip(self, capsys):
        p_values = {e['object_name']: e['p_value'] for e in ITER12_PER_OBJECT}
        denominator_table(p_values, 180, (7, 11))
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip().startswith(('7', '11'))]
        assert any('True' in line for line in lines)
        assert any('False' in line for line in lines)
