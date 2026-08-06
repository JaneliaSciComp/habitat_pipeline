"""Tests for database.lab_notebook.LabNotebook — the append-only provenance
layer for the AI-in-the-loop discovery platform (see
docs/AI_DISCOVERY_LOOP_DESIGN.md §6 and docs/PHASE0_FINDINGS.md).
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from database.lab_notebook import LabNotebook, _json_safe


@pytest.fixture
def notebook():
    temp_dir = Path(tempfile.mkdtemp())
    nb = LabNotebook(temp_dir / "notebook.db")
    yield nb
    nb.engine.dispose()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestJsonSafe:
    def test_numpy_scalars_convert(self):
        safe = _json_safe({'a': np.float64(0.5), 'b': np.int64(3), 'c': np.bool_(True)})
        assert safe == {'a': 0.5, 'b': 3, 'c': True}
        assert isinstance(safe['a'], float) and isinstance(safe['b'], int) and isinstance(safe['c'], bool)

    def test_numpy_array_and_str_keys(self):
        safe = _json_safe({np.str_('rat613'): np.array([1, 2, 3])})
        assert safe == {'rat613': [1, 2, 3]}

    def test_nested_structures(self):
        safe = _json_safe({'outer': [{'x': np.float32(1.5)}, (np.int32(2), 'y')]})
        assert safe == {'outer': [{'x': pytest.approx(1.5)}, [2, 'y']]}


class TestHypothesisAndTestFamily:
    def test_add_hypothesis_round_trips(self, notebook):
        hyp = notebook.add_hypothesis(
            "cell X encodes opponent identity",
            predicted_effect="above-chance LDA accuracy",
            chosen_test="population per-cell LDA vs. label-permutation null",
            citations=[{'source': 'bioRxiv', 'id': '10.1101/xyz'}],
        )
        assert hyp.id is not None
        assert hyp.status == 'proposed'

    def test_create_test_family_defaults(self, notebook):
        family = notebook.create_test_family("opponent-identity campaign, animal 631")
        assert family.correction_method == 'bh_fdr'
        assert family.alpha == 0.05


class TestIterationLogging:
    def test_log_iteration_round_trips(self, notebook):
        hyp = notebook.add_hypothesis("test hypothesis")
        family = notebook.create_test_family("family A")

        iteration = notebook.log_iteration(
            'ephys.decode_opponent_identity',
            params={'animal_id': '631', 'n_shuffles': 200},
            result_summary={
                'status': 'success',
                'population_accuracy_mean': np.float64(0.252),
                'unique_classes': np.array(['rat613', 'rat616']),
                'p_value': 0.01,
            },
            animal_id='631', session_id='20251216',
            hypothesis_id=hyp.id, test_family_id=family.id,
            figure_paths=[Path('a.png'), Path('b.png')],
        )

        assert iteration.id is not None
        assert iteration.git_commit is not None  # this repo has a git history
        assert iteration.status == 'success'
        assert iteration.scientist_decision == 'pending'
        assert iteration.result_summary_dict()['population_accuracy_mean'] == pytest.approx(0.252)
        assert iteration.result_summary_dict()['unique_classes'] == ['rat613', 'rat616']
        assert iteration.figure_paths_list() == ['a.png', 'b.png']

    def test_log_iteration_status_falls_back_to_result_summary(self, notebook):
        iteration = notebook.log_iteration(
            'ephys.decode_event_outcome', params={}, result_summary={'status': 'failed'},
        )
        assert iteration.status == 'failed'

    def test_explicit_git_commit_is_not_overridden(self, notebook):
        iteration = notebook.log_iteration(
            'ephys.decode_event_outcome', params={}, result_summary={},
            git_commit='deadbeef',
        )
        assert iteration.git_commit == 'deadbeef'

    def test_record_decision(self, notebook):
        iteration = notebook.log_iteration('ephys.decode_event_outcome', params={}, result_summary={})
        updated = notebook.record_decision(iteration.id, 'approved', notes='looks real')
        assert updated.scientist_decision == 'approved'
        assert updated.decision_notes == 'looks real'
        assert updated.decision_at is not None

    def test_record_decision_rejects_bad_value(self, notebook):
        iteration = notebook.log_iteration('ephys.decode_event_outcome', params={}, result_summary={})
        with pytest.raises(ValueError):
            notebook.record_decision(iteration.id, 'maybe')

    def test_record_decision_unknown_iteration_raises(self, notebook):
        with pytest.raises(ValueError):
            notebook.record_decision(9999, 'approved')


class TestQueries:
    def test_iterations_for_session(self, notebook):
        notebook.log_iteration('mod.a', params={}, result_summary={}, session_id='20251216')
        notebook.log_iteration('mod.b', params={}, result_summary={}, session_id='20251216')
        notebook.log_iteration('mod.c', params={}, result_summary={}, session_id='other')

        rows = notebook.iterations_for_session('20251216')
        assert {r.analysis_module for r in rows} == {'mod.a', 'mod.b'}

    def test_iterations_for_hypothesis(self, notebook):
        hyp = notebook.add_hypothesis("h")
        notebook.log_iteration('mod.a', params={}, result_summary={}, hypothesis_id=hyp.id)
        notebook.log_iteration('mod.b', params={}, result_summary={})

        rows = notebook.iterations_for_hypothesis(hyp.id)
        assert len(rows) == 1
        assert rows[0].analysis_module == 'mod.a'


class TestCampaignFDR:
    def test_recompute_family_significance(self, notebook):
        family = notebook.create_test_family("campaign")
        it1 = notebook.log_iteration('mod.a', params={}, result_summary={'p_value': 0.01},
                                     test_family_id=family.id)
        it2 = notebook.log_iteration('mod.b', params={}, result_summary={'p_value': 0.2},
                                     test_family_id=family.id)
        # missing p_value_key -> skipped, not an error
        notebook.log_iteration('mod.c', params={}, result_summary={}, test_family_id=family.id)

        q = notebook.recompute_family_significance(family.id)
        assert set(q.keys()) == {it1.id, it2.id}
        assert q[it1.id] < q[it2.id]
        assert all(0.0 <= v <= 1.0 for v in q.values())

    def test_recompute_family_significance_empty_family(self, notebook):
        family = notebook.create_test_family("empty campaign")
        assert notebook.recompute_family_significance(family.id) == {}

    def test_recompute_family_significance_unknown_family_raises(self, notebook):
        with pytest.raises(ValueError):
            notebook.recompute_family_significance(9999)
