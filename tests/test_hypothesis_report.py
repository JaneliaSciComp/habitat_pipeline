"""Tests for the per-hypothesis HTML report.

The load-bearing test is `TestTheBanner`: the "hypothesis-generating only"
sentence must appear whenever the evidence is exploratory, disappear only for a
genuinely confirmatory case, and reappear when any single one of the tier's
preconditions is removed. Because the tier is derived rather than stored, that
parametrization is what proves the banner cannot be dismissed by editing a
field.

`TestSelfContainment` guards the other durability property: a report has to be
readable years later from the file alone, so no CDN, no web font, no external
image.
"""
import re
from datetime import datetime

import pytest

from database.lab_notebook import LabNotebook
from reports.hypothesis_report import (
    HYPOTHESIS_GENERATING_ONLY,
    SECTIONS,
    build_hypothesis_report,
    build_index_report,
    build_orphan_report,
    collect_index_data,
    collect_orphan_report_data,
    collect_report_data,
    render_html,
    render_index_html,
)

FIXED_NOW = datetime(2026, 8, 20, 12, 0, 0)


@pytest.fixture
def nb(tmp_path):
    return LabNotebook(tmp_path / 'notebook.db')


def _render(nb, hypothesis_id, **kwargs):
    data = collect_report_data(nb, hypothesis_id, now=FIXED_NOW,
                              git_commit='deadbeef', **kwargs)
    return render_html(data)


def build_confirmatory(nb, *, post_hoc=False, skip_prediction=False, skip_unlock=False,
                       skip_seed=False, skip_fingerprint=False, skip_decision=False,
                       dirty_denominator=False, miss_threshold=False):
    """The full honest path, optionally with one condition broken."""
    hypothesis = nb.add_hypothesis('Opponent identity generalizes to a held-out session')
    reservation = nb.reserve_holdout('20251218', cohort='cohort7', reason='confirmation',
                                     reserved_by='misha')
    family = nb.get_or_create_test_family(f'confirmation h{hypothesis.id}')
    nb.declare_family_tests(family.id, ['opponent=EC:8way', 'opponent=EC:group'],
                            declared_by='misha')

    prediction = None
    if not skip_prediction:
        prediction = nb.freeze_prediction(
            hypothesis.id, statistic='min_q_value', direction='lt', threshold=0.05,
            falsifier='min q >= 0.05 on the held-out session', n_shuffles_planned=500,
            registered_post_hoc=post_hoc)
        if not skip_unlock and not post_hoc:
            nb.unlock_holdout(reservation.id, hypothesis.id, approved_by='misha',
                              frozen_prediction_id=prediction.id)

    nb.record_family_test(family.id, 'opponent=EC:8way',
                          p_value=0.5 if miss_threshold else 0.001,
                          git_commit='deadbeef')
    if dirty_denominator:
        nb.abandon_family_test(family.id, 'opponent=EC:group',
                               reason='no margin once we looked', outcome_dependent=True)
    else:
        nb.record_family_test(family.id, 'opponent=EC:group', p_value=0.6,
                              git_commit='deadbeef')

    iteration = nb.log_iteration(
        'ephys.decode_opponent_identity', {'behavior_type': 'EC', 'n_shuffles': 500},
        {'status': 'success', 'population_accuracy_mean': 0.31,
         'population_baseline_accuracy': 0.27},
        animal_id='631', session_id='20251218', hypothesis_id=hypothesis.id,
        test_family_id=family.id, seed=None if skip_seed else 0,
        dataset_fingerprint=None if skip_fingerprint else 'abc123')
    if not skip_decision:
        nb.record_decision(iteration.id, 'approved')
    return hypothesis, iteration, family


class TestAllEightSectionsAlwaysRender:
    def test_every_anchor_is_present_for_an_empty_hypothesis(self, nb):
        """An omitted section is itself a claim about the work."""
        hypothesis = nb.add_hypothesis('nothing has been run for this')
        text = _render(nb, hypothesis.id)
        for anchor, title in SECTIONS:
            assert f'id="{anchor}"' in text, f'section {anchor} missing'
            assert title in text

    def test_sections_appear_in_the_fixed_order(self, nb):
        hypothesis = nb.add_hypothesis('x')
        text = _render(nb, hypothesis.id)
        positions = [text.index(f'id="{anchor}"') for anchor, _ in SECTIONS]
        assert positions == sorted(positions)

    def test_empty_sections_say_what_is_missing(self, nb):
        hypothesis = nb.add_hypothesis('x')
        text = _render(nb, hypothesis.id)
        assert 'No frozen prediction exists' in text
        assert 'No iterations are logged' in text
        assert 'No figures were saved' in text
        assert 'No verdict recorded' in text
        assert 'not literature-grounded' in text

    def test_threats_and_falsifier_are_never_empty(self, nb):
        hypothesis = nb.add_hypothesis('x')
        data = collect_report_data(nb, hypothesis.id, now=FIXED_NOW)
        assert data.threats
        assert data.falsifiers

    def test_unknown_hypothesis_raises(self, nb):
        with pytest.raises(ValueError, match='No hypothesis'):
            collect_report_data(nb, 999)


class TestTheBanner:
    def test_present_for_an_exploratory_hypothesis(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert HYPOTHESIS_GENERATING_ONLY in _render(nb, hypothesis.id)

    def test_rendered_verbatim(self, nb):
        """The scope decision was that reports say this in those words."""
        hypothesis = nb.add_hypothesis('x')
        assert ("Without the holdout, the loop's output is hypothesis-generating only."
                in _render(nb, hypothesis.id).replace('&#x27;', "'"))

    def test_absent_for_a_genuinely_confirmatory_case(self, nb):
        hypothesis, _, _ = build_confirmatory(nb)
        assert HYPOTHESIS_GENERATING_ONLY not in _render(nb, hypothesis.id)

    @pytest.mark.parametrize('breakage', [
        {'skip_prediction': True},
        {'post_hoc': True},
        {'skip_unlock': True},
        {'skip_seed': True},
        {'skip_fingerprint': True},
        {'skip_decision': True},
        {'dirty_denominator': True},
        {'miss_threshold': True},
    ])
    def test_reappears_when_any_condition_is_removed(self, nb, breakage):
        hypothesis, _, _ = build_confirmatory(nb, **breakage)
        assert HYPOTHESIS_GENERATING_ONLY in _render(nb, hypothesis.id), (
            f'{breakage} suppressed the banner')

    def test_appears_at_the_top_and_in_the_verdict_section(self, nb):
        hypothesis = nb.add_hypothesis('x')
        text = _render(nb, hypothesis.id)
        assert text.count(HYPOTHESIS_GENERATING_ONLY) >= 2

    def test_unmet_conditions_are_listed(self, nb):
        hypothesis, _, _ = build_confirmatory(nb, skip_decision=True)
        assert 'no durable scientist decision' in _render(nb, hypothesis.id)

    def test_there_is_no_way_to_suppress_it(self):
        """No parameter may exist to turn the banner off."""
        import inspect
        for func in (render_html, collect_report_data, build_hypothesis_report):
            params = set(inspect.signature(func).parameters)
            assert not params & {'banner', 'show_banner', 'suppress_banner', 'quiet'}


class TestRefutationsAreAsProminent:
    def test_refuted_shares_the_supported_css_class(self, nb):
        """Same box, same size, same weight - only the hue differs."""
        supported = nb.add_hypothesis('a')
        nb.record_verdict(supported.id, verdict='supported', rationale='clears threshold')
        refuted = nb.add_hypothesis('b')
        nb.record_verdict(refuted.id, verdict='refuted', rationale='below its baseline')

        supported_html = _render(nb, supported.id)
        refuted_html = _render(nb, refuted.id)
        assert 'class="verdict supported"' in supported_html
        assert 'class="verdict refuted"' in refuted_html
        # The shared class carries the box geometry; the variant only recolours.
        assert 'class="verdict ' in supported_html and 'class="verdict ' in refuted_html
        assert '<span class="label">' in refuted_html

    def test_the_verdict_label_is_shown(self, nb):
        hypothesis = nb.add_hypothesis('x')
        nb.record_verdict(hypothesis.id, verdict='refuted', rationale='r')
        assert 'refuted' in _render(nb, hypothesis.id)

    def test_index_lists_refuted_and_blocked(self, nb):
        refuted = nb.add_hypothesis('refuted one')
        nb.record_verdict(refuted.id, verdict='refuted', rationale='r')
        nb.set_hypothesis_status(refuted.id, 'refuted')
        blocked = nb.add_hypothesis('blocked one')
        nb.record_verdict(blocked.id, verdict='blocked', rationale='no data')
        nb.set_hypothesis_status(blocked.id, 'blocked')

        text = render_index_html(collect_index_data(nb, now=FIXED_NOW))
        assert 'refuted one' in text
        assert 'blocked one' in text
        assert 'refuted' in text and 'blocked' in text

    def test_index_surfaces_iterations_with_no_hypothesis(self, nb):
        """A dropped line of work is only visible in the collection."""
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         session_id='20251210')
        index = collect_index_data(nb, now=FIXED_NOW)
        assert index['orphan_iteration_ids']
        assert 'attached to no hypothesis' in render_index_html(index)

    def test_index_banner_when_nothing_is_confirmatory(self, nb):
        nb.add_hypothesis('x')
        assert HYPOTHESIS_GENERATING_ONLY in render_index_html(
            collect_index_data(nb, now=FIXED_NOW))

    def test_empty_index_is_explicit(self, nb):
        assert 'No hypotheses are registered' in render_index_html(
            collect_index_data(nb, now=FIXED_NOW))


class TestDenominatorSubBlock:
    def test_declared_denominator_is_always_stated(self, nb):
        hypothesis, _, _ = build_confirmatory(nb)
        text = _render(nb, hypothesis.id)
        assert 'Denominator' in text
        assert 'tests used for' in text

    def test_shows_q_as_logged_beside_q_at_the_declared_denominator(self, nb):
        """The two columns that make a post-hoc shrink visible."""
        hypothesis = nb.add_hypothesis('partner position is decodable')
        family = nb.get_or_create_test_family('decode_location family')
        nb.declare_family_tests(family.id, [f'object={o}' for o in
                                            ['613', '615', '616', '617', '620', '621',
                                             '629', '630', '631', '633', '634', '635']],
                                declared_by='backfill')
        nb.record_family_test(family.id, 'object=613', p_value=1.0 / 181,
                              git_commit='0703456')
        nb.abandon_family_test(family.id, 'object=630', reason='near-stationary',
                               outcome_dependent=False,
                               criterion_available_a_priori=True)
        for obj in ['615', '620', '621', '629']:
            nb.abandon_family_test(family.id, f'object={obj}', reason='no margin',
                                   outcome_dependent=True)
        nb.log_iteration(
            'ephys.decode_location', {'n_shuffles': 180, 'null': 'shuffle'},
            {'status': 'success',
             'per_object': [{'object_name': '613', 'q_value': 0.03867403314917127}]},
            session_id='20251210', hypothesis_id=hypothesis.id,
            test_family_id=family.id)

        text = _render(nb, hypothesis.id)
        assert 'q as logged' in text
        assert 'q at declared m=11' in text
        assert '0.0387' in text          # what was reported
        assert '0.0608' in text          # what the declared denominator gives

    def test_outcome_dependent_exclusions_are_flagged(self, nb):
        hypothesis, _, _ = build_confirmatory(nb, dirty_denominator=True)
        text = _render(nb, hypothesis.id)
        assert 'Exclusions' in text
        assert 'outcome-dependent' in text

    def test_unresolvable_budget_is_called_out(self, nb):
        hypothesis = nb.add_hypothesis('x')
        family = nb.get_or_create_test_family('under-resolved')
        nb.declare_family_tests(family.id, [f'cell={i}' for i in range(149)],
                                declared_by='t')
        nb.record_family_test(family.id, 'cell=0', p_value=1.0 / 201)
        nb.log_iteration('ephys.decode_opponent_identity', {'n_shuffles': 200},
                         {'status': 'success'}, hypothesis_id=hypothesis.id,
                         test_family_id=family.id, session_id='20251216')
        text = _render(nb, hypothesis.id)
        assert 'NOT resolvable' in text
        assert 'uninformative, not evidence of absence' in text

    def test_undeclared_family_reports_an_unrecorded_denominator(self, nb):
        """Legacy families predate the ledger; the denominator is unknown, not zero."""
        hypothesis = nb.add_hypothesis('x')
        family = nb.create_test_family('legacy family')
        nb.log_iteration('ephys.decode_opponent_identity', {'n_shuffles': 200},
                         {'status': 'success', 'p_value': 0.005},
                         hypothesis_id=hypothesis.id, test_family_id=family.id,
                         session_id='20251216')
        text = _render(nb, hypothesis.id)
        assert 'declares no tests' in text
        assert 'unrecorded' in text

    def test_reconstructed_denominator_says_it_is_a_lower_bound(self, nb):
        hypothesis = nb.add_hypothesis('x')
        family = nb.get_or_create_test_family('reconstructed family')
        nb.declare_family_tests(family.id, ['a', 'b'], declared_by='backfill',
                                denominator_status='reconstructed')
        nb.record_family_test(family.id, 'a', p_value=0.01)
        nb.log_iteration('ephys.decode_opponent_identity', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, test_family_id=family.id,
                         session_id='20251216')
        text = _render(nb, hypothesis.id)
        assert 'reconstructed after the fact' in text
        assert 'lower bound' in text


class TestThreatChecklist:
    def test_below_baseline_accuracy_is_flagged(self, nb):
        hypothesis = nb.add_hypothesis('outcome is decodable')
        nb.log_iteration('ephys.decode_event_outcome', {},
                         {'status': 'success', 'population_accuracy_mean': 0.606,
                          'population_baseline_accuracy': 0.6315789473684211},
                         hypothesis_id=hypothesis.id, session_id='20251216')
        data = collect_report_data(nb, hypothesis.id, now=FIXED_NOW)
        joined = ' '.join(data.threats)
        assert 'below its majority-class baseline' in joined
        assert 'not 1/n_classes' in joined

    def test_pending_decision_is_flagged(self, nb):
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251210')
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'no durable scientist decision' in joined

    def test_missing_seed_is_flagged(self, nb):
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251210')
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'cannot be reproduced' in joined

    def test_single_session_and_animal_are_flagged(self, nb):
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         animal_id='631', session_id='20251210',
                         hypothesis_id=hypothesis.id)
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'Single session' in joined
        assert 'Single animal' in joined

    def test_pinned_p_value_is_flagged(self, nb):
        hypothesis = nb.add_hypothesis('x')
        family = nb.get_or_create_test_family('pinned')
        nb.declare_family_tests(family.id, ['a'], declared_by='t')
        nb.record_family_test(family.id, 'a', p_value=1.0 / 181)
        nb.log_iteration('ephys.decode_location', {'n_shuffles': 180},
                         {'status': 'success'}, hypothesis_id=hypothesis.id,
                         test_family_id=family.id, session_id='20251210')
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'pinned at the permutation floor' in joined
        assert 'bound, not a measurement' in joined

    def test_mixed_commits_are_flagged(self, nb):
        hypothesis = nb.add_hypothesis('x')
        family = nb.get_or_create_test_family('drift')
        nb.declare_family_tests(family.id, ['a', 'b'], declared_by='t')
        nb.record_family_test(family.id, 'a', p_value=0.01, git_commit='c00a3cb')
        nb.record_family_test(family.id, 'b', p_value=0.02, git_commit='0703456')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, test_family_id=family.id,
                         session_id='20251210')
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'not the same statistic' in joined

    def test_independent_recomputation_is_named_as_a_human_step(self, nb):
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251210')
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert 'independently' in joined and 'recomputed' in joined


class TestFigures:
    def test_a_missing_file_renders_a_placeholder_not_an_error(self, nb, tmp_path):
        """Exactly the fate of this notebook's one recorded figure."""
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_opponent_identity', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251216',
                         figure_paths=[str(tmp_path / 'reaped' / 'summary.png')])
        text = _render(nb, hypothesis.id)
        assert 'no longer exists' in text
        assert 'temporary scratchpad' in text

    def test_an_existing_png_is_embedded_as_a_data_uri(self, nb, tmp_path):
        png = tmp_path / 'fig.png'
        # Smallest valid PNG.
        png.write_bytes(bytes.fromhex(
            '89504e470d0a1a0a0000000d494844520000000100000001080600000'
            '01f15c4890000000a49444154789c6300010000050001'
            '0d0a2db40000000049454e44ae426082'))
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_opponent_identity', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251216',
                         figure_paths=[str(png)])
        text = _render(nb, hypothesis.id)
        assert 'data:image/png;base64,' in text

    def test_decoding_plot_chance_line_caveat_is_added(self, nb, tmp_path):
        png = tmp_path / 'decoding_summary.png'
        png.write_bytes(b'\x89PNG\r\n\x1a\n')
        hypothesis = nb.add_hypothesis('x')
        nb.log_iteration('ephys.decode_opponent_identity', {}, {'status': 'success'},
                         hypothesis_id=hypothesis.id, session_id='20251216',
                         figure_paths=[str(png)])
        joined = ' '.join(collect_report_data(nb, hypothesis.id).threats)
        assert '1/n_classes' in joined


class TestSelfContainment:
    def test_no_external_references(self, nb):
        hypothesis, _, _ = build_confirmatory(nb)
        text = _render(nb, hypothesis.id).lower()
        for forbidden in ('src="http', "src='http", 'href="http', 'cdn',
                          '<script src', '@import', 'fonts.googleapis'):
            assert forbidden not in text, f'found external reference: {forbidden}'

    def test_no_script_tags_at_all(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert '<script' not in _render(nb, hypothesis.id).lower()

    def test_declares_a_utf8_charset(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert '<meta charset="utf-8">' in _render(nb, hypothesis.id)


class TestEscapingAndEncoding:
    def test_html_in_a_statement_is_escaped(self, nb):
        hypothesis = nb.add_hypothesis('<script>alert(1)</script>')
        text = _render(nb, hypothesis.id)
        assert '<script>alert(1)</script>' not in text
        assert '&lt;script&gt;' in text

    def test_quotes_are_escaped(self, nb):
        hypothesis = nb.add_hypothesis('a "quoted" claim')
        assert '&quot;quoted&quot;' in _render(nb, hypothesis.id)

    def test_non_ascii_round_trips(self, nb, tmp_path):
        """Hypothesis 3's statement already carries a U+FFFD from a prior mojibake."""
        hypothesis = nb.add_hypothesis('tuning � for the café rat — dash')
        path = build_hypothesis_report(hypothesis.id, notebook=nb, out_dir=tmp_path,
                                       now=FIXED_NOW, git_commit='x')
        text = path.read_text(encoding='utf-8')
        assert 'café' in text
        assert '�' in text


class TestDeterminism:
    def test_identical_input_renders_identical_bytes(self, nb):
        hypothesis, _, _ = build_confirmatory(nb)
        first = _render(nb, hypothesis.id)
        second = _render(nb, hypothesis.id)
        assert first == second

    def test_injected_timestamp_is_used(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert '2026-08-20 12:00:00' in _render(nb, hypothesis.id)

    def test_injected_commit_is_used(self, nb):
        hypothesis = nb.add_hypothesis('x')
        assert 'deadbeef' in _render(nb, hypothesis.id)


class TestOrphanReports:
    def test_renders_iterations_with_no_hypothesis(self, nb):
        """Most of this notebook's iterations, including its headline result."""
        first = nb.log_iteration('ephys.decode_location', {'null': 'reverse'},
                                 {'status': 'success'}, session_id='20251210')
        second = nb.log_iteration('ephys.decode_location', {'null': 'shuffle'},
                                  {'status': 'success'}, session_id='20251210')
        data = collect_orphan_report_data(nb, [first.id, second.id],
                                          title='exploratory sweep', now=FIXED_NOW)
        text = render_html(data)
        assert data.is_orphan is True
        assert 'not attached to any hypothesis' in text
        for anchor, _ in SECTIONS:
            assert f'id="{anchor}"' in text

    def test_orphan_report_carries_the_banner(self, nb):
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='20251210')
        text = render_html(collect_orphan_report_data(
            nb, [iteration.id], title='t', now=FIXED_NOW))
        assert HYPOTHESIS_GENERATING_ONLY in text

    def test_unknown_iteration_raises(self, nb):
        with pytest.raises(ValueError, match='No iteration'):
            collect_orphan_report_data(nb, [999], title='t')


class TestFileOutput:
    def test_writes_utf8_and_returns_the_path(self, nb, tmp_path):
        hypothesis = nb.add_hypothesis('x')
        path = build_hypothesis_report(hypothesis.id, notebook=nb, out_dir=tmp_path,
                                       now=FIXED_NOW, git_commit='x')
        assert path.exists()
        assert path.name == f'hypothesis_{hypothesis.id}.html'
        assert path.read_text(encoding='utf-8').startswith('<!DOCTYPE html>')

    def test_index_links_to_each_report(self, nb, tmp_path):
        hypothesis = nb.add_hypothesis('x')
        build_hypothesis_report(hypothesis.id, notebook=nb, out_dir=tmp_path,
                                now=FIXED_NOW, git_commit='x')
        index = build_index_report(notebook=nb, out_dir=tmp_path, now=FIXED_NOW,
                                   git_commit='x')
        assert f'hypothesis_{hypothesis.id}.html' in index.read_text(encoding='utf-8')

    def test_orphan_report_filename(self, nb, tmp_path):
        iteration = nb.log_iteration('ephys.decode_location', {}, {'status': 'success'},
                                     session_id='20251210')
        path = build_orphan_report([iteration.id], title='t', notebook=nb,
                                   out_dir=tmp_path, now=FIXED_NOW, git_commit='x')
        assert path.name == f'iterations_{iteration.id}.html'

    def test_creates_the_output_directory(self, nb, tmp_path):
        hypothesis = nb.add_hypothesis('x')
        nested = tmp_path / 'a' / 'b'
        path = build_hypothesis_report(hypothesis.id, notebook=nb, out_dir=nested,
                                       now=FIXED_NOW, git_commit='x')
        assert path.parent == nested
