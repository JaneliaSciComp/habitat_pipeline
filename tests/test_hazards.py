"""Tests for the hazard registry (discovery/hazards.json + discovery/hazards.py).

Two jobs here.

`TestRegistryIntegrity` is the anti-rot guard: a hazard whose detector points
at a renamed function or a deleted test is a decoration that looks like a
detector, which is worse than an honest prose entry. It also enforces the
registry's own earn-your-place rule.

`TestCannotCheckIsNotAPass` pins the module's single most important
invariant: a detector that cannot run reports `ran=False, passed=None`, never
`passed=True`. Collapsing "could not check" into "passed" is what would make
this whole layer report safety it never verified.
"""
import json

import pytest

from discovery._predicates import (
    MissingValue,
    OPS,
    PredicateError,
    evaluate_predicate,
    extract_path,
)
from discovery.hazards import (
    DEFAULT_HAZARDS_PATH,
    DETECTOR_TYPES,
    KINDS,
    SEVERITIES,
    STAGES,
    Hazard,
    HazardRegistryError,
    hazards_by_id,
    hazards_for,
    load_hazards,
    render_digest,
    resolve_callable,
    run_detector,
    validate_registry,
)


# The hazards that must exist because they encode a real, documented failure
# from this project. If one of these disappears, the registry has lost an
# entry that was paid for with a wrong result.
REQUIRED_HAZARD_IDS = (
    'HZ-STAT-001',   # 149 cells x 200 shuffles -> best q 0.74, reported as biology
    'HZ-STAT-002',   # 60.6% accuracy under a 63.2% majority baseline
    'HZ-STAT-005',   # KFold(shuffle=True) leaking autocorrelated position bins
    'HZ-STAT-006',   # null='reverse' is order-only, not a chance null
    'HZ-STAT-007',   # near-stationary target -> spurious decoding win
    'HZ-STAT-008',   # p pinned at the permutation floor
    'HZ-STAT-011',   # post-hoc denominator shrink
    'HZ-STAT-013',   # undeclared class subset / relaxed min_events_per_class
    'HZ-DATA-001',   # tracking with only the focal animal identity-resolved
    'HZ-DATA-002',   # tracking covering only part of the recording
    'HZ-DATA-007',   # three versions of one session's scoring on disk
    'HZ-API-003',    # animal_of_interest passed as an int
)


class TestRegistryIntegrity:
    def test_registry_validates(self):
        problems = validate_registry(check_callables=True, check_tests=False)
        assert problems == [], "hazard registry problems:\n  " + "\n  ".join(problems)

    def test_registry_is_not_empty_and_has_a_ceiling(self):
        hazards = load_hazards()
        assert len(hazards) >= 20
        # The earn-your-place rule exists because a sprawling registry becomes
        # the context tax Layer 0 is supposed to prevent.
        assert len(hazards) <= 40, (
            f"{len(hazards)} hazards. Past ~40 this stops being consultable; "
            "prune prose-only entries or promote them to detectors."
        )

    @pytest.mark.parametrize('hazard_id', REQUIRED_HAZARD_IDS)
    def test_documented_failure_has_an_entry(self, hazard_id):
        assert hazard_id in hazards_by_id()

    def test_ids_and_slugs_are_unique(self):
        hazards = load_hazards()
        assert len({h.id for h in hazards}) == len(hazards)
        assert len({h.slug for h in hazards}) == len(hazards)

    def test_enums_are_respected(self):
        for h in load_hazards():
            assert h.severity in SEVERITIES
            assert h.kind in KINDS
            assert h.detector.type in DETECTOR_TYPES
            for stage in h.applies_to.get('stages', ()):
                assert stage in STAGES

    def test_every_callable_detector_resolves(self):
        """A renamed function must fail here, not silently stop guarding."""
        for h in load_hazards():
            if h.detector.type == 'callable':
                assert callable(resolve_callable(h.detector.callable)), h.id

    def test_every_pass_if_op_is_supported(self):
        for h in load_hazards():
            if h.detector.pass_if is not None:
                assert h.detector.pass_if['op'] in OPS, h.id

    def test_every_hazard_cites_a_reference(self):
        for h in load_hazards():
            assert h.references, f"{h.id} has no references"

    def test_earn_your_place_rule(self):
        """Prose-only entries must have actually caused a wrong result."""
        for h in load_hazards():
            if not h.detector.is_executable:
                assert h.occurred, (
                    f"{h.id} is prose-only and has no recorded occurrence. "
                    "Give it a detector, record the failure it caused, or "
                    "move it to discovery/requirements.py if it is a "
                    "capability gap rather than a trap."
                )
                assert h.evidence.get('where'), f"{h.id} claims occurred but cites no 'where'"

    def test_header_documents_the_sync_policy(self):
        """The prose/JSON relationship must stay stated in the file itself."""
        with open(DEFAULT_HAZARDS_PATH, encoding='utf-8') as fh:
            raw = json.load(fh)
        header = raw['_header']
        assert 'CLAUDE.md' in header['prose_canonical_in']
        assert header['earn_your_place_rule']
        assert header['sync_policy']

    @pytest.mark.slow
    def test_every_test_detector_node_id_collects(self):
        """A deleted or renamed test must not leave a dead detector behind."""
        problems = validate_registry(check_callables=False, check_tests=True)
        assert problems == [], "\n  ".join(problems)


class TestSelection:
    def test_min_severity_is_inclusive(self):
        crit = hazards_for(min_severity='critical')
        high = hazards_for(min_severity='high')
        assert {h.id for h in crit}.issubset({h.id for h in high})
        assert all(h.severity == 'critical' for h in crit)
        assert len(high) > len(crit)

    def test_sorted_most_severe_first(self):
        ranks = [h.severity_rank for h in hazards_for(min_severity='low')]
        assert ranks == sorted(ranks)

    def test_analysis_filter_selects_module_specific_hazards(self):
        loc = {h.id for h in hazards_for(analysis='ephys.decode_location')}
        assert 'HZ-STAT-006' in loc          # null='reverse' is decode_location's default
        assert 'HZ-API-005' not in loc       # social place field occupancy swap

    def test_unscoped_hazard_applies_broadly(self):
        """An unscoped hazard is assumed relevant, not assumed irrelevant.

        Over-reporting costs noise; under-reporting costs a corrupted result.
        HZ-INTERP-002 declares only a stage, so it must survive an analysis
        filter.
        """
        ids = {h.id for h in hazards_for(stage='interpret', analysis='ephys.decode_location')}
        assert 'HZ-INTERP-002' in ids

    def test_bad_min_severity_raises(self):
        with pytest.raises(ValueError):
            hazards_for(min_severity='catastrophic')


class TestDigest:
    def test_line_digest_is_one_line_per_hazard(self):
        selected = hazards_for(stage='propose', min_severity='high')
        text = render_digest(selected, 'line')
        assert len(text.splitlines()) == len(selected)

    def test_line_digest_names_the_detector_kind(self):
        text = render_digest(hazards_for(min_severity='low'), 'line')
        assert 'prose-only' in text
        assert 'callable(' in text

    def test_empty_selection_is_explicit(self):
        assert 'no applicable hazards' in render_digest([], 'line')

    def test_full_digest_surfaces_occurrences(self):
        text = render_digest([hazards_by_id()['HZ-STAT-001']], 'full')
        assert 'OCCURRED' in text

    def test_bad_verbosity_raises(self):
        with pytest.raises(ValueError):
            render_digest(hazards_for(), 'chatty')


class TestCannotCheckIsNotAPass:
    """The module's central invariant, from several directions."""

    def test_missing_context_key_is_not_a_pass(self):
        hazard = hazards_by_id()['HZ-STAT-001']
        result = run_detector(hazard, {'params': {'n_shuffles': 180, 'alpha': 0.05}})
        assert result.ran is False
        assert result.passed is None
        assert result.tripped is False
        assert 'cannot check' in result.message

    def test_missing_namespace_is_not_a_pass(self):
        hazard = hazards_by_id()['HZ-STAT-008']
        result = run_detector(hazard, {})
        assert (result.ran, result.passed) == (False, None)

    def test_none_valued_field_is_not_a_pass(self):
        """A manifest field present but null must not satisfy a comparison."""
        hazard = hazards_by_id()['HZ-DATA-001']
        ctx = {'manifest': {'tracking': {'n_identity_resolved_animals': None}}}
        result = run_detector(hazard, ctx)
        assert (result.ran, result.passed) == (False, None)

    def test_prose_only_detector_does_not_claim_a_pass(self):
        hazard = hazards_by_id()['HZ-INTERP-001']
        result = run_detector(hazard, {})
        assert (result.ran, result.passed) == (False, None)
        assert 'prose-only' in result.message

    def test_test_detector_is_opt_in_and_reports_why(self):
        hazard = hazards_by_id()['HZ-API-005']
        result = run_detector(hazard, {}, allow_tests=False)
        assert (result.ran, result.passed) == (False, None)
        assert 'allow_tests=True' in result.message

    def test_unresolvable_substitution_is_not_a_pass(self):
        """`{animal_id}` with no such param must not silently resolve."""
        hazard = hazards_by_id()['HZ-DATA-005']
        result = run_detector(hazard, {'params': {}, 'manifest': {'ephys': {'per_animal': {}}}})
        assert (result.ran, result.passed) == (False, None)

    def test_a_broken_detector_does_not_abort_the_digest(self):
        """One rotted entry must not take down a whole run."""
        broken = Hazard.from_dict({
            'id': 'HZ-TEST-000', 'slug': 'broken', 'title': 'broken',
            'severity': 'low', 'kind': 'statistical', 'status': 'active',
            'added': '2026-08-20', 'applies_to': {}, 'symptom': 's',
            'mechanism': 'm', 'consequence': 'c', 'references': ['x'],
            'detector': {'type': 'callable', 'callable': 'no.such.module:nope'},
        })
        result = run_detector(broken, {})
        assert (result.ran, result.passed) == (False, None)


class TestRealFailuresAreCaught:
    """Each case is a number this project actually reported."""

    def test_the_phase1_budget_artifact(self):
        """149 cells x 200 shuffles could not have found anything."""
        hazard = hazards_by_id()['HZ-STAT-001']
        result = run_detector(hazard, {'params': {'n_tests': 149, 'n_shuffles': 200, 'alpha': 0.05}})
        assert result.tripped
        assert result.raw['recommended_n_shuffles'] == 2980
        assert result.raw['best_achievable_q'] == pytest.approx(0.741, abs=1e-3)

    def test_the_below_baseline_accuracy(self):
        """60.6% against a 12/7 split whose majority baseline is 63.2%."""
        hazard = hazards_by_id()['HZ-STAT-002']
        ctx = {'results': {'population_accuracy_mean': 0.606,
                           'population_baseline_accuracy': 0.6315789473684211}}
        result = run_detector(hazard, ctx)
        assert result.tripped
        assert result.raw['margin'] < 0

    def test_the_pinned_p_value(self):
        """Iteration 12's rat613 p sat exactly on the 180-shuffle floor."""
        hazard = hazards_by_id()['HZ-STAT-008']
        ctx = {'params': {'n_shuffles': 180}, 'results': {'p_value': 1.0 / 181}}
        result = run_detector(hazard, ctx)
        assert result.tripped
        assert result.raw['pinned'] is True

    def test_a_p_value_clear_of_the_floor_passes(self):
        hazard = hazards_by_id()['HZ-STAT-008']
        ctx = {'params': {'n_shuffles': 180}, 'results': {'p_value': 0.03}}
        assert run_detector(hazard, ctx).passed is True

    def test_the_order_only_reverse_null(self):
        hazard = hazards_by_id()['HZ-STAT-006']
        assert run_detector(hazard, {'params': {'null': 'reverse'}}).tripped
        assert run_detector(hazard, {'params': {'null': 'shuffle'}}).passed is True

    def test_the_near_stationary_target(self):
        """rat630's x/y std of ~11-13 px produced a spurious decoding win."""
        hazard = hazards_by_id()['HZ-STAT-007']
        ctx = {'params': {'object_name': '630'},
               'manifest': {'tracking': {'objects': {'630': {'x_std_px': 11.0, 'y_std_px': 13.0}}}}}
        assert run_detector(hazard, ctx).tripped

    def test_a_mobile_target_passes(self):
        hazard = hazards_by_id()['HZ-STAT-007']
        ctx = {'params': {'object_name': '613'},
               'manifest': {'tracking': {'objects': {'613': {'x_std_px': 143.2, 'y_std_px': 121.9}}}}}
        assert run_detector(hazard, ctx).passed is True

    def test_the_blocked_hypothesis_3_tracking(self):
        """Session 20251216 resolves only rat631, which blocked Hypothesis #3."""
        hazard = hazards_by_id()['HZ-DATA-001']
        ctx = {'manifest': {'tracking': {'n_identity_resolved_animals': 1}}}
        assert run_detector(hazard, ctx).tripped

    def test_the_undocumented_tracking_coverage_gap(self):
        """20251216's tracking covers ~63% of the recording."""
        hazard = hazards_by_id()['HZ-DATA-002']
        ctx = {'manifest': {'tracking': {'frac_of_ephys_duration_covered': 0.63,
                                         'ephys_window': [412.7, 8231.4]}}}
        result = run_detector(hazard, ctx)
        assert result.tripped
        assert '63%' in result.message

    def test_the_int_animal_of_interest(self):
        hazard = hazards_by_id()['HZ-API-003']
        assert run_detector(hazard, {'params': {'animal_of_interest': 631}}).tripped
        assert run_detector(hazard, {'params': {'animal_of_interest': '631'}}).passed is True

    def test_the_post_hoc_denominator_shrink(self):
        hazard = hazards_by_id()['HZ-STAT-011']
        assert run_detector(
            hazard, {'results': {'denominator_status': 'outcome_dependent_exclusions'}}
        ).tripped
        assert run_detector(hazard, {'results': {'denominator_status': 'clean'}}).passed is True

    def test_resolution_guard_is_fooled_by_the_shrunk_denominator(self):
        """The reason HZ-STAT-011 has to exist as its own hazard.

        HZ-STAT-001 is not merely optimistic on a shrunk family - it actively
        certifies it. Fed m=7 it reports resolvable; fed the honest m=11 it
        reports unresolvable. So the exclusion manufactured both the
        significance and the permission to claim it, and no amount of
        resolution checking catches that on its own.
        """
        hazard = hazards_by_id()['HZ-STAT-001']
        shrunk = run_detector(hazard, {'params': {'n_tests': 7, 'n_shuffles': 180, 'alpha': 0.05}})
        honest = run_detector(hazard, {'params': {'n_tests': 11, 'n_shuffles': 180, 'alpha': 0.05}})
        assert shrunk.passed is True
        assert honest.tripped is True


class TestPredicates:
    def test_every_documented_op_works(self):
        assert evaluate_predicate(5, {'op': '>=', 'value': 5})
        assert evaluate_predicate(5, {'op': '>', 'value': 4})
        assert evaluate_predicate(4, {'op': '<', 'value': 5})
        assert evaluate_predicate(5, {'op': '<=', 'value': 5})
        assert evaluate_predicate('a', {'op': '==', 'value': 'a'})
        assert evaluate_predicate('a', {'op': '!=', 'value': 'b'})
        assert evaluate_predicate('a', {'op': 'in', 'value': ['a', 'b']})
        assert evaluate_predicate('c', {'op': 'not_in', 'value': ['a', 'b']})
        assert evaluate_predicate(True, {'op': 'is_true'})
        assert evaluate_predicate(False, {'op': 'is_false'})
        assert evaluate_predicate(0, {'op': 'is_present'})
        assert evaluate_predicate(None, {'op': 'is_absent'})

    def test_unknown_op_raises_rather_than_falling_back(self):
        """No eval fallback: the hazard file is data, and data must not execute."""
        with pytest.raises(PredicateError, match='unknown op'):
            evaluate_predicate(5, {'op': 'matches_regex', 'value': '.*'})

    def test_boolean_combinators_are_not_supported(self):
        with pytest.raises(PredicateError):
            evaluate_predicate(5, {'op': 'and', 'value': [1, 2]})

    def test_is_true_is_not_mere_truthiness(self):
        assert not evaluate_predicate('no', {'op': 'is_true'})
        assert not evaluate_predicate({'a': 1}, {'op': 'is_true'})

    def test_unary_op_rejects_a_value(self):
        with pytest.raises(PredicateError):
            evaluate_predicate(True, {'op': 'is_true', 'value': True})

    def test_binary_op_requires_a_value(self):
        with pytest.raises(PredicateError):
            evaluate_predicate(5, {'op': '>='})

    def test_missing_op_raises(self):
        with pytest.raises(PredicateError):
            evaluate_predicate(5, {'value': 3})

    def test_none_is_not_comparable(self):
        with pytest.raises(MissingValue):
            evaluate_predicate(None, {'op': '>=', 'value': 2})

    def test_field_extraction_walks_dicts_and_sequences(self):
        subject = {'per_object': [{'p_value': 0.01}, {'p_value': 0.5}]}
        assert extract_path(subject, 'per_object.1.p_value') == 0.5

    def test_missing_path_raises_rather_than_defaulting(self):
        with pytest.raises(MissingValue):
            extract_path({'a': 1}, 'a.b.c')
        with pytest.raises(MissingValue):
            extract_path({'a': 1}, 'nope')


class TestLoading:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(HazardRegistryError, match='not found'):
            load_hazards(tmp_path / 'nope.json')

    def test_schema_version_mismatch_raises(self, tmp_path):
        """A version bump must fail loudly, not parse as best it can."""
        path = tmp_path / 'hazards.json'
        path.write_text(json.dumps({'schema_version': 999, 'hazards': []}), encoding='utf-8')
        with pytest.raises(HazardRegistryError, match='schema_version'):
            load_hazards(path)

    def test_bad_callable_string_raises(self):
        with pytest.raises(HazardRegistryError):
            resolve_callable('not_a_dotted_path')
