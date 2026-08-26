"""Tests for the capability manifest's consult path.

The fixture manifest below reproduces the *real* state of two sessions this
project has worked with, so the tests double as a regression encoding of two
failures that cost real time:

- session ``20251216`` resolves only the focal animal in tracking, which
  blocked Hypothesis 3 after the analysis had already been attempted;
- ``behavior_type='F'`` yields one usable opponent for animal 631 while
  ``'EC'`` yields eight, which was found by trial and error.

`TestConsultPathStaysCheap` guards the module's design constraint: consulting
must be a JSON read, never something that can reach the //nearline share.
"""
import json
import sys

import pytest

from discovery.capability_manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    ManifestStale,
    check_testable,
    config_sha256,
    list_sessions,
    manifest_status,
    resolve_params,
    session_capabilities,
    suggest_sessions,
    verify_sources,
)
from discovery.capability_manifest import _session_date


REPO_ROOT = __import__('pathlib').Path(__file__).resolve().parent.parent


def _cohorts():
    """Real config paths with their real hashes, so staleness is exercised."""
    out = []
    for name, rel in (('cohort7', 'config/default_paths.json'),
                      ('cohort5', 'config/cohort5_paths.json')):
        digest = config_sha256(REPO_ROOT / rel)
        if digest is not None:
            out.append({'name': name, 'config_path': rel, 'config_sha256': digest})
    return out


def _manifest_dict():
    from datetime import datetime, timezone
    return {
        'schema_version': MANIFEST_SCHEMA_VERSION,
        'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'generated_by': {'script': 'test fixture', 'probe_level': 'full'},
        'cohorts': _cohorts(),
        'sessions': {
            # Only the focal animal is identity-resolved; tracking covers ~63%.
            '20251216_094334': {
                'session_id': '20251216_094334',
                'session_date': '20251216',
                'cohort': 'cohort7',
                'pixels_per_cm': 4.0,
                'ephys': {
                    'animals': ['rat613', 'rat615', 'rat630', 'rat631'],
                    'n_animals_with_ephys': 4,
                    'per_animal': {
                        'rat631': {'path_exists': True, 'n_clusters': 263,
                                   'n_quality_cells': 149,
                                   'sync': {'ok': True}},
                        'rat613': {'path_exists': True, 'n_clusters': 200,
                                   'n_quality_cells': 120,
                                   'sync': {'ok': True}},
                    },
                },
                'tracking': {
                    'available': True,
                    'n_identity_resolved_animals': 1,
                    'identity_resolved_animals': ['rat631'],
                    # Per animal: one session, four recording lengths.
                    'coverage_by_animal': {'rat631': 0.753, 'rat613': 0.3975},
                    'coverage_reference_animal': 'rat613',
                    'frac_of_ephys_duration_covered': 0.3975,
                    'ephys_window': [687.3, 8187.2],
                    'objects': {
                        'rat631': {'identity_resolved': True, 'frac_frames_present': 0.997,
                                   'x_std_px': 143.2, 'y_std_px': 121.9},
                    },
                },
                'events': {
                    'available': True,
                    'n_events_total': 641,
                    'frac_events_within_recording_by_animal': {
                        'rat631': 0.9376, 'rat613': 0.9376},
                    'events_window_reference_animal': 'rat631',
                    'frac_events_within_ephys_window': 0.9376,
                    'per_animal': {
                        'rat631': {
                            'opponent_labels': {
                                'EC': {'n_events': 173, 'n_classes_usable': 8,
                                       'usable': True},
                                'F': {'n_events': 12, 'n_classes_usable': 1,
                                      'usable': False,
                                      'reason': 'only 1 opponent reaches '
                                                'min_events_per_class=5'},
                            },
                            'outcome_labels': {
                                '__any__': {'n_events': 19, 'usable': True,
                                            'majority_baseline': 0.6315789473684211},
                                'F': {'n_events': 8, 'usable': False},
                            },
                        },
                    },
                },
                'provenance': {'sources': {}, 'probe_level': 'full'},
            },
            # Four animals with ephys and full multi-animal tracking.
            'RatCity_20251210_1359_40Hz': {
                'session_id': 'RatCity_20251210_1359_40Hz',
                'session_date': '20251210',
                'cohort': 'cohort7',
                'pixels_per_cm': 4.0,
                'ephys': {
                    'animals': ['rat613', 'rat631'],
                    'n_animals_with_ephys': 4,
                    'per_animal': {
                        'rat631': {'path_exists': True, 'n_quality_cells': 377,
                                   'sync': {'ok': True}},
                        'rat613': {'path_exists': True, 'n_quality_cells': 300,
                                   'sync': {'ok': True}},
                    },
                },
                'tracking': {
                    'available': True,
                    'n_identity_resolved_animals': 4,
                    'identity_resolved_animals': ['rat613', 'rat630', 'rat631', 'rat635'],
                    'coverage_by_animal': {'rat631': 0.95, 'rat613': 0.95},
                    'coverage_reference_animal': 'rat631',
                    'frac_of_ephys_duration_covered': 0.95,
                    'ephys_window': [0.0, 12000.0],
                    'objects': {
                        'rat613': {'identity_resolved': True, 'frac_frames_present': 0.99,
                                   'x_std_px': 143.2, 'y_std_px': 121.9},
                        # The near-stationary animal that produced a spurious win.
                        'rat630': {'identity_resolved': True, 'frac_frames_present': 0.98,
                                   'x_std_px': 11.0, 'y_std_px': 13.0},
                        'rat631': {'identity_resolved': True, 'frac_frames_present': 0.99,
                                   'x_std_px': 150.0, 'y_std_px': 140.0},
                        'rat635': {'identity_resolved': True, 'frac_frames_present': 0.30,
                                   'x_std_px': 120.0, 'y_std_px': 118.0},
                    },
                },
                'events': {'available': True,
                           'frac_events_within_recording_by_animal': {
                               'rat631': 1.0, 'rat613': 1.0},
                           'frac_events_within_ephys_window': 1.0,
                           'per_animal': {}},
                'provenance': {'sources': {}, 'probe_level': 'full'},
            },
        },
    }


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / 'capability_manifest.json'
    path.write_text(json.dumps(_manifest_dict()), encoding='utf-8')
    return path


class TestConsultPathStaysCheap:
    def test_module_does_not_import_the_data_layer(self):
        """Importing the consult path must not drag in SMB-capable modules.

        If it could reach the share, eventually something would make it, and a
        check meant to cost milliseconds would start costing minutes.

        Checked in a *subprocess* deliberately: asserting on this process's
        `sys.modules` would pass or fail depending on which other test file ran
        first, which is exactly the kind of test that looks like a guard and
        isn't. A fresh interpreter is the only way to see the real import
        closure.
        """
        import subprocess

        probe = (
            "import discovery.capability_manifest, sys;"
            "leaked=[m for m in ('ingestion.data_paths','video.tracking_import',"
            "'ephys.decode_location','ephys._lda_decoding','sqlalchemy')"
            " if m in sys.modules];"
            "print(','.join(leaked))"
        )
        proc = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                              text=True, cwd=str(REPO_ROOT))
        assert proc.returncode == 0, proc.stderr
        leaked = proc.stdout.strip()
        assert leaked == '', (
            f"the consult path pulled in {leaked}. Keep the expensive probing in "
            "discovery/manifest_build.py so consulting stays a JSON read.")

    def test_consulting_touches_only_the_manifest(self, manifest, monkeypatch):
        """No directory listing, no globbing, beyond reading the file itself."""
        import pathlib

        def _boom(*args, **kwargs):
            raise AssertionError('the consult path tried to walk the filesystem')

        monkeypatch.setattr(pathlib.Path, 'iterdir', _boom)
        monkeypatch.setattr(pathlib.Path, 'glob', _boom)
        monkeypatch.setattr(pathlib.Path, 'rglob', _boom)

        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', behavior_type='EC', path=manifest)
        assert report.testable is True

    def test_session_date_matches_the_notebook_implementation(self):
        """The small independent copy must not drift from its counterpart."""
        from database.lab_notebook import normalize_session_key
        for candidate in ['20251210', 'RatCity_20251210_1359_40Hz',
                          'RatCity_20251210_1359_40Hz.rec', '20251210_094334',
                          '2025', '', 'rat631', None]:
            assert _session_date(candidate) == normalize_session_key(candidate)


class TestTheRealFailures:
    def test_hypothesis_3_is_not_testable_on_20251216(self, manifest):
        """Only rat631 is identity-resolved, so no partner positions exist."""
        report = check_testable('ephys.social_spatial_fields', '20251216',
                                animal_id='rat631', path=manifest)
        assert report.testable is False
        joined = ' '.join(u.reason for u in report.unmet)
        assert 'identity-resolved' in joined

    def test_that_failure_offers_an_alternative_session(self, manifest):
        options = suggest_sessions('ephys.social_spatial_fields', 'rat631', path=manifest)
        assert any(o.session_id == 'RatCity_20251210_1359_40Hz' for o in options)

    def test_behavior_type_F_is_not_testable(self, manifest):
        """One usable opponent class; LDA needs two."""
        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', behavior_type='F', path=manifest)
        assert report.testable is False
        assert any('n_classes_usable' in u.requirement for u in report.unmet)
        assert any('HZ-STAT-010' in u.hazard_ids for u in report.unmet)

    def test_behavior_type_EC_is_testable(self, manifest):
        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', behavior_type='EC', path=manifest)
        assert report.testable is True

    def test_leaving_behavior_type_open_enumerates_the_viable_one(self, manifest):
        """The mechanical replacement for finding this out by trial and error."""
        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', path=manifest)
        assert report.testable is False          # underspecified, not impossible
        assert report.undetermined
        chosen = {combo['behavior_type'] for combo in report.viable_params}
        assert chosen == {'EC'}

    def test_the_summary_names_the_remedy(self, manifest):
        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', behavior_type='F', path=manifest)
        text = report.summary()
        assert 'NOT TESTABLE' in text
        assert 'remedy:' in text
        assert 'manifest generated' in text

    def test_tracking_coverage_is_a_warning_not_a_block(self, manifest):
        """63% coverage doesn't make the analysis impossible, only wrong if ignored."""
        report = check_testable('ephys.decode_location', '20251216',
                                animal_id='rat631', object_name='rat631', path=manifest)
        assert any('coverage_by_animal' in w.requirement
                   for w in report.warnings)
        assert any('HZ-DATA-002' in w.hazard_ids for w in report.warnings)

    def test_a_sparsely_tracked_object_warns(self, manifest):
        report = check_testable('ephys.decode_location', '20251210',
                                animal_id='rat631', object_name='rat635', path=manifest)
        assert any('frac_frames_present' in w.requirement for w in report.warnings)

    def test_an_unresolved_object_is_not_testable(self, manifest):
        report = check_testable('ephys.decode_location', '20251216',
                                animal_id='rat631', object_name='rat632', path=manifest)
        assert report.testable is False
        assert any('absent' in u.observed for u in report.unmet)

    def test_coverage_is_read_per_animal(self, manifest):
        """One session, several recording lengths - so coverage is per animal.

        On the real 20251216 the same tracking window covers 39.8% of rat613's
        recording and 75.3% of rat631's. A session-wide scalar reported one of
        those as the answer for all four animals.
        """
        record = session_capabilities('20251216', manifest)
        coverage = record['tracking']['coverage_by_animal']
        assert coverage['rat631'] != coverage['rat613']

        lenient = check_testable('ephys.decode_location', '20251216',
                                 animal_id='rat631', object_name='rat631',
                                 path=manifest)
        strict = check_testable('ephys.decode_location', '20251216',
                                animal_id='rat613', object_name='rat631',
                                path=manifest)
        # rat631 is at 0.753 and rat613 at 0.3975; both are below the 0.8
        # threshold, but the warning must quote each animal's own number.
        observed = {r.animal_id: [w.observed for w in r.warnings
                                  if 'coverage_by_animal' in w.requirement]
                    for r in (lenient, strict)}
        assert observed['rat631'] != observed['rat613']


class TestResolveParams:
    def test_enumerates_tracked_objects_for_decode_location(self, manifest):
        combos = resolve_params('ephys.decode_location', '20251210',
                                animal_id='rat631', path=manifest)
        objects = {c['object_name'] for c in combos}
        assert {'rat613', 'rat630', 'rat631'}.issubset(objects)

    def test_sweeps_animals_when_none_is_named(self, manifest):
        combos = resolve_params('ephys.decode_opponent_identity', '20251216',
                                path=manifest)
        assert all('animal_id' in c for c in combos)

    def test_returns_empty_when_nothing_is_viable(self, manifest):
        assert resolve_params('ephys.social_spatial_fields', '20251216',
                              animal_id='rat631', path=manifest) == ()


class TestSuggestSessions:
    def test_does_not_rank(self, manifest):
        """An unranked set beats a fabricated 'best session' score."""
        options = suggest_sessions('ephys.decode_location', 'rat631', path=manifest)
        assert options
        assert not hasattr(options[0], 'score')
        assert not hasattr(options[0], 'rank')

    def test_carries_the_numbers_needed_to_choose(self, manifest):
        options = suggest_sessions('ephys.social_spatial_fields', 'rat631', path=manifest)
        assert options
        joined = ' '.join(n for o in options for n in o.notes)
        assert 'identity-resolved' in joined
        assert 'tracking covers' in joined

    def test_cohort_filter(self, manifest):
        assert suggest_sessions('ephys.decode_location', 'rat631',
                                cohort='cohort5', path=manifest) == ()


class TestSessionLookup:
    @pytest.mark.parametrize('query', [
        '20251210', 'RatCity_20251210_1359_40Hz', 'RatCity_20251210_1359_40Hz.rec',
    ])
    def test_every_id_form_resolves(self, manifest, query):
        record = session_capabilities(query, manifest)
        assert record['session_date'] == '20251210'

    def test_unknown_session_raises_with_a_count(self, manifest):
        with pytest.raises(KeyError, match='not in the capability manifest'):
            session_capabilities('19990101', manifest)

    def test_list_sessions(self, manifest):
        assert len(list_sessions(path=manifest)) == 2
        assert list_sessions(cohort='cohort5', path=manifest) == ()

    def test_unknown_analysis_raises_rather_than_passing(self, manifest):
        """'No requirements' and 'we have no idea' must not look the same."""
        with pytest.raises(KeyError, match='no requirement bundle'):
            check_testable('ephys.made_up_analysis', '20251216', path=manifest)


class TestStalenessRaisesRatherThanWarns:
    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(ManifestError, match='not found'):
            manifest_status(tmp_path / 'nope.json')

    def test_schema_mismatch_raises(self, tmp_path):
        path = tmp_path / 'm.json'
        path.write_text(json.dumps({'schema_version': 999, 'sessions': {}}),
                        encoding='utf-8')
        with pytest.raises(ManifestError, match='schema_version'):
            manifest_status(path)

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / 'm.json'
        path.write_text('{not json', encoding='utf-8')
        with pytest.raises(ManifestError, match='not valid JSON'):
            manifest_status(path)

    def test_changed_cohort_config_raises(self, tmp_path):
        """A quietly stale manifest is worse than none: the agent would trust it."""
        data = _manifest_dict()
        if not data['cohorts']:
            pytest.skip('no cohort configs present to hash')
        data['cohorts'][0]['config_sha256'] = 'deadbeef' * 8
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.raises(ManifestStale, match='changed since the build'):
            manifest_status(path)

    def test_fresh_manifest_is_fresh(self, manifest):
        status = manifest_status(manifest)
        assert status.state == 'fresh'
        assert status.n_sessions == 2

    def test_probe_level_is_derived_from_sessions_not_the_top_level_claim(self, tmp_path):
        """The top-level field records what the last run *asked for*.

        Probing one session at 'full' leaves it saying 'full' while every other
        session is paths-only, so trusting it makes the manifest overstate
        itself - and manifest_status would then not warn at all.
        """
        data = _manifest_dict()
        data['generated_by']['probe_level'] = 'full'      # the optimistic claim
        for record in data['sessions'].values():
            record['provenance']['probe_level'] = 'paths'  # the reality
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.warns(RuntimeWarning, match='paths-only'):
            status = manifest_status(path)
        assert status.probe_level == 'paths'
        assert status.state == 'partial'
        assert status.n_fully_probed == 0

    def test_a_mixed_manifest_is_partial_and_says_how_many(self, tmp_path):
        """The real state after probing one session out of 34."""
        data = _manifest_dict()
        keys = sorted(data['sessions'])
        data['sessions'][keys[0]]['provenance']['probe_level'] = 'full'
        data['sessions'][keys[1]]['provenance']['probe_level'] = 'paths'
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.warns(RuntimeWarning, match='1 of 2'):
            status = manifest_status(path)
        assert status.probe_level == 'mixed'
        assert status.state == 'partial'
        assert status.n_fully_probed == 1

    def test_a_fully_probed_manifest_is_fresh(self, manifest):
        status = manifest_status(manifest)
        assert status.probe_level == 'full'
        assert status.n_fully_probed == status.n_sessions

    def test_old_manifest_warns_but_is_usable(self, tmp_path):
        data = _manifest_dict()
        data['generated_at'] = '2020-01-01T00:00:00Z'
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.warns(RuntimeWarning, match='days old'):
            assert manifest_status(path).state == 'aging'

    def test_every_report_carries_the_manifest_timestamp(self, manifest):
        """So a stale answer is visible in the skill's own output."""
        report = check_testable('ephys.decode_opponent_identity', '20251216',
                                animal_id='rat631', behavior_type='EC', path=manifest)
        assert report.manifest_generated_at
        assert report.manifest_state == 'fresh'


class TestVerifySources:
    def test_no_recorded_sources_is_trivially_ok(self, manifest):
        result = verify_sources('20251216', manifest)
        assert result.ok is True
        assert result.checked == 0

    def test_detects_a_missing_file(self, tmp_path):
        data = _manifest_dict()
        data['sessions']['20251216_094334']['provenance']['sources'] = {
            'tracking': {'path': str(tmp_path / 'gone.csv'), 'size': 10, 'mtime': 1}
        }
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        result = verify_sources('20251216', path)
        assert result.ok is False
        assert result.missing

    def test_detects_a_changed_size(self, tmp_path):
        target = tmp_path / 'tracking.csv'
        target.write_text('some content', encoding='utf-8')
        data = _manifest_dict()
        data['sessions']['20251216_094334']['provenance']['sources'] = {
            'tracking': {'path': str(target), 'size': 999999, 'mtime': 1}
        }
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        result = verify_sources('20251216', path)
        assert result.ok is False
        assert any('size' in c for c in result.changed)

    def test_unchanged_file_passes(self, tmp_path):
        import os
        target = tmp_path / 'tracking.csv'
        target.write_text('some content', encoding='utf-8')
        stat = os.stat(target)
        data = _manifest_dict()
        data['sessions']['20251216_094334']['provenance']['sources'] = {
            'tracking': {'path': str(target), 'size': stat.st_size,
                         'mtime': int(stat.st_mtime)}
        }
        path = tmp_path / 'm.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        assert verify_sources('20251216', path).ok is True


class TestRequirementBundles:
    def test_every_known_analysis_has_a_bundle(self):
        from discovery.requirements import known_analyses, requirements_for
        for analysis in known_analyses():
            assert requirements_for(analysis)

    def test_every_referenced_hazard_id_exists(self):
        """A requirement citing a hazard that was renamed helps nobody."""
        from discovery.hazards import hazards_by_id
        from discovery.requirements import REQUIREMENTS
        known = set(hazards_by_id())
        for analysis, reqs in REQUIREMENTS.items():
            for req in reqs:
                for hazard_id in req.hazards:
                    assert hazard_id in known, (
                        f"{analysis} requirement {req.path!r} cites unknown "
                        f"hazard {hazard_id}")

    def test_no_boolean_combinators_creep_in(self):
        """The guardrail against the bundles becoming a rules engine."""
        from discovery._predicates import OPS
        from discovery.requirements import REQUIREMENTS
        for reqs in REQUIREMENTS.values():
            for req in reqs:
                assert req.op in OPS
                assert req.op not in ('and', 'or', 'not')

    def test_a_bad_op_is_rejected_at_construction(self):
        from discovery._predicates import PredicateError
        from discovery.requirements import Req
        with pytest.raises(PredicateError):
            Req('some.path', 'matches_regex', '.*')

    def test_placeholders_are_discovered(self):
        from discovery.requirements import Req
        req = Req('events.per_animal.{animal_id}.opponent_labels.{behavior_type}.usable',
                  'is_true')
        assert req.placeholders == ('animal_id', 'behavior_type')
