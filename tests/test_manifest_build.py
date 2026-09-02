"""Tests for discovery/manifest_build.py — the expensive half of Layer 0.

All mock data; nothing here touches //nearline. Two things are pinned that
would otherwise regress silently:

`TestCostGuards` asserts the tracking probe passes `usecols` to `read_csv`.
The merged mask-metrics files run to tens of megabytes and only five columns
are needed, so a well-meaning switch to `load_tracking_data` would turn a
cheap probe into a slow one across every session.

`TestBuildIsNeverFatal` asserts a session whose data is broken produces a
record with the failure written into `provenance.errors` rather than aborting
the sweep. A multi-hour pass that dies on session 9 of 34 is not resumable in
any useful sense.
"""
import json

import numpy as np
import pandas as pd
import pytest

from discovery.manifest_build import (
    DEFAULT_QUALITY_THRESHOLDS,
    _is_identity_resolved,
    _stat_entry,
    atomic_write_manifest,
    build_session_record,
    derive_analysis_readiness,
    merge_session_record,
    new_manifest,
    probe_events,
    probe_tracking,
)


#: Tracking timestamps on disk are Linux **epoch** nanoseconds, so a realistic
#: fixture has to be epoch-based; a naive "seconds since session start scaled by
#: 1e9" would sit below the probe's nanosecond-detection threshold and silently
#: exercise a different code path than production does.
EPOCH_BASE_SECONDS = 1_700_000_000.0


class _StubSync:
    """Linear behaviour->ephys map, mirroring tests/test_decode_location.py.

    Defaults to shifting epoch seconds onto a recording-relative clock, which
    is what a real DataSyncManager does.
    """

    def __init__(self, slope=1.0, intercept=-EPOCH_BASE_SECONDS):
        self.slope = slope
        self.intercept = intercept

    def convert_behavior_to_ephys(self, t):
        return np.asarray(t, dtype=float) * self.slope + self.intercept


def _write_tracking(tmp_path, *, n_frames=400, objects=None, frame_rate=40.0,
                    start_second=100.0):
    """A mask-metrics CSV plus its timestamps sidecar."""
    objects = objects or {
        'rat631': {'x_std': 150.0, 'y_std': 140.0, 'present': 1.0},
        'rat630': {'x_std': 12.0, 'y_std': 11.0, 'present': 1.0},   # near-stationary
        'unknown_1': {'x_std': 90.0, 'y_std': 90.0, 'present': 0.3},
    }
    rng = np.random.default_rng(0)
    rows = []
    for index, (name, spec) in enumerate(objects.items()):
        keep = int(n_frames * spec['present'])
        for frame in range(keep):
            rows.append({
                'object_name': name, 'object_id': index, 'frame': frame,
                'center_x': float(rng.normal(500, spec['x_std'])),
                'center_y': float(rng.normal(400, spec['y_std'])),
                'extra_unused_column': 0.0,
            })
    path = tmp_path / 'merged_20251216_0950_1200_mask_metrics.csv'
    pd.DataFrame(rows).to_csv(path, index=False)

    # load_timestamps replaces '_mask_metrics' with '_ts' in the stem, so the
    # sidecar must be named for the tracking file without that infix. Values are
    # Linux epoch nanoseconds, as on disk.
    seconds = EPOCH_BASE_SECONDS + start_second + np.arange(n_frames) / frame_rate
    np.save(tmp_path / 'merged_20251216_0950_1200_ts.npy', seconds * 1e9)
    return path


class _StubDsm:
    def __init__(self, *, tracking=None, events=None, kilosort=None,
                 session_id='20251216', pixels_per_cm=4.0, raise_on=()):
        self._tracking = [tracking] if tracking else []
        self._events = [events] if events else []
        self._kilosort = kilosort
        self.session_id = session_id
        self._pixels_per_cm = pixels_per_cm
        self._raise_on = set(raise_on)

    def _maybe_raise(self, name):
        if name in self._raise_on:
            raise FileNotFoundError(f'stub failure in {name}')

    def get_tracking_files(self):
        self._maybe_raise('get_tracking_files')
        return list(self._tracking)

    def get_behavioral_event_files(self):
        self._maybe_raise('get_behavioral_event_files')
        return list(self._events)

    def get_kilosort_path(self):
        self._maybe_raise('get_kilosort_path')
        return self._kilosort

    def get_pixels_per_cm(self):
        return self._pixels_per_cm


class TestIdentityResolution:
    @pytest.mark.parametrize('name,expected', [
        ('rat631', True), ('rat613', True), ('RAT631', True),
        ('unknown_1', False), ('food', False), ('block', False),
        ('rat', False), ('631', False), ('rat631_b', False),
    ])
    def test_only_ratNNN_counts_as_an_animal(self, name, expected):
        """Identity resolution is per-session; unassigned blobs are not animals."""
        assert _is_identity_resolved(name) is expected


class TestProbeTracking:
    def test_reports_per_object_variance_and_presence(self, tmp_path):
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert out['available'] is True
        assert out['n_identity_resolved_animals'] == 2
        assert out['identity_resolved_animals'] == ['rat630', 'rat631']
        assert out['unresolved_object_names'] == ['unknown_1']

    def test_flags_the_near_stationary_object(self, tmp_path):
        """The degenerate-target case that produced a spurious decoding win."""
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert out['objects']['rat630']['x_std_px'] < 25
        assert out['objects']['rat631']['x_std_px'] > 100

    def test_records_the_sparse_object(self, tmp_path):
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert out['objects']['unknown_1']['frac_frames_present'] == pytest.approx(0.3,
                                                                                  abs=0.02)

    def test_computes_partial_ephys_coverage(self, tmp_path):
        """The 20251216 case: tracking starts after the recording does."""
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=100.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             durations={'rat631': 200.0})
        assert out['ephys_window'][0] == pytest.approx(100.0, abs=0.5)
        # 100s..110s of a 200s recording -> about 5% covered.
        assert 0.0 < out['coverage_by_animal']['rat631'] < 0.2
        assert out['coverage_reference_animal'] == 'rat631'

    def test_full_coverage_reports_near_one(self, tmp_path):
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=0.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             durations={'rat631': 10.0})
        assert out['coverage_by_animal']['rat631'] == pytest.approx(1.0, abs=0.05)

    def test_coverage_is_per_animal_because_durations_differ(self, tmp_path):
        """Animals in one session share a clock, not a recording length.

        The real 20251216 durations are 18866 / 3651 / 18556 / 9960 s, so the
        same tracking window covers 39.8% / 205% / 40.4% / 75.3% of "the
        recording". An earlier version divided by whichever animal sorted first
        and reported that as the coverage - a confidently wrong scalar.
        """
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=0.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             durations={'rat613': 100.0, 'rat631': 20.0})
        coverage = out['coverage_by_animal']
        assert coverage['rat613'] < coverage['rat631'], 'coverage must differ per animal'
        # The single scalar is the most conservative reading, and names its source.
        assert out['coverage_reference_animal'] == 'rat613'
        assert out['frac_of_ephys_duration_covered'] == coverage['rat613']

    def test_a_ratio_above_one_is_flagged_not_reported(self, tmp_path):
        """205% coverage is an inconsistency, not a measurement."""
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=0.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             durations={'rat615': 2.0})   # recording far shorter
        assert 'rat615' in out['coverage_exceeds_recording']
        assert out['coverage_by_animal']['rat615'] <= 1.0

    def test_no_durations_means_no_coverage_claim(self, tmp_path):
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert out['coverage_by_animal'] == {}
        assert out['frac_of_ephys_duration_covered'] is None

    def test_converts_to_cm_when_the_scale_is_known(self, tmp_path):
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(), pixels_per_cm=4.0)
        record = out['objects']['rat631']
        assert record['x_std_cm'] == pytest.approx(record['x_std_px'] / 4.0, abs=1e-3)

    def test_omits_cm_when_the_scale_is_unknown(self, tmp_path):
        """cohort5 has pixels_per_cm null; *_cm must not be invented."""
        path = _write_tracking(tmp_path)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(), pixels_per_cm=None)
        assert 'x_std_cm' not in out['objects']['rat631']

    def test_no_tracking_file_is_recorded_not_raised(self):
        out = probe_tracking(_StubDsm(tracking=None))
        assert out['available'] is False
        assert 'no tracking file' in out['error']

    def test_a_raising_resolver_is_recorded(self):
        out = probe_tracking(_StubDsm(raise_on=('get_tracking_files',)))
        assert out['available'] is False
        assert 'path resolution failed' in out['error']


class TestAttachmentToTheRightRecording:
    """HZ-DATA-008. The blocks of one day share a clock and run back to back.

    Measured on 20251216/rat613: [6.7, 18897.5], [19017.6, 36569.2] and
    [36732.6, 42054.0] s. The day's only tracking file maps to [687, 8187],
    inside the first block and nowhere near the other two — but the first
    version of this check compared against ``[0, duration]``, and since
    ``duration`` is a *span* every block looked like it started at zero. The
    afternoon block therefore reported ``overlap_verified`` against a file
    recorded five hours before it began, which is precisely the silent wrong
    answer the attachment check exists to prevent.
    """

    def test_the_primary_block_is_attached(self, tmp_path):
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=100.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             windows={'rat613': [5.0, 200.0]})
        assert out['attachment_status'] == 'overlap_verified'
        assert out['attached'] is True
        assert out['available'] is True

    def test_a_later_block_of_the_same_day_is_not_attached(self, tmp_path):
        """Tracking at 100-110 s against a block spanning 19000-36000 s."""
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=100.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             windows={'rat613': [19017.6, 36569.2]})
        assert out['attachment_status'] == 'no_overlap'
        assert out['attached'] is False
        assert out['available'] is False, (
            'a file that does not overlap this recording must not be offered '
            'to it as available')
        assert out['overlap_seconds'] < 0
        assert 'HZ-DATA-008' in out['error']

    def test_zero_start_assumption_would_have_passed_it(self, tmp_path):
        """The bug, pinned: [0, duration] makes every block start at zero."""
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=100.0)
        wrong = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                               durations={'rat613': 36569.2 - 19017.6})
        assert wrong['attached'] is True, (
            'documents the wrong answer the duration-only path gives; the '
            'windows= path above is the correct one')

    def test_windows_take_precedence_over_durations(self, tmp_path):
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0,
                               start_second=100.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             durations={'rat613': 17551.6},
                             windows={'rat613': [19017.6, 36569.2]})
        assert out['attached'] is False

    def test_coverage_uses_the_real_interval_not_the_span(self, tmp_path):
        """rat615's 'impossible 205%' was this bug, not bad data.

        Its recording is [377.0, 4028.7] s — a 3651 s block starting 377 s
        into the day. Dividing by the span while assuming a zero start put
        the tracking window mostly outside it and produced a ratio above one.
        """
        path = _write_tracking(tmp_path, n_frames=4000, frame_rate=40.0,
                               start_second=377.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync(),
                             windows={'rat615': [377.0, 4028.7]})
        # 4000 frames at 40 Hz is a 100 s window, against a 3651.7 s block.
        assert out['coverage_by_animal']['rat615'] == pytest.approx(0.0274,
                                                                    abs=0.002)
        assert out['coverage_by_animal']['rat615'] <= 1.0

    def test_unknown_interval_stays_undetermined(self, tmp_path):
        """No windows and no durations: cannot check, so no verdict."""
        path = _write_tracking(tmp_path, n_frames=400, frame_rate=40.0)
        out = probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert out['attachment_status'] == 'undetermined'
        assert out['attached'] is None
        assert out['available'] is True, (
            'undetermined must not be reported as a confident negative either')


class TestCostGuards:
    def test_tracking_probe_passes_usecols(self, tmp_path, monkeypatch):
        """A ~90 MB CSV must not be read in full to get five columns."""
        path = _write_tracking(tmp_path)
        seen = {}
        original = pd.read_csv

        def _spy(*args, **kwargs):
            seen.setdefault('usecols', kwargs.get('usecols'))
            return original(*args, **kwargs)

        monkeypatch.setattr(pd, 'read_csv', _spy)
        probe_tracking(_StubDsm(tracking=path), _StubSync())
        assert seen['usecols'] is not None, (
            'the tracking probe read every column; keep usecols so this stays cheap')
        assert 'center_x' in seen['usecols']

    def test_falls_back_to_a_full_read_when_columns_differ(self, tmp_path):
        """A differently-shaped file must still probe rather than fail."""
        path = tmp_path / 'odd_mask_metrics.csv'
        pd.DataFrame({'object_name': ['rat631'] * 3, 'object_id': [0, 0, 0],
                      'x': [1.0, 2.0, 3.0]}).to_csv(path, index=False)
        out = probe_tracking(_StubDsm(tracking=path))
        assert out['available'] is True
        assert 'rat631' in out['objects']


class TestProbeEvents:
    def _events_csv(self, tmp_path, *, n_ec=40, n_fight=6):
        """Events where 'EC' supports many opponents and 'F' supports one.

        Reproduces the real asymmetry for animal 631 on session 20251216.
        """
        rows = []
        opponents = ['rat613', 'rat615', 'rat630', 'rat635']
        for i in range(n_ec):
            rows.append({
                'event_id': f'ec{i}', 'type': 'EC',
                'ts_start': 1_700_000_000_000_000_000 + i * 1_000_000_000,
                'ts_end': 1_700_000_000_000_000_000 + i * 1_000_000_000 + 500_000_000,
                'initiator': 'rat631', 'victim': opponents[i % len(opponents)],
                'winner': 'rat631', 'loser': opponents[i % len(opponents)],
            })
        for i in range(n_fight):
            rows.append({
                'event_id': f'f{i}', 'type': 'F',
                'ts_start': 1_700_000_100_000_000_000 + i * 1_000_000_000,
                'ts_end': 1_700_000_100_000_000_000 + i * 1_000_000_000 + 500_000_000,
                'initiator': 'rat631', 'victim': 'rat613',
                'winner': 'rat631', 'loser': 'rat613',
            })
        path = tmp_path / '20251216_behavior_event_df.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_uses_the_real_extractors(self, tmp_path):
        """Reimplementing min_events_per_class here would eventually disagree."""
        path = self._events_csv(tmp_path)
        out = probe_events(_StubDsm(events=path), ['rat631'], _StubSync())
        if not out['sync_probe_ok']:
            pytest.skip(f"stub sync insufficient for this loader: {out['error']}")
        opponent = out['per_animal']['rat631']['opponent_labels']
        assert opponent['EC']['n_classes_usable'] == 4
        assert opponent['EC']['usable'] is True

    def test_a_single_class_type_is_marked_unusable_with_a_reason(self, tmp_path):
        """behavior_type='F' with one opponent: the real trial-and-error failure."""
        path = self._events_csv(tmp_path)
        out = probe_events(_StubDsm(events=path), ['rat631'], _StubSync())
        if not out['sync_probe_ok']:
            pytest.skip('stub sync insufficient for this loader')
        opponent = out['per_animal']['rat631']['opponent_labels']
        if 'F' in opponent:
            assert opponent['F']['usable'] is False
            assert 'min_events_per_class' in opponent['F']['reason']
        else:
            assert 'F' in out['per_animal']['rat631']['unusable_types']

    def test_reports_per_type_counts(self, tmp_path):
        path = self._events_csv(tmp_path)
        out = probe_events(_StubDsm(events=path), ['rat631'], _StubSync())
        assert out['available'] is True
        assert out['by_type'].get('EC') == 40

    def test_no_event_file_is_recorded_not_raised(self):
        out = probe_events(_StubDsm(events=None), ['rat631'])
        assert out['available'] is False
        assert 'no behavioural event file' in out['error']

    def test_without_sync_label_counts_are_withheld(self, tmp_path):
        """Label extraction needs ts_*_ephys; guessing would be worse than nothing."""
        path = self._events_csv(tmp_path)
        out = probe_events(_StubDsm(events=path), ['rat631'], sync=None)
        assert out['per_animal'] == {}
        assert 'not synchronized' in out['error']


class TestDeriveAnalysisReadiness:
    def _record(self, **overrides):
        record = {
            'ephys': {'animals': ['rat631'], 'n_animals_with_ephys': 4,
                      'per_animal': {'rat631': {'path_exists': True,
                                                'n_quality_cells': 149,
                                                'sync': {'ok': True}}}},
            'tracking': {'available': True, 'n_identity_resolved_animals': 1,
                         'identity_resolved_animals': ['rat631'],
                         'coverage_by_animal': {'rat631': 0.753},
                         'coverage_reference_animal': 'rat631',
                         'frac_of_ephys_duration_covered': 0.753,
                         'objects': {'rat631': {'identity_resolved': True,
                                                'frac_frames_present': 0.99}}},
            'events': {'available': True,
                       'frac_events_within_recording_by_animal': {'rat631': 1.0},
                       'frac_events_within_ephys_window': 1.0,
                       'per_animal': {'rat631': {
                           'opponent_labels': {
                               'EC': {'n_classes_usable': 8, 'usable': True},
                               'F': {'n_classes_usable': 1, 'usable': False}},
                           'outcome_labels': {'__any__': {'usable': True}}}}},
            'pixels_per_cm': 4.0,
            'provenance': {'sources': {}, 'probe_level': 'full'},
        }
        record.update(overrides)
        return record

    def test_opponent_decoding_is_testable_only_for_the_viable_type(self):
        readiness = derive_analysis_readiness(self._record())
        block = readiness['ephys.decode_opponent_identity']
        assert block['testable'] is True
        assert [c['behavior_type'] for c in block['viable_params']] == ['EC']

    def test_partner_analyses_are_not_testable_with_one_resolved_animal(self):
        """Session 20251216: exactly the state that blocked Hypothesis 3."""
        readiness = derive_analysis_readiness(self._record())
        assert readiness['ephys.social_spatial_fields']['testable'] is False
        assert readiness['ephys.decode_partner_distance']['testable'] is False

    def test_self_decoding_is_still_testable(self):
        readiness = derive_analysis_readiness(self._record())
        block = readiness['ephys.decode_location']
        assert block['testable'] is True
        assert all(c['object_name'] == 'rat631' for c in block['viable_params'])

    def test_agrees_with_check_testable(self, tmp_path):
        """A requirements change must surface as 'rebuild', not two answers."""
        from discovery.capability_manifest import (MANIFEST_SCHEMA_VERSION,
                                                   check_testable)
        record = self._record()
        record['session_id'] = '20251216_094334'
        record['session_date'] = '20251216'
        record['cohort'] = 'cohort7'
        record['analysis_readiness'] = derive_analysis_readiness(record)

        path = tmp_path / 'm.json'
        path.write_text(json.dumps({
            'schema_version': MANIFEST_SCHEMA_VERSION,
            'generated_at': '2026-08-20T00:00:00Z',
            'generated_by': {'probe_level': 'full'},
            'cohorts': [], 'sessions': {'20251216_094334': record},
        }), encoding='utf-8')

        for analysis, block in record['analysis_readiness'].items():
            for combo in block['viable_params']:
                report = check_testable(analysis, '20251216', path=path, **combo)
                assert report.testable, (
                    f"readiness says {analysis} {combo} is viable but check_testable "
                    f"disagrees: {report.summary()}")


class TestBuildIsNeverFatal:
    def test_a_broken_session_yields_a_record_with_errors(self, tmp_path):
        """A sweep must not die on one bad session."""
        record = build_session_record(
            '20991231_000000', ['rat999'], cohort='cohort7',
            config_path=str(tmp_path / 'no_such_config.json'), probe_level='paths')
        assert record['session_id'] == '20991231_000000'
        assert record['provenance']['errors']
        assert 'analysis_readiness' in record

    def test_paths_level_records_that_it_did_not_probe_content(self, tmp_path):
        record = build_session_record(
            '20251216_094334', [], cohort='cohort7',
            config_path=str(tmp_path / 'nope.json'), probe_level='paths')
        assert record['provenance']['probe_level'] == 'paths'

    def test_records_probe_duration(self, tmp_path):
        record = build_session_record(
            '20251216_094334', [], cohort='cohort7',
            config_path=str(tmp_path / 'nope.json'), probe_level='paths')
        assert record['provenance']['probe_seconds'] >= 0


class TestArtifactWriting:
    def test_atomic_write_leaves_no_temp_file(self, tmp_path):
        manifest = new_manifest([], probe_level='paths', repo_root=tmp_path)
        path = atomic_write_manifest(manifest, tmp_path / 'out' / 'm.json')
        assert path.exists()
        assert not list(path.parent.glob('*.tmp'))

    def test_written_manifest_is_loadable(self, tmp_path):
        from discovery.capability_manifest import load_manifest
        manifest = new_manifest([], probe_level='paths', repo_root=tmp_path)
        path = atomic_write_manifest(manifest, tmp_path / 'm.json')
        assert load_manifest(path, use_cache=False)['schema_version'] == \
            manifest['schema_version']

    def test_merge_collects_session_errors_at_the_top_level(self, tmp_path):
        manifest = new_manifest([], probe_level='full', repo_root=tmp_path)
        merge_session_record(manifest, {
            'session_id': '20251216_094334',
            'provenance': {'errors': [{'stage': 'ephys', 'error': 'boom'}]},
        })
        assert manifest['build_errors'][0]['session'] == '20251216_094334'
        assert manifest['build_errors'][0]['stage'] == 'ephys'

    def test_manifest_records_the_probe_level_and_host(self, tmp_path):
        manifest = new_manifest([], probe_level='full', argv='--probe-level full',
                                repo_root=tmp_path)
        assert manifest['generated_by']['probe_level'] == 'full'
        assert manifest['generated_by']['host']
        assert manifest['generated_by']['argv'] == '--probe-level full'

    def test_stat_entry_marks_a_missing_file(self, tmp_path):
        entry = _stat_entry(tmp_path / 'nope.csv')
        assert entry['missing'] is True
        assert entry['size'] is None

    def test_stat_entry_records_size_and_mtime(self, tmp_path):
        target = tmp_path / 'f.csv'
        target.write_text('x', encoding='utf-8')
        entry = _stat_entry(target)
        assert entry['size'] == 1
        assert entry['mtime'] > 0

    def test_quality_thresholds_are_recorded_not_implied(self):
        """n_quality_cells is meaningless without the thresholds behind it."""
        assert set(DEFAULT_QUALITY_THRESHOLDS) == {
            'min_firing_rate', 'min_presence_ratio', 'max_cv_isi'}


class TestTheRealArtifact:
    """Assertions against the committed manifest, if one has been built."""

    @pytest.fixture
    def manifest(self):
        from discovery.capability_manifest import DEFAULT_MANIFEST_PATH, load_manifest
        if not DEFAULT_MANIFEST_PATH.exists():
            pytest.skip('no capability manifest has been built yet')
        return load_manifest(DEFAULT_MANIFEST_PATH, use_cache=False)

    def test_it_records_sessions(self, manifest):
        assert manifest['sessions']

    def test_every_session_has_a_date_and_cohort(self, manifest):
        for session_id, record in manifest['sessions'].items():
            assert record.get('cohort'), session_id
            assert record.get('session_date'), session_id

    def test_readiness_is_present_for_every_session(self, manifest):
        for session_id, record in manifest['sessions'].items():
            assert 'analysis_readiness' in record, session_id
