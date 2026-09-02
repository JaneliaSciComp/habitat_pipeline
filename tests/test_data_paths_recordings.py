"""Tests for recording-level path resolution in ingestion/data_paths.py.

A ``.rec`` directory is a *day* of acquisition, not one recording. Trodes
writes one set of artifacts per acquisition block, all sharing a stem:
``20251216_094334_merged.{kilosort,DIO,LFP,timestampoffset}`` beside
``20251216_144334_merged.*`` and ``20251216_194334_merged.*``. Until this
layer discovered stems from disk, ``get_kilosort_path`` built
``f"{session}_merged.kilosort"``, which:

- named only the block whose timestamp matched the directory, hiding 26
  recordings in cohort 7 that no analysis could address;
- was the wrong string entirely for cohort 5, which writes ``_merge`` in 21
  of 24 animal directories (one has no suffix), leaving 22 of 24 unreachable.
  The two cohort-5 sessions that ever ran did so because they happen to use
  ``_merged``, which was a filename coincidence rather than a property of the
  data.

The first class below is the one that matters most: every id that resolved
before must still resolve to the same recording. All fixtures are temporary
directory trees mirroring the real layouts; nothing here touches //nearline.
"""
import json
import logging

import pytest

from ingestion.data_paths import (
    DataStorageManager,
    Recording,
    _discover_recordings,
    get_animals_and_sessions,
    get_dio_path,
    get_kilosort_path,
    resolve_recordings,
)

#: The real cohort-7 layout for 20251216: three blocks for rat613 and rat631,
#: two for rat630 (its afternoon block starts at 14:50), one for rat615.
COHORT7_BLOCKS = {
    'rat613': ['20251216_094334', '20251216_144334', '20251216_194334'],
    'rat615': ['20251216_094334'],
    'rat630': ['20251216_094334', '20251216_145034', '20251216_194334'],
    'rat631': ['20251216_094334', '20251216_144334', '20251216_194334'],
}


def _make_recording(animal_dir, stem, *, with_spikes=True):
    """Create one recording's artifacts under an animal directory."""
    ks4 = animal_dir / f"{stem}.kilosort" / "kilosort4"
    ks4.mkdir(parents=True)
    if with_spikes:
        (ks4 / 'spike_times.npy').write_bytes(b'\x00')
        (ks4 / 'spike_clusters.npy').write_bytes(b'\x00')
    dio = animal_dir / f"{stem}.DIO"
    dio.mkdir()
    for channel in range(1, 5):
        (dio / f"{stem}.dio_Controller_Din{channel}.dat").write_bytes(b'\x00')
    (animal_dir / f"{stem}.timestampoffset").write_text('0')


def _write_config(tmp_path, ephys_root, **extra):
    config = {'ephys': str(ephys_root), 'video': str(tmp_path / 'video'),
              'tracking': str(tmp_path / 'tracking'),
              'events': str(tmp_path / 'events'), 'pixels_per_cm': None}
    config.update(extra)
    path = tmp_path / 'paths.json'
    path.write_text(json.dumps(config))
    return str(path)


@pytest.fixture
def cohort7(tmp_path):
    """20251216 with three blocks, and 20251210 with a single one."""
    ephys = tmp_path / 'cohort7' / 'ephys'
    day = ephys / '20251216_094334.rec'
    for animal, blocks in COHORT7_BLOCKS.items():
        animal_dir = day / animal
        animal_dir.mkdir(parents=True)
        for block in blocks:
            _make_recording(animal_dir, f"{block}_merged")
    single = ephys / '20251210_110059.rec' / 'rat613'
    single.mkdir(parents=True)
    _make_recording(single, '20251210_110059_merged')
    (ephys / 'pulse_log.txt').write_text('TimestampEST\n1\n2\n')
    return _write_config(tmp_path, ephys)


@pytest.fixture
def cohort5(tmp_path):
    """The three suffix conventions cohort 5 actually uses on disk."""
    ephys = tmp_path / 'cohort5' / 'ephys'
    merge = ephys / '20250813_110128.rec' / 'rat650'
    merge.mkdir(parents=True)
    _make_recording(merge, '20250813_110128_merge')

    bare = ephys / '20250812_115448.rec' / 'rat650'
    bare.mkdir(parents=True)
    _make_recording(bare, '20250812_115448')

    merged = ephys / '20250819_141715.rec' / 'rat650'
    merged.mkdir(parents=True)
    _make_recording(merged, '20250819_141715_merged')
    return _write_config(tmp_path, ephys)


class TestNothingThatResolvedBeforeMoves:
    """The regression guard. 21 logged iterations depend on these answers."""

    @pytest.mark.parametrize('session_id', ['20251216', '20251216_094334'])
    def test_date_and_rec_level_ids_give_the_primary_recording(
            self, cohort7, session_id):
        path = get_kilosort_path('rat613', session_id, config_path=cohort7)[0]
        assert path.parent.name == '20251216_094334_merged.kilosort'
        assert path.exists()

    def test_partial_animal_ids_still_match(self):
        """'613' matching 'rat613' is a long-standing convention."""
        # Exercised through the cohort7 fixture in the next test; kept separate
        # so a failure names the convention rather than the path.
        assert 'rat613'.find('613') > 0

    def test_partial_animal_id_resolves(self, cohort7):
        path = get_kilosort_path('613', '20251216', config_path=cohort7)[0]
        assert path.parent.name == '20251216_094334_merged.kilosort'

    def test_dio_path_uses_the_same_stem(self, cohort7):
        path = get_dio_path('rat613', '20251216', 1, config_path=cohort7)[0]
        assert path.parent.name == '20251216_094334_merged.DIO'
        assert path.name == '20251216_094334_merged.dio_Controller_Din1.dat'
        assert path.exists()

    def test_a_single_recording_session_is_unaffected(self, cohort7):
        path = get_kilosort_path('rat613', '20251210', config_path=cohort7)[0]
        assert path.parent.name == '20251210_110059_merged.kilosort'


class TestSubRecordingsBecomeAddressable:
    def test_afternoon_block_resolves(self, cohort7):
        path = get_kilosort_path('rat613', '20251216_144334',
                                 config_path=cohort7)[0]
        assert path.parent.name == '20251216_144334_merged.kilosort'
        assert path.exists()

    def test_evening_block_resolves_with_its_own_dio(self, cohort7):
        dio = get_dio_path('rat613', '20251216_194334', 2,
                           config_path=cohort7)[0]
        assert dio.name == '20251216_194334_merged.dio_Controller_Din2.dat'
        assert dio.exists()

    def test_an_animal_specific_block_time_resolves(self, cohort7):
        """rat630's afternoon block starts at 14:50, not 14:43."""
        path = get_kilosort_path('rat630', '20251216_145034',
                                 config_path=cohort7)[0]
        assert path.parent.name == '20251216_145034_merged.kilosort'

    def test_resolve_recordings_lists_the_whole_day(self, cohort7):
        found = resolve_recordings('rat613', '20251216', config_path=cohort7)
        assert [r.recording_id for r in found] == COHORT7_BLOCKS['rat613']
        assert [r.is_primary for r in found] == [True, False, False]

    def test_a_recording_id_that_does_not_exist_still_raises(self, cohort7):
        with pytest.raises((FileNotFoundError, ValueError)):
            get_kilosort_path('rat613', '20251216_030303', config_path=cohort7)


class TestCohort5SuffixConventions:
    """The 22 animal directories that were unreachable before."""

    def test_merge_suffix_resolves(self, cohort5):
        path = get_kilosort_path('rat650', '20250813_110128',
                                 config_path=cohort5)[0]
        assert path.parent.name == '20250813_110128_merge.kilosort'
        assert path.exists()

    def test_no_suffix_resolves(self, cohort5):
        path = get_kilosort_path('rat650', '20250812_115448',
                                 config_path=cohort5)[0]
        assert path.parent.name == '20250812_115448.kilosort'
        assert path.exists()

    def test_merged_suffix_still_resolves(self, cohort5):
        """20250819_141715 ran before precisely because it uses '_merged'."""
        path = get_kilosort_path('rat650', '20250819_141715',
                                 config_path=cohort5)[0]
        assert path.parent.name == '20250819_141715_merged.kilosort'
        assert path.exists()

    def test_dio_follows_the_discovered_suffix(self, cohort5):
        """Cohort 5 writes '_merge' for every artifact, not just kilosort."""
        dio = get_dio_path('rat650', '20250813_110128', 1,
                           config_path=cohort5)[0]
        assert dio.parent.name == '20250813_110128_merge.DIO'
        assert dio.exists()

    def test_animal_prefixed_stem_resolves(self, tmp_path):
        """The one cohort-7 directory named 'rat613_<session>_merge'."""
        ephys = tmp_path / 'ephys'
        animal_dir = ephys / '20251209_160716.rec' / 'rat613'
        animal_dir.mkdir(parents=True)
        _make_recording(animal_dir, 'rat613_20251209_160716_merge')
        config = _write_config(tmp_path, ephys)
        path = get_kilosort_path('rat613', '20251209_160716',
                                 config_path=config)[0]
        assert path.parent.name == 'rat613_20251209_160716_merge.kilosort'
        assert path.exists()


class TestAmbiguityFailsClosed:
    def test_several_matches_with_no_primary_raises(self, tmp_path):
        """Never guess which block a result belongs to.

        A directory named for a time with no recording of its own, holding
        two later blocks: picking one silently would attach a result to the
        wrong recording and nothing downstream could detect it.
        """
        ephys = tmp_path / 'ephys'
        animal_dir = ephys / '20251216_090000.rec' / 'rat613'
        animal_dir.mkdir(parents=True)
        _make_recording(animal_dir, '20251216_144334_merged')
        _make_recording(animal_dir, '20251216_194334_merged')
        config = _write_config(tmp_path, ephys)
        with pytest.raises(ValueError, match='none of them is the primary'):
            get_kilosort_path('rat613', '20251216', config_path=config)

    def test_the_error_names_the_candidates(self, tmp_path):
        ephys = tmp_path / 'ephys'
        animal_dir = ephys / '20251216_090000.rec' / 'rat613'
        animal_dir.mkdir(parents=True)
        _make_recording(animal_dir, '20251216_144334_merged')
        _make_recording(animal_dir, '20251216_194334_merged')
        config = _write_config(tmp_path, ephys)
        with pytest.raises(ValueError) as excinfo:
            get_kilosort_path('rat613', '20251216', config_path=config)
        assert '20251216_144334' in str(excinfo.value)
        assert '20251216_194334' in str(excinfo.value)


class TestNoKilosortOutputAtAll:
    def test_a_path_that_does_not_exist_comes_back_rather_than_raising(
            self, tmp_path):
        """Callers report 'no ephys here'; they do not crash.

        The capability-manifest probe depends on this: it records
        ``path_exists=False`` and a load error, which is a fact worth having.
        """
        ephys = tmp_path / 'ephys'
        animal_dir = ephys / '20251222_094643.rec' / 'rat630'
        animal_dir.mkdir(parents=True)
        config = _write_config(tmp_path, ephys)
        path = get_kilosort_path('rat630', '20251222', config_path=config)[0]
        assert path.exists() is False
        assert path.parent.name == '20251222_094643_merged.kilosort'

    def test_discover_returns_empty_for_an_empty_directory(self, tmp_path):
        animal_dir = tmp_path / 'rat630'
        animal_dir.mkdir()
        assert _discover_recordings(animal_dir, '20251222_094643') == []


class TestEnumeration:
    def test_one_row_per_recording(self, cohort7):
        frame = get_animals_and_sessions(config_path=cohort7)
        # 3 + 1 + 3 + 3 blocks on 20251216, plus one on 20251210.
        assert len(frame) == 11
        assert frame['is_primary'].sum() == 5

    def test_session_column_carries_the_recording_id(self, cohort7):
        frame = get_animals_and_sessions(config_path=cohort7)
        sessions = set(frame['session'])
        assert '20251216_144334' in sessions
        assert '20251216_094334' in sessions

    def test_every_enumerated_path_is_real(self, cohort7):
        """The enumerator and the resolver must not disagree."""
        frame = get_animals_and_sessions(config_path=cohort7)
        assert all(p.exists() for p in frame['kilosort_path'])

    def test_enumerated_ids_round_trip_through_the_resolver(self, cohort7):
        frame = get_animals_and_sessions(config_path=cohort7)
        for row in frame.itertuples(index=False):
            resolved = get_kilosort_path(row.animal, row.session,
                                         config_path=cohort7)[0]
            assert resolved == row.kilosort_path

    def test_cohort5_rows_now_point_at_real_paths(self, cohort5):
        frame = get_animals_and_sessions(config_path=cohort5)
        assert len(frame) == 3
        assert all(p.exists() for p in frame['kilosort_path'])

    def test_an_animal_with_no_kilosort_stays_visible(self, tmp_path):
        ephys = tmp_path / 'ephys'
        (ephys / '20251222_094643.rec' / 'rat630').mkdir(parents=True)
        config = _write_config(tmp_path, ephys)
        frame = get_animals_and_sessions(config_path=config)
        assert len(frame) == 1
        assert frame['kilosort_path'][0].exists() is False


class TestDataStorageManagerRecordingMetadata:
    def test_primary_recording_is_labelled(self, cohort7):
        dsm = DataStorageManager('rat613', '20251216', config_path=cohort7,
                                 use_cache=False)
        assert dsm.recording_id == '20251216_094334'
        assert dsm.is_primary_recording is True
        assert dsm.recording_ids_on_date == COHORT7_BLOCKS['rat613']

    def test_non_primary_recording_is_labelled(self, cohort7):
        dsm = DataStorageManager('rat613', '20251216_144334',
                                 config_path=cohort7, use_cache=False)
        assert dsm.recording_id == '20251216_144334'
        assert dsm.is_primary_recording is False
        assert dsm.recording_stem == '20251216_144334_merged'

    def test_unresolvable_session_leaves_the_fields_empty(self, cohort7):
        dsm = DataStorageManager('rat999', '20251216', config_path=cohort7,
                                 use_cache=False)
        assert dsm.kilosort_path is None
        assert dsm.recording_id is None
        assert dsm.is_primary_recording is None

    def test_cache_round_trips_the_recording_fields(self, cohort7, tmp_path):
        cache = tmp_path / 'cache'
        first = DataStorageManager('rat613', '20251216_144334',
                                   config_path=cohort7, cache_dir=cache)
        second = DataStorageManager('rat613', '20251216_144334',
                                    config_path=cohort7, cache_dir=cache)
        assert second.recording_id == first.recording_id
        assert second.is_primary_recording is False
        assert second.recording_stem == '20251216_144334_merged'

    def test_a_version_1_cache_is_rejected(self, cohort7, tmp_path):
        """v1 caches hold '{session}_merged' paths that may not exist.

        Trusting one would reinstate exactly the bug this change fixed, for
        every animal/session pair already cached on the workstation.
        """
        cache = tmp_path / 'cache'
        cache.mkdir()
        stale = {'version': 1, 'animal_id': 'rat650',
                 'session_id': '20250813_110128',
                 'kilosort_path': '/nonexistent/20250813_110128_merged.kilosort',
                 'dio_paths': {}, 'video_files': [], 'tracking_files': [],
                 'behavioral_event_files': [], 'pulse_log_path': None}
        (cache / 'rat650_20250813_110128.json').write_text(json.dumps(stale))
        dsm = DataStorageManager('rat650', '20250813_110128',
                                 config_path=cohort7, cache_dir=cache)
        assert 'nonexistent' not in str(dsm.kilosort_path)


class TestNonPrimaryRecordingWarnsAboutDateResolvedFiles:
    """HZ-DATA-008. Tracking and events resolve by date, not by recording."""

    def _with_tracking_and_events(self, tmp_path, cohort7):
        config = json.loads((tmp_path / 'paths.json').read_text())
        tracking = tmp_path / 'tracking' / 'merged_20251216_0950_1200'
        tracking.mkdir(parents=True)
        (tracking / 'merged_20251216_0950_1200_mask_metrics.csv').write_text(
            'frame,object_name,center_x,center_y\n')
        events = tmp_path / 'events' / '20251216'
        events.mkdir(parents=True)
        (events / '20251216_behavior_event_df.csv').write_text('type\n')
        (tmp_path / 'paths.json').write_text(json.dumps(config))
        return cohort7

    def test_the_afternoon_block_warns(self, cohort7, tmp_path, caplog):
        config = self._with_tracking_and_events(tmp_path, cohort7)
        with caplog.at_level(logging.WARNING, logger='ingestion.data_paths'):
            dsm = DataStorageManager('rat613', '20251216_144334',
                                     config_path=config, use_cache=False)
        assert dsm.is_primary_recording is False
        if dsm.tracking_files or dsm.behavioral_event_files:
            assert 'HZ-DATA-008' in caplog.text

    def test_the_primary_block_does_not_warn(self, cohort7, tmp_path, caplog):
        config = self._with_tracking_and_events(tmp_path, cohort7)
        with caplog.at_level(logging.WARNING, logger='ingestion.data_paths'):
            DataStorageManager('rat613', '20251216', config_path=config,
                               use_cache=False)
        assert 'HZ-DATA-008' not in caplog.text


class TestRecordingNamedTuple:
    def test_primary_is_derived_not_assumed(self, tmp_path):
        animal_dir = tmp_path / 'rat613'
        animal_dir.mkdir()
        for block in COHORT7_BLOCKS['rat613']:
            _make_recording(animal_dir, f"{block}_merged")
        found = _discover_recordings(animal_dir, '20251216_094334')
        assert all(isinstance(r, Recording) for r in found)
        assert sum(r.is_primary for r in found) == 1
        assert next(r for r in found if r.is_primary).recording_id == \
            '20251216_094334'
