"""The typed ingestion errors: which one is raised, and that old handlers still catch it."""

import json

import numpy as np
import pytest

from ingestion.data_paths import get_animals_and_sessions, get_kilosort_path, verify_kilosort_path
from ingestion.errors import DataRootUnavailable, MissingCompanionFile, RecordingNotFound
from ingestion.kilosort_data_import import _sample_counts_at


def _config(tmp_path, ephys):
    path = tmp_path / 'paths.json'
    path.write_text(json.dumps({'ephys': str(ephys), 'video': str(tmp_path / 'v'),
                                'tracking': str(tmp_path / 't'), 'events': str(tmp_path / 'e')}))
    return str(path)


def _recording(tmp_path):
    animal = tmp_path / 'ephys' / '20251210_110059.rec' / 'rat613'
    ks4 = animal / '20251210_110059_merged.kilosort' / 'kilosort4'
    ks4.mkdir(parents=True)
    for name in ('spike_times.npy', 'spike_clusters.npy'):
        np.save(ks4 / name, np.zeros(1, dtype=np.int64))
    return ks4


@pytest.mark.parametrize('cls', [DataRootUnavailable, RecordingNotFound, MissingCompanionFile])
def test_every_error_is_still_a_file_not_found_error(cls):
    assert issubclass(cls, FileNotFoundError)


def test_unreachable_ephys_root(tmp_path):
    config = _config(tmp_path, tmp_path / 'not_mounted')
    with pytest.raises(DataRootUnavailable, match='Ephys directory not found'):
        get_kilosort_path('613', '20251210', config_path=config)
    with pytest.raises(DataRootUnavailable):
        get_animals_and_sessions(config_path=config)


def test_unknown_session_and_animal(tmp_path):
    _recording(tmp_path)
    config = _config(tmp_path, tmp_path / 'ephys')
    with pytest.raises(RecordingNotFound, match='No session directory'):
        get_kilosort_path('613', '20240101', config_path=config)
    with pytest.raises(RecordingNotFound, match='No animal directory'):
        get_kilosort_path('999', '20251210', config_path=config)
    with pytest.raises(RecordingNotFound, match="No recording '20251210_030303'"):
        get_kilosort_path('613', '20251210_030303', config_path=config)


def test_exists_versus_loadable(tmp_path):
    ks4 = _recording(tmp_path)
    assert verify_kilosort_path(ks4)
    assert not verify_kilosort_path(ks4, check_loadable=True)
    with pytest.raises(MissingCompanionFile, match="timestamps.dat"):
        _sample_counts_at(ks4, np.array([0]))

    (ks4.parent / '20251210_110059_merged.timestamps.dat').write_bytes(np.arange(4, dtype='<u4').tobytes())
    assert verify_kilosort_path(ks4, check_loadable=True)
