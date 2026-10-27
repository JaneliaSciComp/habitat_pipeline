"""Pin the Kilosort loader's internals to the straightforward algorithms they replaced.

The loader memory-maps timestamps.dat, groups spikes with one lexsort and
selects cluster rows with boolean masks. These tests restate the original,
obviously-correct formulations inline and require identical output.
"""

import numpy as np
import pandas as pd
import pytest

from ingestion.kilosort_data_import import (
    SAMPLE_RATE,
    _extract_cluster_properties,
    _group_spikes_by_cluster,
    _sample_counts_at,
    _select_clusters,
    _waveform_to_channel,
)


def _write_timestamps(parent, counts, trodes_header):
    with open(parent / 'rec.timestamps.dat', 'wb') as f:
        if trodes_header:
            f.write(b'<Start settings>\r\nClockrate: 30000\r\nFields: <time uint32>\r\n<End settings>\n')
        else:
            for i in range(25):
                f.write(f"Header line {i}\n".encode())
        f.write(np.asarray(counts, dtype='<u4').tobytes())


@pytest.mark.parametrize('trodes_header', [True, False])
def test_timestamps_lookup_matches_full_read(tmp_path, trodes_header):
    ks4 = tmp_path / 'kilosort4'
    ks4.mkdir()
    counts = np.arange(5000, 7000, 3, dtype=np.uint32)
    _write_timestamps(tmp_path, counts, trodes_header)
    picked = _sample_counts_at(ks4, np.array([0, 5, len(counts) - 1, 5]))
    assert type(picked) is np.ndarray and picked.dtype == np.dtype('<u4')
    assert picked.tolist() == [counts[0], counts[5], counts[-1], counts[5]]
    assert np.array_equal(_sample_counts_at(ks4, np.arange(len(counts))), counts)
    (tmp_path / 'rec.timestamps.dat').unlink()        # not left locked by a mapping


def test_group_spikes_matches_per_cluster_sort(tmp_path):
    rng = np.random.default_rng(0)
    ks4 = tmp_path / 'kilosort4'
    ks4.mkdir()
    counts = np.sort(rng.choice(10**6, 20000, replace=False)).astype(np.uint32)
    _write_timestamps(tmp_path, counts, trodes_header=True)

    spike_times = rng.integers(0, len(counts), 5000)
    spike_clusters = rng.integers(0, 12, 5000)
    ks_ids = [7, 0, 3, 99, 11]            # 99 has no spikes

    got = _group_spikes_by_cluster(ks4, spike_times, spike_clusters, ks_ids)

    seconds = counts.astype(np.uint64)[spike_times].astype(float) / SAMPLE_RATE
    for cid, arr in zip(ks_ids, got):
        expected = np.sort(seconds[spike_clusters == cid])
        assert type(arr) is np.ndarray and arr.dtype == np.float64
        assert np.array_equal(arr, expected)


def test_waveform_to_channel_matches_loop(tmp_path):
    templates = np.random.default_rng(1).normal(size=(6, 20, 9))
    np.save(tmp_path / 'templates.npy', templates)
    expected = [int(np.argmax(t.max(axis=0) - t.min(axis=0))) for t in templates]
    assert _waveform_to_channel(tmp_path).tolist() == expected


def _curated_folder(tmp_path, n_good):
    channel_map = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    np.save(tmp_path / 'channel_positions.npy',
            np.column_stack([np.arange(5) * 100.0, np.arange(5) * 20.0]))
    info = pd.DataFrame({
        'cluster_id': [4, 2, 9, 5],
        'group': ['good'] * n_good + ['mua'] * (4 - n_good),
        'ch': [13, 10, 14, 11],
        'depth': [60.0, 0.0, 80.0, 20.0],
        'Amplitude': [1.5, 2.5, 3.5, 4.5], 'amp': [5.0, 6.0, 7.0, 8.0], 'fr': [0.1, 0.2, 0.3, 0.4],
    })
    return channel_map, info


def test_curated_properties(tmp_path):
    channel_map, info = _curated_folder(tmp_path, n_good=3)
    to_load, curated = _select_clusters(info, None)
    assert to_load.tolist() == [True, True, True, False]
    assert curated.tolist() == [True, True, True]

    props = _extract_cluster_properties(tmp_path, to_load, info, None, channel_map)
    assert props['ks_ids'] == [4, 2, 9]
    assert props['channel'].tolist() == [13, 10, 14]
    assert props['amplitude'].tolist() == [1.5, 2.5, 3.5]
    assert props['DV'].tolist() == [60.0, 0.0, 80.0]        # site 3, 0, 4
    assert props['XX'].tolist() == [300.0, 0.0, 400.0]
    assert props['cell_numbers'].tolist() == [[3, 0], [0, 1], [4, 2]]


def test_single_cluster_properties_stay_one_dimensional(tmp_path):
    channel_map, info = _curated_folder(tmp_path, n_good=1)
    to_load, _ = _select_clusters(info, None)
    props = _extract_cluster_properties(tmp_path, to_load, info, None, channel_map)
    for key in ('channel', 'amplitude', 'amp', 'fr', 'DV', 'XX'):
        assert props[key].shape == (1,), key


def test_uncurated_properties(tmp_path):
    channel_map = np.array([20, 21, 22], dtype=np.int32)
    np.save(tmp_path / 'channel_positions.npy', np.array([[0.0, 5.0], [100.0, 15.0], [200.0, 25.0]]))
    templates = np.zeros((3, 4, 3))
    for template, site in enumerate([2, 0, 1]):
        templates[template, 1, site] = 1.0
    np.save(tmp_path / 'templates.npy', templates)
    labels = pd.DataFrame({'cluster_id': [0, 1, 2], 'KSLabel': ['good', 'mua', 'Good']})
    pd.DataFrame({'cluster_id': [0, 1, 2], 'Amplitude': [9.0, 8.0, 7.0]}).to_csv(
        tmp_path / 'cluster_Amplitude.tsv', sep='\t', index=False)

    to_load, curated = _select_clusters(None, labels)
    assert to_load.tolist() == [True, False, True] and curated is None
    props = _extract_cluster_properties(tmp_path, to_load, None, labels, channel_map)
    assert props['ks_ids'] == [0, 2]
    assert props['channel'].tolist() == [22, 21]
    assert props['amplitude'].tolist() == [9.0, 7.0]
    assert props['DV'].tolist() == [25.0, 15.0]
    assert props['fr'].size == 0 and props['amp'].size == 0
