"""Pin the duration and presence-ratio rules behind the quality-cell set.

``duration_seconds`` comes from the raw sample-index span, and presence bins
start at each cell's own first spike. Both were kept on purpose (backlog item
B1, 2026-10-28): see the ``KilosortData.duration_seconds`` docstring.
"""

import numpy as np

from ingestion.kilosort_data_import import SAMPLE_RATE, KilosortData


def _ks(spike_times_by_cell, recording_sec):
    n = len(spike_times_by_cell)
    return KilosortData(
        animal_id='rat1', session_id='20250101',
        spike_times=np.array([0, int(recording_sec * SAMPLE_RATE)], dtype=np.int64),
        spike_clusters=np.zeros(2, dtype=np.int64),
        spike_times_by_cell=[np.asarray(st, dtype=float) for st in spike_times_by_cell],
        ks_ids=list(range(n)),
        channel=np.zeros(n, dtype=int), amplitude=np.zeros(n), fr=np.zeros(n), amp=np.zeros(n),
        DV=np.zeros(n), XX=np.zeros(n), cell_numbers=np.zeros((n, 2), dtype=int),
        to_load=np.ones(n, dtype=bool),
    )


def test_one_outlier_spike_does_not_move_duration_or_other_cells():
    steady = np.arange(0.5, 1000, 1.0)
    clean = _ks([steady, steady], recording_sec=1000)
    stray = _ks([steady, np.append(steady, 2000.0)], recording_sec=1000)

    assert stray.duration_seconds == clean.duration_seconds == 1000
    assert stray.ephys_window[1] == 2000.0
    m_clean, m_stray = clean.calculate_firing_pattern_metrics(), stray.calculate_firing_pattern_metrics()
    assert m_stray[0] == m_clean[0]
    assert m_stray[0]['presence_ratio'] == 1.0


def test_presence_bins_start_at_each_cells_first_spike():
    late = np.arange(500.5, 1000, 1.0)       # fires through the second half only
    m = _ks([late], recording_sec=1000).calculate_firing_pattern_metrics(time_bin_sec=100)[0]
    # Bins run from 500.5 s for 1000 s, so half of them fall after the recording ends.
    assert m['presence_ratio'] == 0.5
