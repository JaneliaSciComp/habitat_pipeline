"""Shared event-aligned firing-rate tensor builder.

Used by both [ephys/population_geometry.py](population_geometry.py) and
[ephys/_lda_decoding.py](_lda_decoding.py). Keep this thin and dependency-free.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def event_aligned_rates(spike_times_list: Sequence[np.ndarray],
                        align_times: np.ndarray,
                        time_window: Tuple[float, float],
                        bin_starts: np.ndarray,
                        bin_ends: np.ndarray,
                        time_bin_size: float) -> np.ndarray:
    """Return an ``(n_events, n_cells, n_bins)`` firing-rate tensor (Hz).

    For each (event, cell), count spikes in
    ``[align + bin_starts[b], align + bin_ends[b])`` and divide by
    ``time_bin_size``.

    Parameters
    ----------
    spike_times_list
        Length-``n_cells`` sequence of 1-D sorted arrays of spike times (s).
    align_times
        ``(n_events,)`` alignment times (s, same clock as the spikes).
    time_window
        ``(start, end)`` of the analysis window relative to each alignment.
    bin_starts, bin_ends
        ``(n_bins,)`` bin edges relative to alignment.
    time_bin_size
        Bin width used to convert spike counts to rates.
    """
    n_cells = len(spike_times_list)
    n_events = len(align_times)
    n_bins = len(bin_starts)
    rates = np.zeros((n_events, n_cells, n_bins), dtype=np.float32)
    for ci, spike_times in enumerate(spike_times_list):
        for ei, et in enumerate(align_times):
            in_window = (spike_times >= et + time_window[0]) & (spike_times < et + time_window[1])
            rel = np.sort(spike_times[in_window] - et)
            if len(rel) > 0:
                lo = np.searchsorted(rel, bin_starts, side='left')
                hi = np.searchsorted(rel, bin_ends, side='left')
                rates[ei, ci, :] = (hi - lo) / time_bin_size
    return rates
