"""find_sync_mapping must give exactly what the original brute-force double loop gave.

The search was vectorised (sliding windows + a first-interval pre-filter); the
reference below is the original nested loop, kept here as the specification.
"""

import numpy as np
import pytest

from ingestion.ephys_sync import find_sync_mapping, load_pulse_log


def _reference_matches(TSE, TSB, system_time, search_window=300, tol=0.02, L=10):
    intervals_E, intervals_B = np.diff(TSE), np.diff(TSB)
    valid = np.where((TSB >= system_time - search_window / 2) & (TSB <= system_time + search_window / 2))[0]
    search_start = int(valid[0]) if len(valid) else 0
    for start_b in range(search_start, len(intervals_B) - L + 1):
        seq = intervals_B[start_b:start_b + L]
        match_e = -1
        for t in range(len(intervals_E) - L + 1):
            if np.all(np.abs(seq - intervals_E[t:t + L]) <= tol):
                match_e = t
                break
        if match_e == -1:
            continue
        matches = [(TSE[match_e + i + 1], TSB[start_b + i + 1]) for i in range(L)]
        b, e = start_b + L, match_e + L
        while b < len(intervals_B) and e < len(intervals_E):
            if abs(intervals_B[b] - intervals_E[e]) > tol:
                break
            matches.append((TSE[e + 1], TSB[b + 1]))
            b += 1
            e += 1
        return matches
    return []


def _pulse_trains(seed, n_ephys=300, jitter=0.002, offset=1.7e9, gap=0):
    rng = np.random.default_rng(seed)
    intervals = rng.uniform(0.5, 5.0, n_ephys + 400)
    behaviour_clock = np.cumsum(intervals) + offset
    first = 150 + gap
    ephys_idx = np.arange(first, first + n_ephys)
    tse = (behaviour_clock[ephys_idx] - offset) * (1 - 2e-5) + 12.0
    tsb = behaviour_clock + rng.normal(0, jitter, len(behaviour_clock))
    return tse, tsb, float(behaviour_clock[first])


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('jitter', [0.001, 0.009, 0.012])   # 0.012: some intervals break tolerance
def test_matches_reference(seed, jitter):
    tse, tsb, t0 = _pulse_trains(seed, jitter=jitter)
    expected = _reference_matches(tse, tsb, t0)
    if len(expected) < 2:
        with pytest.raises(ValueError):
            find_sync_mapping(tse, tsb, t0)
        return
    got = find_sync_mapping(tse, tsb, t0)
    assert np.array_equal(got['matched_ephys'], [m[0] for m in expected])
    assert np.array_equal(got['matched_behavior'], [m[1] for m in expected])
    assert got['n_matches'] == len(expected)


def test_anchor_outside_log_scans_from_start():
    tse, tsb, _ = _pulse_trains(3)
    expected = _reference_matches(tse, tsb, system_time=0.0)
    got = find_sync_mapping(tse, tsb, 0.0)
    assert np.array_equal(got['matched_behavior'], [m[1] for m in expected])


def test_no_match_raises():
    rng = np.random.default_rng(9)
    tse = np.cumsum(rng.uniform(0.5, 5.0, 50))
    tsb = np.cumsum(np.full(200, 0.1)) + 1.7e9          # regular 0.1 s: nothing like the ephys train
    with pytest.raises(ValueError, match='No sync match'):
        find_sync_mapping(tse, tsb, 1.7e9)


def test_too_few_ephys_pulses_raises():
    with pytest.raises(ValueError, match='No sync match'):
        find_sync_mapping(np.array([1.0, 2.0, 3.5]), np.arange(100.0), 0.0)


def test_pulse_log_matches_line_by_line_parse(tmp_path):
    values = [1765335073639831351, 1765335083339333740, 1765335090000000001]
    path = tmp_path / 'pulse_log.txt'
    path.write_text('TimestampEST\n' + '\n'.join(map(str, values)) + '\n')
    expected = np.array([int(v) / 1e9 for v in values])
    got = load_pulse_log(path)
    assert np.array_equal(got, expected)
    got[0] = -1                                          # the cache hands out copies
    assert np.array_equal(load_pulse_log(path), expected)
