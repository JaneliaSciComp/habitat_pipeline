"""Opponent and group labels skip events that name only one animal (``I`` introduce, ``RM`` remove)."""

import numpy as np
import pandas as pd

from video.behavioral_events import BehavioralEventsData


def _events():
    rows = []
    for i, opponent in enumerate(['rat615'] * 6 + ['rat630'] * 6):
        rows.append(('EC', 'rat613', opponent, float(10 * i)))
    rows.append(('I', 'rat613', np.nan, 500.0))     # introduce: no victim
    rows.append(('RM', np.nan, 'rat613', 600.0))    # remove: no initiator
    df = pd.DataFrame(rows, columns=['type', 'initiator', 'victim', 'ts_start_ephys'])
    df['ts_end_ephys'] = df['ts_start_ephys'] + 1.0
    return BehavioralEventsData(session_id='20251216', events_data=df, synchronized=True)


def test_opponent_labels_skip_events_without_an_opponent():
    starts, _, labels = _events().extract_opponent_labels('613', behavior_type=None)
    assert sorted(set(labels)) == ['rat615', 'rat630']
    assert len(starts) == 12 and 500.0 not in starts and 600.0 not in starts


def test_group_labels_skip_events_without_an_opponent():
    _, _, labels = _events().extract_group_labels('613', behavior_type=None)
    assert len(labels) == 12
    assert set(labels) == {'self', 'others'}
