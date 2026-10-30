"""Behavioural events resolve to one ``<date>_behavior_event_df*.csv`` in the dated directory (HZ-DATA-007)."""

import json

import pytest

from ingestion.data_paths import get_event_files_by_date


def _config(tmp_path):
    path = tmp_path / 'paths.json'
    path.write_text(json.dumps({'ephys': str(tmp_path / 'ephys'), 'video': str(tmp_path / 'v'),
                                'tracking': str(tmp_path / 't'), 'events': str(tmp_path / 'events')}))
    return str(path)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('type\n')
    return path


def test_the_cohort7_layout_resolves_only_the_dated_file(tmp_path):
    events = tmp_path / 'events'
    canonical = _touch(events / '20251216' / '20251216_behavior_event_df.csv')
    # The older exports and the loose copy that live beside it on the share.
    for name in ('behavior_event_df_1.csv', 'behavior_event_df_orig.csv',
                 'after_ud_behavior_event_df_hierarchy_diff.csv'):
        _touch(events / '20251216' / name)
    _touch(events / 'behavior_event_df.csv')
    assert get_event_files_by_date('20251216_094334', config_path=_config(tmp_path)) == [canonical]


def test_the_cohort5_suffixed_name_still_resolves(tmp_path):
    f = _touch(tmp_path / 'events' / '20250819' / '20250819_behavior_event_df_hierarchy_diff.csv')
    assert get_event_files_by_date('20250819_141715', config_path=_config(tmp_path)) == [f]


def test_two_candidate_files_raise_rather_than_concatenate(tmp_path):
    _touch(tmp_path / 'events' / '20251216' / '20251216_behavior_event_df.csv')
    _touch(tmp_path / 'events' / '20251216' / '20251216_behavior_event_df_v2.csv')
    with pytest.raises(ValueError, match='Several behavioural event files'):
        get_event_files_by_date('20251216', config_path=_config(tmp_path))


def test_files_outside_the_dated_directory_are_never_read(tmp_path):
    events = tmp_path / 'events'
    _touch(events / 'archive' / '20251216_behavior_event_df.csv')            # undated directory
    _touch(events / '20251216' / 'old' / '20251216_behavior_event_df.csv')   # below the dated one
    _touch(events / '20251216_behavior_event_df.csv')                        # loose at the root
    assert get_event_files_by_date('20251216', config_path=_config(tmp_path)) == []
