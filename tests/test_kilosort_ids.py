"""``KilosortData`` ids come from the recording stamp in the ``.kilosort`` directory name."""

from pathlib import Path

import pytest

from ingestion.kilosort_data_import import _extract_ids_from_path


@pytest.mark.parametrize('recording_dir, session', [
    ('20251216_094334_merged.kilosort', '20251216_094334'),      # cohort 7
    ('20250812_115448_merge.kilosort', '20250812_115448'),       # cohort 5
    ('20250812_115448.kilosort', '20250812_115448'),             # cohort 5, no suffix
    ('rat613_20251209_160716_merge.kilosort', '20251209_160716'),
    ('rat20260927_103821_merged.kilosort', '20260927_103821'),   # cohort 9
])
def test_session_is_the_recording_stamp(recording_dir, session):
    ks4 = Path('share') / '20251216_094334.rec' / 'rat613' / recording_dir / 'kilosort4'
    assert _extract_ids_from_path(ks4) == ('rat613', session)


@pytest.mark.parametrize('ks4', [
    Path('rat613') / 'no_stamp_merged.kilosort' / 'kilosort4',
    Path('rat613') / '20251216_094334_merged' / 'kilosort4',          # not a .kilosort directory
    Path('rat613') / '20251216_094334_merged.kilosort' / 'other',
])
def test_anything_else_is_unknown(ks4):
    assert _extract_ids_from_path(ks4) == ('unknown_animal', 'unknown_session')
