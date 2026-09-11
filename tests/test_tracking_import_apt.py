"""Tests for the wide APT/TQT tracking format in video/tracking_import.py.

All mock data; nothing here touches //nearline. The fixtures reproduce the
three things the real export does that the older mask-metrics format does not:
a UTF-8 byte-order mark, animals named ``rat4635`` for rat 635, and blank cells
for frames where an animal was not detected.

What is pinned here:

`TestFrameNumbering` is the one that matters. ``VideoTrackingData.timestamps``
is indexed *by frame number* through ``_index_by_frame``, so if wide-format
rows were numbered from 1, or renumbered after dropping undetected frames,
every animal would silently read a neighbour's timestamp — and worse, animals
with different numbers of missing frames would each be wrong by a different
amount, which no downstream check could detect.

`TestBackwardsCompatibility` pins that the long format still parses exactly as
before, since `parse_tracking` now dispatches on the header.
"""
import numpy as np
import pandas as pd
import pytest

from video.tracking_import import (
    TRACKING_FORMAT_APT_TQT,
    TRACKING_FORMAT_MASK_METRICS,
    TRACKING_FORMAT_UNKNOWN,
    detect_tracking_format,
    load_tracking_data,
    load_timestamps,
    normalize_object_name,
    parse_apt_tracking,
    parse_tracking,
    read_tracking_centers,
    read_tracking_header,
)

#: 40 Hz, as the real exports are.
_FRAME_NS = 25_000_000
_T0 = 1_765_904_400_049_644_328


def _apt_frame(n_frames=10, animals=('rat4631', 'rat4613'), missing=None):
    """A wide APT/TQT frame; *missing* maps animal -> frames to blank out."""
    missing = missing or {}
    data = {'timestamp': _T0 + np.arange(n_frames, dtype=np.int64) * _FRAME_NS}
    for i, animal in enumerate(animals):
        x = np.arange(n_frames, dtype=float) + i * 1000.0
        y = np.arange(n_frames, dtype=float) + i * 2000.0
        kp = np.arange(n_frames, dtype=float) + i * 3000.0
        for frame_no in missing.get(animal, ()):
            x[frame_no] = np.nan
            y[frame_no] = np.nan
        data[f'{animal}_center_x'] = x
        data[f'{animal}_center_y'] = y
        data[f'{animal}_kp0_x'] = kp
        data[f'{animal}_kp0_y'] = kp
    return pd.DataFrame(data)


def _write_apt(tmp_path, name='TQT_named.csv', bom=True, **kwargs):
    path = tmp_path / name
    # The real export is written with a byte-order mark.
    _apt_frame(**kwargs).to_csv(path, index=False,
                                encoding='utf-8-sig' if bom else 'utf-8')
    return path


def _write_mask_metrics(tmp_path, name='sess_mask_metrics.csv', n_frames=6):
    path = tmp_path / name
    rows = []
    for frame_no in range(1, n_frames + 1):
        for object_id, object_name in ((1, 'rat631'), (2, 'rat613')):
            rows.append({'frame': frame_no, 'object_id': object_id,
                         'object_name': object_name, 'area': 4000 + frame_no,
                         'center_x': float(frame_no), 'center_y': float(frame_no * 2)})
    pd.DataFrame(rows).to_csv(path, index=False)
    np.save(tmp_path / name.replace('_mask_metrics.csv', '_ts.npy'),
            _T0 + np.arange(n_frames + 1, dtype=np.int64) * _FRAME_NS)
    return path


class TestFormatDetection:
    def test_wide_frame_is_apt(self):
        assert detect_tracking_format(_apt_frame().columns) == TRACKING_FORMAT_APT_TQT

    def test_long_frame_is_mask_metrics(self):
        columns = ['frame', 'object_id', 'object_name', 'center_x', 'center_y']
        assert detect_tracking_format(columns) == TRACKING_FORMAT_MASK_METRICS

    def test_neither_is_unknown(self):
        assert detect_tracking_format(['a', 'b']) == TRACKING_FORMAT_UNKNOWN

    def test_timestamp_alone_is_not_enough(self):
        """A timestamp column without position columns names no objects."""
        assert detect_tracking_format(['timestamp', 'x']) == TRACKING_FORMAT_UNKNOWN

    def test_byte_order_mark_does_not_hide_the_timestamp_column(self, tmp_path):
        """Read as plain utf-8 the first column is '﻿timestamp'."""
        header = read_tracking_header(_write_apt(tmp_path, bom=True))
        assert header[0] == 'timestamp'
        assert detect_tracking_format(header) == TRACKING_FORMAT_APT_TQT


class TestNameNormalisation:
    @pytest.mark.parametrize('raw,expected', [
        ('rat4635', 'rat635'),
        ('rat4613', 'rat613'),
        ('rat4631', 'rat631'),
    ])
    def test_extra_four_is_stripped(self, raw, expected):
        assert normalize_object_name(raw) == expected

    @pytest.mark.parametrize('raw', ['rat631', 'rat46351', 'object_1', 'rat4'])
    def test_other_names_pass_through(self, raw):
        assert normalize_object_name(raw) == raw

    def test_parsed_keys_match_the_ids_used_elsewhere(self):
        parsed = parse_apt_tracking(_apt_frame(animals=('rat4635', 'rat4630')))
        assert set(parsed) == {'rat635', 'rat630'}

    def test_collision_keeps_both_objects(self):
        """Normalising must never silently drop an animal's trajectory.

        The awkward case is a file holding both ``rat4631`` and ``rat631``:
        falling back to the raw name collides a second time.
        """
        parsed = parse_apt_tracking(_apt_frame(animals=('rat4631', 'rat631')))
        assert len(parsed) == 2, 'one animal was overwritten by the other'
        # Positions distinguish them: the second animal is offset by 1000.
        assert parsed['rat631']['center_x'].iloc[0] == 0.0
        assert [obj['center_x'].iloc[0] for obj in parsed.values()] == [0.0, 1000.0]


class TestFrameNumbering:
    """``timestamps[frame]`` must be that row's own timestamp, per animal."""

    def test_frames_are_zero_based_row_indices(self):
        parsed = parse_apt_tracking(_apt_frame(n_frames=5))
        assert parsed['rat631']['frame'].tolist() == [0, 1, 2, 3, 4]

    def test_missing_frames_are_dropped_not_renumbered(self):
        frame = _apt_frame(n_frames=6, missing={'rat4631': (1, 3)})
        parsed = parse_apt_tracking(frame)
        assert parsed['rat631']['frame'].tolist() == [0, 2, 4, 5]
        # The other animal is untouched.
        assert parsed['rat613']['frame'].tolist() == [0, 1, 2, 3, 4, 5]

    def test_trajectory_timestamps_survive_dropped_frames(self, tmp_path):
        """The whole point: a gap in one animal must not shift its clock."""
        path = _write_apt(tmp_path, n_frames=6, missing={'rat4631': (1, 3)})
        tracking = load_tracking_data(path)
        traj = tracking.get_object_trajectory('rat631')
        expected = [_T0 + f * _FRAME_NS for f in (0, 2, 4, 5)]
        assert traj['timestamps'].tolist() == pytest.approx(expected)

    def test_positions_stay_with_their_own_frame(self, tmp_path):
        path = _write_apt(tmp_path, n_frames=6, missing={'rat4631': (1, 3)})
        traj = load_tracking_data(path).get_object_trajectory('rat631')
        # x was built as float(frame), so the check is direct.
        assert traj['center_x'].tolist() == traj['frame'].tolist()


class TestParsedContents:
    def test_columns_are_ordered_and_complete(self):
        parsed = parse_apt_tracking(_apt_frame())
        assert list(parsed['rat631'].columns) == [
            'frame', 'timestamp', 'center_x', 'center_y', 'kp0_x', 'kp0_y']

    def test_keypoints_are_kept(self):
        parsed = parse_apt_tracking(_apt_frame())
        assert parsed['rat613']['kp0_x'].iloc[0] == 3000.0

    def test_empty_frame_yields_empty_objects(self):
        parsed = parse_apt_tracking(_apt_frame(n_frames=0))
        assert set(parsed) == {'rat631', 'rat613'}
        assert all(len(obj) == 0 for obj in parsed.values())

    def test_frame_without_position_columns_is_rejected(self):
        with pytest.raises(ValueError, match='APT/TQT'):
            parse_apt_tracking(pd.DataFrame({'timestamp': [1, 2]}))

    def test_parse_tracking_dispatches_on_the_header(self):
        parsed = parse_tracking(_apt_frame())
        assert set(parsed) == {'rat631', 'rat613'}


class TestLoading:
    def test_load_populates_timestamps_without_a_sidecar(self, tmp_path):
        """The wide format carries its own clock; there is no '*_ts.npy'."""
        tracking = load_tracking_data(_write_apt(tmp_path, n_frames=8))
        assert tracking.timestamps is not None
        assert len(tracking.timestamps) == 8
        assert tracking.timestamps[0] == _T0

    def test_load_timestamps_reads_the_column(self, tmp_path):
        timestamps = load_timestamps(_write_apt(tmp_path, n_frames=8))
        assert len(timestamps) == 8
        assert timestamps[-1] == _T0 + 7 * _FRAME_NS

    def test_timestamps_are_not_truncated_to_float(self, tmp_path):
        """Linux nanoseconds need 61 bits; a float32/int32 cast would corrupt."""
        timestamps = load_timestamps(_write_apt(tmp_path, n_frames=4))
        assert timestamps.dtype == np.int64
        assert int(timestamps[0]) == _T0

    def test_load_ts_false_leaves_timestamps_unset(self, tmp_path):
        tracking = load_tracking_data(_write_apt(tmp_path), load_ts=False)
        assert tracking.timestamps is None

    def test_objects_resolve_by_bare_animal_id(self, tmp_path):
        """`get_object_data('631')` is how the analyses ask for an animal."""
        tracking = load_tracking_data(_write_apt(tmp_path))
        assert tracking.get_object_data('631') is not None
        assert tracking.get_object_data('rat631') is not None


class TestSynchronisation:
    class _Sync:
        """Behaviour seconds -> ephys seconds, offset by a constant."""
        def convert_behavior_to_ephys(self, seconds):
            return np.asarray(seconds, dtype=np.float64) - _T0 / 1e9 + 100.0

    def test_sync_maps_each_animals_own_frames(self, tmp_path):
        path = _write_apt(tmp_path, n_frames=6, missing={'rat4631': (1, 3)})
        tracking = load_tracking_data(path)
        assert tracking.synchronize_with_ephys(self._Sync()) is True
        ephys = tracking.get_object_data('rat631')['ephys_timestamps']
        expected = [100.0 + f * _FRAME_NS / 1e9 for f in (0, 2, 4, 5)]
        assert ephys.tolist() == pytest.approx(expected, abs=1e-6)


class TestReadTrackingCenters:
    def test_wide_file_comes_back_long(self, tmp_path):
        path = _write_apt(tmp_path, n_frames=5, missing={'rat4631': (0,)})
        centers = read_tracking_centers(path)
        assert list(centers.columns) == ['frame', 'object_name',
                                         'center_x', 'center_y']
        counts = centers.groupby('object_name').size().to_dict()
        assert counts == {'rat631': 4, 'rat613': 5}

    def test_long_file_is_passed_through(self, tmp_path):
        centers = read_tracking_centers(_write_mask_metrics(tmp_path))
        assert set(centers['object_name']) == {'rat631', 'rat613'}
        assert 'area' not in centers.columns, 'read more columns than needed'


class TestBackwardsCompatibility:
    def test_long_format_still_parses(self, tmp_path):
        tracking = load_tracking_data(_write_mask_metrics(tmp_path))
        assert set(tracking.get_object_names()) == {'rat631', 'rat613'}
        assert tracking.timestamps is not None, 'sidecar *_ts.npy was not read'

    def test_identifier_columns_are_still_dropped(self, tmp_path):
        parsed = parse_tracking(pd.read_csv(_write_mask_metrics(tmp_path)))
        assert 'object_id' not in parsed['rat631'].columns
        assert 'object_name' not in parsed['rat631'].columns
        assert 'area' in parsed['rat631'].columns

    def test_missing_timestamps_still_raise(self, tmp_path):
        path = tmp_path / 'no_clock.csv'
        pd.DataFrame({'object_name': ['rat631'], 'object_id': [1],
                      'frame': [1]}).to_csv(path, index=False)
        with pytest.raises(FileNotFoundError):
            load_timestamps(path)
