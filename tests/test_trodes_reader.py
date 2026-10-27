"""Tests for ingestion.trodes_to_python on synthetic Trodes extracted files."""

import numpy as np
import pytest

from ingestion.trodes_to_python import parse_fields, read_trodes_dat, read_trodes_header


def _write(path, fields, records, extra=()):
    lines = ['<Start settings>', 'Clockrate: 30000',
             'System_time_at_creation: 1765896217452', *extra, f'Fields: {fields}']
    with open(path, 'wb') as f:
        f.write(('\r\n'.join(lines) + '\r\n<End settings>\n').encode('ascii'))
        f.write(records.tobytes())


def test_dio_file_round_trips(tmp_path):
    dtype = np.dtype([('time', '<u4'), ('state', 'u1')])
    records = np.array([(100, 1), (250, 0), (900, 1)], dtype=dtype)
    _write(tmp_path / 'x.dat', '<time uint32><state uint8>', records)

    out = read_trodes_dat(tmp_path / 'x.dat')
    assert out.header['system_time_at_creation'] == '1765896217452'
    assert out.data.dtype == dtype
    assert out.data['time'].tolist() == [100, 250, 900]
    assert out.data['state'].tolist() == [1, 0, 1]


def test_mmap_matches_read(tmp_path):
    records = np.arange(5000, 6000, dtype='<u4').view([('time', '<u4')])
    _write(tmp_path / 't.timestamps.dat', '<time uint32>', records)
    read = read_trodes_dat(tmp_path / 't.timestamps.dat')
    mapped = read_trodes_dat(tmp_path / 't.timestamps.dat', mmap=True)
    assert np.array_equal(read.data['time'], mapped.data['time'])


def test_header_offset_points_at_payload(tmp_path):
    records = np.array([7], dtype='<u4').view([('time', '<u4')])
    _write(tmp_path / 'x.dat', '<time uint32>', records)
    _, offset = read_trodes_header(tmp_path / 'x.dat')
    raw = (tmp_path / 'x.dat').read_bytes()
    assert raw[offset:] == records.tobytes()


@pytest.mark.parametrize('fields, expected', [
    ('<time uint32><state uint8>', [('time', '<u4'), ('state', 'u1')]),
    ('<data 4*int16>', [('data', '<i2', (4,))]),
    ('<data int16*4><t uint32>', [('data', '<i2', (4,)), ('t', '<u4')]),
])
def test_parse_fields(fields, expected):
    assert parse_fields(fields) == np.dtype(expected)


def test_bad_files_raise_value_error(tmp_path):
    (tmp_path / 'nohdr.dat').write_bytes(b'garbage\n')
    with pytest.raises(ValueError, match='Start settings'):
        read_trodes_dat(tmp_path / 'nohdr.dat')
    with pytest.raises(ValueError, match='Unknown Trodes field type'):
        parse_fields('<time notatype>')
