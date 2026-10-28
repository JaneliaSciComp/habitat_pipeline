"""
Reader for SpikeGadgets Trodes "extracted" binary files (``*.dat``).

Every file Trodes exports — DIO channels, ``*.timestamps.dat``, LFP — has the
same layout: an ASCII header between ``<Start settings>`` and
``<End settings>`` made of ``Key: value`` lines, followed by packed binary
records. The header's ``Fields`` line declares the record layout, e.g.
``<time uint32><state uint8>`` (a field may repeat: ``<data 4*int16>``), so
the header has to be parsed before the payload can be read at all.
"""

import re
from pathlib import Path
from typing import Dict, NamedTuple, Tuple, Union

import numpy as np

HEADER_START = '<Start settings>'
HEADER_END = '<End settings>'


class TrodesFile(NamedTuple):
    """A parsed Trodes file: header (keys lower-cased) and record array."""
    header: Dict[str, str]
    data: np.ndarray


def read_trodes_header(path: Union[str, Path]) -> Tuple[Dict[str, str], int]:
    """Return ``(header, payload_offset)`` for a Trodes extracted file.

    Keys are lower-cased (``'system_time_at_creation'``, ``'fields'``, ...);
    values are the raw strings. ``payload_offset`` is the byte position where
    the binary records start.
    """
    header: Dict[str, str] = {}
    with open(path, 'rb') as f:
        if f.readline().decode('ascii').strip() != HEADER_START:
            raise ValueError(f"{path} does not start with '{HEADER_START}'")
        for raw in f:
            line = raw.decode('ascii').strip()
            if line == HEADER_END:
                return header, f.tell()
            key, sep, value = line.partition(': ')
            if sep:
                header[key.lower()] = value
    raise ValueError(f"{path} has no '{HEADER_END}' line")


def parse_fields(fields: str) -> np.dtype:
    """``'<time uint32><state uint8>'`` -> ``np.dtype([('time', uint32), ('state', uint8)])``.

    A repeated field is written ``<name N*type>`` or ``<name type*N>`` and
    becomes a sub-array of length N.
    """
    tokens = re.sub(r'[<>]', ' ', fields).split()
    if len(tokens) % 2:
        raise ValueError(f"Cannot parse Trodes fields string {fields!r}")

    layout = []
    for name, spec in zip(tokens[::2], tokens[1::2]):
        repeats = 1
        if '*' in spec:
            a, b = spec.split('*')
            spec, repeats = (b, int(a)) if a.isdigit() else (a, int(b))
        try:
            ftype = np.dtype(spec)
        except TypeError:
            raise ValueError(f"Unknown Trodes field type {spec!r} in {fields!r}") from None
        layout.append((name, ftype, repeats) if repeats > 1 else (name, ftype))
    return np.dtype(layout)


def read_trodes_dat(path: Union[str, Path], mmap: bool = False) -> TrodesFile:
    """Read a Trodes extracted file.

    With ``mmap=True`` the records are memory-mapped rather than read, which
    is what you want for a multi-GB ``*.timestamps.dat`` when only some
    entries are needed.
    """
    header, offset = read_trodes_header(path)
    if 'fields' not in header:
        raise ValueError(f"{path} header has no 'Fields' line")
    dtype = parse_fields(header['fields'])
    if mmap:
        data = np.memmap(path, dtype=dtype, mode='r', offset=offset)
    else:
        with open(path, 'rb') as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype)
    return TrodesFile(header, data)
