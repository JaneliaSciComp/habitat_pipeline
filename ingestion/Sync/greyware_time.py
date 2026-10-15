#!/usr/bin/env python3
"""
Convert Trodes hardware sample counts to Greyware-disciplined PC time.

Method (see cohort9_sample_timing report):
  1. Every packet in a .rec file carries a hardware sample count (uint32, +1 per
     sample) and a SysClock stamp (uint64, PC clock in ns since the Unix epoch).
  2. A single stamp is late and noisy, so a straight line is fitted through the
     stamps of many packets spread over the whole recording.
  3. The line gives PC time for any sample count. The latency constant measured
     on cohort 7 is subtracted.

Usage:
    python greyware_time.py recording.rec --counts 261075 5000000 --tz America/New_York
    python greyware_time.py recording.rec --save-fit fit.json          # fit once, reuse
    python greyware_time.py --fit fit.json --counts 261075 5000000     # no .rec needed

In Python:
    fit = fit_recording("20260929_095542.rec")
    t_unix = greyware_time(fit, counts)               # numpy array, Unix seconds UTC
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np

LATENCY_S = 0.1185      # mean stamp lateness on 7 cohort 7 days (SD 4.0 ms across days)
N_BLOCKS = 1200         # blocks of consecutive packets, spread evenly over the recording
BLOCK_PACKETS = 10000   # packets per block (0.33 s at 30 kHz)


def read_layout(path):
    """Header size, packet size and field offsets, from the XML header."""
    with open(path, "rb") as f:
        raw = f.read(1_000_000)
    end = raw.index(b"</Configuration>") + len(b"</Configuration>")
    header_size = raw.index(b"\n", end) + 1
    h = raw[:header_size].decode("latin1")
    n_channels = int(re.search(r'numChannels="(\d+)"', h).group(1))
    devices = []
    for m in re.finditer(r"<Device ([^>]*)>", h):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if a.get("available") == "1":
            devices.append((int(a["packetOrderPreference"]), a["name"], int(a["numBytes"])))
    devices.sort()
    offset, offsets = 1, {}                  # byte 0 is the 0x55 sync byte
    for _, name, nbytes in devices:
        offsets[name] = offset
        offset += nbytes
    if "SysClock" not in offsets:
        raise ValueError("this recording has no SysClock device; cannot recover PC time")
    ts_offset = offset                        # hardware timestamp follows the device fields
    packet_size = ts_offset + 4 + 2 * n_channels
    m = re.search(r'systemTimeAtCreation="\s*(\d+)"', h)
    return dict(header_size=header_size, packet_size=packet_size, sysclock_offset=offsets["SysClock"],
                ts_offset=ts_offset, n_channels=n_channels,
                header_system_time_ms=int(m.group(1)) if m else None)


def sample_stamps(path, layout, n_blocks=N_BLOCKS, block_packets=BLOCK_PACKETS):
    """Read n_blocks blocks of consecutive packets spread evenly over the file."""
    P, H = layout["packet_size"], layout["header_size"]
    size = os.path.getsize(path)
    n_packets = (size - H) // P
    if (size - H) % P:
        print(f"warning: {(size - H) % P} trailing bytes after the last whole packet", file=sys.stderr)
    starts = np.linspace(0, n_packets - block_packets - 1, n_blocks).astype(np.int64)
    counts, stamps = [], []
    with open(path, "rb") as f:
        for i in starts:
            f.seek(H + int(i) * P)
            a = np.frombuffer(f.read(block_packets * P), dtype=np.uint8).reshape(block_packets, P)
            counts.append(np.ascontiguousarray(a[:, layout["ts_offset"]:layout["ts_offset"] + 4]).view("<u4").ravel().astype(np.int64))
            stamps.append(np.ascontiguousarray(a[:, layout["sysclock_offset"]:layout["sysclock_offset"] + 8]).view("<u8").ravel().astype(np.int64))
    if not (np.diff(counts[0]) == 1).all():
        raise ValueError("hardware counts do not step by 1; packet layout was not recognised")
    c = np.concatenate(counts)
    for k in np.flatnonzero(np.diff(c) < -(2 ** 31)):      # uint32 wrap-around (after about 39.8 h)
        c[k + 1:] += 2 ** 32
    return c, np.concatenate(stamps), n_packets


def fit_recording(path, n_blocks=N_BLOCKS, block_packets=BLOCK_PACKETS):
    """Fit PC time (ns) against hardware count. Returns a dict that greyware_time() accepts."""
    layout = read_layout(path)
    counts, stamps, n_packets = sample_stamps(path, layout, n_blocks, block_packets)
    n_ref, s_ref = int(counts[0]), int(stamps[0])
    x = (counts - n_ref).astype(np.float64)
    y = (stamps - s_ref).astype(np.float64)
    ns_per_sample, intercept = np.polyfit(x, y, 1)
    return dict(file=os.path.basename(path), n_ref=n_ref, t_ref_unix_s=(s_ref + intercept) / 1e9,
                rate_hz=1e9 / ns_per_sample, n_blocks=n_blocks, block_packets=block_packets,
                n_packets=int(n_packets), last_count=int(counts[-1]), header_system_time_ms=layout["header_system_time_ms"])


def greyware_time(fit, counts, latency_s=LATENCY_S):
    """Greyware time (Unix seconds, UTC) of hardware sample counts, latency constant subtracted."""
    counts = np.asarray(counts, dtype=np.float64)
    return fit["t_ref_unix_s"] + (counts - fit["n_ref"]) / fit["rate_hz"] - latency_s


def to_datetime(t_unix, tz="America/New_York"):
    from zoneinfo import ZoneInfo
    return [datetime.datetime.fromtimestamp(float(t), ZoneInfo(tz)) for t in np.atleast_1d(t_unix)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", nargs="?", help="Trodes .rec file")
    ap.add_argument("--fit", help="load a saved fit (json) instead of reading the .rec")
    ap.add_argument("--save-fit", help="write the fit to this json file")
    ap.add_argument("--counts", type=int, nargs="*", default=[], help="hardware sample counts to convert")
    ap.add_argument("--latency-ms", type=float, default=LATENCY_S * 1000, help="latency constant to subtract (default 118.5)")
    ap.add_argument("--blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--block-packets", type=int, default=BLOCK_PACKETS)
    ap.add_argument("--tz", default="America/New_York")
    a = ap.parse_args()
    if a.fit:
        fit = json.load(open(a.fit))
    elif a.rec:
        fit = fit_recording(a.rec, a.blocks, a.block_packets)
    else:
        ap.error("give a .rec file or --fit")
    print(f"{fit['file']}: counts {fit['n_ref']} to {fit['last_count']}  T_ref={fit['t_ref_unix_s']:.6f}  R={fit['rate_hz']:.5f} Hz")
    for n in a.counts:
        if not fit["n_ref"] <= n <= fit["last_count"]:
            print(f"warning: count {n} is outside this recording's range "
                  f"({fit['n_ref']} to {fit['last_count']}); wrong file, or an extrapolated value", file=sys.stderr)
    if a.save_fit:
        json.dump(fit, open(a.save_fit, "w"), indent=1)
    t = greyware_time(fit, a.counts, a.latency_ms / 1000)
    for n, tt, dt in zip(a.counts, t, to_datetime(t, a.tz) if len(a.counts) else []):
        print(f"count {n}  ->  {tt:.4f}  {dt:%Y-%m-%d %H:%M:%S.%f}")


if __name__ == "__main__":
    main()
