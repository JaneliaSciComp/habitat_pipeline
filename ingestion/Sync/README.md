# greyware_time.py

Convert Trodes hardware sample counts from a `.rec` file to Greyware-disciplined PC time.

The method was calibrated on cohort 7 and applied to cohort 9 merge day (2026-09-29). It does not use the Trodes header start time or the timekeeper pulse log.

Expected accuracy: about 5 to 15 ms. It is not shown to reach 1 ms. See [Accuracy and limits](#accuracy-and-limits).

## Why not the header start time

The header fields `systemTimeAtCreation` and `timestampAtCreation` are written when the file is opened, while recording is still paused. The first is the PC clock read in the GUI thread, the second is the last hardware count the acquisition thread had received. The two reads are not simultaneous. On cohort 7 the header start was 143 to 448 ms late. On the three cohort 9 merge-day recordings it implies a time 528, 1367 and 661 ms later than the fit below.

## How it works

Every packet in the recording carries:

- a **hardware sample count** (uint32, +1 per sample, so true time is exactly linear in it), and
- a **SysClock stamp** (uint64, nanoseconds since the Unix epoch) from the acquisition PC, whose clock is disciplined by Greyware.

The stamps are applied after the packets arrive, so each one is late and noisy. A single stamp is not used. Instead:

1. Read the header and work out the packet layout from the device list.
2. Read blocks of consecutive packets, evenly spaced over the whole recording.
3. Fit a straight line of SysClock stamp against hardware count (least squares). The slope is the true sample rate (about 29,999.78 Hz, which is 7 ppm below nominal). The intercept is the PC time at a reference count.
4. Subtract the latency constant measured on cohort 7.

```
t_Greyware(n) = T_ref + (n - n_ref) / R - latency
```

- `n` is the hardware sample count (the same value as the timestamps in the DIO export and in the `.rec` packets).
- `n_ref`, `T_ref` (Unix seconds) and `R` (Hz) come from the fit. They are specific to each recording.
- `latency` defaults to **0.1185 s**: the mean stamp lateness over seven cohort 7 days (per-day range 114.4 to 125.5 ms, SD 4.0 ms).

## Usage

Fit a recording and convert counts:

```
python greyware_time.py 20260929_095542.rec --counts 261075 270000000 --tz America/New_York
```

Fit once, save, and reuse without the `.rec`:

```
python greyware_time.py 20260929_095542.rec --save-fit fit_20260929_095542.json
python greyware_time.py --fit fit_20260929_095542.json --counts 261075 270000000
```

From Python:

```python
from greyware_time import fit_recording, greyware_time, to_datetime

fit = fit_recording("20260929_095542.rec")        # reads about 12 M packets by default
t_unix = greyware_time(fit, [261075, 270000000])  # numpy array, Unix seconds UTC
to_datetime(t_unix, "America/New_York")
```

Options: `--latency-ms` (default 118.5), `--blocks` (default 1200), `--block-packets` (default 10000), `--tz`.

The script needs numpy and a recording that contains the `SysClock` device. It stops with an error if the device is missing or the hardware counts do not step by 1.

## Running it for a file and a sample

You need two things: the recording (`.rec`) the data came from, and the hardware sample count of the sample you care about.

**1. Choose the recording.** Sample counts restart in every recording, so the same count appears in more than one file. Use the recording the data came from. Its name is the session date and start time (for example `20260929_095542`). If you are unsure, run the script and read the count range it prints. A count outside that range triggers a warning.

**2. Get the hardware count of your sample.**

- From a DIO export (`*.dio_Controller_Din1.dat`): each record already holds the count, so use it as it is.
- From a sample index in an exported file (for example a Kilosort `.dat`): convert the index to a count with the export's timestamp file first. See [Converting a sample index to a hardware count](#converting-a-sample-index-to-a-hardware-count).

**3. Run the script.**

```
python greyware_time.py 20260929_095542.rec --counts 270000000
```

The first run reads about 264 MB for a 22-byte session file and takes about 75 seconds over the network share. It prints the fit and then the time of each count:

```
20260929_095542.rec: counts 261075 to 541369551  T_ref=1790690149.515685  R=29999.77321 Hz
count 270000000  ->  1790699140.7627  2026-09-29 12:25:40.762656
```

- The first line gives the recording's count range, the reference time and the fitted rate. `T_ref` in this line is before the latency constant, which is subtracted in the conversion.
- The number after the arrow is the Greyware time in Unix seconds (UTC). The last column is the same time in local time. The default zone is America/New_York, and `--tz` changes it.
- Give several counts at once with `--counts 261075 270000000 541369551`.

**4. Reuse the fit.** Save it once, and later runs take a second and do not need the `.rec` or the network share:

```
python greyware_time.py 20260929_095542.rec --save-fit fit_20260929_095542.json
python greyware_time.py --fit fit_20260929_095542.json --counts 270000000
```

**5. Many events at once.** For example every rising edge in a DIO file:

```python
import numpy as np
from greyware_time import greyware_time, to_datetime
import json

fit = json.load(open("fit_20260929_095542.json"))
raw = open("20260929_095542_merged.dio_Controller_Din1.dat", "rb").read()
i = raw.index(b"<End settings>\n") + len(b"<End settings>\n")
rec = np.frombuffer(raw[i:i + ((len(raw) - i) // 5) * 5], dtype=np.uint8).reshape(-1, 5)
counts = np.ascontiguousarray(rec[:, :4]).view("<u4").ravel()      # hardware counts
state = rec[:, 4]
rising = counts[1:][(state[1:] == 1) & (state[:-1] == 0)]
t_unix = greyware_time(fit, rising)                                 # Unix seconds, UTC
```

For a numpy file of spike or event counts, use `greyware_time(fit, np.load("event_counts.npy"))`.

**6. Check the result.**

- The times should fall between the recording's start and end (see the merge day table below).
- The fitted rate should be close to 29,999.78 Hz.
- An error about a missing `SysClock` device, or about counts that do not step by 1, means the file is not one this method applies to.
- To use a different latency constant, add `--latency-ms 114.4`. The default is 118.5.

## Which file to use

| File | Packet size | Notes |
|---|---|---|
| Session-level `.rec` (cohort 9) | 22 bytes | DIO, Multiplexed, SysClock, timestamp. About 12 GB per 5 h. **Preferred.** |
| Per-rat `*_merged.rec` | 856 bytes (cohort 7, 384 channels) | Same stamps, about 40 times more data to read |

Packet layout (bytes from the start of each packet):

| Field | Bytes | Cohort 7 merged | Cohort 9 session |
|---|---|---|---|
| Sync (0x55) | 1 | 0 | 0 |
| Controller_DIO | 1 | 1 | 1 |
| Multiplexed | 8 | 2-9 | 2-9 |
| neuropixels_LFP | 66 | 10-75 | not present |
| **SysClock** (uint64 ns) | 8 | **76-83** | **10-17** |
| **Hardware timestamp** (uint32) | 4 | **84-87** | **18-21** |
| Channel data (int16) | 2 x channels | 88-855 | none |

The script computes these offsets from the header, so nothing is hard-coded. The SysClock and timestamp offsets were confirmed by decoding the files. The other offsets follow from the device order and sizes.

## Sample size

The line is fitted from sampled blocks, not every packet. These figures are the variation in the fitted time caused by sampling alone (worst of the first, middle and last sample of the recording), measured on `20260929_095542` by drawing random subsets from a pool of 2,400 blocks of 10,000 packets.

| Blocks x packets per block | Packets read | SD of fitted time |
|---|---|---|
| 60 x 3,000 | 0.18 M | 6.2 ms |
| 240 x 3,000 | 0.72 M | 3.2 ms |
| 480 x 3,000 | 1.4 M | 2.5 ms |
| 960 x 3,000 | 2.9 M | 1.9 ms |
| 1,200 x 10,000 (default) | 12 M | 1.4 ms |

Recommendations:

- Use **at least 1,000 blocks of 3,000 packets** (about 2 ms of sampling jitter).
- More, shorter blocks beat fewer, longer ones. 1,200 blocks of 1,000 packets reaches about 2.0 ms using 10 times fewer packets than 1,200 blocks of 10,000.
- Going beyond about 1,000 blocks buys little. Sampling jitter is already below the roughly 4 ms uncertainty of the latency constant.
- For the 856-byte merged files the default would read about 10 GB. Use `--blocks 1000 --block-packets 3000` (about 2.6 GB), or the session-level file.

## Timestamps for spikes and events

An event is timed by evaluating the equation at the event's hardware count. How many samples the event spans does not change the precision.

Do **not** average the SysClock stamps of the event's own packets. They arrive in batches and are correlated over roughly 100 ms, so averaging gains nothing for events shorter than that. The precision comes from the line fitted over the whole recording.

## Converting a sample index to a hardware count

The equation takes the hardware count, not the index of a sample in an exported file (for example a Kilosort `.dat`). Packets can be missing from a `.rec` while the counter keeps running (22,810 were missing in the 12/10 cohort 7 recording), so the index is not the count minus the first count. Use the timestamp file written by the Trodes export, or read the count from the packet. This has not been checked for each export type.

## Merge day coefficients (2026-09-29)

The 118.5 ms latency constant is already subtracted in `T_ref` below. Each recording has its own rate; use its own value.

| Recording | n_ref | T_ref (Unix s) | R (Hz) | Start (EDT) | End (EDT) |
|---|---|---|---|---|---|
| 20260929_095542 | 261075 | 1790690149.3972 | 29999.7732 | 09:55:49.397 | 14:56:26.482 |
| 20260929_150139 | 165676 | 1790708503.1745 | 29999.7887 | 15:01:43.174 | 20:02:51.100 |
| 20260929_201052 | 141083 | 1790727054.6164 | 29999.7659 | 20:10:54.616 | 07:53:32.009 (9/30) |

```python
REC = {"20260929_095542": (261075, 1790690149.397185, 29999.77321),
       "20260929_150139": (165676, 1790708503.174497, 29999.788746),
       "20260929_201052": (141083, 1790727054.616411, 29999.765906)}
def greyware_time(session, n):
    n_ref, t_ref, rate = REC[session]
    return t_ref + (n - n_ref) / rate          # Unix seconds, UTC
```

These values are also in `mergeday_equation.json`. Running `greyware_time.py` on `20260929_095542.rec` reproduces the first row (count 261075 maps to 09:55:49.397 EDT, rate 29999.77321 Hz). Note that `T_ref` in a saved fit file does not include the latency constant, which `greyware_time()` subtracts, so the table values differ from a fit file's `t_ref_unix_s` by 0.1185 s.

## Accuracy and limits

| Source | Size |
|---|---|
| Sampling jitter of the fit (1,200 blocks x 10,000 packets) | about 1.4 ms |
| Latency constant, SD across 7 cohort 7 days | about 4 ms (days ranged 114.4 to 125.5 ms) |
| Consistency between consecutive merge-day recordings | 14.1 and 10.7 ms |

Expected accuracy is **5 to 15 ms**. The method is not shown to reach 1 ms.

- **The constant is carried over from cohort 7.** Cohort 9 uses the same controller (serial 00106 00189, firmware 3.20) but Trodes 2.5.3, against 2.6.1 in cohort 7. The release notes list no change to SysClock or the acquisition thread, but the code was not compared. The cohort 9 latency itself could not be measured, because the cohort 9 timekeeper cannot serve as a time reference.
- **Implied merge-day lateness.** Assuming one feature of the stamp distribution sits at the same lateness as in cohort 7, merge-day mean lateness comes out at 114.4 ms (earliest stamps), 121.1 ms (median) or 127.1 ms (latest stamps). The earliest-stamp assumption predicted cohort 7 days best (hold-out error SD 2.5 ms), so 114.4 ms is the best-supported value, about 4 ms below the default. Pass `--latency-ms 114.4` to use it.
- **Two checks are not reconciled.** The stamp-shape method says the three merge-day recordings differ in lateness by 1 to 3.5 ms. A consistency check against the timekeeper drift says 11 to 14 ms. The 5 to 15 ms range keeps both in view.
- **Stamp rate.** The fitted sample rate is 29,999.766 to 29,999.789 Hz on merge day, in line with 29,999.786 Hz from the cohort 7 regression.
- **Only the three merge-day recordings** have been fit with dense sampling. Other cohort 9 sessions have not.
- **Within-session drift.** If the stamp lateness drifts during a session, the line absorbs it. On two of the seven cohort 7 days this gave a line error of 4.5 and 7.6 ms; on the other five it was 0.2 to 1.3 ms.

## Files

- `greyware_time.py`: the script.
- `fit_20260929_095542.json`: saved fit for the first merge-day recording.
- `mergeday_equation.json`: coefficients for all three merge-day recordings.
- `cohort9_sample_timing.html`: the full report (published as a private page).
