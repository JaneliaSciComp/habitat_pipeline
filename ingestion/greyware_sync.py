"""
Greyware (SysClock-fit) ephys↔behavior sync
============================================

Cohort 9's timekeeper pulse log cannot serve as a time reference, so the
pulse-matching sync in :mod:`ingestion.ephys_sync` is not trusted there.
Instead every Trodes packet's SysClock stamp (Greyware-disciplined PC time,
Unix ns) is fitted against its hardware sample count, which gives Unix time
for any count (see ``ingestion/Sync/README.md``)::

    t_unix(n) = T_ref + (n - n_ref) / R - latency

The pipeline's ephys clock is already ``count / 30000`` (spike times are
``timestamps.dat[idx] / 30000``; DIO times are ``time / 30000``), so this is
exactly the linear ephys→behavior map that ``DataSyncManager`` exposes::

    slope     = 30000 / R
    intercept = T_ref - n_ref / R - latency

Fits come from ``ingestion/Sync/greyware_fits.json`` (merge day 2026-09-29
only, for now). ``DataSyncManager`` selects this path when the cohort config
sets ``"sync_method": "greyware"``; pulse matching then runs only as a
diagnostic (:func:`pulse_residuals`).
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FITS_PATH = Path(__file__).parent / "Sync" / "greyware_fits.json"
DEFAULT_LATENCY_S = 0.1185
#: How far from the Greyware-predicted span the diagnostic pulse search looks (s).
#: Cohort 9's timekeeper sits ~11350 s (3.15 h) behind Greyware time on merge
#: day, so the margin must exceed that for the match to see the true pulses.
PULSE_SEARCH_MARGIN_S = 6 * 3600.0


def load_greyware_fit(recording_id: str, fits_path: Optional[Path] = None) -> dict:
    """Return the saved SysClock fit for *recording_id* (``YYYYmmdd_HHMMSS``).

    The dict follows ``greyware_time.fit_recording``'s convention:
    ``t_ref_unix_s`` excludes the latency constant.
    """
    with open(fits_path or FITS_PATH) as fh:
        fits = json.load(fh)
    if recording_id not in fits:
        known = sorted(k for k in fits if not k.startswith("_"))
        raise KeyError(
            f"No Greyware fit for recording {recording_id!r}; fitted: {known}. "
            "Fit its session .rec with ingestion/Sync/greyware_time.py --save-fit "
            "and add it to ingestion/Sync/greyware_fits.json (see ingestion/Sync/README.md)."
        )
    return fits[recording_id]


def fit_to_linear(fit: dict, latency_s: float = DEFAULT_LATENCY_S,
                  sample_rate: float = 30000.0):
    """Express a SysClock fit as ``(slope, intercept)`` on ephys seconds.

    ``behavior_s = slope * ephys_s + intercept`` with ``ephys_s = count / sample_rate``.
    """
    rate = float(fit["rate_hz"])
    slope = sample_rate / rate
    intercept = float(fit["t_ref_unix_s"]) - float(fit["n_ref"]) / rate - latency_s
    return slope, intercept


def windowed_pulse_match(slope: float, intercept: float, ephys_sync, behavior_sync,
                         margin_s: float = PULSE_SEARCH_MARGIN_S) -> dict:
    """Interval-match DIO edges to pulses, searching only near the Greyware prediction.

    ``find_sync_mapping`` alone centres its search on the DIO header time and,
    finding nothing there, scans the whole pulse log (cohort 9's spans weeks:
    ~165k pulses, hours of search). Restricting the pulses to the recording's
    Greyware-mapped span ± *margin_s* keeps it to seconds.
    """
    from ingestion.ephys_sync import find_sync_mapping

    tse = np.asarray(ephys_sync, dtype=float)
    tsb = np.asarray(behavior_sync, dtype=float)
    if len(tse) < 2:
        raise ValueError("fewer than 2 DIO sync edges")
    lo, hi = slope * tse[0] + intercept, slope * tse[-1] + intercept
    near = tsb[(tsb >= lo - margin_s) & (tsb <= hi + margin_s)]
    return find_sync_mapping(tse, near, lo, search_window=2 * margin_s)


def pulse_residuals(slope: float, intercept: float, pulse_mapping: dict) -> dict:
    """Compare the Greyware map to a pulse-match result from ``find_sync_mapping``.

    Offset = Greyware-mapped DIO edge − matched timekeeper pulse (s). This is
    a diagnostic only: the timekeeper is the reference cohort 9 cannot trust.
    On merge day it reads ~11350 s behind Greyware and runs ~46 ppm slow, so a
    large, slowly growing offset is the *expected* result there.
    """
    e = np.asarray(pulse_mapping["matched_ephys"], dtype=float)
    b = np.asarray(pulse_mapping["matched_behavior"], dtype=float)
    offset = slope * e + intercept - b
    out = {
        "ok": True,
        "n_matches": int(len(e)),
        "median_offset_s": float(np.median(offset)),
        "max_abs_offset_s": float(np.max(np.abs(offset))),
        "pulse_slope": float(pulse_mapping["slope"]),
        "error": None,
    }
    logger.info(
        "Pulse cross-check: Greyware - timekeeper = %.3f s (median, %d pulses, range %.3f s); "
        "pulse slope %.9f vs Greyware %.9f. Using Greyware.",
        out["median_offset_s"], out["n_matches"], float(np.ptp(offset)), out["pulse_slope"], slope)
    return out
