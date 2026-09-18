"""Per-day timeline: when each recorded animal had ephys, and when it was
video-tracked.

Companion to :func:`video.tracking_apt_demo.plot_checks` — same imshow
encoding and reference palette (inverted here: blue is *present*, white is
*absent*, so a mostly-white row reads as "mostly missing" at a glance),
applied across the animals of one recording *day* instead of across the
objects inside one tracking file.

    python -m video.tracking_ephys_timeline --date 20251216
    python -m video.tracking_ephys_timeline --all --cohort cohort7
    python -m video.tracking_ephys_timeline --all --cohort cohort5 \
        --config config/cohort5_paths.json

Two panels per day
-------------------
``ephys availability``
    Per animal, the measured ``[first spike, last spike]`` window(s) from
    :mod:`discovery.capability_manifest` (``ephys.per_animal.<animal>.ephys_window``,
    per :func:`discovery.manifest_build.probe_ephys`). A day with several
    recording blocks contributes one window per block; a gap between blocks,
    or a block an animal is missing from entirely, is real and shown as a gap
    rather than smoothed over — see the "session's duration" gotcha in
    CLAUDE.md. This half is cheap: it is a JSON read, not a kilosort reload.

``tracking coverage``
    Per animal, per time bin, the fraction of expected frames actually
    detected. Computed **fresh** from every tracking file
    ``DataStorageManager`` resolves for the date (multi-root: the older
    mask-metrics export and every APT/TQT chunk), *not* from
    ``discovery/capability_manifest.json`` — that artifact's tracking fields
    predate multi-file date resolution and are stale for exactly this (see
    CLAUDE.md, "capability_manifest.json is stale for tracking"). Bins outside
    every tracking file's own time span are masked (drawn in the neutral grid
    colour) rather than counted as a gap, since "no camera was rolling" and "a
    camera was rolling and missed this animal" are different failures and the
    unmasked version conflates them.

    Rows are every ``rat<id>`` object any tracking file names that day, not
    just the animals with ephys — tracking is scored against a wider arena
    census, and an animal with no ephys that day can still be in frame. The
    ephys panel's rows stay ephys-only, since that is the set an ephys window
    can exist for at all; the two row lists can therefore differ.

Both panels share one x-axis: time of day (local, from the day's ephys clock
mapped back through its sync), since every recording block of a day shares one
ephys clock (CLAUDE.md: "All animals in a session share one ephys clock").
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery.capability_manifest import load_manifest  # noqa: E402
from ingestion.data_paths import DataStorageManager  # noqa: E402
from ingestion.ephys_sync import DataSyncManager  # noqa: E402
from video.tracking_import import (  # noqa: E402
    load_timestamps,
    normalize_object_name,
    read_tracking_centers,
)
from video.tracking_apt_demo import (  # noqa: E402
    ACCENT,
    AXIS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_BLUE_FROM_SURFACE,
    SURFACE,
    _style,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = r'\\nearline\karpova\TervoLab\analysis\Tracking\APT\cohort7'

DEFAULT_COHORTS = {
    'cohort7': 'config/default_paths.json',
    'cohort5': 'config/cohort5_paths.json',
}

#: A tracked object counts as a rat only if its canonicalized name is exactly
#: 'rat<digits>' — matches discovery.manifest_build's own identity check, so
#: an unresolved tracklet id never shows up as a phantom "animal" row.
_RAT_ID_RE = re.compile(r'^rat\d+$', re.IGNORECASE)


def _canonical_rat_id(raw_name: Any) -> str:
    """Object name -> ``'rat<id>'``, or unchanged if it isn't a rat at all.

    Two on-disk quirks land on the same animal id: APT's ``'rat4635'``
    (handled by :func:`normalize_object_name`) and this cohort's older
    mask-metrics export, which names objects with the bare number
    (``'613'``, not ``'rat613'`` — confirmed on session 20251210, where every
    object name is a bare digit string). A bare-digit name gets the same
    ``'rat'`` prefix used everywhere else in this pipeline
    (``video/behavioral_visualization.py``'s ``rat_id`` helper does the same
    thing); anything else (an unresolved tracklet id, an arena landmark) is
    left alone so it still fails :data:`_RAT_ID_RE` and is excluded.
    """
    name = normalize_object_name(raw_name)
    return f"rat{name}" if name.isdigit() else name


# ---------------------------------------------------------------------------
# Gathering one day's facts
# ---------------------------------------------------------------------------

def day_records(manifest: Mapping[str, Any], cohort: str,
                session_date: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Every manifest session (one per recording block) for this cohort+date."""
    return sorted(
        (sid, record) for sid, record in manifest.get('sessions', {}).items()
        if record.get('cohort') == cohort and record.get('session_date') == session_date
    )


def cohort_dates(manifest: Mapping[str, Any], cohort: str) -> List[str]:
    dates = {record.get('session_date') for record in manifest.get('sessions', {}).values()
             if record.get('cohort') == cohort and record.get('session_date')}
    return sorted(dates)


def ephys_intervals(records: Sequence[Tuple[str, Dict[str, Any]]]
                    ) -> Dict[str, List[Tuple[float, float]]]:
    """``{animal: [(t_first, t_last), ...]}`` on the day's shared ephys clock.

    One interval per recording block the animal appears in with a measured
    window; a block where ``ephys_window`` is ``None`` (load failed, etc.)
    contributes nothing for that animal rather than a fabricated zero.
    """
    intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for _, record in records:
        for animal, block in (record.get('ephys', {}).get('per_animal') or {}).items():
            window = block.get('ephys_window')
            if window and window[1] > window[0]:
                intervals[animal].append((float(window[0]), float(window[1])))
    for animal in intervals:
        intervals[animal].sort()
    return dict(intervals)


def primary_record(records: Sequence[Tuple[str, Dict[str, Any]]]
                   ) -> Optional[Tuple[str, Dict[str, Any]]]:
    for sid, record in records:
        if (record.get('recording') or {}).get('is_primary'):
            return sid, record
    return records[0] if records else None


def build_day_sync(record: Mapping[str, Any], config_path: Optional[str],
                   dio_channel: int) -> Tuple[Optional[Any], Optional[Any]]:
    """A ``(DataStorageManager, DataSyncManager)`` for this day, or ``(None, None)``.

    Any animal of the primary block works — the sync is per recording, not
    per animal (CLAUDE.md: "any animal's DIO + pulse log works because they
    all share the clock").

    Built with ``use_cache=False``: ``DataStorageManager``'s on-disk path
    cache has no staleness check (only a version bump invalidates it), so a
    cached entry silently keeps whatever tracking files existed the day it was
    written. Confirmed on 20251216: the cache for ``rat613_20251216_094334``
    listed 3 tracking files while the share now has 7 APT chunks for that
    date. A tool whose entire point is to surface tracking gaps must not read
    a path list that can go stale in exactly that direction.
    """
    animals = sorted((record.get('ephys') or {}).get('animals') or [])
    recording_id = (record.get('recording') or {}).get('recording_id') \
        or record.get('session_id')
    last_exc = None
    for animal in animals:
        try:
            dsm = DataStorageManager(animal, recording_id, config_path=config_path,
                                     use_cache=False)
            sync = DataSyncManager(dsm, dio_channel=dio_channel)
            return dsm, sync
        except Exception as exc:
            last_exc = exc
            continue
    if animals:
        logger.warning("could not build a day sync for %s: %s", recording_id, last_exc)
    return None, None


def hours_of_day(ephys_seconds: np.ndarray, sync) -> np.ndarray:
    """Ephys-clock seconds -> local wall-clock hour-of-day (fractional).

    ``sync.convert_ephys_to_behavior`` gives Unix epoch seconds;
    :func:`datetime.fromtimestamp` (no explicit tz) reads that in the
    machine's local zone, which is what this pipeline already does elsewhere
    for display (``gui/runners.py``) and matches directory names such as
    ``20251216_094334`` exactly.
    """
    epoch_seconds = np.asarray(sync.convert_ephys_to_behavior(np.asarray(ephys_seconds)))
    local = [datetime.fromtimestamp(t) for t in epoch_seconds]
    return np.array([dt.hour + dt.minute / 60.0 + dt.second / 3600.0 for dt in local])


# ---------------------------------------------------------------------------
# Coverage grids
# ---------------------------------------------------------------------------

def ephys_grid(intervals: Mapping[str, List[Tuple[float, float]]],
              animals: Sequence[str], edges: np.ndarray) -> np.ndarray:
    """Per animal, per bin: fraction of the bin an ephys window covers."""
    widths = edges[1:] - edges[:-1]
    grid = np.zeros((len(animals), len(widths)))
    for row, animal in enumerate(animals):
        covered = np.zeros(len(widths))
        for start, end in intervals.get(animal, []):
            overlap = np.clip(np.minimum(edges[1:], end) - np.maximum(edges[:-1], start),
                              0, None)
            covered += overlap
        grid[row] = np.clip(covered / widths, 0, 1)
    return grid


def tracking_grid(dsm, sync, edges: np.ndarray, required_animals: Sequence[str] = ()
                  ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Per animal, per bin: fraction of expected frames detected.

    Rows are discovered from the data, not assumed: every normalized
    ``rat<id>`` object *any* tracking file for the date names, unioned with
    ``required_animals`` (so an ephys animal that a tracking file never
    detects still gets a row, rather than silently disappearing). Also
    returns which bins any tracking file actually spans — the rest are
    masked, not zeroed, by the caller.
    """
    n_bins = len(edges) - 1
    counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_bins))
    expected = np.zeros(n_bins)

    try:
        files = dsm.get_tracking_files() or []
    except Exception as exc:
        logger.warning("no tracking files resolved: %s", exc)
        files = []

    for path in files:
        try:
            timestamps = load_timestamps(path).astype(np.float64)
            if timestamps.max() > 1e12:      # Linux nanoseconds
                timestamps = timestamps / 1e9
            day_seconds = np.asarray(sync.convert_behavior_to_ephys(timestamps))
            span = day_seconds[-1] - day_seconds[0]
            if len(day_seconds) < 2 or span <= 0:
                continue
            fps = (len(day_seconds) - 1) / span
            centers = read_tracking_centers(path)
        except Exception as exc:
            logger.warning("skipping tracking file %s: %s", path, exc)
            continue
        if 'object_name' not in centers.columns or 'frame' not in centers.columns:
            continue

        # This file's own span, projected onto the day grid: bins it does not
        # reach stay unmasked-false, so "no camera" is distinguishable from a
        # real gap during a run.
        lo, hi = day_seconds[0], day_seconds[-1]
        overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
        expected += overlap * fps

        names = centers['object_name'].map(_canonical_rat_id).to_numpy()
        frame_idx = np.clip(centers['frame'].to_numpy(), 0, len(day_seconds) - 1)
        centers_seconds = day_seconds[frame_idx]
        for name in np.unique(names):
            if not _RAT_ID_RE.match(str(name)):
                continue   # an unresolved tracklet id, not a rat
            mask = names == name
            hist, _ = np.histogram(centers_seconds[mask], bins=edges)
            counts[str(name)] += hist

    animals = sorted(set(required_animals) | set(counts))
    covered = expected > 1e-6
    grid = np.full((len(animals), n_bins), np.nan)
    safe_expected = np.where(covered, expected, 1.0)
    for row, animal in enumerate(animals):
        fraction = np.where(covered, counts[animal] / safe_expected, np.nan)
        grid[row] = np.clip(fraction, 0, 1)
    return animals, grid, covered


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_day_timeline(session_date: str, cohort: str, ephys_animals: Sequence[str],
                      track_animals: Sequence[str], ephys: np.ndarray, tracking: np.ndarray,
                      covered: np.ndarray, x_edges: np.ndarray, is_time_of_day: bool,
                      out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    # Blue is present, white is absent: a mostly-white row reads as
    # "mostly missing" at a glance, the opposite sense of plot_checks' "darker
    # is a gap" (there the grid is gap-fraction; here it is presence directly).
    cmap = LinearSegmentedColormap.from_list('presence', SEQUENTIAL_BLUE_FROM_SURFACE)
    cmap.set_bad(GRID)   # bins no tracking file reaches: neutral, not "absent"

    total_hours = float(x_edges[-1] - x_edges[0])
    hhmm = FuncFormatter(lambda h, pos: f"{int(h % 24):02d}:{round((h % 1) * 60):02d}")

    # Row counts differ between panels (tracking spans a wider arena census
    # than ephys), so each panel's height follows its own animal count.
    fig_height = 0.55 * (len(ephys_animals) + len(track_animals)) + 2.5
    fig, (ax_ephys, ax_track) = plt.subplots(
        2, 1, figsize=(13, fig_height), facecolor=SURFACE,
        gridspec_kw={'height_ratios': [max(len(ephys_animals), 1),
                                       max(len(track_animals), 1)], 'hspace': 0.5})

    img1 = ax_ephys.imshow(
        ephys, aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[x_edges[0], x_edges[-1], len(ephys_animals) - 0.5, -0.5],
        interpolation='nearest')
    ax_ephys.set_yticks(range(len(ephys_animals)))
    ax_ephys.set_yticklabels(
        [f"{a}  {ephys[i].mean():.0%}" for i, a in enumerate(ephys_animals)], fontsize=8)
    ax_ephys.set_title('Ephys availability — blue is recorded spikes, white is '
                       'no ephys (percentages are hours with ephys)',
                       fontsize=10, color=INK, loc='left', pad=8)
    _style(ax_ephys)
    ax_ephys.tick_params(left=False)
    bar1 = fig.colorbar(img1, ax=ax_ephys, fraction=0.02, pad=0.012)
    bar1.set_ticks([0, 1])
    bar1.set_ticklabels(['absent', 'recorded'])
    bar1.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar1.outline.set_visible(False)

    track_values = np.ma.masked_invalid(tracking)
    img2 = ax_track.imshow(
        track_values, aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[x_edges[0], x_edges[-1], len(track_animals) - 0.5, -0.5],
        interpolation='nearest')
    track_pct = np.nanmean(tracking, axis=1)
    ax_track.set_yticks(range(len(track_animals)))
    ax_track.set_yticklabels(
        [f"{a}  {'n/a' if np.isnan(p) else f'{p:.0%}'}"
         for a, p in zip(track_animals, track_pct)],
        fontsize=8)
    ax_track.set_xlabel('time of day (local)' if is_time_of_day
                        else 'elapsed time (hours) — sync unavailable, no time of day',
                        color=INK_SECONDARY, fontsize=9)
    ax_track.set_title('Tracking coverage — blue is detected while a camera was '
                       'rolling, white is undetected; grid colour is no tracking '
                       'file at all (percentages are of frames a file could see)',
                       fontsize=10, color=INK, loc='left', pad=8)
    _style(ax_track)
    ax_track.tick_params(left=False)
    bar2 = fig.colorbar(img2, ax=ax_track, fraction=0.02, pad=0.012)
    bar2.set_ticks([0, 1])
    bar2.set_ticklabels(['absent', 'tracked'])
    bar2.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar2.outline.set_visible(False)

    if is_time_of_day:
        ax_ephys.xaxis.set_major_formatter(hhmm)
        ax_track.xaxis.set_major_formatter(hhmm)

    any_tracking = bool(covered.any())
    if not any_tracking:
        ax_track.text(0.5, 0.5, 'no tracking file resolved for this date',
                      transform=ax_track.transAxes, ha='center', va='center',
                      fontsize=9, color=INK_MUTED)

    fig.suptitle(f"{cohort}  {session_date}   ephys + tracking coverage, "
                f"{len(ephys_animals)} ephys / {len(track_animals)} tracked animal(s), "
                f"{total_hours:.1f} h",
                fontsize=12, color=INK, x=0.09, ha='left', y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------

def run_one_day(manifest: Mapping[str, Any], cohort: str, session_date: str,
                config_path: Optional[str], out_dir: Path, dio_channel: int = 1,
                bin_seconds: float = 60.0) -> None:
    records = day_records(manifest, cohort, session_date)
    if not records:
        raise ValueError(f"no manifest sessions for {cohort} {session_date}")

    intervals = ephys_intervals(records)
    animals = sorted(intervals)
    if not animals:
        raise ValueError(f"no animal has a measured ephys window on {session_date}")

    total_seconds = max(end for ivs in intervals.values() for _, end in ivs)
    n_bins = max(1, int(np.ceil(total_seconds / bin_seconds)))
    edges = np.arange(n_bins + 1, dtype=np.float64) * bin_seconds

    primary_sid, primary = primary_record(records)
    dsm, sync = build_day_sync(primary, config_path, dio_channel)

    e_grid = ephys_grid(intervals, animals, edges)
    if dsm is not None and sync is not None:
        track_animals, t_grid, covered = tracking_grid(
            dsm, sync, edges, required_animals=animals)
        x_edges = hours_of_day(edges, sync)
        is_time_of_day = True
    else:
        track_animals = animals
        t_grid = np.full((len(animals), n_bins), np.nan)
        covered = np.zeros(n_bins, dtype=bool)
        x_edges = edges / 3600.0
        is_time_of_day = False

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{cohort}_{session_date}_coverage_timeline.png'
    plot_day_timeline(session_date, cohort, animals, track_animals, e_grid, t_grid,
                      covered, x_edges, is_time_of_day, out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--date', default=None,
                        help='an 8-digit session date, e.g. 20251216')
    parser.add_argument('--all', action='store_true',
                        help='process every date for --cohort in the manifest')
    parser.add_argument('--cohort', default='cohort7', choices=sorted(DEFAULT_COHORTS),
                        help='which cohort config to resolve tracking/sync against')
    parser.add_argument('--config', default=None,
                        help='cohort config path (default: the standard one for --cohort)')
    parser.add_argument('--manifest', default=None,
                        help='capability manifest path (default: discovery/capability_manifest.json)')
    parser.add_argument('--dio-channel', type=int, default=1)
    parser.add_argument('--bin-seconds', type=float, default=60.0,
                        help='time bin width for both panels')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                        help='where to write the figures')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    config_path = args.config or DEFAULT_COHORTS[args.cohort]
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)

    if args.all:
        dates = cohort_dates(manifest, args.cohort)
        if not dates:
            raise SystemExit(f"no sessions for cohort {args.cohort!r} in the manifest")
    elif args.date:
        dates = [args.date]
    else:
        raise SystemExit('pass --date YYYYMMDD or --all')

    out_dir = Path(args.out_dir)
    failures = []
    for session_date in dates:
        print(f"\n{'=' * 78}\n{args.cohort} {session_date}")
        try:
            run_one_day(manifest, args.cohort, session_date, config_path, out_dir,
                       dio_channel=args.dio_channel, bin_seconds=args.bin_seconds)
        except Exception as exc:
            logger.exception("failed on %s %s", args.cohort, session_date)
            failures.append((session_date, exc))

    if failures:
        print(f"\n{len(failures)} of {len(dates)} failed:")
        for session_date, exc in failures:
            print(f"  {session_date}: {exc}")
        return 1
    print(f"\ndone: {len(dates)} date(s)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
