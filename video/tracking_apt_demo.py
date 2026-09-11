"""Demo: load a wide APT/TQT tracking file and plot the trajectories.

Runs against a real file on ``//nearline`` (no synthetic data), prints the
checks that say "this parsed correctly", and writes two figures.

    python -m video.tracking_apt_demo
    python -m video.tracking_apt_demo --session 20251217 --file-index 0
    python -m video.tracking_apt_demo --tracking-file <path>.csv --out-dir ./figs

    # every APT export under the configured tracking roots, each figure
    # written into that export's own directory on the share:
    python -m video.tracking_apt_demo --all --save-beside

What the printed checks are for
-------------------------------
Loading a wide file can go wrong in ways that produce a perfectly plausible
DataFrame, so the demo asserts the things a plot would not reveal:

``format``
    Detected from the header alone. If the byte-order mark were not stripped
    the first column would be ``'﻿timestamp'`` and detection would fall
    back to ``unknown``.
``object names``
    Must be ``rat635``, not the file's ``rat4635`` — otherwise
    ``get_object_data('rat631')`` returns ``None`` and an analysis silently
    loses the animal.
``clock``
    Strictly increasing, ~40 Hz, and the int64 nanoseconds must survive intact
    (a float32 cast would quantise them to ~10-minute steps).
``frame alignment``
    The per-animal frame numbers must still index the session timestamp array
    correctly *after* undetected frames were dropped. This is the one that
    cannot be seen by eye: each animal misses a different number of frames, so
    renumbering would skew each animal's clock by a different amount.

Plots
-----
Both are named after the source file, e.g. ``TQT_named_trajectories.png``.

``<stem>_trajectories.png``
    One panel per animal — identity comes from the facet, so colour is free to
    encode elapsed time on a single-hue sequential ramp. A trajectory that
    reads as a coherent walk through one arena, with the colour sweeping
    smoothly from light to dark, is the visual confirmation that positions and
    timestamps are paired correctly.
``<stem>_checks.png``
    Detection coverage per animal over the recording, and one animal's x
    position over a short window with its gaps left as breaks.

Note on units
-------------
Positions are plotted in **pixels**, deliberately. The APT solutions are
tracked on 4500x2050 video while the older mask-metrics solutions are tracked
on 2148x1064, so the cohort config's single ``pixels_per_cm`` cannot be right
for both. See ``--pixels-per-cm``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Importable both as ``python -m video.tracking_apt_demo`` and as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video.tracking_import import (  # noqa: E402
    TRACKING_FORMAT_APT_TQT,
    load_tracking_data,
)

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_FILE = (
    '//nearline/karpova/TervoLab/analysis/Tracking/APT/cohort7/'
    'cohort7_20251216_1159/solution/TQT_named.csv'
)

# Tokens from the validated reference palette (references/palette.md).
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'
ACCENT = '#2a78d6'          # categorical slot 1, used as a single accent
CRITICAL = '#d03b3b'        # status: reserved, used only for "missing"

#: Sequential blue, light -> dark. Steps 200..700 rather than the full 100..700:
#: a trajectory is a thin mark, and the two lightest steps recede so far into
#: the surface that the first minutes of the recording read as missing.
SEQUENTIAL_BLUE = [
    '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]

#: The same ramp anchored at the surface, for filled cells where "none" should
#: disappear into the background rather than read as a light value.
SEQUENTIAL_BLUE_FROM_SURFACE = [SURFACE] + SEQUENTIAL_BLUE


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def report(tracking) -> dict:
    """Print the load-correctness checks and return the facts they used."""
    names = tracking.get_object_names()
    timestamps = tracking.timestamps
    seconds = timestamps / 1e9
    intervals = np.diff(seconds)
    elapsed = float(seconds[-1] - seconds[0])

    print(f"\nfile        {tracking.tracking_file}")
    print(f"format      {tracking.tracking_format}")
    print(f"objects     {len(names)}  {names}")
    print(f"frames      {len(timestamps):,}")
    print(f"clock       int64 ns = {timestamps.dtype == np.int64}, "
          f"strictly increasing = {bool(np.all(intervals > 0))}")
    rate = (len(timestamps) - 1) / elapsed if elapsed else float('nan')
    print(f"            {elapsed:.1f} s  ({rate:.3f} Hz, "
          f"median interval {np.median(intervals) * 1e3:.2f} ms)")

    # The extra '4' must be gone: the rest of the pipeline says 'rat631'/'631'.
    print("\nname normalisation (APT writes 'rat4631' for rat 631)")
    print(f"  all names look like rat<3 digits>: "
          f"{all(len(n) == 6 and n.startswith('rat') for n in names)}")
    print(f"  get_object_data('631') resolves:   "
          f"{tracking.get_object_data('631') is not None}")

    # Frame alignment. Each animal drops a different number of frames, so this
    # is where a renumbering bug would hide.
    print("\nper-animal detection and frame alignment")
    print(f"  {'animal':<10}{'frames':>9}{'detected':>10}"
          f"{'x range (px)':>18}{'y range (px)':>18}   timestamp check")
    facts = {}
    for name in names:
        obj = tracking.get_object_data(name)
        frames = np.clip(obj['frame'].to_numpy(), 0, len(timestamps) - 1)
        # The row's own timestamp, carried through from the source file, must
        # equal the session timestamp array indexed by that row's frame number.
        # Only the wide format carries one: the long format keeps its clock in
        # a sidecar, so there is nothing independent to check it against.
        if 'timestamp' in obj.columns:
            aligned = bool(np.array_equal(obj['timestamp'].to_numpy(),
                                          timestamps[frames]))
            verdict = 'OK' if aligned else 'MISALIGNED'
        else:
            aligned, verdict = None, 'n/a (no per-row timestamp)'
        x, y = obj['center_x'].to_numpy(), obj['center_y'].to_numpy()
        facts[name] = {'frames': frames, 'x': x, 'y': y, 'aligned': aligned,
                       'seconds': seconds[frames] - seconds[0]}
        print(f"  {name:<10}{len(obj):>9,}{len(obj) / len(timestamps):>9.1%}"
              f"{f'{np.nanmin(x):.0f} - {np.nanmax(x):.0f}':>18}"
              f"{f'{np.nanmin(y):.0f} - {np.nanmax(y):.0f}':>18}"
              f"   {verdict}")

    if any(f['aligned'] is False for f in facts.values()):
        raise AssertionError(
            "frame numbers no longer index the session timestamp array; "
            "undetected frames were renumbered rather than dropped")
    return facts


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)


def plot_trajectories(tracking, facts, out_path, pixels_per_cm=None):
    """Small multiples: one animal per panel, path coloured by elapsed time.

    Nine animals is far past the point where categorical colour can tell series
    apart, so identity is carried by the facet and the panel title. That frees
    colour for the thing actually worth encoding here — when the animal was
    where it was, on a single-hue sequential ramp.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQUENTIAL_BLUE)
    names = list(facts)
    n_cols = 3
    n_rows = int(np.ceil(len(names) / n_cols))

    scale = 1.0 / pixels_per_cm if pixels_per_cm else 1.0
    unit = 'cm' if pixels_per_cm else 'px'

    # One shared extent, so panels are comparable and the arena keeps its shape.
    x_min = min(np.nanmin(f['x']) for f in facts.values()) * scale
    x_max = max(np.nanmax(f['x']) for f in facts.values()) * scale
    y_min = min(np.nanmin(f['y']) for f in facts.values()) * scale
    y_max = max(np.nanmax(f['y']) for f in facts.values()) * scale
    pad_x, pad_y = (x_max - x_min) * 0.02, (y_max - y_min) * 0.02

    total_minutes = max(f['seconds'][-1] for f in facts.values()) / 60.0
    norm = Normalize(0.0, total_minutes)

    # Panels are equal-aspect on a shared extent, so their height is fixed by
    # the arena's shape. Derive the figure height from it rather than guessing,
    # or every row carries a band of dead space.
    left, right = 0.09, 0.98
    wspace, hspace = 0.15, 0.45
    fig_width = 14.5
    # Inches, because what has to fit is text: the suptitle plus a row of panel
    # titles above the grid, and tick labels plus the colourbar below it.
    chrome_top, chrome_bottom = 0.95, 1.05
    panel_w = fig_width * (right - left) / (n_cols + (n_cols - 1) * wspace)
    panel_h = panel_w * (y_max - y_min) / (x_max - x_min)
    fig_height = (panel_h * (n_rows + (n_rows - 1) * hspace)
                  + chrome_top + chrome_bottom)
    top = 1.0 - chrome_top / fig_height
    bottom = chrome_bottom / fig_height

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        f = facts[name]
        x, y, minutes = f['x'] * scale, f['y'] * scale, f['seconds'] / 60.0

        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # A gap in detection must not be drawn as a straight dash across the
        # arena, so segments spanning more than a second are cut.
        keep = np.diff(f['seconds']) < 1.0
        lines = LineCollection(segments[keep], cmap=cmap, norm=norm,
                               linewidth=0.35, alpha=0.75)
        lines.set_array(minutes[:-1][keep])
        ax.add_collection(lines)

        _style(ax)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_max + pad_y, y_min - pad_y)   # image coords: y grows down
        ax.set_aspect('equal')
        ax.set_title(f"{name}    {len(x) / len(tracking.timestamps):.0%} of frames",
                     fontsize=10, color=INK, loc='left', pad=8)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    # Explicit geometry: letting colorbar(ax=[...]) steal space shrinks the
    # whole grid and leaves a band of dead space under the title.
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        hspace=hspace, wspace=wspace)
    cax = fig.add_axes([left, 0.60 / fig_height, right - left,
                        0.16 / fig_height])
    bar = fig.colorbar(lines, cax=cax, orientation='horizontal')
    bar.set_label('elapsed time (minutes)', color=INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    fig.suptitle(
        f"Trajectories — {Path(tracking.tracking_file).name}   "
        f"({len(tracking.timestamps):,} frames, {total_minutes:.0f} min, "
        f"positions in {unit})",
        fontsize=12, color=INK, x=left, ha='left',
        y=1.0 - 0.26 / fig_height)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nwrote {out_path}")


def plot_checks(tracking, facts, out_path, window_seconds=120.0):
    """Detection coverage per animal, and one animal's x position over time."""
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    from matplotlib.colors import LinearSegmentedColormap

    names = list(facts)
    seconds = tracking.timestamps / 1e9
    total_minutes = (seconds[-1] - seconds[0]) / 60.0

    fig, (ax_cov, ax_trace) = plt.subplots(
        2, 1, figsize=(12, 7), facecolor=SURFACE,
        gridspec_kw={'height_ratios': [len(names), 7], 'hspace': 0.42})

    # --- where detection is MISSING -----------------------------------------
    # Encoding the gaps rather than the coverage: the gaps are the signal, and
    # a mostly-empty panel with dark marks reads at a glance where a mostly-dark
    # panel with white slivers does not.
    cmap = LinearSegmentedColormap.from_list('gaps', SEQUENTIAL_BLUE_FROM_SURFACE)
    n_bins = 600
    edges = np.linspace(0, total_minutes, n_bins + 1)
    grid = np.zeros((len(names), n_bins))
    expected = len(tracking.timestamps) / n_bins
    for row, name in enumerate(names):
        counts, _ = np.histogram(facts[name]['seconds'] / 60.0, bins=edges)
        grid[row] = np.clip(1.0 - counts / expected, 0, 1)

    image = ax_cov.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                          extent=[0, total_minutes, len(names) - 0.5, -0.5],
                          interpolation='nearest')
    ax_cov.set_yticks(range(len(names)))
    ax_cov.set_yticklabels(
        [f"{n}  {1 - grid[i].mean():.0%}" for i, n in enumerate(names)], fontsize=8)
    ax_cov.set_xlabel('elapsed time (minutes)', color=INK_SECONDARY, fontsize=9)
    ax_cov.set_title('Undetected frames — darker is a longer gap '
                     '(percentages are frames tracked)',
                     fontsize=10, color=INK, loc='left', pad=8)
    _style(ax_cov)
    ax_cov.tick_params(left=False)
    bar = fig.colorbar(image, ax=ax_cov, fraction=0.02, pad=0.012)
    bar.set_ticks([0, 1])
    bar.set_ticklabels(['tracked', 'absent'])
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)

    # --- one animal's x position, gaps left as breaks -----------------------
    name = names[0]
    f = facts[name]
    in_window = f['seconds'] <= window_seconds
    t, x = f['seconds'][in_window], f['x'][in_window]
    # Break the line wherever a frame is missing, rather than interpolating
    # across it — an invented straight segment is exactly what this plot is
    # meant to rule out.
    x = x.copy().astype(float)
    gaps = np.diff(f['frames'][in_window]) > 1
    x[np.append(gaps, False)] = np.nan

    ax_trace.plot(t, x, color=ACCENT, linewidth=1.2, marker='.',
                  markersize=1.5, markeredgewidth=0)
    missing = t[np.append(gaps, False)]
    if len(missing):
        # Pinned just above the bottom spine in axes fraction, so the rug never
        # drags the y-limits around or lands on top of the trace.
        rug = mtransforms.blended_transform_factory(ax_trace.transData,
                                                    ax_trace.transAxes)
        ax_trace.vlines(missing, 0.01, 0.06, transform=rug,
                        color=CRITICAL, linewidth=1.0)
        ax_trace.text(0.0, -0.30, f'{len(missing)} detection gaps (marked below '
                                  'the trace); the line breaks at each one '
                                  'rather than interpolating across it',
                      transform=ax_trace.transAxes, ha='left',
                      fontsize=8, color=INK_SECONDARY)
    ax_trace.grid(True, color=GRID, linewidth=0.6, axis='y')
    ax_trace.set_axisbelow(True)
    _style(ax_trace)
    ax_trace.set_xlabel('elapsed time (seconds)', color=INK_SECONDARY, fontsize=9)
    ax_trace.set_ylabel('x position (px)', color=INK_SECONDARY, fontsize=9)
    ax_trace.set_title(
        f'{name} — x position over the first {window_seconds:.0f} s',
        fontsize=10, color=INK, loc='left', pad=8)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------

def find_apt_files(config_path=None):
    """Every ``TQT_named.csv`` under the configured tracking roots.

    Goes through the same root list and the same filename/subdirectory
    constants that :func:`ingestion.data_paths.get_tracking_files_by_date`
    uses, so what this demo covers cannot drift from what the pipeline can
    actually resolve.
    """
    from ingestion.data_paths import (
        _DATELESS_TRACKING_FILENAMES,
        _DATELESS_TRACKING_SUBDIRS,
        _load_config,
        _tracking_roots,
    )

    found, seen = [], set()
    for root in _tracking_roots(_load_config(config_path)):
        if not root.exists():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            for sub in _DATELESS_TRACKING_SUBDIRS:
                for filename in _DATELESS_TRACKING_FILENAMES:
                    candidate = child / sub / filename
                    if candidate.is_file() and str(candidate) not in seen:
                        seen.add(str(candidate))
                        found.append(candidate)
    return found


def run_one(path, out_dir, pixels_per_cm=None):
    """Load one tracking file, print its checks, write both figures."""
    tracking = load_tracking_data(path)
    if tracking.tracking_format != TRACKING_FORMAT_APT_TQT:
        print(f"\nnote: this is a '{tracking.tracking_format}' file, not an APT "
              "export. The demo runs either way.")
    if tracking.timestamps is None:
        raise ValueError(f"no frame timestamps in {path}")

    facts = report(tracking)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named after the source file, so a figure sitting beside its CSV on the
    # share is unambiguous about which export produced it.
    stem = Path(path).stem
    plot_trajectories(tracking, facts, out_dir / f'{stem}_trajectories.png',
                      pixels_per_cm=pixels_per_cm)
    plot_checks(tracking, facts, out_dir / f'{stem}_checks.png')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--tracking-file', default=None,
                        help='path to a tracking CSV (default: a cohort-7 APT file)')
    source.add_argument('--session', default=None,
                        help='session id; resolves through DataStorageManager')
    source.add_argument('--all', action='store_true',
                        help='process every TQT_named.csv under the configured '
                             'tracking roots')
    parser.add_argument('--animal', default='631',
                        help='animal id, only used to resolve --session')
    parser.add_argument('--file-index', type=int, default=0,
                        help='which of the session tracking files to load')
    parser.add_argument('--config', default=None,
                        help='cohort config, only used with --all')
    parser.add_argument('--pixels-per-cm', type=float, default=None,
                        help='convert positions to cm. Left off by default: the '
                             'APT solutions are tracked on 4500x2050 video and '
                             'the mask-metrics ones on 2148x1064, so the config '
                             "value is not valid for both — measure it for the "
                             'format you are loading.')
    parser.add_argument('--save-beside', action='store_true',
                        help='write each figure next to its own source file '
                             '(on the share) instead of into --out-dir')
    parser.add_argument('--out-dir', default='.', help='where to write the figures')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

    if args.all:
        paths = find_apt_files(args.config)
        if not paths:
            raise SystemExit('no TQT_named.csv found under the configured '
                             'tracking roots')
        print(f"{len(paths)} APT file(s):")
        for path in paths:
            print(f"  {path}")
    elif args.session:
        from ingestion.data_paths import DataStorageManager
        dsm = DataStorageManager(animal_id=args.animal, session_id=args.session)
        files = dsm.get_tracking_files()
        for i, path in enumerate(files):
            print(f"  [{i}]{' <-' if i == args.file_index else '   '} {path}")
        paths = [files[args.file_index]]
    else:
        paths = [Path(args.tracking_file or DEFAULT_TRACKING_FILE)]

    failures = []
    for path in paths:
        out_dir = Path(path).parent if args.save_beside else Path(args.out_dir)
        print(f"\n{'=' * 78}\n{path}")
        try:
            run_one(path, out_dir, pixels_per_cm=args.pixels_per_cm)
        except Exception as exc:
            # One unreadable export must not abandon the rest of the batch.
            logger.exception("failed on %s", path)
            failures.append((path, exc))

    if failures:
        print(f"\n{len(failures)} of {len(paths)} failed:")
        for path, exc in failures:
            print(f"  {path}: {exc}")
        return 1
    print(f"\ndone: {len(paths)} file(s)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
