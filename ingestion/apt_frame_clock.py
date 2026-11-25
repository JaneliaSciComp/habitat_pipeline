#!/usr/bin/env python3
"""
Frame clocks for APT movies that have no ``<movie>_ts.npy`` on the share.

The 2026-11 APT export (``solution_keypoints.csv``) carries frame indices but
no timestamps, and the APT video's real per-frame timestamps are on the share
for only six movies (``social_videos4APT``). FERAL's ``feral_events.csv``
covers all 322 movies, and its bout start/end times are those same APT frame
timestamps truncated to the millisecond. Joined to the movie-browser JSON
(which gives each bout's frames, see :func:`ingestion.feral_import.load_feral_bouts`),
every bout boundary is an exact ``(frame, time)`` pair. Interpolating between
them (:class:`ingestion.feral_import.FrameClock`) rebuilds the clock.

Measured against the six real files (``--check``): median 0.5 ms, p99 <= 28 ms,
worst 125 ms (in the movie with 23 dropped frames). Against the ``timestamp``
column of the 17 archived ``TQT_named.csv`` files without a real ``ts.npy``
(6 days): median <= 1 ms, p99 <= 87 ms, worst 182 ms. A nominal 25 ms grid is
off by up to 575 ms on the same movies, and the raw-video
``social_videos/RatCity_*_40Hz_ts.npy`` has fewer frames than the APT video,
so indexing it by APT frame is ~200 ms off.

Output: ``.cache/apt_frame_ts/<movie>_ts.npy`` (int64 Linux ns, frames
``0..71999``, or further if FERAL's window is longer) plus ``manifest.json``.
:func:`video.tracking_import.load_apt_frame_timestamps` reads these only when
no real ``ts.npy`` exists, and reports ``timestamp_source='feral_anchor'``.

Usage::

    python -m ingestion.apt_frame_clock --build          # ~1 min, needs the share
    python -m ingestion.apt_frame_clock --check          # clock vs real ts files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.data_paths import (  # noqa: E402
    APT_FRAME_TS_DIR, _tracking_roots, apt_chunk_tracking_file, load_config,
)
from ingestion.feral_import import FrameClock, default_dirs, load_feral_bouts  # noqa: E402

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

#: Frames in a 30-minute chunk at 40 fps; APT files hold 71,970-71,998.
FULL_CHUNK_FRAMES = 72000


def apt_movie_files(config: dict) -> Dict[str, Path]:
    """``{movie: APT tracking file}`` for every chunk directory under the tracking roots."""
    out: Dict[str, Path] = {}
    for root in _tracking_roots(config):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if child.is_dir():
                f = apt_chunk_tracking_file(child)
                if f is not None:
                    out.setdefault(child.name, f)
    return out


def real_timestamp_file(tracking_file: Path, config: dict) -> Optional[Path]:
    """The movie's real APT ``ts.npy`` (lookup sources 1-2), or None."""
    from video.tracking_import import apt_frame_timestamp_candidates
    for path, source in apt_frame_timestamp_candidates(tracking_file, config):
        if source == "apt_ts_npy" and path.is_file():
            return path
    return None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clock_inputs(feral_csv: Path, browser_dir: Path):
    bouts, _, windows, _ = load_feral_bouts(feral_csv, browser_dir)
    return FrameClock(bouts), bouts, windows


def build_frame_clocks(feral_csv, browser_dir, out_dir=APT_FRAME_TS_DIR, *,
                       movies: Optional[Iterable[str]] = None,
                       config: Optional[dict] = None,
                       skip_real: bool = True) -> dict:
    """Write one FERAL-anchored ``<movie>_ts.npy`` per movie; return the manifest.

    *movies* defaults to every APT movie found under *config*'s tracking roots
    (or, with no APT movies found, every movie FERAL covers). Movies with a real
    ``ts.npy`` are skipped when *skip_real* is set, because the loader prefers
    the real file anyway.
    """
    feral_csv, browser_dir, out_dir = Path(feral_csv), Path(browser_dir), Path(out_dir)
    config = config if config is not None else load_config()
    clock, bouts, windows = _clock_inputs(feral_csv, browser_dir)
    apt_files = apt_movie_files(config) if config.get("tracking") else {}
    wanted = list(movies) if movies is not None else (sorted(apt_files) or sorted(clock._anchors))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "feral_anchor", "feral_csv": str(feral_csv),
                "feral_csv_md5": _md5(feral_csv), "browser_dir": str(browser_dir),
                "movies": {}, "skipped_real_ts": [], "no_feral_anchors": []}
    for movie in wanted:
        if skip_real and movie in apt_files and real_timestamp_file(apt_files[movie], config):
            manifest["skipped_real_ts"].append(movie)
            continue
        if movie not in clock._anchors:
            manifest["no_feral_anchors"].append(movie)
            logger.warning("No FERAL bouts for %s; no clock written", movie)
            continue
        # At least a full 30-minute chunk: FERAL's window can end a few frames
        # before the APT file's last frame, and frames past the last anchor
        # extrapolate at 40 fps.
        n_frames = max(windows.get(movie, (0, 0))[1] + 1, FULL_CHUNK_FRAMES)
        ts = clock(movie, np.arange(n_frames, dtype=np.int64))
        np.save(out_dir / f"{movie}_ts.npy", ts.astype(np.int64))
        fx = clock._anchors[movie][0]
        manifest["movies"][movie] = {
            "n_frames": int(n_frames),
            "n_anchors": int(len(fx)),
            "first_anchor_frame": int(fx[0]),
            "last_anchor_frame": int(fx[-1]),
            "max_anchor_gap_frames": int(np.diff(fx).max()) if len(fx) > 1 else None,
        }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d frame clocks to %s (%d skipped: real ts; %d without anchors)",
                len(manifest["movies"]), out_dir, len(manifest["skipped_real_ts"]),
                len(manifest["no_feral_anchors"]))
    return manifest


def check_frame_clocks(feral_csv, browser_dir, *, config: Optional[dict] = None) -> dict:
    """Error of the FERAL clock against every real APT ``ts.npy`` (ms)."""
    config = config if config is not None else load_config()
    clock, _, _ = _clock_inputs(Path(feral_csv), Path(browser_dir))
    out = {}
    for movie, f in apt_movie_files(config).items():
        real = real_timestamp_file(f, config)
        if real is None or movie not in clock._anchors:
            continue
        truth = np.load(real).astype(np.int64)
        err = np.abs(clock(movie, np.arange(len(truth), dtype=np.int64)) - truth) / 1e6
        out[movie] = {"n_frames": int(len(truth)), "median_ms": float(np.median(err)),
                      "p99_ms": float(np.percentile(err, 99)), "max_ms": float(err.max()),
                      "dropped_frames": int(np.round(np.diff(truth) / 25e6).sum() - (len(truth) - 1))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="write the frame clocks")
    mode.add_argument("--check", action="store_true", help="compare with real ts files")
    p.add_argument("--config", default=None, help="cohort config (default cohort 7)")
    p.add_argument("--feral_csv", default=None)
    p.add_argument("--browser_dir", default=None)
    p.add_argument("--out_dir", default=str(APT_FRAME_TS_DIR))
    p.add_argument("--movies", nargs="+", default=None)
    p.add_argument("--include_real", action="store_true",
                   help="also write clocks for movies that have a real ts.npy")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_config(args.config)
    feral_dir, browser_dir = default_dirs(args.config)
    feral_csv = Path(args.feral_csv) if args.feral_csv else feral_dir / "feral_events.csv"
    browser_dir = Path(args.browser_dir) if args.browser_dir else browser_dir
    if args.build:
        res = build_frame_clocks(feral_csv, browser_dir, args.out_dir, movies=args.movies,
                                 config=config, skip_real=not args.include_real)
        res = {k: (len(v) if isinstance(v, (dict, list)) else v) for k, v in res.items()}
    else:
        res = check_frame_clocks(feral_csv, browser_dir, config=config)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
