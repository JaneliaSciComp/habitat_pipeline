"""Precompute the session browser's index (and optionally per-day tracking grids).

    python scripts/build_session_index.py --cohort cohort7
    python scripts/build_session_index.py --cohort cohort7 --cohort cohort5 --timelines
    python scripts/build_session_index.py --cohort cohort7 --timelines --dates 20251216

The index itself is cheap (seconds: directory listings, two values per video
``_ts.npy``, one events CSV per scored date) and the GUI builds it on demand.
``--timelines`` is the slow part: per-animal tracking coverage reads every
tracking CSV of a day over SMB. Days whose tracking files are unchanged since
their grid was cached are skipped, so a rerun only pays for new data.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.session_index import (  # noqa: E402
    COHORT_CONFIGS,
    cached_day_tracking_grid,
    day_sync,
    get_day_tracking_grid,
    load_session_index,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", action="append", choices=sorted(COHORT_CONFIGS),
                        help="cohort(s) to index (default: all)")
    parser.add_argument("--refresh", action="store_true",
                        help="rescan the share even if the cached index is valid")
    parser.add_argument("--timelines", action="store_true",
                        help="also build per-animal tracking coverage for each day")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="restrict --timelines to these YYYYMMDD dates")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    for cohort in args.cohort or sorted(COHORT_CONFIGS):
        t0 = time.time()
        index = load_session_index(cohort, refresh=args.refresh,
                                   progress=lambda m: print(f"  {m}", flush=True))
        print(f"{cohort}: {len(index['table'])} recordings, {len(index['days'])} days "
              f"({time.time() - t0:.1f} s)")
        if not args.timelines:
            continue

        for date, day in sorted(index["days"].items()):
            if args.dates and date not in args.dates:
                continue
            if not day.get("tracking_files"):
                continue
            if cached_day_tracking_grid(cohort, day) is not None:
                print(f"  {date}: tracking grid cached")
                continue
            sync = day_sync(index, date)
            if sync is None:
                print(f"  {date}: no sync in manifest, skipped")
                continue
            required = sorted({a["animal"] for rid in day["recordings"]
                               for a in index["details"][rid]["animals"]})
            t1 = time.time()
            try:
                grid = get_day_tracking_grid(cohort, day, sync, required)
            except Exception as exc:
                print(f"  {date}: FAILED {type(exc).__name__}: {exc}")
                continue
            print(f"  {date}: {len(day['tracking_files'])} file(s), "
                  f"{len(grid['animals'])} rows ({time.time() - t1:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
