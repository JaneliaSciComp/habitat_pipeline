#!/usr/bin/env python
"""
Build the capability manifest (Layer 0) from the data on ``//nearline``.

Must run where the share is mounted. Two probe levels:

``--probe-level paths`` (minutes)
    Path resolution only: which sessions and animals exist, and whether their
    directories are there. Fast, catches broken configuration, and tells you
    the shape of the dataset. Recorded in the artifact as ``probe_level:
    'paths'``, which makes :func:`discovery.capability_manifest.manifest_status`
    report ``partial`` so nothing mistakes it for a full inventory.

``--probe-level full`` (hours over ~47 animal/session pairs)
    Adds cluster and quality-cell counts, sync validity, per-object tracking
    statistics and coverage, and the real per-``behavior_type`` usable label
    sets. This is the level that makes a hypothesis fail at generation time.

Sessions are merged into the artifact one at a time via an atomic replace, so a
multi-hour run is resumable and a crash costs only the session in flight.

Examples::

    python scripts/build_capability_manifest.py --probe-level paths
    python scripts/build_capability_manifest.py --probe-level full --sessions 20251216
    python scripts/build_capability_manifest.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_COHORTS = (
    {'name': 'cohort7', 'config_path': 'config/default_paths.json'},
    {'name': 'cohort5', 'config_path': 'config/cohort5_paths.json'},
)


class Reporter:
    """PASS/PARTIAL/FAIL accumulator, matching scripts/phase0_probe.py's style."""

    def __init__(self):
        self.rows = []

    def add(self, verdict: str, label: str, detail: str = '') -> None:
        self.rows.append((verdict, label, detail))
        print(f"  [{verdict:7s}] {label}" + (f" - {detail}" if detail else ''),
              flush=True)

    def summary(self) -> int:
        counts = {}
        for verdict, _, _ in self.rows:
            counts[verdict] = counts.get(verdict, 0) + 1
        print('\n' + '=' * 72)
        print('  ' + '  '.join(f"{verdict}={count}" for verdict, count
                                in sorted(counts.items())))
        print('=' * 72)
        return 1 if counts.get('FAIL') else 0


def cmd_check(args) -> int:
    """Local-only staleness report; never touches the share."""
    from discovery.capability_manifest import (
        ManifestError, load_manifest, manifest_status, verify_sources,
    )

    path = Path(args.out)
    try:
        status = manifest_status(path, repo_root=REPO_ROOT)
    except ManifestError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(status.summary())

    manifest = load_manifest(path)
    if args.verify_sources:
        print("\nverifying recorded sources (this touches the share):")
        stale = 0
        for session_id in sorted(manifest.get('sessions', {})):
            result = verify_sources(session_id, path)
            if not result.ok:
                stale += 1
                print(f"  [STALE] {session_id}: "
                      f"{len(result.changed)} changed, {len(result.missing)} missing")
        print(f"\n{stale} session(s) need a rebuild" if stale
              else "\nall recorded sources unchanged")
    return 0


def cmd_build(args) -> int:
    from discovery.manifest_build import (
        build_session_record, enumerate_targets, merge_session_record,
        atomic_write_manifest, new_manifest,
    )
    from discovery.capability_manifest import load_manifest

    out_path = Path(args.out)
    cohorts = [c for c in DEFAULT_COHORTS
               if not args.cohort or c['name'] in args.cohort]

    # Always resume from an existing artifact unless --rebuild is given.
    # --force re-probes the *selected* sessions; it must not discard the rest,
    # which an earlier version did - one `--force --sessions X` silently dropped
    # 33 already-probed sessions.
    manifest = None
    if out_path.exists() and not args.rebuild:
        try:
            manifest = dict(load_manifest(out_path, use_cache=False))
            print(f"resuming from {out_path} "
                  f"({len(manifest.get('sessions', {}))} session(s) already present)")
        except Exception as exc:
            print(f"could not reuse {out_path} ({exc}); starting fresh")
    if manifest is None:
        manifest = new_manifest(cohorts, probe_level=args.probe_level,
                                argv=' '.join(sys.argv[1:]), repo_root=REPO_ROOT)
    else:
        manifest['generated_by']['probe_level'] = args.probe_level
        manifest['generated_by']['argv'] = ' '.join(sys.argv[1:])

    reporter = Reporter()
    started = time.time()
    n_built = 0

    for cohort in cohorts:
        config_path = str(REPO_ROOT / cohort['config_path'])
        if not Path(config_path).exists():
            reporter.add('FAIL', f"{cohort['name']}: config missing", config_path)
            continue
        print(f"\n=== {cohort['name']} ({cohort['config_path']}) ===", flush=True)
        try:
            targets = enumerate_targets(config_path)
        except Exception as exc:
            reporter.add('FAIL', f"{cohort['name']}: enumeration failed",
                         f"{type(exc).__name__}: {exc}")
            continue
        print(f"  {len(targets)} session(s) discovered", flush=True)

        for session_id in sorted(targets):
            target = targets[session_id]
            if args.sessions and not any(
                    s in session_id or s == target.get('session_date')
                    for s in args.sessions):
                continue
            if session_id in manifest['sessions'] and not args.force:
                existing = manifest['sessions'][session_id]
                if (existing.get('provenance') or {}).get('probe_level') == args.probe_level:
                    reporter.add('SKIP', session_id, 'already probed at this level')
                    continue

            animals = target['animals']
            if args.animals:
                animals = [a for a in animals if any(x in a for x in args.animals)]
            try:
                record = build_session_record(
                    session_id, animals, cohort=cohort['name'],
                    config_path=config_path, probe_level=args.probe_level,
                    dio_channel=args.dio_channel)
            except Exception as exc:
                reporter.add('FAIL', session_id, f"{type(exc).__name__}: {exc}")
                continue

            merge_session_record(manifest, record)
            # Written after every session so a multi-hour pass is resumable and
            # a crash costs only the session in flight.
            atomic_write_manifest(manifest, out_path)
            n_built += 1

            errors = (record.get('provenance') or {}).get('errors', [])
            detail = (f"{len(animals)} animal(s), "
                      f"{record['provenance']['probe_seconds']}s")
            if not animals:
                reporter.add('PARTIAL', session_id, 'no animals with ephys')
            elif errors:
                reporter.add('PARTIAL', session_id,
                             f"{detail}, {len(errors)} probe error(s)")
            else:
                reporter.add('PASS', session_id, detail)

    from datetime import datetime, timezone
    manifest['generated_at'] = datetime.now(timezone.utc).isoformat(
        timespec='seconds').replace('+00:00', 'Z')
    atomic_write_manifest(manifest, out_path)

    elapsed = time.time() - started
    print(f"\nwrote {out_path} - {len(manifest['sessions'])} session(s) total, "
          f"{n_built} built this run, {elapsed:.1f}s")
    if manifest['build_errors']:
        print(f"{len(manifest['build_errors'])} recorded build error(s); "
              "see build_errors in the artifact")
    if args.probe_level == 'paths':
        print("\nNOTE: probe_level='paths'. Content facts (cell counts, event class "
              "counts, tracking coverage) are absent, and manifest_status() will "
              "report 'partial' until a full build runs.")
    return reporter.summary()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='build_capability_manifest.py',
        description=__doc__.strip().splitlines()[1],
        epilog='Run with --probe-level paths first; it is fast and catches broken '
               'configuration before committing to a multi-hour full pass.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',
                        default=str(REPO_ROOT / 'discovery' / 'capability_manifest.json'))
    parser.add_argument('--probe-level', choices=('paths', 'full'), default='paths')
    parser.add_argument('--cohort', action='append', default=[],
                        help='restrict to a cohort name (repeatable)')
    parser.add_argument('--sessions', nargs='*', default=[],
                        help='restrict to session ids or dates')
    parser.add_argument('--animals', nargs='*', default=[])
    parser.add_argument('--dio-channel', type=int, default=1)
    parser.add_argument('--force', action='store_true',
                        help='re-probe the selected sessions even if already present '
                             'at this probe level; other sessions are kept')
    parser.add_argument('--rebuild', action='store_true',
                        help='discard the existing artifact and start from scratch '
                             '(this DOES drop already-probed sessions)')
    parser.add_argument('--check', action='store_true',
                        help='report staleness and exit; builds nothing')
    parser.add_argument('--verify-sources', action='store_true',
                        help='with --check, stat recorded source files (touches the share)')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return cmd_check(args)
    return cmd_build(args)


if __name__ == '__main__':
    raise SystemExit(main())
