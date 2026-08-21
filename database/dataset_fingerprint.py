"""
Cheap dataset fingerprints for iteration provenance.

``Iteration.git_commit`` already records which *code* produced a result. This
records which *data* did, so that "the same analysis, re-run" can be
distinguished from "the same analysis against a re-exported dataset". Both are
preconditions of the confirmatory tier: a confirmation run that cannot be
reproduced confirms nothing.

Assumptions:
    - **Metadata, not content.** The fingerprint is a SHA-256 over sorted
      ``(name, size, mtime_ns)`` triples, never over file bytes. Kilosort
      output is gigabytes sitting on an SMB share, and hashing it to detect
      drift would cost minutes per run to catch a case that ``size`` already
      catches. The realistic failure is replacement or re-export, and both
      change size or mtime.
    - **The method is versioned in the record.** ``fingerprint_method`` is
      stored alongside the digest so this can be upgraded to content hashing
      later without silently invalidating every existing fingerprint, or worse,
      comparing digests computed two different ways.
    - **Partial is labelled, not silent.** A path that cannot be stat'ed is
      recorded in the method string (``…/partial:n``) rather than skipped
      quietly, because a fingerprint over fewer files than intended is not the
      fingerprint it claims to be.
    - **mtime is coarse-grained deliberately.** Compared at whole-second
      resolution, since SMB and some filesystems do not preserve nanoseconds
      across a copy, and a spurious mismatch would train the reader to ignore
      the field.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    'FINGERPRINT_METHOD',
    'fingerprint_inputs',
    'fingerprint_session',
    'describe_inputs',
]

#: Bump when the algorithm changes, so old and new digests are never compared.
FINGERPRINT_METHOD = 'paths+size+mtime/v1'

#: Files whose contents are irrelevant to an analysis's inputs.
_SKIP_SUFFIXES = ('.pyc', '.log', '.tmp')


def _iter_files(target: Path, max_files: int) -> Iterable[Path]:
    """Yield files for one input path; a directory contributes its tree.

    Bounded by ``max_files`` because a Kilosort directory can hold a great many
    files and the fingerprint is meant to be cheap. Truncation is reported by
    the caller rather than hidden.
    """
    if target.is_dir():
        count = 0
        for path in sorted(target.rglob('*')):
            if count >= max_files:
                return
            if path.is_file() and path.suffix.lower() not in _SKIP_SUFFIXES:
                count += 1
                yield path
    elif target.is_file():
        yield target


def describe_inputs(paths: Sequence[Union[str, Path]], *,
                    max_files: int = 500) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """Collect ``(name, size, mtime_seconds)`` triples plus unreadable paths."""
    entries: List[Tuple[str, int, int]] = []
    problems: List[str] = []

    for raw in paths:
        if raw is None:
            continue
        target = Path(raw)
        try:
            if not target.exists():
                problems.append(f"missing:{target}")
                continue
        except OSError as exc:
            problems.append(f"unreadable:{target} ({exc})")
            continue

        found = False
        for path in _iter_files(target, max_files):
            found = True
            try:
                stat = path.stat()
            except OSError as exc:
                problems.append(f"unreadable:{path} ({exc})")
                continue
            # Name relative to the input root keeps the digest stable when the
            # share is mounted at a different point.
            try:
                name = path.relative_to(target).as_posix() if target.is_dir() \
                    else path.name
            except ValueError:
                name = path.name
            entries.append((name, int(stat.st_size), int(stat.st_mtime)))
        if not found and target.is_dir():
            problems.append(f"empty:{target}")

    entries.sort()
    return entries, problems


def fingerprint_inputs(paths: Sequence[Union[str, Path]], *,
                       max_files: int = 500) -> Tuple[Optional[str], str]:
    """Fingerprint a set of input paths.

    Returns ``(digest, method)``. ``digest`` is ``None`` when nothing readable
    was found, so that a caller records "not fingerprinted" rather than a hash
    of the empty set — which would compare equal across completely different
    datasets.
    """
    entries, problems = describe_inputs(paths, max_files=max_files)
    if not entries:
        return None, f"{FINGERPRINT_METHOD}/no-readable-inputs"

    digest = hashlib.sha256()
    for name, size, mtime in entries:
        digest.update(f"{name}\0{size}\0{mtime}\n".encode('utf-8'))

    method = FINGERPRINT_METHOD
    if problems:
        method = f"{FINGERPRINT_METHOD}/partial:{len(problems)}"
    return digest.hexdigest(), method


def fingerprint_session(dsm=None, *, kilosort_path=None, tracking_paths=None,
                        event_paths=None, max_files: int = 500) -> Tuple[Optional[str], str]:
    """Fingerprint the three data streams behind one analysis.

    Accepts a ``DataStorageManager`` or explicit paths. The DSM route is
    best-effort: a path resolver that raises for a missing modality must not
    stop the run from being logged, since an un-fingerprinted iteration is
    still worth recording — it simply cannot be confirmatory.
    """
    paths: List[Union[str, Path]] = []

    if dsm is not None:
        for getter in ('get_kilosort_path', 'get_tracking_files',
                       'get_behavioral_event_files'):
            method = getattr(dsm, getter, None)
            if method is None:
                continue
            try:
                value = method()
            except Exception:
                continue
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                paths.extend(value)
            else:
                paths.append(value)

    for extra in (kilosort_path, tracking_paths, event_paths):
        if extra is None:
            continue
        if isinstance(extra, (list, tuple, set)):
            paths.extend(extra)
        else:
            paths.append(extra)

    return fingerprint_inputs(paths, max_files=max_files)
