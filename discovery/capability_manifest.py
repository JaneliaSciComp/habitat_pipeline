"""
The capability manifest's **consult** path: a cheap, local read that answers
"is analysis X testable on session Y, and if not, why not, and where instead?"

Its job is to make untestable hypotheses fail at generation time. Hypothesis 3
of this project was blocked only after the analysis was attempted, because
session ``20251216``'s tracking resolves only the focal animal and nothing
recorded that anywhere. Likewise ``decode_opponent_identity``'s default
``behavior_type='F'`` yields one usable opponent while ``'EC'`` yields eight,
which was discovered by trial and error.

Deliberate import discipline
----------------------------
This module must never import :mod:`ingestion`, :mod:`video`, or :mod:`ephys`.
Consulting the manifest is a JSON read plus dict traversal; if the consult path
could reach the ``//nearline`` SMB share, then sooner or later something would
make it, and a "cheap check" would start taking minutes. The expensive probing
lives in :mod:`discovery.manifest_build`. A test enforces the separation by
asserting ``ingestion.data_paths`` is absent from ``sys.modules`` after
importing this module.

Assumptions:
    - **A stale manifest is worse than no manifest**, because an agent will
      trust it and propose confidently against data that no longer exists —
      a new failure mode introduced by the fix. So
      :func:`manifest_status` *raises* on a missing file, a schema-version
      mismatch, or a changed cohort config, and only warns about age. Please
      don't soften those raises later; the warning would be ignored.
    - **Availability, not quality.** These checks pre-empt "the data isn't
      there" completely and "the analysis won't work" not at all. Whether
      cross-validation degenerates, or whether 19 events could ever support a
      claim, is not knowable from an inventory.
    - **suggest_sessions does not rank.** It returns the satisfying set with
      the relevant numbers. Ranking would need an objective function nobody
      has agreed on, and a fabricated "best session" score is worse than a
      list.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from discovery._predicates import MissingValue, extract_path
from discovery.requirements import (
    PARAM_SWEEPS,
    REQUIREMENTS,
    Req,
    ReqResult,
    evaluate_req,
    known_analyses,
    requirements_for,
)

__all__ = [
    'MANIFEST_SCHEMA_VERSION',
    'DEFAULT_MANIFEST_PATH',
    'MANIFEST_MAX_AGE_DAYS',
    'ManifestError',
    'ManifestStale',
    'ManifestStatus',
    'Unmet',
    'FeasibilityReport',
    'SessionOption',
    'SourceVerification',
    'load_manifest',
    'manifest_status',
    'list_sessions',
    'session_capabilities',
    'check_testable',
    'resolve_params',
    'suggest_sessions',
    'verify_sources',
    'config_sha256',
]

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / 'capability_manifest.json'
MANIFEST_MAX_AGE_DAYS = 30


class ManifestError(RuntimeError):
    """The manifest is missing, unreadable, or of an incompatible version."""


class ManifestStale(ManifestError):
    """A cohort config changed since the manifest was built.

    Every fact derived from that config is now suspect, so this refuses rather
    than warns.
    """


@dataclass(frozen=True)
class ManifestStatus:
    state: str            # 'fresh' | 'aging' | 'partial'
    generated_at: str
    age_days: Optional[float]
    probe_level: str
    n_sessions: int
    warnings: Tuple[str, ...] = ()

    def summary(self) -> str:
        line = (f"manifest {self.state}: generated {self.generated_at}, "
                f"{self.n_sessions} session(s), probe_level={self.probe_level}")
        return '\n'.join([line] + [f"  warning: {w}" for w in self.warnings])


@dataclass(frozen=True)
class Unmet:
    """One requirement that was not satisfied, with everything needed to act."""

    requirement: str
    observed: str
    reason: str
    remedy: Optional[str] = None
    hazard_ids: Tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f" - {self.requirement}", f"   observed: {self.observed}"]
        if self.reason:
            lines.append(f"   {self.reason}")
        if self.remedy:
            lines.append(f"   remedy: {self.remedy}")
        if self.hazard_ids:
            lines.append(f"   hazards: {', '.join(self.hazard_ids)}")
        return '\n'.join(lines)


@dataclass(frozen=True)
class FeasibilityReport:
    analysis: str
    session_id: str
    animal_id: Optional[str]
    params: Mapping[str, Any]
    testable: bool
    unmet: Tuple[Unmet, ...] = ()
    warnings: Tuple[Unmet, ...] = ()
    undetermined: Tuple[Unmet, ...] = ()
    viable_params: Tuple[Mapping[str, Any], ...] = ()
    manifest_generated_at: str = ''
    manifest_state: str = ''

    def summary(self) -> str:
        """Render for a skill to paste, verdict first."""
        head = 'TESTABLE' if self.testable else 'NOT TESTABLE'
        params = ', '.join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        lines = [f"{head}  {self.analysis}  {self.session_id}"
                 + (f" / {self.animal_id}" if self.animal_id else '')
                 + (f" ({params})" if params else '')]
        for item in self.unmet:
            lines.append(item.render())
        if self.undetermined:
            lines.append("UNDETERMINED (a parameter was left open):")
            lines.extend(item.render() for item in self.undetermined)
        if self.warnings:
            lines.append("WARNINGS (non-blocking, but must be reported):")
            lines.extend(item.render() for item in self.warnings)
        if self.viable_params:
            lines.append(f"viable parameter sets: {len(self.viable_params)}")
            for combo in self.viable_params[:5]:
                lines.append(f"   {combo}")
        lines.append(f" manifest generated {self.manifest_generated_at} "
                     f"(state={self.manifest_state})")
        return '\n'.join(lines)


@dataclass(frozen=True)
class SessionOption:
    session_id: str
    cohort: str
    animal_id: Optional[str]
    params: Mapping[str, Any]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceVerification:
    session_id: str
    ok: bool
    changed: Tuple[str, ...] = ()
    missing: Tuple[str, ...] = ()
    checked: int = 0


# ------------------------------------------------------------------ loading

_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


def config_sha256(config_path: Path) -> Optional[str]:
    """Hash a local cohort config file.

    The cheapest real staleness signal available without touching the share:
    if a cohort's ephys root or ``pixels_per_cm`` changed, everything derived
    from it is suspect. Only small local config files are hashed — hashing
    gigabytes of Kilosort output over SMB to detect drift would be theatre.
    """
    path = Path(config_path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Optional[Path] = None, *, use_cache: bool = True) -> Dict[str, Any]:
    """Read and version-check the manifest. Memoized on ``(path, mtime)``."""
    path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise ManifestError(
            f"capability manifest not found at {path}. Build it with "
            "`python scripts/build_capability_manifest.py` on a machine where the "
            "//nearline share is mounted."
        )
    key = (str(path), path.stat().st_mtime)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"capability manifest at {path} is not valid JSON: {exc}") from exc

    version = raw.get('schema_version')
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"capability manifest schema_version={version!r}, this code expects "
            f"{MANIFEST_SCHEMA_VERSION}. Rebuild it - consulting a manifest written "
            "by a different schema would silently misread fields."
        )
    if use_cache:
        _CACHE[key] = raw
    return raw


def manifest_status(path: Optional[Path] = None, *,
                    repo_root: Optional[Path] = None) -> ManifestStatus:
    """Local-only freshness check. Raises on anything that makes it untrustworthy.

    Runs on every consult, so it must not touch the network or the share.
    """
    from datetime import datetime, timezone

    manifest = load_manifest(path)
    generated_at = manifest.get('generated_at', '')
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent

    problems: List[str] = []
    for cohort in manifest.get('cohorts', ()):
        declared = cohort.get('config_sha256')
        config_path = root / cohort.get('config_path', '')
        actual = config_sha256(config_path)
        if actual is None:
            problems.append(f"config {cohort.get('config_path')} is missing")
        elif declared and actual != declared:
            problems.append(f"config {cohort.get('config_path')} changed since the build")
    if problems:
        raise ManifestStale(
            "capability manifest is stale: " + '; '.join(problems) +
            ". Rebuild with scripts/build_capability_manifest.py. Refusing to answer "
            "from a manifest whose inputs have moved."
        )

    age_days = None
    if generated_at:
        try:
            stamp = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            age_days = (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0
        except ValueError:
            pass

    probe_level = (manifest.get('generated_by') or {}).get('probe_level', 'unknown')
    notes: List[str] = []
    state = 'fresh'
    if probe_level == 'paths':
        state = 'partial'
        notes.append("built with --probe-level paths: content facts (cell counts, "
                     "event class counts, tracking coverage) are absent")
    if age_days is not None and age_days > MANIFEST_MAX_AGE_DAYS:
        state = 'aging' if state == 'fresh' else state
        notes.append(f"{age_days:.0f} days old (limit {MANIFEST_MAX_AGE_DAYS})")

    status = ManifestStatus(
        state=state, generated_at=generated_at, age_days=age_days,
        probe_level=probe_level, n_sessions=len(manifest.get('sessions', {})),
        warnings=tuple(notes),
    )
    for note in notes:
        warnings.warn(f"capability manifest: {note}", RuntimeWarning, stacklevel=2)
    return status


# ------------------------------------------------------------------ reading

def _sessions(path: Optional[Path]) -> Dict[str, Any]:
    return load_manifest(path).get('sessions', {})


_SESSION_DATE_RE = re.compile(r'(20\d{6})')


def _session_date(session_id: Any) -> Optional[str]:
    """The ``YYYYMMDD`` inside a session id, or ``None``.

    Mirrors ``database.lab_notebook.normalize_session_key`` — a small
    independent copy rather than an import, following the same convention as
    ``ephys._stats_utils.benjamini_hochberg`` vs. its counterpart in
    ``social_spatial_fields``. The reason here is import weight: the consult
    path is meant to be a JSON read, and reaching into ``database`` would pull
    SQLAlchemy in behind it. ``tests/test_capability_manifest.py`` asserts the
    two implementations agree.
    """
    if session_id is None:
        return None
    match = _SESSION_DATE_RE.search(str(session_id))
    return match.group(1) if match else None


def _resolve_session(session_id: str, path: Optional[Path]) -> Tuple[str, Dict[str, Any]]:
    """Find a session record by exact key, then by embedded date.

    Session ids are written several ways across this codebase (``'20251210'``
    in the notebook, ``'RatCity_20251210_1359_40Hz'`` on disk), so an exact
    match alone would miss the very id an analysis actually passes.
    """
    sessions = _sessions(path)
    if session_id in sessions:
        return session_id, sessions[session_id]

    wanted = _session_date(session_id)
    if wanted:
        for key, record in sessions.items():
            if record.get('session_date') == wanted or _session_date(key) == wanted:
                return key, record
    raise KeyError(
        f"session {session_id!r} is not in the capability manifest "
        f"({len(sessions)} session(s) recorded). Rebuild the manifest, or check "
        "the id."
    )


def session_capabilities(session_id: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """The raw capability record for one session."""
    return _resolve_session(session_id, path)[1]


def list_sessions(cohort: str = None, path: Optional[Path] = None) -> Tuple[str, ...]:
    return tuple(sorted(
        key for key, record in _sessions(path).items()
        if cohort is None or record.get('cohort') == cohort
    ))


# ------------------------------------------------------------- feasibility

def _to_unmet(result: ReqResult) -> Unmet:
    return Unmet(
        requirement=result.requirement,
        observed=result.observed,
        reason=result.req.reason,
        remedy=result.req.remedy,
        hazard_ids=result.req.hazards,
    )


def check_testable(analysis: str, session_id: str, animal_id: str = None,
                   *, path: Optional[Path] = None, **params) -> FeasibilityReport:
    """Can ``analysis`` be run on this session, with these params?

    Returns a report rather than a bool: a caller that gets ``False`` needs the
    reason and an alternative in order to do anything useful with it.

    Parameters left open become sweep dimensions — the report comes back with
    ``viable_params`` enumerated instead of a failure — which is how
    ``behavior_type`` stops being a guess.
    """
    requirements = requirements_for(analysis)
    status = manifest_status(path)
    session_key, record = _resolve_session(session_id, path)

    supplied = dict(params)
    if animal_id is not None:
        supplied.setdefault('animal_id', animal_id)

    unmet: List[Unmet] = []
    soft: List[Unmet] = []
    undetermined: List[Unmet] = []
    for req in requirements:
        result = evaluate_req(req, record, supplied)
        if result.satisfied is True:
            continue
        item = _to_unmet(result)
        if result.satisfied is None:
            undetermined.append(item)
        elif req.severity == 'warning':
            soft.append(item)
        else:
            unmet.append(item)

    viable: Tuple[Mapping[str, Any], ...] = ()
    if undetermined:
        viable = resolve_params(analysis, session_id, animal_id, path=path, **params)

    return FeasibilityReport(
        analysis=analysis, session_id=session_key, animal_id=animal_id,
        params=supplied, testable=not unmet and not undetermined,
        unmet=tuple(unmet), warnings=tuple(soft), undetermined=tuple(undetermined),
        viable_params=viable,
        manifest_generated_at=status.generated_at, manifest_state=status.state,
    )


def _sweep_values(analysis: str, record: Mapping[str, Any],
                  params: Mapping[str, Any]) -> Dict[str, List[Any]]:
    """Legal values for each sweepable parameter, read from the manifest."""
    out: Dict[str, List[Any]] = {}
    for name, path_template in (PARAM_SWEEPS.get(analysis) or {}).items():
        if params.get(name) is not None:
            continue
        try:
            container = extract_path(record, path_template.format(**dict(params)))
        except (MissingValue, KeyError, IndexError):
            continue
        if isinstance(container, Mapping):
            out[name] = sorted(container.keys())
        elif isinstance(container, (list, tuple)):
            out[name] = list(container)
    return out


def resolve_params(analysis: str, session_id: str, animal_id: str = None,
                   *, path: Optional[Path] = None,
                   **params) -> Tuple[Mapping[str, Any], ...]:
    """Enumerate the parameter combinations that are actually viable here.

    This is what replaces trial and error. Rather than guessing a
    ``behavior_type`` and finding out after a load-and-sync cycle that it
    yields one usable class, ask the manifest which values work.
    """
    import itertools

    _, record = _resolve_session(session_id, path)
    base = dict(params)
    if animal_id is not None:
        base.setdefault('animal_id', animal_id)

    # Sweep over animals with ephys when none was named.
    if base.get('animal_id') is None:
        animals = ((record.get('ephys') or {}).get('animals')) or []
        candidates = [{**base, 'animal_id': a} for a in animals] or [base]
    else:
        candidates = [base]

    viable: List[Mapping[str, Any]] = []
    for candidate in candidates:
        sweeps = _sweep_values(analysis, record, candidate)
        if not sweeps:
            report = _check_concrete(analysis, record, candidate)
            if report:
                viable.append(candidate)
            continue
        names = sorted(sweeps)
        for combo in itertools.product(*(sweeps[n] for n in names)):
            trial = {**candidate, **dict(zip(names, combo))}
            if _check_concrete(analysis, record, trial):
                viable.append(trial)
    return tuple(viable)


def _check_concrete(analysis: str, record: Mapping[str, Any],
                    params: Mapping[str, Any]) -> bool:
    """Do all blocking requirements pass for a fully-specified param set?"""
    for req in requirements_for(analysis):
        if req.severity != 'blocking':
            continue
        result = evaluate_req(req, record, params)
        if result.satisfied is not True:
            return False
    return True


def suggest_sessions(analysis: str, animal_id: str = None, *,
                     cohort: str = None, path: Optional[Path] = None,
                     **params) -> Tuple[SessionOption, ...]:
    """Sessions where this analysis *is* testable, with the relevant numbers.

    Deliberately unranked. Which session is "best" depends on an objective
    nobody has agreed on, and inventing a score would be worse than handing
    back the set.
    """
    options: List[SessionOption] = []
    for session_id in list_sessions(cohort=cohort, path=path):
        record = session_capabilities(session_id, path)
        for combo in resolve_params(analysis, session_id, animal_id, path=path, **params):
            notes = []
            tracking = record.get('tracking') or {}
            if tracking.get('n_identity_resolved_animals') is not None:
                notes.append(f"{tracking['n_identity_resolved_animals']} identity-resolved")
            ephys = record.get('ephys') or {}
            if ephys.get('n_animals_with_ephys') is not None:
                notes.append(f"{ephys['n_animals_with_ephys']} animals with ephys")
            covered = tracking.get('frac_of_ephys_duration_covered')
            if covered is not None:
                notes.append(f"tracking covers {covered:.0%} of the recording")
            options.append(SessionOption(
                session_id=session_id, cohort=record.get('cohort', ''),
                animal_id=combo.get('animal_id'), params=combo, notes=tuple(notes),
            ))
    return tuple(options)


def verify_sources(session_id: str, path: Optional[Path] = None) -> SourceVerification:
    """Opt-in staleness check that *does* touch the share.

    Compares recorded ``mtime``/``size`` against the filesystem. A handful of
    ``stat()`` calls per session, not a content hash — replacement and
    re-export are the realistic drift, and they both change size or mtime.
    """
    _, record = _resolve_session(session_id, path)
    sources = (record.get('provenance') or {}).get('sources') or {}

    changed: List[str] = []
    missing: List[str] = []
    checked = 0

    def _check(label: str, entry: Mapping[str, Any]) -> None:
        nonlocal checked
        target = entry.get('path')
        if not target:
            return
        checked += 1
        try:
            stat = os.stat(target)
        except OSError:
            missing.append(f"{label}: {target}")
            return
        if entry.get('size') is not None and int(entry['size']) != stat.st_size:
            changed.append(f"{label}: size {entry['size']} -> {stat.st_size}")
        elif entry.get('mtime') is not None and int(entry['mtime']) != int(stat.st_mtime):
            changed.append(f"{label}: mtime changed")

    for group, value in sources.items():
        if isinstance(value, Mapping) and 'path' in value:
            _check(group, value)
        elif isinstance(value, Mapping):
            for name, entry in value.items():
                if isinstance(entry, Mapping):
                    _check(f"{group}/{name}", entry)

    return SourceVerification(
        session_id=session_id, ok=not changed and not missing,
        changed=tuple(changed), missing=tuple(missing), checked=checked,
    )
