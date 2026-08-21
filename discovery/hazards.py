"""
Hazard registry — "Layer 0" of the AI-in-the-loop discovery platform
(``docs/AI_DISCOVERY_LOOP_DESIGN.md``; see ``HANDOFF.md`` for the increment
this landed in).

``CLAUDE.md``'s "Gotchas I keep tripping over" section is the prose form of
this registry: a hand-maintained list of traps that have silently corrupted a
result in this project at least once. Prose is the right representation for a
human picking the repo back up, and the wrong one for an agent that must
mechanically check whether the analysis it just ran tripped one. This module
is the machine-readable mirror.

**The two representations are cross-linked, not generated from each other.**
Every hazard carries ``references`` pointing back at ``CLAUDE.md`` and the
implementing code; a test asserts those links resolve. Generating prose from
JSON produces prose nobody wants to read, and generating JSON from prose
requires parsing prose. Both are maintained by hand, deliberately.

Detector kinds
--------------
A hazard is only as useful as the check behind it, so each carries one of:

``none``
    Prose only. Consulting it means stating, in the output, how the trap was
    avoided. The audit record is a ``HazardAcknowledgement`` row in the lab
    notebook — instructional enforcement, but reviewable after the fact.
``callable``
    A repo function (usually in :mod:`ephys._stats_utils` or
    :mod:`discovery.detectors`) plus a ``pass_if`` predicate. Genuinely
    executable.
``test``
    One or more pytest node ids. Passes on exit 0. A *list* means all must
    pass, which matters for guardrails that come in pairs: the self/target
    occupancy swap in ``tests/test_social_spatial_fields.py`` is only caught
    by running both halves, since a swapped implementation still passes
    either one alone.

Assumptions:
    - **A detector that cannot run reports ``ran=False, passed=None`` — never
      ``passed=True``.** A missing context key, an unresolvable manifest
      path, or an un-comparable value must never be reported as a pass. This
      is the single most important invariant in the module: a safety layer
      that reports unverified safety is worse than no safety layer, because
      it displaces the scepticism that would otherwise have been applied.
      :class:`~discovery._predicates.MissingValue` is the mechanism.
    - **Callables are resolved by import, never ``eval``.** A
      ``"module:function"`` string goes through :func:`importlib.import_module`
      plus :func:`getattr`. The registry is data, and data does not execute.
    - **The registry earns its place per entry.** An entry qualifies only if
      it has demonstrably caused a wrong result in this project (recorded in
      ``evidence``), or it carries a real detector. A registry that grows to
      100 prose warnings becomes the context tax it exists to prevent; the
      practical ceiling is ~40 entries.
    - **Test detectors shell out to pytest and are therefore opt-in.**
      ``run_detector(..., allow_tests=True)``. A propose-stage digest should
      not spawn subprocesses.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from discovery._predicates import (
    MissingValue,
    PredicateError,
    describe_predicate,
    evaluate_predicate,
    extract_path,
)

__all__ = [
    'HAZARDS_SCHEMA_VERSION',
    'DEFAULT_HAZARDS_PATH',
    'SEVERITIES',
    'KINDS',
    'STAGES',
    'HAZARD_STATUSES',
    'DETECTOR_TYPES',
    'Detector',
    'Hazard',
    'DetectorResult',
    'HazardRegistryError',
    'load_hazards',
    'hazards_by_id',
    'hazards_for',
    'render_digest',
    'resolve_callable',
    'resolve_context_ref',
    'run_detector',
    'run_detectors_for',
    'validate_registry',
]

HAZARDS_SCHEMA_VERSION = 1
DEFAULT_HAZARDS_PATH = Path(__file__).resolve().parent / 'hazards.json'

SEVERITIES: Tuple[str, ...] = ('critical', 'high', 'medium', 'low')
KINDS: Tuple[str, ...] = ('statistical', 'data', 'api', 'interpretation', 'provenance')
STAGES: Tuple[str, ...] = ('propose', 'implement', 'run', 'interpret')
HAZARD_STATUSES: Tuple[str, ...] = ('active', 'resolved', 'accepted')
DETECTOR_TYPES: Tuple[str, ...] = ('none', 'callable', 'test')

_SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}

# Required top-level keys on every hazard record.
_REQUIRED_KEYS = (
    'id', 'slug', 'title', 'severity', 'kind', 'status', 'added',
    'applies_to', 'symptom', 'mechanism', 'consequence', 'detector',
    'references',
)


class HazardRegistryError(ValueError):
    """The hazard file is malformed. Raised by :func:`validate_registry`."""


@dataclass(frozen=True)
class Detector:
    """How (or whether) a hazard can be checked mechanically."""

    type: str
    callable: Optional[str] = None
    args_from: Mapping[str, str] = field(default_factory=dict)
    pass_if: Optional[Mapping[str, Any]] = None
    test: Tuple[str, ...] = ()
    on_fail: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> 'Detector':
        if not raw:
            return cls(type='none')
        test = raw.get('test') or ()
        if isinstance(test, str):
            test = (test,)
        return cls(
            type=raw.get('type', 'none'),
            callable=raw.get('callable'),
            args_from=dict(raw.get('args_from') or {}),
            pass_if=raw.get('pass_if'),
            test=tuple(test),
            on_fail=raw.get('on_fail'),
        )

    @property
    def is_executable(self) -> bool:
        return self.type in ('callable', 'test')


@dataclass(frozen=True)
class Hazard:
    """One trap that has corrupted a result here, or that a detector guards."""

    id: str
    slug: str
    title: str
    severity: str
    kind: str
    status: str
    added: str
    applies_to: Mapping[str, Tuple[str, ...]]
    symptom: str
    mechanism: str
    consequence: str
    detector: Detector
    references: Tuple[str, ...]
    remedies: Tuple[str, ...] = ()
    known_gaps: Tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'Hazard':
        applies = raw.get('applies_to') or {}
        return cls(
            id=raw['id'],
            slug=raw['slug'],
            title=raw['title'],
            severity=raw['severity'],
            kind=raw['kind'],
            status=raw.get('status', 'active'),
            added=raw.get('added', ''),
            applies_to={k: tuple(v) for k, v in applies.items()},
            symptom=raw['symptom'],
            mechanism=raw['mechanism'],
            consequence=raw['consequence'],
            detector=Detector.from_dict(raw.get('detector')),
            references=tuple(raw.get('references') or ()),
            remedies=tuple(raw.get('remedies') or ()),
            known_gaps=tuple(raw.get('known_gaps') or ()),
            evidence=dict(raw.get('evidence') or {}),
        )

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_RANK.get(self.severity, len(SEVERITIES))

    @property
    def occurred(self) -> bool:
        """Did this trap actually produce a wrong result in this project?"""
        return bool(self.evidence.get('occurred'))

    def one_line(self) -> str:
        if self.detector.type == 'callable':
            det = f"callable({self.detector.callable})"
        elif self.detector.type == 'test':
            det = f"test(x{len(self.detector.test)})"
        else:
            det = 'prose-only'
        return f"{self.id} [{self.severity}] {self.title} - detector: {det}"


@dataclass(frozen=True)
class DetectorResult:
    """Outcome of attempting one hazard's detector.

    ``ran=False`` means the check could not be performed; ``passed`` is then
    ``None`` and must not be read as either safe or unsafe.
    """

    hazard_id: str
    ran: bool
    passed: Optional[bool]
    message: str
    detector_type: str = 'none'
    raw: Any = None

    @property
    def tripped(self) -> bool:
        """True only when the detector ran and the hazard was hit."""
        return self.ran and self.passed is False

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe form, for a ``HazardAcknowledgement`` row."""
        return {
            'hazard_id': self.hazard_id,
            'ran': self.ran,
            'passed': self.passed,
            'message': self.message,
            'detector_type': self.detector_type,
        }


# ---------------------------------------------------------------- loading

_CACHE: Dict[Tuple[str, float], Tuple[Hazard, ...]] = {}


def load_hazards(path: Optional[Path] = None, *, use_cache: bool = True) -> Tuple[Hazard, ...]:
    """Load and parse the hazard file. Memoized on ``(path, mtime)``."""
    path = Path(path) if path is not None else DEFAULT_HAZARDS_PATH
    if not path.exists():
        raise HazardRegistryError(f"hazard file not found: {path}")

    key = (str(path), path.stat().st_mtime)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    with open(path, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)

    version = raw.get('schema_version')
    if version != HAZARDS_SCHEMA_VERSION:
        raise HazardRegistryError(
            f"hazard file schema_version={version!r}, this code expects "
            f"{HAZARDS_SCHEMA_VERSION}. Regenerate or migrate {path}."
        )

    hazards = tuple(Hazard.from_dict(h) for h in raw.get('hazards', ()))
    if use_cache:
        _CACHE[key] = hazards
    return hazards


def hazards_by_id(path: Optional[Path] = None) -> Dict[str, Hazard]:
    return {h.id: h for h in load_hazards(path)}


def hazards_for(
    *,
    stage: Optional[str] = None,
    analysis: Optional[str] = None,
    module: Optional[str] = None,
    kind: Optional[str] = None,
    min_severity: str = 'low',
    include_inactive: bool = False,
    path: Optional[Path] = None,
) -> Tuple[Hazard, ...]:
    """Select applicable hazards, most severe first.

    ``min_severity`` is inclusive: ``'high'`` returns critical and high.
    Filters are conjunctive; omitted filters don't constrain. A hazard whose
    ``applies_to`` omits a dimension is treated as applying to all values of
    it — an unscoped hazard is assumed relevant rather than assumed
    irrelevant, because the failure mode of over-reporting is noise and the
    failure mode of under-reporting is a corrupted result.
    """
    if min_severity not in _SEVERITY_RANK:
        raise ValueError(f"min_severity must be one of {SEVERITIES}, got {min_severity!r}")
    cutoff = _SEVERITY_RANK[min_severity]

    def _matches(haz: Hazard, dimension: str, wanted: Optional[str]) -> bool:
        if wanted is None:
            return True
        declared = haz.applies_to.get(dimension)
        if not declared:
            return True  # unscoped => applies broadly
        return wanted in declared

    out = [
        h for h in load_hazards(path)
        if (include_inactive or h.status == 'active')
        and h.severity_rank <= cutoff
        and (kind is None or h.kind == kind)
        and _matches(h, 'stages', stage)
        and _matches(h, 'analyses', analysis)
        and _matches(h, 'modules', module)
    ]
    out.sort(key=lambda h: (h.severity_rank, h.id))
    return tuple(out)


def render_digest(hazards: Sequence[Hazard], verbosity: str = 'line') -> str:
    """Render hazards for pasting into a skill's context.

    ``'line'`` is one line each — the default, because a skill that pastes 30
    full records spends the context budget Layer 0 exists to protect.
    ``'brief'`` adds the mechanism and remedies; ``'full'`` adds evidence and
    references.
    """
    if verbosity not in ('line', 'brief', 'full'):
        raise ValueError(f"verbosity must be 'line'/'brief'/'full', got {verbosity!r}")
    if not hazards:
        return "(no applicable hazards)"

    chunks: List[str] = []
    for h in hazards:
        if verbosity == 'line':
            chunks.append(h.one_line())
            continue
        lines = [f"{h.id} [{h.severity}/{h.kind}] {h.title}",
                 f"  symptom:     {h.symptom}",
                 f"  mechanism:   {h.mechanism}",
                 f"  consequence: {h.consequence}"]
        if h.detector.type == 'callable':
            lines.append(f"  detector:    callable {h.detector.callable}")
        elif h.detector.type == 'test':
            lines.append(f"  detector:    tests {', '.join(h.detector.test)}")
        else:
            lines.append("  detector:    none (prose-only - state how you avoided it)")
        for r in h.remedies:
            lines.append(f"  remedy:      {r}")
        for g in h.known_gaps:
            lines.append(f"  known gap:   {g}")
        if verbosity == 'full':
            if h.occurred:
                where = ', '.join(str(w) for w in h.evidence.get('where', ()))
                lines.append(f"  OCCURRED:    {h.evidence.get('date', '?')} - {where}")
            for ref in h.references:
                lines.append(f"  see:         {ref}")
        chunks.append('\n'.join(lines))
    return '\n'.join(chunks) if verbosity == 'line' else '\n\n'.join(chunks)


# -------------------------------------------------------------- detectors

def resolve_callable(dotted: str):
    """Resolve ``"package.module:function"`` to the function object.

    Import + ``getattr`` only — never ``eval``. Raises
    :class:`HazardRegistryError` with the offending string on failure, so
    :func:`validate_registry` can report which entry rotted.
    """
    if not dotted or ':' not in dotted:
        raise HazardRegistryError(
            f"callable must be 'module:function', got {dotted!r}"
        )
    module_name, _, func_name = dotted.partition(':')
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # ImportError, but a broken module can raise anything
        raise HazardRegistryError(f"cannot import {module_name!r} for {dotted!r}: {exc}") from exc
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise HazardRegistryError(f"{module_name!r} has no attribute {func_name!r}") from exc


def resolve_context_ref(ref: Any, context: Mapping[str, Any]) -> Any:
    """Resolve one ``args_from`` value against a detector context.

    A string starting with ``$`` is a namespaced path — ``$manifest.…``,
    ``$params.…``, ``$results.…`` — optionally containing ``{placeholder}``
    substitutions filled from ``context['params']``. Anything else is a
    literal constant.

    Raises :class:`MissingValue` when the path cannot be resolved, which is
    what turns into ``ran=False`` rather than a false pass.
    """
    if not isinstance(ref, str) or not ref.startswith('$'):
        return ref  # literal

    body = ref[1:]
    if '{' in body:
        params = context.get('params') or {}
        try:
            body = body.format(**params)
        except (KeyError, IndexError) as exc:
            raise MissingValue(
                f"reference {ref!r} needs a substitution not present in params: {exc}"
            ) from exc

    namespace, _, path = body.partition('.')
    if namespace not in context:
        raise MissingValue(
            f"reference {ref!r}: no {namespace!r} namespace in context "
            f"(have: {sorted(context)})"
        )
    return extract_path(context[namespace], path)


def _format_on_fail(template: Optional[str], raw: Any, kwargs: Mapping[str, Any]) -> str:
    """Best-effort ``on_fail`` message rendering.

    The template names fields of the detector's return value (e.g.
    ``{best_achievable_q:.2f}``). A template that references something absent
    must not mask the real finding, so formatting failure degrades to the
    raw template rather than raising.
    """
    if not template:
        return ''
    fields: Dict[str, Any] = {}
    if isinstance(raw, Mapping):
        fields.update(raw)
    fields.update(kwargs)
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return template


def run_detector(
    hazard: Hazard,
    context: Optional[Mapping[str, Any]] = None,
    *,
    allow_tests: bool = False,
    test_timeout: int = 900,
) -> DetectorResult:
    """Attempt one hazard's detector.

    Never raises for an un-runnable detector — returns ``ran=False,
    passed=None`` with a message saying what was missing. Genuinely broken
    *registry* content (unknown op, unimportable callable) is also reported
    this way rather than raised, so one rotted entry cannot abort a whole
    digest; :func:`validate_registry` is the place that fails loudly.
    """
    context = dict(context or {})
    det = hazard.detector

    if det.type == 'none':
        return DetectorResult(
            hazard_id=hazard.id, ran=False, passed=None, detector_type='none',
            message=("prose-only hazard: no automated check exists. State explicitly "
                     "how this was avoided and record a HazardAcknowledgement."),
        )

    if det.type == 'test':
        if not det.test:
            return DetectorResult(hazard.id, False, None,
                                  "test detector declares no node ids", 'test')
        if not allow_tests:
            return DetectorResult(
                hazard.id, False, None, detector_type='test',
                message=("test-backed detector not run (spawns pytest). "
                         f"Re-run with allow_tests=True to execute: {', '.join(det.test)}"),
            )
        cmd = [sys.executable, '-m', 'pytest', '-q', '--no-header', *det.test]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=test_timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return DetectorResult(hazard.id, False, None,
                                  f"could not run pytest: {exc}", 'test')
        passed = proc.returncode == 0
        tail = (proc.stdout or proc.stderr or '').strip().splitlines()
        msg = f"pytest exit {proc.returncode} for {', '.join(det.test)}"
        if not passed and tail:
            msg += ' | ' + tail[-1]
        return DetectorResult(hazard.id, True, passed, msg, 'test',
                              raw={'returncode': proc.returncode})

    # det.type == 'callable'
    if det.type != 'callable':
        return DetectorResult(hazard.id, False, None,
                              f"unknown detector type {det.type!r}", det.type)
    try:
        fn = resolve_callable(det.callable)
    except HazardRegistryError as exc:
        return DetectorResult(hazard.id, False, None, str(exc), 'callable')

    kwargs: Dict[str, Any] = {}
    for name, ref in det.args_from.items():
        try:
            kwargs[name] = resolve_context_ref(ref, context)
        except MissingValue as exc:
            return DetectorResult(
                hazard.id, False, None, detector_type='callable',
                message=f"cannot check: {exc}. Detector needs {name}={ref!r}.",
            )

    try:
        raw = fn(**kwargs)
    except MissingValue as exc:
        return DetectorResult(hazard.id, False, None,
                              f"cannot check: {exc}", 'callable', raw=None)
    except Exception as exc:
        return DetectorResult(
            hazard.id, False, None, detector_type='callable',
            message=f"detector {det.callable} raised {type(exc).__name__}: {exc}",
        )

    if det.pass_if is None:
        # No predicate: the callable itself is the verdict.
        try:
            passed = bool(raw)
        except Exception as exc:  # pragma: no cover - defensive
            return DetectorResult(hazard.id, False, None,
                                  f"detector result not truth-testable: {exc}", 'callable')
        requirement = f"{det.callable} is truthy"
    else:
        try:
            passed = evaluate_predicate(raw, det.pass_if)
        except MissingValue as exc:
            return DetectorResult(
                hazard.id, False, None, detector_type='callable',
                message=f"cannot check: {exc}", raw=raw,
            )
        except PredicateError as exc:
            return DetectorResult(
                hazard.id, False, None, detector_type='callable',
                message=f"malformed pass_if in {hazard.id}: {exc}", raw=raw,
            )
        requirement = describe_predicate(det.pass_if)

    if passed:
        message = f"OK: {requirement}"
    else:
        detail = _format_on_fail(det.on_fail, raw, kwargs)
        message = f"TRIPPED: {requirement} not satisfied."
        if detail:
            message += ' ' + detail
    return DetectorResult(hazard.id, True, passed, message, 'callable', raw=raw)


def run_detectors_for(
    hazards: Sequence[Hazard],
    context: Optional[Mapping[str, Any]] = None,
    *,
    allow_tests: bool = False,
) -> Tuple[DetectorResult, ...]:
    """Run every hazard's detector, returning results in the input order."""
    return tuple(run_detector(h, context, allow_tests=allow_tests) for h in hazards)


# -------------------------------------------------------------- validation

def validate_registry(
    path: Optional[Path] = None,
    *,
    check_callables: bool = True,
    check_tests: bool = False,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Return a list of problems with the hazard file; empty means valid.

    This is what keeps the registry from rotting silently: a renamed function
    or a deleted test turns a "detector" into a decoration, and a decoration
    that looks like a detector is worse than an honest prose entry.

    ``check_tests`` shells out to ``pytest --collect-only`` and is therefore
    slow — the test suite runs it under the ``slow`` marker.
    """
    problems: List[str] = []
    try:
        hazards = load_hazards(path, use_cache=False)
    except HazardRegistryError as exc:
        return [str(exc)]

    if not hazards:
        problems.append("registry is empty")

    seen_ids: Dict[str, int] = {}
    seen_slugs: Dict[str, int] = {}
    for i, h in enumerate(hazards):
        where = f"hazards[{i}] ({h.id})"
        if h.id in seen_ids:
            problems.append(f"{where}: duplicate id (also at index {seen_ids[h.id]})")
        seen_ids[h.id] = i
        if h.slug in seen_slugs:
            problems.append(f"{where}: duplicate slug {h.slug!r}")
        seen_slugs[h.slug] = i

        if h.severity not in SEVERITIES:
            problems.append(f"{where}: severity {h.severity!r} not in {SEVERITIES}")
        if h.kind not in KINDS:
            problems.append(f"{where}: kind {h.kind!r} not in {KINDS}")
        if h.status not in HAZARD_STATUSES:
            problems.append(f"{where}: status {h.status!r} not in {HAZARD_STATUSES}")
        if h.detector.type not in DETECTOR_TYPES:
            problems.append(f"{where}: detector type {h.detector.type!r} not in {DETECTOR_TYPES}")

        for stage in h.applies_to.get('stages', ()):
            if stage not in STAGES:
                problems.append(f"{where}: unknown stage {stage!r} (expected {STAGES})")

        if not h.references:
            problems.append(f"{where}: no references - every hazard must cite where it's documented")

        # The earn-your-place rule, mechanically checked.
        if not h.occurred and not h.detector.is_executable:
            problems.append(
                f"{where}: prose-only AND never observed to cause a wrong result. "
                "Every entry must either have bitten (evidence.occurred) or carry a detector."
            )
        if h.occurred and not h.evidence.get('where'):
            problems.append(f"{where}: evidence.occurred is true but 'where' is empty")

        if h.detector.type == 'callable':
            if not h.detector.callable:
                problems.append(f"{where}: callable detector with no 'callable'")
            elif check_callables:
                try:
                    resolve_callable(h.detector.callable)
                except HazardRegistryError as exc:
                    problems.append(f"{where}: {exc}")
            if h.detector.pass_if is not None:
                try:
                    describe_predicate(h.detector.pass_if)
                    op = h.detector.pass_if.get('op')
                    from discovery._predicates import OPS
                    if op not in OPS:
                        problems.append(f"{where}: pass_if op {op!r} not in {OPS}")
                except Exception as exc:
                    problems.append(f"{where}: malformed pass_if: {exc}")

        if h.detector.type == 'test':
            if not h.detector.test:
                problems.append(f"{where}: test detector with no node ids")
            elif check_tests:
                problems.extend(_check_test_ids(h, where, repo_root))

    return problems


def _check_test_ids(hazard: Hazard, where: str, repo_root: Optional[Path]) -> List[str]:
    """Assert every declared pytest node id is collectable."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    cmd = [sys.executable, '-m', 'pytest', '--collect-only', '-q', '--no-header',
           *hazard.detector.test]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"{where}: could not collect test ids: {exc}"]
    if proc.returncode != 0:
        tail = (proc.stdout or proc.stderr or '').strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        return [f"{where}: pytest cannot collect {hazard.detector.test}: {detail}"]
    return []


# -------------------------------------------------------------------- CLI

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument('--digest', action='store_true', help='print the hazard digest')
    p.add_argument('--stage', choices=STAGES, default=None)
    p.add_argument('--analysis', default=None)
    p.add_argument('--kind', choices=KINDS, default=None)
    p.add_argument('--min-severity', choices=SEVERITIES, default='low')
    p.add_argument('--verbosity', choices=('line', 'brief', 'full'), default='line')
    p.add_argument('--validate', action='store_true', help='validate the registry and exit non-zero on problems')
    p.add_argument('--check-tests', action='store_true', help='also verify pytest node ids collect (slow)')
    args = p.parse_args(argv)

    if args.validate:
        problems = validate_registry(check_tests=args.check_tests)
        if problems:
            print(f"{len(problems)} problem(s) in the hazard registry:")
            for prob in problems:
                print(f"  - {prob}")
            return 1
        print(f"hazard registry OK ({len(load_hazards())} entries)")
        return 0

    selected = hazards_for(stage=args.stage, analysis=args.analysis,
                           kind=args.kind, min_severity=args.min_severity)
    print(render_digest(selected, args.verbosity))
    if not args.digest:
        print(f"\n({len(selected)} hazards; use --verbosity brief|full for detail)")
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(_main())
