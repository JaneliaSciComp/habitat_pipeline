"""
Declarative per-analysis data requirements, evaluated against the capability
manifest so that a hypothesis which cannot be tested fails at *generation*
time with a reason and an alternative, rather than forty minutes into an
extraction.

The distinction from :mod:`discovery.hazards` is worth keeping sharp:

- A **requirement** answers "does the data needed for this exist?" A missing
  prerequisite — no partner tracking, no validated head direction, no arena
  geometry — is a capability gap, and it belongs here.
- A **hazard** answers "will this silently lie?" A trap that corrupts a
  result which does run belongs there.

Both share one restricted predicate evaluator
(:mod:`discovery._predicates`), so a requirement's ``op`` and a hazard's
``pass_if`` can never drift apart in meaning.

Assumptions:
    - **The op table is fixed and flat**, with no boolean combinators. This is
      the guardrail against the bundles below quietly becoming a rules engine.
      A check that needs real logic becomes a named function in
      :mod:`discovery.detectors` and is referenced from a hazard instead.
    - **An unrecorded field is unmet, not satisfied.** If the manifest has no
      record of a session's tracking, testability cannot be asserted — the
      honest answer is "not according to what we know", surfaced as
      ``observed='absent'``. A partial probe is reported separately by
      :func:`discovery.capability_manifest.manifest_status`, so a caller can
      tell "no such data" from "we never looked".
    - **Requirements are necessary, not sufficient.** They pre-empt
      *availability* failures completely and *quality* failures not at all:
      whether cross-validation degenerates, or whether 19 events were ever
      going to support a claim, is not knowable from an inventory. Some
      hypotheses will still die late, and that is the design working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from discovery._predicates import (
    MissingValue,
    OPS,
    PredicateError,
    describe_predicate,
    evaluate_predicate,
    extract_path,
)

__all__ = [
    'Req',
    'REQUIREMENTS',
    'PARAM_SWEEPS',
    'requirements_for',
    'evaluate_req',
    'known_analyses',
]

SEVERITIES = ('blocking', 'warning')


@dataclass(frozen=True)
class Req:
    """One condition an analysis places on a session's capability record.

    ``path`` is a dotted path into the per-session manifest record and may
    contain ``{placeholder}`` substitutions filled from the caller's params —
    ``ephys.per_animal.{animal_id}.n_quality_cells`` needs ``animal_id``.
    When a needed substitution is absent the requirement becomes a *sweep*
    dimension rather than a failure: that is how
    :func:`discovery.capability_manifest.resolve_params` enumerates the viable
    parameter grid instead of demanding one up front.
    """

    path: str
    op: str
    value: Any = None
    reason: str = ''
    remedy: Optional[str] = None
    hazards: Tuple[str, ...] = ()
    severity: str = 'blocking'

    def __post_init__(self):
        if self.op not in OPS:
            raise PredicateError(
                f"Req({self.path!r}) uses unsupported op {self.op!r}. "
                f"Supported: {', '.join(OPS)}. Anything richer belongs in "
                "discovery.detectors as a named function."
            )
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    @property
    def placeholders(self) -> Tuple[str, ...]:
        """Substitution keys this requirement needs, e.g. ``('animal_id',)``."""
        out, rest = [], self.path
        while '{' in rest:
            _, _, rest = rest.partition('{')
            name, _, rest = rest.partition('}')
            if name:
                out.append(name)
        return tuple(out)

    def pass_if(self) -> Dict[str, Any]:
        spec: Dict[str, Any] = {'op': self.op}
        if self.value is not None or self.op in ('==', '!='):
            spec['value'] = self.value
        return spec

    def describe(self) -> str:
        return f"{self.path} {describe_predicate({'op': self.op, 'value': self.value})}"


@dataclass(frozen=True)
class ReqResult:
    """Outcome of one requirement against one session record."""

    req: Req
    satisfied: Optional[bool]     # None => could not determine
    observed: str
    requirement: str

    @property
    def blocking(self) -> bool:
        return self.satisfied is not True and self.req.severity == 'blocking'


def evaluate_req(req: Req, session_record: Mapping[str, Any],
                 params: Mapping[str, Any]) -> ReqResult:
    """Evaluate one requirement, reporting what was observed either way."""
    try:
        path = req.path.format(**dict(params or {}))
    except (KeyError, IndexError) as exc:
        return ReqResult(req, None, f"needs {exc} to be specified", req.describe())

    requirement = f"{path} {describe_predicate(req.pass_if())}"
    try:
        observed_value = extract_path(session_record, path)
    except MissingValue:
        return ReqResult(req, False, 'absent (not recorded in the manifest)', requirement)

    try:
        satisfied = evaluate_predicate(observed_value, req.pass_if())
    except MissingValue as exc:
        return ReqResult(req, None, f"not comparable: {exc}", requirement)

    return ReqResult(req, satisfied, _render(observed_value), requirement)


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        shown = list(value)[:6]
        suffix = f" (+{len(value) - len(shown)} more)" if len(value) > len(shown) else ''
        return f"{shown}{suffix}"
    return repr(value)


# --------------------------------------------------------------- bundles

#: Applies to every analysis: without a clock map and some cells, nothing
#: downstream means anything.
_COMMON = (
    Req('ephys.per_animal.{animal_id}.path_exists', 'is_true',
        reason='No Kilosort output resolved for this animal/session.',
        remedy='Check the cohort config, or pick an animal the manifest lists as '
               'having ephys for this session.',
        hazards=('HZ-API-001',)),
    Req('ephys.per_animal.{animal_id}.sync.ok', 'is_true',
        reason='Ephys/behaviour clock alignment failed for this session, so nothing '
               'event- or tracking-aligned can be trusted.',
        remedy='Any animal in the session shares the clock - try another animal, or '
               'another DIO channel.',
        hazards=('HZ-DATA-005',)),
    Req('ephys.per_animal.{animal_id}.n_quality_cells', '>=', 10,
        reason='Fewer than 10 quality cells makes a population claim meaningless.',
        remedy='Loosen the quality thresholds deliberately and say so, or pick a '
               'session with more units.',
        hazards=('HZ-STAT-001',)),
)

_EVENT_DECODING = _COMMON + (
    Req('events.available', 'is_true',
        reason='No scored behavioural events for this session.',
        hazards=('HZ-API-002',)),
    Req('events.frac_events_within_ephys_window', '>=', 0.95,
        severity='warning',
        reason='Some scored events fall outside the recording, so the usable event '
               'count is lower than it looks.',
        hazards=('HZ-DATA-006',)),
)

_TRACKING_BASED = _COMMON + (
    Req('tracking.available', 'is_true',
        reason='No tracking file resolved for this session.',
        hazards=('HZ-DATA-001',)),
    Req('tracking.frac_of_ephys_duration_covered', '>=', 0.8,
        severity='warning',
        reason='Tracking covers only part of the recording; an analysis over the whole '
               'session mixes in time with no position data.',
        remedy='Pass t_window_ephys from the manifest\'s tracking.ephys_window.',
        hazards=('HZ-DATA-002',)),
    Req('pixels_per_cm', 'is_present', severity='warning',
        reason='pixels_per_cm is unset for this cohort, so any parameter named *_cm '
               'is really in pixels.',
        hazards=('HZ-DATA-004',)),
)

_PARTNER_TRACKING = _TRACKING_BASED + (
    Req('tracking.n_identity_resolved_animals', '>=', 2,
        reason='Only the focal animal is identity-resolved in this session\'s tracking, '
               'so no partner positions exist.',
        remedy='Use suggest_sessions() to find a session with multi-animal tracking.',
        hazards=('HZ-DATA-001',)),
)

#: Keyed by the analysis module name used in ``Iteration.analysis_module``.
REQUIREMENTS: Dict[str, Tuple[Req, ...]] = {
    'ephys.decode_opponent_identity': _EVENT_DECODING + (
        Req('events.per_animal.{animal_id}.opponent_labels.{behavior_type}.n_classes_usable',
            '>=', 2,
            reason='Fewer than two opponent classes reach min_events_per_class, so LDA '
                   'has nothing to separate and label extraction returns empty arrays.',
            remedy='Consult the manifest for a behavior_type with >= 2 usable classes, '
                   "or use label_mode='group' to pool opponents.",
            hazards=('HZ-STAT-010',)),
    ),
    'ephys.decode_event_outcome': _EVENT_DECODING + (
        Req('events.per_animal.{animal_id}.outcome_labels.{behavior_type}.usable',
            'is_true',
            reason='Not enough winner/loser events of this type to cross-validate.',
            remedy='Leave behavior_type unset to pool every event with a winner and a '
                   'loser - but say so explicitly, since that is not just fights.',
            hazards=('HZ-STAT-010', 'HZ-API-004')),
    ),
    'ephys.decode_location': _TRACKING_BASED + (
        Req('tracking.objects.{object_name}.identity_resolved', 'is_true',
            reason='That object is not identity-resolved in this session\'s tracking.',
            remedy='Use suggest_sessions() to find a session where it is.',
            hazards=('HZ-DATA-001',)),
        Req('tracking.objects.{object_name}.frac_frames_present', '>=', 0.5,
            severity='warning',
            reason='That object is tracked in only a fraction of frames.',
            hazards=('HZ-DATA-003',)),
    ),
    'ephys.social_spatial_fields': _PARTNER_TRACKING,
    'ephys.decode_partner_distance': _PARTNER_TRACKING,
    'ephys.inter_brain_dynamics': _COMMON + (
        Req('ephys.n_animals_with_ephys', '>=', 2,
            reason='A shared subspace needs two simultaneously-recorded animals.',
            hazards=('HZ-API-007',)),
    ),
}

#: Manifest paths enumerating the legal values of each sweepable parameter, so
#: `resolve_params` can build the viable grid instead of guessing.
PARAM_SWEEPS: Dict[str, Dict[str, str]] = {
    'ephys.decode_opponent_identity': {
        'behavior_type': 'events.per_animal.{animal_id}.opponent_labels',
    },
    'ephys.decode_event_outcome': {
        'behavior_type': 'events.per_animal.{animal_id}.outcome_labels',
    },
    'ephys.decode_location': {
        'object_name': 'tracking.identity_resolved_animals',
    },
}


def known_analyses() -> Tuple[str, ...]:
    return tuple(sorted(REQUIREMENTS))


def requirements_for(analysis: str) -> Tuple[Req, ...]:
    """Requirement bundle for an analysis module name.

    Raises :class:`KeyError` for an unknown analysis rather than returning an
    empty tuple, because "no requirements" and "we have no idea what this
    needs" must not look the same to a caller deciding whether to run
    something.
    """
    try:
        return REQUIREMENTS[analysis]
    except KeyError:
        raise KeyError(
            f"no requirement bundle for {analysis!r}. Known: "
            f"{', '.join(known_analyses())}. Add one to discovery/requirements.py "
            "rather than letting an unknown analysis report itself testable."
        ) from None
