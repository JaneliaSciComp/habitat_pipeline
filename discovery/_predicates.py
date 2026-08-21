"""
The one restricted predicate evaluator shared by the hazard registry
(:mod:`discovery.hazards`) and the capability-requirement bundles
(:mod:`discovery.requirements`).

Both layers need to express "this observed value must satisfy this condition"
in a *data* file rather than in code, so both need a way to evaluate a
`{field, op, value}` triple. Single-sourcing it here means a hazard's
``pass_if`` and a requirement's ``op`` can never drift apart in meaning.

Assumptions:
    - **The op table is deliberately fixed and flat.** There are no boolean
      combinators (no ``and``/``or``/``not``), no nesting, and no arithmetic.
      This is the guardrail against the file format quietly becoming a
      homegrown rules engine: the moment a check needs real logic it must
      become a named function in :mod:`discovery.detectors`, where it is
      readable, typed, and unit-testable. If you find yourself wanting to add
      an operator, that is the signal to write a function instead.
    - **No ``eval``/``exec``, ever.** An unknown operator raises
      :class:`PredicateError` rather than falling back to Python evaluation.
      The hazard file is data authored to be trusted, but it is still data,
      and a data file that can execute arbitrary code is a different kind of
      artifact than the one this design intends.
    - **A missing value is not a passing value.** Absent, ``None``, and
      un-comparable subjects raise :class:`MissingValue` so that callers can
      distinguish "the check ran and passed" from "the check could not run".
      Collapsing those two into ``True`` is the specific failure mode that
      makes a safety layer worse than no safety layer, because it reports
      safety it never verified.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

__all__ = [
    'PredicateError',
    'MissingValue',
    'OPS',
    'UNARY_OPS',
    'extract_path',
    'evaluate_predicate',
    'describe_predicate',
]


class PredicateError(ValueError):
    """The predicate itself is malformed (unknown op, missing operand)."""


class MissingValue(LookupError):
    """The subject of the predicate is absent, ``None``, or un-comparable.

    Distinct from a failing predicate: callers must surface this as "could
    not check", never as "checked and passed".
    """


def _cmp(op_name, fn):
    """Wrap an ordering comparison so ``None`` becomes MissingValue.

    ``None < 5`` is a TypeError on Python 3, and ``None`` in this codebase
    means "not recorded" (e.g. a manifest field a partial probe never
    filled). Treating it as un-comparable rather than letting the TypeError
    escape keeps the "cannot run" path distinct from a crash.
    """

    def _inner(a, b):
        if a is None or b is None:
            raise MissingValue(
                f"cannot evaluate {op_name!r}: operand is None "
                f"(left={a!r}, right={b!r})"
            )
        try:
            return bool(fn(a, b))
        except TypeError as exc:
            raise MissingValue(
                f"cannot evaluate {op_name!r} on {type(a).__name__} "
                f"vs {type(b).__name__}: {exc}"
            ) from exc

    return _inner


def _contains(a, b):
    if b is None:
        raise MissingValue("'in' requires a collection as its value")
    try:
        return a in b
    except TypeError as exc:
        raise MissingValue(f"'in' operand is not a collection: {exc}") from exc


#: Binary operators: ``(observed, value) -> bool``.
_BINARY_OPS = {
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '<': _cmp('<', lambda a, b: a < b),
    '<=': _cmp('<=', lambda a, b: a <= b),
    '>': _cmp('>', lambda a, b: a > b),
    '>=': _cmp('>=', lambda a, b: a >= b),
    'in': _contains,
    'not_in': lambda a, b: not _contains(a, b),
}

#: Unary operators: ``(observed,) -> bool``. These take no ``value``.
_UNARY_OPS = {
    # `is True`, not truthiness: a hazard asserting `resolvable is_true`
    # should not be satisfied by the string "no" or by a non-empty dict.
    # numpy bools are accepted via the == True comparison.
    'is_true': lambda a: a is True or a == True,  # noqa: E712 - numpy.bool_
    'is_false': lambda a: a is False or a == False,  # noqa: E712
    'is_present': lambda a: a is not None,
    'is_absent': lambda a: a is None,
}

OPS: Tuple[str, ...] = tuple(sorted(_BINARY_OPS)) + tuple(sorted(_UNARY_OPS))
UNARY_OPS: Tuple[str, ...] = tuple(sorted(_UNARY_OPS))

_MISSING = object()


def extract_path(subject: Any, path: str) -> Any:
    """Follow a dotted ``path`` into nested mappings / sequences.

    Supports mapping keys and integer sequence indices, so a manifest path
    like ``tracking.objects.rat631.x_std_px`` and a result path like
    ``per_object.0.p_value`` both work.

    Raises :class:`MissingValue` if any step is absent — never returns a
    default, because a default is indistinguishable from a real value at the
    call site.
    """
    if not path:
        return subject
    current = subject
    walked = []
    for part in path.split('.'):
        walked.append(part)
        here = '.'.join(walked)
        if isinstance(current, Mapping):
            if part not in current:
                raise MissingValue(f"path {path!r}: no key {here!r}")
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise MissingValue(f"path {path!r}: bad index at {here!r}: {exc}") from exc
        else:
            raise MissingValue(
                f"path {path!r}: cannot descend into {type(current).__name__} at {here!r}"
            )
    return current


def evaluate_predicate(subject: Any, pass_if: Mapping[str, Any]) -> bool:
    """Evaluate a ``{field?, op, value?}`` predicate against ``subject``.

    ``field`` (optional) is a dotted path extracted from ``subject`` first,
    so a detector can return a whole dict (e.g. ``fdr_resolution``'s result)
    and the predicate can name the one key that matters.

    Raises :class:`PredicateError` for a malformed predicate and
    :class:`MissingValue` when the subject cannot be resolved or compared.
    """
    if not isinstance(pass_if, Mapping):
        raise PredicateError(f"pass_if must be a mapping, got {type(pass_if).__name__}")

    op = pass_if.get('op')
    if op is None:
        raise PredicateError(f"pass_if is missing 'op': {dict(pass_if)!r}")
    if op not in _BINARY_OPS and op not in _UNARY_OPS:
        raise PredicateError(
            f"unknown op {op!r}. Supported: {', '.join(OPS)}. "
            "Checks needing anything richer belong in discovery.detectors "
            "as a named function, not in the predicate language."
        )

    field = pass_if.get('field')
    observed = extract_path(subject, field) if field else subject

    if op in _UNARY_OPS:
        if 'value' in pass_if:
            raise PredicateError(f"op {op!r} is unary but a 'value' was supplied")
        return bool(_UNARY_OPS[op](observed))

    value = pass_if.get('value', _MISSING)
    if value is _MISSING:
        raise PredicateError(f"op {op!r} requires a 'value'")
    return bool(_BINARY_OPS[op](observed, value))


def describe_predicate(pass_if: Mapping[str, Any]) -> str:
    """Render a predicate as a short human-readable requirement string.

    Used for the ``requirement`` line in an :class:`~discovery.hazards.
    DetectorResult` message and in a feasibility report's unmet list, so the
    stated requirement and the evaluated one can't disagree.
    """
    op = pass_if.get('op', '?')
    field = pass_if.get('field')
    lhs = field if field else 'value'
    if op in _UNARY_OPS:
        return f"{lhs} {op}"
    return f"{lhs} {op} {pass_if.get('value')!r}"
