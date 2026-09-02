"""
Executable checks backing the hazard registry (:mod:`discovery.hazards`).

Every function here is deliberately thin. All of the statistics lives in
:mod:`ephys._stats_utils` (``fdr_resolution``, ``empirical_p_value``,
``benjamini_hochberg``, ``majority_class_baseline``); these functions only
marshal a detector context into that module's arguments and apply a
threshold. A detector that reimplements a statistic will eventually disagree
with the analysis it is supposed to be guarding, at which point it is worse
than no detector.

Return convention
-----------------
Return a ``dict`` whose keys a hazard's ``pass_if`` can name, plus enough
context for the ``on_fail`` message template. Returning a bare ``bool`` is
allowed (the hazard then omits ``pass_if``) but a dict is preferred, because
the numbers behind a verdict are what make it actionable.

Assumptions:
    - **Raise :class:`~discovery._predicates.MissingValue` when the inputs
      needed to decide are absent**, rather than returning a pass or a fail.
      :func:`discovery.hazards.run_detector` converts that into
      ``ran=False, passed=None``. Guessing in either direction is worse: a
      guessed pass hides a real hazard, and a guessed fail trains the reader
      to ignore the layer.
    - **The repo-scanning detectors are ratchets, not audits.** They
      allowlist the call sites that exist and are understood today, and fire
      on new ones. They deliberately do not claim the allowlisted sites are
      correct — only that they were reviewed. See
      :func:`kfold_shuffle_audit`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from discovery._predicates import MissingValue

__all__ = [
    'REPO_ROOT',
    'passthrough',
    'accuracy_beats_baseline',
    'class_selection_is_prespecified',
    'pinned_at_p_floor',
    'target_position_variance_ok',
    'tracking_coverage_ok',
    'date_resolved_files_belong_to_recording',
    'param_equals',
    'param_is_explicit',
    'param_is_str',
    'resolved_from_dated_directory',
    'method_implemented',
    'repo_pattern_absent',
    'kfold_shuffle_audit',
]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise MissingValue(f"{name} is required but was None/absent")
    return value


def _as_float(value: Any, name: str) -> float:
    _require(value, name)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MissingValue(f"{name}={value!r} is not numeric: {exc}") from exc
    if not np.isfinite(out):
        raise MissingValue(f"{name}={value!r} is not finite")
    return out


# --------------------------------------------------------------- generic

def passthrough(value: Any = None) -> Any:
    """Return ``value`` unchanged.

    Lets a hazard express a pure manifest/param lookup — e.g. "this session
    must have >= 2 identity-resolved animals" — as a normal detector with a
    ``pass_if``, without inventing a one-off function per field. Raises
    :class:`MissingValue` for ``None`` so an absent manifest field is
    "cannot check", not "passes".
    """
    return _require(value, 'value')


# ---------------------------------------------------------- statistical

#: The decoder's raw result dict and the notebook's curated summary spell the
#: majority-class baseline differently, and this detector is used against both.
#: Reading only one key made it silently report "cannot check" on correctly-logged
#: iterations - on the very runs where the accuracy was below baseline.
_BASELINE_KEYS = ('population_baseline_accuracy', 'baseline_accuracy')
_ACCURACY_KEYS = ('population_accuracy_mean', 'accuracy', 'mean_accuracy')


def _first_present(source: Any, keys: Sequence[str]) -> Any:
    if not isinstance(source, Mapping):
        return None
    for key in keys:
        if source.get(key) is not None:
            return source[key]
    return None


def accuracy_beats_baseline(
    accuracy: Any = None,
    class_counts: Any = None,
    baseline: Any = None,
    margin: float = 0.0,
    results: Any = None,
) -> Dict[str, Any]:
    """Compare an observed accuracy against the *majority-class* baseline.

    ``1 / n_classes`` is prevalence-blind and understates the bar whenever
    classes are imbalanced: on this project's real 12-winner/7-loser split,
    always guessing the majority scores 63.2%, so a reported 60.6% was
    actually *below* naive guessing while looking comfortably "above chance".

    Supply either ``class_counts`` (a mapping label -> count, or a sequence of
    labels, from which the baseline is derived via
    :func:`ephys._stats_utils.majority_class_baseline`) or a precomputed
    ``baseline``. Usable at propose time from manifest counts alone, before
    anything runs.
    """
    from ephys._stats_utils import majority_class_baseline

    # Accept a whole result mapping and find the keys, since the raw result dict
    # and the curated summary use different names for the same quantities.
    if results is not None:
        if accuracy is None:
            accuracy = _first_present(results, _ACCURACY_KEYS)
        if baseline is None and class_counts is None:
            baseline = _first_present(results, _BASELINE_KEYS)

    acc = _as_float(accuracy, 'accuracy')

    if baseline is not None:
        base = _as_float(baseline, 'baseline')
    else:
        _require(class_counts, 'class_counts or baseline')
        if isinstance(class_counts, Mapping):
            counts = np.asarray([int(v) for v in class_counts.values()], dtype=np.int64)
            if counts.size == 0 or counts.sum() <= 0:
                raise MissingValue(f"class_counts is empty: {class_counts!r}")
            base = float(counts.max() / counts.sum())
        else:
            labels = np.asarray(list(class_counts))
            if labels.size == 0:
                raise MissingValue("class_counts sequence is empty")
            base = float(majority_class_baseline(labels))

    if not np.isfinite(base):
        raise MissingValue("majority-class baseline is not finite")

    return {
        'accuracy': acc,
        'baseline_accuracy': base,
        'margin': acc - base,
        'required_margin': float(margin),
        'beats_baseline': bool(acc > base + float(margin)),
    }


def pinned_at_p_floor(p_value: Any, n_shuffles: Any, tolerance: float = 1e-9) -> Dict[str, Any]:
    """Is a permutation p-value sitting exactly on its own floor?

    A permutation test with ``n_shuffles`` draws cannot report a p below
    ``1 / (n_shuffles + 1)``. A result *at* that floor means "no shuffle beat
    the observed value" — which is consistent with a huge effect and equally
    consistent with a budget too small to resolve anything. It is a lower
    bound masquerading as a measurement, so it is not, by itself, a finding.

    This fired for real on iteration 12: ``rat613``'s q=0.0387 came from a
    p pinned at 1/181 with ``n_shuffles=180``.
    """
    p = _as_float(p_value, 'p_value')
    n = int(_as_float(n_shuffles, 'n_shuffles'))
    if n < 1:
        raise MissingValue(f"n_shuffles={n} is not a usable permutation budget")
    floor = 1.0 / (n + 1)
    pinned = bool(p <= floor + tolerance)
    return {
        'p_value': p,
        'n_shuffles': n,
        'p_floor': floor,
        'pinned': pinned,
        'not_pinned': not pinned,
    }


# ----------------------------------------------------------------- data

def target_position_variance_ok(
    x_std: Any,
    y_std: Any,
    min_std: float = 40.0,
    units: str = 'px',
) -> Dict[str, Any]:
    """Does a decoding target actually move enough for the error to mean anything?

    Any position decoder gets near-zero error on a target that barely moves,
    because predicting the mean is already correct. This produced a spurious
    partner-position "win" in this project for an animal whose x/y standard
    deviations were ~11-13 px.

    ``min_std`` is a deliberately blunt threshold in tracking units; the point
    is to catch the degenerate case, not to grade mobility.
    """
    sx = _as_float(x_std, 'x_std')
    sy = _as_float(y_std, 'y_std')
    thresh = float(min_std)
    return {
        'x_std': sx,
        'y_std': sy,
        'min_std': thresh,
        'units': units,
        'smaller_std': min(sx, sy),
        'moves_enough': bool(min(sx, sy) >= thresh),
    }


def tracking_coverage_ok(
    frac_covered: Any,
    min_frac: float = 0.8,
    ephys_window: Any = None,
) -> Dict[str, Any]:
    """Does the tracking file span enough of the recording to analyse it whole?

    Tracking coverage is a time window, not a boolean. Session ``20251216``'s
    tracking file (``merged_20251216_0950_1200``) covers roughly 63% of a
    recording that starts at 09:43:34, and both
    :func:`video.tracking_import.resolve_tracking_on_ephys_clock` and
    :func:`ephys.social_spatial_fields.compute_social_place_fields` accept a
    time window that defaults to "everything". An analysis run over the full
    recording therefore mixes no-position time into its occupancy maps
    silently.
    """
    frac = _as_float(frac_covered, 'frac_covered')
    return {
        'frac_covered': frac,
        'min_frac': float(min_frac),
        'ephys_window': list(ephys_window) if ephys_window is not None else None,
        'covers_enough': bool(frac >= float(min_frac)),
    }


def date_resolved_files_belong_to_recording(
    attachment_status: Any = None,
    is_primary: Any = None,
    recording_ids_on_date: Any = None,
) -> Dict[str, Any]:
    """Do this recording's date-resolved tracking/events actually cover it?

    Tracking files and behavioural event files resolve by 8-digit date, but a
    date is not a recording. ``20251216_094334.rec/rat613/`` holds three
    recordings — 09:43, 14:43 and 19:43 — and the day's single tracking file
    spans 09:50-12:00. All three are offered that file; for two of them it
    maps outside the recording entirely once put on their clock.

    Passes when the capability manifest has *verified* the overlap
    (``attachment_status='overlap_verified'``), or when only one recording
    exists on the date and there is nothing to confuse. An unverified
    attachment on a day with several recordings is a fail, not a pass: the
    file may well be the right one, but nothing has checked, and the wrong
    one produces a plausible rate map rather than an error.
    """
    status = attachment_status
    on_date = list(recording_ids_on_date or [])

    if status is None and not on_date and is_primary is None:
        raise MissingValue(
            'attachment_status, is_primary and recording_ids_on_date are all '
            'absent; cannot tell which recording these files belong to')

    single_recording_on_date = len(on_date) == 1
    verified = status == 'overlap_verified'
    return {
        'attachment_status': status,
        'is_primary': is_primary,
        'recording_ids_on_date': on_date,
        'n_recordings_on_date': len(on_date),
        'single_recording_on_date': single_recording_on_date,
        'attachment_verified': verified,
        'belongs_to_recording': bool(verified or single_recording_on_date),
    }


# ------------------------------------------------------------------ api

def param_equals(value: Any, expected: Any) -> Dict[str, Any]:
    """Assert a parameter was set to a specific value.

    Used for hazards where a *default* is the trap — e.g.
    :mod:`ephys.decode_location`'s ``null='reverse'``, which per its own
    docstring only reverses trajectory order and therefore preserves every
    marginal and the autocorrelation. It tests temporal-order asymmetry, not
    "beats chance", and a claim needs ``null='shuffle'``.
    """
    _require(value, 'value')
    return {'value': value, 'expected': expected, 'matches': bool(value == expected)}


def param_is_explicit(value: Any = None, sentinel: Any = None) -> Dict[str, Any]:
    """Assert a parameter was passed rather than left at a permissive default.

    ``decode_event_outcome_*`` defaults ``behavior_type=None``, which includes
    every event where both ``winner`` and ``loser`` are populated — not just
    fights. That is a legitimate analysis and a silent one; the hazard exists
    so the choice is stated.
    """
    return {'value': value, 'is_explicit': bool(value is not sentinel)}


def param_is_str(value: Any) -> Dict[str, Any]:
    """Assert a parameter is a ``str``.

    ``video.behavioral_events.extract_opponent_labels`` calls
    ``.startswith('rat')`` on ``animal_of_interest`` directly. Only the CLI's
    ``argparse(type=str)`` enforces this; an in-process caller passing an
    ``int`` fails deep inside label extraction with a confusing
    ``'int' object has no attribute 'startswith'``. This happened (logged as
    a failed iteration).
    """
    _require(value, 'value')
    return {'value': value, 'type': type(value).__name__, 'is_str': isinstance(value, str)}


#: Parameter names by which an analysis narrows its own class set.
_CLASS_SUBSET_KEYS = ('selected_opponents', 'selected_classes', 'selected_objects')


def class_selection_is_prespecified(
    params: Mapping[str, Any],
    min_floor: int = 5,
) -> Dict[str, Any]:
    """Did this run narrow its own class set, or relax the event floor?

    Two moves make a decoding result easier without making it truer, and both
    are invisible in the result dict:

    1. Hand-picking a subset of classes (``selected_opponents=['rat634',
       'rat635']``) turns an 8-way problem into a 2-way one and raises the
       apparent accuracy while raising the baseline less.
    2. Lowering ``min_events_per_class`` below the repo's own default of 5
       admits classes with too few events to cross-validate.

    Neither is illegitimate; both are forking paths. This detector exists so
    that they have to be *declared* as part of a test family before running,
    rather than discovered afterwards from a params dict.

    Takes the whole params mapping rather than individual keys, because
    "absent" is the safe value here and a per-key reference would report
    "cannot check" for the common well-behaved case.
    """
    if params is None:
        raise MissingValue("params mapping is required")
    if not isinstance(params, Mapping):
        raise MissingValue(f"params must be a mapping, got {type(params).__name__}")

    floor = params.get('min_events_per_class')
    subset_key = next((k for k in _CLASS_SUBSET_KEYS if params.get(k)), None)
    selected = list(params.get(subset_key) or ()) if subset_key else []

    floor_ok = floor is None or int(floor) >= int(min_floor)
    return {
        'min_events_per_class': floor,
        'min_floor': int(min_floor),
        'floor_ok': bool(floor_ok),
        'subset_key': subset_key,
        'selected_classes': selected,
        'n_selected': len(selected),
        'narrowed_class_set': bool(selected),
        'is_prespecified': bool(floor_ok and not selected),
    }


_DATED_DIR_RE = re.compile(r'^\D*(20\d{6})')


def resolved_from_dated_directory(event_files: Any) -> Dict[str, Any]:
    """Is the resolved behavioural-event file inside a date-named directory?

    The cohort-7 events root holds three versions of session 20251216's
    scoring: ``20251216/20251216_behavior_event_df.csv`` (2026-03-02, 641
    rows) and two loose files one level up (2026-08-13, 688 and 693 rows,
    with 29 rather than 19 usable outcome events for animal 631).
    ``DataStorageManager`` only looks inside date-named directories, so it
    resolves the dated one and the newer loose files are invisible to it.

    The dated directory is canonical by decision. This detector exists so that
    nobody silently starts reading a loose file — the row counts differ by
    enough to change a conclusion, and the two conventions must not be mixed
    within one analysis.
    """
    if event_files is None:
        raise MissingValue('event_files is required')
    files = [event_files] if isinstance(event_files, (str, Path)) else list(event_files)
    if not files:
        raise MissingValue('no behavioural event files resolved')

    dated, loose = [], []
    for raw in files:
        path = Path(str(raw))
        (dated if _DATED_DIR_RE.match(path.parent.name) else loose).append(path.as_posix())

    return {
        'dated_files': dated,
        'loose_files': loose,
        'n_files': len(files),
        'all_from_dated_directory': not loose,
    }


def method_implemented(module: str, attr: str) -> Dict[str, Any]:
    """Is a named function actually implemented in a module?

    Guards hypotheses that assume a method exists. Multi-set CCA for N>2
    brains is a documented stretch goal, not code: the 4-animal data exists
    for sessions already worked with, so the blocker is missing math, and a
    hypothesis proposing it should fail at generation time rather than at
    implementation time.
    """
    import importlib
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        raise MissingValue(f"cannot import {module!r}: {exc}") from exc
    present = hasattr(mod, attr)
    return {'module': module, 'attr': attr, 'implemented': bool(present)}


# --------------------------------------------------------- repo scanning

# The analysis packages only. `discovery/` is excluded deliberately: this
# module necessarily contains the very patterns it searches for (in its
# regexes and its docstrings), so including it would make every scanning
# detector self-match and report a permanent false positive.
_PY_GLOBS = ('ephys/*.py', 'ingestion/*.py', 'video/*.py', 'database/*.py')


def _iter_repo_py(repo_root: Optional[Path] = None) -> Iterable[Path]:
    root = Path(repo_root) if repo_root else REPO_ROOT
    for pattern in _PY_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.name.startswith('_test'):
                continue
            yield path


def repo_pattern_absent(
    pattern: str,
    repo_root: Optional[Path] = None,
    allow_files: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assert a literal substring appears nowhere in the analysis packages.

    Guards against a stale API creeping back in — e.g.
    ``KilosortData(data_input=`` , the pre-dataclass constructor whose callers
    were all migrated and for which back-compat shims are explicitly
    unwanted.
    """
    allowed = {str(a).replace('\\', '/') for a in (allow_files or ())}
    hits: List[str] = []
    for path in _iter_repo_py(repo_root):
        rel = path.relative_to(Path(repo_root) if repo_root else REPO_ROOT).as_posix()
        if rel in allowed:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if pattern in text:
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    hits.append(f"{rel}:{i}")
    return {
        'pattern': pattern,
        'hits': hits,
        'n_hits': len(hits),
        'absent': not hits,
    }


#: Files whose ``shuffle=True`` cross-validation has been reviewed and has a
#: documented rationale. Allowlisted at *file* granularity on purpose: line
#: numbers rot on the first unrelated edit, and a stale allowlist silently
#: stops guarding.
#:
#: ``ephys/_lda_decoding.py`` — ``StratifiedKFold(shuffle=True,
#: random_state=42)`` operates on one row per *behavioral event*, and events
#: are seconds-to-minutes apart, so it is not the adjacent-bin leakage that
#: was fixed in ``decode_location``. The fixed ``random_state`` across the
#: observed run and every permutation is also correct — it makes the
#: comparison paired. The genuine residual risk is narrower (bouts of events
#: clustered in time, which would want grouped folds keyed on event time) and
#: changing it would move a published result, so it is a scientist decision.
#:
#: Keep this list minimal. :func:`kfold_shuffle_audit` reports entries that no
#: longer match anything as ``stale_allowlist_entries``: a stale entry means
#: either the site was fixed (delete the entry) or the file moved (update it),
#: and either way the allowlist has stopped guarding what it claims to.
_KFOLD_SHUFFLE_ALLOWLIST = (
    'ephys/_lda_decoding.py',
)

_KFOLD_SHUFFLE_RE = re.compile(r'(?:Stratified)?(?:Group)?KFold\s*\([^)]*shuffle\s*=\s*True', re.S)


def kfold_shuffle_audit(
    repo_root: Optional[Path] = None,
    allow_files: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Find cross-validation splitters constructed with ``shuffle=True``.

    Shuffled folds leak whenever adjacent samples are near-identical draws
    from an autocorrelated signal. This was a real bug in
    :mod:`ephys.decode_location`, whose ``_cv_decode`` shuffled 0.5 s position
    bins; fixing it to contiguous folds moved self-position decoding from
    p=0.43 to p=0.138 (both the real and null errors grew and became more
    honest — the leakage had been inflating both comparably, hiding a real
    effect under two artificially tight numbers).

    This is a **ratchet, not an audit**: it allowlists the sites that exist
    and are understood today (see :data:`_KFOLD_SHUFFLE_ALLOWLIST` for the
    per-file rationale) and fires on new ones. It does not assert the
    allowlisted sites are correct — only that they were reviewed. New code on
    a time-ordered axis should use ``KFold(shuffle=False)``, matching
    :mod:`ephys.decode_partner_distance` and
    :func:`ephys.inter_brain_dynamics._fit_r2`.
    """
    allowed = set(_KFOLD_SHUFFLE_ALLOWLIST)
    allowed.update(str(a).replace('\\', '/') for a in (allow_files or ()))

    root = Path(repo_root) if repo_root else REPO_ROOT
    reviewed: List[str] = []
    unapproved: List[str] = []
    for path in _iter_repo_py(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if not _KFOLD_SHUFFLE_RE.search(text):
            continue
        (reviewed if rel in allowed else unapproved).append(rel)

    stale = sorted(allowed - set(reviewed))
    return {
        'reviewed_sites': reviewed,
        'unapproved_sites': unapproved,
        'n_unapproved': len(unapproved),
        'no_unapproved_sites': not unapproved,
        'allowlist': sorted(allowed),
        'stale_allowlist_entries': stale,
        'allowlist_is_current': not stale,
    }
