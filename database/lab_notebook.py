"""
Lab notebook — the append-only provenance/rigor layer for the AI-in-the-loop
discovery platform described in ``docs/AI_DISCOVERY_LOOP_DESIGN.md`` (§6).

Extends the existing ``database/`` layer (imports ``Base`` from
``database_core`` so its tables land in the same ``habitat_pipeline.db``,
registered on the same declarative metadata) without touching
``database_core.py``'s ``Animal``/``ExperimentSession``/``DataFile`` models
at all.

Two distinct multiple-comparison problems are handled at two different
granularities:

- **Within one analysis run**, many cells get tested at once — corrected by
  ``ephys._lda_decoding.compute_population_significance`` (per-cell BH-FDR).
- **Across a campaign**, many analysis iterations get run against one
  dataset — corrected here, per :class:`TestFamily`, via
  :meth:`LabNotebook.recompute_family_significance`.

Holdout registry
----------------
:class:`HoldoutReservation` plus :meth:`LabNotebook.assert_not_held_out` are
the enforcement surface for the design doc's §5 held-out-session rule. They
replace an earlier check that could not work: ``Iteration.held_out`` is a
column on an *iteration*, so the prescribed test — look for iterations on
this session — returns nothing for a session that has never been touched, and
passes. That gate could only fire *after* it had already been violated, and
only if the violator flagged their own violation. The registry inverts it:
reservations are created by a human act that is independent of, and prior to,
any analysis, so "never touched" is exactly the state it protects.

Assumptions:
    - **Session keys normalize to the 8-digit date.** Path resolution in
      ``ingestion/data_paths.py`` is substring-based, and this database
      stores ``'20251210'`` while the directory on disk is
      ``RatCity_20251210_1359_40Hz.rec``. A holdout check keyed on exact
      string equality would silently fail to match the very id an analysis
      actually passes.
    - **An unresolvable session id fails closed.** If a query cannot be
      normalized to a date, :meth:`LabNotebook.holdout_status` raises
      :class:`HoldoutIndeterminate` rather than reporting "not held out".
      This is the single most important behavioural difference from the gate
      it replaces.
    - **Animal-scoped reservations block the whole session for multi-animal
      analyses.** Most cohort-7 sessions have four simultaneously implanted
      animals, and ``decode_location``, ``run_inter_brain``,
      ``social_spatial_fields`` and ``decode_partner_distance`` all read a
      partner's data through a focal animal. Reserving one animal while
      running another therefore holds out nothing. Whole-session scope is the
      default for that reason.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    create_engine, text as sa_text,
)
from sqlalchemy.orm import Session, relationship, sessionmaker

from database.database_core import Base


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays (common in result-dicts) to
    plain JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    return obj


def _to_json(obj: Any) -> str:
    return json.dumps(_json_safe(obj))


class HoldoutViolation(RuntimeError):
    """An analysis targeted a session reserved as held-out.

    Raised by :meth:`LabNotebook.assert_not_held_out`. Deliberately an
    exception rather than a warning: a holdout that can be ignored is not a
    holdout, and the whole value of the reserved set is that the loop has
    provably never seen it.
    """


class HoldoutIndeterminate(RuntimeError):
    """A session id could not be resolved well enough to check the registry.

    Fails closed. "I could not tell whether this is held out" must never be
    reported as "this is not held out" — that is precisely how the gate this
    replaces let everything through.
    """


class UndeclaredTestError(RuntimeError):
    """A result was recorded for a test that was never declared.

    The ledger refuses it rather than adding it, because a family that grows
    as results arrive is not a denominator — it is a running total of what
    happened to work.
    """


class TestBudgetExhausted(RuntimeError):
    """Recording this test would exceed the family's declared size.

    Not a compute quota. Its function is to make the denominator a decision
    taken before results are seen, and extending it an explicit, attributable
    act rather than a silent drift.
    """


_SESSION_DATE_RE = re.compile(r'(20\d{6})')

#: Public entry point per analysis module, used by :func:`canonical_params` to
#: fill in defaults the caller left implicit. Only affects run-identity
#: hashing, so an entry going stale degrades duplicate detection rather than
#: corrupting anything.
_ANALYSIS_ENTRY_POINTS = {
    'ephys.decode_opponent_identity':
        'ephys.decode_opponent_identity:decode_opponent_identity_population',
    'ephys.decode_event_outcome':
        'ephys.decode_event_outcome:decode_event_outcome_population',
    'ephys.decode_location': 'ephys.decode_location:decode_all_locations',
    'ephys.social_spatial_fields':
        'ephys.social_spatial_fields:compute_social_place_fields',
    'ephys.decode_partner_distance':
        'ephys.decode_partner_distance:decode_partner_distance',
}

#: Params that describe *where* a run wrote things rather than *what* it
#: computed. Excluded from the run identity so two identical analyses whose
#: outputs went to different directories still hash equal.
_NON_IDENTITY_PARAMS = frozenset({
    'output_dir', 'out_dir', 'save_plots', 'show_plots', 'verbose', 'note',
    'notes', 'figure_dir', 'config_path',
})


def canonical_params(analysis_module: str, params: Dict) -> Dict:
    """Normalize a params dict into a comparable form for run identity.

    Two runs of the same analysis can log different param *keys* with
    identical param *values* — one caller passes ``cv_folds=5`` explicitly
    while another leaves it at its default — and then look like two tests
    when they are one. This fills in defaults from the entry point's
    signature so the two collapse.

    Best-effort by design: if the module can't be imported or isn't in
    :data:`_ANALYSIS_ENTRY_POINTS`, the supplied params are canonicalized on
    their own. That weakens duplicate detection and corrupts nothing.
    """
    import importlib
    import inspect

    merged: Dict[str, Any] = {}
    entry = _ANALYSIS_ENTRY_POINTS.get(analysis_module)
    if entry:
        module_name, _, func_name = entry.partition(':')
        try:
            func = getattr(importlib.import_module(module_name), func_name)
            for name, parameter in inspect.signature(func).parameters.items():
                if parameter.default is not inspect.Parameter.empty:
                    merged[name] = parameter.default
        except Exception:
            pass  # best-effort; fall through to the supplied params alone

    merged.update({k: v for k, v in (params or {}).items()})
    return {
        k: _json_safe(v) for k, v in sorted(merged.items())
        if k not in _NON_IDENTITY_PARAMS
    }


def compute_run_key(analysis_module: str, params: Dict, *, git_commit: str = None,
                    dataset_fingerprint: str = None) -> str:
    """Stable hash identifying "the same test run again".

    Includes the code version and the dataset fingerprint, because the same
    parameters against changed code or changed data are genuinely a different
    test — iterations 11 and 12 of this project's own notebook are the same
    declared family evaluated under a leaky and a fixed cross-validation
    splitter, and mixing them in one correction would be incoherent.
    """
    import hashlib

    payload = json.dumps({
        'module': analysis_module,
        'params': canonical_params(analysis_module, params),
        'git_commit': git_commit,
        'dataset_fingerprint': dataset_fingerprint,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def normalize_session_key(session_id: Any) -> Optional[str]:
    """Canonical holdout key for a session: the ``YYYYMMDD`` date inside it.

    Path resolution in :mod:`ingestion.data_paths` matches directories by
    substring and parses dates as ``session_id[:8]``, so one recording is
    referred to by several strings across this codebase::

        '20251210'                        # what the lab notebook stores
        'RatCity_20251210_1359_40Hz'      # the directory on disk
        'RatCity_20251210_1359_40Hz.rec'  # with the suffix
        '20251210_094334'                 # ephys session id form

    All four must resolve to the same reservation. Returns ``None`` when no
    date can be found, which callers must treat as indeterminate rather than
    as absent.
    """
    if session_id is None:
        return None
    match = _SESSION_DATE_RE.search(str(session_id))
    return match.group(1) if match else None


def normalize_animal_key(animal_id: Any) -> Optional[str]:
    """Canonical animal key: the trailing digits of an id.

    This codebase refers to the same animal as ``'631'``, ``'rat631'``, and
    ``631`` in different places (the DB stores ``'631'``, tracking object
    names are ``'rat631'``, and ``.cache/data_paths`` contains both forms).
    A holdout check keyed on the raw string would match one form and miss the
    others.
    """
    if animal_id is None:
        return None
    match = re.search(r'(\d+)\s*$', str(animal_id))
    return match.group(1) if match else str(animal_id).strip() or None


def _same_animal(left: Any, right: Any) -> bool:
    left_key, right_key = normalize_animal_key(left), normalize_animal_key(right)
    return left_key is not None and left_key == right_key


#: Columns added to pre-existing tables after their first release, as
#: ``{table: [(column_name, sqlite_type), ...]}``.
#:
#: These cannot be handled by ``Base.metadata.create_all``, which creates
#: *missing tables* and never ``ALTER``s an existing one. Declaring a new
#: ``Column`` on :class:`Iteration` puts it in the ORM and not in the live
#: database file, so every query touching it raises
#: ``OperationalError: no such column``. New *tables* need no entry here.
_ADDED_COLUMNS: Dict[str, List[Tuple[str, str]]] = {
    'iterations': [
        ('tier', 'VARCHAR(20)'),
        ('frozen_prediction_id', 'INTEGER'),
        ('seed', 'VARCHAR(64)'),
        ('dataset_fingerprint', 'VARCHAR(64)'),
        ('fingerprint_method', 'VARCHAR(40)'),
    ],
    'test_families': [
        ('declared_n_tests', 'INTEGER'),
        ('budget_max_tests', 'INTEGER'),
        ('status', 'VARCHAR(20)'),
        ('superseded_by_id', 'INTEGER'),
        ('denominator_status', 'VARCHAR(20)'),
    ],
}


def _ensure_added_columns(engine) -> List[str]:
    """Idempotently add any missing :data:`_ADDED_COLUMNS`; return what it added.

    Must run *after* ``create_all``. A table that ``create_all`` just created
    already has every ORM column, so this is a no-op for fresh databases and
    a one-time migration for the existing one.

    Every added column is nullable with no default, on purpose: rows written
    before the column existed then read as ``NULL`` meaning "not recorded",
    which is the honest value. A ``DEFAULT False`` would retroactively assert
    something about those rows that nobody actually checked.

    SQLite's ``ADD COLUMN`` is O(1) and cannot lose data. It also cannot add a
    foreign-key constraint, which is why ``frozen_prediction_id`` is a plain
    ``INTEGER`` rather than a declared ``ForeignKey``. There is no Alembic in
    this repo and this helper exists so there needn't be.
    """
    added: List[str] = []
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            # Table/column names come from the module-level constant above,
            # never from caller input, so interpolation here is not an
            # injection surface. PRAGMA does not accept bound parameters.
            rows = conn.execute(sa_text(f"PRAGMA table_info({table})")).fetchall()
            if not rows:
                continue  # table absent; create_all will build it complete
            existing = {row[1] for row in rows}
            for name, sqlite_type in columns:
                if name not in existing:
                    conn.execute(sa_text(f"ALTER TABLE {table} ADD COLUMN {name} {sqlite_type}"))
                    added.append(f"{table}.{name}")
    return added


def _git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """Best-effort git commit of the code that ran; ``None`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


class TestFamily(Base):
    """Groups analysis iterations for campaign-level multiple-comparison
    accounting (design doc §5: "the notebook counts every test the agent
    runs in a campaign")."""

    __tablename__ = 'test_families'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    correction_method = Column(String(20), default='bh_fdr')  # 'bh_fdr' | 'bonferroni'
    alpha = Column(Float, default=0.05)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    # --- added after first release; see _ADDED_COLUMNS ---
    #: Family size fixed *before* any result is seen. This is the denominator
    #: BH must divide by. Kept separate from "how many actually ran" so that
    #: shrinking the family after the fact cannot silently improve q-values.
    declared_n_tests = Column(Integer)
    #: Declared family size cap. Not a compute quota — its function is to fix
    #: the denominator up front. See ``LabNotebook.record_family_test``.
    budget_max_tests = Column(Integer)
    #: 'open' | 'closed' | 'superseded' | 'invalidated'
    status = Column(String(20))
    superseded_by_id = Column(Integer)
    #: 'clean' | 'reconstructed' | 'pipeline_changed' | 'outcome_dependent_exclusions'
    denominator_status = Column(String(20))

    iterations = relationship("Iteration", back_populates="test_family")

    def __repr__(self):
        return f"<TestFamily(name={self.name!r}, n_iterations={len(self.iterations)})>"


class Hypothesis(Base):
    """A pre-registered hypothesis (design doc §5's held-out/pre-registration
    concept; §6's "hypothesis text" captured per iteration)."""

    __tablename__ = 'hypotheses'

    id = Column(Integer, primary_key=True)
    statement = Column(Text, nullable=False)
    predicted_effect = Column(Text)
    chosen_test = Column(Text)
    citations = Column(Text)  # JSON-encoded list[{source, id, title, url}]
    status = Column(String(20), default='proposed')  # proposed|approved|rejected|confirmed
    scientist_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    iterations = relationship("Iteration", back_populates="hypothesis")

    def __repr__(self):
        return f"<Hypothesis(id={self.id}, status={self.status})>"


class Iteration(Base):
    """One logged run of the discovery loop — design doc §6's "one genuinely
    new artifact": hypothesis, params, code version, dataset/session,
    result, figures, test family, and the scientist's decision."""

    __tablename__ = 'iterations'

    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey('hypotheses.id'), nullable=True)
    test_family_id = Column(Integer, ForeignKey('test_families.id'), nullable=True)

    animal_id = Column(String(50))
    session_id = Column(String(50))
    held_out = Column(Boolean, default=False)

    analysis_module = Column(String(200), nullable=False)
    params = Column(Text, nullable=False)  # JSON-encoded dict
    git_commit = Column(String(40))

    status = Column(String(20))
    result_summary = Column(Text)  # JSON-encoded dict (curated, not the raw result-dict)
    figure_paths = Column(Text)  # JSON-encoded list[str]

    scientist_decision = Column(String(20), default='pending')  # pending|approved|rejected
    decision_notes = Column(Text)
    decision_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)

    # --- added after first release; see _ADDED_COLUMNS ---
    #: 'exploratory' | 'confirmatory'. Descriptive only: the authoritative
    #: answer is derived by ``LabNotebook.evidence_tier``, which recomputes it
    #: from preconditions so that nobody can promote a result by editing a
    #: field.
    tier = Column(String(20))
    #: Plain Integer, not a ForeignKey: SQLite's ALTER TABLE ADD COLUMN cannot
    #: add a constraint, and this column is retrofitted.
    frozen_prediction_id = Column(Integer)
    #: Recorded as a string so "0" and "not recorded" stay distinguishable.
    #: NULL means the module hardcodes its seed rather than accepting one.
    seed = Column(String(64))
    dataset_fingerprint = Column(String(64))
    fingerprint_method = Column(String(40))

    hypothesis = relationship("Hypothesis", back_populates="iterations")
    test_family = relationship("TestFamily", back_populates="iterations")

    def result_summary_dict(self) -> Dict:
        return json.loads(self.result_summary) if self.result_summary else {}

    def params_dict(self) -> Dict:
        return json.loads(self.params) if self.params else {}

    def figure_paths_list(self) -> List[str]:
        return json.loads(self.figure_paths) if self.figure_paths else []

    def __repr__(self):
        return (f"<Iteration(id={self.id}, module={self.analysis_module}, "
                f"session={self.session_id}, decision={self.scientist_decision})>")


class FrozenPrediction(Base):
    """A prediction recorded *before* the data were seen.

    The surface behind "no post-hoc redefinition of a statistic without a new
    frozen record". There is deliberately **no update method** anywhere in
    this module: changing a statistic, a threshold, or a family means
    inserting a new row and setting ``superseded_by_id`` on the old one, so
    the sequence of things that were predicted stays legible. A mutable
    pre-registration is not a pre-registration.

    ``registered_post_hoc=True`` marks a record transcribed from an analysis
    that had already run. Such a record is useful — it says what was claimed —
    but it is contagious: it can never satisfy the confirmatory tier, and the
    report is required to say the prediction was written after the fact.
    """

    __tablename__ = 'frozen_predictions'

    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey('hypotheses.id'), nullable=False)
    version = Column(Integer, default=1)

    #: Dotted path into the iteration's ``result_summary``, or one of the
    #: ledger-derived names ``'min_q_value'`` / ``'q_value'``.
    statistic = Column(Text, nullable=False)
    direction = Column(String(4), nullable=False)  # 'lt' | 'gt'
    threshold = Column(Float, nullable=False)
    alpha = Column(Float, default=0.05)
    n_shuffles_planned = Column(Integer)

    test_family_id = Column(Integer, ForeignKey('test_families.id'))
    declared_test_keys = Column(Text)  # JSON list
    n_tests_declared = Column(Integer)

    holdout_required = Column(Boolean, default=True)
    #: 'replication' (same animals/conditions) vs 'generalization' (different
    #: animals). A failed generalization test is *not* a refutation of the
    #: original claim, and conflating them misreports both.
    holdout_kind = Column(String(20), default='replication')

    #: What result would count against this. Required: a prediction with no
    #: falsifier is not testable, and an agent-authored falsifier for someone
    #: else's hypothesis is not a falsifier.
    falsifier = Column(Text, nullable=False)

    registered_post_hoc = Column(Boolean, nullable=False, default=False)
    spec_hash = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)

    superseded_by_id = Column(Integer, ForeignKey('frozen_predictions.id'))
    supersede_reason = Column(Text)

    def declared_test_keys_list(self) -> List[str]:
        return json.loads(self.declared_test_keys) if self.declared_test_keys else []

    def __repr__(self):
        tag = ' POST-HOC' if self.registered_post_hoc else ''
        return (f"<FrozenPrediction(id={self.id}, hyp={self.hypothesis_id}, "
                f"v{self.version}, {self.statistic} {self.direction} "
                f"{self.threshold}{tag})>")


class HypothesisVerdict(Base):
    """An append-only call on a hypothesis, refutations included.

    ``'refuted'`` is a first-class outcome, not an absence of one. A hypothesis
    that quietly stops being mentioned leaves no row here, which is exactly
    the failure mode the index report exists to surface.
    """

    __tablename__ = 'hypothesis_verdicts'

    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(Integer, ForeignKey('hypotheses.id'), nullable=False)
    tier = Column(String(20), nullable=False)  # 'exploratory' | 'confirmatory'
    #: supported | refuted | inconclusive | hypothesis_generating_only | blocked
    verdict = Column(String(40), nullable=False)

    holdout_iteration_id = Column(Integer, ForeignKey('iterations.id'))
    denominator_known = Column(Boolean, default=False)
    n_tests_in_denominator = Column(Integer)

    rationale = Column(Text, nullable=False)
    decided_by = Column(String(20), default='scientist')  # 'scientist' | 'agent'
    created_at = Column(DateTime, default=datetime.utcnow)
    supersedes_id = Column(Integer, ForeignKey('hypothesis_verdicts.id'))

    def __repr__(self):
        return (f"<HypothesisVerdict(hyp={self.hypothesis_id}, {self.tier}/"
                f"{self.verdict})>")


class FamilyTest(Base):
    """One *declared* test — the ledger's atomic unit.

    The distinction from :class:`Iteration` is the whole point. An iteration
    is a run that happened; a declared test exists from the moment the family
    is declared, survives being abandoned, and is counted in the denominator
    whether or not it ever produced a number. Without that, "how many tests
    were behind this claim" is unanswerable after the fact, because tests that
    were dropped leave no trace.

    One iteration can hold many tests: ``decode_all_locations`` sweeps every
    tracked object and returns them in a single result dict, which is how a
    12-test sweep came to be recorded as one row whose q-values had been
    computed internally against a caller-chosen denominator.
    """

    __tablename__ = 'family_tests'

    id = Column(Integer, primary_key=True)
    test_family_id = Column(Integer, ForeignKey('test_families.id'), nullable=False)
    #: Stable identifier for *what* is being tested, e.g. ``'object=rat613'``
    #: or ``'opponent:EC:8way:pooled'``. Declared before running.
    test_key = Column(String(200), nullable=False)

    #: 'declared' | 'run' | 'failed' | 'abandoned' | 'excluded_prespecified'
    status = Column(String(30), nullable=False, default='declared')
    declared_at = Column(DateTime, default=datetime.utcnow)
    declared_by = Column(String(100))

    iteration_id = Column(Integer, ForeignKey('iterations.id'))
    #: sha256 over (module, canonical params, git commit, dataset fingerprint).
    #: Equal run_key means the same test was run twice, not two tests.
    run_key = Column(String(64))
    p_value = Column(Float)
    git_commit = Column(String(40))

    exclusion_reason = Column(Text)
    #: Required, not defaulted, when abandoning. The single field that makes a
    #: post-hoc family shrink impossible to perform silently.
    exclusion_outcome_dependent = Column(Boolean)
    #: Could the exclusion criterion have been evaluated without running
    #: anything? (A target's positional variance could; "showed no margin"
    #: could not.)
    criterion_available_a_priori = Column(Boolean)
    applied_after_seeing_results = Column(Boolean)

    rerun_of_id = Column(Integer, ForeignKey('family_tests.id'))
    notes = Column(Text)

    def __repr__(self):
        return (f"<FamilyTest(family={self.test_family_id}, key={self.test_key!r}, "
                f"status={self.status}, p={self.p_value})>")


class HoldoutReservation(Base):
    """A session (or animal within one) the discovery loop must not touch.

    Populated by an explicit human act — ``notebook_cli reserve-holdout`` —
    which is what makes it a valid gate: the reservation exists before any
    analysis, so it protects a session that has never been run rather than
    recording that one already was.

    Releases never delete. A released reservation stops blocking but stays on
    the record, because "this was held out until we spent it" and "this was
    never held out" are different histories and the difference is the whole
    basis for calling a result confirmatory.
    """

    __tablename__ = 'holdout_reservations'

    id = Column(Integer, primary_key=True)
    cohort = Column(String(50), nullable=False)
    #: Normalized 8-digit date; see :func:`normalize_session_key`.
    session_key = Column(String(20), nullable=False)
    #: The resolved directory name, when known. Recorded for provenance only —
    #: matching is always on ``session_key``, never on this.
    session_dir = Column(String(200))
    #: NULL for a whole-session reservation.
    animal_id = Column(String(50))
    scope = Column(String(20), nullable=False, default='session')  # 'session' | 'animal'

    reason = Column(Text, nullable=False)
    reserved_by = Column(String(100), nullable=False)
    reserved_at = Column(DateTime, default=datetime.utcnow)

    released_at = Column(DateTime)
    released_by = Column(String(100))
    release_reason = Column(Text)

    #: Set by :meth:`LabNotebook.unlock_holdout`. One unlock admits exactly
    #: one hypothesis, and only for a confirmatory run.
    unlocked_for_hypothesis_id = Column(Integer)
    unlocked_frozen_prediction_id = Column(Integer)
    unlocked_at = Column(DateTime)
    unlocked_by = Column(String(100))

    @property
    def active(self) -> bool:
        return self.released_at is None

    def __repr__(self):
        state = 'active' if self.active else 'released'
        target = self.animal_id or 'whole session'
        return (f"<HoldoutReservation(id={self.id}, {self.cohort}/{self.session_key}, "
                f"{target}, {state})>")


@dataclass(frozen=True)
class HoldoutStatus:
    """Result of a holdout check — a reason, not a bare boolean.

    A caller that gets ``held_out=True`` needs to know *why* and *by which
    reservation* in order to say anything useful, and a caller that gets
    ``False`` needs to know the registry was actually consulted.
    """

    session_id: str
    session_key: str
    held_out: bool
    reason: Optional[str] = None
    reservation_ids: Tuple[int, ...] = ()
    scope: Optional[str] = None
    unlocked_for_hypothesis_id: Optional[int] = None
    registry_is_empty: bool = False

    def summary(self) -> str:
        if not self.held_out:
            if self.registry_is_empty:
                return (f"{self.session_id}: not held out (the holdout registry is "
                        "empty - no session is reserved yet)")
            return f"{self.session_id}: not held out"
        return (f"{self.session_id}: HELD OUT by reservation(s) "
                f"{list(self.reservation_ids)} ({self.scope}) - {self.reason}")


VERDICTS = ('supported', 'refuted', 'inconclusive',
            'hypothesis_generating_only', 'blocked')

HYPOTHESIS_STATUSES = ('proposed', 'approved', 'rejected', 'confirmed',
                       'refuted', 'blocked')


@dataclass(frozen=True)
class TierAssessment:
    """Whether a hypothesis's evidence is exploratory or confirmatory.

    **Derived, never stored.** A stored tier is a field somebody sets to make a
    warning banner go away; the seven conditions below are recomputed on every
    read, so the only way to reach ``'confirmatory'`` is to actually reserve a
    session, pre-register a prediction, unlock the reservation for that one
    hypothesis, run it with a recorded seed and dataset fingerprint, record a
    decision, and clear the threshold at an honest denominator.

    ``blocking_reasons`` is empty if and only if the tier is confirmatory, and
    each entry is written to be printed verbatim in a report.
    """

    hypothesis_id: int
    tier: str
    blocking_reasons: Tuple[str, ...] = ()
    holdout_iteration_id: Optional[int] = None
    frozen_prediction_id: Optional[int] = None

    @property
    def is_confirmatory(self) -> bool:
        return self.tier == 'confirmatory'

    def summary(self) -> str:
        if self.is_confirmatory:
            return (f"hypothesis {self.hypothesis_id}: CONFIRMATORY "
                    f"(holdout iteration {self.holdout_iteration_id})")
        lines = [f"hypothesis {self.hypothesis_id}: EXPLORATORY - "
                 f"{len(self.blocking_reasons)} condition(s) unmet:"]
        lines.extend(f"  - {r}" for r in self.blocking_reasons)
        return '\n'.join(lines)


class LabNotebook:
    """Main interface to the lab notebook, mirroring ``HabitatDatabase``'s
    CRUD-method style in ``database_core.py``."""

    def __init__(self, db_path: Union[str, Path, None] = None):
        if db_path is None:
            db_path = Path.cwd() / "habitat_pipeline.db"
        self.db_path = Path(db_path)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        # create_all builds missing tables but never ALTERs an existing one,
        # so retrofitted columns need their own idempotent pass. Recorded on
        # the instance so a caller (or a test) can assert what migrated.
        self.migrated_columns = _ensure_added_columns(self.engine)

    def get_db_session(self) -> Session:
        return self.SessionLocal()

    # -- Holdout registry -------------------------------------------------
    def reserve_holdout(self, session_id: str, *, cohort: str, reason: str,
                         reserved_by: str, animal_id: str = None,
                         session_dir: str = None) -> HoldoutReservation:
        """Reserve a session (or one animal in it) as held-out.

        Whole-session scope is the default and the recommended one. An
        ``animal_id``-scoped reservation is honoured, but see
        :meth:`holdout_status` — it blocks the entire session for any
        multi-animal analysis, because those read a partner's data through a
        focal animal and would otherwise leak the reserved animal.
        """
        key = normalize_session_key(session_id)
        if key is None:
            raise HoldoutIndeterminate(
                f"cannot reserve {session_id!r}: no YYYYMMDD date found in it. "
                "Reserve by a session id containing the recording date."
            )
        if not reason or not str(reason).strip():
            raise ValueError("a holdout reservation requires a reason")
        if not reserved_by or not str(reserved_by).strip():
            raise ValueError("a holdout reservation requires reserved_by")

        with self.get_db_session() as db_session:
            reservation = HoldoutReservation(
                cohort=cohort,
                session_key=key,
                session_dir=session_dir or (str(session_id) if session_id != key else None),
                animal_id=animal_id,
                scope='animal' if animal_id else 'session',
                reason=reason,
                reserved_by=reserved_by,
            )
            db_session.add(reservation)
            db_session.commit()
            db_session.refresh(reservation)
            return reservation

    def release_holdout(self, reservation_id: int, *, reason: str,
                         released_by: str) -> HoldoutReservation:
        """Stop a reservation blocking, without deleting the record."""
        with self.get_db_session() as db_session:
            reservation = db_session.get(HoldoutReservation, reservation_id)
            if reservation is None:
                raise ValueError(f"No holdout reservation with id {reservation_id}")
            if reservation.released_at is not None:
                raise ValueError(
                    f"reservation {reservation_id} was already released at "
                    f"{reservation.released_at}"
                )
            reservation.released_at = datetime.utcnow()
            reservation.released_by = released_by
            reservation.release_reason = reason
            db_session.commit()
            db_session.refresh(reservation)
            return reservation

    def unlock_holdout(self, reservation_id: int, hypothesis_id: int, *,
                        approved_by: str, frozen_prediction_id: int) -> HoldoutReservation:
        """Open a reservation for exactly one hypothesis's confirmation run.

        The only legal way into the held-out set. Requires a frozen prediction
        that was *not* registered post hoc — confirming against held-out data
        means nothing if the prediction was written after seeing the
        exploratory result.
        """
        with self.get_db_session() as db_session:
            reservation = db_session.get(HoldoutReservation, reservation_id)
            if reservation is None:
                raise ValueError(f"No holdout reservation with id {reservation_id}")
            if not reservation.active:
                raise ValueError(f"reservation {reservation_id} is already released")
            if reservation.unlocked_for_hypothesis_id is not None:
                raise ValueError(
                    f"reservation {reservation_id} is already unlocked for hypothesis "
                    f"{reservation.unlocked_for_hypothesis_id}. One unlock, one hypothesis."
                )
            self._require_prospective_prediction(db_session, frozen_prediction_id, hypothesis_id)
            reservation.unlocked_for_hypothesis_id = hypothesis_id
            reservation.unlocked_frozen_prediction_id = frozen_prediction_id
            reservation.unlocked_at = datetime.utcnow()
            reservation.unlocked_by = approved_by
            db_session.commit()
            db_session.refresh(reservation)
            return reservation

    def _require_prospective_prediction(self, db_session, frozen_prediction_id: int,
                                         hypothesis_id: int) -> None:
        """Validate an unlock's frozen prediction, if that table exists yet.

        Split out so the holdout registry is usable before the
        pre-registration layer lands, without silently skipping the check
        once it has.
        """
        model = globals().get('FrozenPrediction')
        if model is None:  # pragma: no cover - pre-registration layer not built
            return
        prediction = db_session.get(model, frozen_prediction_id)
        if prediction is None:
            raise ValueError(f"No frozen prediction with id {frozen_prediction_id}")
        if prediction.hypothesis_id != hypothesis_id:
            raise ValueError(
                f"frozen prediction {frozen_prediction_id} belongs to hypothesis "
                f"{prediction.hypothesis_id}, not {hypothesis_id}"
            )
        if prediction.registered_post_hoc:
            raise ValueError(
                f"frozen prediction {frozen_prediction_id} was registered post hoc. "
                "A holdout confirmation requires a prediction written before the "
                "data were seen; otherwise the holdout buys nothing."
            )
        if prediction.superseded_by_id is not None:
            raise ValueError(
                f"frozen prediction {frozen_prediction_id} has been superseded; "
                "unlock against the current one."
            )

    def list_holdout(self, active_only: bool = True) -> List[HoldoutReservation]:
        with self.get_db_session() as db_session:
            query = db_session.query(HoldoutReservation)
            if active_only:
                query = query.filter(HoldoutReservation.released_at.is_(None))
            return query.order_by(HoldoutReservation.reserved_at).all()

    def holdout_status(self, session_id: str, animal_id: str = None, *,
                        cohort: str = None, multi_animal: bool = False) -> HoldoutStatus:
        """Is this session/animal reserved? Returns a reason, not a bare bool.

        Consults ``holdout_reservations`` only — never ``iterations``. That is
        the fix for the inverted gate: a reserved session with no iterations
        is exactly the case that must be blocked, and an iteration-based check
        sees nothing there.

        ``multi_animal=True`` must be passed by any analysis that reads more
        than one animal's data (``decode_location`` with a partner
        ``object_name``, ``run_inter_brain``, ``social_spatial_fields``,
        ``decode_partner_distance``). It promotes animal-scoped reservations to
        block the whole session, because such an analysis reads the reserved
        animal regardless of which animal is focal.

        ``cohort`` is optional; when omitted, a reservation in *any* cohort
        for that date blocks. Over-blocking is the safe direction.
        """
        key = normalize_session_key(session_id)
        if key is None:
            raise HoldoutIndeterminate(
                f"cannot determine holdout status for session_id={session_id!r}: "
                "no YYYYMMDD date found in it. Pass a resolvable session id "
                "(e.g. '20251210' or 'RatCity_20251210_1359_40Hz'). Refusing to "
                "report 'not held out' for an id that could not be checked."
            )

        with self.get_db_session() as db_session:
            total_active = db_session.query(HoldoutReservation).filter(
                HoldoutReservation.released_at.is_(None)
            ).count()
            query = db_session.query(HoldoutReservation).filter(
                HoldoutReservation.released_at.is_(None),
                HoldoutReservation.session_key == key,
            )
            if cohort is not None:
                query = query.filter(HoldoutReservation.cohort == cohort)
            candidates = query.all()

            blocking = []
            for reservation in candidates:
                if reservation.scope == 'session':
                    blocking.append(reservation)
                elif multi_animal:
                    # An animal-scoped reservation cannot be honoured by a
                    # multi-animal analysis: it reads the reserved animal's
                    # data whichever animal is nominally focal.
                    blocking.append(reservation)
                elif animal_id is not None and _same_animal(reservation.animal_id, animal_id):
                    blocking.append(reservation)

            if not blocking:
                return HoldoutStatus(
                    session_id=str(session_id), session_key=key, held_out=False,
                    registry_is_empty=(total_active == 0),
                )

            scopes = sorted({r.scope for r in blocking})
            unlocked = [r.unlocked_for_hypothesis_id for r in blocking
                        if r.unlocked_for_hypothesis_id is not None]
            reasons = '; '.join(f"[{r.id}] {r.reason}" for r in blocking)
            return HoldoutStatus(
                session_id=str(session_id), session_key=key, held_out=True,
                reason=reasons,
                reservation_ids=tuple(r.id for r in blocking),
                scope=','.join(scopes),
                unlocked_for_hypothesis_id=unlocked[0] if len(set(unlocked)) == 1 else None,
            )

    def assert_not_held_out(self, session_id: str, animal_id: str = None, *,
                             purpose: str = 'exploratory', hypothesis_id: int = None,
                             cohort: str = None, multi_animal: bool = False) -> HoldoutStatus:
        """Raise :class:`HoldoutViolation` if this target is reserved.

        Call this *before* constructing a ``DataStorageManager`` — the point is
        to refuse before any data is read.

        ``purpose='confirmatory'`` passes only when the blocking reservation
        has been unlocked for exactly ``hypothesis_id``. Everything else,
        including a confirmatory run against a different hypothesis, still
        raises.
        """
        if purpose not in ('exploratory', 'confirmatory'):
            raise ValueError(
                f"purpose must be 'exploratory' or 'confirmatory', got {purpose!r}"
            )
        status = self.holdout_status(session_id, animal_id, cohort=cohort,
                                      multi_animal=multi_animal)
        if not status.held_out:
            return status

        if purpose == 'confirmatory':
            if hypothesis_id is None:
                raise HoldoutViolation(
                    f"{status.summary()}\nA confirmatory run needs hypothesis_id= so the "
                    "unlock can be checked against it."
                )
            if status.unlocked_for_hypothesis_id == hypothesis_id:
                return status
            raise HoldoutViolation(
                f"{status.summary()}\nNot unlocked for hypothesis {hypothesis_id} "
                f"(unlocked for: {status.unlocked_for_hypothesis_id}). Use "
                "unlock_holdout(reservation_id, hypothesis_id, ...) with a frozen "
                "prediction registered before the data were seen."
            )

        raise HoldoutViolation(
            f"{status.summary()}\nThis session is reserved for confirmation and must not be "
            "used for exploratory work. Pick another session, or promote a hypothesis and "
            "unlock the reservation for it."
        )

    # -- Hypotheses -------------------------------------------------------
    def add_hypothesis(self, statement: str, predicted_effect: str = None,
                        chosen_test: str = None,
                        citations: Optional[List[Dict]] = None) -> Hypothesis:
        with self.get_db_session() as db_session:
            hyp = Hypothesis(
                statement=statement,
                predicted_effect=predicted_effect,
                chosen_test=chosen_test,
                citations=_to_json(citations) if citations is not None else None,
            )
            db_session.add(hyp)
            db_session.commit()
            db_session.refresh(hyp)
            return hyp

    def set_hypothesis_status(self, hypothesis_id: int, status: str,
                               notes: str = None) -> Hypothesis:
        """Transition a pre-registered hypothesis's status (design doc §6:
        the scientist's approval-gate decision, but for a hypothesis rather
        than a logged iteration — see ``record_decision`` for the iteration
        equivalent).

        ``'refuted'`` and ``'blocked'`` were added to the original four so
        that refutations are reportable as such rather than filed under
        ``'rejected'`` (which means "the scientist declined to pursue this",
        a different thing), and so ``'blocked'`` stops being expressed only in
        free-text notes — hypothesis 3 needed it and had no way to say it.
        """
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(
                f"status must be one of {'/'.join(repr(s) for s in HYPOTHESIS_STATUSES)}, "
                f"got {status!r}"
            )
        with self.get_db_session() as db_session:
            hyp = db_session.get(Hypothesis, hypothesis_id)
            if hyp is None:
                raise ValueError(f"No hypothesis with id {hypothesis_id}")
            hyp.status = status
            if notes is not None:
                hyp.scientist_notes = notes
            db_session.commit()
            db_session.refresh(hyp)
            return hyp

    def list_hypotheses(self, status: Optional[str] = None) -> List[Hypothesis]:
        """List pre-registered hypotheses, optionally filtered by status —
        used to check for prior/duplicate proposals before registering a
        new one."""
        with self.get_db_session() as db_session:
            query = db_session.query(Hypothesis)
            if status is not None:
                query = query.filter(Hypothesis.status == status)
            return query.order_by(Hypothesis.created_at).all()

    # -- Pre-registration ---------------------------------------------------
    def freeze_prediction(self, hypothesis_id: int, *, statistic: str, direction: str,
                           threshold: float, falsifier: str, alpha: float = 0.05,
                           n_shuffles_planned: int = None, test_family_id: int = None,
                           declared_test_keys: Sequence[str] = None,
                           holdout_required: bool = True,
                           holdout_kind: str = 'replication',
                           registered_post_hoc: bool = False) -> FrozenPrediction:
        """Record a prediction before running anything.

        ``falsifier`` is required. A prediction that cannot say what would
        count against it is not a prediction, and the report's "what would
        change the verdict" section has nothing to print without it.

        Set ``registered_post_hoc=True`` when transcribing a claim from an
        analysis that has already run. It is honest and it is permanent: such a
        record can never satisfy the confirmatory tier.
        """
        import hashlib

        if direction not in ('lt', 'gt'):
            raise ValueError(f"direction must be 'lt' or 'gt', got {direction!r}")
        if not falsifier or not str(falsifier).strip():
            raise ValueError(
                "a frozen prediction requires a falsifier: what result would count "
                "against this hypothesis?"
            )
        if holdout_kind not in ('replication', 'generalization'):
            raise ValueError(
                f"holdout_kind must be 'replication' or 'generalization', got {holdout_kind!r}"
            )

        keys = list(declared_test_keys or ())
        with self.get_db_session() as db_session:
            if db_session.get(Hypothesis, hypothesis_id) is None:
                raise ValueError(f"No hypothesis with id {hypothesis_id}")
            version = 1 + db_session.query(FrozenPrediction).filter(
                FrozenPrediction.hypothesis_id == hypothesis_id).count()

            spec = json.dumps({
                'hypothesis_id': hypothesis_id, 'statistic': statistic,
                'direction': direction, 'threshold': float(threshold),
                'alpha': float(alpha), 'n_shuffles_planned': n_shuffles_planned,
                'declared_test_keys': sorted(keys), 'holdout_kind': holdout_kind,
            }, sort_keys=True)

            prediction = FrozenPrediction(
                hypothesis_id=hypothesis_id, version=version,
                statistic=statistic, direction=direction, threshold=float(threshold),
                alpha=float(alpha), n_shuffles_planned=n_shuffles_planned,
                test_family_id=test_family_id,
                declared_test_keys=_to_json(keys) if keys else None,
                n_tests_declared=len(keys) or None,
                holdout_required=bool(holdout_required), holdout_kind=holdout_kind,
                falsifier=falsifier, registered_post_hoc=bool(registered_post_hoc),
                spec_hash=hashlib.sha256(spec.encode('utf-8')).hexdigest(),
            )
            db_session.add(prediction)
            db_session.commit()
            db_session.refresh(prediction)
            return prediction

    def supersede_prediction(self, prediction_id: int, *, reason: str,
                              **new_prediction_kwargs) -> FrozenPrediction:
        """Replace a frozen prediction by inserting a new one.

        The only sanctioned way to change a pre-registered statistic. The old
        row keeps its values and gains a pointer to the new one, so the report
        can show that the target moved and why — which is the whole point of
        freezing it in the first place.
        """
        if not reason or not str(reason).strip():
            raise ValueError("superseding a frozen prediction requires a reason")
        with self.get_db_session() as db_session:
            old = db_session.get(FrozenPrediction, prediction_id)
            if old is None:
                raise ValueError(f"No frozen prediction with id {prediction_id}")
            if old.superseded_by_id is not None:
                raise ValueError(
                    f"prediction {prediction_id} was already superseded by "
                    f"{old.superseded_by_id}"
                )
            hypothesis_id = old.hypothesis_id

        replacement = self.freeze_prediction(hypothesis_id, **new_prediction_kwargs)
        with self.get_db_session() as db_session:
            old = db_session.get(FrozenPrediction, prediction_id)
            old.superseded_by_id = replacement.id
            old.supersede_reason = reason
            db_session.commit()
        return replacement

    def frozen_predictions_for(self, hypothesis_id: int) -> List[FrozenPrediction]:
        with self.get_db_session() as db_session:
            return db_session.query(FrozenPrediction).filter(
                FrozenPrediction.hypothesis_id == hypothesis_id
            ).order_by(FrozenPrediction.version).all()

    def current_prediction(self, hypothesis_id: int) -> Optional[FrozenPrediction]:
        """The live prediction: not superseded. ``None`` if there is none."""
        for prediction in self.frozen_predictions_for(hypothesis_id):
            if prediction.superseded_by_id is None:
                return prediction
        return None

    # -- Verdicts -----------------------------------------------------------
    def record_verdict(self, hypothesis_id: int, *, verdict: str, rationale: str,
                        tier: str = None, holdout_iteration_id: int = None,
                        denominator_known: bool = False,
                        n_tests_in_denominator: int = None,
                        decided_by: str = 'scientist',
                        supersedes_id: int = None) -> HypothesisVerdict:
        """Append a verdict. Refutations are recorded, never omitted.

        ``tier`` defaults to the derived assessment, so a caller cannot label
        an exploratory result confirmatory by passing a string.
        """
        if verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {'/'.join(repr(v) for v in VERDICTS)}, "
                f"got {verdict!r}"
            )
        if not rationale or not str(rationale).strip():
            raise ValueError("a verdict requires a rationale")
        if tier is None:
            tier = self.evidence_tier(hypothesis_id).tier
        elif tier not in ('exploratory', 'confirmatory'):
            raise ValueError(f"tier must be 'exploratory' or 'confirmatory', got {tier!r}")

        with self.get_db_session() as db_session:
            if db_session.get(Hypothesis, hypothesis_id) is None:
                raise ValueError(f"No hypothesis with id {hypothesis_id}")
            row = HypothesisVerdict(
                hypothesis_id=hypothesis_id, tier=tier, verdict=verdict,
                holdout_iteration_id=holdout_iteration_id,
                denominator_known=bool(denominator_known),
                n_tests_in_denominator=n_tests_in_denominator,
                rationale=rationale, decided_by=decided_by,
                supersedes_id=supersedes_id,
            )
            db_session.add(row)
            db_session.commit()
            db_session.refresh(row)
            return row

    def verdicts_for(self, hypothesis_id: int) -> List[HypothesisVerdict]:
        with self.get_db_session() as db_session:
            return db_session.query(HypothesisVerdict).filter(
                HypothesisVerdict.hypothesis_id == hypothesis_id
            ).order_by(HypothesisVerdict.created_at, HypothesisVerdict.id).all()

    def latest_verdict(self, hypothesis_id: int) -> Optional[HypothesisVerdict]:
        rows = self.verdicts_for(hypothesis_id)
        return rows[-1] if rows else None

    # -- Derived evidence tier ---------------------------------------------
    def evidence_tier(self, hypothesis_id: int) -> TierAssessment:
        """Recompute the tier from its seven preconditions.

        Every condition that fails contributes a sentence a report can print
        verbatim. Nothing here is stored, so nothing here can be edited to
        change the answer.
        """
        reasons: List[str] = []

        prediction = self.current_prediction(hypothesis_id)
        if prediction is None:
            reasons.append(
                "No frozen prediction exists: this analysis was not pre-registered, "
                "so every number in it was chosen after the data were seen.")
        elif prediction.registered_post_hoc:
            reasons.append(
                f"Frozen prediction {prediction.id} was registered post hoc "
                f"({prediction.created_at:%Y-%m-%d}); it records what was claimed but "
                "cannot serve as a pre-registration.")
            prediction = None

        iterations = self.iterations_for_hypothesis(hypothesis_id)
        candidate = None
        if prediction is not None:
            later = [it for it in iterations
                     if it.created_at and it.created_at > prediction.created_at]
            if not later:
                reasons.append(
                    f"No iteration was run after frozen prediction {prediction.id} was "
                    "registered, so nothing has tested it yet.")
            else:
                candidate = self._pick_holdout_iteration(later, hypothesis_id)
                if candidate is None:
                    reasons.append(
                        "No iteration ran against a session reserved and unlocked for "
                        f"hypothesis {hypothesis_id}; all of its runs used exploratory "
                        "sessions the loop had already seen.")

        if candidate is not None:
            if not candidate.seed or not candidate.dataset_fingerprint:
                missing = [name for name, value in
                           (('seed', candidate.seed),
                            ('dataset_fingerprint', candidate.dataset_fingerprint))
                           if not value]
                reasons.append(
                    f"Iteration {candidate.id} did not record {' and '.join(missing)}, "
                    "so the confirmation run is not reproducible.")
            if candidate.scientist_decision in (None, 'pending'):
                reasons.append(
                    f"Iteration {candidate.id} has no durable scientist decision "
                    "(scientist_decision is still 'pending').")
            reasons.extend(self._spent_holdout_reasons(hypothesis_id))
            reasons.extend(self._denominator_reasons(candidate))
            reasons.extend(self._threshold_reasons(candidate, prediction))

        if reasons:
            return TierAssessment(
                hypothesis_id=hypothesis_id, tier='exploratory',
                blocking_reasons=tuple(reasons),
                holdout_iteration_id=candidate.id if candidate else None,
                frozen_prediction_id=prediction.id if prediction else None,
            )
        return TierAssessment(
            hypothesis_id=hypothesis_id, tier='confirmatory',
            holdout_iteration_id=candidate.id if candidate else None,
            frozen_prediction_id=prediction.id if prediction else None,
        )

    def _pick_holdout_iteration(self, iterations: Sequence[Iteration],
                                 hypothesis_id: int) -> Optional[Iteration]:
        """The first iteration run against a session unlocked for this hypothesis."""
        with self.get_db_session() as db_session:
            unlocked_keys = {
                r.session_key for r in db_session.query(HoldoutReservation).filter(
                    HoldoutReservation.unlocked_for_hypothesis_id == hypothesis_id
                ).all()
            }
        if not unlocked_keys:
            return None
        for iteration in iterations:
            if normalize_session_key(iteration.session_id) in unlocked_keys:
                return iteration
        return None

    def _spent_holdout_reasons(self, hypothesis_id: int) -> List[str]:
        """Was the reserved session already seen before it was unlocked?

        A holdout's entire value is that the loop has provably never looked at
        it. If any iteration — for *any* hypothesis — ran against that session
        before the unlock, the set was already spent and a later
        "confirmation" against it is just another exploratory run with a
        pre-registration attached.

        Checked across all hypotheses on purpose: the loop having seen the
        session while chasing a different question contaminates it just as
        thoroughly.
        """
        reasons: List[str] = []
        with self.get_db_session() as db_session:
            reservations = db_session.query(HoldoutReservation).filter(
                HoldoutReservation.unlocked_for_hypothesis_id == hypothesis_id
            ).all()
            for reservation in reservations:
                cutoff = reservation.unlocked_at or reservation.reserved_at
                if cutoff is None:
                    continue
                prior = [
                    it for it in db_session.query(Iteration).filter(
                        Iteration.created_at < cutoff).all()
                    if normalize_session_key(it.session_id) == reservation.session_key
                ]
                if prior:
                    ids = ', '.join(str(it.id) for it in prior)
                    reasons.append(
                        f"Session {reservation.session_key} was already analysed before "
                        f"reservation {reservation.id} was unlocked (iteration(s) {ids}), "
                        "so the holdout was spent and this is not a confirmation.")
        return reasons

    def _denominator_reasons(self, iteration: Iteration) -> List[str]:
        if iteration.test_family_id is None:
            return [f"Iteration {iteration.id} is not attached to a test family, so the "
                    "number of tests behind its claim is unknown."]
        denom = self.family_denominator(iteration.test_family_id)
        if denom['denominator_status'] != 'clean':
            detail = denom['denominator_status'].replace('_', ' ')
            return [f"Test family {iteration.test_family_id} has denominator status "
                    f"'{detail}' rather than 'clean' "
                    f"({denom['n_tests_for_correction']} tests for correction)."]
        return []

    def _threshold_reasons(self, iteration: Iteration,
                            prediction: Optional[FrozenPrediction]) -> List[str]:
        """Was the frozen threshold actually met, at the ledger's denominator?"""
        if prediction is None:
            return []
        met, detail = self._threshold_met(iteration, prediction)
        if met:
            return []
        return [f"The frozen prediction was not met: {detail}"]

    def _threshold_met(self, iteration: Iteration,
                        prediction: FrozenPrediction) -> Tuple[bool, str]:
        summary = iteration.result_summary_dict()
        statistic = prediction.statistic

        if statistic in ('min_q_value', 'q_value'):
            if iteration.test_family_id is None:
                return False, (f"statistic {statistic!r} needs a test family and "
                               f"iteration {iteration.id} has none")
            per_test = self.family_fdr(iteration.test_family_id)['per_test']
            if not per_test:
                return False, f"test family {iteration.test_family_id} has no scored tests"
            observed = min(v['q_value'] for v in per_test.values())
        else:
            observed = summary
            for part in str(statistic).split('.'):
                if isinstance(observed, dict) and part in observed:
                    observed = observed[part]
                else:
                    return False, (f"statistic {statistic!r} is not present in iteration "
                                   f"{iteration.id}'s result summary, so it cannot be "
                                   "verified")

        try:
            observed = float(observed)
        except (TypeError, ValueError):
            return False, f"statistic {statistic!r} is not numeric ({observed!r})"
        if not np.isfinite(observed):
            return False, f"statistic {statistic!r} is not finite ({observed!r})"

        threshold = float(prediction.threshold)
        passed = observed < threshold if prediction.direction == 'lt' else observed > threshold
        comparison = '<' if prediction.direction == 'lt' else '>'
        return passed, (f"{statistic} = {observed:.6g}, predicted "
                        f"{comparison} {threshold:.6g}")

    # -- Test families ------------------------------------------------------
    def create_test_family(self, name: str, correction_method: str = 'bh_fdr',
                            alpha: float = 0.05, notes: str = None) -> TestFamily:
        with self.get_db_session() as db_session:
            family = TestFamily(name=name, correction_method=correction_method,
                                alpha=alpha, notes=notes)
            db_session.add(family)
            db_session.commit()
            db_session.refresh(family)
            return family

    def get_or_create_test_family(self, name: str, correction_method: str = 'bh_fdr',
                                   alpha: float = 0.05, notes: str = None) -> TestFamily:
        """Fetch a family by name, or create it.

        Prefer this over :meth:`create_test_family`, which has no such check
        and already produced two identically-named families eight minutes
        apart in this notebook — one of them empty. Since family membership
        *is* the denominator, a duplicated family silently splits it.
        """
        with self.get_db_session() as db_session:
            existing = db_session.query(TestFamily).filter(
                TestFamily.name == name
            ).order_by(TestFamily.created_at).first()
            if existing is not None:
                return existing
        return self.create_test_family(name, correction_method=correction_method,
                                        alpha=alpha, notes=notes)

    # -- Ledger: declared tests -------------------------------------------
    def declare_family_tests(self, test_family_id: int, test_keys: Sequence[str], *,
                              declared_by: str, budget_max_tests: int = None,
                              denominator_status: str = 'clean',
                              extend: bool = False, notes: str = None) -> List[FamilyTest]:
        """Fix a family's membership *before* any of it is run.

        This is the load-bearing call of the whole layer. ``test_keys`` becomes
        the denominator every q-value in the family is divided by, and it is
        recorded now so that dropping a member later cannot change it
        retroactively.

        Refuses to re-declare an already-declared family unless ``extend=True``
        — growing a family after results are in is the forking path this
        exists to measure, so it has to be a deliberate, recorded act.
        """
        if not test_keys:
            raise ValueError("declare at least one test key")
        keys = list(dict.fromkeys(str(k) for k in test_keys))  # de-dup, keep order

        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            if family is None:
                raise ValueError(f"No test family with id {test_family_id}")

            already = {t.test_key for t in db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id).all()}
            if already and not extend:
                raise ValueError(
                    f"family {test_family_id} already declares {len(already)} test(s). "
                    "Pass extend=True to add more, which is recorded as an extension "
                    "rather than a fresh declaration - a family that grows as results "
                    "arrive is not a denominator."
                )

            new_keys = [k for k in keys if k not in already]
            if budget_max_tests is not None:
                total = len(already) + len(new_keys)
                if total > int(budget_max_tests):
                    raise TestBudgetExhausted(
                        f"declaring {total} tests exceeds budget_max_tests="
                        f"{budget_max_tests} for family {test_family_id}"
                    )
                family.budget_max_tests = int(budget_max_tests)
            elif family.budget_max_tests is None:
                # Default the budget to the declared size: the family is
                # exactly as big as it said it would be.
                family.budget_max_tests = len(already) + len(new_keys)

            created = []
            for key in new_keys:
                test = FamilyTest(
                    test_family_id=test_family_id, test_key=key,
                    status='declared', declared_by=declared_by, notes=notes,
                )
                db_session.add(test)
                created.append(test)

            family.declared_n_tests = len(already) + len(new_keys)
            if family.status is None:
                family.status = 'open'
            if extend and already:
                family.notes = ((family.notes or '') +
                                f"\n[{datetime.utcnow().isoformat()}] extended by "
                                f"{declared_by}: +{len(new_keys)} test(s)"
                                f"{' - ' + notes if notes else ''}").strip()
            else:
                family.denominator_status = denominator_status

            db_session.commit()
            for test in created:
                db_session.refresh(test)
            return created

    def record_family_test(self, test_family_id: int, test_key: str, *,
                            iteration_id: int = None, p_value: float = None,
                            git_commit: str = None, run_key: str = None,
                            status: str = 'run', notes: str = None) -> FamilyTest:
        """Attach a result to a previously declared test.

        Raises :class:`UndeclaredTestError` for a key that was never declared.
        That refusal is where the compute seam and the record seam separate:
        nothing can stop an agent running an analysis forty times in ad-hoc
        Python, but the ledger will not let forty runs be recorded as seven
        tests with a clean q-value.
        """
        if status not in ('run', 'failed'):
            raise ValueError(f"status must be 'run' or 'failed', got {status!r}")

        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            if family is None:
                raise ValueError(f"No test family with id {test_family_id}")

            declared = db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id,
                FamilyTest.test_key == str(test_key),
            ).order_by(FamilyTest.id).all()
            if not declared:
                known = [t.test_key for t in db_session.query(FamilyTest).filter(
                    FamilyTest.test_family_id == test_family_id).all()]
                raise UndeclaredTestError(
                    f"test_key {test_key!r} was never declared for family "
                    f"{test_family_id}. Declared: {known}. Call "
                    "declare_family_tests(...) before running, or "
                    "declare_family_tests(..., extend=True) to record that the "
                    "family grew."
                )

            open_slot = next((t for t in declared if t.status == 'declared'), None)
            if open_slot is not None:
                target = open_slot
            else:
                # Already recorded once: this is a rerun, kept as its own row
                # linked back to the original so it is not double-counted.
                budget = family.budget_max_tests
                n_rows = db_session.query(FamilyTest).filter(
                    FamilyTest.test_family_id == test_family_id).count()
                if budget is not None and n_rows >= int(budget):
                    raise TestBudgetExhausted(
                        f"family {test_family_id} already holds {n_rows} test rows at "
                        f"budget_max_tests={budget}. To record another, first call "
                        f"extend_family_budget({test_family_id}, {int(budget) + 1}, "
                        "actor=..., reason=...)."
                    )
                target = FamilyTest(
                    test_family_id=test_family_id, test_key=str(test_key),
                    declared_by=declared[0].declared_by, rerun_of_id=declared[0].id,
                )
                db_session.add(target)

            target.status = status
            target.iteration_id = iteration_id
            target.p_value = float(p_value) if p_value is not None else None
            target.git_commit = git_commit
            target.run_key = run_key
            if notes:
                target.notes = notes
            db_session.commit()
            db_session.refresh(target)
            return target

    def abandon_family_test(self, test_family_id: int, test_key: str, *,
                             reason: str, outcome_dependent: bool,
                             criterion_available_a_priori: bool = None,
                             applied_after_seeing_results: bool = None,
                             actor: str = None) -> FamilyTest:
        """Drop a declared test, on the record.

        ``outcome_dependent`` is required and has no default: you cannot
        remove a test from a family without stating whether the reason came
        from looking at a result. That one required argument is what makes
        this project's own iteration 11/12 pattern — five of twelve objects
        dropped, four of them because of what the first pass showed —
        impossible to repeat silently.

        An outcome-dependent exclusion does not reduce the denominator. The
        test stays counted, because deciding to drop it *was* a test.
        """
        if not isinstance(outcome_dependent, bool):
            raise TypeError(
                "abandon_family_test requires outcome_dependent=True/False explicitly: "
                "was this test dropped because of something a result showed?"
            )
        if not reason or not str(reason).strip():
            raise ValueError("abandoning a test requires a reason")

        with self.get_db_session() as db_session:
            test = db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id,
                FamilyTest.test_key == str(test_key),
            ).order_by(FamilyTest.id).first()
            if test is None:
                raise UndeclaredTestError(
                    f"test_key {test_key!r} was never declared for family {test_family_id}"
                )

            # A criterion evaluable without running anything (a target's
            # positional variance) may leave the denominator; one that needed a
            # result may not.
            a_priori = (criterion_available_a_priori
                        if criterion_available_a_priori is not None
                        else not outcome_dependent)
            test.status = 'excluded_prespecified' if (a_priori and not outcome_dependent) \
                else 'abandoned'
            test.exclusion_reason = reason
            test.exclusion_outcome_dependent = bool(outcome_dependent)
            test.criterion_available_a_priori = bool(a_priori)
            test.applied_after_seeing_results = (
                bool(applied_after_seeing_results)
                if applied_after_seeing_results is not None else bool(outcome_dependent)
            )
            if actor:
                test.notes = ((test.notes or '') + f"\nabandoned by {actor}").strip()
            db_session.commit()
            db_session.refresh(test)
            return test

    def extend_family_budget(self, test_family_id: int, new_budget: int, *,
                              actor: str, reason: str) -> TestFamily:
        """Raise a family's declared size, leaving an attributable trail."""
        if not reason or not str(reason).strip():
            raise ValueError("extending a test budget requires a reason")
        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            if family is None:
                raise ValueError(f"No test family with id {test_family_id}")
            old = family.budget_max_tests
            if old is not None and int(new_budget) < int(old):
                raise ValueError(
                    f"new budget {new_budget} is below the current {old}; budgets "
                    "are not lowered retroactively"
                )
            family.budget_max_tests = int(new_budget)
            family.notes = ((family.notes or '') +
                            f"\n[{datetime.utcnow().isoformat()}] budget {old} -> "
                            f"{new_budget} by {actor}: {reason}").strip()
            db_session.commit()
            db_session.refresh(family)
            return family

    def family_tests(self, test_family_id: int) -> List[FamilyTest]:
        with self.get_db_session() as db_session:
            return db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id
            ).order_by(FamilyTest.id).all()

    def family_denominator(self, test_family_id: int) -> Dict[str, Any]:
        """The denominator, computed by the ledger rather than by the caller.

        ``n_tests_for_correction`` is the declared family size minus only the
        tests excluded on a criterion that was available *a priori*. Anything
        dropped because of what a result showed stays in the count.

        Never hand a count of surviving tests to
        :func:`ephys._stats_utils.fdr_resolution`. Feeding it the number a
        post-hoc exclusion produced is how a design that guard would have
        rejected as unresolvable came to be certified as resolvable.
        """
        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            if family is None:
                raise ValueError(f"No test family with id {test_family_id}")
            tests = db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id
            ).order_by(FamilyTest.id).all()
            declared_n_tests = family.declared_n_tests
            explicit_status = family.denominator_status

        primary = [t for t in tests if t.rerun_of_id is None]
        by_status: Dict[str, int] = {}
        for t in primary:
            by_status[t.status] = by_status.get(t.status, 0) + 1

        n_declared = declared_n_tests if declared_n_tests is not None else len(primary)
        n_excluded_prespecified = by_status.get('excluded_prespecified', 0)
        outcome_dependent = [
            {'test_key': t.test_key, 'reason': t.exclusion_reason,
             'applied_after_seeing_results': t.applied_after_seeing_results}
            for t in primary if t.exclusion_outcome_dependent
        ]
        commits = sorted({t.git_commit for t in tests
                          if t.status == 'run' and t.git_commit})

        # Precedence: an unrecorded denominator dominates, then an
        # unrecoverable one, then selection on the outcome, then an incoherent
        # pipeline. Each is a strictly weaker claim than the one after it.
        if not tests:
            # A family that predates the ledger, or one nobody declared. Its
            # denominator is *unrecorded* rather than zero, and reporting it as
            # a number would invent a denominator out of an absence.
            status = 'undeclared'
        elif explicit_status == 'reconstructed':
            status = 'reconstructed'
        elif outcome_dependent:
            status = 'outcome_dependent_exclusions'
        elif len(commits) > 1:
            status = 'pipeline_changed'
        else:
            status = explicit_status or 'clean'

        return {
            'test_family_id': test_family_id,
            'n_declared': int(n_declared),
            'n_run': by_status.get('run', 0),
            'n_failed': by_status.get('failed', 0),
            'n_abandoned': by_status.get('abandoned', 0),
            'n_excluded_prespecified': int(n_excluded_prespecified),
            'n_still_declared': by_status.get('declared', 0),
            'n_reruns': len(tests) - len(primary),
            'n_tests_for_correction': int(n_declared) - int(n_excluded_prespecified),
            'denominator_status': status,
            'outcome_dependent_exclusions': outcome_dependent,
            'distinct_commits': commits,
        }

    def family_fdr(self, test_family_id: int, *, alpha: float = None,
                    n_shuffles: int = None) -> Dict[str, Any]:
        """BH-FDR over the *declared* family, not over what happened to run.

        Members that were declared but produced no p-value are entered as
        ``p = 1.0``. That reproduces conservative Benjamini-Hochberg at the
        declared denominator exactly — the step-up minimum-accumulate collapses
        the padded tail to 1.0 and leaves each real p-value adjusted by
        ``p_i * m / i`` — so :func:`ephys._stats_utils.benjamini_hochberg` is
        reused unchanged and the statistics core needs no edit.

        Pass ``n_shuffles`` to also get the resolution verdict computed against
        the ledger's denominator rather than a hand count.
        """
        from ephys._stats_utils import benjamini_hochberg, fdr_resolution

        denom = self.family_denominator(test_family_id)
        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            alpha = float(alpha) if alpha is not None else float(family.alpha or 0.05)
            tests = db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == test_family_id
            ).order_by(FamilyTest.id).all()

        scored = [t for t in tests if t.rerun_of_id is None
                  and t.status == 'run' and t.p_value is not None]
        m = denom['n_tests_for_correction']
        n_padded = max(0, m - len(scored))

        pvals = [t.p_value for t in scored] + [1.0] * n_padded
        if not pvals:
            return {'per_test': {}, 'n_tests_for_correction': m,
                    'alpha': alpha, 'denominator': denom, 'fdr_resolution': None}

        qvals = benjamini_hochberg(np.array(pvals, dtype=np.float64))
        per_test = {
            t.test_key: {
                'p_value': float(t.p_value),
                'q_value': float(qvals[i]),
                'significant': bool(qvals[i] < alpha),
                'padded': False,
                'git_commit': t.git_commit,
            }
            for i, t in enumerate(scored)
        }

        resolution = None
        if n_shuffles:
            resolution = fdr_resolution(n_tests=m, n_shuffles=int(n_shuffles), alpha=alpha)

        return {
            'per_test': per_test,
            'n_tests_for_correction': m,
            'n_scored': len(scored),
            'n_padded': n_padded,
            'alpha': alpha,
            'denominator': denom,
            'fdr_resolution': resolution,
        }

    # -- Iterations -----------------------------------------------------
    def log_iteration(self, analysis_module: str, params: Dict, result_summary: Dict, *,
                       animal_id: str = None, session_id: str = None,
                       hypothesis_id: int = None, test_family_id: int = None,
                       held_out: bool = None,
                       figure_paths: Optional[List[Union[str, Path]]] = None,
                       git_commit: str = None, status: str = None,
                       seed: Union[int, str] = None,
                       dataset_fingerprint: str = None,
                       fingerprint_method: str = None) -> Iteration:
        """Log one iteration of the discovery loop.

        ``result_summary`` should be a *curated* dict of key scalar metrics
        (e.g. population accuracy, best-cell id, significance q-values) —
        not the raw, potentially large per-cell result-dict a decoder
        returns. ``git_commit`` auto-fills via ``git rev-parse HEAD`` when
        omitted.

        ``seed`` and ``dataset_fingerprint`` are optional but are preconditions
        of the confirmatory tier: a confirmation run that cannot be reproduced
        confirms nothing. ``seed`` is stored as a string so that ``0`` and
        "not recorded" stay distinguishable — several modules still hardcode
        their seed and have none to report.

        ``held_out`` is derived from the holdout registry when not given, so
        the flag and the registry cannot disagree. It is descriptive; the
        registry is the authority.
        """
        if git_commit is None:
            git_commit = _git_commit()
        if status is None:
            status = result_summary.get('status', 'unknown')
        if held_out is None:
            held_out = False
            if session_id is not None and normalize_session_key(session_id):
                try:
                    held_out = self.holdout_status(session_id, animal_id).held_out
                except HoldoutIndeterminate:
                    held_out = False

        with self.get_db_session() as db_session:
            iteration = Iteration(
                hypothesis_id=hypothesis_id,
                test_family_id=test_family_id,
                animal_id=animal_id,
                session_id=session_id,
                held_out=held_out,
                analysis_module=analysis_module,
                params=_to_json(params),
                git_commit=git_commit,
                status=status,
                result_summary=_to_json(result_summary),
                figure_paths=_to_json([str(p) for p in figure_paths]) if figure_paths else None,
                seed=str(seed) if seed is not None else None,
                dataset_fingerprint=dataset_fingerprint,
                fingerprint_method=fingerprint_method,
            )
            db_session.add(iteration)
            db_session.commit()
            db_session.refresh(iteration)
            return iteration

    def record_decision(self, iteration_id: int, decision: str, notes: str = None) -> Iteration:
        """Record the scientist's approval-gate decision on a logged iteration."""
        if decision not in ('approved', 'rejected', 'pending'):
            raise ValueError(f"decision must be 'approved'/'rejected'/'pending', got {decision!r}")
        with self.get_db_session() as db_session:
            iteration = db_session.get(Iteration, iteration_id)
            if iteration is None:
                raise ValueError(f"No iteration with id {iteration_id}")
            iteration.scientist_decision = decision
            iteration.decision_notes = notes
            iteration.decision_at = datetime.utcnow()
            db_session.commit()
            db_session.refresh(iteration)
            return iteration

    # -- Queries ----------------------------------------------------------
    def iterations_for_session(self, session_id: str) -> List[Iteration]:
        with self.get_db_session() as db_session:
            return db_session.query(Iteration).filter(
                Iteration.session_id == session_id
            ).order_by(Iteration.created_at).all()

    def iterations_for_hypothesis(self, hypothesis_id: int) -> List[Iteration]:
        with self.get_db_session() as db_session:
            return db_session.query(Iteration).filter(
                Iteration.hypothesis_id == hypothesis_id
            ).order_by(Iteration.created_at).all()

    # -- Campaign-level multiple-comparison control ------------------------
    def recompute_family_significance(self, test_family_id: int,
                                       p_value_key: str = 'p_value') -> Dict[int, float]:
        """Campaign-level BH-FDR: one p-value per iteration in the family.

        Distinct from ``ephys._lda_decoding.compute_population_significance``,
        which corrects across many *cells* within a single iteration — this
        corrects across many *iterations* (analyses) in one campaign. Every
        member iteration's ``result_summary`` must contain ``p_value_key``
        (typically the analysis's own permutation-test p-value for its
        headline claim, e.g. population accuracy vs. chance); iterations
        missing it are skipped.

        Returns ``{iteration_id: q_value}``.
        """
        from ephys._stats_utils import benjamini_hochberg

        with self.get_db_session() as db_session:
            family = db_session.get(TestFamily, test_family_id)
            if family is None:
                raise ValueError(f"No test family with id {test_family_id}")

            ids, pvals = [], []
            for iteration in family.iterations:
                summary = iteration.result_summary_dict()
                if p_value_key in summary:
                    ids.append(iteration.id)
                    pvals.append(summary[p_value_key])

        if not ids:
            return {}

        q_values = benjamini_hochberg(np.array(pvals, dtype=np.float64))
        return {iteration_id: float(q) for iteration_id, q in zip(ids, q_values)}

    def __repr__(self):
        with self.get_db_session() as db_session:
            n_hyp = db_session.query(Hypothesis).count()
            n_families = db_session.query(TestFamily).count()
            n_iter = db_session.query(Iteration).count()
        return f"<LabNotebook(hypotheses={n_hyp}, test_families={n_families}, iterations={n_iter})>"
