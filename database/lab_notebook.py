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
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
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

    def get_db_session(self) -> Session:
        return self.SessionLocal()

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

    # -- Iterations -----------------------------------------------------
    def log_iteration(self, analysis_module: str, params: Dict, result_summary: Dict, *,
                       animal_id: str = None, session_id: str = None,
                       hypothesis_id: int = None, test_family_id: int = None,
                       held_out: bool = False,
                       figure_paths: Optional[List[Union[str, Path]]] = None,
                       git_commit: str = None, status: str = None) -> Iteration:
        """Log one iteration of the discovery loop.

        ``result_summary`` should be a *curated* dict of key scalar metrics
        (e.g. population accuracy, best-cell id, significance q-values) —
        not the raw, potentially large per-cell result-dict a decoder
        returns. ``git_commit`` auto-fills via ``git rev-parse HEAD`` when
        omitted.
        """
        if git_commit is None:
            git_commit = _git_commit()
        if status is None:
            status = result_summary.get('status', 'unknown')

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
