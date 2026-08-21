"""
Per-hypothesis HTML reports over the lab notebook.

A fixed eight-section structure, rendered in the same order every time:

    1. Claim
    2. Frozen prediction
    3. What actually ran
    4. Figures
    5. Statistic against null   (with a mandatory denominator sub-block)
    6. Verdict
    7. Threats to validity
    8. What would change the verdict

**No section is ever omitted.** An absent section is itself a claim about the
work — "no frozen prediction exists" and "there was no need for one" look
identical if the heading simply disappears — so an empty section renders an
explicit sentence saying what is missing and what that implies.

Design notes
------------
*No templating dependency.* The repo has no jinja2, no ``to_html``, and its
statistics core deliberately avoids ``statsmodels`` to stay installable on the
Janelia workstation. Adding a template engine for one document would be the
wrong trade, so this is ``string.Template``-free f-strings plus
:func:`html.escape` and :mod:`base64`.

*Three layers, split for testability.* :func:`collect_report_data` reads the
database and returns dataclasses with no HTML in them; :func:`render_html`
turns dataclasses into a string and touches neither the database nor the
filesystem; the ``build_*`` functions are thin IO wrappers. Rendering is
byte-deterministic when ``now`` and ``git_commit`` are injected.

Assumptions:
    - **The "hypothesis-generating only" banner is driven by a derived tier.**
      :meth:`database.lab_notebook.LabNotebook.evidence_tier` recomputes its
      seven preconditions on every read, so the only way to remove the banner
      is to actually satisfy them. There is deliberately no parameter to
      suppress it.
    - **A refuted verdict renders with the same CSS class, box and weight as a
      supported one**, differing only in hue. Prominence is the mechanical part
      of "refutations reported as prominently as confirmations"; the rest is
      the index report, because a hypothesis that gets quietly dropped has no
      per-hypothesis document at all and only a collection can show that.
    - **Section 5 always states its denominator.** Reporting a q-value without
      the number of tests it was divided by is how this project's own
      partner-position result came to be read as significant, so the declared
      family size, the exclusions, and the q-values recomputed at the declared
      denominator sit next to the ones that were logged.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    'HYPOTHESIS_GENERATING_ONLY',
    'SECTIONS',
    'MAX_EMBEDDED_BYTES',
    'FigureRef',
    'ReportData',
    'collect_report_data',
    'collect_orphan_report_data',
    'collect_index_data',
    'render_html',
    'render_index_html',
    'build_hypothesis_report',
    'build_index_report',
]

#: Rendered verbatim, per the scope decision that the reports "should say so in
#: those words".
HYPOTHESIS_GENERATING_ONLY = (
    "Without the holdout, the loop's output is hypothesis-generating only."
)

SECTIONS: Tuple[Tuple[str, str], ...] = (
    ('claim', 'Claim'),
    ('prediction', 'Frozen prediction'),
    ('ran', 'What actually ran'),
    ('figures', 'Figures'),
    ('statistic', 'Statistic against null'),
    ('verdict', 'Verdict'),
    ('threats', 'Threats to validity'),
    ('falsifier', 'What would change the verdict'),
)

#: Total base64 payload cap. Past this, figures link out instead of embedding.
MAX_EMBEDDED_BYTES = 15 * 1024 * 1024


# ----------------------------------------------------------------- data

@dataclass(frozen=True)
class FigureRef:
    path: str
    exists: bool
    caption: str = ''
    iteration_id: Optional[int] = None
    n_bytes: int = 0
    data_uri: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass(frozen=True)
class ReportData:
    title: str
    hypothesis: Dict[str, Any] = field(default_factory=dict)
    citations: Tuple[Dict[str, Any], ...] = ()
    current_prediction: Optional[Dict[str, Any]] = None
    superseded_predictions: Tuple[Dict[str, Any], ...] = ()
    iterations: Tuple[Dict[str, Any], ...] = ()
    deviations: Tuple[str, ...] = ()
    figures: Tuple[FigureRef, ...] = ()
    families: Tuple[Dict[str, Any], ...] = ()
    tier: Optional[Dict[str, Any]] = None
    verdict: Optional[Dict[str, Any]] = None
    threats: Tuple[str, ...] = ()
    falsifiers: Tuple[str, ...] = ()
    generated_at: str = ''
    git_commit: Optional[str] = None
    db_path: str = ''
    is_orphan: bool = False

    @property
    def hypothesis_generating_only(self) -> bool:
        return not (self.tier or {}).get('is_confirmatory', False)


def _row_to_dict(row, fields: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in fields:
        value = getattr(row, name, None)
        out[name] = value.isoformat(sep=' ', timespec='seconds') \
            if isinstance(value, datetime) else value
    return out


_ITERATION_FIELDS = (
    'id', 'analysis_module', 'animal_id', 'session_id', 'status', 'git_commit',
    'seed', 'dataset_fingerprint', 'fingerprint_method', 'held_out',
    'scientist_decision', 'decision_notes', 'test_family_id', 'created_at',
    'hypothesis_id', 'frozen_prediction_id',
)
_PREDICTION_FIELDS = (
    'id', 'version', 'statistic', 'direction', 'threshold', 'alpha',
    'n_shuffles_planned', 'test_family_id', 'n_tests_declared', 'holdout_required',
    'holdout_kind', 'falsifier', 'registered_post_hoc', 'spec_hash', 'created_at',
    'superseded_by_id', 'supersede_reason',
)
_VERDICT_FIELDS = (
    'id', 'tier', 'verdict', 'holdout_iteration_id', 'denominator_known',
    'n_tests_in_denominator', 'rationale', 'decided_by', 'created_at',
)


def _collect_figures(iterations, *, embed: bool) -> Tuple[FigureRef, ...]:
    out: List[FigureRef] = []
    budget = MAX_EMBEDDED_BYTES
    for iteration in iterations:
        for raw in iteration.figure_paths_list():
            path = Path(raw)
            if not path.exists():
                out.append(FigureRef(
                    path=str(raw), exists=False, iteration_id=iteration.id,
                    skipped_reason=(
                        'recorded here but the file no longer exists - it was written '
                        'to a temporary scratchpad rather than a durable output '
                        'directory'),
                ))
                continue
            size = path.stat().st_size
            if not embed:
                out.append(FigureRef(str(raw), True, iteration_id=iteration.id,
                                     n_bytes=size, skipped_reason='embedding disabled'))
                continue
            if size > budget:
                out.append(FigureRef(
                    str(raw), True, iteration_id=iteration.id, n_bytes=size,
                    skipped_reason=f'too large to embed ({size / 1e6:.1f} MB); '
                                   'linked instead'))
                continue
            budget -= size
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            payload = base64.b64encode(path.read_bytes()).decode('ascii')
            out.append(FigureRef(
                path=str(raw), exists=True, iteration_id=iteration.id, n_bytes=size,
                caption=f"{path.name} (iteration {iteration.id})",
                data_uri=f"data:{mime};base64,{payload}",
            ))
    return tuple(out)


def _family_block(nb, family_id: int, iterations) -> Dict[str, Any]:
    """Declared denominator, exclusions, and q recomputed at that denominator."""
    denom = nb.family_denominator(family_id)
    fdr = nb.family_fdr(family_id)

    logged: Dict[str, float] = {}
    for iteration in iterations:
        if iteration.test_family_id != family_id:
            continue
        summary = iteration.result_summary_dict()
        for entry in summary.get('per_object', []) or []:
            key = f"object={entry.get('object_name')}"
            if entry.get('q_value') is not None:
                logged[key] = float(entry['q_value'])

    rows = []
    for key, values in sorted(fdr['per_test'].items()):
        rows.append({
            'test_key': key,
            'p_value': values['p_value'],
            'q_as_logged': logged.get(key),
            'q_at_declared': values['q_value'],
            'significant_at_declared': values['significant'],
            'git_commit': values.get('git_commit'),
        })

    exclusions = [
        {'test_key': t.test_key, 'status': t.status,
         'reason': t.exclusion_reason,
         'outcome_dependent': bool(t.exclusion_outcome_dependent),
         'a_priori': bool(t.criterion_available_a_priori),
         'applied_after_seeing_results': bool(t.applied_after_seeing_results)}
        for t in nb.family_tests(family_id)
        if t.status in ('abandoned', 'excluded_prespecified')
    ]

    n_shuffles = None
    for iteration in iterations:
        if iteration.test_family_id == family_id:
            n_shuffles = iteration.params_dict().get('n_shuffles') or n_shuffles
    # Only meaningful once the family has a declared size. A legacy family
    # (created before the ledger existed) has none, and asking fdr_resolution
    # about zero tests would raise.
    resolution = None
    if n_shuffles and denom['n_tests_for_correction'] >= 1:
        from ephys._stats_utils import fdr_resolution
        resolution = fdr_resolution(n_tests=denom['n_tests_for_correction'],
                                    n_shuffles=int(n_shuffles),
                                    alpha=fdr['alpha'])

    return {'family_id': family_id, 'denominator': denom, 'alpha': fdr['alpha'],
            'rows': rows, 'exclusions': exclusions, 'n_shuffles': n_shuffles,
            'fdr_resolution': resolution, 'n_padded': fdr.get('n_padded', 0)}


def _deviations(iterations, prediction: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """What differed between what was pre-registered and what ran."""
    if prediction is None:
        return ()
    out: List[str] = []
    planned = prediction.get('n_shuffles_planned')
    for iteration in iterations:
        actual = iteration.params_dict().get('n_shuffles')
        if planned and actual is not None and int(actual) != int(planned):
            out.append(f"iteration {iteration.id}: n_shuffles={actual}, "
                       f"{planned} planned")
        if prediction.get('test_family_id') and iteration.test_family_id and \
                iteration.test_family_id != prediction['test_family_id']:
            out.append(f"iteration {iteration.id}: ran against test family "
                       f"{iteration.test_family_id}, prediction named "
                       f"{prediction['test_family_id']}")
    return tuple(out)


def _threats(nb, iterations, families, tier, figures) -> Tuple[str, ...]:
    """Machine-checkable checklist. Never empty."""
    out: List[str] = []

    if not iterations:
        out.append("No iterations are logged for this hypothesis, so there is nothing "
                   "to assess.")
        return tuple(out)

    sessions = {it.session_id for it in iterations if it.session_id}
    animals = {it.animal_id for it in iterations if it.animal_id}
    if len(sessions) <= 1:
        out.append(f"Single session ({', '.join(sorted(sessions)) or 'unknown'}): "
                   "nothing here speaks to whether the effect replicates.")
    if len(animals) <= 1:
        out.append(f"Single animal ({', '.join(sorted(animals)) or 'unknown'}): "
                   "the result may be idiosyncratic to this subject.")

    for iteration in iterations:
        summary = iteration.result_summary_dict()
        accuracy = summary.get('population_accuracy_mean')
        baseline = summary.get('population_baseline_accuracy',
                               summary.get('baseline_accuracy'))
        if accuracy is not None and baseline is not None:
            try:
                if float(accuracy) <= float(baseline):
                    out.append(
                        f"Iteration {iteration.id}: accuracy {float(accuracy):.3f} is at "
                        f"or below its majority-class baseline {float(baseline):.3f}. "
                        "Compare against the baseline, not 1/n_classes.")
            except (TypeError, ValueError):
                pass
        if iteration.scientist_decision in (None, 'pending'):
            out.append(f"Iteration {iteration.id} has no durable scientist decision; "
                       "the approval gate for it was conversational and left no record.")
        if not iteration.seed or not iteration.dataset_fingerprint:
            out.append(f"Iteration {iteration.id} did not record "
                       f"{'a seed' if not iteration.seed else 'a dataset fingerprint'}, "
                       "so it cannot be reproduced exactly.")
        if iteration.test_family_id is None:
            out.append(f"Iteration {iteration.id} is not attached to a test family, so "
                       "the number of tests behind its claim is unknown.")

    for block in families:
        denom = block['denominator']
        if denom['denominator_status'] != 'clean':
            out.append(f"Test family {block['family_id']} has denominator status "
                       f"'{denom['denominator_status']}' "
                       f"({denom['n_tests_for_correction']} tests for correction).")
        for exclusion in block['exclusions']:
            if exclusion['outcome_dependent']:
                out.append(f"Test '{exclusion['test_key']}' was dropped for an "
                           f"outcome-dependent reason ({exclusion['reason']}), which is "
                           "selection on the result.")
            elif exclusion['applied_after_seeing_results']:
                out.append(f"Test '{exclusion['test_key']}' was excluded on an a-priori "
                           "criterion, but the criterion was applied after the first "
                           "results were seen.")
        if len(denom['distinct_commits']) > 1:
            out.append(f"Test family {block['family_id']} mixes {len(denom['distinct_commits'])} "
                       f"code versions ({', '.join(denom['distinct_commits'])}); those "
                       "are not the same statistic.")
        resolution = block['fdr_resolution']
        if resolution and not resolution['resolvable']:
            out.append(
                f"Test family {block['family_id']} could not resolve FDR significance at "
                f"{block['n_shuffles']} shuffles across {resolution['n_tests']} tests "
                f"(best achievable q {resolution['best_achievable_q']:.3f}). A null "
                "result here is uninformative, not evidence of absence; "
                f"{resolution['recommended_n_shuffles']} shuffles would be needed.")
        for row in block['rows']:
            if block['n_shuffles'] and row['p_value'] is not None:
                floor = 1.0 / (int(block['n_shuffles']) + 1)
                if row['p_value'] <= floor + 1e-12:
                    out.append(
                        f"'{row['test_key']}' has p pinned at the permutation floor "
                        f"({floor:.6f}); that is a bound, not a measurement.")
            if row['q_as_logged'] is not None and row['q_at_declared'] is not None and \
                    abs(row['q_as_logged'] - row['q_at_declared']) > 1e-9:
                out.append(
                    f"'{row['test_key']}' was logged with q={row['q_as_logged']:.4f} but "
                    f"is q={row['q_at_declared']:.4f} at the declared denominator.")

    if any(f.exists and 'decoding' in Path(f.path).name for f in figures):
        out.append("Figures from ephys/decoding_plots.py draw their chance line at "
                   "1/n_classes, which is not the majority-class baseline. Read the "
                   "'above chance' marks accordingly.")

    if tier and not tier.get('is_confirmatory'):
        out.append("No headline number in this report has been independently "
                   "recomputed from its inputs by the report generator; that check is "
                   "a human step and is how every trap in this project was found.")
    return tuple(out)


def _falsifiers(prediction, families, tier) -> Tuple[str, ...]:
    out: List[str] = []
    if prediction and prediction.get('falsifier'):
        out.append(prediction['falsifier'])

    for block in families:
        resolution = block['fdr_resolution']
        if resolution and not resolution['resolvable']:
            out.append(f"Raising the permutation budget to "
                       f"{resolution['recommended_n_shuffles']} shuffles would make a "
                       f"null result in family {block['family_id']} interpretable.")
        denom = block['denominator']
        for row in block['rows']:
            if row['q_as_logged'] is not None and row['q_at_declared'] is not None and \
                    row['q_as_logged'] < block['alpha'] <= row['q_at_declared']:
                out.append(
                    f"'{row['test_key']}' clears q<{block['alpha']} only if the declared "
                    f"family is smaller than {denom['n_tests_for_correction']} tests; at "
                    "the declared denominator it does not.")

    if tier and not tier.get('is_confirmatory'):
        reasons = tier.get('blocking_reasons') or ()
        if reasons:
            out.append("Reaching a confirmatory tier requires resolving: "
                       + '; '.join(reasons))
    if not out:
        out.append("No falsifier is on record. A claim with no stated falsifier cannot "
                   "be tested.")
    return tuple(out)


def collect_report_data(nb, hypothesis_id: int, *, now: datetime = None,
                        git_commit: str = None, embed_figures: bool = True) -> ReportData:
    """Assemble everything one hypothesis's report needs. No HTML here."""
    from database.lab_notebook import Hypothesis

    with nb.get_db_session() as db_session:
        hypothesis = db_session.get(Hypothesis, hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"No hypothesis with id {hypothesis_id}")
        hypothesis_dict = _row_to_dict(hypothesis, (
            'id', 'statement', 'predicted_effect', 'chosen_test', 'status',
            'scientist_notes', 'created_at'))
        citations = json.loads(hypothesis.citations) if hypothesis.citations else []

    predictions = nb.frozen_predictions_for(hypothesis_id)
    current = next((p for p in predictions if p.superseded_by_id is None), None)
    current_dict = _row_to_dict(current, _PREDICTION_FIELDS) if current else None
    superseded = tuple(_row_to_dict(p, _PREDICTION_FIELDS)
                       for p in predictions if p.superseded_by_id is not None)

    iterations = nb.iterations_for_hypothesis(hypothesis_id)
    tier = nb.evidence_tier(hypothesis_id)
    verdict = nb.latest_verdict(hypothesis_id)

    family_ids = sorted({it.test_family_id for it in iterations
                         if it.test_family_id is not None})
    families = tuple(_family_block(nb, fid, iterations) for fid in family_ids)
    figures = _collect_figures(iterations, embed=embed_figures)

    tier_dict = {'tier': tier.tier, 'is_confirmatory': tier.is_confirmatory,
                 'blocking_reasons': list(tier.blocking_reasons),
                 'holdout_iteration_id': tier.holdout_iteration_id,
                 'frozen_prediction_id': tier.frozen_prediction_id}

    return ReportData(
        title=f"Hypothesis {hypothesis_id}",
        hypothesis=hypothesis_dict,
        citations=tuple(citations),
        current_prediction=current_dict,
        superseded_predictions=superseded,
        iterations=tuple(dict(_row_to_dict(it, _ITERATION_FIELDS),
                              params=it.params_dict())
                         for it in iterations),
        deviations=_deviations(iterations, current_dict),
        figures=figures,
        families=families,
        tier=tier_dict,
        verdict=_row_to_dict(verdict, _VERDICT_FIELDS) if verdict else None,
        threats=_threats(nb, iterations, families, tier_dict, figures),
        falsifiers=_falsifiers(current_dict, families, tier_dict),
        generated_at=(now or datetime.now()).isoformat(sep=' ', timespec='seconds'),
        git_commit=git_commit,
        db_path=str(nb.db_path),
    )


def collect_orphan_report_data(nb, iteration_ids: Sequence[int], *, title: str,
                               now: datetime = None, git_commit: str = None,
                               embed_figures: bool = True) -> ReportData:
    """Report over iterations with no hypothesis attached.

    Not optional: most of this notebook's iterations have ``hypothesis_id``
    NULL, including the one behind its headline opponent-identity finding, so a
    hypothesis-only generator could not render the project's own result.
    """
    from database.lab_notebook import Iteration

    with nb.get_db_session() as db_session:
        iterations = [db_session.get(Iteration, int(i)) for i in iteration_ids]
        missing = [i for i, row in zip(iteration_ids, iterations) if row is None]
        if missing:
            raise ValueError(f"No iteration(s) with id {missing}")
        db_session.expunge_all()

    family_ids = sorted({it.test_family_id for it in iterations
                         if it.test_family_id is not None})
    families = tuple(_family_block(nb, fid, iterations) for fid in family_ids)
    figures = _collect_figures(iterations, embed=embed_figures)
    tier_dict = {'tier': 'exploratory', 'is_confirmatory': False,
                 'blocking_reasons': [
                     'These iterations are not attached to any hypothesis, so no '
                     'prediction was pre-registered for them.'],
                 'holdout_iteration_id': None, 'frozen_prediction_id': None}

    return ReportData(
        title=title,
        hypothesis={},
        iterations=tuple(dict(_row_to_dict(it, _ITERATION_FIELDS),
                              params=it.params_dict())
                         for it in iterations),
        figures=figures,
        families=families,
        tier=tier_dict,
        threats=_threats(nb, iterations, families, tier_dict, figures),
        falsifiers=_falsifiers(None, families, tier_dict),
        generated_at=(now or datetime.now()).isoformat(sep=' ', timespec='seconds'),
        git_commit=git_commit,
        db_path=str(nb.db_path),
        is_orphan=True,
    )


def collect_index_data(nb, *, now: datetime = None,
                       git_commit: str = None) -> Dict[str, Any]:
    """Every hypothesis with its tier, verdict and denominator status.

    A first-class deliverable rather than an extra: prominence is a property of
    the collection. A hypothesis that was quietly dropped has no per-hypothesis
    report at all, so only an index can show that it exists and went nowhere.
    """
    rows: List[Dict[str, Any]] = []
    for hypothesis in nb.list_hypotheses():
        iterations = nb.iterations_for_hypothesis(hypothesis.id)
        tier = nb.evidence_tier(hypothesis.id)
        verdict = nb.latest_verdict(hypothesis.id)
        statuses = sorted({
            nb.family_denominator(fid)['denominator_status']
            for fid in {it.test_family_id for it in iterations
                        if it.test_family_id is not None}
        })
        rows.append({
            'id': hypothesis.id,
            'statement': hypothesis.statement,
            'status': hypothesis.status,
            'tier': tier.tier,
            'n_blocking_reasons': len(tier.blocking_reasons),
            'verdict': verdict.verdict if verdict else None,
            'n_iterations': len(iterations),
            'denominator_statuses': statuses,
        })

    # Iterations with no hypothesis attached are the majority of this notebook,
    # including the one behind its headline finding. Surfaced here so the index
    # doesn't quietly present the hypothesis list as the whole record.
    from database.lab_notebook import Iteration
    with nb.get_db_session() as db_session:
        orphans = [row.id for row in db_session.query(Iteration).filter(
            Iteration.hypothesis_id.is_(None)).order_by(Iteration.id).all()]

    return {
        'rows': rows,
        'orphan_iteration_ids': orphans,
        'generated_at': (now or datetime.now()).isoformat(sep=' ', timespec='seconds'),
        'git_commit': git_commit,
        'db_path': str(nb.db_path),
    }


# -------------------------------------------------------------- rendering

_CSS = """
:root { --fg:#1a1a1a; --muted:#5c5c5c; --line:#d8d8d8; --bg:#fff;
        --warn-bg:#fff4e5; --warn-line:#e8a33d;
        --ok-bg:#eef7ee; --ok-line:#4a8f4a;
        --bad-bg:#fdecec; --bad-line:#c0392b;
        --neutral-bg:#f1f3f5; --neutral-line:#8a8a8a; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--fg); background: var(--bg); line-height: 1.55;
       max-width: 62rem; margin: 0 auto; padding: 2rem 1.25rem 5rem; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 2px solid var(--line); }
h2 .num { color: var(--muted); font-weight: 400; margin-right: .5rem; }
h3 { font-size: .95rem; margin: 1.25rem 0 .4rem; }
p, li { margin: .4rem 0; }
code, .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
              font-size: .86em; }
.sub { color: var(--muted); font-size: .88rem; }
table { border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .88rem; }
th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line);
         vertical-align: top; }
th { font-weight: 600; background: var(--neutral-bg); }
td.num { text-align: right; font-family: ui-monospace, Consolas, monospace; }
.banner { border: 2px solid var(--warn-line); background: var(--warn-bg);
          padding: .8rem 1rem; margin: 1rem 0; font-weight: 600; border-radius: 4px; }
/* Verdict boxes share one class on purpose: a refuted result must be as
   prominent as a supported one, differing only in hue. */
.verdict { border: 2px solid var(--neutral-line); background: var(--neutral-bg);
           padding: 1rem 1.15rem; margin: .8rem 0; border-radius: 4px; }
.verdict .label { font-size: 1.25rem; font-weight: 700; letter-spacing: .02em;
                  text-transform: uppercase; display: block; margin-bottom: .35rem; }
.verdict.supported { border-color: var(--ok-line); background: var(--ok-bg); }
.verdict.refuted { border-color: var(--bad-line); background: var(--bad-bg); }
.verdict.inconclusive, .verdict.hypothesis_generating_only, .verdict.blocked {
    border-color: var(--warn-line); background: var(--warn-bg); }
.missing { color: var(--muted); font-style: italic; }
.flag { color: var(--bad-line); font-weight: 600; }
.empty { border-left: 3px solid var(--warn-line); background: var(--warn-bg);
         padding: .6rem .85rem; margin: .5rem 0; }
figure { margin: 1.25rem 0; }
figure img { max-width: 100%; border: 1px solid var(--line); }
figcaption { font-size: .84rem; color: var(--muted); margin-top: .3rem; }
ul.checklist { padding-left: 1.15rem; }
.denominator { border: 1px solid var(--neutral-line); background: var(--neutral-bg);
               padding: .7rem .9rem; margin: .7rem 0; border-radius: 4px; }
.footer { margin-top: 3rem; padding-top: .8rem; border-top: 1px solid var(--line);
          color: var(--muted); font-size: .82rem; }
"""


def _e(value: Any) -> str:
    """Escape for HTML. Every database-sourced string goes through here."""
    if value is None:
        return ''
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return '&mdash;'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return _e(value)


def _empty(message: str) -> str:
    return f'<div class="empty">{message}</div>'


def _banner() -> str:
    """The hypothesis-generating-only banner, rendered verbatim.

    Deliberately *not* passed through :func:`_e`. It is a module constant with
    no data interpolated into it, and escaping would render the apostrophe as
    ``&#x27;`` — which browsers display correctly but which makes the sentence
    ungreppable in the output file. The requirement is that reports say this in
    these words, so the words have to survive into the file.
    """
    return f'<div class="banner">{HYPOTHESIS_GENERATING_ONLY}</div>'


def _heading(index: int, anchor: str, title: str) -> str:
    return f'<h2 id="{anchor}"><span class="num">{index}.</span>{_e(title)}</h2>'


def _section_claim(data: ReportData) -> str:
    out = []
    if data.is_orphan:
        out.append(_empty(
            'These iterations are not attached to any hypothesis. They were run as '
            'exploratory work, so there is no claim on record for them.'))
    elif data.hypothesis:
        out.append(f'<p><strong>{_e(data.hypothesis.get("statement"))}</strong></p>')
        if data.hypothesis.get('predicted_effect'):
            out.append(f'<p><span class="sub">Predicted effect:</span> '
                       f'{_e(data.hypothesis["predicted_effect"])}</p>')
        if data.hypothesis.get('chosen_test'):
            out.append(f'<p><span class="sub">Chosen test:</span> '
                       f'<span class="mono">{_e(data.hypothesis["chosen_test"])}</span></p>')
        out.append(f'<p class="sub">Status: {_e(data.hypothesis.get("status"))} '
                   f'&middot; registered {_e(data.hypothesis.get("created_at"))}</p>')
        if data.hypothesis.get('scientist_notes'):
            out.append(f'<p><span class="sub">Notes:</span> '
                       f'{_e(data.hypothesis["scientist_notes"])}</p>')

    if data.citations:
        out.append('<h3>References</h3><ul>')
        for citation in data.citations:
            label = citation.get('title') or citation.get('id') or 'reference'
            url = citation.get('url')
            inner = (f'<a href="{_e(url)}">{_e(label)}</a>' if url else _e(label))
            source = f" <span class=\"sub\">({_e(citation.get('source'))})</span>" \
                if citation.get('source') else ''
            out.append(f'<li>{inner}{source}</li>')
        out.append('</ul>')
    else:
        out.append(_empty(
            'No citations are recorded. This hypothesis is <strong>not '
            'literature-grounded</strong> &mdash; it was proposed while the literature '
            'MCP servers were unauthenticated, so nothing here rests on a cited '
            'result.'))
    return '\n'.join(out)


def _section_prediction(data: ReportData) -> str:
    out = []
    prediction = data.current_prediction
    if prediction is None:
        out.append(_empty(
            '<strong>No frozen prediction exists.</strong> This analysis was not '
            'pre-registered, so every number below was chosen after the data were '
            'seen. Nothing in this report can distinguish a predicted result from a '
            'selected one.'))
    else:
        if prediction.get('registered_post_hoc'):
            out.append(_empty(
                f'<strong>Registered retroactively</strong> on '
                f'{_e(prediction.get("created_at"))}. This records what was claimed; '
                'it is <strong>not</strong> a pre-registration and cannot support a '
                'confirmatory claim.'))
        comparison = '&lt;' if prediction['direction'] == 'lt' else '&gt;'
        out.append(
            '<table><tr><th>Field</th><th>Value</th></tr>'
            f'<tr><td>statistic</td><td class="mono">{_e(prediction["statistic"])}</td></tr>'
            f'<tr><td>prediction</td><td class="mono">{_e(prediction["statistic"])} '
            f'{comparison} {_fmt(prediction["threshold"], 6)}</td></tr>'
            f'<tr><td>alpha</td><td class="num">{_fmt(prediction.get("alpha"))}</td></tr>'
            f'<tr><td>n_shuffles planned</td>'
            f'<td class="num">{_fmt(prediction.get("n_shuffles_planned"))}</td></tr>'
            f'<tr><td>declared tests</td>'
            f'<td class="num">{_fmt(prediction.get("n_tests_declared"))}</td></tr>'
            f'<tr><td>holdout required</td>'
            f'<td>{_fmt(prediction.get("holdout_required"))}</td></tr>'
            f'<tr><td>holdout kind</td><td>{_e(prediction.get("holdout_kind"))} '
            '<span class="sub">(a failed generalization test is not a refutation of '
            'the original claim)</span></td></tr>'
            f'<tr><td>registered</td><td>{_e(prediction.get("created_at"))}</td></tr>'
            f'<tr><td>spec hash</td>'
            f'<td class="mono">{_e((prediction.get("spec_hash") or "")[:16])}</td></tr>'
            '</table>')

    if data.superseded_predictions:
        out.append('<h3>Superseded versions</h3>'
                   '<p class="sub">The audit trail for &ldquo;no post-hoc redefinition '
                   'of a statistic without a new frozen record&rdquo;.</p>'
                   '<table><tr><th>v</th><th>statistic</th><th>threshold</th>'
                   '<th>why it was replaced</th></tr>')
        for prior in data.superseded_predictions:
            out.append(f'<tr><td class="num">{_fmt(prior.get("version"))}</td>'
                       f'<td class="mono">{_e(prior.get("statistic"))}</td>'
                       f'<td class="num">{_fmt(prior.get("threshold"), 6)}</td>'
                       f'<td>{_e(prior.get("supersede_reason"))}</td></tr>')
        out.append('</table>')
    return '\n'.join(out)


def _section_ran(data: ReportData) -> str:
    if not data.iterations:
        return _empty('<strong>No iterations are logged.</strong> Nothing has been run '
                      'for this hypothesis.')
    out = ['<table><tr><th>#</th><th>module</th><th>animal / session</th>'
           '<th>commit</th><th>seed</th><th>dataset fingerprint</th>'
           '<th>decision</th></tr>']
    for iteration in data.iterations:
        seed = (f'<span class="mono">{_e(iteration["seed"])}</span>'
                if iteration.get('seed') else
                '<span class="flag">not recorded</span> '
                '<span class="sub">(module hardcodes it)</span>')
        fingerprint = (f'<span class="mono">{_e(iteration["dataset_fingerprint"][:12])}</span>'
                       if iteration.get('dataset_fingerprint') else
                       '<span class="flag">not recorded</span>')
        decision = iteration.get('scientist_decision') or 'pending'
        decision_html = (f'<span class="flag">{_e(decision)}</span>'
                         if decision == 'pending' else _e(decision))
        held = ' <span class="sub">[held out]</span>' if iteration.get('held_out') else ''
        out.append(
            f'<tr><td class="num">{_fmt(iteration["id"])}</td>'
            f'<td class="mono">{_e(iteration["analysis_module"])}</td>'
            f'<td>{_e(iteration.get("animal_id"))} / {_e(iteration.get("session_id"))}{held}</td>'
            f'<td class="mono">{_e((iteration.get("git_commit") or "")[:8])}</td>'
            f'<td>{seed}</td><td>{fingerprint}</td><td>{decision_html}</td></tr>')
    out.append('</table>')

    out.append('<h3>Parameters</h3>')
    for iteration in data.iterations:
        params = json.dumps(iteration.get('params') or {}, indent=2, sort_keys=True,
                            default=str)
        out.append(f'<p class="sub">iteration {_fmt(iteration["id"])} '
                   f'({_e(iteration["analysis_module"])})</p>'
                   f'<pre class="mono">{_e(params)}</pre>')

    out.append('<h3>Deviations from the frozen prediction</h3>')
    if data.deviations:
        out.append('<ul>' + ''.join(f'<li>{_e(d)}</li>' for d in data.deviations) + '</ul>')
    elif data.current_prediction is None:
        out.append('<p class="missing">No frozen prediction to compare against.</p>')
    else:
        out.append('<p>None detected.</p>')
    return '\n'.join(out)


def _section_figures(data: ReportData) -> str:
    if not data.figures:
        return _empty('<strong>No figures were saved for this analysis.</strong> '
                      'Nothing here can be checked visually.')
    out = []
    for figure in data.figures:
        if figure.data_uri:
            out.append(f'<figure><img alt="{_e(figure.caption)}" src="{figure.data_uri}">'
                       f'<figcaption>{_e(figure.caption)}</figcaption></figure>')
        else:
            note = figure.skipped_reason or 'not embedded'
            out.append(_empty(
                f'Figure <span class="mono">{_e(figure.path)}</span> &mdash; {_e(note)}.'))
    return '\n'.join(out)


def _section_statistic(data: ReportData) -> str:
    if not data.families:
        return _empty('<strong>No test family is attached</strong>, so no corrected '
                      'statistic can be computed and the number of tests behind any '
                      'claim here is unknown.')
    out = []
    for block in data.families:
        denom = block['denominator']
        out.append(f'<h3>Test family {_fmt(block["family_id"])}</h3>')

        # The mandatory denominator sub-block.
        resolution = block['fdr_resolution']
        resolution_html = ''
        if resolution:
            verdict = ('resolvable' if resolution['resolvable'] else
                       '<span class="flag">NOT resolvable</span>')
            resolution_html = (
                f'<br>FDR resolution at {_fmt(block["n_shuffles"])} shuffles: {verdict} '
                f'(best achievable q {_fmt(resolution["best_achievable_q"], 3)}; '
                f'{_fmt(resolution["recommended_n_shuffles"])} shuffles would be needed)')
        status = denom['denominator_status']
        status_html = (_e(status) if status == 'clean'
                       else f'<span class="flag">{_e(status)}</span>')
        out.append(
            '<div class="denominator"><strong>Denominator</strong><br>'
            f'declared {_fmt(denom["n_declared"])} test(s); '
            f'{_fmt(denom["n_run"])} run, {_fmt(denom["n_abandoned"])} abandoned, '
            f'{_fmt(denom["n_excluded_prespecified"])} excluded a priori '
            f'&rarr; <strong>{_fmt(denom["n_tests_for_correction"])} tests used for '
            'correction</strong><br>'
            f'status: {status_html}'
            f'{resolution_html}</div>')

        if denom['denominator_status'] == 'undeclared':
            out.append(_empty(
                'This family <strong>declares no tests</strong>, so the denominator its '
                'q-values should be divided by is <strong>unrecorded</strong> &mdash; not '
                'zero, and not one. Any corrected statistic below is uninterpretable '
                'until the family is declared or reconstructed.'))
        elif denom['denominator_status'] == 'reconstructed':
            out.append(_empty(
                'This denominator was <strong>reconstructed after the fact</strong> from '
                'what happened to be logged. Unlogged parameter sweeps and informal '
                'exploration are unrecoverable, so the true number of tests is a '
                '<strong>lower bound</strong>, not a count.'))

        out.append('<table><tr><th>test</th><th>p</th><th>q as logged</th>'
                   f'<th>q at declared m={_fmt(denom["n_tests_for_correction"])}</th>'
                   '<th>significant</th><th>commit</th></tr>')
        for row in block['rows']:
            logged = row['q_as_logged']
            declared = row['q_at_declared']
            differs = (logged is not None and declared is not None
                       and abs(logged - declared) > 1e-9)
            logged_html = ('&mdash;' if logged is None else
                           (f'<span class="flag">{logged:.4f}</span>' if differs
                            else f'{logged:.4f}'))
            significant = ('yes' if row['significant_at_declared']
                           else '<span class="flag">no</span>')
            out.append(f'<tr><td class="mono">{_e(row["test_key"])}</td>'
                       f'<td class="num">{_fmt(row["p_value"], 6)}</td>'
                       f'<td class="num">{logged_html}</td>'
                       f'<td class="num">{_fmt(declared)}</td>'
                       f'<td>{significant}</td>'
                       f'<td class="mono">{_e((row.get("git_commit") or "")[:8])}</td></tr>')
        out.append('</table>')
        if block['n_padded']:
            out.append(f'<p class="sub">{_fmt(block["n_padded"])} declared test(s) '
                       'produced no p-value and were entered as p=1.0, which is '
                       'conservative Benjamini-Hochberg at the declared denominator.</p>')

        if block['exclusions']:
            out.append('<h3>Exclusions</h3><table>'
                       '<tr><th>test</th><th>status</th><th>reason</th>'
                       '<th>outcome-dependent?</th><th>applied after seeing results?</th>'
                       '</tr>')
            for exclusion in block['exclusions']:
                dependent = ('<span class="flag">YES</span>'
                             if exclusion['outcome_dependent'] else 'no')
                after = ('<span class="flag">yes</span>'
                         if exclusion['applied_after_seeing_results'] else 'no')
                out.append(f'<tr><td class="mono">{_e(exclusion["test_key"])}</td>'
                           f'<td>{_e(exclusion["status"])}</td>'
                           f'<td>{_e(exclusion["reason"])}</td>'
                           f'<td>{dependent}</td><td>{after}</td></tr>')
            out.append('</table>')
    return '\n'.join(out)


def _section_verdict(data: ReportData) -> str:
    out = []
    verdict = data.verdict
    if verdict is None:
        out.append('<div class="verdict"><span class="label">No verdict recorded</span>'
                   'Nobody has made a call on this hypothesis. That is not the same as '
                   'an inconclusive result.</div>')
    else:
        css = _e(verdict['verdict'])
        label = _e(verdict['verdict'].replace('_', ' '))
        out.append(
            f'<div class="verdict {css}"><span class="label">{label}</span>'
            f'{_e(verdict["rationale"])}'
            f'<p class="sub">tier: {_e(verdict["tier"])} &middot; decided by '
            f'{_e(verdict["decided_by"])} &middot; {_e(verdict["created_at"])} &middot; '
            f'denominator known: {_fmt(verdict.get("denominator_known"))}'
            + (f' ({_fmt(verdict["n_tests_in_denominator"])} tests)'
               if verdict.get('n_tests_in_denominator') is not None else '')
            + '</p></div>')

    tier = data.tier or {}
    if not tier.get('is_confirmatory'):
        out.append(_banner())
        reasons = tier.get('blocking_reasons') or ()
        if reasons:
            out.append('<p class="sub">Unmet conditions for a confirmatory tier:</p>'
                       '<ul class="checklist">'
                       + ''.join(f'<li>{_e(r)}</li>' for r in reasons) + '</ul>')
    else:
        out.append('<p class="sub">Tier: <strong>confirmatory</strong> &mdash; '
                   'pre-registered, run against a reserved session unlocked for this '
                   'hypothesis alone, and reproducible.</p>')
    return '\n'.join(out)


def _section_threats(data: ReportData) -> str:
    if not data.threats:
        return ('<p>No automated threat was detected. That is not the same as there '
                'being none: this checklist only covers traps that have already been '
                'encoded.</p>')
    return ('<ul class="checklist">'
            + ''.join(f'<li>{_e(t)}</li>' for t in data.threats) + '</ul>')


def _section_falsifier(data: ReportData) -> str:
    return ('<ul class="checklist">'
            + ''.join(f'<li>{_e(f)}</li>' for f in data.falsifiers) + '</ul>')


_RENDERERS = {
    'claim': _section_claim,
    'prediction': _section_prediction,
    'ran': _section_ran,
    'figures': _section_figures,
    'statistic': _section_statistic,
    'verdict': _section_verdict,
    'threats': _section_threats,
    'falsifier': _section_falsifier,
}


def render_html(data: ReportData) -> str:
    """Render a :class:`ReportData` to a complete self-contained document.

    Pure: no database access, no filesystem access, and byte-deterministic for
    a given ``ReportData``.
    """
    parts = [
        '<!DOCTYPE html>', '<html lang="en">', '<head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<title>{_e(data.title)}</title>',
        f'<style>{_CSS}</style>', '</head>', '<body>',
        f'<h1>{_e(data.title)}</h1>',
    ]
    if data.hypothesis.get('statement'):
        parts.append(f'<p class="sub">{_e(data.hypothesis["statement"])}</p>')

    if data.hypothesis_generating_only:
        parts.append(_banner())

    parts.append('<p class="sub">Contents: ' + ' &middot; '.join(
        f'<a href="#{anchor}">{index}. {_e(title)}</a>'
        for index, (anchor, title) in enumerate(SECTIONS, start=1)) + '</p>')

    for index, (anchor, title) in enumerate(SECTIONS, start=1):
        parts.append(_heading(index, anchor, title))
        parts.append(_RENDERERS[anchor](data))

    parts.append(
        f'<div class="footer">Generated {_e(data.generated_at)}'
        + (f' at commit <span class="mono">{_e(data.git_commit)}</span>'
           if data.git_commit else '')
        + f' from <span class="mono">{_e(data.db_path)}</span>.'
        ' Self-contained: no external requests.</div>')
    parts.extend(['</body>', '</html>'])
    return '\n'.join(parts) + '\n'


def render_index_html(index: Mapping[str, Any]) -> str:
    """Render the index over every hypothesis, dropped ones included."""
    parts = [
        '<!DOCTYPE html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
        '<title>Hypothesis index</title>', f'<style>{_CSS}</style>', '</head>',
        '<body>', '<h1>Hypothesis index</h1>',
        '<p class="sub">Every hypothesis on record, including refuted and blocked '
        'ones. A hypothesis that was quietly dropped has no report of its own, so '
        'this list is the only place its absence is visible.</p>',
    ]
    rows = index.get('rows') or []
    if not rows:
        parts.append(_empty('No hypotheses are registered.'))
    else:
        parts.append('<table><tr><th>#</th><th>statement</th><th>status</th>'
                     '<th>tier</th><th>verdict</th><th>iterations</th>'
                     '<th>denominator</th></tr>')
        for row in rows:
            verdict = row.get('verdict')
            verdict_html = (f'<span class="flag">{_e(verdict.replace("_", " "))}</span>'
                            if verdict in ('refuted', 'blocked')
                            else (_e(verdict.replace('_', ' ')) if verdict
                                  else '<span class="missing">none</span>'))
            tier = row.get('tier')
            tier_html = (_e(tier) if tier == 'confirmatory'
                         else f'<span class="flag">{_e(tier)}</span>')
            statuses = row.get('denominator_statuses') or []
            statuses_html = ', '.join(
                _e(s) if s == 'clean' else f'<span class="flag">{_e(s)}</span>'
                for s in statuses) or '<span class="missing">none</span>'
            parts.append(
                f'<tr><td class="num">{_fmt(row["id"])}</td>'
                f'<td><a href="hypothesis_{_fmt(row["id"])}.html">'
                f'{_e((row.get("statement") or "")[:140])}</a></td>'
                f'<td>{_e(row.get("status"))}</td><td>{tier_html}</td>'
                f'<td>{verdict_html}</td>'
                f'<td class="num">{_fmt(row.get("n_iterations"))}</td>'
                f'<td>{statuses_html}</td></tr>')
        parts.append('</table>')

    orphans = index.get('orphan_iteration_ids') or []
    if orphans:
        parts.append(_empty(
            f'<strong>{len(orphans)} logged iteration(s) are attached to no hypothesis '
            f'at all</strong> (ids: {_e(", ".join(str(i) for i in orphans))}). They are '
            'part of the search that produced everything above, and they are invisible '
            'to any per-hypothesis correction.'))

    if not any(row.get('tier') == 'confirmatory' for row in rows):
        parts.append(_banner())

    parts.append(f'<div class="footer">Generated {_e(index.get("generated_at"))} from '
                 f'<span class="mono">{_e(index.get("db_path"))}</span>.</div>')
    parts.extend(['</body>', '</html>'])
    return '\n'.join(parts) + '\n'


# --------------------------------------------------------------- IO layer

DEFAULT_OUT_DIR = Path('reports/out')


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 explicitly: at least one statement in this notebook already carries
    # a replacement character from a prior mojibake, and the default codepage on
    # this workstation would mangle it further or raise.
    path.write_text(text, encoding='utf-8')
    return path


def build_hypothesis_report(hypothesis_id: int, *, notebook=None,
                            out_dir: Path = DEFAULT_OUT_DIR,
                            embed_figures: bool = True,
                            now: datetime = None,
                            git_commit: str = None) -> Path:
    from database.lab_notebook import LabNotebook, _git_commit

    nb = notebook or LabNotebook()
    data = collect_report_data(nb, hypothesis_id, now=now,
                              git_commit=git_commit or _git_commit(),
                              embed_figures=embed_figures)
    return _write(Path(out_dir) / f'hypothesis_{hypothesis_id}.html', render_html(data))


def build_orphan_report(iteration_ids: Sequence[int], *, title: str, notebook=None,
                        out_dir: Path = DEFAULT_OUT_DIR, embed_figures: bool = True,
                        now: datetime = None, git_commit: str = None,
                        filename: str = None) -> Path:
    from database.lab_notebook import LabNotebook, _git_commit

    nb = notebook or LabNotebook()
    data = collect_orphan_report_data(nb, iteration_ids, title=title, now=now,
                                      git_commit=git_commit or _git_commit(),
                                      embed_figures=embed_figures)
    name = filename or ('iterations_' + '_'.join(str(i) for i in iteration_ids) + '.html')
    return _write(Path(out_dir) / name, render_html(data))


def build_index_report(*, notebook=None, out_dir: Path = DEFAULT_OUT_DIR,
                       now: datetime = None, git_commit: str = None) -> Path:
    from database.lab_notebook import LabNotebook, _git_commit

    nb = notebook or LabNotebook()
    index = collect_index_data(nb, now=now, git_commit=git_commit or _git_commit())
    return _write(Path(out_dir) / 'index.html', render_index_html(index))
