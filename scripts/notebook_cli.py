#!/usr/bin/env python
"""
Command-line surface for the lab notebook: holdout reservations, the
multiple-comparisons ledger, pre-registration, verdicts, and report generation.

Exists mainly to remove friction. ``scientist_decision`` sat at ``'pending'``
on every logged iteration not because anyone decided to skip the gate, but
because recording a decision meant opening a Python session. A one-line
command is the fix::

    python scripts/notebook_cli.py decide 7 approved -m "real, corrected"

Style follows ``database/database_cli.py`` and ``scripts/phase0_probe.py``.

Reading commands are safe to run any time. Writing commands (``reserve-holdout``,
``declare-family``, ``freeze``, ``verdict``, ``decide``, ``abandon-test``) change
the notebook, and the notebook is a single gitignored SQLite file with no other
copy — ``--snapshot`` is available on the destructive-ish ones and
``scripts/backfill_ledger.py`` always takes one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.lab_notebook import (  # noqa: E402
    HYPOTHESIS_STATUSES,
    VERDICTS,
    HoldoutIndeterminate,
    HoldoutViolation,
    LabNotebook,
    TestBudgetExhausted,
    UndeclaredTestError,
)


def _notebook(args) -> LabNotebook:
    return LabNotebook(args.db) if getattr(args, 'db', None) else LabNotebook()


def _snapshot(nb: LabNotebook) -> Path:
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    target = nb.db_path.with_name(f"{nb.db_path.name}.bak_{stamp}")
    shutil.copy2(nb.db_path, target)
    print(f"snapshot: {target}")
    return target


def _emit(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))


# ----------------------------------------------------------------- holdout

def cmd_reserve_holdout(args) -> int:
    nb = _notebook(args)
    reservation = nb.reserve_holdout(
        args.session_id, cohort=args.cohort, reason=args.reason,
        reserved_by=args.by, animal_id=args.animal)
    scope_note = ('' if args.animal is None else
                  "\n  NOTE: animal-scoped. Any multi-animal analysis "
                  "(decode_location with a partner, run_inter_brain, "
                  "social_spatial_fields, decode_partner_distance) reads this animal "
                  "through a focal animal, so the whole session is blocked for those.")
    print(f"reserved {reservation.cohort}/{reservation.session_key} "
          f"(id {reservation.id}){scope_note}")
    return 0


def cmd_release_holdout(args) -> int:
    nb = _notebook(args)
    reservation = nb.release_holdout(args.reservation_id, reason=args.reason,
                                     released_by=args.by)
    print(f"released reservation {reservation.id} (the record is kept)")
    return 0


def cmd_unlock_holdout(args) -> int:
    nb = _notebook(args)
    reservation = nb.unlock_holdout(
        args.reservation_id, args.hypothesis_id, approved_by=args.by,
        frozen_prediction_id=args.prediction)
    print(f"reservation {reservation.id} unlocked for hypothesis "
          f"{reservation.unlocked_for_hypothesis_id} only")
    return 0


def cmd_list_holdout(args) -> int:
    nb = _notebook(args)
    rows = nb.list_holdout(active_only=not args.all)
    if not rows:
        print("no holdout reservations"
              + ("" if args.all else " (nothing is currently reserved)"))
        return 0
    for row in rows:
        state = 'active' if row.active else f"released {row.released_at}"
        unlocked = (f" unlocked for hypothesis {row.unlocked_for_hypothesis_id}"
                    if row.unlocked_for_hypothesis_id else '')
        print(f"[{row.id}] {row.cohort}/{row.session_key} "
              f"{row.animal_id or '(whole session)'} - {state}{unlocked}")
        print(f"      {row.reason}  (by {row.reserved_by})")
    return 0


def cmd_check_holdout(args) -> int:
    nb = _notebook(args)
    try:
        status = nb.holdout_status(args.session_id, args.animal,
                                   multi_animal=args.multi_animal)
    except HoldoutIndeterminate as exc:
        print(f"INDETERMINATE: {exc}")
        return 2
    print(status.summary())
    return 1 if status.held_out else 0


# ------------------------------------------------------------------ ledger

def cmd_declare_family(args) -> int:
    nb = _notebook(args)
    family = nb.get_or_create_test_family(args.name, alpha=args.alpha)
    keys = args.keys or [line.strip() for line in sys.stdin if line.strip()]
    created = nb.declare_family_tests(
        family.id, keys, declared_by=args.by, budget_max_tests=args.budget,
        denominator_status=args.denominator_status, extend=args.extend)
    print(f"family {family.id} ({family.name!r}): declared {len(created)} new test(s)")
    print(json.dumps(nb.family_denominator(family.id), indent=2, default=str))
    return 0


def cmd_abandon_test(args) -> int:
    nb = _notebook(args)
    test = nb.abandon_family_test(
        args.family_id, args.test_key, reason=args.reason,
        outcome_dependent=args.outcome_dependent,
        criterion_available_a_priori=args.a_priori,
        applied_after_seeing_results=args.after_results, actor=args.by)
    print(f"{test.test_key}: {test.status} "
          f"(outcome-dependent: {test.exclusion_outcome_dependent})")
    if test.exclusion_outcome_dependent:
        print("  This test STAYS in the denominator: deciding to drop it because of "
              "what a result showed was itself a test.")
    return 0


def cmd_extend_budget(args) -> int:
    nb = _notebook(args)
    family = nb.extend_family_budget(args.family_id, args.budget, actor=args.by,
                                     reason=args.reason)
    print(f"family {family.id} budget -> {family.budget_max_tests}")
    return 0


def cmd_denominator(args) -> int:
    nb = _notebook(args)
    denom = nb.family_denominator(args.family_id)
    print(json.dumps(denom, indent=2, default=str))
    if denom['denominator_status'] != 'clean':
        print(f"\nWARNING: denominator status is {denom['denominator_status']!r}. "
              "Do not pass a hand-counted n_tests to fdr_resolution - use "
              "n_tests_for_correction above.")
    return 0


def cmd_ledger(args) -> int:
    nb = _notebook(args)
    result = nb.family_fdr(args.family_id, n_shuffles=args.n_shuffles)
    denom = result['denominator']
    print(f"family {args.family_id}: {denom['n_declared']} declared, "
          f"{denom['n_run']} run, {denom['n_abandoned']} abandoned, "
          f"{denom['n_excluded_prespecified']} excluded a priori")
    print(f"  m used for correction: {result['n_tests_for_correction']} "
          f"(status: {denom['denominator_status']})")
    if result.get('fdr_resolution'):
        resolution = result['fdr_resolution']
        print(f"  resolvable at {args.n_shuffles} shuffles: {resolution['resolvable']}"
              f" (best achievable q {resolution['best_achievable_q']:.4f}; "
              f"{resolution['recommended_n_shuffles']} would be needed)")
    if result['n_padded']:
        print(f"  {result['n_padded']} declared test(s) entered as p=1.0")
    print()
    for key, values in sorted(result['per_test'].items()):
        mark = '*' if values['significant'] else ' '
        print(f" {mark} {key:36s} p={values['p_value']:.6f}  q={values['q_value']:.4f}")
    return 0


# --------------------------------------------------------- pre-registration

def cmd_freeze(args) -> int:
    nb = _notebook(args)
    prediction = nb.freeze_prediction(
        args.hypothesis_id, statistic=args.statistic, direction=args.direction,
        threshold=args.threshold, falsifier=args.falsifier, alpha=args.alpha,
        n_shuffles_planned=args.n_shuffles, test_family_id=args.family,
        declared_test_keys=args.keys, holdout_kind=args.holdout_kind,
        registered_post_hoc=args.post_hoc)
    comparison = '<' if prediction.direction == 'lt' else '>'
    print(f"frozen prediction {prediction.id} (v{prediction.version}) for hypothesis "
          f"{prediction.hypothesis_id}: {prediction.statistic} {comparison} "
          f"{prediction.threshold}")
    if prediction.registered_post_hoc:
        print("  registered POST HOC: records what was claimed, cannot support a "
              "confirmatory tier")
    return 0


def cmd_verdict(args) -> int:
    nb = _notebook(args)
    row = nb.record_verdict(
        args.hypothesis_id, verdict=args.verdict, rationale=args.rationale,
        holdout_iteration_id=args.iteration,
        denominator_known=args.denominator_known,
        n_tests_in_denominator=args.n_tests, decided_by=args.by)
    print(f"hypothesis {row.hypothesis_id}: {row.verdict} (tier {row.tier}, derived)")
    return 0


def cmd_status(args) -> int:
    nb = _notebook(args)
    hypothesis = nb.set_hypothesis_status(args.hypothesis_id, args.status,
                                          notes=args.notes)
    print(f"hypothesis {hypothesis.id}: status {hypothesis.status}")
    return 0


def cmd_decide(args) -> int:
    """The one-liner whose absence left every iteration at 'pending'."""
    nb = _notebook(args)
    iteration = nb.record_decision(args.iteration_id, args.decision, notes=args.notes)
    print(f"iteration {iteration.id}: {iteration.scientist_decision}")
    return 0


def cmd_tier(args) -> int:
    nb = _notebook(args)
    print(nb.evidence_tier(args.hypothesis_id).summary())
    return 0


# ----------------------------------------------------------------- reports

def cmd_report(args) -> int:
    from reports.hypothesis_report import build_hypothesis_report, build_orphan_report

    nb = _notebook(args)
    out_dir = Path(args.out)
    if args.iterations:
        ids = [int(i) for i in args.iterations.split(',')]
        path = build_orphan_report(ids, title=args.title or f"Iterations {ids}",
                                   notebook=nb, out_dir=out_dir,
                                   embed_figures=not args.no_figures)
    elif args.hypothesis is not None:
        path = build_hypothesis_report(args.hypothesis, notebook=nb, out_dir=out_dir,
                                       embed_figures=not args.no_figures)
    else:
        print("pass --hypothesis N or --iterations 10,11,12", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


def cmd_index(args) -> int:
    from reports.hypothesis_report import build_index_report

    nb = _notebook(args)
    path = build_index_report(notebook=nb, out_dir=Path(args.out))
    print(f"wrote {path}")
    return 0


def cmd_summary(args) -> int:
    nb = _notebook(args)
    print(repr(nb))
    print()
    for hypothesis in nb.list_hypotheses():
        tier = nb.evidence_tier(hypothesis.id)
        verdict = nb.latest_verdict(hypothesis.id)
        iterations = nb.iterations_for_hypothesis(hypothesis.id)
        print(f"[{hypothesis.id}] {hypothesis.status:10s} {tier.tier:13s} "
              f"{(verdict.verdict if verdict else 'no verdict'):28s} "
              f"{len(iterations)} iteration(s)")
        print(f"     {hypothesis.statement[:110]}")
    from database.lab_notebook import Iteration
    with nb.get_db_session() as db_session:
        pending = [row.id for row in db_session.query(Iteration).filter(
            Iteration.scientist_decision.in_(('pending', None))
        ).order_by(Iteration.id).all()]
        orphans = [row.id for row in db_session.query(Iteration).filter(
            Iteration.hypothesis_id.is_(None)).order_by(Iteration.id).all()]

    if pending:
        print(f"\n{len(pending)} iteration(s) awaiting a scientist decision: {pending}")
        print("  record one with: notebook_cli.py decide <id> approved -m '...'")
    if orphans:
        print(f"\n{len(orphans)} iteration(s) attached to no hypothesis: {orphans}")
        print("  these are part of the search but invisible to per-hypothesis "
              "correction")
    return 0


# -------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='notebook_cli.py',
        description=__doc__.strip().splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', help='path to habitat_pipeline.db (default: ./)')
    sub = parser.add_subparsers(dest='command', required=True)

    # -- holdout
    p = sub.add_parser('reserve-holdout', help='reserve a session as held-out')
    p.add_argument('session_id')
    p.add_argument('--cohort', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--by', required=True)
    p.add_argument('--animal', default=None,
                   help='animal-scoped reservation (prefer whole-session)')
    p.set_defaults(func=cmd_reserve_holdout)

    p = sub.add_parser('release-holdout', help='stop a reservation blocking (kept on record)')
    p.add_argument('reservation_id', type=int)
    p.add_argument('--reason', required=True)
    p.add_argument('--by', required=True)
    p.set_defaults(func=cmd_release_holdout)

    p = sub.add_parser('unlock-holdout',
                       help='open a reservation for one hypothesis\'s confirmation')
    p.add_argument('reservation_id', type=int)
    p.add_argument('hypothesis_id', type=int)
    p.add_argument('--prediction', type=int, required=True,
                   help='frozen prediction id (must not be post hoc)')
    p.add_argument('--by', required=True)
    p.set_defaults(func=cmd_unlock_holdout)

    p = sub.add_parser('list-holdout', help='list reservations')
    p.add_argument('--all', action='store_true', help='include released ones')
    p.set_defaults(func=cmd_list_holdout)

    p = sub.add_parser('check-holdout', help='is this session off-limits?')
    p.add_argument('session_id')
    p.add_argument('--animal', default=None)
    p.add_argument('--multi-animal', action='store_true',
                   help='set for any analysis reading more than one animal')
    p.set_defaults(func=cmd_check_holdout)

    # -- ledger
    p = sub.add_parser('declare-family',
                       help='fix a test family\'s membership before running it')
    p.add_argument('name')
    p.add_argument('keys', nargs='*', help='test keys (or pipe them on stdin)')
    p.add_argument('--by', required=True)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--budget', type=int, default=None)
    p.add_argument('--denominator-status', default='clean',
                   choices=('clean', 'reconstructed'))
    p.add_argument('--extend', action='store_true',
                   help='add to an already-declared family (recorded as an extension)')
    p.set_defaults(func=cmd_declare_family)

    p = sub.add_parser('abandon-test', help='drop a declared test, on the record')
    p.add_argument('family_id', type=int)
    p.add_argument('test_key')
    p.add_argument('--reason', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--outcome-dependent', dest='outcome_dependent',
                       action='store_true',
                       help='the reason came from looking at a result')
    group.add_argument('--not-outcome-dependent', dest='outcome_dependent',
                       action='store_false',
                       help='the criterion is independent of any result')
    p.add_argument('--a-priori', dest='a_priori', action='store_true', default=None,
                   help='the criterion was computable before running anything')
    p.add_argument('--after-results', dest='after_results', action='store_true',
                   default=None, help='it was applied after seeing results')
    p.add_argument('--by', default=None)
    p.set_defaults(func=cmd_abandon_test)

    p = sub.add_parser('extend-budget', help='raise a family\'s declared size')
    p.add_argument('family_id', type=int)
    p.add_argument('budget', type=int)
    p.add_argument('--reason', required=True)
    p.add_argument('--by', required=True)
    p.set_defaults(func=cmd_extend_budget)

    p = sub.add_parser('denominator', help='the denominator, computed by the ledger')
    p.add_argument('family_id', type=int)
    p.set_defaults(func=cmd_denominator)

    p = sub.add_parser('ledger', help='BH-FDR over the declared family')
    p.add_argument('family_id', type=int)
    p.add_argument('--n-shuffles', type=int, default=None,
                   help='also report the resolution verdict at this budget')
    p.set_defaults(func=cmd_ledger)

    # -- pre-registration and verdicts
    p = sub.add_parser('freeze', help='pre-register a prediction')
    p.add_argument('hypothesis_id', type=int)
    p.add_argument('--statistic', required=True)
    p.add_argument('--direction', required=True, choices=('lt', 'gt'))
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--falsifier', required=True,
                   help='what result would count against this')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n-shuffles', type=int, default=None)
    p.add_argument('--family', type=int, default=None)
    p.add_argument('--keys', nargs='*', default=None)
    p.add_argument('--holdout-kind', default='replication',
                   choices=('replication', 'generalization'))
    p.add_argument('--post-hoc', action='store_true',
                   help='transcribing a claim from an analysis that already ran')
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser('verdict', help='record a verdict (refutations included)')
    p.add_argument('hypothesis_id', type=int)
    p.add_argument('verdict', choices=VERDICTS)
    p.add_argument('--rationale', required=True)
    p.add_argument('--iteration', type=int, default=None)
    p.add_argument('--denominator-known', action='store_true')
    p.add_argument('--n-tests', type=int, default=None)
    p.add_argument('--by', default='scientist')
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser('status', help='set a hypothesis status')
    p.add_argument('hypothesis_id', type=int)
    p.add_argument('status', choices=HYPOTHESIS_STATUSES)
    p.add_argument('--notes', '-m', default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('decide', help='record the scientist decision on an iteration')
    p.add_argument('iteration_id', type=int)
    p.add_argument('decision', choices=('approved', 'rejected', 'pending'))
    p.add_argument('--notes', '-m', default=None)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser('tier', help='why a hypothesis is exploratory or confirmatory')
    p.add_argument('hypothesis_id', type=int)
    p.set_defaults(func=cmd_tier)

    # -- reports
    p = sub.add_parser('report', help='write a self-contained HTML report')
    p.add_argument('--hypothesis', type=int, default=None)
    p.add_argument('--iterations', default=None,
                   help='comma-separated iteration ids with no hypothesis attached')
    p.add_argument('--title', default=None)
    p.add_argument('--out', default='reports/out')
    p.add_argument('--no-figures', action='store_true')
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('index', help='write the index over every hypothesis')
    p.add_argument('--out', default='reports/out')
    p.set_defaults(func=cmd_index)

    p = sub.add_parser('summary', help='one-screen state of the notebook')
    p.set_defaults(func=cmd_summary)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (HoldoutViolation, HoldoutIndeterminate) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3
    except (UndeclaredTestError, TestBudgetExhausted) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 4
    except (ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
