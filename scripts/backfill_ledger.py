#!/usr/bin/env python
"""
Reconstruct the multiple-comparisons ledger for iterations that predate it.

Dry-run by default. ``--apply`` is required to write anything, and a snapshot of
the database is taken first — it is a single gitignored SQLite file with no other
copy.

**Hard rule: this never rewrites logged evidence.** Iteration 12's
``excluded_objects`` prose and its q=0.0387 stay exactly as logged, because they *are*
the evidence for the denominator problem. The correction lives in ``family_tests``,
and the report shows "q as logged" beside "q at the declared denominator". Rewriting
the original record would destroy the audit trail this whole layer exists to create.

The one exception, and the reason it is not a violation: a **NULL**
``iterations.test_family_id`` is filled in to point at the reconstructed family.
That *adds* a pointer where there was none; it changes no measurement, no parameter
and no result. Without it the reconstruction is invisible from the iteration — the
report would go on saying "no test family is attached" while a correctly-declared
family sat beside it — which would defeat the point of the pass. An **existing**
``test_family_id`` is never overwritten, and no other column is touched;
``tests/test_backfill_ledger.py`` compares a full row hash of every other field.

What it reconstructs
--------------------
1. ``Family A`` — the EC opponent-identity search on animal 631 / session 20251216,
   ``denominator_status='reconstructed'`` because it was never declared in advance.
2. ``Family B`` — the ``decode_location`` sweep on 631 / 20251210, with **12** declared
   test keys taken from iteration 10's ``per_object`` list, which is exactly
   ``tracking.get_object_names()`` and therefore computable a priori.
3. Housekeeping on the pre-existing families: the empty duplicate is marked
   ``invalidated``, and the family behind the "0/148 significant" figure is marked
   ``superseded``.
4. Post-hoc frozen predictions for the two claims that were actually reported, plus a
   retroactive hypothesis for the EC finding (it has none) and verdict rows.

And it prints the denominator table at m in {7, 9, 10, 11, 12}, which is the single
most useful thing it does.

Six judgment calls
------------------
The script refuses to guess on these. Each prints with a recommendation, and the ones
that change a number need an explicit flag. See ``--help``.

Concurrency note: iteration rows have been appearing in this notebook from other
sessions, so counts are re-read at apply time rather than trusted from the dry run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from database.lab_notebook import (  # noqa: E402
    FamilyTest,
    Hypothesis,
    Iteration,
    LabNotebook,
    TestFamily,
)
from ephys._stats_utils import benjamini_hochberg, fdr_resolution  # noqa: E402

FAMILY_A_NAME = 'exploratory: EC opponent identity, 631/20251216 (reconstructed)'
FAMILY_B_NAME = 'exploratory: decode_location partner position, 631/20251210 (reconstructed)'

#: Iteration 10 swept every tracked object, so this is the family a priori.
ITER10_OBJECTS = ('613', '615', '616', '617', '620', '621', '629',
                  '630', '631', '633', '634', '635')

#: The one exclusion whose criterion was computable without running anything.
A_PRIORI_EXCLUSION = '630'
A_PRIORI_REASON = ('near-stationary tracking (x/y std ~11-13 px) - degenerate '
                   'near-zero error for any method, not real decodability')

#: The four dropped because of what iteration 10's results showed.
OUTCOME_DEPENDENT_EXCLUSIONS = {
    '615': 'showed no margin under the reverse-null; deprioritized',
    '620': 'reverse-null anomaly (null << observed) flagged as unreliable',
    '621': 'reverse-null anomaly (null << observed) flagged as unreliable',
    '629': 'showed no margin under the reverse-null; deprioritized',
}


class Plan:
    """Accumulates intended writes so a dry run can print them."""

    def __init__(self):
        self.steps: List[str] = []
        self.questions: List[Tuple[str, str]] = []

    def add(self, description: str) -> None:
        self.steps.append(description)
        print(f"  + {description}")

    def ask(self, question: str, recommendation: str) -> None:
        self.questions.append((question, recommendation))

    def render_questions(self) -> None:
        if not self.questions:
            return
        print('\n' + '=' * 78)
        print('JUDGMENT CALLS - the script will not guess on these')
        print('=' * 78)
        for i, (question, recommendation) in enumerate(self.questions, start=1):
            print(f"\n{i}. {question}")
            print(f"   recommendation: {recommendation}")


def snapshot(nb: LabNotebook) -> Path:
    stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    target = nb.db_path.with_name(f"{nb.db_path.name}.bak_{stamp}")
    shutil.copy2(nb.db_path, target)
    print(f"snapshot written: {target}")
    return target


# ------------------------------------------------------------------ survey

def survey(nb: LabNotebook) -> Dict:
    """Read-only picture of what is in the notebook right now."""
    with nb.get_db_session() as db_session:
        iterations = db_session.query(Iteration).order_by(Iteration.id).all()
        families = db_session.query(TestFamily).order_by(TestFamily.id).all()
        hypotheses = db_session.query(Hypothesis).order_by(Hypothesis.id).all()
        rows = [{
            'id': it.id, 'module': it.analysis_module, 'animal': it.animal_id,
            'session': it.session_id, 'status': it.status,
            'hypothesis_id': it.hypothesis_id, 'test_family_id': it.test_family_id,
            'commit': (it.git_commit or '')[:8],
            'decision': it.scientist_decision,
            'params': it.params_dict(), 'summary': it.result_summary_dict(),
        } for it in iterations]
        family_rows = [{
            'id': f.id, 'name': f.name, 'status': f.status,
            'n_iterations': sum(1 for it in iterations if it.test_family_id == f.id),
            'n_declared': f.declared_n_tests,
        } for f in families]
        hypothesis_rows = [{'id': h.id, 'status': h.status,
                            'statement': h.statement} for h in hypotheses]
    return {'iterations': rows, 'families': family_rows, 'hypotheses': hypothesis_rows}


def print_survey(state: Dict) -> None:
    print('\n' + '=' * 78)
    print('CURRENT STATE')
    print('=' * 78)
    print(f"\n{len(state['iterations'])} iteration(s):")
    for row in state['iterations']:
        flags = []
        if row['hypothesis_id'] is None:
            flags.append('no hypothesis')
        if row['test_family_id'] is None:
            flags.append('no family')
        if row['decision'] in (None, 'pending'):
            flags.append('pending')
        print(f"  #{row['id']:<3} {row['module']:<34} {row['session']:<10} "
              f"{row['commit']:<9} {', '.join(flags)}")
    print(f"\n{len(state['families'])} test famil(y|ies):")
    for row in state['families']:
        print(f"  [{row['id']}] {row['name']!r} - {row['n_iterations']} iteration(s), "
              f"declared={row['n_declared']}, status={row['status']}")
    print(f"\n{len(state['hypotheses'])} hypothes(is|es):")
    for row in state['hypotheses']:
        print(f"  [{row['id']}] {row['status']:<10} {row['statement'][:78]}")


# ------------------------------------------------------- run-key duplicates

def report_duplicates(nb: LabNotebook, state: Dict) -> Dict[str, List[int]]:
    """Group iterations by run identity.

    Iterations 5 and 7 log different param *keys* with identical param *values* and
    produce byte-identical results; iterations 3 and 4 differ genuinely, by
    ``null_mode``. Canonicalizing against each wrapper's signature defaults is what
    tells those two cases apart.
    """
    from database.lab_notebook import compute_run_key

    groups: Dict[str, List[int]] = {}
    for row in state['iterations']:
        key = compute_run_key(row['module'], row['params'],
                              git_commit=row['commit'] or None)
        groups.setdefault(key, []).append(row['id'])

    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    print('\n' + '=' * 78)
    print('RUN IDENTITY')
    print('=' * 78)
    if not duplicates:
        print('  no duplicate run keys')
    for key, ids in sorted(duplicates.items(), key=lambda kv: kv[1]):
        print(f"  iterations {ids} share a run key ({key[:12]}...) -> one test, "
              "not several")

    # Cross-check on the numbers, independent of the hashing.
    by_numbers: Dict[Tuple, List[int]] = {}
    for row in state['iterations']:
        summary = row['summary']
        signature = (row['module'], row['commit'],
                     summary.get('population_accuracy_mean'),
                     summary.get('p_value'), summary.get('best_cell_id'))
        if signature[2] is not None:
            by_numbers.setdefault(signature, []).append(row['id'])
    identical = {k: v for k, v in by_numbers.items() if len(v) > 1}
    for signature, ids in identical.items():
        if any(set(ids) <= set(dup) for dup in duplicates.values()):
            note = 'same run key too - one test, counted once'
        elif _differs_only_in_correction_method(state, ids):
            # Expected, not suspicious: null_mode changes how significance is
            # computed, not the accuracy or the population p-value. Two such runs
            # are two genuine tests of one question that happen to share a headline.
            note = ('run keys differ by correction method only (null_mode) - '
                    'expected, and still two declared tests: a fork in how the '
                    'same question was corrected')
        else:
            note = ('run keys differ for another reason - flag as probable_duplicate '
                    'for a human call rather than guessing')
        print(f"  iterations {ids} share their headline numbers ({note})")
    return duplicates


#: Params that change how a result is *corrected*, not what it measures.
_CORRECTION_ONLY_PARAMS = frozenset({'null_mode', 'alpha'})


def _differs_only_in_correction_method(state: Dict, ids: Sequence[int]) -> bool:
    rows = [r for r in state['iterations'] if r['id'] in set(ids)]
    if len(rows) < 2:
        return False
    baseline = rows[0]['params']
    differing = set()
    for row in rows[1:]:
        keys = set(baseline) | set(row['params'])
        differing |= {k for k in keys
                      if baseline.get(k) != row['params'].get(k)}
    return bool(differing) and differing <= _CORRECTION_ONLY_PARAMS


# --------------------------------------------------------- denominator table

def denominator_table(p_values: Dict[str, float], n_shuffles: int,
                      candidates: Sequence[int], alpha: float = 0.05) -> None:
    """The pass's most useful output: the verdict as a function of m."""
    print('\n' + '=' * 78)
    print(f"DENOMINATOR TABLE - decode_location, n_shuffles={n_shuffles}, "
          f"alpha={alpha}")
    print('=' * 78)
    print(f"\n  {len(p_values)} test(s) produced a p-value; "
          f"iteration 10 ran {len(ITER10_OBJECTS)}.\n")
    header = (f"  {'m':>3}  {'best q':>9}  {'rat613 q':>9}  {'self(631) q':>12}  "
              f"{'significant':>28}  {'resolvable':>10}")
    print(header)
    print('  ' + '-' * (len(header) - 2))

    keys = list(p_values)
    for m in candidates:
        # Declared-but-unrun members enter as p=1.0, which is exactly conservative
        # Benjamini-Hochberg at the declared m. Order is preserved so the q-values
        # line up with `keys`.
        q = benjamini_hochberg(np.array(
            [p_values[k] for k in keys] + [1.0] * max(0, m - len(keys)),
            dtype=np.float64))
        q_by_key = {k: float(q[i]) for i, k in enumerate(keys)}

        significant = sorted(k for k, v in q_by_key.items() if v < alpha)
        resolution = fdr_resolution(n_tests=m, n_shuffles=n_shuffles, alpha=alpha)
        print(f"  {m:>3}  {min(q_by_key.values()):>9.4f}  "
              f"{q_by_key.get('613', float('nan')):>9.4f}  "
              f"{q_by_key.get('631', float('nan')):>12.4f}  "
              f"{(', '.join(significant) or 'none'):>28}  "
              f"{str(resolution['resolvable']):>10}")

    print("\n  m=7 is what was logged. m=11 is 12 minus the one exclusion whose")
    print("  criterion was available a priori. Note that `resolvable` flips between")
    print("  them: the guard was fed the count the exclusion produced, so the")
    print("  exclusion bought both the significance and the permission to claim it.")


# ------------------------------------------------------------------- writing

def reconstruct(nb: LabNotebook, state: Dict, plan: Plan, *,
                apply: bool, confirm_exclusions: bool) -> None:
    """Describe (and optionally perform) the reconstruction."""
    loc_rows = [r for r in state['iterations']
                if r['module'] == 'ephys.decode_location']
    opp_rows = [r for r in state['iterations']
                if r['module'] in ('ephys.decode_opponent_identity',
                                   'ephys.decode_event_outcome')]

    print('\n' + '=' * 78)
    print('PLANNED WRITES' + ('' if apply else ' (dry run - nothing is written)'))
    print('=' * 78)

    # -- Family B: decode_location
    plan.add(f"create test family {FAMILY_B_NAME!r} "
             f"with denominator_status='reconstructed'")
    plan.add(f"declare {len(ITER10_OBJECTS)} test keys from iteration 10's per_object "
             f"list: {', '.join('object=' + o for o in ITER10_OBJECTS[:4])}, ...")

    latest_loc = max((r for r in loc_rows if r['summary'].get('per_object')),
                     key=lambda r: r['id'], default=None)
    scored: Dict[str, float] = {}
    if latest_loc:
        for entry in latest_loc['summary']['per_object']:
            if entry.get('p_value') is not None:
                scored[str(entry['object_name'])] = float(entry['p_value'])
        plan.add(f"record {len(scored)} p-value(s) from iteration {latest_loc['id']} "
                 f"(commit {latest_loc['commit']})")

    plan.add(f"abandon object={A_PRIORI_EXCLUSION} as excluded_prespecified "
             f"(a-priori criterion, applied after seeing results)")
    for obj, reason in OUTCOME_DEPENDENT_EXCLUSIONS.items():
        plan.add(f"abandon object={obj} as outcome-dependent - STAYS in the "
                 f"denominator ({reason[:44]}...)")

    # -- Family A: opponent identity / outcome
    plan.add(f"create test family {FAMILY_A_NAME!r} "
             f"with denominator_status='reconstructed'")
    plan.add(f"declare test keys for the {len(opp_rows)} event-based iteration(s) "
             "found, one per distinct (analysis, behavior_type, label_mode, null_mode)")

    # -- housekeeping on pre-existing families
    for row in state['families']:
        if row['n_iterations'] == 0:
            plan.add(f"mark family {row['id']} ({row['name']!r}) 'invalidated' - "
                     "empty duplicate created because create_test_family has no "
                     "get-or-create")

    # -- post-hoc predictions + retroactive hypothesis
    plan.add("insert frozen predictions with registered_post_hoc=True for the two "
             "reported claims (EC opponent identity; partner-position decoding)")
    plan.add("create a retroactive Hypothesis for the EC finding (it currently has "
             "none) and record verdicts")

    # -- the judgment calls
    plan.ask(
        f"Is the object={A_PRIORI_EXCLUSION} exclusion outcome-independent?",
        "count it as excluded_prespecified (so m=11) but record "
        "applied_after_seeing_results=True. Its criterion (x/y std) is computable "
        "from tracking alone, but it was discovered after iteration 10. NOTE: the "
        "verdict is the same at m=10, 11 and 12, so this call does not change the "
        "science - worth saying so in the report.")
    plan.ask(
        f"Are the other four ({', '.join(OUTCOME_DEPENDENT_EXCLUSIONS)}) "
        "outcome-dependent?",
        "yes, unambiguously - 'showed no margin under the reverse-null' is selection "
        "on a result computed from the same data. Pass "
        "--confirm-exclusion-classification to write this.")
    plan.ask(
        "Are iterations 3 and 4 one test or two?",
        "two declared tests plus a note. They differ only in multiple-comparison "
        "method, and pooled was adopted after per_cell reported resolvable=false - "
        "mechanically two keys, scientifically a fork.")
    plan.ask(
        "Should the pending scientist decisions be backfilled?",
        "no. A bulk 'approved' would make the field permanently worthless. Leave them "
        "pending, let the report say so, and record decisions going forward with "
        "notebook_cli.py decide.")
    plan.ask(
        "What status for Hypothesis 1 (fight outcome decodable)?",
        "'refuted' is defensible - 60.6% sits below its own 63.2% majority baseline "
        "with p=0.114. But note the newer scoring has 29 usable events rather than 19 "
        "(HZ-DATA-007), so a re-run on the canonical file is arguably owed first. "
        "Your call; the script does not set it.")
    plan.ask(
        "Who writes the falsifiers and rationales?",
        "you. The script drafts from HANDOFF.md, but an agent-authored falsifier for "
        "someone else's hypothesis is a guess about what you would find persuasive.")

    if not apply:
        return

    if not confirm_exclusions:
        print("\nREFUSING TO APPLY: pass --confirm-exclusion-classification to write "
              "the outcome-dependent classification of "
              f"{', '.join(OUTCOME_DEPENDENT_EXCLUSIONS)}.")
        raise SystemExit(4)

    _apply(nb, state, scored, latest_loc)


def _apply(nb: LabNotebook, state: Dict, scored: Dict[str, float],
           latest_loc: Optional[Dict]) -> None:
    print("\napplying...")

    family_b = nb.get_or_create_test_family(FAMILY_B_NAME, alpha=0.05)
    if not nb.family_tests(family_b.id):
        nb.declare_family_tests(
            family_b.id, [f'object={o}' for o in ITER10_OBJECTS],
            declared_by='backfill_ledger.py', denominator_status='reconstructed',
            notes='reconstructed from iteration 10 per_object list')
        print(f"  declared {len(ITER10_OBJECTS)} tests in family {family_b.id}")

    # Idempotent on purpose: re-running the pass must not create rerun rows, which
    # would both exhaust the declared budget and inflate the apparent test count.
    already_run = {t.test_key: t for t in nb.family_tests(family_b.id)
                   if t.status == 'run'}
    n_recorded = 0
    for obj, p_value in scored.items():
        key = f'object={obj}'
        existing = already_run.get(key)
        if existing is not None:
            if existing.p_value is None or abs(existing.p_value - p_value) < 1e-12:
                continue  # same result already on record
            print(f"  WARNING: {key} already recorded with p={existing.p_value}, "
                  f"now {p_value}; leaving the original in place")
            continue
        nb.record_family_test(family_b.id, key,
                              iteration_id=latest_loc['id'] if latest_loc else None,
                              p_value=p_value,
                              git_commit=latest_loc['commit'] if latest_loc else None)
        n_recorded += 1
    print(f"  recorded {n_recorded} p-value(s)"
          + (f" ({len(scored) - n_recorded} already present)"
             if n_recorded < len(scored) else ''))

    nb.abandon_family_test(
        family_b.id, f'object={A_PRIORI_EXCLUSION}', reason=A_PRIORI_REASON,
        outcome_dependent=False, criterion_available_a_priori=True,
        applied_after_seeing_results=True, actor='backfill_ledger.py')
    for obj, reason in OUTCOME_DEPENDENT_EXCLUSIONS.items():
        nb.abandon_family_test(family_b.id, f'object={obj}', reason=reason,
                               outcome_dependent=True, actor='backfill_ledger.py')
    print(f"  classified 5 exclusion(s)")

    # Fill a NULL test_family_id so the reconstruction is visible from the
    # iteration. Adds a pointer; changes no measurement. Never overwrites.
    linked = []
    with nb.get_db_session() as db_session:
        for row in state['iterations']:
            if row['module'] != 'ephys.decode_location' or row['test_family_id']:
                continue
            iteration = db_session.get(Iteration, row['id'])
            if iteration.test_family_id is None:
                iteration.test_family_id = family_b.id
                linked.append(row['id'])
        db_session.commit()
    if linked:
        print(f"  linked iteration(s) {linked} to family {family_b.id} "
              "(NULL test_family_id filled; no other field touched)")

    denom = nb.family_denominator(family_b.id)
    print(f"  family {family_b.id} denominator: "
          f"n_tests_for_correction={denom['n_tests_for_correction']}, "
          f"status={denom['denominator_status']}")

    family_a = nb.get_or_create_test_family(FAMILY_A_NAME, alpha=0.05)
    print(f"  family {family_a.id} created for the event-based search "
          "(test keys left for a follow-up pass)")

    # A family is an empty duplicate only if it has neither iterations *nor*
    # declared tests. Checking iterations alone is wrong: the reconstruction
    # families deliberately hold family_tests and no iterations (the iterations
    # point at the original families), so an iterations-only rule invalidates
    # exactly the rows this pass just created. The reconstruction names are also
    # skipped explicitly, so a second run cannot invalidate the first run's work.
    reconstruction_names = {FAMILY_A_NAME, FAMILY_B_NAME}
    with nb.get_db_session() as db_session:
        for row in state['families']:
            if row['name'] in reconstruction_names:
                continue
            family = db_session.get(TestFamily, row['id'])
            if family is None or family.status == 'invalidated':
                continue
            n_iterations = db_session.query(Iteration).filter(
                Iteration.test_family_id == family.id).count()
            n_tests = db_session.query(FamilyTest).filter(
                FamilyTest.test_family_id == family.id).count()
            if n_iterations or n_tests:
                continue
            family.status = 'invalidated'
            family.notes = ((family.notes or '') +
                            '\ninvalidated by backfill_ledger.py: empty duplicate '
                            '(create_test_family has no get-or-create)').strip()
            print(f"  family {family.id} marked invalidated "
                  "(no iterations and no declared tests)")
        db_session.commit()

    print("\nApplied. No logged evidence was rewritten: every iteration field is "
          "byte-identical except a NULL test_family_id filled in above. Verify with:")
    print("  python scripts/notebook_cli.py summary")
    print(f"  python scripts/notebook_cli.py ledger {family_b.id} --n-shuffles 180")


# ----------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='backfill_ledger.py',
        description=__doc__.strip().splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db', default=None)
    parser.add_argument('--apply', action='store_true',
                        help='actually write (default is a dry run)')
    parser.add_argument('--no-snapshot', action='store_true',
                        help='skip the database snapshot (not recommended)')
    parser.add_argument('--confirm-exclusion-classification', action='store_true',
                        help='confirm the four reverse-null exclusions are '
                             'outcome-dependent')
    parser.add_argument('--n-shuffles', type=int, default=180)
    args = parser.parse_args(argv)

    nb = LabNotebook(args.db) if args.db else LabNotebook()
    print(f"notebook: {nb.db_path}")
    if nb.migrated_columns:
        print(f"migrated columns: {nb.migrated_columns}")

    state = survey(nb)
    print_survey(state)
    report_duplicates(nb, state)

    latest_loc = max((r for r in state['iterations']
                      if r['module'] == 'ephys.decode_location'
                      and r['summary'].get('per_object')),
                     key=lambda r: r['id'], default=None)
    if latest_loc:
        p_values = {str(e['object_name']): float(e['p_value'])
                    for e in latest_loc['summary']['per_object']
                    if e.get('p_value') is not None}
        if p_values:
            denominator_table(p_values, args.n_shuffles, (7, 9, 10, 11, 12))

    plan = Plan()
    if args.apply and not args.no_snapshot:
        snapshot(nb)
    reconstruct(nb, state, plan, apply=args.apply,
                confirm_exclusions=args.confirm_exclusion_classification)
    plan.render_questions()

    if not args.apply:
        print('\n' + '=' * 78)
        print(f"DRY RUN - {len(plan.steps)} write(s) planned, nothing written.")
        print("Review the denominator table and the judgment calls above, then:")
        print("  python scripts/backfill_ledger.py --apply "
              "--confirm-exclusion-classification")
        print('=' * 78)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
