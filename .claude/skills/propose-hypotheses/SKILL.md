---
name: propose-hypotheses
description: Generate, critique, and rank candidate hypotheses from a logged analysis result, ground them in literature when available, and pre-register the scientist's pick in the lab notebook. Use after interpret-results delivers a verdict, or when asked to propose/suggest a hypothesis, next experiment, or follow-up analysis for a session.
user-invocable: true
allowed-tools:
  - Read
  - Bash
  - Glob
  - Agent
---

# propose-hypotheses — Hypothesizer role for the discovery loop

Implements the **Hypothesizer** role from `docs/AI_DISCOVERY_LOOP_DESIGN.md` §4④,
including its explicit instruction: "Do not accept an agent's first idea." Run the
bounded generate → critique → rank cycle below and hand the scientist a ranked
shortlist with rationale — not one idea dressed up as a conclusion.

No approval gate applies to *this* skill — generating and pre-registering hypotheses
is free. The gate is downstream: testing against a held-out session, or writing new
code (`implement-module`), each need their own explicit approval per `CLAUDE.md` /
`docs/AI_DISCOVERY_LOOP_DESIGN.md` §6.

## 0. Consult Layer 0 first — before generating anything

Two machine-readable artifacts exist so that an untestable hypothesis fails here
rather than forty minutes into an extraction.

### 0a. The capability manifest

```python
from discovery.capability_manifest import (manifest_status, session_capabilities,
                                            suggest_sessions)
print(manifest_status().summary())
print(session_capabilities(session_id))
```

`manifest_status()` **raises** on a missing file, a schema mismatch, or a changed
cohort config. If it raises, stop and tell the scientist to run
`python scripts/build_capability_manifest.py`. Do not proceed from a stale manifest —
an agent that trusts one will propose confidently against data that no longer exists,
which is a *new* failure mode created by the fix.

State the session's actual inventory in one line before generating: animals with
ephys, quality cells, identity-resolved tracked animals, and which behaviour types
have usable label sets.

**Availability as of the 2026-08-20 build** — verify against the manifest rather than
trusting this paragraph, but the shape matters for what is worth proposing at all:

| | cohort 7 | cohort 5 |
|---|---|---|
| sessions | 14 | 20 |
| with tracking | 2 (`20251210`, `20251216`) | 0 |
| with scored events | 1 (`20251216`) | 2 |
| 4-animal ephys | 7 | 0 |

So every event-based hypothesis — anything using `decode_opponent_identity` or
`decode_event_outcome` — is testable on **one** session, and cross-day or held-out
confirmation of an event-based claim is not currently possible. Say so rather than
proposing a multi-session design that cannot run.

### 0b. The hazard registry

```python
from discovery.hazards import hazards_for, render_digest
print(render_digest(hazards_for(stage='propose', min_severity='high'), 'line'))
```

Read this *before* generating, so the traps are in context rather than discovered
afterwards.

## 1. Ground in the actual finding

Don't propose in a vacuum. Pull the triggering result and the session's history:

```python
from database.lab_notebook import LabNotebook
nb = LabNotebook()
rows = nb.iterations_for_session(session_id)
summary = rows[-1].result_summary_dict()          # the finding you're building on
prior_hypotheses = nb.list_hypotheses()            # everything already proposed
```

State what you're building on in one line (module, headline number vs. baseline,
population p-value — same numbers `interpret-results` already surfaced) before
generating anything. Skim `prior_hypotheses` so you don't re-propose something
already on record; if a candidate duplicates one, say so instead of hiding it.

## 2. Literature pass — degrade gracefully, never block

This environment has literature MCP servers installed (PubMed, bioRxiv,
Scholar_Gateway, Consensus, Elicit, Scite, Open_Targets, ChEMBL). Try them. As of
this writing **none are authenticated** — only their `authenticate`/
`complete_authentication` tools are exposed, not real search.

- **Do not silently trigger an OAuth flow.** If a server's real search tools aren't
  available, tell the scientist plainly which MCP needs `/mcp` auth and continue
  without it — this is a one-time setup step for them, not something to shortcut.
- Every hypothesis produced without a working literature search must say so
  explicitly: `citations: []`, flagged **"not literature-grounded"**. Never present
  an unsupported hypothesis as if it were backed by a citation you don't actually have.
- If a server *is* authenticated, use it and attach real citations
  (`{source, id, title, url}` — the same shape `Hypothesis.citations` already stores,
  see `database/lab_notebook.py::add_hypothesis`).

## 3. Generate 3-5 candidates

Each candidate is a `{statement, predicted_effect, chosen_test, citations}` tuple,
tied to the concrete result from step 1 — not a generic "cells might encode X"
idea. For `chosen_test`, be explicit about which of two buckets it falls in:

- **Existing-module** — names a wrapper that already exists (`decode_opponent_identity`,
  `decode_event_outcome`, `run_inter_brain`, `compute_social_place_fields`,
  `decode_partner_distance`) with specific params (behavior_type, label_mode, etc.).
  This bucket only ever needs `run-analysis` next — no new code.
- **New-module** — needs an analysis the repo doesn't have yet. This bucket is
  `implement-module` material; say plainly what the new module would compute and
  why the existing ones can't answer it.

## 4. Critique each candidate

Before ranking, stress-test every candidate:

- **Testability — mechanically, not by judgement.** Every candidate's `chosen_test`
  must name an analysis, a session, an animal, and concrete params, because that is
  `check_testable`'s signature:
  ```python
  from discovery.capability_manifest import check_testable, suggest_sessions
  report = check_testable(analysis, session_id, animal_id=animal_id, **params)
  ```
  A candidate that comes back `testable=False` is **either dropped with its stated
  reason, or re-pointed at a session from `suggest_sessions(...)`** — never silently
  ranked as if it were runnable. Carry `report.warnings` through to the presentation
  even for candidates that pass.

  This is a genuine tightening: `chosen_test` used to be free prose, and Hypothesis 3
  was pre-registered and only found to be impossible after the analysis was attempted.

- **Power.** Same math `run-analysis` applies:
  ```python
  from ephys._stats_utils import fdr_resolution, majority_class_baseline
  ```
  Flag a candidate as under-powered if the likely event count per class is too small
  for a meaningful CV split (`min_events_per_class=5` is the decoders' own floor) or
  if `fdr_resolution` at a realistic `n_shuffles` budget couldn't detect the claimed
  effect size. Use the manifest's `majority_baseline` as the bar a predicted accuracy
  must clear — a candidate predicting "60%" on the 12/7 outcome split is predicting a
  result *below* naive guessing, and that can be caught here before anything runs.

- **Propose-stage hazard detectors.** Run them against the candidate's params. In
  particular a candidate that hand-picks a class subset or lowers
  `min_events_per_class` trips `HZ-STAT-013`; that is allowed but must be declared as
  part of a test family up front, not discovered later from a params dict.
- **Redundancy.** Cross-check against `prior_hypotheses` and `iterations_for_session`
  — has this exact test already run? Say so if it has.
- **Existing vs. new code**, restated plainly (from step 3) — this is what determines
  whether the scientist's next step is `run-analysis` or `implement-module`.

## 5. Rank and present

Present a numbered shortlist: statement, predicted effect, chosen test, existing/new
call, citations (or the explicit "not literature-grounded" flag), and the critique
findings. Wait for the scientist to pick one, several, or none — do not proceed
automatically.

## 6. Pre-register the pick

On a pick, register it — this does **not** need approval, it's pre-registration, not
execution (design doc §5: "reserve at least one recording session... until a
hypothesis is *pre-registered*... in the lab notebook"):

```python
hyp = nb.add_hypothesis(
    statement=...,
    predicted_effect=...,
    chosen_test=...,          # analysis + session + animal + concrete params
    citations=citations_or_none,
)
```

### Then freeze the prediction — this is the moment for it

Pre-registration is only pre-registration if it happens before the run. This skill is
the last point at which that is still true, so do it here:

```python
prediction = nb.freeze_prediction(
    hyp.id,
    statistic='min_q_value',        # or a dotted path into the result summary
    direction='lt', threshold=0.05,
    alpha=0.05, n_shuffles_planned=n_shuffles,
    declared_test_keys=test_keys,   # the family, decided now
    holdout_kind='replication',     # or 'generalization' - they are different claims
    falsifier='<what result would count against this>',
)
```

**`falsifier` is required and must come from the scientist.** An agent-authored
falsifier for someone else's hypothesis is not a falsifier — it is a guess about what
they would find persuasive. Ask for it; refuse to freeze without one.

Get `holdout_kind` right, because it decides how a failure is read. A held-out session
with *different animals* cannot replicate a finding about animal 631 — it can only
test whether the effect generalizes, and a failed generalization is **not** a
refutation of the original claim. Conflating the two misreports both.

There is deliberately **no update method** for a frozen prediction. Changing a
statistic or a threshold means `nb.supersede_prediction(...)`, which inserts a new row
and leaves the old one visible with its reason — so the report can show that the
target moved. That is the whole point of freezing it.

If the analysis has already run and you are recording what was claimed rather than
predicting it, pass `registered_post_hoc=True`. That is honest and it is permanent:
such a record can never satisfy the confirmatory tier.

Status stays at the DB default (`'proposed'`) — don't flip it here. It becomes
`'approved'`/`'rejected'` at whichever gate comes next: `run-analysis`'s existing
approval-gate check if `chosen_test` names an existing module, or
`implement-module`'s code-writing gate if it names a new one. Tell the scientist the
new `Hypothesis.id` and which of those two skills to run next.
