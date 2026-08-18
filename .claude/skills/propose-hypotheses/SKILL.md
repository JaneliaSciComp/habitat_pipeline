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

- **Power/feasibility.** Use the event/class counts already on record (from step 1,
  or a quick check against `BehavioralEventsData`) with the same math `run-analysis`
  already applies:
  ```python
  from ephys._stats_utils import fdr_resolution
  from ephys._stats_utils import majority_class_baseline
  ```
  Flag a candidate as under-powered if the likely event count per class is too small
  for a meaningful CV split (rule of thumb already in this repo: `min_events_per_class=5`
  is the decoders' own floor) or if `fdr_resolution` at a realistic `n_shuffles` budget
  couldn't detect the claimed effect size.
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
    chosen_test=...,
    citations=citations_or_none,
)
```

Status stays at the DB default (`'proposed'`) — don't flip it here. It becomes
`'approved'`/`'rejected'` at whichever gate comes next: `run-analysis`'s existing
approval-gate check if `chosen_test` names an existing module, or
`implement-module`'s code-writing gate if it names a new one. Tell the scientist the
new `Hypothesis.id` and which of those two skills to run next.
