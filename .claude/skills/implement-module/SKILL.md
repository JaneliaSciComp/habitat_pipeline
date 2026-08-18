---
name: implement-module
description: Turn an approved hypothesis (or a direct request for a new analysis) into a concrete build plan, get explicit scientist approval, then dispatch the coder subagent to write the new ephys/ module and its mock test. Use when a hypothesis's chosen_test needs code the repo doesn't have yet, or when asked to implement/build/add a new analysis module.
user-invocable: true
allowed-tools:
  - Read
  - Bash
  - Glob
  - Agent
---

# implement-module — Coder-orchestration role for the discovery loop

Implements the code-writing side of the **Coder** role from
`docs/AI_DISCOVERY_LOOP_DESIGN.md` §4⑤, enforcing its hardest requirement (§6):
**explicit scientist approval before writing or modifying repo code.** This skill
deliberately has no `Write`/`Edit` in its own `allowed-tools` — it can only plan and
talk. The only thing that can actually write a file is the `coder` subagent
(`.claude/agents/coder.md`), and this skill only dispatches it after approval. That
makes the gate structural, not just a promise in prose.

## 1. Take the input

Either:
- A `Hypothesis` id from `propose-hypotheses` whose `chosen_test` names an analysis
  the repo doesn't have yet, or
- A direct ad hoc request ("build a decoder for X").

```python
from database.lab_notebook import LabNotebook
nb = LabNotebook()
hyp = next(h for h in nb.list_hypotheses() if h.id == hypothesis_id)
```
Read `hyp.statement` / `hyp.predicted_effect` / `hyp.chosen_test` before doing
anything else.

## 2. Restate the concrete build plan — do this before asking for approval

Work out everything the `coder` subagent (see its file for the full contract) will
need, so the scientist is approving a specific, inspectable plan rather than a blank
check:

- **New module path** (`ephys/<name>.py`) and the wrapper function signature.
- **Which existing core it reuses**: classification-shaped → `ephys/_lda_decoding.py`'s
  `run_population_per_cell_decode`/`single_cell_lda_decode`; regression/custom-shaped →
  `ephys/decode_partner_distance.py`'s I/O-free-core + thin-wrapper split. Say which,
  and why.
- **Result-dict fields** it must carry (`status`, `parameters` with
  `class_label`/`analysis_title`, a `behavioral_summary`-equivalent block, and
  `significance*` keys if the rigor layer applies).
- **The statistical assumptions** it will need to document (module "Statistical
  notes" + function "Assumptions:" — see `coder.md` for the exact convention/examples).
- **The new test file** and which existing fixtures it reuses
  (`tests/test_social_spatial_fields.py`'s session-shaped mocks, or
  `tests/test_decode_partner_distance.py`'s pure-array generators).
- **Whether a domain-guardrail test applies** (an identity/axis/self-target swap this
  analysis could get backwards) — name the specific swap risk, or state there isn't one.

## 3. Hard stop — explicit approval required

Present the plan from step 2 to the scientist and **wait for an affirmative
go-ahead in this same turn** before doing anything else. This is the design doc's
§6 gate (a) — "explicit scientist approval before writing/modifying repo code" —
applied the same way `run-analysis`'s "Approval gates" section applies to held-out
sessions and `config/`/`.gui_cache/` changes. Do not treat silence, a vague
acknowledgment, or "sounds good in general" as approval of the specific plan — get a
clear yes to *this* plan.

If the scientist wants changes to the plan, revise and re-present; don't dispatch on
a plan that changed after the last explicit approval.

## 4. On approval: register and dispatch

```python
if hypothesis_id is not None:
    nb.set_hypothesis_status(hypothesis_id, 'approved')
```

Then dispatch the `coder` subagent (`Agent` tool, `subagent_type='coder'`) with the
**exact plan just approved** as its prompt — file paths, signature, result-dict
contract, assumptions to document, test fixtures to reuse, and the guardrail test
call, verbatim. Don't hand it a vaguer version of the plan than what was approved.

## 5. Verify independently — don't just relay the subagent's self-report

```bash
python -m pytest tests/test_<new>.py -q
python -m pytest tests/ -q -m "not slow"
git status
git diff --stat
```

`coder` runs these itself too, but re-run them yourself before telling the scientist
anything passed. If either suite fails, say so plainly — don't round a partial pass
up to success.

## 6. Report and hand off

Tell the scientist: files created/changed, the real test outcome, the statistical
assumptions `coder` documented (in plain language), whether a guardrail test was
added and what it guards, and an explicit reminder that **nothing is committed** —
review the diff (`git diff`) and commit when ready, per this repo's convention of
never auto-committing. Point them at `run-analysis` as the natural next step once
they're satisfied with the code.

## 7. On rejection

If the scientist declines at step 3:

```python
if hypothesis_id is not None:
    nb.set_hypothesis_status(hypothesis_id, 'rejected', notes=...)
```

Stop there — do not dispatch `coder`.
