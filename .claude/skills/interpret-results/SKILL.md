---
name: interpret-results
description: Read a logged analysis result (numbers first, figures second) and produce a structured interpretation with an explicit call on whether the finding is statistically real. Use after run-analysis completes, or when asked to interpret/explain/assess a decoding or analysis result.
user-invocable: true
allowed-tools:
  - Read
  - Bash
  - Glob
  - Agent
---

# interpret-results — Interpreter role for the discovery loop

Implements the **Interpreter** role from `docs/AI_DISCOVERY_LOOP_DESIGN.md` §4,
including its explicit caveat: the neuroscience case study behind this design flags
**weak visual self-evaluation** as a top failure mode for agents in a
data-to-discovery loop. The rule this skill enforces: reason over the logged numbers
*first*; look at figures only *after* you've stated the numbers-based conclusion, and
only to sanity-check it — never let a figure override what the metrics say.

## 1. Pull the numbers

If you have an `Iteration` id (from `run-analysis`) or a session id, load it:

```python
from database.lab_notebook import LabNotebook
nb = LabNotebook()
rows = nb.iterations_for_session(session_id)   # or look up by id directly
summary = rows[-1].result_summary_dict()
```

State, in this order:
1. **What was tested** (`analysis_module`, `params_dict()['behavior_type']` /
   `label_mode`, event counts from `unique_classes`/`n_events`).
2. **The headline number** (`population_accuracy_mean` ± std, or `best_cell_accuracy`)
   against chance (`1 / n_classes`).
3. **Whether it's corrected for multiple comparisons.** Check
   `summary.get('n_significant_cells')` / whether the run used `n_shuffles > 0`:
   - If the rigor layer ran (`significance` was populated): report
     `n_significant / n_tested` cells surviving Benjamini-Hochberg FDR — that's the
     number that matters, not the raw "successful cells" count.
   - If it did **not** run (`n_shuffles=0`, the default): say so explicitly and
     recommend re-running via `run-analysis` with `n_shuffles>=200` before treating
     "above chance" as a real finding. Do not report an uncorrected accuracy as if it
     were a discovery — Phase 0's own synthetic demo showed exactly this over-calls
     significance (9/24 flagged vs. 6 truly tuned cells).

## 2. Only then, look at the figures

If PNGs exist (`figure_paths_list()`), read them for a **sanity check** against the
numbers you already stated — e.g. does the confusion matrix / accuracy-distribution
plot look consistent with the reported mean and best-cell accuracy — not as a source
of new claims. If a figure needs real visual inspection (e.g. judging whether a
rate-map or raster looks structured), delegate to the `interpreter` subagent
(`.claude/agents/interpreter.md`) rather than reasoning over the image inline — it
keeps image tokens out of the main conversation and it's built to follow the same
numbers-first rule independently.

**If the numbers and the figure seem to disagree, flag the mismatch explicitly**
rather than quietly trusting whichever one looks more convincing — that disagreement
is itself worth reporting to the scientist.

## 3. Deliver a verdict, not a description

End with one of:
- "Real, corrected": significant cells survive FDR at a real `n_shuffles`.
- "Suggestive, not yet corrected": accuracy is above chance but the rigor layer
  hasn't run — recommend the re-run, don't call it a finding yet.
- "Not supported": accuracy near chance / no cells survive correction.

If a `Hypothesis` is attached to this iteration, this verdict is what the scientist
uses at the approval gate — say plainly whether you'd advance it, and why, in terms
of the numbers above.
