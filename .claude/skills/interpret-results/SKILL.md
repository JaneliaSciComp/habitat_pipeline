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

2. **The headline number against the right baseline.** Compare
   `population_accuracy_mean` to **`baseline_accuracy`** (the majority-class rate),
   *not* to `1 / n_classes`. With imbalanced classes these differ a lot and only the
   former is honest: a real run reporting 60.6% accuracy on a 12/7 winner/loser split
   was actually *below* its 63.2% majority-class baseline, while `1/n_classes` would
   have called 50% "chance" and made it look like signal. If you want a metric whose
   chance level really is `1/n_classes`, use `balanced_accuracy`.

3. **The population-level p-value — usually the headline.**
   `significance_population['p_value']` tests whether the population's mean accuracy
   beats its label-permutation null. It is a *single* test, so it needs no FDR
   correction and is well resolved even at modest `n_shuffles`. When events are few
   (tens), this is the only decoding claim the data can actually support.

4. **Whether the per-cell screen could have detected anything.** Check
   `significance_resolution['resolvable']` **before** interpreting the per-cell count:
   - `resolvable == False` → a null per-cell result is **uninformative, not evidence
     of absence**. With `n_shuffles=200` across 149 cells the p-value floor is 1/201,
     so the best reachable q is 0.74 — no single cell could pass at any effect size.
     Say this explicitly and recommend a re-run at
     `significance_resolution['recommended_n_shuffles']` or with `null_mode='pooled'`.
     Do **not** report "0 significant cells" as a biological finding; that mistake was
     made once already on this project.
   - `resolvable == True` → report `n_significant / n_tested` surviving BH-FDR. That
     is the number that matters, not the raw "successful cells" count.
   - Rigor layer not run at all (`n_shuffles=0`, the default) → say so, and recommend
     re-running before treating "above chance" as a finding. Phase 0's synthetic demo
     showed a naive screen over-calls significance (9/24 flagged vs. 6 truly tuned).

5. **The denominator.** Read q from the ledger, not from `result_summary`:

   ```python
   fdr = nb.family_fdr(iteration.test_family_id)
   denom = fdr['denominator']
   ```

   Report `denom['n_tests_for_correction']` alongside every q-value, and compare
   `q_at_declared` against whatever the analysis logged. If `denom['denominator_status']`
   is not `'clean'`, say which it is and what it means:

   | status | what to say |
   |---|---|
   | `undeclared` | The family declares no tests, so the denominator is **unrecorded** — not zero. Any corrected statistic is uninterpretable. |
   | `reconstructed` | Recovered after the fact; the true number of tests is a **lower bound**. |
   | `outcome_dependent_exclusions` | Tests were dropped because of what results showed. That is selection on the outcome. |
   | `pipeline_changed` | The family mixes code versions. Those are not the same statistic. |

   This is the check that matters most. Iteration 12's `rat613` was logged at
   q=0.0387 and significant; at the declared denominator of 11 it is q=0.0608 and
   **not** significant, and the resolution guard flips from resolvable to not.

6. **Hazards, at the interpret stage.**

   ```python
   from discovery.hazards import hazards_for, render_digest, run_detectors_for
   ```

   Run the interpret-stage detectors against the result. A detector reporting
   `ran=False` is **could not check** — never report it as a pass. `HZ-INTERP-002`
   has no detector on purpose: re-derive the headline number from its inputs by hand
   and say what you recomputed versus what you took on trust. Every trap in this
   registry was found that way and by nothing else.

## 2. Only then, look at the figures

If PNGs exist (`figure_paths_list()`), read them for a **sanity check** against the
numbers you already stated — e.g. does the confusion matrix / accuracy-distribution
plot look consistent with the reported mean and best-cell accuracy — not as a source
of new claims.

⚠️ **The saved decoding figures draw their chance line at `1/n_classes`** and their
"Cells > Chance" counts against it (`ephys/decoding_plots.py`). On imbalanced classes
that line sits *below* the honest majority-class baseline, so a figure can show most
cells "above chance" for a result that doesn't beat naive guessing. Trust
`baseline_accuracy` from step 2 over the plotted line, and say so if they conflict.

If a figure needs real visual inspection (e.g. judging whether a
rate-map or raster looks structured), delegate to the `interpreter` subagent
(`.claude/agents/interpreter.md`) rather than reasoning over the image inline — it
keeps image tokens out of the main conversation and it's built to follow the same
numbers-first rule independently.

**If the numbers and the figure seem to disagree, flag the mismatch explicitly**
rather than quietly trusting whichever one looks more convincing — that disagreement
is itself worth reporting to the scientist.

## 3. Deliver a verdict, not a description

End with one of:
- **"Real, corrected"** — accuracy beats `baseline_accuracy`, the population-level
  p-value is significant, and/or cells survive FDR in a `resolvable` screen.
- **"Suggestive, not yet corrected"** — beats baseline but the rigor layer hasn't run.
  Recommend the re-run; don't call it a finding yet.
- **"Cannot tell — under-resolved"** — the per-cell screen wasn't `resolvable` and
  there's no population-level p-value to fall back on. This is *distinct from* "not
  supported": the analysis was incapable of answering, so the honest report is that
  nothing was learned. Recommend the specific re-run configuration.
- **"Not supported"** — accuracy at or below `baseline_accuracy`, and/or a
  well-resolved test found nothing.
- **"Refuted"** — a well-resolved test contradicts the hypothesis's own prediction.
  Distinct from "not supported", which is an absence of evidence. Record it as such:
  `notebook_cli.py verdict <id> refuted --rationale "..."`. A refutation is a result
  and must be reported as prominently as a confirmation, not quietly dropped.

### Always state the tier

```python
print(nb.evidence_tier(hypothesis_id).summary())
```

Unless the tier is `confirmatory`, say in these words: **"Without the holdout, the
loop's output is hypothesis-generating only."** Then list the unmet conditions the
assessment returns. This is not boilerplate — a result on the same data the
hypothesis was generated from is a different kind of claim from one that survived
held-out confirmation, and the wording is what keeps the two from being read alike.

Note the current constraint: only one cohort-7 session has scored behavioural events,
so **no event-based hypothesis has a path to holdout confirmation right now**. Say
that rather than implying confirmation is merely pending.

If a `Hypothesis` is attached to this iteration, this verdict is what the scientist
uses at the approval gate — say plainly whether you'd advance it, and why, in terms
of the numbers above. Then point at:

```
python scripts/notebook_cli.py report --hypothesis <id>
```
