# HANDOFF — AI-in-the-loop discovery platform work

*Context for continuing this thread in a fresh Claude Code session. Read this + `CLAUDE.md`, then see the docs referenced below.*

## What this thread is about

Designing an **AI-in-the-loop discovery platform** on top of `habitat_pipeline`: an agent that runs analyses using the existing code, interprets results (numbers + figures), searches the literature, generates/ranks hypotheses, implements them as new `ephys/` modules following repo conventions, re-runs on data, and iterates — with a scientist gating the decisions.

## Artifacts already created (in this repo)

- `docs/AI_DISCOVERY_LOOP_DESIGN.md` — the merged design doc. Core = repo-native discovery loop + a scientific-rigor layer; hardening layer = right-sized MLOps (containers, CI gating agent code, experiment tracking, sandboxing). Explicitly excludes enterprise overkill (K8s/Terraform/Domino/RLHF). Corrected for the real modality (electrophysiology / Kilosort, **not** calcium imaging).
- `docs/PHASE0_FINDINGS.md` — results of the first loop attempt.
- `scripts/phase0_probe.py` — reproducible probe: env check, core mock tests, real-session decode CLI, synthetic Runner→Interpreter loop with a multiple-comparison guardrail. Runs on data-less machines (REAL leg reports gracefully) and on the workstation (REAL leg should pass).

## Phase 0 findings (key facts)

- Code is healthy and agent-drivable: **104 core mock tests pass** in ~13s; the LDA core returns clean result-dicts; CLIs fail *gracefully* with parseable errors.
- **One hard blocker: data locality.** The configs point at `//nearline/karpova/...` SMB shares. The Runner must execute where that share is mounted (this Janelia Windows workstation or an HPC node), not in a remote Linux sandbox.
- The rigor guardrail justified itself on run #1: a naive "above chance" screen over-called cells vs known ground truth — multiple-comparison correction belongs in the core, pulled forward to Phase 1.

## Immediate next step

Run the probe on THIS workstation, where `//nearline` is mounted:

```
python scripts/phase0_probe.py --animal-id 631 --session-id 20251216
```

If the REAL leg turns green (produces a real result-dict + figures), that's the baseline for **Phase 1**: codify the loop as skills/subagents + a **lab notebook** (build it on the existing `database/` + `habitat_pipeline.db`), and wire in the rigor layer — replacing the probe's crude binomial FDR illustration with proper **label-permutation nulls**.

## Open decisions / TODO

- [x] Confirm the real-session decode runs end-to-end on the workstation. **Confirmed 2026-08-05**: data locality is resolved (real load succeeds: 263 clusters, 641 events); the CLI sync gap is fixed (see below). Both decoders now produce real validated results for animal 631/session 20251216: `decode_event_outcome` (behavior_type default → 19 events, 146/149 cells decoded, 55.9%±11.1% pop. accuracy, best cell 80.0%) and `decode_opponent_identity` (`--behavior_type EC` → 8 opponents, 149/149 cells decoded, 25.2%±3.6% pop. accuracy vs. 12.5% chance, best cell 36.4%). See "Update 2026-08-05" in `docs/PHASE0_FINDINGS.md`.
- [x] Fixed the CLI sync gap: both `main()` functions now build a `DataSyncManager(dio_channel=1)` and call `synchronize_with_ephys` before decoding, mirroring `gui/loaders.py`. Also fixed two bugs found while verifying: a `✓` print in `video/behavioral_events.py` that crashes on Windows `cp1252` consoles, and `decode_event_outcome.py` passing `DataStorageManager` directly to `load_kilosort_data` (the stale-API gotcha `CLAUDE.md` already warns about). 104 mock tests still pass.
- [x] Re-ran `decode_opponent_identity` against `--behavior_type EC` (8 opponents with ≥5 events each, vs. only 1 for the default `'F'`) — real validated result, above chance. Phase 0 is now fully closed out; both decoder REAL legs are green.
- [x] Design the lab-notebook schema on the `database/` layer. **Done 2026-08-05** — see "Phase 1 backbone" below.
- [ ] Decide experiment-tracking tool (MLflow self-hosted vs W&B). Still deferred — that's Phase 2 (§7 hardening), not Phase 1.
- [ ] Do NOT touch `config/` or delete `.gui_cache/` without asking (per CLAUDE.md).

## Phase 1 backbone — done 2026-08-05

Built per the approved plan (`~/.claude/plans/vivid-wobbling-pillow.md`): the lab notebook, the rigor layer, and the two lowest-risk skills/subagent. Deferred (per explicit scope decision): `propose-hypotheses`, `implement-module`, and the Coder subagent — they need literature-MCP auth and code-writing approval gates that deserve their own review.

**Rigor layer** (`ephys/_stats_utils.py`, `ephys/_lda_decoding.py::compute_population_significance`): opt-in (`n_shuffles=0` by default) label-permutation significance + Benjamini-Hochberg FDR across cells, wired additively into `decode_opponent_identity_population`/`decode_event_outcome_population` (new `results['significance']` key, new `--n_shuffles`/`--alpha` CLI flags) without touching the pinned result-dict schema or `run_population_per_cell_decode`'s signature. New guardrail test (`tests/test_lda_decoding_significance.py`) recreates Phase 0's synthetic 24-cell/6-tuned scenario and asserts the FDR correction doesn't manufacture more false positives than the naive screen did.

**Real-data verification** (animal 631/session 20251216, `decode_event_outcome`, default params, 200 shuffles): 148/149 cells decoded (population accuracy 60.6%±11.4%, best cell 95.0%), but **0/148 cells survive FDR correction** — a materially different, more honest conclusion than "146 successful cells at 55-60% accuracy" reads as on its own. Worth noting: the 12-winner/7-loser class split means naive majority-class guessing already scores 63.2%, i.e. the per-cell CV accuracy here doesn't even clear that naive baseline — with only 19 events split across 5 CV folds, this is a plausible, real result (underpowered, not a bug), and exactly the kind of over-claiming the rigor layer exists to catch before it reaches a hypothesis. Runtime: ~370s for 149 cells × 200 shuffles on this workstation — opt-in, not default, for a reason.

**Lab notebook** (`database/lab_notebook.py`, new file, `database_core.py` untouched): `Hypothesis`/`TestFamily`/`Iteration` tables on the same `Base`/`habitat_pipeline.db` as the existing Animal/Session/DataFile models. `LabNotebook.log_iteration(...)` logged the real run above as iteration id 1 (git commit `bd61df5`), round-tripped correctly via `iterations_for_session('20251216')`. `recompute_family_significance` does the *campaign*-level BH-FDR (one p-value per iteration) — a distinct correction from the rigor layer's *within-run* per-cell correction above. `sqlalchemy` (already declared under `pyproject.toml`'s `[all]` extra, just not installed) is now installed.

**Skills + subagent**: `.claude/skills/run-analysis/SKILL.md`, `.claude/skills/interpret-results/SKILL.md`, `.claude/agents/interpreter.md`. Not yet exercised live — **newly added project skills aren't hot-loaded mid-session**; they'll register on the next fresh session. The underlying recipe both skills describe was verified directly (the real-data run above follows it exactly: load → sync → decode with `n_shuffles` → curate a summary → `log_iteration`).

**Test suite**: 233 passed, 1 skipped (was 217 before this pass), full run ~2 min (was ~14-20s for the original 104-test core suite; the 200-shuffle guardrail test is the main new cost, and it's a one-time regression check, not something re-run on every commit necessarily).

**Next**: on a fresh session, actually invoke `/run-analysis` then `/interpret-results` conversationally to confirm the skills read as expected once loaded. After that, `propose-hypotheses`/`implement-module`/Coder is the natural next Phase 1 increment.

> ⚠️ **Superseded in part by Phase 1.5 below.** The "0/148 cells survive FDR" figure above was an *artifact of an under-resolved permutation budget*, not a biological null. See the correction.

## Phase 1.5 backbone — statistics correctness — done 2026-08-06

A review of the Phase 1 work found three verified miscalibrations. All are now fixed. Sequencing rationale: an agent generating and screening hypotheses on top of a miscalibrated significance test is exactly the false-positive factory the design doc §5 exists to prevent, so this came before the remaining agentic pieces.

**1. The FDR had no resolution to detect anything.** A permutation test floors p at `1/(n_shuffles+1)`; BH multiplies the smallest p by the number of tests. At `n_shuffles=200` across 149 cells the best achievable q was **0.74** — no cell could pass at any effect size, and ≥15 cells would have had to hit the p-floor simultaneously. The "0/148 significant" result was predetermined by the budget. New [`fdr_resolution()`](ephys/_stats_utils.py) computes this up front; `compute_population_significance` calls it, raises a `RuntimeWarning`, and returns the verdict as `results['significance_resolution']` so an under-resolved run is self-labelling. Two escape hatches added: `null_mode='pooled'` (ranks each cell against all cells' nulls pooled → ~150× finer p-floor at *identical* compute) and a properly-resolved single population-level test (below).

**2. Accuracy was compared against the wrong baseline.** `chance_level = 1/n_classes` is prevalence-blind; the per-cell path returned no baseline at all. For the real 12-winner/7-loser split the majority-class baseline is **63.2%**, so the previously-reported 55.9–60.6% accuracies were at or *below* naive guessing. Added (purely additive) `baseline_accuracy` + `balanced_accuracy` per cell and `population_baseline_accuracy` + `population_balanced_accuracy_mean` per run, plus shared `print_baseline_block`/`print_significance_block` CLI reporters that flag "BELOW majority-class baseline" explicitly. Left alone by decision: `scoring='accuracy'` and `decoding_plots.py`'s `1/n_classes` chance lines (documented as a gotcha in CLAUDE.md instead of changing existing figures).

**3. Campaign-level FDR was dead code.** `recompute_family_significance` reads a `p_value` from `result_summary` but nothing produced one. Now `results['significance_population']` gives one analysis-level permutation p-value (mean accuracy vs. its null) — **free**, since it reuses the retained null matrix. Being a single test it needs no correction and is well resolved at modest `n_shuffles`, which makes it the right headline when events are few. Subtlety handled: all cells now share one permutation per shuffle index, because independent per-cell permutations would shrink the population-mean null's variance by ~1/n_cells and make that test anti-conservative (there's a regression test for this).

**Also fixed (pre-existing).** `ephys/social_spatial_fields.py`'s `_p_geq`/`_p_leq` used `k/n`, which returns exactly `p=0.0` when no shuffle beats the observed value — invalid from a finite permutation test, and it survives BH as `q=0` reading as infinite confidence. Now the add-one `(1+k)/(n+1)` form, consistent with the decoding path. **This turned out to be masking the same resolution bug**: four social-place-field tests only passed because `p=0.0` short-circuited a configuration (3 targets, 100 shuffles, `sig_alpha=0.01` → best q 0.030) that could never have classified a cell as tuned. Their shuffle budget is now 500. Note `compute_social_place_fields` does **not** yet self-check resolution — flagged in CLAUDE.md as a known gap.

**Also fixed (found during verification).** Non-finite null draws — degenerate CV folds, common with few events — were being silently counted as "did not exceed", because `nan >= x` is False. That biases every p-value *downward*. Now `ephys._stats_utils.empirical_p_value` single-sources the add-one form and drops non-finite draws from both numerator and denominator; runs report `n_valid_shuffles`/`n_nan_draws`.

**Test suite**: `slow` marker added to `pyproject.toml` (none existed). Fast pass `pytest -q -m "not slow"` → 253 passed, 1 skipped, 5 deselected in ~90 s (was ~170 s with everything inline). Heavy permutation tests run in the full pass.

### First real finding — opponent identity is decodable from `EC` events

With the corrected stack, animal 631 / session 20251216, `n_shuffles=200`:

| run | accuracy | majority baseline | balanced acc (chance) | population p | per-cell FDR |
|---|---|---|---|---|---|
| `decode_event_outcome` (19 events, 12/7) | 60.6% ± 11.4% | **63.2%** — below | 54.3% (50%) | **0.114** | 0/148 |
| `decode_opponent_identity` `EC` (173 events, 8 opponents), per-cell null | 28.8% ± 4.2% | 27.7% | 14.5% (12.5%) | **0.005** | **17/149** |
| same, pooled null | 28.8% ± 4.2% | 27.7% | 14.5% (12.5%) | **0.005** | **16/149** |

- **The outcome analysis is a clean, honest null**: accuracy below its own majority-class baseline and a non-significant population p-value. Consistent across every statistic, and now reported as such rather than as "55.9–60.6% accuracy, best cell 80–95%".
- **The `EC` opponent analysis is a real effect**: observed mean 28.8% against a null of 25.5% ± 0.3% — roughly 11σ — with the population p-value pinned at its floor (no permutation beat the observed value), and 16–17 of 149 cells surviving per-cell FDR under *both* null modes. This is the project's first genuine positive finding, and it is the better-powered analysis (173 events, ~21/class) as predicted.
- **The resolution guard behaved exactly as its analytics predicted.** It warned that 149 cells × 200 shuffles cannot resolve a *lone* significant cell — and indeed the 17 detections only got through because ≥15 cells hit the p-floor simultaneously, which is precisely the `min_tests_at_floor = 15` condition. Per-cell (17) and pooled (16) agreeing is a good cross-check. For a single-cell claim here, `n_shuffles≥2980` or the pooled null is still required.
- **Campaign-level FDR now has inputs** (test family 3): q = 0.0075 for both `EC` runs, q = 0.114 for the outcome run.

**Deferred, unchanged**: the multi-session sweep (feasible — `get_animals_and_sessions(config_path)` yields 47 pairs across 2 cohorts; wants resumability since a rigor-on sweep is multi-hour), notebook read surface, and held-out-session enforcement (`Iteration.held_out` is still a column nobody checks).

## `run-analysis` → `interpret-results` live-tested — done 2026-08-17

Exercised the two Phase 1 skills conversationally, end-to-end, for the first time
(they'd only been unit-verified as recipes before). Ran `decode_opponent_identity`
(`behavior_type='EC'`, `null_mode='pooled'`, `n_shuffles=200`) in-process against
animal 631/session 20251216, logged it as iteration 7, then ran `interpret-results`
against that iteration: population accuracy 28.8% vs. 27.7% baseline, population
p = 0.005, 16/149 cells survive FDR — **verdict: Real, corrected**, reproducing
iteration 5's numbers exactly (same seed, same inputs — a clean reproducibility check).

One real friction point surfaced: `decode_opponent_identity_population`'s
`animal_of_interest` must be a `str` — `extract_opponent_labels` calls
`.startswith('rat')` on it directly — but nothing enforces this for an in-process
caller (only the CLI's `argparse(type=str)` does). Passing an `int` fails deep inside
label extraction with a confusing `'int' object has no attribute 'startswith'`. Not
yet fixed in the library code; logged as iteration 6 (a failed run) and flagged here
for whoever touches `extract_opponent_labels` next.

## Phase 1 second increment — Hypothesizer + Coder — done 2026-08-17

Built the remaining two discovery-loop roles deferred at the end of Phase 1:
`propose-hypotheses` (Hypothesizer, design doc §4④/§5) and `implement-module` +
a `coder` subagent (Coder, §4⑤). Per the approved plan
(`~/.claude/plans/curried-giggling-sutherland.md`).

**`database/lab_notebook.py`**: added `LabNotebook.set_hypothesis_status(id, status,
notes=None)` and `list_hypotheses(status=None)` — small, additive, same file,
`database_core.py` untouched. Needed so `propose-hypotheses` can check for prior
proposals before registering a duplicate, and `implement-module` can flip a
hypothesis to `approved`/`rejected` at its code-writing gate. New tests in
`tests/test_lab_notebook.py::TestHypothesisAndTestFamily` (20 → covers round-trip,
bad-status, unknown-id, and status-filtered listing).

**`.claude/skills/propose-hypotheses/SKILL.md`**: generate → critique → rank cycle
per the Co-Scientist pattern the design doc calls for. Grounds every candidate in
the triggering iteration's actual numbers and checks `list_hypotheses()` for
duplicates before proposing. **Literature MCP servers in this environment
(PubMed, bioRxiv, Scholar_Gateway, Consensus, Elicit, Scite, Open_Targets, ChEMBL)
are installed but not authenticated** — confirmed via `ToolSearch`, only their
`authenticate`/`complete_authentication` tools are exposed. The skill is written to
degrade gracefully: tell the scientist which MCP needs `/mcp` auth, continue without
it, and flag every hypothesis produced this way as "not literature-grounded" rather
than silently proceeding as if it were cited. On a scientist pick, pre-registers via
`nb.add_hypothesis(...)` (status stays `'proposed'` — no gate at this stage).

**`.claude/agents/coder.md`**: writes a new `ephys/` module + matching mock test.
Picks between the two conventions already in the repo — reuse the classification
core (`ephys/_lda_decoding.py`) or `decode_partner_distance.py`'s I/O-free-core +
thin-wrapper split — and is required to document statistical assumptions the way
`decode_partner_distance.py` (module "Statistical notes") and
`_lda_decoding.py::compute_population_significance` (function "Assumptions:") already
do, plus add a domain-guardrail test modeled on
`tests/test_social_spatial_fields.py`'s self/target-swap pair (or explain why none
applies). Tools: `Read, Write, Edit, Bash, Glob, Grep` — no `Agent`, no MCP/web
access (citations arrive pre-resolved in its prompt).

**`.claude/skills/implement-module/SKILL.md`**: the orchestration layer around
`coder`, and where the design doc's §6 hard requirement — "explicit scientist
approval before writing/modifying repo code" — is enforced *structurally*: this
skill's own `allowed-tools` has no `Write`/`Edit`, so it is physically incapable of
writing a file itself. It restates a concrete build plan, gets an explicit approval
in-turn, only then dispatches `coder`, and independently re-runs the fast test suite
afterward rather than trusting the subagent's self-report.

**Not yet done**: live-exercising `propose-hypotheses`/`implement-module` against a
real hypothesis and an actual new module — that's a separate, larger next step
(the same "build it, then exercise it in a later turn" sequencing Phase 1's first
two skills followed). Also not done: authenticating any literature MCP (requires the
user's own OAuth browser flow).

**Test suite**: `python -m pytest tests/test_lab_notebook.py -q` → 20 passed.
`python -m pytest tests/ -q -m "not slow"` unaffected (only additive `LabNotebook`
methods changed; no existing signatures touched).
