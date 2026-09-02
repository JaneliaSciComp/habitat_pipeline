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

## `propose-hypotheses` live-tested; two hypotheses run — done 2026-08-17/18

Exercised `propose-hypotheses` for the first time, grounded in iteration 7's
opponent-identity finding. Generated and ranked 4 candidates (all flagged
"not literature-grounded" — the literature MCPs are still unauthenticated); the
scientist picked two, pre-registered as Hypothesis #2 and #3.

- **Hypothesis #2** (does opponent decodability survive coarsening to a 2-group
  split?) — run via `run-analysis`, iteration 8: population accuracy 57.5% vs. 57.2%
  baseline (negligible raw margin), but population p=0.005 and 20/149 cells survive
  FDR (more than the original 8-way analysis's 16). `interpret-results` verdict:
  **Real, corrected**, but complicates rather than confirms the hypothesis's own
  wording — the effect persists under coarsening but its raw accuracy margin
  collapses, arguing against a clean individual-vs-coarse-rank dichotomy.
- **Hypothesis #3** (are the 16 FDR-significant opponent-identity cells actually
  spatially confounded with opponent location?) — **blocked**, iteration 9: session
  `20251216`'s tracking file only has the focal animal (`rat631`) identity-resolved,
  no opponent positions exist for that session. Not a code problem — logged as
  `status='blocked'`, `Hypothesis.status` left at `'proposed'` with notes explaining
  why, rather than forced into `'rejected'`. Re-derived the 16 significant cluster
  IDs anyway for future use: `470, 519, 528, 590, 707, 752, 766, 862, 955, 1016,
  1019, 1020, 1107, 1159, 1167, 1168`.

## Hypothesis backlog — done 2026-08-18/19

Brainstormed novel analyses exploiting the multi-animal/simultaneous-ephys setup,
merged with a second list contributed directly (`docs/HYPOTHESIS_LIST.md`) into one
feasibility-checked, difficulty-tiered backlog: **`docs/HYPOTHESIS_BACKLOG.md`**.
Cross-checking against the actual repo surfaced several corrections worth
remembering:

- Allocentric social place fields and conjunctive self×partner cells are **already
  built** (`ephys/social_spatial_fields.py`) — not new work, contrary to how the
  contributed list framed them.
- `ephys/decode_location.py` already supports decoding *any* tracked animal's
  position from *any* animal's spikes (`object_name` is independent of whose
  `KilosortData` is passed), with CV and a null baseline built in —
  partner-position decoding is nearly free, not a new-module item.
- **Most cohort-7 sessions actually have 4 simultaneously-implanted animals**
  (`rat613`, `rat615`, `rat630`, `rat631` — confirmed via `get_animals_and_sessions()`),
  including session `20251216` itself. This corrects an initial assumption that
  N>2-ephys was unconfirmed — multi-brain (N>2) methods are blocked on missing
  *math* (multi-set CCA isn't implemented), not missing data.
- Real head direction doesn't exist yet (`orientation` passes through tracking
  unvalidated; heading is derived from movement velocity instead) — gaze/attention
  ideas are genuinely blocked on that prerequisite.

## Partner-position decoding exercise — done 2026-08-19

Ran the backlog's cheapest item (partner-position decoding via `decode_location.py`)
against session `20251210` (chosen because, unlike `20251216`, it has full
multi-animal tracking *and* 4-animal ephys — see the sync/tracking check in this
session's transcript). Result is a caution, not a finding — full writeup in
`docs/HYPOTHESIS_BACKLOG.md`'s "Update 2026-08-19" section:

- The module's default `null='reverse'` is a weak, order-only null (explicitly
  preserves autocorrelation per its docstring) — a first pass using it produced a
  spurious "win" for a near-stationary animal and cases where the null beat real
  decoding outright. Re-ran with `null='shuffle'` + `ephys._stats_utils.empirical_p_value`
  + BH-FDR instead (iterations 10 then 11) — the same "always use a proper
  permutation null" lesson Phase 1.5 already taught the LDA decoders, now confirmed
  to generalize to this older module too.
- **Self-position decoding failed the proper shuffle-null test** (p=0.43) with
  default parameters — the basic sanity check any spatial decoder needs to clear
  before a partner-decoding claim means anything.

## `decode_location.py` CV-leakage fix — done 2026-08-19

Root-caused the self-decoding failure above: `ephys/decode_location.py::_cv_decode`
used `KFold(n_splits=cv_folds, shuffle=True, random_state=42)`. Position is
autocorrelated in time (like distance in `decode_partner_distance.py`), so shuffled
folds leak adjacent, near-identical time bins between train and test — this repo
already has an established convention against exactly this
(`decode_partner_distance.py`/`ephys.inter_brain_dynamics._fit_r2` both use
`KFold(shuffle=False)`), just not yet applied here. Confirmed empirically before
fixing: patching in contiguous folds alone dropped self-decoding's p-value from 0.43
to 0.138 (both real and null errors got larger and more honest — the leakage had
been inflating both comparably, hiding a real effect under two artificially-tight
numbers); adding `rate_smoothing_sigma=2.0` (standard for coarse-bin Bayesian
place decoders) on top got it to p=0.022.

**Fixed**: `_cv_decode` now uses `KFold(shuffle=False)`. Added a "Statistical notes"
docstring section to the module (matching the convention in
`decode_partner_distance.py`/`_lda_decoding.py`) documenting the contiguous-fold
requirement and the null's autocorrelation-preserving design. Added
`tests/test_decode_location.py` — **this module had zero test coverage before**:
one test spies on `sklearn.model_selection.KFold`'s call to assert `shuffle=False`
(direct regression guard), one confirms contiguous folds actually produce
contiguous index blocks, one is a synthetic place-tuned-population sanity check
(self-position must beat a shuffle-null by a real margin). 270 passed (was 267),
no regressions in the rest of the suite.

**Final honest result** (iteration 12, animal 631/session `20251210`, 7-object
family, `n_shuffles=180`, contiguous folds + `rate_smoothing_sigma=2.0`): self
(631) improves to p=0.022 nominally but does **not** survive BH-FDR (q=0.077,
7 tests) — the sanity check is only borderline at this shuffle budget. One partner
(`rat613`) does clear FDR (q=0.039), but it's pinned at the `n_shuffles=180` p-floor,
and given self-decoding's own borderline status, **not treated as a finding yet**.
The leakage fix is a confirmed, real correctness improvement; the underlying
neuroscience question (does this population encode partner position) is still
open, likely needs a larger shuffle budget (~500) to resolve self-decoding
conclusively one way or the other before revisiting partner claims.

**Still not done**: live-exercising `implement-module`/`coder` against a real new
module (every hypothesis run so far only needed existing modules); authenticating
any literature MCP; the multi-session sweep; held-out-session enforcement; raising
`decode_location`'s shuffle budget to resolve self-decoding conclusively.

## Layers 0 + 7 + HTML reports — done 2026-08-20

Built the first increment of the scientist's 8-layer architecture (plan:
`~/.claude/plans/i-want-to-brainstorm-lazy-emerson.md`). **Layer 0** (capability
manifest + hazard registry), **Layer 7** (multiple-comparisons ledger, declared-family
budget, holdout registry), and **Layer 5's** per-hypothesis HTML report. Layers 1, 2,
3, 4 and 6 remain out of scope, except for the minimum slice of Layer 2 that Layer 7
requires: a frozen prediction has to exist for "no post-hoc redefinition without a new
frozen record" to mean anything.

**Test suite: 743 passed, 3 skipped** (`-m "not slow"`, ~70 s), up from 270. Nothing
committed.

### The finding that justified the increment

Iteration 12's headline claim was computed on a post-hoc-shrunk denominator. Iteration
10 ran all 12 tracked objects; iterations 11–12 dropped 5, moving m from 12 to 7, and
four of those five exclusions are justified in prose *by iteration 10's own results*.

| declared m | `rat613` q | verdict | `fdr_resolution(m, 180).resolvable` |
|---|---|---|---|
| 7 (as logged) | 0.0387 | significant | **True** |
| 11 (honest: 12 − the one degenerate target) | **0.0608** | **not significant** | **False** |

The exclusion did not merely improve q-values — it flipped the repo's *own* resolution
guard from "unresolvable" to "resolvable", because the guard was handed the count the
exclusion produced. So it manufactured both the significance and the permission to
claim it. `tests/test_lab_notebook_ledger.py::TestIteration12GoldenRegression` pins
both answers side by side so the bug cannot come back quietly.

### What Layer 0's first manifest build found

`python scripts/build_capability_manifest.py --probe-level paths` — 34 sessions,
101 s, zero build errors. The inventory is the finding:

| | cohort 7 | cohort 5 |
|---|---|---|
| sessions | 14 | 20 |
| with tracking | **2** (`20251210`, `20251216`) | **0** (no tracking directory at all) |
| with scored events | **1** (`20251216`) | 2 |
| 4-animal ephys | 7 | 0 |

**Only one session in the entire dataset has scored behavioural events.** "47
animal/session pairs" is true of *ephys* only. Consequences:

- Every event-based analysis (`decode_opponent_identity`, `decode_event_outcome`) has
  **n = 1 session**.
- The EC opponent-identity finding has **no path to holdout confirmation** — there is
  no second event-scored session to reserve. Per the scope decision the holdout
  mechanism ships built and *empty*, and this constraint is recorded rather than
  worked around. It is not "confirmation pending"; it is "confirmation not currently
  possible".
- `docs/HYPOTHESIS_BACKLOG.md`'s cross-day-stability item and the deferred
  multi-session sweep are both far more constrained than they read.

**Three versions of session 20251216's scoring exist on disk**, and
`DataStorageManager` resolves the oldest (it only looks inside date-named
directories):

| file | modified | rows | 631 outcome events |
|---|---|---|---|
| `20251216/20251216_behavior_event_df.csv` ← analyses use this | 2026-03-02 | 641 | **19** (12 W / 7 L) |
| `behavior_event_df.csv` (loose) | 2026-08-13 | 688 | **29** (20 W / 9 L) |
| `behavior_event_df_update.csv` (loose) | 2026-08-13 | 693 | **29** (20 W / 9 L) |

**Decision: the dated directory is canonical.** Guarded by `HZ-DATA-007` so nobody
wires a loose file in by accident. Worth knowing that Hypothesis 1's "underpowered
null" was computed on the 19-event version; the discrepancy between the three files
has not been explained.

### Layer 0 — `discovery/`

`hazards.json` holds **29 entries**, 16 with executable detectors (`callable` →
`ephys._stats_utils` or `discovery/detectors.py`; `test` → a pytest node id).
`validate_registry` enforces an **earn-your-place rule** — an entry qualifies only if
it demonstrably caused a wrong result here, or it carries a real detector — plus
importlib-resolvability of every callable and (under `slow`) collectability of every
test node id, so a renamed function can't leave a decoration that looks like a guard.

The module's central invariant: **a detector that cannot run reports
`ran=False, passed=None`, never `passed=True`.** `TestCannotCheckIsNotAPass` attacks
that from seven directions. A safety layer that reports unverified safety is worse
than none, because it displaces the scepticism that would otherwise have applied.

`capability_manifest.py` is the **consult** path and must never import
`ingestion`/`video`/`ephys`; a subprocess test enforces it, because if the cheap check
could reach the SMB share, eventually something would make it. `manifest_status()`
**raises** on a missing file, a schema mismatch or a changed cohort config, and only
warns on age — a quietly stale manifest is worse than no manifest, since the agent
will trust it. Don't soften those raises.

Requirements vs. hazards is a deliberate split: a **requirement** answers "does the
data exist" (a missing prerequisite, e.g. no validated head direction, belongs in
`discovery/requirements.py`); a **hazard** answers "will this silently lie".

### Layer 7 — additive to `database/lab_notebook.py`

Four new tables (`family_tests`, `holdout_reservations`, `frozen_predictions`,
`hypothesis_verdicts`) plus 10 retrofitted columns via `_ensure_added_columns`. The
real database migrated cleanly (snapshot taken first); legacy rows read `NULL` =
"not recorded".

- **The atomic unit is a *declared test*, not an iteration** — it exists before it
  runs and survives being abandoned, which is what makes the denominator knowable.
  One iteration can hold many tests (`decode_all_locations` sweeps 12 objects into one
  result dict).
- `family_fdr` pads declared-but-unrun members with p=1.0 and calls the existing
  `benjamini_hochberg` **unchanged** — that reproduces conservative BH at the declared
  m exactly, so `ephys/_stats_utils.py` needed no edit at all.
- `abandon_family_test`'s `outcome_dependent` is **required with no default**. An
  outcome-dependent exclusion *stays* in the denominator: deciding to drop it was
  itself a test.
- **The old holdout gate was inverted.** `held_out` was an `Iteration` column, so the
  prescribed check returned nothing for an untouched session and passed — it could
  only fire after being violated. Now a session-level registry, keyed on the
  normalized 8-digit date (the DB says `'20251210'`, the directory says
  `RatCity_20251210_1359_40Hz`), failing **closed** on an unresolvable id.
- **The tier is derived, never stored** — a stored flag is a field someone edits to
  make a banner go away. `evidence_tier` recomputes **eight** conditions. Every
  hypothesis today fails at the first, which is the correct current state.

Honest scoping of the budget: **structural at the record seam, instructional at the
compute seam.** Nothing can stop an agent running a decoder forty times in ad-hoc
Python. What becomes impossible is laundering forty runs into the record as seven
tests with a clean q-value. That is the right target — the problem is the garden of
forking paths, not compute cost.

### Design gaps found while building, all closed

1. **A spent holdout still counted.** Nothing checked whether a reserved session had
   already been analysed *before* being unlocked, which destroys the only thing a
   holdout provides. Now condition 8, and it catches contamination from a *different*
   hypothesis too — the loop having seen the session while chasing another question
   contaminates it just as thoroughly.
2. **`HZ-STAT-001` bound `n_tests` to `n_quality_cells`** (377) — right for per-cell
   decoding, wrong for `decode_location`, whose family is the object sweep. Now bound
   to the ledger's declared denominator, and it reports "cannot check" rather than
   confidently using the wrong number.
3. **An undeclared family reported a denominator of 0** and crashed `fdr_resolution`
   on real data. Legacy families predate the ledger: their denominator is
   *unrecorded*, not zero. New `denominator_status='undeclared'`.
4. **The manifest's label counts were read from the wrong tuple slot.** All three
   `extract_*_labels` return `(start_times, end_times, labels)`; the probe
   destructured index 1. Caught by a mock test *before* the multi-hour full probe ran.

### Correction to earlier framing in this file

`ephys/_lda_decoding.py`'s three `StratifiedKFold(shuffle=True, random_state=42)`
sites are **not** the same defect as the `decode_location` leak. That one shuffled
0.5 s *position bins* — adjacent, near-identical samples of an autocorrelated signal.
These split **one row per behavioural event**, seconds to minutes apart. The genuine
residual risk is narrower (bouts of events clustered in time, which would want grouped
folds keyed on event time), and the fixed `random_state` across the observed run *and*
every permutation is correct because it makes the comparison paired. So it is an
*audit* item (`HZ-STAT-005`, allowlisted with a documented rationale and a stale-entry
check), not a bug — and changing it would move the published 16-significant-cells
result, so it is a scientist decision explicitly off-limits to `coder`.

### Live observation: the pathology happening in real time

Iterations 13–18 were logged during this session from a concurrent process. Iterations
14 → 15 → 16 are one search with a growing headline number and no recorded
denominator: all 8 EC opponents (28.8% vs 27.7% baseline) → a hand-picked
`rat613`/`rat635` pair (55.5% vs 50.0%) → `rat634`/`rat635` with
`min_events_per_class=1` and `cv_folds=4` (**68.4%** vs 55.6%). All have
`test_family_id=None`. Seeded as `HZ-STAT-013`; left in place at the scientist's
direction.

### Backfill applied 2026-08-21

`scripts/backfill_ledger.py --apply --confirm-exclusion-classification` ran against the
real notebook (snapshot taken first). Result, readable via
`notebook_cli.py ledger 4 --n-shuffles 180`:

```
family 4: 12 declared, 7 run, 4 abandoned, 1 excluded a priori
  m used for correction: 11 (status: reconstructed)
  resolvable at 180 shuffles: False (best achievable q 0.0608; 220 would be needed)
  object=613  p=0.005525  q=0.0608     <- logged as q=0.0387, significant
  object=631  p=0.022099  q=0.1215     <- logged as q=0.0773
```

No test in the family is significant at the declared denominator. The report shows
"q as logged" beside "q at declared m=11" so both numbers stay visible.

Judgment calls as applied: the near-stationary target counts as `excluded_prespecified`
with `applied_after_seeing_results=True` (and the verdict is identical at m=10, 11 and
12, so the call doesn't change the science); the four reverse-null drops are
outcome-dependent and stay in the denominator; iterations 3/4 are two declared tests;
the pending scientist decisions were **not** bulk-set. Hypothesis 1 was left at
`'proposed'` per the scientist's decision — no verdict row was written for it.

Two bugs surfaced during the apply and are fixed with regression tests:

1. **The emptiness check ate its own output.** "Invalidate any family with zero
   iterations" wrongly invalidated the reconstruction families, which deliberately hold
   `family_tests` and no iterations. It only appeared on a second run, because the first
   run's families aren't in the first run's survey. Now requires zero of *both* and skips
   the reconstruction names by name. The two wrongly-invalidated families were repaired
   back to `'open'` with a note recording why.
2. **The hard rule was too strict to be useful.** "Never UPDATE an iterations row" meant
   the reconstructed family was invisible from the iteration, so the report kept saying
   "no test family is attached" while a correctly-declared family sat beside it. Narrowed
   to: a **NULL** `test_family_id` may be filled (adds a pointer, changes no
   measurement); an existing one is never overwritten; no other column is touched, which
   a full row-hash test enforces. Iterations 10-12 are now linked to family 4.

### One unreproduced test flake — worth watching

`tests/test_kilosort_data_loading.py::TestKilosortDataLoading::test_load_complete_dataset`
failed **once**, in a full-suite run, and I could not reproduce it: it passes in
isolation, passed 5/5 in a targeted loop with its neighbouring file, and passed in 3
consecutive full fast runs and one full slow run afterwards. The assertion text wasn't
captured before it stopped recurring.

Not code this increment touched. The plausible mechanism, unverified: `_find_cached_file`
in `ingestion/kilosort_data_import.py` picks a cached pkl by `sorted(..., key=st_mtime,
reverse=True)`, which is order-nondeterministic when two files land in the same clock
tick — and there was a concurrent session writing to this repo throughout. Flagging it
rather than calling the suite unconditionally green; if it recurs, capture the assertion
and start there.

### Full probe of 20251216, and three bugs it exposed — 2026-08-26

`--probe-level full --sessions 20251216`, 87 s. The probe validated against known
ground truth (rat631: 263 clusters, 149 quality cells, EC 8 classes / 173 events,
outcome 19 events at a 0.632 majority baseline — all matching the record exactly),
and turned up something new: **the EC analysis is runnable on two more animals in
the same session.** rat630 has 8 usable opponent classes / 123 EC events / 114
quality cells; rat613 has 5 classes / 118 events / 187 quality cells. The EC finding
has only ever been tested on 631. rat615 has 269 clusters but **0 quality cells**, so
it is out.

Then three bugs, two of them mine, all fixed with regression tests:

**1. "Fraction of the recording covered" was never a coherent session-wide
quantity.** Animals in one session share a clock, not a recording length: 20251216's
four durations are 18866 / 3651 / 18556 / 9960 s. The single tracking window
`[687.3, 8187.2]` (span 7499.9 s) therefore covers 39.8% / 205% / 40.4% / 75.3% of
"the recording" — and 205% is impossible, which is the tell. My probe divided by
whichever animal sorted first and recorded 0.3975 as *the* coverage. Now
`tracking.coverage_by_animal`, with the scalar kept only as the most conservative
value and naming its reference animal, and ratios above 1.0 flagged in
`coverage_exceeds_recording` rather than reported as coverage. Same fix applied to
`events.frac_events_within_recording_by_animal` after the same bug appeared there:
**93.8%** of events fall inside rat631's recording, against 39.8% inside rat615's.

**Correction to this file and the backlog**: the "roughly 63%" coverage figure
recorded on 2026-08-20 was filename arithmetic, never measured, and wrong for every
animal. For rat631 it is **75.3%**.

**2. `manifest_status` trusted a field that records an intention, not a fact.**
`generated_by.probe_level` says what the last run *asked for*, so probing one session
at 'full' left it claiming 'full' while 33 sessions were paths-only — and the status
then reported `fresh` with no warning. Now derived from the sessions themselves,
yielding `paths` / `mixed` / `full`, with the count of fully-probed sessions.

**3. `check_testable` could not tell "this data doesn't exist" from "nobody
looked".** Both surfaced as an unmet requirement, so a paths-level session reported
`NOT TESTABLE` for content requirements — a confident claim about data nobody had
examined. Absent-because-unprobed is now `undetermined`, with the exact rebuild
command in the message.

**Also fixed**: `--force` discarded the entire artifact rather than re-probing the
selected sessions, contradicting its own help text. One `--force --sessions X`
silently dropped 33 already-probed sessions (recovered by re-running paths, 101 s).
`--force` now re-probes the selection and keeps the rest; `--rebuild` is the explicit
start-over flag.

Manifest state after all this: 34 sessions, `probe_level=mixed`, 1 fully probed,
correctly self-reporting as `partial`. Suite: 781 passing.

## First fully pre-registered cycle — hypothesis 4 refuted — 2026-08-26/27

Not previously recorded here. The Layer 7 machinery ran end to end for the first
time, on its own terms: **frozen prediction #1** (`min_q_value < 0.05`, alpha 0.05,
200 shuffles planned, 3 declared test keys, `registered_post_hoc=False`,
`spec_hash 2c21bb0e…`), **family 6 declared before any run**, then iterations
19-21, then a verdict.

Hypothesis 4: the EC opponent-identity effect is a property of this cohort's
coding rather than of animal 631 — it should also appear in rat630 and rat613,
recorded simultaneously in the same session. Family 6 is the first clean
denominator in the project: *3 declared, 3 run, 0 abandoned, m=3, status
`clean`, resolvable at 200 shuffles* (best achievable q 0.0149).

| test | p | q | accuracy vs own majority baseline |
|---|---|---|---|
| `animal=rat631` | 0.004975 | 0.0075 | 0.2881 vs 0.2775 — clears it |
| `animal=rat630` | 0.004975 | 0.0075 | 0.2184 vs 0.2358 — **below** |
| `animal=rat613` | 0.920398 | 0.9204 | 0.3449 vs 0.4068 — **below**, 0/187 cells |

**Verdict: `refuted`** (tier `exploratory`, denominator known, m=3, decided_by
`agent`). rat630 beats its permutation null while failing to beat
always-guessing-the-most-frequent-opponent, which is exactly the pair of
statistics Phase 1.5 exists to report together. `evidence_tier(4)` still returns
`exploratory`, blocked at condition 1 — no holdout was reserved or spent.

## Recording-level path resolution — 2026-09-02

`DataStorageManager` could address **47 of the 94 animal-recordings that exist on
the share**, and the "47 animal/session pairs" figure quoted throughout this file
was that reach, not the dataset. Two independent causes, both in
`ingestion/data_paths.py`, which hard-coded `f"{session}_merged.kilosort"` in three
places:

1. **A `.rec` directory is a day, not a recording.** `20251216_094334.rec/rat613/`
   holds `20251216_094334_merged.*`, `20251216_144334_merged.*` and
   `20251216_194334_merged.*` — morning, afternoon and evening blocks, each with its
   own kilosort output, DIO (all Din channels), LFP and timestampoffset. Only the
   block whose timestamp matched the directory name was ever resolvable. **26 nested
   recordings in cohort 7, 25 with real spike files**, that no analysis has touched.
2. **The suffix is not fixed.** Cohort 5 writes `_merge` (21 of 24 animal
   directories), one directory has no suffix, and one cohort-7 directory is
   `rat613_20251209_160716_merge`. So **22 of 24 cohort-5 animal directories were
   unreachable while containing spike files.** The two cohort-5 sessions that ever
   ran (`20250819_141715`, iterations 17-18) did so because they happen to use
   `_merged` — a filename coincidence, not a property of the data.

**Fixed** by discovering the stem from disk (`_discover_recordings`) instead of
building it. `session_id` may now name any recording. Backwards compatibility is
the load-bearing property and is pinned by
`tests/test_data_paths_recordings.py::TestNothingThatResolvedBeforeMoves`:
`'20251216'` and `'20251216_094334'` still resolve to the morning block, so all 21
logged iterations still mean what they said. A full stamp matching nothing now
**raises** rather than falling back to the primary (the first version silently
returned the morning recording for `'20251216_030303'` — caught by its own test);
several matches with no primary among them also raise. `get_animals_and_sessions`
yields one row per recording with `recording_id`/`stem`/`session_dir`/`is_primary`,
and the GUI picker, `enumerate_targets` and `suggest_sessions` all follow from it.
Paths cache bumped to `_CACHE_VERSION = 2`, because a v1 entry holds a `_merged`
path that may not exist.

Counts after the fix: **cohort 7 72 rows / 28 sessions, cohort 5 24 rows / 20
sessions** — 96 animal-recordings, 94 with spike files, against 47 addressable
before. The 14:43 block of 20251216 loads (177 clusters) and syncs cleanly.

**But addressable is not loadable, and the full probe is what established the
difference** (see below): 63 of the 96 load end to end, up from 38. Cohort 5 gains
almost nothing in practice.

### The hazard the fix created — `HZ-DATA-008`

Tracking and events resolve by **8-digit date**, so on a multi-recording day every
block is offered the same files. `20251216` has exactly one tracking file spanning
09:50-12:00 — inside the morning block. Mapped through the 14:43 block's own sync it
falls outside that recording entirely, and since that recording loads and syncs
cleanly, **nothing would have errored**.

**The blocks of a day share one clock and run back to back**, which is what makes
the overlap check possible and what made the first version of it wrong. Measured on
20251216/rat613: `[6.7, 18897.5]`, `[19017.6, 36569.2]`, `[36732.6, 42054.0]` s —
sequential, non-overlapping, so the nested blocks are genuinely new data rather than
re-sortings of the same period. But my first attachment check compared the tracking
window against `[0, duration]`, and `duration_seconds` is a *span*, so every block
looked like it started at the origin: the afternoon block reported
`overlap_verified` against a file recorded five hours before it began. Now
`probe_ephys` records the measured `ephys_window` and everything downstream uses the
real interval. Pinned by
`tests/test_manifest_build.py::TestAttachmentToTheRightRecording`, which keeps the
wrong answer beside the right one.

**Two documented numbers were wrong as a result**, both corrected in `CLAUDE.md`:

- rat615's "impossible 205% coverage" — offered on 2026-08-26 as the *tell* that a
  session-wide scalar was incoherent — was this bug, not bad data. Its recording is
  `[377.0, 4028.7]`, a 3651 s block starting 377 s into the day; against its real
  interval the coverage is **91.5%**.
- rat631's duration of **9960 s is not real**. It comes from a cached pkl whose raw
  `spike_times` is empty and whose stale `_duration_seconds` therefore wins, while
  the `spike_times_by_cell` in the *same file* span 18370 s. So the 75.3% coverage
  recorded for rat631 on 2026-08-26 — itself the correction to an earlier error — is
  also wrong; it is **40.8%**. `probe_ephys` now sets
  `duration_disagrees_with_window` when the two disagree by more than 5%.

Per the scope decision, attachment is decided **by overlap, computed at probe time**:
`probe_tracking`/`probe_events` map the file's window through *this* recording's sync
and record `attachment_status` ∈ {`overlap_verified`, `no_overlap`, `undetermined`},
setting `available=False` on `no_overlap`. An uncomputable answer stays
`undetermined` and never becomes a pass, per the layer's central invariant. Path
resolution cannot decide this (it needs a DIO read and must stay cheap), so
`DataStorageManager` only logs a warning — hence "structural in the manifest,
advisory in the resolver". Registry now 30 entries; two warning-severity
requirements added so `check_testable` surfaces it.

## The full manifest probe — done 2026-09-02

`--probe-level full --rebuild`, 3156 s (53 min), 48 sessions, 96 animal-recordings,
zero fatal build errors. `manifest_status()` now reports **fresh, `probe_level=full`,
48 of 48 fully probed** — the first time the artifact has been anything other than
`partial`, and the item has been open since 2026-08-20.

### The inventory, honestly

| | count |
|---|---|
| animal-recordings enumerated | 96 |
| with spike files | 94 |
| **loadable end to end** | **63** (was 38 before the resolver fix) |
| of those, newly-addressable non-primary blocks | 25, carrying **4320 of 12227 quality cells** |

**"Addressable" and "loadable" are not the same thing, and the gap is 31
recordings.** `load_kilosort_data` needs a `*.timestamps.dat` beside `kilosort4/` —
`_read_timestamps` uses it to map Kilosort's spike *indices* onto the day's absolute
sample clock — and that ~2 GB file is absent for:

- **23 of cohort 5's 24 animal-recordings.** So cohort 5's paths now resolve and its
  data still cannot be loaded; the practical gain there is one recording
  (`20250819_141715`, the one that already worked). Anything planned on cohort 5
  is blocked on those files, not on code.
- **every animal of cohort-7 `20251209_160716` and `20251213_102418`** — two entire
  sessions.

`verify_kilosort_path` reports these as verified because it only checks
`spike_times.npy`/`spike_clusters.npy`, so the failure appears at load time. Recorded
per animal in the manifest as a `FileNotFoundError` load error and documented in
`CLAUDE.md`; it is a missing-file problem on the share.

### Other things the probe established

- **`duration_disagrees_with_window` fires on 7 animal-recordings**, six of them
  rat631 (`20251212`, `20251214` ×2 with rat630, `20251215`, `20251216`,
  `20251219_151017`, `20251220_104752`). So the stale-cached-duration problem is not
  a one-off on 20251216 — any coverage figure derived from `duration_seconds` for
  those animals is suspect.
- **20 recordings sit on multi-block days**, which is the population `HZ-DATA-008`
  guards. The three non-primary blocks of 20251216 correctly report `no_overlap` for
  both tracking and events, and `check_testable` returns not-testable for every
  tracking- and event-based analysis on them.
- Tracking attachment is `overlap_verified` for exactly two sessions
  (`20251210_110059`, `20251216_094334`) and events for two
  (`20251216_094334`, cohort-5 `20250819_141715`) — the rest is `undetermined`
  because no file exists for those dates at all.

### Still not done

- **Holdout reservations.** The decision was to probe first, then reserve from the
  inventory — the probe reads metadata only and tests no hypothesis, so it spent
  nothing. The 25 newly-loadable blocks are the obvious candidate pool.
- **~25 recordings have never been analysed.** They have no video and no scoring, so
  they are **ephys-only**: inter-brain, population geometry, and anything that does
  not need behaviour. Whether they are behaviourally comparable to the morning
  blocks is unknown.
- The 31 unloadable recordings need their `*.timestamps.dat` produced or copied.
- `docs/AI_DISCOVERY_LOOP_DESIGN.md` §5/§6 amendments.
- Every gate at the *compute* seam is still instructional. Per the scope decision only
  the git entries were removed from `.claude/settings.local.json`; the broad
  `python -c` wildcards remain, so `python -c` is still an unprompted write primitive
  and `implement-module`'s "structural" claim holds only against the literal
  `Write`/`Edit` tool names.
- 18 iterations still have `scientist_decision='pending'`, and 15 are attached to no
  hypothesis. Deliberately **not** bulk-set: record them going forward via
  `notebook_cli.py decide`.
