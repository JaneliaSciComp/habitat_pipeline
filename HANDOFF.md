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
