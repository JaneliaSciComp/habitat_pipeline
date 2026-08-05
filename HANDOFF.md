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
- [ ] Design the lab-notebook schema on the `database/` layer (per-iteration: hypothesis, params, git commit, session, metrics, figure paths, citations, test-family for MC accounting, scientist decision).
- [ ] Decide experiment-tracking tool (MLflow self-hosted vs W&B).
- [ ] Do NOT touch `config/` or delete `.gui_cache/` without asking (per CLAUDE.md).
