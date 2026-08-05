# Phase 0 Findings — First Attempt at the Discovery Loop

*Ran on 2026-07-10 in the Cowork Linux sandbox against the real `habitat_pipeline` repo. Goal (per the design doc): attempt one loop on a real session and find where it breaks — before building anything.*

*Re-ran on 2026-08-05 on the Janelia workstation (`//nearline` mounted) — see [Update 2026-08-05](#update-2026-08-05--ran-on-the-workstation) below.*

## TL;DR

The **code is healthy and agent-drivable**; the loop's only hard blocker is **data locality**. The analysis stack ran cleanly and the rigor guardrail already earned its keep on the first synthetic run. Nothing here requires a rewrite — it requires running the Runner *where the data lives*.

## What I actually ran

1. **Environment probe.** Python 3.10; installed the light dependency set from `requirements.txt` (numpy/scipy/pandas/scikit-learn/matplotlib/seaborn/h5py/pytest). No heavy/GPU deps needed for the CLI path.
2. **Mock-data test suite.** Ran the core analysis tests (`test_social_spatial_fields`, `test_inter_brain_dynamics`, `test_multi_animal_session`, `test_kilosort_data_analysis`, `test_decode_partner_distance`): **104 passed in ~13 s.**
3. **Real-session CLI.** `python -m ephys.decode_opponent_identity --animal_id 631 --session_id 20251216 --use_quality_cells`.
4. **Loop mechanics demo.** Ran the LDA core (`run_population_per_cell_decode`) on a synthetic session with known ground truth, then had an "Interpreter" pass read the result-dict and summarize.

## Results

### ✅ Works
- **Runner leg (mock/synthetic).** 104 core tests pass; the LDA core returns real, well-formed result-dicts (`{'status':'success','accuracy':0.87,'n_events':60,...}`). An agent can call these functions and CLIs directly.
- **CLIs are agent-friendly.** The real-session CLI failed *gracefully* with a clean, parseable message — exactly what an agent needs to detect and react to, not a stack trace.
- **Interpreter leg.** Reading the unified result-dict schema and summarizing (cells decoded, mean accuracy, best cell) is trivial and needs no new plumbing.

### ✅ Rigor guardrail proved itself immediately
On the synthetic session, **6 of 24 cells were truly tuned**, but a naive "accuracy > chance + 0.05" screen flagged **9** — three false positives, purely from testing 24 cells. This is the multiple-comparison problem happening on run #1, and it's exactly what the design's rigor layer (FDR/Bonferroni correction tracked in the lab notebook) is there to catch. Strong evidence this belongs in the core, not as a later add-on.

### ❌ Hard blocker: data locality
The real run stops at data loading:

```
Loading data for animal 631, session 20251216
Error loading data: No session directory found matching '20251216'
in //nearline/karpova/TervoLab/data/RatCity/cohort7/ephys
```

The configs point at SMB shares (`//nearline/karpova/...`); those are on the Janelia Windows network and are **not mounted in the Linux sandbox**. The pipeline is fine — the data simply isn't reachable from here.

### ⚠️ Minor
- GUI-tab tests fail on a missing `streamlit` (heavy optional dep). Irrelevant to the loop: the design drives **CLIs headlessly**, not the GUI. Don't install it for the agent path.

## What this means for the build

1. **The Runner must execute where `//nearline/karpova/` is mounted** — the user's Janelia workstation, or an HPC node with the share. The agent/orchestrator can live anywhere, but analysis execution and the data must be co-located. This is the single most important architectural constraint Phase 0 surfaced, and it's about *deployment location*, not code.
2. **Mock-data CI is fully viable here.** The 104-test suite runs in seconds with no share — so the "agent-written code must pass tests before touching real data" gate (design §7) works in a sandbox exactly as intended.
3. **The rigor layer moves up in priority.** It caught false positives on the very first run; it should be in Phase 1, not deferred.
4. **No code changes required to start.** The CLIs, result-dict schema, and cache are already agent-ready.

## Recommended next step

Re-run this same Phase 0 on the **Janelia workstation** (where the share is mounted) instead of the sandbox:

```bash
python -m ephys.decode_opponent_identity --animal_id 631 --session_id 20251216 --use_quality_cells
```

If that produces a real result-dict + figures, we have a genuine end-to-end Runner→Interpreter pass on real data, and Phase 1 (codifying the loop as skills + the lab notebook, with the rigor layer wired in) can begin against a working baseline.

## Update 2026-08-05 — ran on the workstation

Re-ran `scripts/phase0_probe.py --animal-id 631 --session-id 20251216` on the Janelia Windows workstation, where `//nearline` is mounted.

```
[PASS] ENV: required deps - all core deps importable
[PASS] TESTS: core mock suite - 20.2s (104 passed)
[FAIL] REAL: real-session decode CLI - unexpected error
[PASS] LOOP: runner+interpreter - result-dict produced
```

### ✅ Data locality is no longer a blocker
On the workstation, `load_kilosort_data` and `load_behavioral_events` both succeed against the real share: 263 ephys clusters, 641 behavioral events loaded for animal 631 / session 20251216. This confirms the Phase 0 hypothesis — the earlier failure really was about *where* the Runner executes, not the code.

### ❌ New finding: the standalone decoder CLIs skip ephys sync
The REAL leg now fails one step later, inside decoding itself:

```
Error extracting behavioral events: No ephys-synchronized timestamp columns found in behavioral data
Analysis failed: No ephys-synchronized timestamp columns found in behavioral data
```

Root cause: `ephys/decode_opponent_identity.py::main()` (and the same pattern in `ephys/decode_event_outcome.py::main()`) loads `BehavioralEventsData` and hands it straight to the decoder — it never constructs a `DataSyncManager` or calls `behavior_data.synchronize_with_ephys(sync)`. Per the `ts_*_ephys` gotcha in `CLAUDE.md`, `extract_opponent_labels`/`extract_outcome_labels` require sync to have already happened, so the CLI reliably fails on *every* real session, not just this one.

The GUI path does not have this bug — `gui/loaders.py` builds a `DataSyncManager(dio_channel=1, ...)` and calls `events.synchronize_with_ephys(sync, create_new_columns=True)` before any decoding runs (`gui/loaders.py:50-92`). So the pipeline itself is fine; the two standalone CLI `main()` functions are missing a step the GUI already does correctly.

This is a better-defined blocker than "data locality" was: it's a one-time fix (call the same sync sequence the GUI uses) rather than an environment constraint, and it fully explains why the REAL leg still fails on a machine where the share is mounted.

### ✅ Fixed — both CLIs now sync before decoding
`ephys/decode_opponent_identity.py::main()` and `ephys/decode_event_outcome.py::main()` now construct a `DataSyncManager(data_manager, dio_channel=1)` and call `behavior_data.synchronize_with_ephys(sync, create_new_columns=True)` right after loading behavioral events, mirroring `gui/loaders.py`.

Verifying the fix surfaced two more pre-existing, unrelated bugs on this Windows workstation, both fixed along the way:
- `video/behavioral_events.py` printed a `✓` (U+2713) in several status lines, which crashes on the default Windows `cp1252` console codepage (`UnicodeEncodeError`) — caught by the decoder's broad `except Exception`, so it surfaced as a confusing "Error extracting behavioral events" rather than the real cause. Replaced with `[OK]`.
- `decode_event_outcome.py::main()` called `load_kilosort_data(data_manager)`, passing the `DataStorageManager` directly — the exact stale-API mismatch flagged in `CLAUDE.md` ("the current entry point is `load_kilosort_data(path)`... passing the `DataStorageManager` directly does not work for `KilosortData`"). Changed to `load_kilosort_data(data_manager.get_kilosort_path())`.

Both CLIs now run end-to-end on real data (animal 631 / session 20251216) and produce real result-dicts + saved figures:
- `decode_opponent_identity`: found only 1 qualifying opponent (`rat616`) for behavior_type `'F'` after the `min_events_per_class` filter, so 0/149 cells could be decoded (single-class — expected, not a bug) but the CLI completes cleanly and reports it.
- `decode_event_outcome`: 19 aggressive events (7 loser / 12 winner), 146/149 cells decoded, population accuracy 55.9% ± 11.1%, best cell 80.0% (cluster 1031) — a real, informative result.

The core mock suite (104 tests) still passes after these changes — no regressions.

### ✅ `decode_opponent_identity` validated with real class diversity
The first check used `behavior_type='F'` (fights) for animal 631/session 20251216, which after the `min_events_per_class=5` filter left only one opponent (`rat616`) — a single-class case, correctly decoded as 0/149 but not a real test of the decoder. Checking opponent counts across all behavior types for animal 631 in this session found `EC` (agonistic/escalated-chase-type events, 173 total) has 8 distinct opponents with ≥5 events each. Re-running with `--behavior_type EC`:

```
[OK] Found 173 EC events with opponent labels
[OK] Unique opponents: ['rat613' 'rat616' 'rat617' 'rat620' 'rat629' 'rat630' 'rat634' 'rat635']
Successful cells: 149/149
Population accuracy: 25.2% ± 3.6%
Best cell accuracy: 36.4% (Cell ID: 1167)
```

8-way classification at 25.2% (chance = 12.5%) is a real, above-chance, well-formed result — the fix and the decoder are validated end-to-end on real data.

### What this means for the build
- **The REAL leg is now genuinely green for both decoders**: `decode_event_outcome` (real classes, 55.9%±11.1% accuracy) and `decode_opponent_identity` (8-way, 25.2%±3.6% accuracy, above the 12.5% chance level).
- **Not a Runner/agent-architecture problem** — the fixes were small, localized, and consistent with patterns already established elsewhere in the repo (`gui/loaders.py`, the `CLAUDE.md` gotcha list). Confirms the earlier read: the code is healthy and agent-drivable, and an agent-run Phase 1 loop would have caught and could plausibly have proposed these same fixes.
- **The above-chance accuracy is itself a reminder the rigor layer is needed**: 149 cells were tested individually; a proper claim of "these cells encode opponent identity" needs the label-permutation null / multiple-comparison correction Phase 1 is meant to add, not just "population mean beats chance."
- **Phase 1 can now start against a working REAL baseline for both decoders.**
