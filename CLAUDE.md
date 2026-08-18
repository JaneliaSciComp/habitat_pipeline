# CLAUDE.md — Notes for Claude

Short, opinionated brief for myself when I come back to this repo. Skips anything already in [README.md](README.md) or [ARCHITECTURE.md](ARCHITECTURE.md); read those first for the big picture.

## What this repo is

A multi-animal electrophysiology + behavior analysis pipeline for the RatCity project (Janelia, Tervo lab). Each "session" is one date of recording from one of several freely-behaving rats in a shared arena; data comes in three streams (Kilosort 4 spikes, video tracking, manually-scored behavioral events) on three different clocks that have to be aligned before anything else works.

The work surface is mostly:
- a Streamlit dashboard ([gui/app.py](gui/app.py)) — primary user-facing interface,
- a Panel/Bokeh dashboard ([gui/interactive_app.py](gui/interactive_app.py)) — linked timeline ↔ rastermap ↔ 3D PCA,
- per-module CLIs for the decoders ([ephys/decode_opponent_identity.py](ephys/decode_opponent_identity.py), [ephys/decode_event_outcome.py](ephys/decode_event_outcome.py)),
- notebooks for exploration ([test.ipynb](test.ipynb) is the active scratchpad).

`workflow.py` is older glue; new work happens in the modules and GUI.

## Mental model

```
DataStorageManager (config-driven path discovery)
    │
    ├── load_kilosort_data(dsm.get_kilosort_path())          → KilosortData  (dataclass, pickled cache)
    ├── load_behavioral_events(dsm.get_behavioral_event_files(), session_id=…) → BehavioralEventsData (dataclass)
    ├── load_tracking_data(dsm)                               → VideoTrackingData (dataclass)
    └── DataSyncManager(dsm, dio_channel=1)                   → linear ephys↔behavior clock map
            └── behavior.synchronize_with_ephys(sync)         → adds ts_*_ephys columns to events_data
```

Three dataclasses + one sync object is the whole data plane. Every analysis module takes some combination of these.

The decoding modules went through a recent refactor (commits `9eab430`, `797de45`): a shared label-agnostic core ([ephys/_lda_decoding.py](ephys/_lda_decoding.py)) and shared plots ([ephys/decoding_plots.py](ephys/decoding_plots.py)) now back both `decode_opponent_identity` and `decode_event_outcome`. Result-dict schemas are unified. Plot titles/axis labels are driven by `results['parameters']['class_label']` and `['analysis_title']` set by the wrapper — when adding a new decoder, follow that pattern.

## Gotchas I keep tripping over

- **`KilosortData` is now a dataclass, not a class with a constructor that takes a path.** The old test/README docs that say `KilosortData(data_input=path)` are stale. The current entry point is `load_kilosort_data(path)`. The Streamlit loader does `load_kilosort_data(dsm.get_kilosort_path())` — passing the `DataStorageManager` directly does not work for `KilosortData` (it does for `load_tracking_data`).
- **Behavioral timestamps are in Linux nanoseconds in the CSV.** `synchronize_with_ephys` divides by 1e9 internally; don't double-convert. Sync writes `ts_start_ephys` / `ts_end_ephys` (seconds in ephys time). Anything that calls `extract_*_labels` needs sync to have happened first.
- **`BehavioralEventsData` is a session-level object, not per-animal**, because events involve multiple animals (initiator/victim/winner/loser). To filter by focal animal, use `extract_opponent_labels(animal_of_interest=…)` rather than constructing one events object per animal.
- **`label_mode='opponent' vs 'group'`** in opponent decoding: `'group'` pools opponents into `'low'`/`'high'` halves by their trailing numeric ID using `_assign_id_groups`; for odd N the middle rat goes to the half its ID is numerically closer to. Useful when you don't have enough events per individual opponent.
- **`decode_event_outcome_*` defaults `behavior_type=None`**, which includes every event where both `winner` and `loser` are populated (not just fights). Pass `behavior_type='F'` to restrict.
- **GUI cache lives in `.gui_cache/`** (git-ignored). Decoding/population results are pickled and keyed on `(session, params)`. Changing any param invalidates automatically; force a recompute by deleting the relevant pkl.
- **Rastermap and `umap-learn` are optional installs.** Code that needs them imports lazily; if you see an ImportError there, that's why.
- **DIO channel 1 is the sync channel** by default. There are channels 1–4 in `dsm.dio_paths`.
- **Two cohorts have their own config files** ([config/default_paths.json](config/default_paths.json) = cohort 7, [config/cohort5_paths.json](config/cohort5_paths.json) = cohort 5). The GUI lets you pick between them; the API takes `config_path=` on `DataStorageManager`.
- **`spike_times_by_cell` indexes parallel to `ks_ids`** — `spike_times_by_cell[i]` is the spike-time array for cluster `ks_ids[i]`. Several places build a `cluster_id → index` map (`_cluster_index_map` in [ephys/decoding_plots.py](ephys/decoding_plots.py)); reuse rather than reinventing.
- **Social place field occupancy must be computed over the *target's* `(x, y)`, not the focal animal's.** Speed-gating by target (not focal) is the default and important — when the partner is stationary, occupancy variance dominates the rate map. See [ephys/social_spatial_fields.py](ephys/social_spatial_fields.py) (`compute_social_place_fields`, default `speed_filter_subject='target'`); synthetic test #2 in [tests/test_social_spatial_fields.py](tests/test_social_spatial_fields.py) exists to catch a self/target swap.
- **Tracking↔ephys time conversion lives only in [`resolve_tracking_on_ephys_clock`](video/tracking_import.py)** (a free function taking `VideoTrackingData` + a `DataSyncManager` + `pixels_per_cm` + `animal_ids`). [`MultiAnimalSession.get_tracking_on_ephys_clock`](ingestion/multi_animal_session.py) is a thin wrapper that loads the tracking and delegates to it; single-focal analyses that lack partner ephys (e.g. [ephys/decode_partner_distance.py](ephys/decode_partner_distance.py)) call the free function directly. Anything else that needs both clocks must call it too. It also does px→cm via the optional `pixels_per_cm` config key (null by default → positions stay in pixels, `*_cm` params then mean pixels, single warning logged).
- **NWB files import into the same three dataclasses via [`load_nwb_session`](ingestion/nwb_import.py).** `load_nwb_session(path)` returns a `NwbSession` namedtuple `(ks, events, tracking, sync)` — the standard `KilosortData`/`BehavioralEventsData`/`VideoTrackingData` plus an `IdentitySyncManager` (slope=1, intercept=0). NWB streams already share one clock, so behavioral events come back **pre-synchronized** (`ts_*_ephys` populated, `synchronized=True`) — do **not** call `synchronize_with_ephys` on them. The module is purely additive (no GUI/DSM wiring) and imports `pynwb` lazily. NWB `units` spike times are already in seconds, so they map straight to `spike_times_by_cell`; per-cluster geometry (channel/DV/XX) is best-effort from the `electrodes` table, else zero stubs (the decoders only need `ks_ids` + `spike_times_by_cell`). Behavioral mapping is **generic**: each NWB `TimeIntervals`/`trials` table becomes events with `type`=table name, and identity columns (initiator/victim/winner/loser) pass through *only if present* — DANDI 001749 (the test set, mouse reward-competition) has just `cs_onsets`/`us_deliveries`, so outcome/opponent decoding doesn't apply there but spike-to-event alignment does.
- **A permutation budget that's too small for the number of tests makes a null result meaningless.** BH-FDR multiplies the smallest p-value by the number of tests, and a permutation test floors p at `1/(n_shuffles+1)`. With `n_shuffles=200` across 149 cells the best reachable q is **0.74** — no cell can be significant at any effect size. This bit this project for real: a "0/148 cells significant" result was reported as biology when it was predetermined by the budget. Always check [`fdr_resolution(n_tests, n_shuffles, alpha)`](ephys/_stats_utils.py) first; `compute_population_significance` calls it, warns, and returns the verdict under `significance_resolution`. Escape hatches: `null_mode='pooled'` (≈n_cells better resolution, same compute, assumes a shared null), the single-test `significance_population` p-value, or `recommended_n_shuffles`. The same trap applies to [social place fields](ephys/social_spatial_fields.py) — its default `sig_alpha=0.01` across 3–4 targets needs ≥300 shuffles, and it does **not** yet self-check.
- **`chance_level = 1/n_classes` is not the baseline to beat when classes are imbalanced.** Plain `'accuracy'` (what every `cross_val_score` here uses) must be compared to the *majority-class* rate: on a 12/7 winner/loser split that's 63.2%, not 50%, so a 60.6% "result" is actually below naive guessing. Use `baseline_accuracy` / `population_baseline_accuracy` (from [`majority_class_baseline`](ephys/_stats_utils.py)), or `balanced_accuracy`, whose chance level genuinely is `1/n_classes`. **[ephys/decoding_plots.py](ephys/decoding_plots.py) still draws its chance lines and "Cells > Chance" counts at `1/n_classes`** — deliberately left alone to avoid changing published figures, so read those plots with this in mind.
- **Permutation p-values use the add-one form `(1+k)/(n+1)`, never `k/n`.** A finite permutation test cannot justify `p == 0`; the plain mean form returns exactly 0.0 when no shuffle beats the observed value, which survives BH as `q == 0` and reads as infinite confidence. Both [ephys/_lda_decoding.py](ephys/_lda_decoding.py) and [ephys/social_spatial_fields.py](ephys/social_spatial_fields.py) (`_p_geq`/`_p_leq`) use the add-one form; keep new nulls consistent.
- **All animals in a session share one ephys clock.** Kilosort spike times across animal directories are already comparable in seconds. For multi-animal analyses, bin them on a common ephys-second grid via [`MultiAnimalSession.get_common_binned_rates`](ingestion/multi_animal_session.py) — do **not** attempt cross-animal clock conversion. Behavior↔ephys sync still goes through a single per-session [`DataSyncManager`](ingestion/ephys_sync.py); any animal's DIO + pulse log works because they all share the clock. See [ephys/README.md](ephys/README.md) for the full inter-brain module.

## Where to look first

| If asked about… | Start at |
|---|---|
| Path resolution / "where is X" | [ingestion/data_paths.py](ingestion/data_paths.py) — `DataStorageManager` |
| Spike data shape / loading | [ingestion/kilosort_data_import.py](ingestion/kilosort_data_import.py) — `KilosortData`, `load_kilosort_data` |
| Clock alignment | [ingestion/ephys_sync.py](ingestion/ephys_sync.py) — `DataSyncManager`, `find_sync_mapping` |
| Importing NWB-format datasets | [ingestion/nwb_import.py](ingestion/nwb_import.py) — `load_nwb_session`, `nwb_to_kilosort_data`, `nwb_to_behavioral_events`, `nwb_to_tracking_data`, `IdentitySyncManager` |
| Behavioral events / labels | [video/behavioral_events.py](video/behavioral_events.py) — `extract_opponent_labels`, `extract_outcome_labels`, `extract_group_labels` |
| LDA decoding internals | [ephys/_lda_decoding.py](ephys/_lda_decoding.py) — `single_cell_lda_decode`, `run_population_per_cell_decode`, `run_time_resolved_population_decode` |
| Significance / multiple-comparison correction | [ephys/_lda_decoding.py](ephys/_lda_decoding.py) — `compute_population_significance`; [ephys/_stats_utils.py](ephys/_stats_utils.py) — `benjamini_hochberg`, `fdr_resolution`, `majority_class_baseline` |
| Lab notebook (per-iteration provenance, campaign FDR) | [database/lab_notebook.py](database/lab_notebook.py) — `LabNotebook`, `Hypothesis`, `TestFamily`, `Iteration` |
| AI-in-the-loop discovery platform | [docs/AI_DISCOVERY_LOOP_DESIGN.md](docs/AI_DISCOVERY_LOOP_DESIGN.md), [docs/PHASE0_FINDINGS.md](docs/PHASE0_FINDINGS.md), [HANDOFF.md](HANDOFF.md), [scripts/phase0_probe.py](scripts/phase0_probe.py); skills in `.claude/skills/{run-analysis,interpret-results,propose-hypotheses,implement-module}/`, subagents `.claude/agents/interpreter.md` and `.claude/agents/coder.md` |
| Decoding plots | [ephys/decoding_plots.py](ephys/decoding_plots.py) |
| Result-dict schema | [ephys/_lda_decoding.py](ephys/_lda_decoding.py) returns; wrappers in [ephys/decode_opponent_identity.py](ephys/decode_opponent_identity.py) and [ephys/decode_event_outcome.py](ephys/decode_event_outcome.py) add `parameters` and `behavioral_summary` |
| Streamlit GUI plumbing | [gui/state.py](gui/state.py) (typed `SessionKey`/`AnalysisParams`), [gui/loaders.py](gui/loaders.py), [gui/runners.py](gui/runners.py) (`cached_step`) |
| Adding a new analysis tab | Pattern: write a `render(session_key, params)` in [gui/tabs/](gui/tabs/), wire it into [gui/app.py](gui/app.py) |
| Inter-brain shared subspace (multi-animal CCA, nulls, behavior regression) | [ephys/README.md](ephys/README.md) is the entry point; modules in [ephys/inter_brain_dynamics.py](ephys/inter_brain_dynamics.py), [ephys/inter_brain_plots.py](ephys/inter_brain_plots.py), [ephys/run_inter_brain.py](ephys/run_inter_brain.py), [ingestion/multi_animal_session.py](ingestion/multi_animal_session.py), [video/behavior_features.py](video/behavior_features.py), [gui/tabs/inter_brain.py](gui/tabs/inter_brain.py) |

## Test data note

The user's working directory is on a Janelia Windows workstation; data paths in the configs point at `\\nearline\karpova\...` SMB shares (cohort 5 and cohort 7). Anything I write that depends on real data should be runnable from there. Tests under [tests/](tests/) use mock data and don't need the share.

## Build/test cheatsheet

```bash
# Run the Streamlit GUI (primary entry point)
streamlit run gui/app.py

# Run tests (run_tests.py crashes on Windows cp1252 consoles - emoji in its banner;
# use pytest directly). Heavy permutation tests are marked `slow`.
python -m pytest tests/ -q -m "not slow"     # fast pass
python -m pytest tests/ -q                   # everything

# Run a decoding analysis from the CLI
python -m ephys.decode_opponent_identity --animal_id 631 --session_id 20251216 --use_quality_cells
python -m ephys.decode_event_outcome    --animal_id 631 --session_id 20251216 --use_quality_cells

# Run inter-brain shared-subspace analysis (two animals)
python -m ephys.run_inter_brain --session_id 20251216 --animal_ids 631 632 \
    --bin_size 0.5 --smoothing 0.25 --max_K 20 --n_shuffles 200 \
    --output_dir ./results
```

## What I should NOT do without asking

- Modify or delete anything under `config/` — those paths are environment-specific and shared.
- Delete `.gui_cache/` wholesale; it can hold hours of compute.
- Refactor the result-dict schema in [ephys/_lda_decoding.py](ephys/_lda_decoding.py) — both decoder wrappers and all of [ephys/decoding_plots.py](ephys/decoding_plots.py) depend on it, and there's a pinned `class_label`/`analysis_title` contract that the user just stabilised in the most recent commits.
- Commit changes to `test.ipynb` unless asked — it's an active scratchpad, the diff is usually huge.
- Add backwards-compat shims for the old `KilosortData(data_input=...)` API. The user has already migrated all callers; keep the new dataclass API clean.
