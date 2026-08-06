---
name: run-analysis
description: Run an existing habitat_pipeline analysis (opponent/outcome decoding, inter-brain, social place fields, partner distance) against a session and log the run to the lab notebook. Use when asked to run/execute/decode an analysis for an animal+session, to re-run something for a hypothesis, or to produce a fresh result before interpreting it.
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# run-analysis — Runner role for the discovery loop

Implements the **Runner** role from `docs/AI_DISCOVERY_LOOP_DESIGN.md` §4. Your job is
to run one of the repo's *existing* analyses in-process (not just shell out to a CLI
and throw away the result), and log the real result to the lab notebook
(`database/lab_notebook.py`). You do not write new analysis code here — that's the
deferred `implement-module` skill/Coder role.

## 0. Approval gates — check before running anything

Per §6 of the design doc, get explicit scientist approval before:
- running against a session flagged `held_out=True` in the notebook (query
  `LabNotebook.iterations_for_session(...)` or ask if unsure),
- anything that would touch `config/` or delete `.gui_cache/` (`CLAUDE.md`: don't, without asking).

Running a cached/known analysis against a normal (non-held-out) session needs no
special approval — that's the whole point of a repo-native Runner.

## 1. Pick the right module for the ask

| Ask is about... | Import | Wrapper function |
|---|---|---|
| which opponent was involved | `ephys.decode_opponent_identity` | `decode_opponent_identity_population(...)` |
| who won/lost a fight | `ephys.decode_event_outcome` | `decode_event_outcome_population(...)` |
| shared cross-animal dynamics | `ephys.run_inter_brain` / `ephys.inter_brain_dynamics` | see `ephys/README.md` |
| social place fields / partner tuning | `ephys.social_spatial_fields` | `compute_social_place_fields(...)` |
| distance-to-partner tuning | `ephys.decode_partner_distance` | see module docstring |

For the two decoders above, run the wrapper function **in-process** (import it and
call it directly, e.g. via `python -c "..."` or a short throwaway script) rather than
invoking `main()`/the CLI — you need the real result-dict object to log curated
numbers to the notebook, not just stdout text. The CLIs
(`python -m ephys.decode_opponent_identity ...`) are still useful for a human-facing
smoke check or when you only need the printed summary + saved PNGs.

## 2. Repo gotchas (don't rediscover these — see `CLAUDE.md` for the full list)

- Load order: `load_kilosort_data(dsm.get_kilosort_path())` — passing the
  `DataStorageManager` itself does not work for `KilosortData` (it does for
  `load_tracking_data`).
- **Sync is now automatic** inside both decoder CLIs' `main()` (fixed after Phase 0
  found it missing), but if you call the wrapper functions directly in-process, *you*
  must sync behavioral events yourself first:
  ```python
  from ingestion.data_paths import DataStorageManager
  from ingestion.ephys_sync import DataSyncManager
  from ingestion.kilosort_data_import import load_kilosort_data
  from video.behavioral_events import load_behavioral_events

  dm = DataStorageManager(animal_id, session_id)
  ks_data = load_kilosort_data(dm.get_kilosort_path())
  behavior_data = load_behavioral_events(dm.get_behavioral_event_files(), session_id=dm.session_id)
  sync = DataSyncManager(dm, dio_channel=1)
  behavior_data.synchronize_with_ephys(sync, create_new_columns=True)
  ```
- If a session/behavior_type combination has too few events for a given opponent
  (`min_events_per_class` filters it out), try `label_mode='group'` (pools opponents
  into `'low'`/`'high'` halves) or a different `behavior_type` — check event counts
  first rather than guessing (see `docs/PHASE0_FINDINGS.md`'s `EC`-vs-`F` example).
- `decode_event_outcome_*` defaults `behavior_type=None` (any event with both
  `winner`/`loser` populated); pass `'F'` to restrict to fights.

## 3. Turn on the rigor layer for anything a hypothesis rides on

Both decoder wrappers accept `n_shuffles` (default `0` = off), `alpha` (default
`0.05`), `seed`, and `null_mode` (`'per_cell'` default / `'pooled'`). Leave
`n_shuffles=0` for quick exploratory/cache-friendly runs. When a claim rides on it,
turn it on — it populates three keys (see
`ephys/_lda_decoding.py::compute_population_significance`):

| key | what it is |
|---|---|
| `significance` | `{cluster_id: {p_value, q_value, significant, n_shuffles}}`, BH-FDR across cells |
| `significance_population` | one well-resolved test of population mean accuracy vs. its null — needs no FDR |
| `significance_resolution` | whether the per-cell screen *could* detect anything at this budget |

### Pick `n_shuffles` from the math, not by habit

**Do not just default to 200.** BH multiplies the smallest p-value by the number of
cells, and a permutation test floors p at `1/(n_shuffles+1)`. Across 149 cells,
`n_shuffles=200` gives a best-achievable q of **0.74** — no cell can ever be
significant, so a null result means nothing. Check first:

```python
from ephys._stats_utils import fdr_resolution
fdr_resolution(n_tests=149, n_shuffles=200, alpha=0.05)
# -> {'resolvable': False, 'recommended_n_shuffles': 2980, 'min_tests_at_floor': 15, ...}
```

Three ways to get a real answer, cheapest first:
1. **`null_mode='pooled'`** — ranks each cell against all cells' nulls pooled
   (`n_cells * n_shuffles` draws), so the p-floor drops by ~the cell count at
   *identical* compute. Assumes cells share a null; good for a screen, slightly
   anti-conservative for atypical (very low firing rate) cells.
2. **Lean on `significance_population`** — a single test, well resolved at 200
   shuffles. With few events (tens) this is the only claim the data supports.
3. **Raise `n_shuffles`** to `recommended_n_shuffles` — correct but expensive
   (~2980 shuffles on 149 cells is roughly an hour and a half).

The function warns loudly when a run is under-resolved; don't ignore it, and never
report "0 significant cells" from an under-resolved run as a biological result.

Runtime: roughly linear in `n_cells * n_shuffles`. ~150 cells × 200 shuffles took
~370 s on this workstation, so plan for minutes, not seconds — don't block on it in
the same turn if you can do something else first.

## 4. Log the run to the lab notebook

```python
from database.lab_notebook import LabNotebook

nb = LabNotebook()  # defaults to ./habitat_pipeline.db
params = {
    'animal_of_interest': animal_id, 'behavior_type': behavior_type,
    'time_window': time_window, 'time_bin_size': time_bin_size,
    'cv_folds': cv_folds, 'n_shuffles': n_shuffles, 'alpha': alpha,
    'null_mode': null_mode,
}
population = results.get('significance_population') or {}
resolution = results.get('significance_resolution') or {}
result_summary = {
    'status': results['status'],
    'n_successful_cells': results.get('n_successful_cells'),
    'n_total_cells': results.get('n_total_cells'),
    'population_accuracy_mean': results.get('population_accuracy_mean'),
    'population_accuracy_std': results.get('population_accuracy_std'),
    # The honest comparison point — NOT 1/n_classes.
    'baseline_accuracy': results.get('population_baseline_accuracy'),
    'balanced_accuracy': results.get('population_balanced_accuracy_mean'),
    'best_cell_accuracy': results.get('best_cell_accuracy'),
    'best_cell_id': results.get('best_cell_id'),
    'unique_classes': results.get('behavioral_summary', {}).get('unique_classes'),
    'n_events': results.get('behavioral_summary', {}).get('n_events'),
    'n_significant_cells': sum(v['significant'] for v in results['significance'].values())
                            if results.get('significance') else None,
    # `p_value` is the key LabNotebook.recompute_family_significance reads for
    # campaign-level FDR — always include it when the rigor layer ran.
    'p_value': population.get('p_value'),
    'population_null_mean': population.get('null_mean'),
    'fdr_resolvable': resolution.get('resolvable'),
    'recommended_n_shuffles': resolution.get('recommended_n_shuffles'),
}
iteration = nb.log_iteration(
    'ephys.decode_opponent_identity', params, result_summary,
    animal_id=animal_id, session_id=session_id,
    hypothesis_id=hypothesis_id,      # None if this is exploratory, not hypothesis-testing
    test_family_id=test_family_id,    # None unless you're accumulating a campaign
    figure_paths=saved_png_paths,     # if you saved plots
)
```

Log a **curated summary**, not the raw `cell_results` dict (hundreds of per-cell
entries) — `log_iteration`'s `_json_safe` handles numpy types in whatever you pass,
but keep it to the scalars a scientist or the `interpret-results` skill actually needs.

If this run is part of an ongoing campaign (multiple hypotheses/analyses against one
session), create or reuse a `TestFamily` (`nb.create_test_family(...)`) and pass its
id — that's what lets `nb.recompute_family_significance(...)` correct across the
whole campaign later, on top of the within-run per-cell correction from step 3.

## 5. Hand off

After logging, tell the scientist what you found in one or two sentences (numbers,
not a data dump) and, if this came out of `interpret-results`/a hypothesis, point at
the `Iteration.id` so they can `record_decision(...)` on it.
