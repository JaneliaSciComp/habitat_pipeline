# Claude Code task: Inter-brain neural dynamics module for habitat_pipeline

## Context

We have simultaneous Kilosort 4 recordings from multiple rats in the same arena (RatCity, Tervo lab, Janelia). The pipeline currently treats each session as one focal animal (`DataStorageManager(animal_id, session_id, ...)`) and does event-aligned LDA decoding + PCA/UMAP population geometry per animal. We want to add a new analysis module that implements the **shared / unique neural subspace** decomposition from:

> Zhang, Phi, Li et al., **"Inter-brain neural dynamics in biological and artificial intelligence systems."** *Nature* **645**, 991–1001 (2025). DOI: 10.1038/s41586-025-09196-4. Companion code: https://github.com/hongw-lab/code_for_2024_zhang-phi.

In one line: for two simultaneously recorded animals A and B, find a low-dimensional **shared subspace** whose dynamics correlate across brains (via canonical correlation), and a **unique subspace** in each brain orthogonal to it; quantify variance partitioning, statistical significance against shuffles, and regression of shared dimensions onto self- and partner-behavior.

Read `CLAUDE.md`, `README.md`, and `ARCHITECTURE.md` first. Conform to the existing data plane and result-dict conventions before writing code.

## What to build

### 1. Multi-animal session loader — extend `ingestion/`

**Important fact about this dataset:** all animals recorded in the same session share a single ephys clock. Kilosort spike times across animal directories are already on a common timeline. So there is **no cross-animal clock conversion** to do. Behavior↔ephys sync is still needed once per session (for the behavior-regression step in §2/§5), but only one `DataSyncManager` is needed — any animal's DIO will do, since they all share the clock.

The current `DataStorageManager` is per-animal. Add multi-animal support without breaking the existing API.

- **New file: `ingestion/multi_animal_session.py`**
  - `MultiAnimalSession(session_id, animal_ids: list[str], config_path=None, dio_channel=1, sync_from_animal: str | None = None)` — a thin orchestrator that holds:
    - `dsm_by_animal: dict[str, DataStorageManager]` (one per animal)
    - `ks_by_animal: dict[str, KilosortData]` (lazy, cached)
    - one shared `BehavioralEventsData` and one shared `DataSyncManager` for the session. `sync_from_animal` selects which animal's DIO + pulse-log to use; defaults to `animal_ids[0]`. Assert that constructing a sync from any other animal in the session yields the same slope/intercept within tolerance (`r_squared > 0.999`, `|slope - 1| < 1e-4` on a sanity-check pass; emit a warning if not).
  - `get_common_binned_rates(bin_size_sec, t_start_ephys=None, t_end_ephys=None, filter_kwargs=None, smoothing_sigma_sec=None) -> tuple[np.ndarray, dict[str, np.ndarray]]` — returns `(bin_centers_ephys, {animal_id: rates})`, where each `rates` array is shape `(n_cells_animal, n_bins)` and **all animals share identical bin edges in ephys seconds**. Use the existing `KilosortData.bin_spike_times` directly; no time conversion needed. If `smoothing_sigma_sec` is set, apply 1-D Gaussian smoothing over time per cell after binning.
  - `get_common_time_window()` — returns `(t_start, t_end)` in **ephys seconds**, taking the intersection of all animals' `KilosortData.duration_seconds`.
- **No new config**: reuse `default_paths.json` / `cohort5_paths.json`. The constructor builds one DSM per animal_id and verifies they share `session_id`.
- **Cache**: pickle the binned-rates output under `.gui_cache/multi_animal/{session_id}_{sorted(animal_ids)}_{bin_size}.pkl`, keyed on `(session_id, animal_ids, bin_size, smoothing, filter_kwargs, t_window)`. Reuse `gui/cache.py` helpers if they fit; otherwise write a tiny local cache helper.

### 2. Inter-brain analysis module — new under `ephys/`

- **New file: `ephys/inter_brain_dynamics.py`**

Core API (label-agnostic, mirrors `ephys/_lda_decoding.py`'s style):

```python
def fit_shared_subspace(
    X_A: np.ndarray,  # (T, N_A) z-scored firing rates, animal A
    X_B: np.ndarray,  # (T, N_B) z-scored firing rates, animal B
    n_components: int | None = None,        # if None, choose via CV correlation drop
    reg: float = 1e-3,                       # ridge for CCA whitening
    method: str = "cca",                     # "cca" | "pls" | "gfa"
    cv_folds: int = 5,
) -> SharedSubspaceFit
```

`SharedSubspaceFit` is a dataclass containing:
- `U_A, U_B`: loadings (N_A × K) and (N_B × K) for the shared subspace in each animal
- `canonical_correlations`: length-K vector of train and per-fold-CV correlations
- `S_A, S_B`: shared-dim time courses (T × K) for each animal
- `V_A_unique, V_B_unique`: orthonormal bases of the unique subspace in each animal (N_A × (N_A-K) etc.), obtained by projecting out `U_A` and re-orthonormalizing the residual PCA basis
- `variance_partition`: dict with `shared_var_A`, `shared_var_B`, `unique_var_A`, `unique_var_B`, each as a fraction of total per-animal variance, plus equivalent partitions on z-scored vs raw rates
- `parameters`: dict echoing all inputs (n_components, reg, method, bin_size_sec, smoothing_sigma_sec, t_window, animal_ids)
- `analysis_title` and `class_label` mirror the convention from the LDA decoders so plotting can reuse it

Other functions:

```python
def choose_n_components(X_A, X_B, max_K=20, reg=1e-3, n_shuffles=200, cv_folds=5)
    # Returns: per-K mean CV canonical correlation, per-K shuffle null (circular shift on X_B),
    # and a recommended K (largest K where train-CC exceeds 95th-percentile shuffle null).

def shuffle_null_subspace(X_A, X_B, n_shuffles=200, n_components=K, kind="circular_shift", seed=0)
    # Returns canonical_correlations under circular time shifts of X_B (preserves autocorr).

def project_onto_shared(X, U)
    # Trivial helper for downstream regression.

def regress_shared_on_behavior(
    S_A: np.ndarray, S_B: np.ndarray, behavior_features: pd.DataFrame, alpha=1.0
)
    # Ridge regression of each shared dim on self-behavior, partner-behavior, and both;
    # report R^2 for {self only, partner only, both, both-minus-self, both-minus-partner}.
    # Returns dict per animal.

def time_lagged_cca(X_A, X_B, max_lag_bins, n_components=K)
    # Sweep integer lags; returns canonical correlations vs lag. Used to detect leader-follower.

def cross_animal_correlation_matrix(X_A, X_B)
    # Full N_A × N_B Pearson correlation matrix for diagnostic plots.
```

Implementation notes:
- Use `sklearn.cross_decomposition.CCA` for the baseline; offer a regularized variant via numpy (whitening with ridge then SVD of the cross-covariance) because CCA's vanilla version is unstable when N > T.
- The shared subspace is defined on the firing-rate **time series**, not event-aligned trials — this matches the paper, which works on continuous imaging frames. Default `bin_size_sec=0.5`; allow override.
- All inputs are pre-z-scored per cell. Z-score on the training fold only inside the CV loop.
- Variance partition: variance in animal A explained by the shared subspace = `||U_A U_A^T X_A^T||_F^2 / ||X_A^T||_F^2`, using the projection matrix in the standardized space.
- The "shuffle null" must preserve within-animal autocorrelation. Use circular shifts (random shift drawn from `[0.1*T, 0.9*T]`) on `X_B` relative to `X_A`. Do **not** simple-permute time bins — that destroys autocorr and inflates p-values.

### 3. Plots — new under `ephys/`

- **New file: `ephys/inter_brain_plots.py`**, with `SharedSubspaceFit` as the input contract. Functions:

  - `plot_canonical_correlations(fit, shuffle_null=None)` — bar chart of per-K canonical correlation, with shuffle null band.
  - `plot_variance_partition(fit)` — stacked bars per animal for shared vs unique variance.
  - `plot_shared_dimensions(fit, t_bins, k_dims=(0,1,2))` — overlaid time courses of shared dims for animal A vs B; correlation annotated.
  - `plot_cross_animal_correlation(corr_matrix)` — heatmap of cell-pair correlations sorted by hierarchical clustering.
  - `plot_time_lagged_cca(lags, ccs)` — leader-follower lag profile.
  - `plot_shared_vs_behavior(fit, regression_results)` — bar chart of R² self vs partner vs joint, per shared dim.
  - `plot_inter_brain_summary(fit, ...)` — 6-panel dashboard combining the above (like `plot_decoding_summary` in `ephys/decoding_plots.py`).

Match the title/axis-label pattern from `ephys/decoding_plots.py` (titles driven by `fit.parameters['analysis_title']`).

### 4. CLI entry — new

- **New file: `ephys/run_inter_brain.py`** with `main()` and `argparse`:

```bash
python -m ephys.run_inter_brain \
    --session_id 20251216 --animal_ids 631 632 \
    --bin_size 0.5 --smoothing 0.25 \
    --max_K 20 --n_shuffles 200 \
    --behavior_type EC --output_dir ./results
```

Behaviors:
- Loads via `MultiAnimalSession`, builds common binned rates, fits the subspace, runs the shuffle null, runs the behavior regression (using `BehavioralEventsData` + tracking-derived features — see §5), saves a results pickle, and writes the six-panel summary PNG.
- Mirror the CLI structure in `ephys/decode_opponent_identity.py` (argument parsing, output dir layout, logging).

### 5. Behavior feature builder

The regression in §2 (`regress_shared_on_behavior`) needs continuous per-bin behavior features. Add:

- **New file: `video/behavior_features.py`**
  - `build_behavior_feature_matrix(tracking: VideoTrackingData, events: BehavioralEventsData, sync: DataSyncManager, t_grid_ephys: np.ndarray, focal: str, partner: str) -> pd.DataFrame`
  - Tracking timestamps and event timestamps live in behavior seconds; convert them into ephys seconds via `sync.convert_behavior_to_ephys` once at the top, then resample onto `t_grid_ephys` so the behavior features line up bin-for-bin with the firing-rate matrices from §1.
  - Columns: focal speed, focal angular speed, focal head direction (if available from pose), focal-partner distance, focal-partner relative bearing, focal-partner relative speed, indicator columns for each event type in a ±1 s window around event timestamps.
  - This is the "self vs other behavior" input used to ask whether shared dims are driven by self, other, or both — directly matches Fig. 4 of the paper.

### 6. GUI integration

- **New tab: `gui/tabs/inter_brain.py`** that:
  - Adds a multi-animal selector (multiselect over `get_animals_and_sessions(config_path=...)` filtered to the chosen session_id).
  - Calls `fit_shared_subspace` via `cached_step()` (see `gui/runners.py`).
  - Renders the six-panel summary plus a slider for K.
- Wire it into `gui/app.py` alongside the existing Tracking / Behavioral / Decoding / Population tabs.
- Add typed params to `gui/state.py` (e.g., `InterBrainParams` dataclass with `animal_ids: tuple[str, ...]`, `bin_size`, `smoothing`, `K`, `n_shuffles`, `t_window`).
- Disk cache key includes all of those so changing any one invalidates.

### 7. Tests — new under `tests/`

- **New file: `tests/test_inter_brain_dynamics.py`**
  - Synthetic test: build two N=50, T=2000 datasets that share K=3 latent factors plus independent noise; assert `fit_shared_subspace` recovers cosine similarity > 0.9 with the planted shared directions, and that shuffle-null correlations are clearly below the true ones.
  - Edge cases: K=0 (no shared structure → all canonical correlations should be near shuffle null), N_A ≠ N_B, T < N (regularization required), missing data (NaNs in rates).
  - `MultiAnimalSession.get_common_binned_rates`: assert that with synthetic spike-time lists, both animals' rate matrices have identical bin edges, the bin centers are returned in ephys seconds, and the time window equals the intersection of the inputs' durations.
- Update `tests/run_tests.py` to include the new file.

### 8. Documentation

- Add a section to `README.md` under "ephys" describing the new module, the CLI invocation, and a 6-line example.
- Add a row to `ARCHITECTURE.md`'s Capabilities table linking to `ephys/inter_brain_dynamics.py`.
- Append a "Gotchas" entry to `CLAUDE.md`: all animals in a session share one ephys clock — bin spike times directly on a common ephys-second grid via `MultiAnimalSession.get_common_binned_rates`; do not attempt cross-animal clock conversion. Behavior↔ephys sync still goes through a single per-session `DataSyncManager`.
- Notebook stub: `ephys/inter_brain_demo.ipynb` that walks through one session end-to-end (load → fit → variance partition → shuffle null → behavior regression → plots).

## Acceptance criteria

1. `python -m ephys.run_inter_brain --session_id 20251216 --animal_ids 631 632 --bin_size 0.5 --max_K 10 --n_shuffles 100 --output_dir ./results` runs end-to-end on real data, writes a pickle and a PNG, exits 0.
2. Synthetic test recovers planted shared subspace; shuffle null is below true canonical correlations by ≥3 SD.
3. `from ephys.inter_brain_dynamics import fit_shared_subspace, shuffle_null_subspace, regress_shared_on_behavior` and `from ephys.inter_brain_plots import plot_inter_brain_summary` work from a fresh Python session in the conda env.
4. Streamlit GUI shows an Inter-Brain tab, the multi-animal selector populates from the session's available rats, and the summary plot updates when K changes (using cached fits).
5. `MultiAnimalSession.get_common_binned_rates` returns identical bin edges across animals, in ephys seconds, with no cross-clock arithmetic anywhere in the code path — verified by grep for `convert_ephys` / `convert_behavior` outside of `ingestion/`.
6. No regressions: existing pytest suite passes (`cd tests && python run_tests.py`).
7. Result-dict / dataclass conventions match `_lda_decoding.py`'s schema: `parameters` carries `class_label` and `analysis_title` so plots can be driven from the dataclass without code branches.

## Gotchas (from CLAUDE.md, restated because they will bite)

- **All animals in a session share a single ephys clock.** Spike times are already comparable across Kilosort directories. Bin them on a common ephys-second grid; do not attempt cross-animal clock conversion. Only one `DataSyncManager` per session is needed for behavior↔ephys.
- `KilosortData` is a dataclass; entry point is `load_kilosort_data(path)` — do not pass a DSM to it.
- Behavioral CSV timestamps are Linux nanoseconds; `synchronize_with_ephys` divides by 1e9. Don't double-convert.
- `BehavioralEventsData` is session-level (initiator/victim/winner/loser), shared across all animals. Don't build one per animal.
- DIO channel 1 is the sync channel by default. There are channels 1–4 in `dsm.dio_paths`.
- Two cohorts have separate configs — read `config_path` through to all loaders.
- Don't touch `config/`, don't refactor the existing decoder result-dict schema, and don't commit `test.ipynb`.
- Do not add backwards-compat shims for the old `KilosortData(data_input=...)` API.

## Out of scope (do NOT do these)

- Multi-animal **MARL agent** analyses from Figs. 5–6 of the paper. Just biological data.
- Cell-type / opsin labeling — RatCity ephys does not have molecular labels.
- Any modification to `ephys/_lda_decoding.py`, `ephys/decoding_plots.py`, or the existing decoder result-dict schemas.

## Stretch goals (only after the above is green and tested)

- Replace plain CCA with **probabilistic CCA / Group Factor Analysis** (Klami et al., 2013) for principled K selection via ARD.
- Three-or-more-brain extension via multi-set CCA (sum-correlation criterion).
- Cross-validated **shared dynamics decoder**: train a decoder that uses *partner's* neural shared dims to decode self behavior, vs self's shared dims as control. This is the natural extension of Fig. 3.
- **Causal ablation analog of Fig. 6**: in tied-cohort sessions, project out the shared subspace from one animal's spike count vectors before feeding them to the existing LDA decoders, and ask whether decoding of opponent identity or outcome degrades.

## Build order

1. Multi-animal loader (§1) → write tests for the common-grid binner first.
2. Core CCA / shared-subspace fit (§2) with synthetic test.
3. Shuffle null + variance partition.
4. Behavior feature builder (§5).
5. Behavior regression on shared dims.
6. Plots (§3) and CLI (§4).
7. GUI tab (§6).
8. Docs + notebook + run on a real session.

Commit after each numbered step. Keep commits small and use the existing commit style (look at recent commits in `git log`).
