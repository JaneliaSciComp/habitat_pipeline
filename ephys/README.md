# ephys/ — Neural analysis

Analysis modules over the spike-sorted Kilosort 4 outputs. Three groups
of tooling live here:

| Group | Modules | Result-dict pattern |
|---|---|---|
| **LDA decoding** | [_lda_decoding.py](_lda_decoding.py), [decode_opponent_identity.py](decode_opponent_identity.py), [decode_event_outcome.py](decode_event_outcome.py), [decoding_plots.py](decoding_plots.py) | `results['parameters']['class_label']` + `['analysis_title']` drive plot titles. |
| **Location / population** | [decode_location.py](decode_location.py), [population_geometry.py](population_geometry.py), [rastermap_viz.py](rastermap_viz.py) | Stand-alone per-module dicts. |
| **Inter-brain dynamics** ↓ | [inter_brain_dynamics.py](inter_brain_dynamics.py), [inter_brain_plots.py](inter_brain_plots.py), [run_inter_brain.py](run_inter_brain.py) | `SharedSubspaceFit` dataclass with the same `class_label` / `analysis_title` convention. |

Plus QA: [plot_ephys_qa_stats.py](plot_ephys_qa_stats.py).

---

## Inter-brain neural dynamics

For two animals recorded simultaneously in the same arena, find a
low-dimensional **shared subspace** whose dynamics correlate across
brains, plus a **unique subspace** in each brain orthogonal to it.
Quantify variance partitioning, statistical significance against
circular-shift shuffles, regression of shared dimensions onto self vs
partner behavior, and leader/follower temporal lag.

Reference: Zhang, Phi, Li et al., *Inter-brain neural dynamics in
biological and artificial intelligence systems*. **Nature** 645, 991–1001
(2025). DOI: 10.1038/s41586-025-09196-4 — biological data only;
MARL agent analyses are out of scope.

### Key facts about this dataset

- All animals recorded in one session share a **single ephys clock** —
  Kilosort spike times across animal directories are already comparable
  in seconds, so binning onto a shared time grid needs no cross-clock
  arithmetic.
- Behavior↔ephys sync still goes through one
  [`DataSyncManager`](../ingestion/ephys_sync.py) per session; any
  animal's DIO + pulse log works because they all share the clock.
- One [`BehavioralEventsData`](../video/behavioral_events.py) per
  session (events involve multiple animals as initiator/victim/etc.).

### Module map

```
ingestion/multi_animal_session.py    ← multi-animal orchestrator
ephys/inter_brain_dynamics.py        ← CCA + nulls + helpers + regression
ephys/inter_brain_plots.py           ← seven plot functions
ephys/run_inter_brain.py             ← CLI
video/behavior_features.py           ← per-bin behavior feature matrix
gui/tabs/inter_brain.py              ← Streamlit tab
gui/state.py                         ← InterBrainParams dataclass
```

---

### 1. `MultiAnimalSession` — common ephys-second grid

[`ingestion/multi_animal_session.py`](../ingestion/multi_animal_session.py)

```python
session = MultiAnimalSession(
    session_id="20251216",
    animal_ids=["631", "632"],
    config_path=None,                 # cohort 7 default
    dio_channel=1,
    sync_from_animal=None,            # default: animal_ids[0]
)
bin_centers, rates_by_animal = session.get_common_binned_rates(
    bin_size_sec=0.5,
    smoothing_sigma_sec=0.25,
    filter_kwargs={"min_firing_rate": 0.5,
                   "min_presence_ratio": 0.8,
                   "max_cv_isi": 5.0},
    use_cache=True,                   # .gui_cache/multi_animal/
)
# bin_centers: (T,) ephys seconds, identical across animals.
# rates_by_animal: {animal_id: (n_cells, T) firing-rate matrix}
```

- One `DataStorageManager` is built per animal at construction.
  `KilosortData` is loaded lazily via `session.get_ks(animal_id)`.
- `session.sync` and `session.events` are lazy properties — single
  canonical `DataSyncManager` and ephys-synced `BehavioralEventsData`
  for the whole session.
- `session.verify_sync_consistency()` builds a sync from every animal
  and warns (does not raise) if slopes or `r²` disagree with the
  canonical sync.
- `get_common_binned_rates` directly forwards filter kwargs to
  `KilosortData.filter_cells_by_firing_patterns` so quality filtering
  is per-animal; bin edges are guaranteed identical across animals.
- Caches under `.gui_cache/multi_animal/{session_id}_{hash(params)}.pkl`
  keyed on session, sorted animal IDs, bin size, time window, filter
  kwargs, and smoothing.

---

### 2. `SharedSubspaceFit` + `fit_shared_subspace`

[`ephys/inter_brain_dynamics.py`](inter_brain_dynamics.py)

```python
from ephys.inter_brain_dynamics import fit_shared_subspace

fit = fit_shared_subspace(
    X_A,                              # (T, N_A) firing rates
    X_B,                              # (T, N_B) firing rates
    n_components=5,                   # K
    reg=1e-3,                         # ridge for whitening
    method="regularized",             # | "cca" | "pls"
    cv_folds=5,                       # contiguous-block CV (no shuffle)
    animal_ids=("631", "632"),
    t_window=(0.0, 1800.0),
    bin_size_sec=0.5,
    smoothing_sigma_sec=0.25,
)
```

Returns a [`SharedSubspaceFit`](inter_brain_dynamics.py) dataclass:

| Field | Shape | Meaning |
|---|---|---|
| `U_A`, `U_B` | `(N_A, K)`, `(N_B, K)` | Cell-space → shared-subspace projection; `S_A = X_A_z @ U_A`. |
| `S_A`, `S_B` | `(T_valid, K)` | Shared-dim time courses on z-scored data. |
| `V_A_unique`, `V_B_unique` | `(N_A, N_A−K)`, `(N_B, N_B−K)` | Orthonormal bases of each animal's unique subspace. |
| `canonical_correlations` | `dict` | `{train: (K,), cv: (n_folds, K), cv_mean: (K,), cv_std: (K,)}` Pearson correlations between matched shared dims. |
| `variance_partition` | `dict` | `shared_var_A_z`, `unique_var_A_z`, ... (and `_raw` equivalents on mean-centered raw rates); z-scored entries sum to 1 by construction. |
| `parameters` | `dict` | Echoes inputs + `class_label="Shared dim"`, `analysis_title="Inter-brain shared subspace"`. |
| `valid_mask` | `(T_input,)` bool | True for bins kept after NaN drop. |

#### Method choice — `method="regularized"` (default)

Cohort recordings frequently have N comparable to or larger than T
after quality filtering. sklearn's `CCA` is numerically unstable in
that regime; `"regularized"` is the default and uses ridge-whitened
SVD of the cross-covariance:

1. Z-score `X_A`, `X_B` (per cell, on the training fold inside CV).
2. Whiten: `W_A = (C_AA + reg·I)^(−½)`, same for `B`.
3. SVD of `W_A^T C_AB W_B = U Σ V^T`.
4. Loadings `U_A = W_A · U[:, :K]`, `U_B = W_B · V[:, :K]`.

`"cca"` and `"pls"` route through sklearn's `CCA` / `PLSCanonical`
respectively (well-conditioned only when T ≫ N). `"gfa"` raises
`NotImplementedError` (stretch goal).

#### Cross-validation

Contiguous-block time-series CV (`KFold(shuffle=False)`-style). Shuffled
K-fold would leak across folds because firing rates are autocorrelated.
Z-scoring happens on the training portion only, per fold.

#### Variance partition

CCA loadings are **not** column-orthonormal in cell space, so a literal
`U_A U_A^T` is not the orthogonal projector onto `span(U_A)`. The
partition uses `Q_A = qr(U_A).Q` (the QR-orthonormalized loadings) so
that `shared_var = ‖X·Q‖_F² / ‖X‖_F²` and `unique_var = 1 − shared_var`
exactly. Both z-scored and raw (mean-centered) partitions are reported.

#### NaN handling

Bins with any NaN in either `X_A` or `X_B` are dropped before fitting;
`fit.valid_mask` records which input bins were kept.

---

### 3. Shuffle null + K selection

```python
from ephys.inter_brain_dynamics import shuffle_null_subspace, choose_n_components

null = shuffle_null_subspace(
    X_A, X_B,
    n_components=5, n_shuffles=200,
    kind="circular_shift",            # only supported kind today
    seed=0,
)
# null: (n_shuffles, K) train canonical correlations under the null.
```

For each shuffle, an integer shift drawn from `[0.1·T, 0.9·T]` is
applied to `X_B` via `np.roll`, then the fit is repeated. **Never**
permute time bins — that destroys autocorrelation and inflates
significance.

```python
result = choose_n_components(
    X_A, X_B,
    max_K=20, n_shuffles=200, cv_folds=5, seed=0,
)
# result['train_ccs']         (max_K,)
# result['cv_mean']           (max_K,)
# result['shuffle_null']      (n_shuffles, max_K)
# result['shuffle_p95']       (max_K,)
# result['recommended_K']     int — train-CC > p95-null rule (5%/dim FPR)
# result['recommended_K_cv']  int — cv_mean > p95-null rule (conservative)
```

Two K recommendations are surfaced because the train-CC rule
(specified by the paper / prompt) is lax — noise-dim train CCs can
marginally exceed the 95th-pctl null by chance. The CV-mean rule is
sharper because held-out R² ≈ 0 for noise dims.

---

### 4. Time-lagged CCA, cross-animal correlation matrix, helpers

```python
from ephys.inter_brain_dynamics import (
    project_onto_shared,
    time_lagged_cca,
    cross_animal_correlation_matrix,
)

# Trivial: X @ U.  Caller responsible for z-scoring consistently.
S_A_new = project_onto_shared(X_A_z, fit.U_A)

# Sweep integer lags ∈ [-max, +max], pair X_A[t] with X_B[t-lag].
# Positive lag means X_B leads X_A by `lag` bins.
lags, ccs = time_lagged_cca(
    X_A, X_B,
    max_lag_bins=10, n_components=fit.n_components,
)
# lags: (2*max_lag_bins+1,);  ccs: (n_lags, K)

# Full (N_A, N_B) Pearson cross-correlation matrix.
C = cross_animal_correlation_matrix(X_A, X_B)
```

---

### 5. Behavior regression

```python
from ephys.inter_brain_dynamics import regress_shared_on_behavior

result = regress_shared_on_behavior(
    fit,
    behavior_by_animal={
        "631": features_A,            # built with 631 as focal
        "632": features_B,            # built with 632 as focal
    },
    alpha=1.0,
    cv_folds=5,
)
# result["631"][k] = {
#     "R2_self":           float,    # CV-mean R²
#     "R2_partner":        float,
#     "R2_both":           float,
#     "R2_partner_unique": R2_both - R2_self,
#     "R2_self_unique":    R2_both - R2_partner,
# }
# result["632"][k] = {...}
# result["feature_names"] = {animal_id: {self:[...], partner:[...]}}
# result["parameters"]    = {alpha, cv_folds, animal_ids, class_label, analysis_title}
```

For each animal, regresses each shared dim onto **self** features (the
focal-as-this-animal feature matrix), **partner** features (the
focal-as-the-other-animal feature matrix), and **both** (column
concat). R² is CV-mean (no shuffle) with per-fold z-scoring on the
training portion.

The prompt's nominal signature is `(S_A, S_B, behavior_features, alpha)`;
we deviated because a single `behavior_features` cannot distinguish self
from partner features per animal. The function takes the full `fit` and
a dict of per-focal feature DataFrames.

#### Building the feature matrices

[`video/behavior_features.py`](../video/behavior_features.py)

```python
from video.behavior_features import build_behavior_feature_matrix

features_A = build_behavior_feature_matrix(
    tracking=tracking,                # VideoTrackingData
    events=session.events,            # BehavioralEventsData (ephys-synced)
    sync=session.sync,                # DataSyncManager (fallback for timestamps)
    t_grid_ephys=bin_centers,         # from MultiAnimalSession
    focal="631",                      # animal whose behavior is "self"
    partner="632",
    event_window_sec=1.0,             # ± window for event indicators
    event_types=None,                 # default: all unique types in events
)
```

Columns:

| Column | Source |
|---|---|
| `speed` | `‖d(focal_xy)/dt‖` |
| `angular_speed` | `d(unwrap(atan2(vy, vx)))/dt` |
| `head_dir` | only present if focal tracking has a head-direction column |
| `distance` | `‖focal_xy − partner_xy‖` |
| `relative_bearing` | bearing focal→partner, wrapped to `[-π, π]` in focal heading frame |
| `relative_speed` | `d(distance)/dt`, positive = retreating |
| `event_<TYPE>` | 1.0 in bins within `±event_window_sec` of any `ts_start_ephys` of `TYPE` |

Timestamps are converted into ephys seconds once at the top of the call
(`sync.convert_behavior_to_ephys` if tracking is not already synced);
continuous features are linearly resampled onto `t_grid_ephys` with NaN
outside the tracked range.

---

### 6. Plots

[`ephys/inter_brain_plots.py`](inter_brain_plots.py) — seven functions,
all returning `matplotlib.figure.Figure`. Titles and class labels are
pulled from `fit.parameters['analysis_title']` /
`fit.parameters['class_label']`, mirroring the convention in
[`decoding_plots.py`](decoding_plots.py).

```python
from ephys.inter_brain_plots import (
    plot_canonical_correlations,      # train + CV bars + p95 null line
    plot_variance_partition,          # stacked bars per animal
    plot_shared_dimensions,           # overlaid time courses A vs B
    plot_cross_animal_correlation,    # heatmap + hierarchical sort
    plot_time_lagged_cca,             # CC vs lag, peak marked
    plot_shared_vs_behavior,          # R² self/partner/both per K
    plot_inter_brain_summary,         # six-panel dashboard
)
```

The 6-panel summary handles missing optional inputs (cross_corr,
time_lagged, regression_results) by drawing helpful placeholder text
instead of failing.

---

### 7. CLI

[`ephys/run_inter_brain.py`](run_inter_brain.py)

```bash
python -m ephys.run_inter_brain \
    --session_id 20251216 --animal_ids 631 632 \
    --bin_size 0.5 --smoothing 0.25 \
    --max_K 20 --n_shuffles 200 \
    --behavior_type EC --output_dir ./results
```

Loads via `MultiAnimalSession`, builds common binned rates, fits the
subspace, runs the shuffle null + time-lagged CCA + cross-animal
correlation, attempts to load tracking and build per-focal behavior
matrices for the regression (gracefully skips with a warning if
tracking is unavailable), and writes:

```
results/inter_brain_20251216_631_632/
├── results.pkl   # SharedSubspaceFit + shuffle_null + time_lagged
│                 #   + cross_corr + regression_results + bin_centers
└── summary.png   # 6-panel dashboard
```

Internally factored into `_analyze(...)` (pure compute) and
`_analyze_and_save(...)` (pure compute + I/O) so the GUI can call the
former directly.

All CLI flags are documented under `python -m ephys.run_inter_brain
--help`; key ones beyond the prompt's example: `--method`, `--reg`,
`--cv_folds`, `--max_lag_bins`, `--alpha`, `--event_window`, `--t_start`,
`--t_end`, `--skip_regression`, `--seed`, `--config_path`,
`--log_level`.

---

### 8. Streamlit GUI tab

[`gui/tabs/inter_brain.py`](../gui/tabs/inter_brain.py) — a fifth tab in
`gui/app.py`. The focal animal is the one already selected in the
sidebar; the tab adds a multiselect for partner animals (populated from
`get_animals_and_sessions` filtered to the same session), parameter
widgets, and a view picker. The heavy run is wrapped by
[`gui.runners.cached_step`](../gui/runners.py); cache key includes
focal, session, all partners, and every analysis parameter, so any
change invalidates correctly.

`InterBrainParams` lives on [`gui/state.py`](../gui/state.py).

---

### 9. Conventions worth remembering

| | |
|---|---|
| **Shared ephys clock** | All animals' Kilosort spike times in one session are already on a common timeline. Do **not** convert per-animal ephys clocks; use `MultiAnimalSession.get_common_binned_rates`. |
| **Behavior↔ephys sync** | Still goes through one `DataSyncManager` per session (any animal's DIO works). |
| **Z-scoring** | Internal to `fit_shared_subspace` and `_fit_r2`; done on training folds only inside CV. Do not pre-z-score. |
| **CV** | Contiguous-block (no shuffling) everywhere — for CCA, regression, and the K-selection sweep. |
| **Shuffle null** | Circular shifts only. Permuting time bins is a bug. |
| **Variance partition** | QR-orthonormalize loadings before forming the projector. Shared + unique = 1 by construction. |
| **NaN bins** | Dropped at fit time; `fit.valid_mask` records which input bins survived. |
| **Plot titles** | Driven by `fit.parameters['analysis_title']` / `['class_label']` — same convention as the LDA decoders. |

---

### Out of scope (per the prompt)

- MARL agent analyses from Figs. 5–6 of the paper.
- Cell-type / opsin labeling.
- Edits to `_lda_decoding.py`, `decoding_plots.py`, or existing decoder
  result-dict schemas.

### Stretch goals (not implemented)

- Probabilistic CCA / Group Factor Analysis (Klami et al., 2013) with
  ARD-based K selection (`method="gfa"` currently raises).
- Multi-set CCA for three-or-more brains.
- Partner→self decoder (Fig. 3 extension): train a decoder on partner's
  shared dims to predict self behavior, vs self's shared dims as control.
- Causal-ablation analog of Fig. 6: project out the shared subspace
  before feeding spikes into the LDA opponent / outcome decoders, ask
  whether decoding degrades.
