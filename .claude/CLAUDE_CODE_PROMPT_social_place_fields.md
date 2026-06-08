# Claude Code task: Allocentric social place fields for habitat_pipeline

## Context

We have simultaneous Kilosort 4 recordings from several rats in a shared large arena (RatCity, Tervo lab, Janelia), with allocentric `(x, y)` tracking for every animal. We already have:

- `decode_location.py` — Bayesian decoding of the focal animal's own `(x, y)` from its own spikes.
- `MultiAnimalSession` (under `ingestion/multi_animal_session.py`, recently added) — loads all animals from one session on the shared ephys clock.
- `VideoTrackingData` (`video/tracking_import.py`) — per-animal position dataframes.

We want to add a new module that computes **allocentric "social place fields"**: for each sorted unit in a focal animal's brain, build occupancy-normalized firing-rate maps as a function of *another* (or *each other*) animal's allocentric `(x, y)`. Quantify which cells are tuned to self position vs to one or more conspecifics, compare fields across partners (broadcast vs partner-specific), and provide proper shuffle-based significance.

Read `CLAUDE.md`, `README.md`, and `ARCHITECTURE.md` first. Conform to the existing data plane, result-dict conventions, and the `MultiAnimalSession` API before writing code.

Relevant prior literature this design is informed by (not to replicate verbatim — we want the multi-rat, multi-target, complex-arena version that hasn't been done):

- Ray et al., *Science* 2025 ([10.1126/science.adk9385](https://doi.org/10.1126/science.adk9385)) — fruit-bat CA1 multi-conspecific identity + position coding.
- Zhang et al., *Nat. Commun.* 2024 ([10.1038/s41467-024-47453-8](https://doi.org/10.1038/s41467-024-47453-8)) — mouse CA1 social-vector cells (egocentric).
- Danjo et al., *Science* 2018; Omer et al., *Science* 2018 — original social-place-cell findings.
- Skaggs et al., *NIPS* 1993 — spatial information measure (bits/spike).

## What to build

### 1. Core module — `ephys/social_spatial_fields.py`

**Public dataclasses:**

```python
@dataclass
class RateMap:
    rates: np.ndarray            # (n_y_bins, n_x_bins), Hz, NaN where occupancy < min_occupancy_sec
    occupancy: np.ndarray         # (n_y_bins, n_x_bins), seconds
    spike_counts: np.ndarray      # (n_y_bins, n_x_bins), counts
    x_edges: np.ndarray
    y_edges: np.ndarray
    focal_animal: str             # animal whose spikes generated this map
    target_animal: str            # animal whose (x, y) the map is over (== focal_animal for self-maps)
    cluster_id: int
    parameters: dict              # bin_size_cm, smoothing_sigma_cm, speed_threshold_cms,
                                  # arena_bounds, min_occupancy_sec, t_window_ephys,
                                  # speed_filter_subject ('focal'|'target'|'none'),
                                  # plus class_label='target_position' and analysis_title for plotting

@dataclass
class FieldStats:
    cluster_id: int
    target_animal: str
    skaggs_bits_per_spike: float
    skaggs_bits_per_sec: float
    sparsity: float
    coherence: float
    split_half_corr: float
    peak_rate_hz: float
    mean_rate_hz: float
    n_spikes_in_window: int

@dataclass
class FieldSignificance:
    cluster_id: int
    target_animal: str
    null_method: str              # 'circular_shift' | 'position_shuffle'
    n_shuffles: int
    p_skaggs: float
    p_sparsity: float
    p_split_half: float
    shuffle_skaggs: np.ndarray    # length n_shuffles, kept for plotting
```

**Public functions (label-agnostic, mirrors `_lda_decoding.py` style):**

```python
def compute_rate_map(
    spike_times: np.ndarray,           # ephys seconds
    target_xy: pd.DataFrame,            # columns: t (ephys seconds), x, y (cm)
    bin_size_cm: float = 5.0,
    arena_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    smoothing_sigma_cm: float | None = 5.0,
    min_occupancy_sec: float = 0.1,
    speed_xy: pd.DataFrame | None = None,    # columns: t, speed (cm/s) for the subject we want to gate on
    speed_threshold_cms: float | None = 5.0,
    t_window_ephys: tuple[float, float] | None = None,
    focal_animal: str = "",
    target_animal: str = "",
    cluster_id: int = -1,
) -> RateMap
```

- Bin spikes and target positions on identical time edges (use `pd.cut` over target_xy.t, take per-bin x/y means).
- Occupancy is the per-spatial-bin sum of dwell time (frame interval in seconds).
- Smoothing: 2-D Gaussian on **both** occupancy and spike counts before division (standard; avoids spurious peaks at low-occupancy bins).
- Bins with occupancy below `min_occupancy_sec` become NaN in `rates`.
- `speed_xy` and `speed_threshold_cms` filter time samples *before* binning. **Default subject for the speed gate is the target animal** — when the target is stationary, very little occupancy variance is informative. Make the subject configurable via a parameter `speed_filter_subject: Literal['focal', 'target', 'none']`.

```python
def spatial_information(rate_map: RateMap) -> tuple[float, float]
    # Returns (bits_per_spike, bits_per_second) using the Skaggs formula with occupancy weighting.

def spatial_sparsity(rate_map: RateMap) -> float
    # (<r>)^2 / <r^2> with occupancy weighting; lower = more spatially selective.

def spatial_coherence(rate_map: RateMap) -> float
    # Pearson correlation between each bin's smoothed rate and the mean of its 8-neighborhood, Fisher-z'd then averaged.

def split_half_stability(
    spike_times, target_xy, **rate_map_kwargs
) -> float
    # Split the t_window in half (first vs second), compute two rate maps, return Pearson correlation
    # over bins where both halves have valid occupancy.

def field_significance(
    spike_times, target_xy,
    n_shuffles: int = 500,
    null_method: Literal['circular_shift', 'position_shuffle'] = 'circular_shift',
    seed: int = 0,
    **rate_map_kwargs,
) -> FieldSignificance
    # circular_shift: rigid time-shift of spike train (preserves rate + autocorr); shift drawn from
    #                 [0.1 * T, 0.9 * T] of t_window.
    # position_shuffle: cyclic shift of target_xy.t relative to spike_times by the same draw rule.
    # Compute Skaggs/sparsity/split-half for each shuffle; p-value = fraction with shuffle stat >= true stat.
```

**Multi-target sweeps (the part that's new):**

```python
def compute_social_place_fields(
    ks: KilosortData,
    mas: 'MultiAnimalSession',                       # already-loaded session
    focal_animal: str,
    target_animals: list[str] | None = None,         # default = all animals in mas
    bin_size_cm: float = 5.0,
    smoothing_sigma_cm: float = 5.0,
    speed_threshold_cms: float = 5.0,
    speed_filter_subject: str = 'target',
    n_shuffles: int = 500,
    min_n_spikes: int = 50,
    use_quality_cells: bool = True,
    quality_thresholds: dict | None = None,          # falls through to ks.filter_cells_by_firing_patterns
    t_window_ephys: tuple[float, float] | None = None,
    arena_bounds: tuple | None = None,               # if None, infer from mas tracking with a small pad
) -> SocialFieldResults
```

```python
@dataclass
class SocialFieldResults:
    rate_maps: dict[str, dict[int, RateMap]]         # target_animal -> cluster_id -> RateMap
    stats:    dict[str, dict[int, FieldStats]]
    signif:   dict[str, dict[int, FieldSignificance]]
    cell_classification: pd.DataFrame                 # see below
    population_field_similarity: dict[str, dict[str, np.ndarray]]
        # target_pair -> {'similarity_matrix': (n_cells, n_cells) Pearson over rate maps,
        #                 'diag_distribution': per-cell self-pair correlation}
    parameters: dict                                  # carries class_label='target_position' and analysis_title
```

`cell_classification` columns:

- `cluster_id`
- `n_target_significant` — number of targets the cell is significantly tuned to (p < 0.01 on Skaggs bits/spike, after Benjamini–Hochberg FDR across targets)
- `category` ∈ {`self_only`, `partner_only`, `conjunctive` (self + ≥1 partner), `broadcast` (≥2 partners), `none`}
- `dominant_target` — argmax Skaggs across targets, or NaN if none significant
- per-target `bits_per_spike_<target>`, `sparsity_<target>`, `split_half_<target>`, `p_value_<target>`

Two helpers exported for downstream notebooks:

```python
def field_similarity_across_targets(rate_maps: dict[str, RateMap]) -> pd.DataFrame
    # For one cluster, the Pearson correlation between its rate map under each pair of targets,
    # computed only over bins where both maps have valid occupancy.

def compute_arena_bounds_from_tracking(mas: 'MultiAnimalSession', pad_cm: float = 5.0) -> tuple
    # Aggregate min/max x and y across all animals' tracking, pad, return ((xmin, xmax), (ymin, ymax)).
```

### 2. Tracking → ephys-clock helper — extend `ingestion/multi_animal_session.py`

The rate-map machinery needs each animal's `(t_ephys, x, y, speed)` on the ephys clock. Tracking timestamps live in behavior time (Linux ns / 1e9 → seconds), so they have to go through the session-level `DataSyncManager`.

Add to `MultiAnimalSession`:

```python
def get_tracking_on_ephys_clock(
    self,
    t_start_ephys: float | None = None,
    t_end_ephys: float | None = None,
) -> dict[str, pd.DataFrame]
    # Returns {animal_id: DataFrame with columns t (ephys seconds), x, y, speed}.
    # 'speed' is computed as the gradient of (x, y) over t with a small Gaussian smoothing (sigma ~ 100 ms).
    # Drops rows outside the window; both x and y are in cm. If tracking is in pixels, convert here using
    # a single per-session pixels-per-cm value read from config (or warn loudly and pass through if unset).
```

This is the *only* place tracking↔ephys time conversion happens. Everything downstream takes `t` in ephys seconds.

### 3. Plots — new file `ephys/social_spatial_plots.py`

Driven by `SocialFieldResults.parameters['analysis_title']` and `['class_label']`, mirroring the convention from `ephys/decoding_plots.py`.

- `plot_rate_maps_grid(results, cluster_id)` — one panel per target animal, side by side. Color bar shared; NaN bins masked light gray; cluster ID and per-target Skaggs/p in subtitle.
- `plot_field_similarity_grid(results, cluster_id)` — heatmap of pairwise rate-map correlations across targets for a single cell.
- `plot_cell_classification_summary(results)` — stacked bar of category counts, plus scatter of `bits_per_spike_self` vs `max_bits_per_spike_partner` colored by category.
- `plot_population_field_similarity(results, target_pair)` — `(n_cells, n_cells)` heatmap of rate-map correlations between cells under two targets, sorted by hierarchical clustering of the self target.
- `plot_skaggs_vs_shuffle(results, target_animal, top_k=20)` — for top-K cells, true Skaggs vs shuffle null distribution.
- `plot_field_stability(results)` — split-half stability per target as violins; horizontal stripes at canonical thresholds (0.3, 0.5).
- `plot_social_place_summary(results, ...)` — 6-panel dashboard combining the above (analog of `plot_decoding_summary`).

### 4. CLI — new file `ephys/run_social_spatial.py`

```bash
python -m ephys.run_social_spatial \
    --session_id 20251216 --animal_ids 631 632 633 \
    --focal 631 \
    --bin_size 5 --smoothing 5 --speed_threshold 5 \
    --speed_filter_subject target \
    --n_shuffles 500 \
    --output_dir ./results
```

Same shape as `ephys/decode_opponent_identity.py`: load via `MultiAnimalSession`, build per-animal tracking on ephys clock via `get_tracking_on_ephys_clock`, call `compute_social_place_fields`, save a `SocialFieldResults` pickle, and emit the six-panel summary PNG plus a multi-page PDF with the per-cluster grid plots for the top-N cells by max Skaggs across targets.

### 5. GUI integration — new tab `gui/tabs/social_spatial.py`

- Multi-animal selector inside the chosen session (reuses the session-level animal listing that the inter-brain tab also uses).
- Focal animal radio button, target animals multiselect (default all).
- Sliders: `bin_size_cm`, `smoothing_sigma_cm`, `speed_threshold_cms`; radio for `speed_filter_subject`.
- Cell selector listing clusters ranked by `max_bits_per_spike across targets`.
- For the selected cluster: render the grid plot (one panel per target).
- Aggregate panel: cell-classification bar + Skaggs scatter.
- Disk-cached via `cached_step()` (see `gui/runners.py`); cache key includes all params, focal, target list, and session.

Add typed params to `gui/state.py`:

```python
@dataclass(frozen=True)
class SocialSpatialParams:
    focal: str
    targets: tuple[str, ...]
    bin_size_cm: float
    smoothing_sigma_cm: float
    speed_threshold_cms: float
    speed_filter_subject: Literal['focal', 'target', 'none']
    n_shuffles: int
    use_quality_cells: bool
```

Wire into `gui/app.py` next to the inter-brain tab.

### 6. Tests — `tests/test_social_spatial_fields.py`

- **Synthetic 1 — planted partner field**: simulate two animals' positions as smooth random walks in a known arena; sample spikes from a Poisson process whose rate is a Gaussian bump centered at a chosen `(x*, y*)` on the *partner's* coordinates (not self's). Assert that `compute_rate_map` recovers the bump location within 1 bin and the shuffle p-value < 0.001.
- **Synthetic 2 — planted self field, partner shuffled**: rate driven by self position. Assert `category == 'self_only'` for the planted cell, and *not* significant for any partner target.
- **Synthetic 3 — no place tuning**: rate is constant. Assert no target is significant after FDR.
- **Synthetic 4 — conjunctive**: rate driven by a Gaussian over the 4-D `(x_self, y_self, x_partner, y_partner)`. Marginal rate maps over self and partner should both be significant; `category == 'conjunctive'`.
- **Edge cases**: very few spikes (< `min_n_spikes`) → graceful skip with stats present but flagged; arena_bounds inferred vs explicit; speed_threshold so high it filters everything → graceful failure with informative message.
- Update `tests/run_tests.py`.

### 7. Documentation

- Add a row to `ARCHITECTURE.md`'s Capabilities table linking to `ephys/social_spatial_fields.py`.
- Add a short section to `README.md` under "ephys" with the API summary and CLI example.
- Append to `CLAUDE.md` Gotchas:
  - "Social place field occupancy must be computed over the *target's* `(x, y)`, not the focal animal's. Speed-gating by target (not focal) is the default and important — when the partner is stationary, occupancy variance dominates the rate map."
  - "Tracking↔ephys time conversion lives only in `MultiAnimalSession.get_tracking_on_ephys_clock`. Anything else that needs both clocks must call that method."
- Notebook stub `ephys/social_place_fields_demo.ipynb`: load session → compute_social_place_fields → render grid plots for a few example cells → cell-classification summary.

## Acceptance criteria

1. `python -m ephys.run_social_spatial --session_id 20251216 --animal_ids 631 632 633 --focal 631 --n_shuffles 200 --output_dir ./results` runs end-to-end on real data, writes a pickle, a PNG summary, and a PDF, exits 0.
2. Synthetic test #1 (planted partner field) recovers the bump within one bin; shuffle p < 0.001 with 500 shuffles.
3. Synthetic test #2 (self-only planted) correctly classifies the cell as `self_only`.
4. `from ephys.social_spatial_fields import compute_social_place_fields, RateMap, FieldStats, SocialFieldResults` works in a fresh Python session.
5. Streamlit GUI shows a Social Place Fields tab; selecting focal + targets + cluster renders the grid plot; cache hits on re-render at the same params.
6. No tracking↔ephys time conversion outside `MultiAnimalSession.get_tracking_on_ephys_clock` — verified by grep for `convert_behavior_to_ephys` / `convert_ephys_to_behavior` outside `ingestion/`.
7. Result-dict conventions match the existing decoders: `parameters` carries `class_label='target_position'` and `analysis_title` so the plot module can be driven from the dataclass without code branches per target.
8. Existing pytest suite still passes (`cd tests && python run_tests.py`).
9. `decode_location.py` is **not** modified. The new module reads from `KilosortData` and `MultiAnimalSession`; it doesn't reach into the existing location decoder.

## Gotchas (will bite if ignored)

- **Occupancy is over the target.** The focal animal's `(x, y)` is irrelevant for a partner-target rate map (it might be used as a covariate in stretch goals). Don't accidentally swap them — the synthetic test #2 exists to catch this.
- **Speed-gating default is `target`**, not `focal`. The point is to remove time during which the partner is stationary and any apparent tuning is dominated by where the partner happens to be parked.
- **Smooth then divide.** Apply Gaussian smoothing to occupancy and spike-count maps **before** dividing — smoothing rates after dividing amplifies noise in low-occupancy bins.
- **Shuffle nulls preserve temporal structure.** Use circular shifts of the spike train (rigid rotation of one of the two signals), not random permutation of bins. Position-shuffle is the alternative null; offer both, default to `circular_shift`.
- **Tracking is in behavior seconds (after Linux-ns → seconds via `synchronize_with_ephys`).** Conversion to ephys seconds must happen exactly once, inside `get_tracking_on_ephys_clock`.
- **All animals in a session share an ephys clock** (see `CLAUDE.md`); each animal's spikes are already comparable in seconds. No cross-animal ephys conversion.
- `KilosortData` is a dataclass; use `load_kilosort_data(path)` and the dataclass methods (`bin_spike_times`, `filter_cells_by_firing_patterns`); do not pass a `DataStorageManager` to `KilosortData`.
- Don't touch `config/`, `_lda_decoding.py`, `decoding_plots.py`, or the existing decoder result-dict schemas.
- Don't add backwards-compat shims for the old `KilosortData(data_input=...)` API.

## Out of scope (do NOT do these)

- Egocentric (head-direction-referenced) social receptive fields. That's a related but separate module — `ephys/social_egocentric_fields.py` — and depends on a clean head-direction signal we may not yet extract.
- Conjunctive 4-D `(x_self, y_self, x_partner, y_partner)` rate maps. Marginal 2-D rate maps + the `conjunctive` category classification are enough for v1.
- Cross-day stability / drift correction. Single-session scope.
- Anything that modifies `decode_location.py` or the LDA decoders.

## Stretch goals (only after green tests)

- **GLM disambiguation**: per-cell Poisson GLM with `(self_x, self_y, partner_x, partner_y, focal_speed, head_dir)` as covariates; report relative deviance explained per term. This is the proper way to control for the focal animal correlating with the partner's position.
- **Spike-position information across multiple partners simultaneously** via a single multivariate GLM (joint rather than per-target marginals), with elastic-net regularization.
- **Permutation-test of "broadcast vs partner-specific" classification**: shuffle partner identities and recompute per-cell field-similarity-across-targets; categories should be robust to partner-identity shuffles only for broadcast cells.
- **Egocentric stretch**: once head direction is reliably extracted, add an `egocentric=True` flag that transforms target `(x, y)` into a head-centered frame before binning.

## Build order

1. `MultiAnimalSession.get_tracking_on_ephys_clock` (§2) — write the test for this first; the rest depends on it.
2. `compute_rate_map`, `spatial_information`, `spatial_sparsity`, `spatial_coherence`, `split_half_stability` (§1 core) with synthetic test for a single planted field.
3. `field_significance` with circular-shift null; synthetic test asserting null < true.
4. `compute_social_place_fields` (multi-target sweep) and `cell_classification`; synthetic tests for self-only and conjunctive cases.
5. Plots (§3).
6. CLI (§4).
7. GUI tab (§5).
8. Docs + notebook + run on a real session.

Commit after each numbered step. Match the existing commit-message style (see `git log`).
