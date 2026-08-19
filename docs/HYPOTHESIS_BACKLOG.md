# Hypothesis backlog — multi-animal, simultaneous ephys + tracking

Merges two sources into one ranked list: the 7 ideas brainstormed in-session (see
git history of this file) and the more extensive list contributed in
`docs/HYPOTHESIS_LIST.md`. Every item below was cross-checked against the actual
repo state — several of the contributed list's ideas turned out to already exist,
and a few implicit assumptions (N>2 simultaneous ephys, usable head direction)
don't hold yet. `docs/HYPOTHESIS_LIST.md` is kept as-is for its full citations and
prose; this file is the actionable, feasibility-checked backlog to draw from.

## Difficulty tiers

- **Tier 0** — already implemented; a demonstration/read of existing results, not new work.
- **Tier 1** — existing module, different parameters or conditioning; a `run-analysis` exercise.
- **Tier 2** — new code, but reuses most of an existing pipeline; a moderate `implement-module` lift.
- **Tier 3** — a genuinely new analysis/module; a larger lift.
- **Tier 4** — needs a prerequisite that doesn't exist yet (new infra, new dependency, or unconfirmed data availability) before the neuroscience question is even testable.

## Cross-cutting caveats (apply to multiple items below)

1. **Tracking coverage is session-specific.** Session `20251216`'s tracking only
   has the focal animal (`rat631`) identity-resolved — discovered while testing
   Hypothesis #3 (`database/lab_notebook.py`, iteration 9, status `blocked`). Check
   `object_name` values in a session's `*_mask_metrics.csv` before assuming
   multi-animal position data exists for it.
2. **N>2 simultaneous ephys is actually routine — confirmed 2026-08-19.**
   `get_animals_and_sessions()` shows most cohort-7 sessions (including `20251216`,
   the session behind the opponent-identity finding) have **4** simultaneously-
   implanted animals (`rat613`, `rat615`, `rat630`, `rat631`), not 2. This corrects
   the original caveat here — the data-availability blocker doesn't apply. What's
   still missing is the *math*: `ephys/inter_brain_dynamics.py::fit_shared_subspace`
   and every test/CLI/README example are pairwise-only; multi-set CCA is an
   unimplemented stretch goal per `ephys/README.md`. So Tier assignments in §I below
   reflect a **missing-method** gap, not a missing-data one — meaningfully easier to
   greenlight than originally written, since the 4-animal ephys data to test against
   already exists for a session we've already worked with.
3. **Real head direction doesn't exist yet.** The raw tracking CSV's `orientation`
   column passes through unvalidated; `video/behavior_features.py::_kinematics`
   derives heading from movement velocity instead. Gaze/attention-based ideas need
   head-direction extraction (from pose keypoints or a validated `orientation`
   signal) built first — movement-heading is a reasonable proxy for "which way am I
   walking" but not "which way am I looking."

---

## A. Already there or nearly free

- **[Tier 0] Allocentric social place fields.** Already built —
  `ephys/social_spatial_fields.py::compute_social_place_fields`. Recapitulates
  Danjo et al. 2018 / Omer et al. 2018 / Sarel et al. 2022, but in a much larger
  arena with multiple identifiable partners than any of those studies. Nothing to
  implement; worth a fresh write-up of what's already on record (16 significant
  opponent-identity cells, iteration 7) framed through this lens.
- **[Tier 0] Conjunctive self×partner cells.** Already detected —
  `social_spatial_fields.py::_classify_cells` outputs a `"conjunctive"` category
  (significant tuning to both the focal animal's own position and ≥1 partner's).
  With 8 distinguishable opponents in the EC event set, worth checking whether any
  conjunctive cells are partner-*specific* (conjunctive only with opponent X) vs.
  partner-general.
- **[Tier 1] Partner-position decoding.** `ephys/decode_location.py::decode_all_locations`
  already loops every tracked object against one animal's spikes with cross-validation
  (`_cv_decode`) and a reverse/shuffle null (`_null_position_decode`) built in — decoding
  rat B's position from rat A's spikes is a parameter choice (`object_name='rat632'`),
  not new code. Missing only a thin wrapper matching the repo's unified result-dict
  schema (`class_label`/`analysis_title`/`parameters`) to fit `run-analysis` cleanly.
- **[Tier 1] Pre-event / anticipatory outcome signatures.** `decode_event_outcome`
  already exists; flip `time_window` negative (e.g. `(-3, 0)`) to ask whether the
  focal animal's neural state *before* an encounter's outcome is decided already
  differentiates winning from losing trials. Cheapest genuinely new finding available.
- **[Tier 1] Cohort-shuffle nulls for inter-brain coupling.** Re-pair animal A's
  spikes with animal B's *behavior from a different session* and re-run
  `regress_shared_on_behavior` — tests how much apparent coupling is real
  within-session structure vs. "rats just do similar things in this arena." A null
  variant of existing code, not a new method.

## B. Social spatial & vector coding

- **[Tier 2] Egocentric "social vector" cells.** Does a cell encode a partner's
  position in the focal animal's own egocentric frame (distance + bearing) rather
  than, or in addition to, absolute allocentric position? `video/behavior_features.py
  ::build_behavior_feature_matrix` already computes `distance` and `relative_bearing`
  per bin (using movement-derived heading, see caveat 3 — a usable proxy for this,
  not for gaze). New piece: re-bin `compute_rate_map`'s occupancy/significance
  machinery against `(distance, relative_bearing)` instead of allocentric `(x, y)`.
  Classic boundary-vector-cell statistics (Hartley/Lever) apply directly.
- **[Tier 2] Personal-space boundary cells.** Extends `ephys/decode_partner_distance.py`
  with a changepoint/nonlinear-fit variant instead of assuming a smooth linear
  distance→rate relationship — tests whether some cells are literally
  personal-space-violation detectors.
- **[Tier 3] N-body social fields.** Rate maps conditioned on the *nearest* partner,
  the *dominant* partner, or cohort centroid — needs 3+ simultaneously-tracked
  animals (routine, per the 12-animal `RatCity_20251210` session) but the analysis
  itself (multi-argument conditioning on `compute_rate_map`) is new code.
- **[Tier 4] Topology-aware (graph-geodesic) representations.** Compare allocentric
  Euclidean maps against graph-geodesic maps if the arena has corridors/junctions —
  needs an arena-topology graph that doesn't exist in the repo yet, on top of new
  analysis code. Rodent analog of Babichev/Dabaghian topological coding.

## C. Inter-brain shared dynamics (pairwise — existing infra)

- **[Tier 1–2] Event/behavior-conditioned shared subspace.** Does the shared
  subspace between two animals' populations (`fit_shared_subspace`) tighten around
  agonistic events vs. neutral coexistence? `regress_shared_on_behavior` already
  accepts event-indicator columns from `build_behavior_feature_matrix` — likely a
  different conditioning/windowing of existing code, worth trying as `run-analysis`
  before assuming new code is needed.
- **[Tier 1–2] Leader/follower directionality.** `time_lagged_cca` already supports
  a lag sweep between two animals' activity; running it specifically inside
  agonistic-event windows vs. baseline and reading off the peak-lag sign gives a
  quantitative leadership signal per encounter. Prediction worth testing: dominant
  animals lead during active interactions, follow during quiescence.
- **[Tier 2] Anticipatory inter-brain coupling.** Does shared-subspace coupling
  strength rise *before* approach events (coupling driving interaction, not just
  reflecting it)? Extends the event-conditioned analysis above with a predictive
  framing — predict "interaction in the next 5 s" from current coupling strength.
- **[Tier 2] Subspace projection-out ("in-silico ablation").** Project animal A's
  spike vectors out of the fitted shared subspace, re-run the existing
  `decode_opponent_identity`/`decode_event_outcome` LDA decoders, and measure
  degradation — an in-silico causal-ish test reusing entirely existing decoders and
  the existing subspace fit, just composed differently.
- **[Tier 2] Behavior-residual decoding.** Regress out speed/position/distance-to-partner
  from spikes (already-available covariates via `build_behavior_feature_matrix`),
  then ask whether the residual still decodes opponent identity — directly answers
  the "are you just decoding running speed" confound, a sharper version of the
  spatial-confound question Hypothesis #3 tried to ask (and was blocked on, for lack
  of opponent position data — this variant only needs the *focal* animal's own
  kinematics, which are always available).

## D. Identity, rank, and social history

- **[Tier 2] Representational geometry of dominance/rank.** Reframes
  `decode_opponent_identity`'s classification (already established: 28.8% vs 27.7%
  baseline, animal 631/session 20251216) as representational similarity analysis —
  does neural distance between opponent pairs' representations correlate with their
  rank/ID distance? Reuses `ephys/_lda_decoding.py::extract_firing_rate_features`
  entirely; only the analysis on top (RSA instead of LDA) is new. Directly relevant
  to why Hypothesis #2's group-level split found a real-but-tiny effect — rank may
  be encoded continuously, not categorically.
- **[Tier 2] Relationship-history regression.** Does the neural representation of a
  given opponent depend on cumulative aggressive history with that specific
  opponent? Regress single-event neural distance on cumulative wins/losses — reuses
  the lab notebook's per-event outcome data plus existing feature extraction.
- **[Tier 2–3] Cross-day stability of the identity code.** Does a given animal's
  opponent-identity or partner-place-field tuning stay stable across days, or
  drift/reassign? `get_animals_and_sessions(config_path)` already enumerates 47
  animal/session pairs (deferred "multi-session sweep" in `HANDOFF.md`). First
  concrete pair to try: session `20251216` (animal 631, opponent-identity finding
  already on record) vs. `RatCity_20251210_1359_40Hz` (full 12-animal tracking,
  overlapping opponent IDs) — **confirmed 2026-08-19**: animal 631 (plus 613, 615,
  630) does have usable ephys on session `20251210` too (verified while running
  partner-position decoding there, iterations 10-11), so this comparison is ready
  to run, not blocked on a data check anymore.
- **[Tier 2–3, opportunistic] Familiarity/novelty axis.** If cohort composition
  ever changed (a new rat introduced), compare the neural representation of the
  novel animal in the first sessions vs. after stabilization. Can't be scheduled —
  depends on whether such a transition exists in the recorded history.

## E. Partner-state decoding, extended

- **[Tier 2] Partner-behavior-state decoding.** Decode B's instantaneous behavioral
  state (resting/locomoting/rearing — derivable cheaply from existing speed
  thresholds in `_kinematics`) from A's spikes. Worth a leave-one-partner-out
  scheme across multiple opponents to test whether the code generalizes across
  identities or is identity-specific.
- **[Tier 2] Asymmetric / time-lagged partner-position decoding.** Sweep lag on top
  of the existing `decode_location` cross-validation: does A predict B's *future*
  position/behavior better than B's current state? The social analog of
  motor-prediction; asymmetry across a dyad is a second, independent leadership signal
  alongside C's CCA-based one.
- **[Tier 3] Occlusion analysis.** The strongest "internal model, not just sensory
  tracking" test: split time into line-of-sight vs. no-line-of-sight bins (needs
  arena geometry — doesn't exist in the repo) and check whether partner-position
  decoding stays above chance during occlusion. High scientific value, blocked on
  building arena/line-of-sight geometry first.

## F. Directional information flow (new methods)

- **[Tier 3] Transfer entropy / mutual information rate between brains.** A more
  general (and more expensive) alternative to time-lagged CCA's directionality
  signal; would reuse `ephys/_stats_utils.py`'s permutation/surrogate machinery for
  significance rather than rebuilding it.
- **[Tier 3–4] Cross-brain "communication subspace" (Semedo et al. 2019 style).**
  Find the low-d projection of A's activity that best predicts B's activity at lag
  τ, quantify its dimensionality and asymmetry against a within-brain null. Genuinely
  new method on top of existing binned-rate infra.

## G. Joint behavior discovery

- **[Tier 2–3] Co-locomotion / joint-movement states.** Detect bouts of correlated
  heading+speed between two animals (using `_kinematics`, already computed) and ask
  whether a distinct population state accompanies moving together vs. coincidental
  proximity — a lighter-weight, hypothesis-driven cousin of the unsupervised
  approach below.
- **[Tier 3–4] Unsupervised joint behavior segmentation ("social syllables").**
  keypoint-MoSeq/VAME-style segmentation on a joint dyadic feature stream (relative
  distance, bearing, mutual orientation, egocentric pose) to discover motifs
  (chase, parallel locomotion, mirroring) not captured by manual scoring. New
  dependency (a segmentation library) plus a new `BehavioralEventsData` event-source
  extension to feed discovered syllables back into the existing LDA/decoding stack.

## H. Gaze & attention — blocked on head-direction infra (caveat 3)

- **[Tier 4] Mutual-gaze epochs.** Cells that fire only when both animals' head
  directions point at each other.
- **[Tier 4] Joint-attention cells.** Both animals oriented toward the same external
  location (food port, novel object) — essentially unstudied in rodent ephys, but
  needs real head direction, not movement heading.
- **[Tier 4] Being-watched representation.** Does A's neural state differ when B is
  looking at it vs. not, controlling for distance/behavior? Same prerequisite.

All three are scientifically the most novel items in this backlog if the
head-direction prerequisite gets built — currently blocked, not merely hard.

## I. Multi-brain (N>2) methods — blocked on missing math, not missing data (see caveat 2)

Data is no longer the blocker here (most cohort-7 sessions have 4 simultaneously-
implanted animals) — these are Tier 3/4 on algorithmic lift alone now, downgraded
from "blocked."

- **[Tier 3–4] Generalized / multi-set CCA (Kettenring 1971) or JIVE/SLIDE.** Finds
  directions correlated across all N brains, partitioning variance into
  cohort-shared / dyad-shared / individual. Explicitly listed as an unimplemented
  stretch goal in `ephys/README.md` — genuinely new math (not in any existing
  dependency), but the 4-animal data to run it on already exists for session
  `20251216`.
- **[Tier 3–4] Identity-aware per-dyad subspace comparison.** Fit one shared subspace
  per dyad (A↔B, A↔C, …) and ask whether A's loadings are similar across dyads
  (broadcast) or partner-specific ("a channel per friend") — a multi-brain analog
  of communication subspaces. Cheaper than multi-set CCA since it reuses the
  existing pairwise `fit_shared_subspace` N-choose-2 times rather than needing new math.
- **[Tier 4, opportunistic] Cohort reorganization events.** If an animal is ever
  added/removed from a cohort, track representational change in the rest of the
  group. Data-ready if such a transition exists in the recorded history; otherwise
  can't be scheduled.

## J. Replay

- **[Tier 3] Trajectory replay during quiescence.** Standard Pfeiffer-Foster-style
  compressed-sequence detection during low-velocity epochs — `decode_location.py`'s
  Bayesian decoder is a real head start, but sequence/event detection on top of it
  doesn't exist yet.
- **[Tier 4] Vicarious replay.** Does A's rest-period activity replay paths B
  recently took? The single cleanest rodent demonstration of observational replay if
  it works, but compounds trajectory-replay detection with partner-position decoding
  — build trajectory replay first.
- **[Tier 4] Joint trajectory replay / cross-brain replay coupling.** Replay of the
  *pair's* joint trajectory, and whether replay events couple across brains at a
  compressed timescale. Hardest items in this backlog — compound of two Tier 3/4
  pieces each.

## K. Modern / heavier methods

- **[Tier 3] Pose-conditioned encoding GLM.** Per-cell mixed-selectivity model
  (position, speed, head direction if available, partner identity/position, social
  syllable if built). The encoding-model analog of the existing decoding work;
  well-scoped but genuinely new infra.
- **[Tier 3] Tensor Component Analysis (cells × time × encounter).**
  `ephys/_rate_tensor.py::event_aligned_rates` already builds exactly this tensor
  shape — the data-prep half of this is done; only the TCA decomposition itself
  (e.g. Williams et al. 2018) and correlating component loadings with
  identity/outcome/rank is new.
- **[Tier 3–4] HMM over cohort joint behavioral state.** A discrete "what is the
  cohort doing" state variable (foraging-alone, dyadic-chase, group-cluster) to
  condition any analysis above on. New infra, but conceptually simpler than full
  unsupervised syllable segmentation (G).
- **[Tier 4] Cross-brain CEBRA / contrastive embedding.** Joint-brain contrastive
  learning (Schneider et al., *Nature* 2023) as an alternative to CCA for the shared
  subspace. Would introduce a new ML dependency not currently in the repo — biggest
  new-infra lift here, positioned as a `method="cebra"` branch alongside the
  existing `fit_shared_subspace` per `ephys/README.md`'s own stretch-goal list.

## Update 2026-08-19 — partner-position decoding (item A) run; a caution, not a result

Ran `ephys/decode_location.py::decode_all_locations` (item A, Tier 1) for animal
631/session `20251210` against all 12 tracked objects. Two things worth recording
for whoever picks up any `decode_location`-based item next:

- **The module's default `null='reverse'` is a weak, easily-misread null.** Per its
  own docstring it only reverses trajectory order, preserving every marginal and
  autocorrelation — it tests temporal-order asymmetry, not "beats chance." A first
  pass using it showed several partners nominally "beating" the null, including a
  spurious win for a near-stationary animal (`rat630`, x/y std ~11-13px — any method
  gets near-zero error on a target that barely moves) and cases where the null
  outperformed real decoding entirely (likely stereotyped/patterned trajectories).
  **Always use `null='shuffle'` with a proper `empirical_p_value`/BH-FDR pass
  (`ephys/_stats_utils.py`) for any real claim from this module** — same lesson the
  LDA decoders already learned the hard way (see Phase 1.5 above).
- **Self-position decoding failed a proper shuffle-null test** (p=0.43, animal 631
  decoding its own position from its own spikes, default params: `bin_size=0.5`,
  `n_spatial_bins=20`, `smoothing_sigma=1.0`, no `pixels_per_cm` calibration, default
  quality-cell thresholds). This is the basic sanity check any spatial decoder should
  clear before a partner-decoding claim means anything, so **partner-position
  decoding is not ready to yield a trustworthy finding with default parameters** —
  needs either arena-scale-tuned spatial binning, a wider quality-cell net, or a
  check on whether this recorded population/region is classically place-tuned at
  all, before revisiting. One partner (`rat613`) nominally cleared FDR (q=0.039 of 7
  tests) but given self-decoding's failure this reads as a likely false discovery,
  not a real effect — logged as iterations 10 (reverse-null) and 11 (shuffle-null,
  real p/q values) rather than presented as a finding.
