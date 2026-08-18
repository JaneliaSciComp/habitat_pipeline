# Hypothesis backlog — multi-animal, simultaneous ephys + tracking

A running list of scientifically motivated analysis ideas that specifically exploit
this dataset's rare combination: **multiple animals, each with simultaneously
recorded electrophysiology, freely moving and individually tracked in a shared
large arena.** None of these are pre-registered `Hypothesis` rows yet (see
`database/lab_notebook.py`) — this is a backlog to draw from via `propose-hypotheses`
once a specific session/finding grounds one of them, or to hand directly to
`implement-module` once the shape is clear enough.

Each entry notes what existing repo infrastructure it would reuse and a rough
feasibility tier:

- **Tier 1** — existing module, different parameters/conditioning; likely just a
  `run-analysis` exercise.
- **Tier 2** — new code, but reuses most of an existing pipeline; a moderate
  `implement-module` lift.
- **Tier 3** — a genuinely new analysis; bigger lift.

## 1. Egocentric "social vector" cells — Tier 2

**Question.** Does a cell encode a partner's position in the focal animal's own
egocentric reference frame (distance + bearing — "how far and which direction is he
from my nose") rather than, or in addition to, the partner's absolute allocentric
position?

**Why it's novel here.** `ephys/social_spatial_fields.py::compute_social_place_fields`
already tests allocentric tuning to a partner's absolute `(x, y)`. Nothing yet tests
the egocentric-vector reference frame — the distinction between "encodes where the
opponent is in the room" and "encodes where the opponent is relative to me" is exactly
the kind of question that needs a real arena and a real conspecific, not a fixed cue
or a small box.

**Reuses.** `video/behavior_features.py::build_behavior_feature_matrix` already
computes `distance` (focal–partner Euclidean) and `relative_bearing` (egocentric
bearing to partner) per time bin. The new piece is re-binning firing rate against
`(distance, relative_bearing)` instead of allocentric `(x, y)` — the rate-map,
occupancy-normalization, and permutation-significance machinery in
`compute_rate_map`/`FieldSignificance` should transfer almost directly, just with a
polar/vector grid instead of a Cartesian one.

## 2. Event-conditioned inter-brain coupling — Tier 1

**Question.** Does the shared neural subspace between two animals' populations
tighten specifically around agonistic encounters (fights, chases) compared to neutral
coexistence?

**Why it's novel here.** Requires two brains recorded on the same clock at the same
time — this is the "two brains, one clock" property CLAUDE.md flags as the defining
feature of the inter-brain module.

**Reuses.** `ephys/inter_brain_dynamics.py::fit_shared_subspace` and
`regress_shared_on_behavior` already accept behavior/event-indicator columns from
`build_behavior_feature_matrix` (which builds one indicator column per event type).
This may be closer to a different **conditioning/windowing** of an existing pipeline
than genuinely new code — worth trying as a `run-analysis` exercise before assuming
it needs `implement-module`.

## 3. Who's driving the interaction — leader/follower directionality — Tier 1–2

**Question.** During a chase/escalation event, does one animal's population activity
predict the *other's future movement* more than the reverse — i.e., which brain is
"leading" the social decision?

**Why it's novel here.** Impossible without simultaneous multi-animal ephys; this is
a directional, causal-flavored question standard single-animal decoding can't touch.

**Reuses.** `ephys/inter_brain_dynamics.py::time_lagged_cca` already supports a
lag sweep between two animals' population activity. The new piece is running that
sweep specifically inside agonistic-event windows vs. baseline and comparing the
lag/asymmetry — likely a thin wrapper around existing infra rather than new math.

## 4. Representational geometry of dominance/rank — Tier 2

**Question.** Rather than asking whether opponent identity is *decodable*
(classification accuracy — already established for animal 631/session 20251216,
see `HANDOFF.md`), do the population's *representational distances* between
opponents respect a linear dominance hierarchy? I.e., is the neural distance between
opponent A's and B's representations correlated with their rank/ID distance
(a representational similarity analysis, RSA-style reframing)?

**Why it's interesting.** Might explain why the group-level (`label_mode='group'`)
opponent-decoding test (`Hypothesis` id 2 in the notebook) found a real-but-tiny
effect: rank could be encoded more continuously/geometrically than the categorical
group split can detect.

**Reuses.** `ephys/_lda_decoding.py::extract_firing_rate_features` for the same
feature extraction already used by `decode_opponent_identity`; the new piece is an
RSA-style distance/correlation analysis instead of an LDA classifier.

## 5. Personal-space boundary cells — Tier 2

**Question.** Does firing rate show a threshold/step nonlinearity at a specific
inter-animal distance (a "personal space violation" detector), rather than the
smooth linear relationship the current regression assumes?

**Reuses.** `ephys/decode_partner_distance.py`'s data loading and distance-binned
regression pipeline; the new piece is a changepoint/nonlinear-fit variant instead of
(or alongside) the existing linear regression.

## 6. Co-locomotion / joint-movement states — Tier 3

**Question.** During bouts when both animals move together (correlated heading and
speed — not just coincidentally close), is there a distinct population state
compared to bouts of similar proximity without correlated movement?

**Why it's novel here.** Only meaningful with two simultaneously tracked,
simultaneously recorded animals in an arena large enough for genuinely independent
locomotion — a static distance/place-field framing (already covered by #1 and #5)
can't distinguish "moving together" from "coincidentally close."

**Reuses.** Per-animal kinematics already computed in
`video/behavior_features.py::_kinematics`; the new pieces are a co-movement bout
detector and a population-state comparison (e.g. shared-subspace occupancy or a
state-decoding pass) — a genuinely new module.

## 7. Cross-day stability of the social code — Tier 3 (infra mostly exists)

**Question.** Does a given animal's opponent-identity code or partner-place-field
tuning remain stable across recording days, or does it drift/reassign?

**Why it's novel here.** Needs the same identified individuals recorded on multiple
days — this cohort's design (same rats, same arena, repeated sessions) supports it
directly.

**Reuses.** `get_animals_and_sessions(config_path)` already enumerates 47
animal/session pairs across both cohorts (flagged as a deferred "multi-session
sweep" in `HANDOFF.md`). A concrete first pair worth checking: session `20251216`
(animal 631, rich ephys + agonistic events, opponent-identity finding already on
record) vs. `RatCity_20251210_1359_40Hz` (full 12-animal tracking, including most
of the same opponent IDs) — *if* animal 631 has usable ephys on the 2025-12-10
session too.

## Notes from the session this backlog came out of

- Two supporting-infrastructure facts worth remembering when picking one of these up:
  `build_behavior_feature_matrix` already computes egocentric bearing/distance, and
  `inter_brain_dynamics.py` already has a permutation-null layer
  (`shuffle_null_subspace`) and lag-sweep support (`time_lagged_cca`) — several of
  the ideas above are cheaper than they look because of this.
- Tracking coverage is **session-specific**, not just cohort-specific: session
  `20251216`'s merged tracking file only has the focal animal (`rat631`)
  identity-resolved (discovered while trying to test a spatial-confound hypothesis
  against opponent-identity decoding — see `Hypothesis` id 3 / iteration 9 in the
  lab notebook, status `blocked`). Check per-session tracking coverage
  (`object_name` values in the merged `*_mask_metrics.csv`) before assuming
  multi-animal position data is available.
