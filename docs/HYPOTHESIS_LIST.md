## 1. Social spatial codes — the cleanest first wins

You have all-animal allocentric tracking plus sorted spikes. The most direct novel-but-tractable analyses are spatial firing maps with respect to *other rats* rather than self.

- **Allocentric "social place fields"** — for each cell, build an occupancy-normalized rate map over rat B's `(x, y)`, ignoring rat A's own position. This recapitulates Danjo et al. 2018 (CA1 social place cells in observation), Omer et al. 2018 (bat hippocampus encoding another bat), and Sarel et al. 2022 — but in a much larger arena with rich navigation and multiple partners. Skaggs spatial information + spatial sparsity + split-half stability gives you the cell-level statistics you already use in `decode_location.py`.
- **Egocentric social receptive fields** — bearing and distance to each conspecific in head-centered coordinates. Drop next to allocentric maps as a complementary view; classic boundary-vector-cell statistics (Hartley/Lever) apply directly. Need head direction from pose — if not already extracted, derive from body axis or two head keypoints.
- **N-body social fields** — rate maps conditioned on the relative configuration of 2+ conspecifics (e.g., position of the *nearest* partner, position of the *dominant* partner, mean centroid of cohort). Almost nobody has the N to compute these.
- **Conjunctive (self × partner) cells** — Sarel-style: cells whose firing depends on the joint position of two animals. With several rats, you can ask whether conjunction is partner-specific (rat 6 vs rat 7) or partner-general.
- **Topology-aware representations** — if the arena has corridors/junctions, build a graph and compare allocentric Euclidean maps with graph-geodesic maps; places that *share a corridor* with the partner may share more representation than equidistant places. This is the rodent analog of Babichev/Dabaghian topological-coding work.

Drops in as `ephys/social_spatial_fields.py` alongside `decode_location.py`. The unifying API is a `compute_rate_map(spikes, target_xy, occupancy, smoothing)` that doesn't care whose `(x, y)` it gets.

## 2. Decoding the partner from the self-brain

The single most striking demonstration of "the brain represents the other" is to decode the partner's state from the focal rat's neural activity.

- **Partner-position decoding** — train the existing Bayesian location decoder to predict rat B's `(x, y)` from rat A's population. Compare to (i) decoding A's own position, (ii) shuffled-partner null, (iii) using B's own brain. Plot accuracy as a function of inter-rat distance, visibility, and behavioral state.
- **Partner-behavior decoding** — decode B's instantaneous behavioral state (resting, locomoting, rearing) from A's spikes, ideally with a leave-one-rat-out scheme across multiple partners (does the code generalize across identities or is it identity-specific?).
- **Asymmetric / time-lagged decoding** — sweep lag: does A predict B's *future* behavior better than B's current? This is the social analog of motor-prediction. Asymmetry across rats is a quantitative leadership signal.
- **Occlusion analysis** — the gold-standard "internal representation" test: split data into time bins where A can vs cannot directly sense B (use line-of-sight from arena geometry + tracking). If partner-position decoding stays above chance during occlusion, you've shown A maintains a mental model of B, not just sensory tracking. This is hard to do in any other setup.

Lives naturally as `ephys/decode_partner.py`. Reuses `decode_location.py` with a different `object_name`.

## 3. From pair to N — multi-brain shared dynamics

The CCA-based two-brain decomposition you're about to build is the special case K=2 of a larger family. With several rats per session you can do things nobody else can.

- **Generalized / multi-set CCA** (Kettenring 1971; Tenenhaus & Tenenhaus 2011) — find directions that are simultaneously correlated across all N brains. Variance-explained partitions into: cohort-shared (all N), dyad-shared (subsets), self-only.
- **JIVE / SLIDE / multi-block PCA** (Lock et al. 2013; Gaynanova & Li 2019) — explicitly factorizes variance into "joint across all", "partially shared (specific subsets)", and "individual". Tells you, for any dyad, what's shared with this partner specifically vs broadcast across the cohort.
- **Identity-aware shared subspace** — fit one shared subspace per dyad (A↔B, A↔C, …); ask whether the loadings on rat A's neurons are similar across dyads (broadcast representation) or different (partner-specific subspaces, i.e., a "channel per friend"). This is conceptually a multi-brain analog of "communication subspaces" (Semedo et al. 2019).
- **Behaviorally-conditioned shared subspace** — fit the shared subspace separately during chase, side-by-side, no-interaction, post-aggression; ask whether the shared dimensions rotate as a function of social state.

Extends the inter-brain module you're building. Add `fit_multibrain_shared` and a `JointVariancePartition` dataclass.

## 4. Directional information flow between brains

Shared subspace gives you symmetric coupling. Information *flow* is asymmetric and more interesting.

- **Transfer entropy / mutual information rate** between binned population states across animals, with appropriate surrogates (circular shifts). Good baseline.
- **Granger causality on shared-subspace time courses** — much more reliable than per-cell GC because the dimensionality is small and the signal is clean.
- **Cross-brain "communication subspace"** — generalize Semedo et al.: find low-d projection of A's activity that best predicts B's activity at lag τ, and vice versa. Quantify the dimensionality and the *asymmetry* of the predictive subspace. Compare against a within-brain communication subspace as null.
- **Leader-follower index** — for each dyad, sweep time-lagged CCA and locate the peak; the sign of the peak lag defines who leads. Plot leader-follower per behavioral motif. In a hierarchy, prediction: dominant rats lead during active interactions, follow during quiescence.

`ephys/inter_brain_flow.py` next to the subspace module.

## 5. Joint behavior segmentation and neural correlates of *discovered* social motifs

Manual scoring captures a few categories. The arena is full of behavior. Run unsupervised segmentation on the *joint* state of multiple animals — not just one rat's pose.

- **Dyadic feature stream**: relative distance, relative bearing, approach speed, mutual orientation, time-since-contact, plus each animal's egocentric pose features.
- **keypoint-MoSeq** or **VAME** on this joint feature stream → discovers "social syllables" (chase, parallel locomotion, mount, escape, mirror, leader-follower).
- Once you have N_syllables time-resolved labels, feed them into the existing LDA/decoding stack as new event types. The interesting question is whether *unsupervised* syllables are decoded better than *manual* events — if yes, the brain agrees with the data-driven taxonomy more than with the human one.
- Cohort-level: a 5-rat session produces a graph of syllable transitions per dyad; cluster across days to find stable "interaction styles" between individuals.

New module `video/social_segmentation.py` plus an event-source extension to `BehavioralEventsData`.

## 6. Replay during quiet wakefulness — vicarious and joint

The arena is large enough that real navigation happens, which means hippocampal-style sequence reactivation is on the table.

- **Trajectory replay** (Pfeiffer-Foster style) — during low-velocity epochs, decode position with a Bayesian decoder and detect compressed sequences that match recently traversed trajectories.
- **Vicarious replay** — does rat A's neural activity during rest replay paths that rat *B* recently took? This would be the cleanest single-experiment demonstration of "observational" or "vicarious" replay in rodents.
- **Joint trajectory replay** — replay the pair's joint trajectory (a sequence of joint position pairs). Tests whether the social structure of past experience is consolidated as a unit.
- **Cross-brain replay coupling** — when one rat replays a recent trajectory, is the other rat's brain coupled to it at the same compressed timescale?

Needs only sleep/quiescence detection + the existing place-cell decoder; `ephys/replay_detection.py`.

## 7. Social identity, recognition, and hierarchy

Multiple distinguishable partners + repeated days = a clean window onto identity coding.

- **Identity subspace** — fit a linear classifier on event-aligned activity to discriminate all N partners; the K most-discriminating directions define the "identity subspace". Ask whether it's stable across days (drift), across behavioral contexts (chase vs feed), and across the cohort hierarchy.
- **Dominance / status decoding** — extract a pairwise rank order from behavior (e.g., wins/losses in aggressive encounters). Decode opponent rank from neural activity at encounter onset. Track how the neural representation reorganizes when dominance flips (rare but valuable events). References: Zhou et al. 2017; Williamson et al. 2024.
- **Relationship-specific subspaces** — for each *pair*, compute the cross-brain shared subspace and the per-brain partner-evoked subspace; cluster across pairs to discover whether "best friends" and "rivals" have qualitatively different inter-brain geometries.
- **Familiarity / novelty axis** — if cohort composition has ever changed (new rat introduced), the neural representation of the novel rat in the first few sessions vs after stabilization is a textbook contrast.

Lives alongside the opponent decoders in `ephys/`.

## 8. Cross-brain CEBRA / contrastive embedding

This is the modern shape of the shared-subspace idea and is uniquely well-suited to your data because contrastive learning thrives on lots of trials and rich behavioral covariates.

- Joint-brain CEBRA: positive pairs = same-time slices from rats A and B; negative pairs = shuffled times. Learns a *single* low-d embedding into which both brains map, with explicit alignment to behavior labels (Schneider et al. *Nature* 2023).
- Extend to N brains and to conditioning on dyad identity — the embedding "remembers" which pair is which.
- Use the embedding for downstream decoding instead of raw rates; near-state-of-the-art on social datasets in recent work.

Wires into `ephys/inter_brain_dynamics.py` as an alternative `method="cebra"` branch.

## 9. Behavioral and neural anticipation

Anticipation is the cleanest readout of internal models.

- **Pre-event spike rate signatures** — sort encounters by outcome and ask whether the focal rat's neural state at, say, t = −3 s already differentiates winning vs losing encounters. The existing `decode_event_outcome` does t≥0; flip the time window.
- **Inter-rat anticipatory coupling** — does the cross-brain shared subspace strength rise *before* approach events? If yes, coupling drives interaction rather than reflecting it. Test causally (in the statistical sense) by predicting "interaction or not in next 5 s" from current shared-subspace energy.
- **Path-integration toward the partner** — fit a Bayesian model where the focal rat's predicted next location is influenced by partner position; quantify whether the brain encodes the model's prediction error.

## 10. Neural correlates of being-watched and joint attention

These need head direction + line-of-sight from arena geometry but are otherwise within reach.

- **Mutual-gaze epochs** — short windows where both rats' head directions point at each other within some tolerance. Look for cells that fire only during mutual orientation.
- **Joint-attention cells** — both rats look at the same external location (food port, novel object). Joint attention is a developmentally important precursor to theory of mind and is essentially unstudied in rodent ephys.
- **Being-watched representation** — does rat A's neural state differ when rat B is looking at it vs not, controlling for distance and behavior?

## 11. Long-timescale cohort dynamics

If recordings span days or weeks, the multi-rat aspect gives you something rare: a *social biography*.

- **Stability of identity subspaces across days** — chronic-recording drift correction (Yu et al.; Steinmetz et al. NeuroPixels alignment) then asks whether the rat's representation of "rat 6" is stable or drifts faster than its own place fields.
- **Relationship history** — does the neural representation of an opponent depend on the cumulative aggressive history with that opponent? Test by regressing single-event neural distance on cumulative wins/losses.
- **Cohort reorganization events** — if a rat is removed or added, track neural representational change in the rest of the cohort. Rare in any single experiment; routine in a long-running RatCity.

## 12. Embodied causal-ish tests within current data

You don't have closed-loop optogenetics, but several "natural ablations" are available.

- **Subspace projection-out** — project rat A's spike vectors out of the shared subspace, refit your existing opponent/outcome LDA decoders, measure degradation. This is your in-silico version of Hong-lab Fig. 6.
- **Behavior-residual decoding** — regress out predictable behavioral covariates from spikes (speed, position, distance to partner) and ask whether *what remains* still decodes opponent identity. Controls for "you're just decoding running speed."
- **Cohort-shuffle nulls** — re-pair rat A's spikes with rat B's behavior from a *different* session. Tests how much of the apparent coupling is true within-session as opposed to general "rats do similar things."

## 13. Methods that need a tiny bit of extra infra but pay off

- **Pose-conditioned encoding GLM** (per cell, mixed selectivity: position, speed, head direction, partner identity, partner position, social syllable). This is the encoding analog of your decoding work and immediately tells you each cell's contribution to each construct. References: Hardcastle/Giocomo, Mimica 2023.
- **Tensor Component Analysis** (Williams et al. 2018) on the cells × time × encounter tensor — produces trial-loading vectors you can correlate with opponent identity, outcome, hierarchy rank.
- **HMM over the cohort's joint behavioral state** — a single discrete state variable for "what is the cohort doing" (foraging-alone, dyadic-chase, group-cluster). Condition any of the above analyses on this state.
