# Hub co-occupancy decoding (rat630) - analysis spec, DRAFT

Status: draft for the scientist; nothing here is pre-registered, no holdout is reserved, no code beyond the
spread gate exists. Supersedes the exclusivity-GLM for rat630 (see "Why not the GLM").

## Question
While rat630 rests at its usual spot (the *hub*, about (3450, 610) px, shared with 1-5 other rats), does its
population activity carry **who else is there**, beyond what hub-local position, speed, time-drift and
partner distance explain?

This is a social co-occupancy question at fixed position. It is **not a territory claim**: the hub is shared
(other rats rest there in nearly every chunk; rat630's exclusivity there peaks at about 0.45), so there is no
owner whose presence or absence could be decoded. Calling it "owner-present" would be wrong.

## Why not the exclusivity GLM
On 20251216_145034 the focal occupies 3-6 effective grid bins per chunk (spread gate: eff_bins 3.4-6.2, exclusivity
range 0.03-0.14). Exclusivity is then a label for 3 positions; the 17/408 nominal hits were place-rate
differences (V-shaped, non-monotonic, sign-inconsistent across chunks). The spread gate now stops this.

## Data
- Focal: rat630, recordings 094334 / 145034 / 194334 (20251216), 095130 / 145130 (20251217), 111544 (20251212),
  161815 (20251215); chunks with full ephys overlap and >= 5 min hub rest (coverage_630.csv).
- Exploration: 20251216 only. Confirmation (reserved by the scientist before looking): 20251217.
- Rows: 0.5 s bins in which the focal is still (speed < 20 px/s) and within `hub_frac_diag * diag` of the hub.

## Labels (declared before running)
1. Per partner P with >= `min_min` minutes both present and absent in a chunk: `present_P(t)` = P within
   `r_near` px of the hub. One binary decode per partner (family = number of eligible partners x chunks).
2. Secondary: partner **count** at the hub (ordinal), reported separately.

## Statistic and null
- Cross-validated balanced accuracy of [baseline + neural] minus [baseline], baseline = fine hub-local position
  (x, y), speed, focal-partner distance, slow time trend. Blocked folds with purge; fit-in-fold z-scoring.
- Null: circular shift of `present_P` within the hub-rest segments (keeps its autocorrelation, breaks the link to
  neural activity). Add-one p. `fdr_resolution` must be checked against `family_denominator`, never a hand count.
- **Known weakness:** presence at the hub is nearly the same variable as focal-partner distance, which
  `decode_partner_distance` already tests. The incremental claim is *identity-specific* presence beyond
  distance; if the baseline already decodes presence near-perfectly there is nothing left to add, and the
  analysis must report that as "untestable", not as a null.

## Gates (all must pass before any neural statistic is read)
- Hub rest >= `min_min` minutes in the chunk, and both classes >= 40 rows per fold.
- Baseline BA below a ceiling (e.g. 0.9), otherwise untestable.
- Position-control gate (existing) for the focal.

## Calibration required before real data (not written yet)
Synthetic sessions with: (a) null cells with hub-local place tuning and drift; (b) presence labels
autocorrelated like the real ones and correlated with partner distance; (c) injected presence modulation of known
size. Show type-I within the binomial bound on (a)/(b) and power by effect size on (c).

## Open decisions for the scientist
- Is hub co-occupancy decoding worth running given decode_partner_distance already exists?
- `r_near`, hub radius and min minutes (all provisional).
- Holdout reservation (20251217), family declaration and frozen predictions.
