# Territory / social-context coding: feasibility note (2026-10)

Status: exploratory record, not a result. Nothing here is pre-registered; no holdout is reserved; the lab
notebook was not touched. Everything is in pixels (APT calibration unresolved). Thresholds marked "provisional"
were chosen by me and not validated.

## Conclusion
With the data in reach, neural coding of territory cannot be tested for rats 630, 631 or 613, and neither can
social-context coding at a fixed resting spot. Three separate reasons, each measured:

1. **No stable exclusive territory.** Behaviour screen over all 19 APT chunks (11 days).
2. **Covariates collapse to position** for the rats that have ephys (3-6 effective occupied grid bins).
3. **Company at a resting spot changes between chunks, not within them**, so presence is a chunk label and
   event-triggered designs have too few events.

This says the question is untestable *here*, not that the cells do not encode territory or social context.

## 1. Behaviour: territory is not exclusive (`video/territory_behavior.py`, `scripts/territory_behavior_screen.py`)
- Only two days have more than one APT chunk: 20251216 (7) and 20251217 (3). The other nine have one.
- Within-day median self-reliability of utilisation distributions is 0.36 on both days; median territoriality
  (self overlap minus strongest same-chunk neighbour overlap) is -0.12 and -0.11.
- Across all chunks only rat630 has a persistent home base (median shift 0.12 of the arena diagonal, self overlap
  0.73 vs 0.52 with other rats), but its resting spot is shared: in nearly every chunk 1-5 other rats also rest
  within 4% of the diagonal of it (631 in 9 chunks; 629, 634, 635 in 5-6 each). It is a hub, not a territory.
- Caveats: chunks span 09:59-20:29, so time of day is confounded; occupancy is arena-wide; home base is one
  argmax on a 30x30 grid; no thresholds were set.

## 2. Neurons: the exclusivity GLM was degenerate for rat630
- Gates on 20251216_145034 (chunks 1459/1659/1829, 136 cells): exclusivity replication median r=0.86 and position
  control 46% of cells both passed, but only because the rat sits in the same few bins every chunk.
- Exploratory GLM: 0 cells significant at q<.05 in each chunk (FDR resolvable, 2720 draws, best q 0.05);
  17/408 nominal p<=.05 vs ~20 expected; effect estimates did not replicate (r = 0.20, -0.09, -0.49).
- Diagnostic plot (`results/territory/20251216_145034_630_exploratory/example_cells.png`): in 1659 the rat occupies
  2-3 bins and exclusivity spans 0.38-0.46, so the "effect" is a rate difference between three positions;
  rate-vs-exclusivity is V-shaped for the top cells.
- Fix added: `exclusivity_spread_gate` in `ephys/run_territory.py` (eff_bins >= 10 and exclusivity 5-95% range
  >= 0.2 in every chunk; provisional). Rerun on 630: eff_bins 3.4-6.2, range 0.03-0.14, fails all chunks.
- Earlier rat631 run (20251216, two chunks, gates ignored) was also null and is likewise uninformative.

## 3. Social context at a fixed spot (`scripts/territory_coverage_table.py`, `scripts/hub_event_counts.py`)
Arrival/departure of a partner while the focal rests (strict: 30 s before, 10 s after; loose: 10 s / 5 s).

| Focal | Spot | Chunks with >=5 min rest | Best partner, strict | Best partner, loose |
|---|---|---|---|---|
| 630 | hub | 10 of 14 | 1 arrival / 1 departure total | 635: 7 / 1 |
| 631 | hub | 2 of 6 | 0 | 629: 1 / 0 |
| 630 | own home | 10 of 14 | <=1 | 635: 3 / 3 |
| 631 | own home | 3 of 6 | 0 | 635: 2 / 0 |
| 613 | own home | 9 of 10 | 634: 1 / 0 | 616: 4 / 6, 634: 5 / 2 |

Event-triggered analysis needs dozens per partner. Co-resting is long and stable (629 and 631 shared 31 and 18 min
with 630 on 20251216), so presence varies between chunks. Any remaining design is a between-chunk comparison that
confounds identity with drift and behavioural/arousal state (huddling, sleep). Hub co-occupancy decoding also
overlaps `decode_partner_distance` (see `docs/TERRITORY_HUB_SPEC.md`, draft, not pursued).

## Corrections to earlier statements in this work
- Rat630's afternoon recording is `20251216_145034`; the earlier gates runs for 630 used `144334`, which does not
  exist for that animal (those runs' "no ephys" was a wrong-stamp artifact).
- 20251217 was proposed as a confirmatory day because it has "more chunks"; it has three, so confirmation there
  rests on little.
- "Owner-present" does not apply to the hub: it is shared, so there is no owner whose presence can be decoded.

## Coverage (rat630, full ephys overlap, >=5 min hub rest)
20251212 (111544): 1559. 20251215 (161815): 1659. 20251216: 094334 -> 1329; 145034 -> 1459, 1659, 1829;
194334 -> 2029. 20251217: 095130 -> 1129; 145130 -> 1529, 1929. `20251209_160716` fails to load (missing
`*.timestamps.dat`). Table: `results/territory_behavior/coverage_630.csv` (also 631, 613).

## What would change the conclusion
- More 30-min APT chunks per day (or other days with several), so territory stability and company changes can be
  tested within a day.
- A rat that moves between groups within a chunk, or finer-grained company changes.
- Verified tracking attachment per recording (`tracking.attachment_status`, HZ-DATA-008); my overlap numbers use
  the DIO sync but the manifest is stale for tracking until a `--probe-level full` rebuild.
- If pursued: reserve holdouts, declare the family (`declare_family_tests`), and freeze predictions before looking
  at any confirmatory day.

## Artifacts
Code: `video/territory.py`, `video/territory_behavior.py`, `ephys/territory_encoding.py`,
`ephys/territory_population.py`, `ephys/run_territory.py`, `scripts/territory_behavior_screen.py`,
`scripts/territory_coverage_table.py`, `scripts/hub_event_counts.py`, `scripts/territory_cell_examples.py`;
tests under `tests/test_territory*.py`, `tests/test_run_territory.py`. Results (git-ignored) in
`results/territory/` and `results/territory_behavior/`.
