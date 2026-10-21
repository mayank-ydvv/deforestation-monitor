# Change detection vs Hansen Global Forest Change

Hansen/UMD GFC-2025-v1.13 (loss years 2001-2025, 30 m) is used for validation only. Our pairs compare the same month a year apart; numbers come from `python -m training.evaluate --hansen ...` and are stored in `web/public/data/metrics.json` under `change_validation`.

| Pair | Hansen years | Detected | Hansen loss (≥ 0.5 ha) | Pixel precision | Pixel recall | Patch precision | Precision, any loss year |
|---|---|---:|---:|---:|---:|---:|---:|
| Hasdeo Jan 2025 → Jan 2026 | 2025 | 164.9 ha, 33 patches | 149.2 ha | **0.725** | **0.801** | 0.394 | 0.737 |
| Rondônia Jul 2024 → Jul 2025 | 2024, 2025 | 376.6 ha, 198 patches | 2,070.1 ha | 0.142 | 0.024 | 0.116 | **0.991** |

Definitions: pixel precision = share of our cleared pixels that Hansen marks as lost in those years; pixel recall = share of Hansen loss we flag (Hansen patches under 0.5 ha excluded, as they are below our minimum mapping unit); patch precision = share of our patches with at least 25 % of their area in Hansen loss; "any loss year" = share of our cleared pixels Hansen marks as lost in any year up to the last listed year.

## Hasdeo: agrees with Hansen

The detected patches trace the southern expansion of the Parsa East–Kete Basan mine and sit on Hansen's 2025 loss ([map](phase4_hasdeo_change.png)). All 8 patches of 2 ha or more match Hansen (60–99 % overlap) and hold 145 of 165 ha. The weak spot is small patches: 20 of the 25 under 2 ha do not match, which is what pulls patch precision down. Temporal confirmation (a patch must still be non-forest next month) targets exactly these.

## Rondônia: a definition mismatch, not a match failure

The map ([phase4_rondonia_change.png](phase4_rondonia_change.png)) shows two different kinds of change:

1. **Hansen's 2024–25 loss here is mostly fire-damaged canopy.** 2024 was a record Amazon fire year. On those pixels median NDVI only falls from 0.79 to 0.71, and the forest model still sees forest at t2 (2–9 % of them are non-forest). The spec defines our product as forest *cleared*, enforced by "model says non-forest at t2 and NDVI dropped > 0.2", so standing burned forest is out of scope; hence the low recall.
2. **99 % of what we flag Hansen also records as tree-cover loss, but in an earlier year (2005–2023).** Hansen keeps only the first loss since 2000, so clearing of regrown vegetation on old clearings never shows up as 2024–25 loss; hence the low same-year precision. Whether re-clearing of secondary vegetation counts as "deforestation" is a product decision; this pipeline reports it.

**Takeaways.** In a region of new clear-cutting (Hasdeo) the detector matches Hansen well by area. Against Hansen in the Amazon it should be read as "clear-cut and re-clearing detector", not "all tree-cover loss": it misses fire degradation and finds re-clearing Hansen does not count. Detecting degradation would need a different signal (e.g. NBR/SWIR change or Sentinel-1), listed as future work.
