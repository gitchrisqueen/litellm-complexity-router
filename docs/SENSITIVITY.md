# Threshold sensitivity

`results/sensitivity-2026-09-04.json` is the committed output of
`python -m evals.sensitivity --write`. This page says what it measures, what
it does not, and how to read the numbers in it.

## What it measures

The scorer produces one number per prompt that does not depend on the cut
points, so every row is scored once and each candidate setting of
`TIER_BOUNDARIES` only re-maps those scores to tiers (then applies the
tool-bearing floor). Two sweeps are run:

- **one-at-a-time** — each of `simple_medium` (0.01–0.08), `medium_complex`
  (0.04–0.20) and `complex_reasoning` (0.15–0.60) over its grid, the other two
  held at the shipped values `0.05 / 0.09 / 0.25`;
- **joint** — every valid combination of the three grids
  (`0 <= a < b < c <= 1`), sorted by accuracy.

Accuracy and the two directional-error counts are computed on the **dev split
only** (20 rows). The **test split is frozen and is never consulted** by this
script; picking a setting on test would turn the headline into a fit
statistic (`CONTRIBUTING.md`). Train is not used either, so the numbers are
comparable with the dev figure in `results/latest.json`.

The realised score distribution is label-free, so it is reported for every
split and for all 120 rows: min, max, quantiles, the share of rows scoring
exactly zero, the share below the `0.25` cut, and a histogram with edges at
the shipped boundaries.

## What it does not do

- It does **not** change `TIER_BOUNDARIES`. The shipped values are marked in
  the output (`"shipped": true`) and the joint best is labelled
  `joint_best_dev`, not "recommended".
- It does not tune anything. With 20 dev rows one row is 5 accuracy points,
  so the differences between neighbouring grid settings are noise-level; the
  artifact exists to show the shape of the surface, not to pick a point on it.
- It does not report test accuracy at any setting.

## Reading the 2026-09-04 run

Shipped boundaries on dev: accuracy **0.40** (8 of 20), 0 SIMPLE rows routed
above COMPLEX, 3 REASONING rows routed below COMPLEX.

One-at-a-time: moving `simple_medium` anywhere in 0.02–0.08 leaves dev
accuracy at 0.40; only 0.01 reaches 0.50, by pushing near-zero-scoring MEDIUM
rows out of SIMPLE. Moving `medium_complex` from 0.06 to 0.09 leaves 0.40;
0.10 and above give 0.35. Moving `complex_reasoning` between 0.15 and 0.30
leaves 0.40; 0.35 and above give 0.35. Every column is flat or nearly flat:
the cut points are not what limits accuracy on this set.

Joint best on dev is `0.01 / 0.04 / 0.15` at **0.50** — a one-row-per-five-points
improvement obtained by compressing all three cuts toward zero, which is the
sweep saying the scores themselves are compressed.

Realised score distribution, all 120 rows: minimum 0.000, **median 0.013**,
95th percentile 0.346, maximum 0.514; **22% of rows score exactly 0.0** and
**85% score below the 0.25 cut**. The README states the achievable maximum
(0.95, the sum of the positive weights) and this artifact measures the
realised spread: the bulk of the labeled set sits in the bottom 3% of the
achievable range, and the three lower tiers share a band of width 0.25. That
is the dynamic-range question this artifact exists to face, and the
answer is that on this dataset the score is compressed near zero, largely
because prompts without code, reasoning keywords or domain terms — including
competition-math problems — score only their length.

## Reproducing

```bash
python -m evals.sensitivity            # print the summary
python -m evals.sensitivity --write    # write results/sensitivity-<date>.json
```

The output carries the same provenance block as the eval results (harness
SHA, dataset and split-assignment digests, thresholds digest, Python version).
