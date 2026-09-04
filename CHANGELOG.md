# Changelog

All notable changes to this project are recorded here. Dates are ISO-8601.

## 0.2.0 - 2026-09-04

### Empty-text default lowered to 0.0

`EMPTY_TEXT_SCORE` is now `0.0`, so a request with no scorable text (no user
turn, empty or whitespace-only content, a content list with no text block)
lands in SIMPLE, the lowest tier, and a tool-bearing request with empty text is
promoted to COMPLEX by the floor. The inherited value was `0.3` (see 0.1.0
below), which resolved to REASONING under the shipped `0.25` cut. Reasoning:
nothing to score means nothing to pay for; a mid-scale default routed an empty
request to the most expensive tier. The value stays configurable
(`empty_text_score`); `tests/test_scoring.py` pins the new default and
`evals/datasets/e5_malformed.jsonl` pins the resulting tiers. This entry
describes the extracted library only; it makes no claim about any deployment.

### The labeled set: 120 rows, 60 outside-anchored

`evals/datasets/e1_routing.jsonl` grows from a 36-row seed to 120 rows, split
50 / 20 / 50 (train / dev / test) with the assignment committed per row and
digested into every results file (`split_assignment_sha256`). 60 rows are
outside-anchored: sampled deterministically from five public datasets whose
licenses were verified at fetch time (`evals/datasets/SOURCES.md`), with the
tier label a pure function of each dataset's own level or category field under
the committed mapping `anchor-mapping-v1`. The seed rows keep their original
split assignment. The 24 new synthetic rows were labeled by the rubric before
the scorer was run on them. Sampler: `scripts/sample_public_rows.py`.

### Harness

- Dataset-composition gates (row count, split sizes, anchored share per split,
  anchored rows in the frozen test split) are read from `evals/thresholds.yaml`
  and **enforced**. `min_test_rows: 50` is now enforced.
- The two directional rules stay hard gates. Violations on the frozen split
  are compared against a published `known_failures` list in
  `evals/thresholds.yaml`; any violation not on that list fails the run, and a
  listed row that starts passing is reported as resolved. The current list is
  five outside-anchored competition-math rows that the scoring stage sends to
  SIMPLE (see `README.md`, "What the anchored subset shows").
- Anchored and synthetic subsets of the test split are reported separately,
  with their gap and a per-source breakdown.
- E4 computes its drop over the rows the baseline and the current run share,
  so adding rows cannot mask or manufacture a regression. `results/baseline.json`
  is re-recorded for the 120-row corpus in this release.
- Every results file carries `thresholds_sha256`, both dataset digests and the
  split-assignment digest. Results schema is now version 2.

### Threshold sensitivity artifact

`evals/sensitivity.py` sweeps the three cut points one at a time and jointly,
reporting dev-split accuracy and directional-error counts only (the test split
is never consulted), plus the label-free realised score distribution per split.
Output: `results/sensitivity-<date>.json`; reading guide: `docs/SENSITIVITY.md`.
`TIER_BOUNDARIES` were not changed.

### Denylist guard hardening

`scripts/denylist_guard.py` now normalises text (NFKC, zero-width and
soft-hyphen removal, a Cyrillic/Greek look-alike fold) before extraction;
emits every separator-bounded span of an identifier-like token and each
span's canonicalised forms (`-`, `_`, joined); drops the 12-character cap on
bare-word substrings; and decodes base64 and hex runs before scanning them.
Output is unchanged: a per-hash zero/non-zero table and a summary line, never
matched text.

## 0.1.0 - 2026-09-04

First public-bound extraction. The scoring stage, the tier map, the
tool-bearing floor, the LiteLLM hook shell, configuration loading, and a no-op
observer interface, rewritten as a standalone package with a fresh history.
Not extracted (they remain private): the classifier layer, the frontier
distribution pool, reasoning-effort injection into requests, the context gate
and truncation, compaction routing, every analytics integration, and the
deployment's proxy configuration.

### Provenance of the cut points

`TIER_BOUNDARIES` ships as `0.05 / 0.09 / 0.25`. In the private source the
first cut carried the annotation `[was 0.02]` and the third `[tuned iter3]`:
the values were tuned by hand against data that was never recorded. They are
shipped here as inherited constants, not as values derived from the committed
dataset. `CONTRIBUTING.md` sets the rule for changing them from now on.

### Empty-text default

`score_complexity` returns `EMPTY_TEXT_SCORE = 0.3` when no scorable text can
be extracted (no user turn, empty or whitespace-only content, or a content list
with no text block). This is the inherited behaviour, kept as-is and made
configurable (`empty_text_score`). `tests/test_scoring.py` pins the value and
`evals/datasets/e5_malformed.jsonl` pins the resulting tier so any change is
visible. Whether the default should change is an open engineering question for
this repository; no claim about it is made in the README.

### Renames

- The virtual model the hook intercepts is `tier-router`.
- Tier targets are `tier-simple` / `tier-medium` / `tier-complex` / `tier-reasoning`.
- Injected-context stripping is a configurable `strip_patterns` list with one
  neutral default (`<injected-context>...</injected-context>`).
