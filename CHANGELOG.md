# Changelog

All notable changes to this project are recorded here. Dates are ISO-8601.

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
