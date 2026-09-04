# Contributing

## The one rule that protects the accuracy number

`DIMENSION_WEIGHTS` (`src/complexity_router/scoring.py`) and `TIER_BOUNDARIES`
(`src/complexity_router/tiers.py`) may only be changed against the `train` and
`dev` splits of `evals/datasets/e1_routing.jsonl`. The `test` split is frozen:
it is never inspected while tuning and never relabeled to fit a change. A pull
request that touches either constant must:

1. say which `train`/`dev` rows motivated it;
2. include the E1 output before and after (`python -m evals.harness --family e1`);
3. refresh `results/baseline.json` in the same PR (`--write-baseline`) and
   explain every row in E4's `changed_rows`;
4. add a line to `CHANGELOG.md` recording the old and new values.

Without this rule the headline is a fit statistic dressed as a generalisation
claim, which is exactly what the original private tuning produced (see
`CHANGELOG.md`, "Provenance of the cut points").

## Adding dataset rows

- Every row carries `source: synthetic` or `source: public:<name>`; a public
  name needs a heading with a verified license line in
  `evals/datasets/SOURCES.md` first.
- Never a row from production traffic, a client, a prospect, or a log.
- Assign `split` when the row is written and do not move it afterwards.
- `anchored: true` only when the label comes from outside the heuristic
  (a benchmark's own difficulty tier, or "the cheapest model that solves it").
- Label `{prompt, has_tools}` before running the scorer on the row, and say who
  labeled (`rater`). A rater who has read the pattern lists is not blind; say so.

## Redaction

`scripts/denylist.sha256` is a list of hashes. If the guard fails on your
change, the change contains a term the maintainer has denied; do not try to
learn which. Rephrase, or open an issue describing the change without the term.

## Checks that must be green

```
ruff check . && ruff format --check . && mypy
pytest --cov=complexity_router
python -m evals.harness --family e1 e4 e5
python scripts/denylist_guard.py
```
