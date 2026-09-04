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
- Assign `split` when the row is written and do not move it afterwards. The
  split assignment is digested into every results file; a moved row changes
  the digest.
- `anchored: true` only when the label comes from outside the heuristic
  (a benchmark's own difficulty tier or category under the committed mapping
  in `SOURCES.md`, or "the cheapest model that solves it"). Anchored rows
  must be `public:*` and carry an `anchor` block naming the dataset,
  revision, config, split, row index, field and value the label came from.
  Public rows are sampled by `scripts/sample_public_rows.py`, never by hand.
- Label `{prompt, has_tools}` before running the scorer on the row, and say who
  labeled (`rater`). A rater who has read the pattern lists is not blind; say so.
- A row that violates a directional rule is **not** relabeled and the
  boundaries are **not** moved to fit it. If the label follows the rubric,
  add the row to `known_failures` in `evals/thresholds.yaml` with the rule
  it violates, in the same PR, so the gate stays hard for anything new.
- The dataset-composition gates in `evals/thresholds.yaml` (row count, split
  sizes, anchored share per split) are enforced; a PR that lowers any of
  them must say why.

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

## Known limits of the denylist guard

Stated so nobody over-trusts a green run:

- Normalisation is NFKC plus zero-width/soft-hyphen removal plus a small
  Cyrillic/Greek look-alike table. Look-alikes outside that table, and text
  in images or binary formats, are not matched.
- Base64 and hex runs are decoded one level deep; compressed, encrypted or
  doubly-encoded content is not.
- Bare-word substrings are enumerated only for words up to 512 characters;
  longer unbroken runs are treated as blobs and reach the decoders instead.
- CI cannot regenerate `scripts/denylist.sha256`; the sources are private and
  the maintainer regenerates it by hand when they change.
- GitHub-side surfaces (repository description, topics, issues, releases) are
  not in the tree or the history and are checked by hand before a public flip.
