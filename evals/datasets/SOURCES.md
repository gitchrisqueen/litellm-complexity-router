# Dataset sources

Every row in `evals/datasets/*.jsonl` carries a `source` field. The harness
refuses a row without one. Allowed values: `synthetic`, or `public:<name>`
where `<name>` is a heading in this file with a verified license line.

**Never production traffic.** No row here was taken from a live system,
a client, a prospect, or any logged conversation.

## Current status: the 120-row labeled set (2026-09-04)

| Dataset | Rows | Sources | Rater | Anchored rows |
|---|---|---|---|---|
| `e1_routing.jsonl` | 120 | 60 `synthetic`, 60 `public:*` (six headings below) | synthetic rows: one pass by the extracting agent, **not blind** (`agent-seed-not-blind` for the 36 seed rows, `agent-single-pass-not-blind` for the 24 added rows); anchored rows: `anchor-mapping-v1`, a pure function of the source dataset's own field (the rater did not read the prompt to choose the tier) | 60 (50% of every split) |
| `e5_malformed.jsonl` | 9 | `synthetic` | same | n/a (behaviour pins, not labels) |

Composition per split (the committed sizing: 50 / 20 / 50, at least 40% anchored
in every split, at least 20 anchored rows in the frozen test split):

| Split | Rows | Anchored | Anchored share | `synthetic` | `public:math` | `public:dolly-15k` | `public:arc-easy` | `public:arc-challenge` | `public:gsm8k` | `public:humaneval` |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 50 | 25 | 50% | 25 | 8 | 6 | 3 | 3 | 2 | 3 |
| dev | 20 | 10 | 50% | 10 | 4 | 3 | 1 | 1 | 1 | 0 |
| test | 50 | 25 | 50% | 25 | 8 | 8 | 3 | 2 | 2 | 2 |
| **all** | **120** | **60** | **50%** | 60 | 20 | 17 | 7 | 6 | 5 | 5 |

Split assignment digest (sorted `id:split` lines, SHA-256) is written into
every results file as `split_assignment_sha256`. The 36 seed rows keep the
splits they were committed with; the 24 new synthetic rows were assigned at
write time; the 60 anchored rows are assigned by the sampler's fixed pattern.
No row has been moved.

**Single rater, no agreement statistic.** Phase 1 runs one labeling pass. The
headline therefore rests on a single-rater label set with no kappa; blind
double labeling is deferred (see `README.md`, "the four controls"). The
anchored rows are the part of the set that does not depend on that rater's
judgement: their label is fixed by the source dataset's field and the mapping
below, both committed before any row was scored.

## Labeling rubric (frozen before scoring; written by a rater who had read the pattern lists)

- **SIMPLE** — a greeting or acknowledgement; a single-fact lookup or recall;
  a one-line translation, spelling or definition; a yes/no or one-choice
  answer from common knowledge. One retrieval step, no generation beyond a
  phrase.
- **MEDIUM** — short generation or transformation with no analysis: rewrite,
  short summary, a short poem or message, a short list of ideas or names, a
  two-or-three-sentence explanation, simple arithmetic with a formula, a
  choice that needs two facts combined.
- **COMPLEX** — a task that needs code, a multi-step procedure, a multi-step
  calculation, or a technical explanation, diagnosis, comparison or design.
  Any tool-bearing request is at least COMPLEX (the floor).
- **REASONING** — a proof, competition-level mathematics, an analysis of
  trade-offs across options with a recommendation, a critique of a hypothesis
  or causal claim, or a theoretical argument.

## Anchor mapping v1 (`anchor-mapping-v1`)

The tier of an anchored row is this function of the source's own field, fixed
before sampling and never adjusted afterwards:

| Source | Field | Value | Tier |
|---|---|---|---|
| `public:arc-easy` | config | `ARC-Easy` | SIMPLE |
| `public:arc-challenge` | config | `ARC-Challenge` | MEDIUM |
| `public:dolly-15k` | category | `open_qa`, `classification` | SIMPLE |
| `public:dolly-15k` | category | `brainstorming`, `creative_writing`, `general_qa` | MEDIUM |
| `public:gsm8k` | config | `main` | COMPLEX |
| `public:humaneval` | task | every task | COMPLEX |
| `public:math` | level | `Level 1`, `Level 2` | COMPLEX |
| `public:math` | level | `Level 4`, `Level 5` | REASONING |

MATH `Level 3` is not sampled: it sits on the COMPLEX/REASONING boundary and
an anchor should not be ambiguous. Dolly rows are sampled only from the five
context-free categories (no `closed_qa`, `summarization`,
`information_extraction`, whose prompts depend on an attached passage).

Sampling (`scripts/sample_public_rows.py`, seed `20260904`): candidates are
read from the Hugging Face datasets-server JSON API (the first 100–200 rows
of each config's split in the dataset's own order; Dolly from the raw JSONL),
filtered — no `@`, no URL, no run of seven or more digits, word count within
the per-source cap (40 for ARC and Dolly, 60 for GSM8K and MATH, 90 for
HumanEval), no `[asy]` diagram blocks, and for the non-code sources no
capitalised word outside a short allowlist of sentence starters and common
nouns (a crude proxy for "no personal names", which also drops sentence-initial
first names) — then sampled with the fixed seed. Every anchored row carries an
`anchor` block: `dataset`, `revision` (the Hugging Face repo SHA at retrieval),
`config`, `split`, `row_idx`, `field`, `value`, `mapping`. Retrieval:
2026-09-04T16:23Z (licenses) and 2026-09-04T16:23Z (rows).

**Licenses of the anchored rows.** The repository is MIT. Rows sampled from
MIT datasets are redistributed under MIT. The 17 Dolly rows are redistributed
under **CC BY-SA 3.0** and the 13 ARC rows under **CC BY-SA 4.0**, with
attribution below; each row's `license` field names its license. Nothing else
in the repository is affected.

## Public datasets

Each heading is a `source` name. Each license was checked at fetch time by
reading the dataset card's `license:` field, the Hugging Face API license tag,
and — where one exists — the upstream `LICENSE` file, on 2026-09-04.

### `public:math` — MATH (Hendrycks et al.)

- Hugging Face: https://huggingface.co/datasets/EleutherAI/hendrycks_math
  (revision `21a5633873b6a120296cce3e2df9d5550074f4a3`); upstream
  https://github.com/hendrycks/math
- License: card `license: mit`; API tag `license:mit`; upstream `LICENSE`
  fetched from https://raw.githubusercontent.com/hendrycks/math/main/LICENSE —
  "MIT License / Copyright (c) 2021 Dan Hendrycks".
- Rows sampled: 20 (configs `algebra`, `prealgebra`, `number_theory`,
  `counting_and_probability`, `geometry`; split `test`; 5 at Level 1–2, 15 at
  Level 4–5, one and three per config respectively). Field used: `level`.

### `public:gsm8k` — GSM8K (OpenAI)

- Hugging Face: https://huggingface.co/datasets/openai/gsm8k (revision
  `740312add88f781978c0658806c59bc2815b9866`); upstream
  https://github.com/openai/grade-school-math
- License: card `license: - mit` and the card's text "The GSM8K dataset is
  licensed under the MIT License"; API tag `license:mit`; upstream `LICENSE`
  fetched from
  https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE —
  "MIT License / Copyright (c) 2021 OpenAI".
- Rows sampled: 5 (config `main`, split `test`). Field used: the config
  (every row is a multi-step word problem).

### `public:humaneval` — HumanEval (OpenAI)

- Hugging Face: https://huggingface.co/datasets/openai/openai_humaneval
  (revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`); upstream
  https://github.com/openai/human-eval
- License: card `license: - mit`; API tag `license:mit`; upstream `LICENSE`
  fetched from https://raw.githubusercontent.com/openai/human-eval/master/LICENSE
  — "The MIT License / Copyright (c) OpenAI".
- Rows sampled: 5 (config `openai_humaneval`, split `test`), each prefixed
  with "Complete this Python function:" so the row is a request rather than a
  bare stub. Field used: `task_id` (every task is a code-completion task).

### `public:arc-easy` and `public:arc-challenge` — AI2 Reasoning Challenge (Allen Institute for AI)

- Hugging Face: https://huggingface.co/datasets/allenai/ai2_arc (revision
  `210d026faf9955653af8916fad021475a3f00453`); upstream
  https://allenai.org/data/arc
- License: card `license: - cc-by-sa-4.0`; API tag `license:cc-by-sa-4.0`.
  The allenai.org data page was fetched (HTTP 200) but renders its license
  text client-side, so the card and the API tag are the verified sources.
  Rows are redistributed under CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/) with this attribution.
- Rows sampled: 7 from config `ARC-Easy`, 6 from config `ARC-Challenge`
  (split `test`), each with its answer choices appended as `(A) … (B) …`.
  Field used: the config.

### `public:dolly-15k` — databricks-dolly-15k (Databricks)

- Hugging Face: https://huggingface.co/datasets/databricks/databricks-dolly-15k
  (revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`)
- License: card `license: cc-by-sa-3.0`; card text "This dataset was developed
  at Databricks (https://www.databricks.com) and its use is subject to the
  CC BY-SA 3.0 license" and a link to
  https://creativecommons.org/licenses/by-sa/3.0/legalcode; API tag
  `license:cc-by-sa-3.0`. Rows are redistributed under CC BY-SA 3.0 with this
  attribution.
- Rows sampled: 17 (`open_qa` 5, `classification` 3, `brainstorming` 3,
  `creative_writing` 3, `general_qa` 3), from rows with an empty `context`.
  Field used: `category`.

### Considered and not used

- **OpenAssistant oasst1** (Apache-2.0, verified from the card): not used —
  it carries no difficulty or category field to anchor to, and its prompts
  are user-written and often personal.
- **LMSYS-Chat-1M**: the dataset card returned HTTP 401 (gated); the license
  could not be verified at fetch time, so it does not ship.
- **Alpaca** (`tatsu-lab/alpaca`, `license: cc-by-nc-4.0` on the card):
  non-commercial; not acceptable for redistribution in an MIT repository.

## Row schema (E1)

```json
{"id": "e1-078", "prompt": "...", "has_tools": false, "expected_tier": "COMPLEX",
 "source": "public:math", "split": "test", "anchored": true,
 "anchor": {"dataset": "EleutherAI/hendrycks_math", "revision": "21a5633…",
            "config": "prealgebra", "split": "test", "row_idx": 12,
            "field": "level", "value": "Level 2", "mapping": "anchor-mapping-v1"},
 "license": "MIT", "rater": "anchor-mapping-v1",
 "rationale": "anchor-mapping-v1: EleutherAI/hendrycks_math level=Level 2 -> COMPLEX"}
```

Synthetic rows omit `anchor` and `license` and carry `"anchored": false`.
`split` is assigned per row and committed. `DIMENSION_WEIGHTS` and
`TIER_BOUNDARIES` may only be changed against `train` / `dev` rows
(see `CONTRIBUTING.md`); the headline is reported on `test` only.

## Row schema (E5)

```json
{"id": "e5-001", "class": "no-user-message", "request": {"messages": []},
 "expected_tier": "SIMPLE", "expect_empty_branch": true,
 "source": "synthetic", "note": "..."}
```

E5 rows pin the router's current behaviour on an input class so a change is
visible in CI. They are behaviour pins, not claims about what the behaviour
should be.
