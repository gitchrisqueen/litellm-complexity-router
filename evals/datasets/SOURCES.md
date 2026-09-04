# Dataset sources

Every row in `evals/datasets/*.jsonl` carries a `source` field. The harness
refuses a row without one. Allowed values: `synthetic`, or `public:<name>`
where `<name>` is a heading in this file with a verified license line.

**Never production traffic.** No row here was taken from a live system,
a client, a prospect, or any logged conversation.

## Current status: seed

| Dataset | Rows | Source | Rater | Anchored rows |
|---|---|---|---|---|
| `e1_routing.jsonl` | 36 | `synthetic` | one pass by the extracting agent, **not blind** (the rater had read the scorer's pattern lists) | 0 |
| `e5_malformed.jsonl` | 9 | `synthetic` | same | n/a (behaviour pins, not labels) |

This is a **seed**, not the labeled set the accuracy headline depends on. The
planned set is 120 rows minimum, split 50 / 20 / 50 (train / dev / test), with
at least 40% of every split anchored to something other than the heuristic
(public benchmark items carrying the benchmark's own difficulty tier, or rows
whose ground truth is the cheapest model that solves the task). Until that set
lands, `thresholds.yaml` leaves the exact-accuracy gate unenforced and the
README reports no accuracy headline.

## Row schema (E1)

```json
{"id": "e1-001", "prompt": "...", "has_tools": false, "expected_tier": "SIMPLE",
 "source": "synthetic", "split": "test", "anchored": false,
 "rater": "agent-seed-not-blind", "rationale": "why this tier"}
```

`split` is assigned per row and committed. `DIMENSION_WEIGHTS` and
`TIER_BOUNDARIES` may only be changed against `train` / `dev` rows
(see `CONTRIBUTING.md`); the headline is reported on `test` only.

## Row schema (E5)

```json
{"id": "e5-001", "class": "no-user-message", "request": {"messages": []},
 "expected_tier": "REASONING", "expect_empty_branch": true,
 "source": "synthetic", "note": "..."}
```

E5 rows pin the router's current behaviour on an input class so a change is
visible in CI. They are behaviour pins, not claims about what the behaviour
should be.

## Public datasets

None yet. When one is added: name, URL, license (verbatim), row count sampled,
and the sampling method go here before any row referencing it is committed.
