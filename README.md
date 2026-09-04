# litellm-complexity-router

A LiteLLM pre-call hook that scores each prompt's complexity on seven weighted dimensions and routes the request to one of four model tiers.

```
score = 0.10 * tokenCount          (words / 150, capped at 1)
      + 0.30 * codePresence        (code-pattern hits / 4)
      + 0.25 * reasoningMarkers    (reasoning-word hits / 3)
      + 0.25 * technicalTerms      (domain-pattern hits / 3)
      - 0.05 * simpleIndicators    (simple-phrase hits / 2; full penalty on short questions)
      + 0.03 * multiStepPatterns   (multi-step hits / 3)
      + 0.02 * questionComplexity  (0.15 per "?" + 0.10 per open wh-word, capped at 1)

tier  = SIMPLE     if score < 0.05
        MEDIUM     if score < 0.09
        COMPLEX    if score < 0.25
        REASONING  otherwise
floor = a request carrying tools is promoted to at least COMPLEX
```

Positive weights sum to 0.95, so the achievable maximum is 0.95 and three of the four tiers sit in the bottom ~26% of that range. Whether real traffic is correspondingly compressed is something the eval set measures, not something this README asserts.

**Tests:** 92 test functions in `tests/` (129 cases once parametrized), all written for this repository (`pytest`). Coverage of `scoring.py` and `tiers.py` is gated at 95% in CI.
**Eval accuracy (directional):** on the frozen 50-row test split the scoring stage lands **22 of 50 rows on the labeled tier — 0.44, Wilson 95% [0.31, 0.58]**; off-by-one rate 0.36. Split by where the labels come from: the **outside-anchored** half (25 rows from public datasets, labeled by each dataset's own level or category) scores **7/25 = 0.28 [0.14, 0.48]**; the synthetic half (25 rows labeled by the author's rubric) scores 15/25 = 0.60 [0.41, 0.77]. The 0.80 threshold is not met; it is reported, not enforced. Five anchored competition-math rows violate the "no REASONING row below COMPLEX" rule and are published as known failures rather than relabeled. Details: `results/latest.json`, `evals/datasets/SOURCES.md`, and "What the anchored subset shows" below.

**Extraction disclosure.** This code was extracted from a private, single-operator production hook the author runs in front of his own agent tooling. It was rewritten here as a standalone library with a fresh git history and is maintained independently; the deployment, its proxy configuration, its model list and its hostnames stay private and do not track this repository.

**Client work is never extracted.** Code delivered to a client — under a client's NDA, in the client's repository — is never extracted, excerpted or published in any form. This repository comes from the author's own system only.

**Costs are the harness's own.** Any cost or latency figure that ever appears in this repository describes the eval harness's own runs. None describes the production router, which holds no cost figure.

## Install and use

```bash
pip install git+https://github.com/gitchrisqueen/litellm-complexity-router
```

As a library:

```python
from complexity_router import RouterConfig, decide

cfg = RouterConfig()  # tier-router -> tier-simple/-medium/-complex/-reasoning
request = {
    "model": "tier-router",
    "messages": [{"role": "user", "content": "Prove there are infinitely many primes."}],
}
d = decide(request, cfg)
round(d.score, 4), d.scored_tier, d.tier, d.target, d.effort
# (0.0873, 'MEDIUM', 'MEDIUM', 'tier-medium', None)
# 6 words -> 0.10 * 6/150 = 0.0040; one reasoning word ("prove") -> 0.25 * 1/3 = 0.0833
```

As a LiteLLM proxy hook — see `examples/config.yaml` for a complete illustrative config:

```yaml
litellm_settings:
  callbacks:
    - complexity_router.hook.complexity_router_hook
complexity_router:
  router_model_name: tier-router
  tier_models: {SIMPLE: tier-simple, MEDIUM: tier-medium, COMPLEX: tier-complex, REASONING: tier-reasoning}
  strip_patterns: ["<injected-context[^>]*>.*?</injected-context>"]
```

```bash
COMPLEXITY_ROUTER_CONFIG=examples/config.yaml litellm --config examples/config.yaml
```

Clients call `model: tier-router`. The hook scores the last user message, picks a tier, writes the tier's model into `model`, injects `tool_choice` for tool-bearing requests when the caller set none, and records the decision under `metadata.complexity_router`. Requests for any other model pass through untouched. `RouterConfig` round-trips exactly through `to_dict` / `from_dict` and YAML.

## What ships, what does not

Ships, each exercised by a test or eval family: the scorer (`scoring.py`), the tier map and the tool-bearing floor (`tiers.py`), the score-to-effort map as a pure function, the hook shell (`hook.py`), configuration loading (`config.py`), and an observer interface with a no-op default (`observability.py`).

Stays private in this first cut: the classifier layer that reassigns targets by lane, the frontier distribution pool, reasoning-effort injection into the request body, the context gate and truncation, compaction routing, and every analytics integration. They are named here because the deployed router applies them **on top of** the score, which bounds what the evals below can claim.

## Evals

Three free families run on every push to `main` and on every pull request. No model is called and nothing costs money.

| Family | What it measures | Gate |
|---|---|---|
| **E1** routing accuracy | `score_complexity` → `score_to_tier` → floor, as pure functions, against 120 labeled rows (50 train / 20 dev / 50 frozen test; 50% outside-anchored in every split); exact-tier accuracy on the frozen `test` split with a Wilson 95% interval, off-by-one rate, confusion matrix, the anchored and synthetic subsets separately, accuracy per source, every miss with its rationale | **Hard:** the two directional rules — no SIMPLE-labeled row routed above COMPLEX, no REASONING-labeled row routed below COMPLEX — against a published `known_failures` list in `evals/thresholds.yaml` (any violation not on the list fails). **Hard:** dataset composition (≥120 rows, ≥50/20/50, ≥40% anchored per split, ≥20 anchored test rows). **Reported only:** the 0.80 exact-accuracy threshold, with its interval. |
| **E4** frozen regression | The *regression corpus* (every E1 row, all splits) against `results/baseline.json` | Fails on a drop of more than 2 accuracy points over the rows both sides share; lists every row whose predicted tier changed. Defensible only because the scorer is deterministic; the same rule is never reused on a sampled eval. |
| **E5** malformed and adversarial input | Requests with no user turn, empty or whitespace content, image-only content, an injection-only message (expected non-trigger), one enormous token, non-English text, a prompt that names its own tier, tools with empty text | Every row must produce its pinned tier without raising. These are behaviour pins, not findings. The extracted code returns the lowest tier for empty text (`EMPTY_TEXT_SCORE = 0.0` → SIMPLE; with tools, the floor lifts it to COMPLEX); `CHANGELOG.md` records the change from the inherited 0.3. |

**What E1 does not measure.** In the deployed hook the score is computed and then repeatedly overridden: the tool floor, a classifier override by lane, a frontier pool picked by correlation-id hash, and rollout variants. For agentic traffic the score barely selects anything. E1 measures **the scoring stage**, and every claim about it is scoped to that.

**The circularity problem, and the four controls.** The scorer is a keyword counter over four regex lists; any rubric written after reading the source shares its intuitions, so accuracy against such a set is a fit statistic. Four controls break the loop, and this README states which are in place:

1. **Train / dev / frozen-test split**, assigned per row and committed, digested into every results file (`split_assignment_sha256`); `DIMENSION_WEIGHTS` and `TIER_BOUNDARIES` may only change against train/dev (`CONTRIBUTING.md`). *In place.* The headline is reported on `test` only, with its interval.
2. **Blind labeling** — raters see `{prompt, has_tools}` and nothing else, two passes, quadratic-weighted kappa. *Deferred:* the synthetic rows were labeled in a single pass by a rater who had read the pattern lists, and the rows say so; no agreement statistic exists. The anchored rows do not depend on that rater: their label is a function of the source dataset's own field under a mapping committed before sampling (`SOURCES.md`).
3. **An outside-anchored subset** whose labels come from something other than the heuristic, reported separately. *In place:* 60 of 120 rows, 50% of every split, from five public datasets with verified licenses; reported separately with its gap to the synthetic subset.
4. **A committed threshold-sensitivity sweep** across all three cut points, publishing the realized score distribution. *In place:* `results/sensitivity-2026-09-04.json`, read via `docs/SENSITIVITY.md`; dev split only, boundaries unchanged.

Until blind double labeling with kappa lands (control 2), the accuracy number is directional; the hard gates are the directional rules (against the published known-failures list), the dataset-composition checks, E4 and E5.

**What the anchored subset shows.** On the frozen test split every MATH, GSM8K and HumanEval row misses (0 of 12), while ARC and Dolly rows land at 0.5–0.67 and the synthetic rows at 0.60. Competition-math problems carry none of the words the scorer counts — no code pattern, no reasoning verb, no domain term — and many open with "what is" or "how many", which triggers the simple-question penalty, so a Level-5 problem scores 0.0 and routes to SIMPLE. Five such test rows violate the "no REASONING row below COMPLEX" rule; they are listed in `evals/thresholds.yaml` under `known_failures` with the rule they break, the gate fails on any addition, and neither the labels nor the boundaries were moved to make them pass. The same pattern holds across all splits (13 REASONING-labeled rows below COMPLEX in the 120, all MATH Level 4–5). The sensitivity sweep says the cut points are not the cause: dev accuracy is flat across the grid, and 85% of all rows score below the 0.25 cut (median 0.013). That is the result the anchored subset exists to produce — a keyword scorer measures vocabulary, not difficulty — and it bounds what this router can claim for traffic that does not announce its own complexity.

**Provenance in every result.** `results/latest.json`, each dated run file and the sensitivity artifact carry `schema_version`, the harness git SHA, the SHA-256 of both datasets, the split-assignment digest, the thresholds digest, the config digest, the Python version and `cost_usd: 0.0`.

**The harness is a shape, not a deliverable.** The method here is free and MIT so that anyone can copy it. What a client engagement buys is what this repository cannot contain: replay cases recorded from their own conversations, their tool schemas and failure modes, thresholds set against their traffic, tier boundaries tuned to their prompts, wiring into their CI, and a written hand-off.

## Redaction, enforced in CI

The repository was built private, redacted, and checked before any public flip.

- `scripts/denylist.sha256` holds SHA-256 hashes of denied terms (the private system's name, hostnames, tracker ids, every internal model alias, and more). It is generated by rule from the private sources by `scripts/generate_denylist.py`, which runs only on the maintainer's machine; the committed artifact contains hashes only.
- `.github/workflows/denylist-guard.yml` hashes every candidate string in the working tree **and** the full history (`git log --all -p`, commit messages, author lines, file paths, refs) and compares. Candidates are extracted after NFKC normalisation, zero-width stripping and a look-alike fold, and include whole tokens, every separator-bounded span of a token with its `-`/`_`/joined variants, every bare word and every substring of it (no length cap), and the decoded contents of base64 and hex runs. Its output is a per-hash zero/non-zero table; it never prints matched text, so a run URL is safe to cite. Remaining limits are listed in `CONTRIBUTING.md`.
- `.github/workflows/secret-scan.yml` runs gitleaks over the full history.
- Limitation, stated plainly: CI cannot regenerate the denylist because the sources it is generated from are private. The maintainer regenerates it when those sources change.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy
pytest --cov=complexity_router
python -m evals.harness --family e1 e4 e5
python -m evals.sensitivity
python scripts/denylist_guard.py
```

Public rows are sampled by `scripts/sample_public_rows.py` (maintainer-run; needs the network) — never by hand.

## License

MIT — see `LICENSE`. `NOTICE` names LiteLLM as the framework this hook plugs into; LiteLLM is not vendored. Thirty of the anchored dataset rows are redistributed under their source licenses (CC BY-SA 3.0 for Dolly, CC BY-SA 4.0 for ARC) with attribution in `evals/datasets/SOURCES.md`; each such row's `license` field says so.
