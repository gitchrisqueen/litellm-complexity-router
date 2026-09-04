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

**Tests:** 78 test functions in `tests/` (115 cases once parametrized), all written for this repository (`pytest`). Coverage of `scoring.py` and `tiers.py` is gated at 95% in CI.
**Eval accuracy:** not yet reported as a headline. The committed set is a 36-row **seed** (18 frozen test rows, one not-blind rater, no outside-anchored rows). On that seed the scoring stage lands 10 of 18 test rows on the labeled tier — 0.556, Wilson 95% interval [0.34, 0.75] — with both directional rules holding. See `results/latest.json` and `evals/datasets/SOURCES.md` for what the number is and is not.

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
| **E1** routing accuracy | `score_complexity` → `score_to_tier` → floor, as pure functions, against labeled rows; exact-tier accuracy on the frozen `test` split with a Wilson 95% interval, off-by-one rate, confusion matrix, every miss with its rationale | Two absolute directional rules: no SIMPLE-labeled row routed above COMPLEX, no REASONING-labeled row routed below COMPLEX. The 0.80 exact-accuracy threshold is computed but **unenforced** while the set is a seed. |
| **E4** frozen regression | The *regression corpus* (every E1 row, all splits) against `results/baseline.json` | Fails on a drop of more than 2 accuracy points; lists every row whose predicted tier changed. Defensible only because the scorer is deterministic; the same rule is never reused on a sampled eval. |
| **E5** malformed and adversarial input | Requests with no user turn, empty or whitespace content, image-only content, an injection-only message (expected non-trigger), one enormous token, non-English text, a prompt that names its own tier, tools with empty text | Every row must produce its pinned tier without raising. These are behaviour pins, not findings. |

**What E1 does not measure.** In the deployed hook the score is computed and then repeatedly overridden: the tool floor, a classifier override by lane, a frontier pool picked by correlation-id hash, and rollout variants. For agentic traffic the score barely selects anything. E1 measures **the scoring stage**, and every claim about it is scoped to that.

**The circularity problem, and the four controls.** The scorer is a keyword counter over four regex lists; any rubric written after reading the source shares its intuitions, so accuracy against such a set is a fit statistic. Four controls break the loop, and this README states which are in place:

1. **Train / dev / frozen-test split**, assigned per row and committed; `DIMENSION_WEIGHTS` and `TIER_BOUNDARIES` may only change against train/dev (`CONTRIBUTING.md`). *In place.* The headline is reported on `test` only, with its interval.
2. **Blind labeling** — raters see `{prompt, has_tools}` and nothing else. *Deferred:* the seed was labeled in a single pass by a rater who had read the pattern lists, and the rows say so.
3. **An outside-anchored subset** whose labels come from something other than the heuristic, reported separately. *Deferred:* zero anchored rows in the seed; the planned set carries at least 40% anchored rows in every split.
4. **A committed threshold-sensitivity sweep** across all three cut points, publishing the realized score distribution. *Deferred.*

Until 2–4 land, the accuracy number is directional and the only hard gates are the directional rules and E4.

**Provenance in every result.** `results/latest.json` and each dated run file carry `schema_version`, the harness git SHA, the dataset SHA-256, the config digest, the Python version and `cost_usd: 0.0`.

**The harness is a shape, not a deliverable.** The method here is free and MIT so that anyone can copy it. What a client engagement buys is what this repository cannot contain: replay cases recorded from their own conversations, their tool schemas and failure modes, thresholds set against their traffic, tier boundaries tuned to their prompts, wiring into their CI, and a written hand-off.

## Redaction, enforced in CI

The repository was built private, redacted, and checked before any public flip.

- `scripts/denylist.sha256` holds SHA-256 hashes of denied terms (the private system's name, hostnames, tracker ids, every internal model alias, and more). It is generated by rule from the private sources by `scripts/generate_denylist.py`, which runs only on the maintainer's machine; the committed artifact contains hashes only.
- `.github/workflows/denylist-guard.yml` hashes every candidate token in the working tree **and** the full history (`git log --all -p`, commit messages, author lines, file paths, refs) and compares. Its output is a per-hash zero/non-zero table; it never prints matched text, so a run URL is safe to cite.
- `.github/workflows/secret-scan.yml` runs gitleaks over the full history.
- Limitation, stated plainly: CI cannot regenerate the denylist because the sources it is generated from are private. The maintainer regenerates it when those sources change.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy
pytest --cov=complexity_router
python -m evals.harness --family e1 e4 e5
python scripts/denylist_guard.py
```

## License

MIT — see `LICENSE`. `NOTICE` names LiteLLM as the framework this hook plugs into; LiteLLM is not vendored.
