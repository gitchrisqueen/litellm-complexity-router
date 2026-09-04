#!/usr/bin/env python3
"""Threshold sensitivity: sweep the three cut points, report on the dev split only.

Scores are a pure function of the prompt and never depend on the cut points,
so every row is scored once; each grid setting then re-maps those scores to
tiers and applies the tool floor. Accuracy and the two directional-error
counts are reported for the ``dev`` split only - the ``test`` split is frozen
and is never used to pick a setting (CONTRIBUTING.md). The realised score
distribution is label-free, so it is reported for every split.

Two sweeps:

- **one-at-a-time**: each cut point over its grid with the other two held at
  the shipped values;
- **joint**: the full grid over all three, keeping every setting that is
  valid (``0 <= a < b < c <= 1``).

Nothing here changes ``TIER_BOUNDARIES``. The output is a committed artifact
(``results/sensitivity-YYYY-MM-DD.json``) and ``docs/SENSITIVITY.md`` explains
how to read it.

Usage::

    python -m evals.sensitivity            # print a summary
    python -m evals.sensitivity --write    # also write results/sensitivity-<date>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from typing import Any

from complexity_router.config import RouterConfig
from complexity_router.tiers import TIER_BOUNDARIES, apply_tool_floor, score_to_tier, tier_index
from evals.harness import (
    DATASETS,
    RESULTS,
    RowResult,
    load_jsonl,
    provenance,
    score_rows,
    split_assignment_sha256,
)

GRID = {
    "simple_medium": [round(0.01 * i, 2) for i in range(1, 9)],  # 0.01 .. 0.08
    "medium_complex": [round(0.01 * i, 2) for i in range(4, 21)],  # 0.04 .. 0.20
    "complex_reasoning": [round(0.05 * i, 2) for i in range(3, 13)],  # 0.15 .. 0.60
}
HIST_EDGES = [0.0, 0.02, 0.05, 0.09, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0]


def evaluate(rows: list[RowResult], boundaries: dict[str, float], min_tier: str) -> dict[str, Any]:
    exact = 0
    simple_above = 0
    reasoning_below = 0
    for r in rows:
        tier = apply_tool_floor(score_to_tier(r.score, boundaries), r.has_tools, min_tier)
        exact += tier == r.expected
        simple_above += r.expected == "SIMPLE" and tier_index(tier) > 2
        reasoning_below += r.expected == "REASONING" and tier_index(tier) < 2
    n = len(rows)
    return {
        "n": n,
        "accuracy": exact / n if n else 0.0,
        "simple_routed_above_complex": simple_above,
        "reasoning_routed_below_complex": reasoning_below,
    }


def distribution(rows: list[RowResult]) -> dict[str, Any]:
    scores = sorted(r.score for r in rows)
    if not scores:
        return {"n": 0}
    hist = []
    for lo, hi in zip(HIST_EDGES[:-1], HIST_EDGES[1:], strict=True):
        hist.append(
            {
                "range": [lo, hi],
                "count": sum(lo <= s < hi for s in scores)
                + (sum(s == 1.0 for s in scores) if hi == 1.0 else 0),
            }
        )
    q = statistics.quantiles(scores, n=20) if len(scores) >= 2 else scores
    return {
        "n": len(scores),
        "min": scores[0],
        "max": scores[-1],
        "median": statistics.median(scores),
        "mean": statistics.fmean(scores),
        "p05": q[0],
        "p25": q[4],
        "p75": q[14],
        "p95": q[18],
        "share_exactly_zero": sum(s == 0.0 for s in scores) / len(scores),
        "share_below_0_25": sum(s < 0.25 for s in scores) / len(scores),
        "histogram": hist,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    config = RouterConfig()
    rows = load_jsonl(DATASETS / "e1_routing.jsonl")
    results = score_rows(rows, config)
    by_split = {s: [r for r in results if r.split == s] for s in ("train", "dev", "test")}
    dev = by_split["dev"]
    shipped = dict(TIER_BOUNDARIES)
    min_tier = config.tool_session_min_tier or "COMPLEX"

    one_at_a_time: dict[str, list[dict[str, Any]]] = {}
    for key, grid in GRID.items():
        one_at_a_time[key] = []
        for v in grid:
            b = dict(shipped)
            b[key] = v
            if not (
                0.0 <= b["simple_medium"] < b["medium_complex"] < b["complex_reasoning"] <= 1.0
            ):
                continue
            one_at_a_time[key].append(
                {"value": v, "shipped": v == shipped[key], **evaluate(dev, b, min_tier)}
            )

    joint: list[dict[str, Any]] = []
    for a in GRID["simple_medium"]:
        for bb in GRID["medium_complex"]:
            for c in GRID["complex_reasoning"]:
                if not (a < bb < c):
                    continue
                b = {"simple_medium": a, "medium_complex": bb, "complex_reasoning": c}
                joint.append({"boundaries": b, **evaluate(dev, b, min_tier)})
    joint.sort(
        key=lambda e: (
            -e["accuracy"],
            e["reasoning_routed_below_complex"] + e["simple_routed_above_complex"],
        )
    )
    shipped_dev = evaluate(dev, shipped, min_tier)
    best = joint[0] if joint else None

    report = {
        "provenance": provenance(config),
        "note": (
            "Accuracy columns use the dev split only; the test split is frozen and was not "
            "consulted. Score distributions are label-free and reported for every split. "
            "TIER_BOUNDARIES were not changed by this run."
        ),
        "split_assignment_sha256": split_assignment_sha256(rows),
        "shipped_boundaries": shipped,
        "dev": {"n": len(dev), "shipped": shipped_dev},
        "grid": GRID,
        "one_at_a_time": one_at_a_time,
        "joint_top_10": joint[:10],
        "joint_settings_evaluated": len(joint),
        "joint_best_dev": best,
        "score_distribution": {s: distribution(v) for s, v in by_split.items()},
        "score_distribution_all": distribution(results),
    }

    print(
        f"dev n={len(dev)} shipped {shipped} -> accuracy={shipped_dev['accuracy']:.3f} "
        f"simple_above={shipped_dev['simple_routed_above_complex']} "
        f"reasoning_below={shipped_dev['reasoning_routed_below_complex']}"
    )
    for key, entries in one_at_a_time.items():
        line = " ".join(
            f"{e['value']}:{e['accuracy']:.2f}{'*' if e['shipped'] else ''}" for e in entries
        )
        print(f"  {key}: {line}")
    if best:
        print(
            f"joint best on dev (n={len(dev)}, noise-level - not applied): "
            f"{best['boundaries']} accuracy={best['accuracy']:.3f}"
        )
    d = report["score_distribution_all"]
    print(
        f"scores (all {d['n']} rows): min={d['min']:.3f} median={d['median']:.3f} "
        f"p95={d['p95']:.3f} max={d['max']:.3f} "
        f"exactly_zero={d['share_exactly_zero']:.2f} below_0.25={d['share_below_0_25']:.2f}"
    )

    if args.write:
        RESULTS.mkdir(exist_ok=True)
        day = datetime.now(UTC).date().isoformat()
        path = RESULTS / f"sensitivity-{day}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"written: results/{path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
