#!/usr/bin/env python3
"""Free eval families for the scoring stage: E1, E4, E5.

Every family runs ``score_complexity`` -> ``score_to_tier`` -> the tool floor
as pure functions. No model is called, nothing costs money, and every run is
deterministic, which is the only reason E4's point-drop gate is defensible.

- **E1 routing accuracy** - rows in ``datasets/e1_routing.jsonl`` carry a
  labeled ``expected_tier``; the harness reports exact-tier accuracy on the
  frozen ``test`` split with a Wilson 95% interval, an off-by-one rate, a
  confusion matrix, and two absolute directional rules (no SIMPLE-labeled row
  routed above COMPLEX, no REASONING-labeled row routed below COMPLEX).
- **E4 frozen regression** - the *regression corpus* is every E1 row (all
  splits) compared against ``results/baseline.json``; the run fails when
  accuracy drops by more than the configured points, and every row whose
  predicted tier changed is listed.
- **E5 malformed and adversarial input** - rows in ``datasets/e5_malformed.jsonl``
  carry a full request and an ``expected_tier`` that pins the router's
  current behaviour on that input class.

Usage::

    python -m evals.harness --family e1 e4 e5            # run and print
    python -m evals.harness --family e1 e4 e5 --write    # also write results/
    python -m evals.harness --family e1 --write-baseline # refresh results/baseline.json

Thresholds live in ``evals/thresholds.yaml``. Each family's ``passed`` flag
is the AND of its enforced thresholds only; unenforced thresholds are still
computed and reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from complexity_router import __version__
from complexity_router.config import RouterConfig
from complexity_router.hook import decide
from complexity_router.tiers import TIER_ORDER, tier_index

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evals" / "datasets"
RESULTS = ROOT / "results"
THRESHOLDS = ROOT / "evals" / "thresholds.yaml"
SCHEMA_VERSION = 1
ALLOWED_SOURCES = ("synthetic",)  # plus "public:<name>" - checked by prefix
ALLOWED_SPLITS = ("train", "dev", "test")


# ── helpers ───────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name}:{i}: invalid JSON ({exc})") from exc
    return rows


def validate_source(row: dict[str, Any], path: Path) -> None:
    src = row.get("source")
    if not isinstance(src, str) or not (src in ALLOWED_SOURCES or src.startswith("public:")):
        raise SystemExit(
            f"{path.name}: row {row.get('id')!r} has no valid 'source' "
            f"(expected one of {ALLOWED_SOURCES} or 'public:<name>')"
        )


def load_thresholds() -> dict[str, Any]:
    return yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8")) or {}


@dataclass
class RowResult:
    id: str
    expected: str
    predicted: str
    scored_tier: str
    score: float
    split: str
    has_tools: bool
    anchored: bool
    rationale: str

    @property
    def exact(self) -> bool:
        return self.expected == self.predicted

    @property
    def distance(self) -> int:
        return tier_index(self.predicted) - tier_index(self.expected)


# ── E1 ────────────────────────────────────────────────────────────────────────


def score_rows(rows: list[dict[str, Any]], config: RouterConfig) -> list[RowResult]:
    out: list[RowResult] = []
    for row in rows:
        data: dict[str, Any] = {
            "model": config.router_model_name,
            "messages": [{"role": "user", "content": row["prompt"]}],
        }
        if row.get("has_tools"):
            data["tools"] = [{"type": "function", "function": {"name": "tool"}}]
        d = decide(data, config)
        out.append(
            RowResult(
                id=str(row["id"]),
                expected=row["expected_tier"],
                predicted=d.tier,
                scored_tier=d.scored_tier,
                score=d.score,
                split=row.get("split", "test"),
                has_tools=bool(row.get("has_tools")),
                anchored=bool(row.get("anchored", False)),
                rationale=str(row.get("rationale", "")),
            )
        )
    return out


def summarise(results: list[RowResult]) -> dict[str, Any]:
    n = len(results)
    exact = sum(r.exact for r in results)
    off_by_one = sum(abs(r.distance) == 1 for r in results)
    lo, hi = wilson_interval(exact, n)
    confusion = {e: {p: 0 for p in TIER_ORDER} for e in TIER_ORDER}
    for r in results:
        confusion[r.expected][r.predicted] += 1
    simple_above_complex = [
        r.id for r in results if r.expected == "SIMPLE" and tier_index(r.predicted) > 2
    ]
    reasoning_below_complex = [
        r.id for r in results if r.expected == "REASONING" and tier_index(r.predicted) < 2
    ]
    return {
        "n": n,
        "exact": exact,
        "accuracy": (exact / n) if n else 0.0,
        "wilson_95": [lo, hi],
        "off_by_one": off_by_one,
        "off_by_one_rate": (off_by_one / n) if n else 0.0,
        "over_routed": sum(r.distance > 0 for r in results),
        "under_routed": sum(r.distance < 0 for r in results),
        "confusion": confusion,
        "directional": {
            "simple_routed_above_complex": simple_above_complex,
            "reasoning_routed_below_complex": reasoning_below_complex,
        },
        "misses": [
            {
                "id": r.id,
                "expected": r.expected,
                "predicted": r.predicted,
                "score": round(r.score, 4),
                "rationale": r.rationale,
            }
            for r in results
            if not r.exact
        ],
    }


def run_e1(config: RouterConfig, thresholds: dict[str, Any]) -> dict[str, Any]:
    path = DATASETS / "e1_routing.jsonl"
    rows = load_jsonl(path)
    for row in rows:
        validate_source(row, path)
        if row.get("split") not in ALLOWED_SPLITS:
            raise SystemExit(
                f"{path.name}: row {row.get('id')!r} split must be one of {ALLOWED_SPLITS}"
            )
        if row.get("expected_tier") not in TIER_ORDER:
            raise SystemExit(f"{path.name}: row {row.get('id')!r} has an unknown expected_tier")
    results = score_rows(rows, config)
    by_split = {s: [r for r in results if r.split == s] for s in ALLOWED_SPLITS}
    test = by_split["test"]
    t = thresholds.get("e1", {})
    test_summary = summarise(test)
    anchored = [r for r in test if r.anchored]
    synthetic = [r for r in test if not r.anchored]
    directional_ok = (
        not test_summary["directional"]["simple_routed_above_complex"]
        and not (test_summary["directional"]["reasoning_routed_below_complex"])
    )
    exact_ok = test_summary["accuracy"] >= float(t.get("exact_min", 0.80))
    checks = {
        "directional_rules": {"passed": directional_ok, "enforced": True},
        "exact_min": {
            "passed": exact_ok,
            "enforced": bool(t.get("enforce_exact", False)),
            "threshold": float(t.get("exact_min", 0.80)),
        },
        "min_test_rows": {
            "passed": len(test) >= int(t.get("min_test_rows", 0)),
            "enforced": bool(t.get("enforce_min_test_rows", False)),
            "threshold": int(t.get("min_test_rows", 0)),
        },
    }
    passed = all(c["passed"] for c in checks.values() if c["enforced"])
    return {
        "family": "E1",
        "dataset": path.name,
        "dataset_sha256": sha256_file(path),
        "rows": len(rows),
        "splits": {s: len(v) for s, v in by_split.items()},
        "test": test_summary,
        "test_anchored": summarise(anchored) if anchored else None,
        "test_synthetic": summarise(synthetic) if synthetic else None,
        "dev": summarise(by_split["dev"]) if by_split["dev"] else None,
        "train": summarise(by_split["train"]) if by_split["train"] else None,
        "checks": checks,
        "passed": passed,
        "per_row": [
            {
                "id": r.id,
                "split": r.split,
                "expected": r.expected,
                "predicted": r.predicted,
                "scored_tier": r.scored_tier,
                "score": round(r.score, 4),
            }
            for r in results
        ],
    }


# ── E4 ────────────────────────────────────────────────────────────────────────


def run_e4(config: RouterConfig, thresholds: dict[str, Any], e1: dict[str, Any]) -> dict[str, Any]:
    baseline_path = RESULTS / "baseline.json"
    t = thresholds.get("e4", {})
    max_drop = float(t.get("max_accuracy_drop_points", 2.0))
    current_rows = {r["id"]: r for r in e1["per_row"]}
    n = len(current_rows)
    current_acc = (
        sum(r["expected"] == r["predicted"] for r in current_rows.values()) / n if n else 0.0
    )
    if not baseline_path.is_file():
        return {
            "family": "E4",
            "baseline": None,
            "corpus_rows": n,
            "current_accuracy": current_acc,
            "passed": False,
            "reason": "results/baseline.json missing - run with --write-baseline",
        }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_rows = {r["id"]: r for r in baseline["per_row"]}
    changed = [
        {"id": rid, "baseline": base_rows[rid]["predicted"], "current": cur["predicted"]}
        for rid, cur in current_rows.items()
        if rid in base_rows and base_rows[rid]["predicted"] != cur["predicted"]
    ]
    added = sorted(set(current_rows) - set(base_rows))
    removed = sorted(set(base_rows) - set(current_rows))
    drop_points = (baseline["accuracy"] - current_acc) * 100
    dataset_changed = baseline.get("dataset_sha256") != e1["dataset_sha256"]
    passed = drop_points <= max_drop
    return {
        "family": "E4",
        "baseline": {
            "recorded": baseline.get("recorded"),
            "harness_git_sha": baseline.get("harness_git_sha"),
            "dataset_sha256": baseline.get("dataset_sha256"),
            "accuracy": baseline["accuracy"],
            "rows": len(base_rows),
        },
        "dataset_changed_since_baseline": dataset_changed,
        "corpus_rows": n,
        "current_accuracy": current_acc,
        "drop_points": drop_points,
        "max_drop_points": max_drop,
        "changed_rows": changed,
        "rows_added": added,
        "rows_removed": removed,
        "passed": passed,
    }


def write_baseline(e1: dict[str, Any]) -> Path:
    rows = e1["per_row"]
    acc = sum(r["expected"] == r["predicted"] for r in rows) / len(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_git_sha": git_sha(),
        "dataset_sha256": e1["dataset_sha256"],
        "accuracy": acc,
        "per_row": [
            {"id": r["id"], "expected": r["expected"], "predicted": r["predicted"]} for r in rows
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "baseline.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# ── E5 ────────────────────────────────────────────────────────────────────────


def run_e5(config: RouterConfig, thresholds: dict[str, Any]) -> dict[str, Any]:
    path = DATASETS / "e5_malformed.jsonl"
    rows = load_jsonl(path)
    outcomes = []
    for row in rows:
        validate_source(row, path)
        request = dict(row["request"])
        request.setdefault("model", config.router_model_name)
        try:
            d = decide(request, config)
            predicted: str | None = d.tier
            score: float | None = round(d.score, 4)
            empty = bool(d.breakdown.get("empty"))
            error = None
        except Exception as exc:  # the router must never raise on malformed input
            predicted, score, empty, error = None, None, False, f"{type(exc).__name__}: {exc}"
        ok = error is None and predicted == row["expected_tier"]
        if "expect_empty_branch" in row:
            ok = ok and (empty == bool(row["expect_empty_branch"]))
        outcomes.append(
            {
                "id": row["id"],
                "class": row.get("class"),
                "expected": row["expected_tier"],
                "predicted": predicted,
                "score": score,
                "empty_branch": empty,
                "error": error,
                "passed": ok,
                "note": row.get("note", ""),
            }
        )
    failed = [o for o in outcomes if not o["passed"]]
    return {
        "family": "E5",
        "dataset": path.name,
        "dataset_sha256": sha256_file(path),
        "rows": len(rows),
        "failed": failed,
        "passed": not failed,
        "per_row": outcomes,
    }


# ── driver ────────────────────────────────────────────────────────────────────


def provenance(config: RouterConfig) -> dict[str, Any]:
    cfg_digest = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_git_sha": git_sha(),
        "package_version": __version__,
        "config_sha256": cfg_digest,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cost_usd": 0.0,
        "note": "Pure-function run over committed datasets; no model was called.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", nargs="+", choices=["e1", "e4", "e5"], default=["e1", "e4", "e5"])
    ap.add_argument(
        "--config", type=Path, default=None, help="RouterConfig YAML (default: library defaults)"
    )
    ap.add_argument(
        "--write", action="store_true", help="write results/<date>.json and results/latest.json"
    )
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="refresh results/baseline.json from this E1 run",
    )
    args = ap.parse_args(argv)

    config = RouterConfig.load(args.config) if args.config else RouterConfig()
    thresholds = load_thresholds()
    report: dict[str, Any] = {"provenance": provenance(config), "families": {}}
    families = set(args.family)
    e1 = None
    if "e1" in families or "e4" in families or args.write_baseline:
        e1 = run_e1(config, thresholds)
        if "e1" in families:
            report["families"]["E1"] = e1
    if args.write_baseline and e1 is not None:
        p = write_baseline(e1)
        print(f"baseline written: {p.relative_to(ROOT)}")
    if "e4" in families and e1 is not None:
        report["families"]["E4"] = run_e4(config, thresholds, e1)
    if "e5" in families:
        report["families"]["E5"] = run_e5(config, thresholds)
    report["passed"] = all(f["passed"] for f in report["families"].values())

    for name, fam in report["families"].items():
        status = "PASS" if fam["passed"] else "FAIL"
        if name == "E1":
            t = fam["test"]
            lo, hi = t["wilson_95"]
            print(
                f"E1 {status}: test n={t['n']} exact={t['accuracy']:.3f} "
                f"[{lo:.3f}, {hi:.3f}] off_by_one={t['off_by_one_rate']:.3f} "
                f"splits={fam['splits']} checks="
                + ",".join(
                    f"{k}:{'ok' if v['passed'] else 'x'}{'' if v['enforced'] else '(unenforced)'}"
                    for k, v in fam["checks"].items()
                )
            )
            for m in t["misses"]:
                print(
                    f"   miss {m['id']}: expected {m['expected']} got {m['predicted']} "
                    f"(score {m['score']}) - {m['rationale']}"
                )
        elif name == "E4":
            if fam["baseline"] is None:
                print(f"E4 {status}: {fam['reason']}")
            else:
                print(
                    f"E4 {status}: corpus n={fam['corpus_rows']} "
                    f"accuracy={fam['current_accuracy']:.3f} "
                    f"baseline={fam['baseline']['accuracy']:.3f} drop={fam['drop_points']:+.1f}pt "
                    f"(max {fam['max_drop_points']}) changed_rows={len(fam['changed_rows'])}"
                    + (" DATASET CHANGED" if fam["dataset_changed_since_baseline"] else "")
                )
        elif name == "E5":
            print(f"E5 {status}: rows={fam['rows']} failed={len(fam['failed'])}")
            for f in fam["failed"]:
                print(
                    f"   fail {f['id']}: expected {f['expected']} got {f['predicted']} "
                    f"error={f['error']}"
                )

    if args.write:
        RESULTS.mkdir(exist_ok=True)
        day = datetime.now(UTC).date().isoformat()
        for name in (f"{day}.json", "latest.json"):
            (RESULTS / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"results written: results/{day}.json, results/latest.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
