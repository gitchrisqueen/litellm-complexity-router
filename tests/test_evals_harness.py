"""Smoke tests for the free eval harness against the committed datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals import harness

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval_basic_properties() -> None:
    lo, hi = harness.wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert harness.wilson_interval(0, 0) == (0.0, 0.0)
    lo50, hi50 = harness.wilson_interval(40, 50)
    lo20, hi20 = harness.wilson_interval(16, 20)
    assert (hi50 - lo50) < (hi20 - lo20)  # a larger n gives a narrower interval


def test_e1_rows_all_carry_source_split_and_known_tier() -> None:
    rows = harness.load_jsonl(harness.DATASETS / "e1_routing.jsonl")
    assert rows, "seed dataset must not be empty"
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["source"] == "synthetic" or r["source"].startswith("public:")
        assert r["split"] in harness.ALLOWED_SPLITS
        assert r["expected_tier"] in harness.TIER_ORDER
        assert isinstance(r["has_tools"], bool)
        assert r["rater"]  # who labeled is always stated


def test_e1_runs_and_directional_rules_hold() -> None:
    out = harness.run_e1(harness.RouterConfig(), harness.load_thresholds())
    assert out["rows"] == sum(out["splits"].values())
    assert out["checks"]["directional_rules"]["passed"] is True
    assert out["passed"] is True
    t = out["test"]
    assert 0.0 <= t["accuracy"] <= 1.0
    assert t["wilson_95"][0] <= t["accuracy"] <= t["wilson_95"][1]


def test_e5_runs_clean_and_never_raises() -> None:
    out = harness.run_e5(harness.RouterConfig(), harness.load_thresholds())
    assert out["rows"] >= 8
    assert all(o["error"] is None for o in out["per_row"])
    assert out["passed"] is True, out["failed"]


def test_e4_against_committed_baseline() -> None:
    th = harness.load_thresholds()
    e1 = harness.run_e1(harness.RouterConfig(), th)
    out = harness.run_e4(harness.RouterConfig(), th, e1)
    assert out["baseline"] is not None, out.get("reason")
    assert out["passed"] is True, out["changed_rows"]
    assert out["dataset_changed_since_baseline"] is False


def test_row_without_source_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    with pytest.raises(SystemExit):
        harness.validate_source({"id": "x"}, p)
    with pytest.raises(SystemExit):
        harness.validate_source({"id": "x", "source": "production"}, p)
    harness.validate_source({"id": "x", "source": "public:example"}, p)


def test_latest_results_carry_provenance() -> None:
    latest = json.loads((ROOT / "results" / "latest.json").read_text())
    prov = latest["provenance"]
    for key in ("schema_version", "run_at", "harness_git_sha", "config_sha256", "cost_usd"):
        assert key in prov
    assert prov["cost_usd"] == 0.0
    assert {"E1", "E4", "E5"} <= set(latest["families"])
    assert latest["families"]["E1"]["dataset_sha256"] == harness.sha256_file(
        harness.DATASETS / "e1_routing.jsonl"
    )


def test_cli_exit_code_reflects_pass(capsys: pytest.CaptureFixture[str]) -> None:
    rc = harness.main(["--family", "e1", "e5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "E1 PASS" in out and "E5 PASS" in out


# ── composition gates, anchored subset, known failures, split digest ─────────


def test_dataset_meets_the_committed_sizing() -> None:
    out = harness.run_e1(harness.RouterConfig(), harness.load_thresholds())
    assert out["rows"] >= 120
    assert out["splits"]["test"] >= 50 and out["splits"]["dev"] >= 20
    assert out["splits"]["train"] >= 50
    for split, n in out["splits"].items():
        assert out["anchored_per_split"][split] / n >= 0.40, split
    assert out["anchored_per_split"]["test"] >= 20
    for name, check in out["checks"].items():
        if check["enforced"]:
            assert check["passed"], name


def test_anchored_rows_are_public_and_carry_an_anchor_block() -> None:
    rows = harness.load_jsonl(harness.DATASETS / "e1_routing.jsonl")
    anchored = [r for r in rows if r.get("anchored")]
    assert len(anchored) >= 48
    for r in anchored:
        assert r["source"].startswith("public:")
        a = r["anchor"]
        for key in (
            "dataset",
            "revision",
            "config",
            "split",
            "row_idx",
            "field",
            "value",
            "mapping",
        ):
            assert key in a, (r["id"], key)
        assert r["license"]


def test_anchored_and_synthetic_subsets_are_reported_separately() -> None:
    out = harness.run_e1(harness.RouterConfig(), harness.load_thresholds())
    assert out["test_anchored"]["n"] + out["test_synthetic"]["n"] == out["test"]["n"]
    assert out["test_anchored_minus_synthetic"] is not None
    assert set(out["test_by_source"]) == {
        r["source"] for r in out["per_row"] if r["split"] == "test"
    }


def test_new_directional_violation_fails_but_known_one_passes(tmp_path: Path) -> None:
    th = harness.load_thresholds()
    out = harness.run_e1(harness.RouterConfig(), th)
    known = out["checks"]["directional_rules"]["known_violations"]
    assert out["checks"]["directional_rules"]["new_violations"] == []
    if known:
        th2 = json.loads(json.dumps(th))
        th2["e1"]["directional"]["known_failures"] = [
            k for k in th2["e1"]["directional"]["known_failures"] if k["id"] != known[0]
        ]
        out2 = harness.run_e1(harness.RouterConfig(), th2)
        assert out2["checks"]["directional_rules"]["new_violations"] == [known[0]]
        assert out2["passed"] is False


def test_split_digest_changes_when_a_row_moves() -> None:
    rows = harness.load_jsonl(harness.DATASETS / "e1_routing.jsonl")
    d1 = harness.split_assignment_sha256(rows)
    moved = json.loads(json.dumps(rows))
    moved[0]["split"] = "dev" if moved[0]["split"] != "dev" else "train"
    assert harness.split_assignment_sha256(moved) != d1


def test_row_validation_refuses_bad_rows(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    good = {
        "id": "x", "prompt": "p", "has_tools": False, "expected_tier": "SIMPLE",
        "source": "synthetic", "split": "train", "rater": "r",
    }  # fmt: skip
    harness.validate_e1_row(good, p)
    with pytest.raises(SystemExit):
        harness.validate_e1_row({**good, "anchored": True}, p)  # anchored but synthetic
    with pytest.raises(SystemExit):
        harness.validate_e1_row({**good, "source": "public:x", "anchored": True}, p)  # no anchor
    with pytest.raises(SystemExit):
        harness.validate_e1_row({k: v for k, v in good.items() if k != "rater"}, p)


def test_sensitivity_sweep_reports_dev_only_and_leaves_boundaries_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from evals import sensitivity

    from complexity_router.tiers import TIER_BOUNDARIES

    before = dict(TIER_BOUNDARIES)
    assert sensitivity.main([]) == 0
    out = capsys.readouterr().out
    assert "dev n=" in out and "test" not in out.split("dev n=")[1].split("\n")[0]
    assert dict(TIER_BOUNDARIES) == before
