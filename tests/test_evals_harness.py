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
