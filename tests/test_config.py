"""Configuration defaults, validation, and dict/YAML round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from complexity_router.config import (
    DEFAULT_ROUTER_MODEL_NAME,
    DEFAULT_TIER_MODELS,
    ENV_CONFIG_PATH,
    RouterConfig,
)
from complexity_router.scoring import DEFAULT_STRIP_PATTERNS, DIMENSION_WEIGHTS
from complexity_router.tiers import TIER_BOUNDARIES


def test_defaults() -> None:
    cfg = RouterConfig()
    assert cfg.router_model_name == DEFAULT_ROUTER_MODEL_NAME == "tier-router"
    assert cfg.tier_models == DEFAULT_TIER_MODELS
    assert cfg.tier_boundaries == TIER_BOUNDARIES
    assert cfg.dimension_weights == DIMENSION_WEIGHTS
    assert cfg.tool_session_min_tier == "COMPLEX"
    assert cfg.strip_patterns == list(DEFAULT_STRIP_PATTERNS)
    assert cfg.empty_text_score == 0.3


def test_dict_round_trip_is_exact() -> None:
    cfg = RouterConfig(
        router_model_name="my-router",
        tier_models={
            "SIMPLE": "a",
            "MEDIUM": "b",
            "COMPLEX": "c",
            "REASONING": "d",
        },
        tier_boundaries={"simple_medium": 0.1, "medium_complex": 0.2, "complex_reasoning": 0.5},
        tool_session_min_tier="MEDIUM",
        strip_patterns=[r"<x>.*?</x>", r"<y>.*?</y>"],
        empty_text_score=0.0,
    )
    assert RouterConfig.from_dict(cfg.to_dict()) == cfg
    assert RouterConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_yaml_round_trip_is_exact() -> None:
    cfg = RouterConfig(router_model_name="r", strip_patterns=[r"<a[^>]*>.*?</a>"])
    assert RouterConfig.from_yaml(cfg.to_yaml()) == cfg


def test_from_dict_accepts_partial_and_rejects_unknown_keys() -> None:
    cfg = RouterConfig.from_dict({"router_model_name": "x"})
    assert cfg.router_model_name == "x"
    assert cfg.tier_models == DEFAULT_TIER_MODELS
    assert RouterConfig.from_dict(None) == RouterConfig()
    with pytest.raises(ValueError, match="unknown config keys"):
        RouterConfig.from_dict({"router_model": "x"})


def test_from_yaml_accepts_bare_mapping_and_sectioned_mapping() -> None:
    bare = "router_model_name: bare\n"
    sectioned = "model_list: []\ncomplexity_router:\n  router_model_name: sectioned\n"
    assert RouterConfig.from_yaml(bare).router_model_name == "bare"
    assert RouterConfig.from_yaml(sectioned).router_model_name == "sectioned"
    assert RouterConfig.from_yaml("") == RouterConfig()
    with pytest.raises(ValueError):
        RouterConfig.from_yaml("- a list\n")


def test_load_from_path_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "router.yaml"
    path.write_text("complexity_router:\n  router_model_name: from-file\n", encoding="utf-8")
    assert RouterConfig.load(path).router_model_name == "from-file"
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    assert RouterConfig.load().router_model_name == "from-file"
    monkeypatch.delenv(ENV_CONFIG_PATH)
    assert RouterConfig.load() == RouterConfig()


def test_example_config_loads_and_round_trips() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "config.yaml"
    cfg = RouterConfig.load(example)
    assert cfg.router_model_name == "tier-router"
    assert RouterConfig.from_yaml(cfg.to_yaml()) == cfg


@pytest.mark.parametrize(
    "bad",
    [
        {"router_model_name": ""},
        {"tier_models": {"SIMPLE": "a"}},
        {"tier_models": {**DEFAULT_TIER_MODELS, "ULTRA": "z"}},
        {
            "tier_boundaries": {
                "simple_medium": 0.3,
                "medium_complex": 0.2,
                "complex_reasoning": 0.5,
            }
        },
        {"dimension_weights": {"tokenCount": 1.0}},
        {"tool_session_min_tier": "ULTRA"},
        {"default_tier": "ULTRA"},
        {"tier_tool_choice": {"COMPLEX": "always"}},
        {"tier_tool_choice": {"ULTRA": "auto"}},
        {"empty_text_score": 1.5},
        {"strip_patterns": [""]},
    ],
)
def test_validation_rejects(bad: dict) -> None:
    with pytest.raises(ValueError):
        RouterConfig.from_dict(bad)


def test_floor_can_be_disabled() -> None:
    cfg = RouterConfig.from_dict({"tool_session_min_tier": None})
    assert cfg.tool_session_min_tier is None
    assert RouterConfig.from_dict(cfg.to_dict()) == cfg
