"""Thin hook-level tests: a request comes in, a tier is assigned, the target
model lands on the outgoing payload, and non-router requests pass through."""

from __future__ import annotations

import logging

import pytest

from complexity_router.config import RouterConfig
from complexity_router.hook import ComplexityRouterHook, RouteDecision, apply_decision, decide
from complexity_router.observability import LoggingObserver, NoopObserver, RecordingObserver


def request(text: str, *, tools: bool = False, model: str = "tier-router", **extra: object) -> dict:
    data: dict = {"model": model, "messages": [{"role": "user", "content": text}]}
    if tools:
        data["tools"] = [{"type": "function", "function": {"name": "read_file"}}]
    data.update(extra)
    return data


# ── decide (pure) ─────────────────────────────────────────────────────────────


def test_decide_simple_text_request() -> None:
    d = decide(request("hi"))
    assert d.scored_tier == "SIMPLE"
    assert d.tier == "SIMPLE"
    assert d.target == "tier-simple"
    assert d.has_tools is False
    assert d.floor_applied is False
    assert d.effort is None
    assert d.tool_choice is None


def test_decide_applies_tool_floor() -> None:
    d = decide(request("hi", tools=True))
    assert d.scored_tier == "SIMPLE"
    assert d.tier == "COMPLEX"
    assert d.target == "tier-complex"
    assert d.floor_applied is True
    assert d.effort == "low"
    assert d.tool_choice == "required"


def test_decide_reasoning_request() -> None:
    text = (
        "Analyze and compare the tradeoffs between optimistic and pessimistic concurrency "
        "control in a distributed database, and prove which one preserves serializability."
    )
    d = decide(request(text))
    assert d.tier == "REASONING"
    assert d.target == "tier-reasoning"
    assert d.effort in {"medium", "high"}


def test_decide_respects_config_overrides() -> None:
    cfg = RouterConfig(
        tier_models={"SIMPLE": "s", "MEDIUM": "m", "COMPLEX": "c", "REASONING": "r"},
        tool_session_min_tier=None,
    )
    d = decide(request("hi", tools=True), cfg)
    assert d.tier == "SIMPLE"
    assert d.target == "s"
    assert d.floor_applied is False
    assert d.tool_choice == "auto"


def test_decide_does_not_mutate_request() -> None:
    data = request("hi", tools=True)
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in data.items()}
    decide(data)
    assert data == before


def test_decision_metadata_rounds_score() -> None:
    d = RouteDecision(
        score=0.123456789,
        scored_tier="COMPLEX",
        tier="COMPLEX",
        target="t",
        has_tools=False,
        floor_applied=False,
        effort="low",
        tool_choice=None,
    )
    assert d.as_metadata()["score"] == 0.1235


# ── apply_decision ────────────────────────────────────────────────────────────


def test_apply_decision_sets_model_tool_choice_and_metadata() -> None:
    cfg = RouterConfig()
    data = request("hi", tools=True)
    d = decide(data, cfg)
    apply_decision(data, d, cfg)
    assert data["model"] == "tier-complex"
    assert data["tool_choice"] == "required"
    meta = data["metadata"]["complexity_router"]
    assert meta["tier"] == "COMPLEX"
    assert meta["routed_to"] if "routed_to" in meta else meta["target"] == "tier-complex"


def test_apply_decision_respects_caller_tool_choice() -> None:
    cfg = RouterConfig()
    data = request("hi", tools=True, tool_choice="auto")
    apply_decision(data, decide(data, cfg), cfg)
    assert data["tool_choice"] == "auto"


def test_apply_decision_replaces_non_dict_metadata() -> None:
    cfg = RouterConfig()
    data = request("hi", metadata="not-a-dict")
    apply_decision(data, decide(data, cfg), cfg)
    assert isinstance(data["metadata"], dict)
    assert "complexity_router" in data["metadata"]


# ── the hook ──────────────────────────────────────────────────────────────────


async def test_hook_routes_router_model_requests() -> None:
    hook = ComplexityRouterHook(RouterConfig())
    data = await hook.async_pre_call_hook({}, None, request("hi", tools=True), "completion")
    assert data["model"] == "tier-complex"
    assert data["tool_choice"] == "required"
    assert data["metadata"]["complexity_router"]["floor_applied"] is True


async def test_hook_passes_through_other_models_untouched() -> None:
    hook = ComplexityRouterHook(RouterConfig())
    original = request("hi", model="some-other-model", tools=True)
    snapshot = dict(original)
    data = await hook.async_pre_call_hook({}, None, original, "completion")
    assert data is original
    assert data == snapshot
    assert "metadata" not in data


async def test_hook_notifies_observer() -> None:
    obs = RecordingObserver()
    hook = ComplexityRouterHook(RouterConfig(), observer=obs)
    await hook.async_pre_call_hook({}, None, request("hi"), "completion")
    await hook.async_pre_call_hook({}, None, request("x", model="other"), "completion")
    assert len(obs.decisions) == 1
    assert obs.decisions[0].target == "tier-simple"


async def test_hook_falls_back_to_default_tier_on_scoring_error() -> None:
    cfg = RouterConfig()
    hook = ComplexityRouterHook(cfg)
    # A boundaries dict that breaks lookup after construction simulates a runtime fault.
    cfg.tier_boundaries.clear()
    data = await hook.async_pre_call_hook({}, None, request("hi"), "completion")
    assert data["model"] == cfg.tier_models[cfg.default_tier]
    assert "error" in data["metadata"]["complexity_router"]


async def test_hook_uses_custom_router_name() -> None:
    cfg = RouterConfig(router_model_name="my-router")
    hook = ComplexityRouterHook(cfg)
    data = await hook.async_pre_call_hook({}, None, request("hi", model="my-router"), "x")
    assert data["model"] == "tier-simple"


def test_hook_default_observer_is_noop() -> None:
    assert isinstance(ComplexityRouterHook(RouterConfig()).observer, NoopObserver)


def test_logging_observer_emits_one_line(caplog: pytest.LogCaptureFixture) -> None:
    obs = LoggingObserver(logging.getLogger("test.router"))
    with caplog.at_level(logging.INFO, logger="test.router"):
        obs.on_route(decide(request("hi", tools=True)))
    assert len(caplog.records) == 1
    assert "final=COMPLEX" in caplog.records[0].getMessage()


def test_module_level_instance_exists() -> None:
    from complexity_router.hook import complexity_router_hook

    assert isinstance(complexity_router_hook, ComplexityRouterHook)
