"""Tier mapping, the tool-bearing floor, and the effort map."""

from __future__ import annotations

import pytest

from complexity_router.tiers import (
    DEFAULT_TOOL_SESSION_MIN_TIER,
    EFFORT_BREAKPOINTS,
    TIER_BOUNDARIES,
    TIER_ORDER,
    apply_tool_floor,
    score_to_effort,
    score_to_tier,
    tier_index,
    validate_boundaries,
)


def test_boundaries_are_the_published_cuts() -> None:
    assert TIER_BOUNDARIES == {
        "simple_medium": 0.05,
        "medium_complex": 0.09,
        "complex_reasoning": 0.25,
    }


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (0.0, "SIMPLE"),
        (0.0499, "SIMPLE"),
        (0.05, "MEDIUM"),
        (0.0899, "MEDIUM"),
        (0.09, "COMPLEX"),
        (0.2499, "COMPLEX"),
        (0.25, "REASONING"),
        (0.95, "REASONING"),
        (1.0, "REASONING"),
    ],
)
def test_score_to_tier_boundaries_are_lower_inclusive(score: float, tier: str) -> None:
    assert score_to_tier(score) == tier


def test_score_to_tier_with_custom_boundaries() -> None:
    b = {"simple_medium": 0.2, "medium_complex": 0.4, "complex_reasoning": 0.6}
    assert score_to_tier(0.1, b) == "SIMPLE"
    assert score_to_tier(0.5, b) == "COMPLEX"


def test_validate_boundaries_rejects_unordered_and_missing() -> None:
    with pytest.raises(ValueError):
        validate_boundaries({"simple_medium": 0.5, "medium_complex": 0.4, "complex_reasoning": 0.6})
    with pytest.raises(ValueError):
        validate_boundaries({"simple_medium": 0.1})
    validate_boundaries(TIER_BOUNDARIES)


def test_tier_order_and_index() -> None:
    assert TIER_ORDER == ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
    assert [tier_index(t) for t in TIER_ORDER] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        tier_index("ULTRA")


@pytest.mark.parametrize("tier", TIER_ORDER)
def test_floor_promotes_only_tool_bearing_requests(tier: str) -> None:
    assert apply_tool_floor(tier, has_tools=False) == tier
    floored = apply_tool_floor(tier, has_tools=True)
    assert tier_index(floored) >= tier_index(DEFAULT_TOOL_SESSION_MIN_TIER)
    if tier_index(tier) >= tier_index(DEFAULT_TOOL_SESSION_MIN_TIER):
        assert floored == tier


def test_floor_with_custom_minimum() -> None:
    assert apply_tool_floor("SIMPLE", True, min_tier="MEDIUM") == "MEDIUM"
    assert apply_tool_floor("REASONING", True, min_tier="MEDIUM") == "REASONING"


def test_effort_breakpoints_are_ascending_and_cover_unit_interval() -> None:
    maxes = [m for m, _, _ in EFFORT_BREAKPOINTS]
    assert maxes == sorted(maxes)
    assert maxes[0] == TIER_BOUNDARIES["medium_complex"]
    assert maxes[-1] > 1.0


@pytest.mark.parametrize(
    ("score", "tier", "has_tools", "effort"),
    [
        (0.02, "SIMPLE", False, None),
        (0.07, "MEDIUM", False, None),
        (0.02, "COMPLEX", True, "low"),  # floor-elevated tool session
        (0.02, "REASONING", True, "medium"),  # unusual floor-to-REASONING
        (0.10, "COMPLEX", False, "low"),
        (0.20, "COMPLEX", True, "low"),
        (0.30, "REASONING", False, "medium"),
        (0.30, "REASONING", True, "low"),
        (0.50, "REASONING", False, "high"),
        (0.50, "REASONING", True, "medium"),
        (0.95, "REASONING", False, "high"),
        (0.95, "REASONING", True, "medium"),
        (1.5, "REASONING", False, "high"),  # guard: beyond the last breakpoint
    ],
)
def test_score_to_effort(score: float, tier: str, has_tools: bool, effort: str | None) -> None:
    assert score_to_effort(score, tier, has_tools) == effort


def test_tools_never_raise_effort_above_no_tools() -> None:
    order = {None: 0, "low": 1, "medium": 2, "high": 3}
    for s in [x / 100 for x in range(0, 101)]:
        tier = score_to_tier(s)
        with_tools = score_to_effort(s, apply_tool_floor(tier, True), True)
        without = score_to_effort(s, tier, False)
        if tier_index(tier) >= tier_index("COMPLEX"):
            assert order[with_tools] <= order[without]
