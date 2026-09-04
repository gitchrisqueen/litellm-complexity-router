"""Tier boundaries, the tool-bearing floor, and the score-to-effort map.

The scorer produces a number in ``[0, 1]``; this module turns it into one of
four ordered tiers and, optionally, a reasoning-effort level. Everything here
is a pure function of its arguments so it can be exercised by the evals as-is.
"""

from __future__ import annotations

from typing import Final

TIER_ORDER: Final[tuple[str, ...]] = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")

#: Cut points. A score below ``simple_medium`` is SIMPLE, below ``medium_complex``
#: is MEDIUM, below ``complex_reasoning`` is COMPLEX, otherwise REASONING.
#: Provenance of the values is recorded in CHANGELOG.md; see CONTRIBUTING.md
#: for the rule on changing them.
TIER_BOUNDARIES: Final[dict[str, float]] = {
    "simple_medium": 0.05,
    "medium_complex": 0.09,
    "complex_reasoning": 0.25,
}

#: Minimum tier for a request that carries tools. Lower tiers may narrate a
#: tool call as text instead of emitting a structured call; the floor keeps
#: tool-bearing traffic on a tier expected to execute tools.
DEFAULT_TOOL_SESSION_MIN_TIER: Final[str] = "COMPLEX"

#: ``(max_score_exclusive, effort_without_tools, effort_with_tools)``.
#: Tool execution is largely mechanical, so effort is capped one level lower
#: when tools are present. Scores below ``medium_complex`` land on tiers whose
#: models are not expected to support an effort parameter.
EFFORT_BREAKPOINTS: Final[tuple[tuple[float, str | None, str | None], ...]] = (
    (0.09, None, None),  # SIMPLE / MEDIUM
    (0.15, "low", "low"),  # lower COMPLEX
    (0.25, "low", "low"),  # upper COMPLEX
    (0.40, "medium", "low"),  # lower REASONING
    (0.60, "high", "medium"),  # mid REASONING
    (1.01, "high", "medium"),  # upper REASONING
)


def validate_boundaries(boundaries: dict[str, float]) -> None:
    """Raise ``ValueError`` unless the three cut points are present and ascending."""
    missing = [k for k in TIER_BOUNDARIES if k not in boundaries]
    if missing:
        raise ValueError(f"missing tier boundaries: {missing}")
    a, b, c = (
        boundaries["simple_medium"],
        boundaries["medium_complex"],
        boundaries["complex_reasoning"],
    )
    if not (0.0 <= a < b < c <= 1.0):
        raise ValueError(f"tier boundaries must satisfy 0 <= {a} < {b} < {c} <= 1")


def score_to_tier(score: float, boundaries: dict[str, float] | None = None) -> str:
    """Map a score to SIMPLE / MEDIUM / COMPLEX / REASONING."""
    b = TIER_BOUNDARIES if boundaries is None else boundaries
    if score < b["simple_medium"]:
        return "SIMPLE"
    if score < b["medium_complex"]:
        return "MEDIUM"
    if score < b["complex_reasoning"]:
        return "COMPLEX"
    return "REASONING"


def tier_index(tier: str) -> int:
    """Position of ``tier`` in ``TIER_ORDER``; raises ``ValueError`` for unknown names."""
    try:
        return TIER_ORDER.index(tier)
    except ValueError as exc:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIER_ORDER}") from exc


def apply_tool_floor(
    tier: str,
    has_tools: bool,
    min_tier: str = DEFAULT_TOOL_SESSION_MIN_TIER,
) -> str:
    """Promote ``tier`` to at least ``min_tier`` when the request carries tools.

    Requests without tools keep their scored tier unchanged.
    """
    if not has_tools:
        return tier
    if tier_index(tier) < tier_index(min_tier):
        return min_tier
    return tier


def score_to_effort(
    score: float,
    tier: str,
    has_tools: bool,
    boundaries: dict[str, float] | None = None,
) -> str | None:
    """Map score + final tier + tool presence to ``"low"``/``"medium"``/``"high"``/``None``.

    ``tier`` is the tier *after* the floor. A floor-elevated request (score
    below ``medium_complex`` but tier raised because tools are present) gets a
    tier-based default rather than the misleadingly low raw score.
    """
    b = TIER_BOUNDARIES if boundaries is None else boundaries
    if score < b["medium_complex"]:
        if tier == "COMPLEX":
            return "low"
        if tier == "REASONING":
            return "medium"
        return None
    for max_score, effort_no_tools, effort_with_tools in EFFORT_BREAKPOINTS:
        if score < max_score:
            return effort_with_tools if has_tools else effort_no_tools
    return "high"
