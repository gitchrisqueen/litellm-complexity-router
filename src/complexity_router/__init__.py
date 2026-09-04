"""litellm-complexity-router: score prompt complexity, route to a model tier.

Public surface:

- :mod:`complexity_router.scoring` - the seven-dimension weighted scorer.
- :mod:`complexity_router.tiers` - tier boundaries, the tool-bearing floor, effort mapping.
- :mod:`complexity_router.config` - the router configuration and YAML round-trip.
- :mod:`complexity_router.observability` - the observer interface (no-op by default).
- :mod:`complexity_router.hook` - the LiteLLM pre-call hook and the pure ``decide`` function.
"""

from complexity_router.config import RouterConfig
from complexity_router.hook import ComplexityRouterHook, RouteDecision, decide
from complexity_router.observability import LoggingObserver, NoopObserver, RouterObserver
from complexity_router.scoring import (
    DIMENSION_WEIGHTS,
    EMPTY_TEXT_SCORE,
    extract_text,
    score_breakdown,
    score_complexity,
)
from complexity_router.tiers import (
    TIER_BOUNDARIES,
    TIER_ORDER,
    apply_tool_floor,
    score_to_effort,
    score_to_tier,
)

__all__ = [
    "DIMENSION_WEIGHTS",
    "EMPTY_TEXT_SCORE",
    "TIER_BOUNDARIES",
    "TIER_ORDER",
    "ComplexityRouterHook",
    "LoggingObserver",
    "NoopObserver",
    "RouteDecision",
    "RouterConfig",
    "RouterObserver",
    "apply_tool_floor",
    "decide",
    "extract_text",
    "score_breakdown",
    "score_complexity",
    "score_to_effort",
    "score_to_tier",
]

__version__ = "0.2.0"
