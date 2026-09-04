"""The LiteLLM pre-call hook, and the pure ``decide`` function behind it.

``decide(data, config)`` is the whole routing decision as a value. The hook
class applies that decision to the outgoing request: it sets ``model``, injects
``tool_choice`` when the caller set none, and records the decision under
``metadata[config.metadata_key]``. Nothing else in the request is touched.

LiteLLM is an optional dependency: when it is installed the hook subclasses
``litellm.integrations.custom_logger.CustomLogger``; when it is not, a
minimal stand-in lets the hook be constructed and tested.
"""

from __future__ import annotations

import re
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from complexity_router.config import RouterConfig
from complexity_router.observability import NoopObserver, RouterObserver
from complexity_router.scoring import compile_strip_patterns, score_breakdown
from complexity_router.tiers import apply_tool_floor, score_to_effort, score_to_tier

try:  # pragma: no cover - exercised only when litellm is installed
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover

    class CustomLogger:  # type: ignore[no-redef]
        """Stand-in for ``litellm.integrations.custom_logger.CustomLogger``."""


@dataclass(frozen=True)
class RouteDecision:
    """The complete outcome of scoring one request."""

    score: float
    scored_tier: str
    tier: str
    target: str
    has_tools: bool
    floor_applied: bool
    effort: str | None
    tool_choice: str | None
    breakdown: dict[str, float] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """The dict stored under ``metadata[<metadata_key>]``."""
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


def _compiled(config: RouterConfig) -> list[re.Pattern[str]]:
    return compile_strip_patterns(config.strip_patterns)


def decide(data: dict[str, Any], config: RouterConfig | None = None) -> RouteDecision:
    """Score ``data["messages"]`` and choose a tier and target model.

    Steps: score the last user message; map the score to a tier; apply the
    tool-bearing floor when ``data["tools"]`` is non-empty; look up the target
    model; derive the effort level; pick the ``tool_choice`` the hook would
    inject. The request is not modified.
    """
    cfg = config or RouterConfig()
    messages = data.get("messages") or []
    breakdown = score_breakdown(
        messages,
        strip_patterns=_compiled(cfg),
        weights=cfg.dimension_weights,
        empty_text_score=cfg.empty_text_score,
    )
    score = breakdown["score"]
    scored_tier = score_to_tier(score, cfg.tier_boundaries)
    has_tools = bool(data.get("tools"))
    tier = scored_tier
    if cfg.tool_session_min_tier is not None:
        tier = apply_tool_floor(scored_tier, has_tools, cfg.tool_session_min_tier)
    target = cfg.tier_models.get(tier, cfg.tier_models[cfg.default_tier])
    effort = score_to_effort(score, tier, has_tools, cfg.tier_boundaries)
    tool_choice = cfg.tier_tool_choice.get(tier, "auto") if has_tools else None
    return RouteDecision(
        score=score,
        scored_tier=scored_tier,
        tier=tier,
        target=target,
        has_tools=has_tools,
        floor_applied=tier != scored_tier,
        effort=effort,
        tool_choice=tool_choice,
        breakdown=breakdown,
    )


def apply_decision(
    data: dict[str, Any], decision: RouteDecision, config: RouterConfig
) -> dict[str, Any]:
    """Write ``decision`` into ``data`` in place and return it."""
    data["model"] = decision.target
    if decision.has_tools and decision.tool_choice and not data.get("tool_choice"):
        data["tool_choice"] = decision.tool_choice
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
    metadata[config.metadata_key] = decision.as_metadata()
    return data


class ComplexityRouterHook(CustomLogger):  # type: ignore[misc]
    """LiteLLM ``CustomLogger`` whose ``async_pre_call_hook`` routes ``tier-router`` requests.

    Register it in the proxy config's ``litellm_settings.callbacks`` as
    ``complexity_router.hook.complexity_router_hook``, and point the client at
    ``model: tier-router`` (or whatever ``router_model_name`` is configured).
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        observer: RouterObserver | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RouterConfig.load()
        self.observer: RouterObserver = observer or NoopObserver()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        if data.get("model") != self.config.router_model_name:
            return data
        try:
            decision = decide(data, self.config)
            apply_decision(data, decision, self.config)
            self.observer.on_route(decision)
        except Exception:
            # Never block a request: on any scoring failure, fall through to
            # the default tier's model so the proxy can still serve the call.
            fallback = self.config.tier_models[self.config.default_tier]
            data["model"] = fallback
            data.setdefault("metadata", {})[self.config.metadata_key] = {
                "error": traceback.format_exc(limit=3),
                "routed_to": fallback,
            }
        return data


#: Module-level instance for ``litellm_settings.callbacks``. Reads
#: ``$COMPLEXITY_ROUTER_CONFIG`` if set, otherwise uses the defaults.
complexity_router_hook = ComplexityRouterHook()
