"""Router configuration: defaults, validation, and dict/YAML round-trip.

A :class:`RouterConfig` is a plain dataclass. ``from_dict`` / ``to_dict`` are
exact inverses for every field, and ``load`` reads the same shape from a YAML
file (the ``complexity_router:`` section of a LiteLLM proxy config, or a
standalone file - see ``examples/config.yaml``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from complexity_router.scoring import DEFAULT_STRIP_PATTERNS, DIMENSION_WEIGHTS, EMPTY_TEXT_SCORE
from complexity_router.tiers import (
    DEFAULT_TOOL_SESSION_MIN_TIER,
    TIER_BOUNDARIES,
    TIER_ORDER,
    tier_index,
    validate_boundaries,
)

#: The virtual model name the hook intercepts. Requests for any other model
#: pass through untouched.
DEFAULT_ROUTER_MODEL_NAME = "tier-router"

#: Illustrative tier -> model_list entry names. A deployment maps these to
#: whatever its proxy config defines.
DEFAULT_TIER_MODELS: dict[str, str] = {
    "SIMPLE": "tier-simple",
    "MEDIUM": "tier-medium",
    "COMPLEX": "tier-complex",
    "REASONING": "tier-reasoning",
}

#: ``tool_choice`` injected per final tier when the request carries tools and
#: the caller set none. The two working tiers require a structured call; the
#: conversational tiers leave it to the model.
DEFAULT_TIER_TOOL_CHOICE: dict[str, str] = {
    "SIMPLE": "auto",
    "MEDIUM": "auto",
    "COMPLEX": "required",
    "REASONING": "required",
}

ENV_CONFIG_PATH = "COMPLEXITY_ROUTER_CONFIG"


@dataclass
class RouterConfig:
    """Everything the hook needs to make a decision."""

    router_model_name: str = DEFAULT_ROUTER_MODEL_NAME
    tier_models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TIER_MODELS))
    tier_boundaries: dict[str, float] = field(default_factory=lambda: dict(TIER_BOUNDARIES))
    dimension_weights: dict[str, float] = field(default_factory=lambda: dict(DIMENSION_WEIGHTS))
    tool_session_min_tier: str | None = DEFAULT_TOOL_SESSION_MIN_TIER
    tier_tool_choice: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TIER_TOOL_CHOICE))
    strip_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_STRIP_PATTERNS))
    empty_text_score: float = EMPTY_TEXT_SCORE
    default_tier: str = "COMPLEX"
    metadata_key: str = "complexity_router"

    def __post_init__(self) -> None:
        self.validate()

    # ── validation ────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ``ValueError`` on any inconsistency."""
        if not self.router_model_name:
            raise ValueError("router_model_name must be non-empty")
        validate_boundaries(self.tier_boundaries)
        missing_models = [t for t in TIER_ORDER if t not in self.tier_models]
        if missing_models:
            raise ValueError(f"tier_models is missing tiers: {missing_models}")
        unknown_models = [t for t in self.tier_models if t not in TIER_ORDER]
        if unknown_models:
            raise ValueError(f"tier_models has unknown tiers: {unknown_models}")
        missing_weights = [k for k in DIMENSION_WEIGHTS if k not in self.dimension_weights]
        if missing_weights:
            raise ValueError(f"dimension_weights is missing: {missing_weights}")
        if self.tool_session_min_tier is not None:
            tier_index(self.tool_session_min_tier)
        tier_index(self.default_tier)
        for tier, choice in self.tier_tool_choice.items():
            tier_index(tier)
            if choice not in ("auto", "required", "none"):
                raise ValueError(f"tier_tool_choice[{tier!r}] must be auto|required|none")
        if not 0.0 <= self.empty_text_score <= 1.0:
            raise ValueError("empty_text_score must be within [0, 1]")
        for p in self.strip_patterns:
            if not isinstance(p, str) or not p:
                raise ValueError("strip_patterns entries must be non-empty strings")

    # ── round-trip ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "router_model_name": self.router_model_name,
            "tier_models": dict(self.tier_models),
            "tier_boundaries": dict(self.tier_boundaries),
            "dimension_weights": dict(self.dimension_weights),
            "tool_session_min_tier": self.tool_session_min_tier,
            "tier_tool_choice": dict(self.tier_tool_choice),
            "strip_patterns": list(self.strip_patterns),
            "empty_text_score": self.empty_text_score,
            "default_tier": self.default_tier,
            "metadata_key": self.metadata_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RouterConfig:
        """Build a config from a mapping; absent keys take their defaults."""
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        kwargs: dict[str, Any] = {}
        for key in allowed:
            if key not in data:
                continue
            value = data[key]
            if key in ("tier_models", "tier_boundaries", "dimension_weights", "tier_tool_choice"):
                value = dict(value)
            elif key == "strip_patterns":
                value = list(value)
            kwargs[key] = value
        return cls(**kwargs)

    def to_yaml(self) -> str:
        return yaml.safe_dump({"complexity_router": self.to_dict()}, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> RouterConfig:
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config YAML must be a mapping")
        section = loaded.get("complexity_router", loaded)
        return cls.from_dict(section)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> RouterConfig:
        """Load from ``path``, else ``$COMPLEXITY_ROUTER_CONFIG``, else defaults."""
        candidate = path or os.environ.get(ENV_CONFIG_PATH)
        if not candidate:
            return cls()
        return cls.from_yaml(Path(candidate).read_text(encoding="utf-8"))
