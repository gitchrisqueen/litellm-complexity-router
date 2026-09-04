"""Observer interface for routing decisions.

The hook calls ``observer.on_route(decision)`` after every routing decision.
The default observer does nothing. A deployment can plug in its own analytics
by implementing :class:`RouterObserver`; the library ships no vendor
integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from complexity_router.hook import RouteDecision


@runtime_checkable
class RouterObserver(Protocol):
    """Anything with an ``on_route(decision)`` method."""

    def on_route(self, decision: RouteDecision) -> None:  # pragma: no cover - protocol
        ...


class NoopObserver:
    """The default: observe nothing."""

    def on_route(self, decision: RouteDecision) -> None:
        return None


class LoggingObserver:
    """Log one line per decision through the standard ``logging`` module."""

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        self._logger = logger or logging.getLogger("complexity_router")
        self._level = level

    def on_route(self, decision: RouteDecision) -> None:
        self._logger.log(
            self._level,
            "score=%.3f scored=%s final=%s target=%s floor=%s has_tools=%s effort=%s",
            decision.score,
            decision.scored_tier,
            decision.tier,
            decision.target,
            decision.floor_applied,
            decision.has_tools,
            decision.effort,
        )


class RecordingObserver:
    """Keep every decision in a list. Intended for tests and offline analysis."""

    def __init__(self) -> None:
        self.decisions: list[RouteDecision] = []

    def on_route(self, decision: RouteDecision) -> None:
        self.decisions.append(decision)
