"""Name -> agent factory, so every entry point can take ``--agent NAME``."""

from __future__ import annotations

from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent, BoardEvaluator
from .expectimax import ExpectimaxAgent

AGENT_NAMES = ("random", "heuristic", "expectimax", "learned")


def make_agent(name: str, seed=None, **kwargs) -> Agent:
    """Build an agent by name.

    ``learned`` loads an n-tuple network checkpoint; pass ``checkpoint=PATH``
    (defaults to the latest one in ``checkpoints/``).
    """
    name = name.lower()
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "heuristic":
        return HeuristicAgent(seed=seed, **kwargs)
    if name == "expectimax":
        return ExpectimaxAgent(seed=seed, **kwargs)
    if name == "learned":
        from .learned import LearnedAgent      # deferred: pulls in training/
        return LearnedAgent(seed=seed, **kwargs)
    raise ValueError(f"unknown agent {name!r}; choose from {AGENT_NAMES}")


__all__ = ["make_agent", "AGENT_NAMES", "Agent", "RandomAgent",
           "HeuristicAgent", "ExpectimaxAgent", "BoardEvaluator"]
