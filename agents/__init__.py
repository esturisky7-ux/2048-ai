"""Playing agents: random, heuristic, expectimax and the learned n-tuple net."""
from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent, BoardEvaluator
from .expectimax import ExpectimaxAgent
from .registry import make_agent, AGENT_NAMES

__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "BoardEvaluator",
           "ExpectimaxAgent", "make_agent", "AGENT_NAMES"]
