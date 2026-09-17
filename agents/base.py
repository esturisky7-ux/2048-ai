"""Agent interface.

An agent is anything that can look at a bitboard and name an action. Agents
are deliberately stateless with respect to the game object: they receive a
raw ``int`` board, which keeps them usable from the training loop, the
evaluator and the dashboard alike without adapters.
"""

from __future__ import annotations

from random import Random

from engine import board as B


class Agent:
    """Base class. Subclasses must implement :meth:`act`."""

    name = "agent"

    def __init__(self, seed: int | None = None):
        self.rng = Random(seed)

    def act(self, board: int) -> int:
        """Return an action for ``board``. Must return a *legal* action.

        Callers guarantee the board is not terminal.
        """
        raise NotImplementedError

    def new_game(self) -> None:
        """Hook called at the start of each game (clears per-game caches)."""

    def describe(self) -> dict:
        """Configuration of this agent, recorded alongside results."""
        return {"name": self.name}


class RandomAgent(Agent):
    """Uniformly random legal move. The performance floor."""

    name = "random"

    def act(self, board: int) -> int:
        return self.rng.choice(B.legal_actions(board))
