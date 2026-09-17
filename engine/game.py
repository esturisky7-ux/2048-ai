"""Stateful 2048 game object.

A thin, convenient wrapper around :mod:`engine.board`. Agents and the
dashboard use this; the training loop does **not** -- it drives the bitboard
functions directly to avoid attribute lookups and object churn.

Nothing here renders anything. Visual output lives entirely in the dashboard.
"""

from __future__ import annotations

from random import Random

from . import board as B


class Game:
    """One game of 2048.

    Parameters
    ----------
    seed:
        Seed for the tile-spawn RNG. Pass an int for a reproducible game.
    rng:
        An existing :class:`random.Random` to draw from instead of creating
        one. Takes precedence over ``seed``.
    """

    __slots__ = ("board", "score", "moves", "rng", "_over")

    def __init__(self, seed=None, rng=None):
        self.rng = rng if rng is not None else Random(seed)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> int:
        self.board = B.new_game(self.rng)
        self.score = 0
        self.moves = 0
        self._over = False
        return self.board

    # -- queries -----------------------------------------------------------
    @property
    def game_over(self) -> bool:
        if not self._over:
            self._over = B.is_game_over(self.board)
        return self._over

    def legal_actions(self):
        return B.legal_actions(self.board)

    @property
    def max_tile(self) -> int:
        return B.max_tile(self.board)

    @property
    def empty_count(self) -> int:
        return B.empty_count(self.board)

    def rows(self):
        return B.board_to_rows(self.board)

    # -- transitions -------------------------------------------------------
    def step(self, action: int):
        """Play ``action``: slide, then spawn a random tile.

        Returns ``(reward, done, moved)`` where ``reward`` is the merge score
        gained. An illegal action is a no-op and returns ``moved=False``.
        """
        nb, gained, moved = B.move(self.board, action)
        if not moved:
            return 0, self.game_over, False
        self.board = B.random_spawn(nb, self.rng)
        self.score += gained
        self.moves += 1
        self._over = False
        return gained, self.game_over, True

    def __str__(self) -> str:
        return B.render(self.board)
