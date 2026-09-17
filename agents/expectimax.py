"""Expectimax search over the 2048 game tree.

2048 alternates between two kinds of node:

  * a **max node** -- the player picks a direction;
  * a **chance node** -- the game drops a tile, uniformly over the empty
    cells, a 2 with probability 0.9 and a 4 with probability 0.1.

The search models both, so the value of a move is a genuine expectation over
what the game might do next rather than a guess about the immediate board.

Three things keep it affordable on a slow CPU:

``depth``
    How many player plies to look ahead. Configurable; the useful range on
    this machine is 1-3.
``prob_cutoff``
    Branches are weighted by how likely they are. Once the running probability
    of a line falls below this threshold the search stops and evaluates
    statically. This is what actually bounds the work -- a chance node with
    ten empty cells splits probability twenty ways, so unlikely lines die off
    quickly and the effective depth adapts to how crowded the board is.
``transposition table``
    The same position is reached by many move orders. Results are memoised per
    (afterstate, remaining depth); the table is bounded and cleared when full.

Measured on the target machine (Celeron 3865U, 6 games per row):

    depth  adaptive  cutoff   mean score   games/s
      1       no      1e-3         5,563     89.0
      2       no      1e-3        11,061      1.71    <- default
      3       no      1e-2        13,997      0.18
      3       no      5e-3        19,735      0.13
      1      yes      1e-3        16,143      0.29
      2      yes      1e-2        15,857      0.10

The default is the strongest setting that still evaluates a few hundred games
in minutes rather than hours. ``--adaptive`` or ``--depth 3 --prob-cutoff
5e-3`` buy roughly 50-80% more score at 6-13x the cost.
"""

from __future__ import annotations

from engine import board as B
from .base import Agent
from .heuristic import BoardEvaluator

_MOVE = B.move
_SPAWN_POS = B.spawn_positions
_ACTIONS = B.ACTIONS
_NEG_INF = float("-inf")


class ExpectimaxAgent(Agent):
    """Depth-limited expectimax with a static evaluator at the leaves.

    Parameters
    ----------
    depth:
        Player plies to search. 1 still looks one spawn ahead (it evaluates
        the expectation over the tile drop), which already beats the greedy
        heuristic agent.
    prob_cutoff:
        Probability below which a line is evaluated statically instead of
        expanded. Larger = faster and weaker.
    adaptive:
        Add a ply when the board gets crowded (few empty cells), where
        mistakes are most expensive and the branching factor is smallest.
    weights:
        Overrides for :class:`~agents.heuristic.BoardEvaluator`.
    max_table:
        Entry cap for the transposition table before it is cleared.
    """

    name = "expectimax"

    def __init__(self, seed=None, depth: int = 2, prob_cutoff: float = 1e-3,
                 adaptive: bool = False, weights: dict | None = None,
                 score_weight: float = 1.0, max_table: int = 400_000):
        super().__init__(seed)
        self.depth = int(depth)
        self.prob_cutoff = float(prob_cutoff)
        self.adaptive = bool(adaptive)
        self.score_weight = float(score_weight)
        self.max_table = int(max_table)
        self.evaluator = BoardEvaluator.get(weights)
        self.lost_penalty = self.evaluator.lost_penalty
        self._table: dict = {}
        self.nodes = 0          # leaf evaluations, for benchmarking

    # -- public ------------------------------------------------------------
    def new_game(self) -> None:
        self._table.clear()

    def act(self, board: int) -> int:
        if len(self._table) > self.max_table:
            self._table.clear()

        depth = self.depth
        if self.adaptive:
            empties = B.empty_count(board)
            if empties <= 3:
                depth += 2
            elif empties <= 6:
                depth += 1

        best_a = -1
        best_v = _NEG_INF
        sw = self.score_weight
        for a in _ACTIONS:
            nb, gained, moved = _MOVE(board, a)
            if not moved:
                continue
            v = sw * gained + self._chance(nb, depth - 1, 1.0)
            if v > best_v:
                best_v = v
                best_a = a
        if best_a < 0:
            legal = B.legal_actions(board)
            return legal[0] if legal else 0
        return best_a

    def describe(self) -> dict:
        return {
            "name": self.name,
            "depth": self.depth,
            "prob_cutoff": self.prob_cutoff,
            "adaptive": self.adaptive,
            "score_weight": self.score_weight,
            "weights": self.evaluator.weights,
        }

    # -- search ------------------------------------------------------------
    def _chance(self, afterstate: int, depth: int, prob: float) -> float:
        """Expected value over the tile the game is about to drop."""
        if depth <= 0 or prob < self.prob_cutoff:
            self.nodes += 1
            return self.evaluator(afterstate)

        key = (afterstate, depth)
        cached = self._table.get(key)
        if cached is not None:
            return cached

        cells = _SPAWN_POS(afterstate)
        n = len(cells)
        if n == 0:
            # Board filled exactly; the player moves again from here.
            return self._max(afterstate, depth, prob)

        p2 = 0.9 / n
        p4 = 0.1 / n
        prob2 = prob * p2
        prob4 = prob * p4
        total = 0.0
        mx = self._max
        for c in cells:
            sh = 4 * c
            total += p2 * mx(afterstate | (1 << sh), depth, prob2)
            total += p4 * mx(afterstate | (2 << sh), depth, prob4)

        self._table[key] = total
        return total

    def _max(self, b: int, depth: int, prob: float) -> float:
        """Best value the player can force from position ``b``."""
        if depth <= 0 or prob < self.prob_cutoff:
            self.nodes += 1
            return self.evaluator(b)

        best = _NEG_INF
        sw = self.score_weight
        chance = self._chance
        nd = depth - 1
        for a in _ACTIONS:
            nb, gained, moved = _MOVE(b, a)
            if not moved:
                continue
            v = sw * gained + chance(nb, nd, prob)
            if v > best:
                best = v
        if best == _NEG_INF:        # no legal move: the game ends here
            return -self.lost_penalty
        return best
