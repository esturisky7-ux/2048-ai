"""Hand-written board evaluation and the one-ply heuristic agent.

Design note: every feature that can be computed from a single line of four
cells is baked into one 65536-entry table at construction time. Evaluating a
board is then a transpose plus twelve array lookups (~3 us) instead of tens of
microseconds of Python arithmetic. This matters enormously for expectimax,
which calls the evaluator at every leaf.

Features
--------
Line-local (summed over the 4 rows *and* the 4 columns):
  empty         number of empty cells                                (bonus)
  merges        adjacent equal pairs, i.e. merges available right now (bonus)
  monotonicity  how far the line is from being sorted                (penalty)
  smoothness    total gap between adjacent tiles                     (penalty)
  sum           total tile mass, discourages hoarding junk            (penalty)

Whole-board:
  max_tile      the largest tile present                             (bonus)
  corner        positional weights pulling big tiles to one corner    (bonus)
  mobility      how many directions are still legal                  (bonus)
"""

from __future__ import annotations

from array import array

from engine import board as B
from .base import Agent

# Positional weight matrix: a "snake" that decreases along row 0 then reverses,
# so the natural optimum keeps the biggest tile in the top-left corner and the
# rest in a descending chain. Only used when the ``corner`` weight is non-zero.
SNAKE = (
    (15.0, 14.0, 13.0, 12.0),
    (8.0, 9.0, 10.0, 11.0),
    (7.0, 6.0, 5.0, 4.0),
    (0.0, 1.0, 2.0, 3.0),
)

# Known-good starting point, in the spirit of the widely used
# empty/merges/monotonicity weighting for expectimax 2048 players.
DEFAULT_WEIGHTS = {
    "empty": 270.0,
    "merges": 700.0,
    "monotonicity": 47.0,
    "monotonicity_power": 4.0,
    "smoothness": 0.0,
    "sum": 11.0,
    "sum_power": 3.5,
    "max_tile": 0.0,
    "corner": 0.0,
    "mobility": 0.0,
    "lost_penalty": 200000.0,
}


def _line_score(line, w):
    """Score one line of four exponents using the line-local features."""
    empty = merges = 0
    # merge count: scan for adjacent equal non-zero pairs, never reusing a tile
    i = 0
    prev = 0
    counter = 0
    for v in line:
        if v == 0:
            empty += 1
        else:
            if prev == v:
                counter += 1
            elif counter:
                merges += 1 + counter
                counter = 0
            prev = v
    if counter:
        merges += 1 + counter

    # tile-mass penalty
    total = 0.0
    p = w["sum_power"]
    for v in line:
        if v:
            total += v ** p

    # monotonicity: the cheaper of "decreasing left-to-right" and the reverse
    mp = w["monotonicity_power"]
    mono_l = mono_r = 0.0
    for i in range(3):
        a, b = line[i], line[i + 1]
        av = a ** mp if a else 0.0
        bv = b ** mp if b else 0.0
        if av > bv:
            mono_l += av - bv
        else:
            mono_r += bv - av

    # smoothness: how different adjacent occupied tiles are
    smooth = 0.0
    occupied = [v for v in line if v]
    for i in range(len(occupied) - 1):
        smooth += abs(occupied[i] - occupied[i + 1])

    return (w["empty"] * empty
            + w["merges"] * merges
            - w["monotonicity"] * min(mono_l, mono_r)
            - w["smoothness"] * smooth
            - w["sum"] * total)


class BoardEvaluator:
    """Precomputed, weight-configurable static evaluation of a board.

    Constructing one costs roughly a second (it fills a 65536-entry table), so
    build it once and reuse it. Instances are cached by weight set.
    """

    _cache: dict = {}

    def __init__(self, weights: dict | None = None):
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"unknown evaluator weights: {sorted(unknown)}")
            w.update(weights)
        self.weights = w

        self._line = array("d", bytes(8 * 65536))
        for row in range(65536):
            line = [(row >> (4 * c)) & 0xF for c in range(4)]
            self._line[row] = _line_score(line, w)

        # Positional term, one table per row position (not symmetric, so it
        # cannot share the line table).
        self.use_corner = w["corner"] != 0.0
        self._pos = [array("d", bytes(8 * 65536)) for _ in range(4)]
        if self.use_corner:
            cw = w["corner"]
            for row in range(65536):
                for r in range(4):
                    s = 0.0
                    for c in range(4):
                        e = (row >> (4 * c)) & 0xF
                        if e:
                            s += SNAKE[r][c] * (1 << e)
                    self._pos[r][row] = cw * s

        self.use_max = w["max_tile"] != 0.0
        self.use_mobility = w["mobility"] != 0.0
        self.lost_penalty = w["lost_penalty"]

    def __call__(self, b: int) -> float:
        line = self._line
        r0 = b & 0xFFFF
        r1 = (b >> 16) & 0xFFFF
        r2 = (b >> 32) & 0xFFFF
        r3 = b >> 48
        t = B.transpose(b)
        s = (line[r0] + line[r1] + line[r2] + line[r3]
             + line[t & 0xFFFF] + line[(t >> 16) & 0xFFFF]
             + line[(t >> 32) & 0xFFFF] + line[t >> 48])
        if self.use_corner:
            p = self._pos
            s += p[0][r0] + p[1][r1] + p[2][r2] + p[3][r3]
        if self.use_max:
            s += self.weights["max_tile"] * (1 << B.max_tile_exp(b))
        if self.use_mobility:
            s += self.weights["mobility"] * len(B.legal_actions(b))
        return s

    @classmethod
    def get(cls, weights: dict | None = None) -> "BoardEvaluator":
        """Cached construction -- avoids rebuilding tables per process."""
        key = tuple(sorted((weights or {}).items()))
        ev = cls._cache.get(key)
        if ev is None:
            ev = cls._cache[key] = cls(weights)
        return ev


class HeuristicAgent(Agent):
    """One-ply greedy player: pick the move with the best evaluated afterstate.

    No search and no lookahead over the random spawn -- it is the reference
    point that expectimax and the learned agent have to beat.
    """

    name = "heuristic"

    def __init__(self, seed=None, weights: dict | None = None):
        super().__init__(seed)
        self.evaluator = BoardEvaluator.get(weights)

    def act(self, board: int) -> int:
        best_a = -1
        best_v = float("-inf")
        ev = self.evaluator
        for a in B.ACTIONS:
            nb, gained, moved = B.move(board, a)
            if not moved:
                continue
            v = gained + ev(nb)
            if v > best_v:
                best_v = v
                best_a = a
        if best_a < 0:                      # terminal board; caller erred
            return B.legal_actions(board)[0]
        return best_a

    def describe(self) -> dict:
        return {"name": self.name, "weights": self.evaluator.weights}
