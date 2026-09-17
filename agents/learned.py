"""Agent wrapper around a trained n-tuple network.

Loads a run's weight file read-only so that watching or evaluating an agent
can never disturb a training process writing to the same file.
"""

from __future__ import annotations

import os

from engine import board as B
from .base import Agent


class LearnedAgent(Agent):
    """Greedy one-ply policy over a learned afterstate value function.

    Parameters
    ----------
    run:
        Name of the training run to load (default ``"default"``).
    checkpoint:
        Explicit path to a ``.f32`` weight file, overriding ``run``. Use this
        to evaluate a frozen snapshot.
    tuple_set:
        Must match the network the weights were trained with. Read from the
        run's metadata when not given.
    depth:
        1 (default) is the plain greedy policy. Values above 1 wrap the
        learned value function in an expectimax search, which is markedly
        stronger and markedly slower.
    """

    name = "learned"

    def __init__(self, seed=None, run: str = "default",
                 checkpoint: str | None = None, tuple_set: str | None = None,
                 depth: int = 1, prob_cutoff: float = 1e-3):
        super().__init__(seed)
        from training.checkpoint import Run
        from training.ntuple import NTupleNetwork

        r = Run(run)
        meta = r.load_meta() or {}
        path = checkpoint or r.weights_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no trained weights at {path} "
                f"(run {run!r} has not been trained yet)")
        self.tuple_set = tuple_set or meta.get("tuple_set", "4x6")
        self.run_name = run
        self.games_trained = meta.get("games", 0)
        # Kept as a string: it goes straight into JSON result files.
        self.checkpoint_path = os.fspath(path)
        self.net = NTupleNetwork(self.tuple_set, path=path, readonly=True)
        self.depth = int(depth)
        self.prob_cutoff = float(prob_cutoff)

    # -- policy ------------------------------------------------------------
    def act(self, board: int) -> int:
        if self.depth <= 1:
            value = self.net.value
            best_a, best_v = -1, float("-inf")
            for a in (0, 1, 2, 3):
                nb, gained, moved = B.move(board, a)
                if not moved:
                    continue
                v = gained + value(nb)
                if v > best_v:
                    best_v, best_a = v, a
            if best_a < 0:
                legal = B.legal_actions(board)
                return legal[0] if legal else 0
            return best_a
        return self._search_act(board)

    # -- optional expectimax on top of the learned values -------------------
    def _search_act(self, board: int) -> int:
        best_a, best_v = -1, float("-inf")
        for a in (0, 1, 2, 3):
            nb, gained, moved = B.move(board, a)
            if not moved:
                continue
            v = gained + self._chance(nb, self.depth - 1, 1.0)
            if v > best_v:
                best_v, best_a = v, a
        if best_a < 0:
            legal = B.legal_actions(board)
            return legal[0] if legal else 0
        return best_a

    def _chance(self, afterstate: int, depth: int, prob: float) -> float:
        if depth <= 0 or prob < self.prob_cutoff:
            return self.net.value(afterstate)
        cells = B.spawn_positions(afterstate)
        n = len(cells)
        if n == 0:
            return self.net.value(afterstate)
        p2, p4 = 0.9 / n, 0.1 / n
        total = 0.0
        for c in cells:
            sh = 4 * c
            total += p2 * self._max(afterstate | (1 << sh), depth, prob * p2)
            total += p4 * self._max(afterstate | (2 << sh), depth, prob * p4)
        return total

    def _max(self, b: int, depth: int, prob: float) -> float:
        best = float("-inf")
        for a in (0, 1, 2, 3):
            nb, gained, moved = B.move(b, a)
            if not moved:
                continue
            v = gained + self._chance(nb, depth - 1, prob)
            if v > best:
                best = v
        return 0.0 if best == float("-inf") else best

    def describe(self) -> dict:
        return {
            "name": self.name,
            "run": self.run_name,
            "checkpoint": self.checkpoint_path,
            "tuple_set": self.tuple_set,
            "games_trained": self.games_trained,
            "depth": self.depth,
        }

    def close(self) -> None:
        self.net.close()
