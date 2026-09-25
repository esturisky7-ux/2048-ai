"""Agent wrapper around a trained n-tuple network.

Loads a run's weight file read-only, so watching or evaluating an agent can
never *write* to a training run. Reading is another matter: a read-only map of
a run's current weights still sees every update a trainer makes, so a fixed
evaluation of them also needs the run held still -- see ``live_run`` and
:func:`training.runlock.hold_frozen`. Snapshots never change.

The agent plays the policy training followed, minus exploration: it scores
moves with the run's own reward configuration and discount, through the same
scorer training uses (:func:`training.learner.best_move`). With the default
configuration -- pure merge score, gamma 1 -- that is ``merge score +
V(afterstate)``.
"""

from __future__ import annotations

import os
import re

from engine import board as B
from .base import Agent

_SNAPSHOT_GAMES = re.compile(r"games-(\d+)\.f32$")


def live_run_for(run: str = "default", checkpoint=None) -> str | None:
    """The run whose *current* weights an agent loaded like this would play.

    ``None`` for a snapshot, and for a weight file that belongs to no run:
    neither of those ever changes underneath an evaluation.
    """
    if not checkpoint:
        return run
    from training.checkpoint import run_for_weights
    found = run_for_weights(checkpoint)
    return found[0] if found and found[1] else None


class LearnedAgent(Agent):
    """Greedy one-ply policy over a learned afterstate value function.

    Parameters
    ----------
    run:
        Name of the training run to load (default ``"default"``).
    checkpoint:
        Explicit path to a ``.f32`` weight file, overriding ``run``. Use this
        to evaluate a frozen snapshot. A snapshot inside a run's directory is
        described by that run's metadata and configuration.
    tuple_set:
        Must match the network the weights were trained with. Read from the
        run's metadata when not given.
    depth:
        1 (default) is the trained policy itself. Values above 1 wrap the
        learned value function in an expectimax search, which is markedly
        stronger and markedly slower. The search optimises the same objective
        the values were trained on: every ply scores ``r + gamma * value``
        with the run's reward and discount, so its leaves and its inner nodes
        are measured in the same units.
    reward, gamma:
        Override the reward configuration and discount the policy scores
        moves with. By default they are the run's own.
    """

    name = "learned"

    def __init__(self, seed=None, run: str = "default",
                 checkpoint: str | None = None, tuple_set: str | None = None,
                 depth: int = 1, prob_cutoff: float = 1e-3,
                 reward: dict | None = None, gamma: float | None = None):
        super().__init__(seed)
        from training.checkpoint import Run, run_for_weights
        from training.learner import GreedyPolicy
        from training.ntuple import NTupleNetwork
        from training.reward import RewardFunction

        found = run_for_weights(checkpoint) if checkpoint else None
        source = found[0] if found else run
        r = Run(source)
        meta = r.load_meta() or {}
        cfg = r.load_config() or {}
        path = checkpoint or r.weights_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no trained weights at {path} "
                f"(run {run!r} has not been trained yet)")
        rf = RewardFunction(reward if reward is not None
                            else cfg.get("reward"))
        if gamma is None:
            gamma = (cfg.get("learning") or {}).get("gamma", 1.0)
        self.tuple_set = tuple_set or meta.get("tuple_set", "4x6")
        self.run_name = source
        self.live_run = live_run_for(source, checkpoint)
        snap = _SNAPSHOT_GAMES.search(os.fspath(path)) \
            if found and not found[1] else None
        self.games_trained = int(snap.group(1)) if snap \
            else meta.get("games", 0)
        # Kept as a string: it goes straight into JSON result files.
        self.checkpoint_path = os.fspath(path)
        self.net = NTupleNetwork(self.tuple_set, path=path, readonly=True)
        self.policy = GreedyPolicy(self.net, rf, gamma)
        self.depth = int(depth)
        self.prob_cutoff = float(prob_cutoff)
        # What a position with no legal move is worth: the TD target a
        # trainer uses for the afterstate that led to it.
        self._dead = rf.terminal() or 0.0

    # -- policy ------------------------------------------------------------
    def new_game(self) -> None:
        self.policy.new_game()

    def act(self, board: int) -> int:
        a = self.policy.act(board) if self.depth <= 1 \
            else self._search_act(board)
        if a < 0:
            legal = B.legal_actions(board)
            return legal[0] if legal else 0
        return a

    # -- optional expectimax on top of the learned values -------------------
    def _search_act(self, board: int) -> int:
        pol = self.policy
        rf = pol.reward
        pure = rf.is_pure_score
        gamma = pol.gamma
        best_a, best_v, best_mx = -1, float("-inf"), pol.max_exp
        for a in (0, 1, 2, 3):
            nb, gained, moved = B.move(board, a)
            if not moved:
                continue
            r, mx = (gained, pol.max_exp) if pure else \
                rf.step(board, nb, gained, pol.max_exp)
            v = r + gamma * self._chance(nb, self.depth - 1, 1.0, mx)
            if v > best_v:
                best_v, best_a, best_mx = v, a, mx
        if best_a >= 0:
            pol.max_exp = best_mx         # as training would, after this move
        return best_a

    def _chance(self, afterstate: int, depth: int, prob: float,
                max_exp: int) -> float:
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
            total += p2 * self._max(afterstate | (1 << sh), depth, prob * p2,
                                    max_exp)
            total += p4 * self._max(afterstate | (2 << sh), depth, prob * p4,
                                    max_exp)
        return total

    def _max(self, b: int, depth: int, prob: float, max_exp: int) -> float:
        rf = self.policy.reward
        pure = rf.is_pure_score
        gamma = self.policy.gamma
        best = float("-inf")
        for a in (0, 1, 2, 3):
            nb, gained, moved = B.move(b, a)
            if not moved:
                continue
            r, mx = (gained, max_exp) if pure else \
                rf.step(b, nb, gained, max_exp)
            v = r + gamma * self._chance(nb, depth - 1, prob, mx)
            if v > best:
                best = v
        return self._dead if best == float("-inf") else best

    def describe(self) -> dict:
        return {
            "name": self.name,
            "run": self.run_name,
            "checkpoint": self.checkpoint_path,
            "tuple_set": self.tuple_set,
            "games_trained": self.games_trained,
            "depth": self.depth,
            "gamma": self.policy.gamma,
            "reward": self.policy.reward.describe(),
        }

    def close(self) -> None:
        self.net.close()
