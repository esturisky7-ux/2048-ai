"""Reward functions for the learning agent.

The default reward is **the merge score and nothing else**, because it is the
only signal in 2048 that cannot be farmed: the sum of merge scores over a game
is exactly the game's final score, so maximising expected return is literally
maximising expected score. Every other signal people reach for has a failure
mode:

``survival``
    Paying per move rewards stalling. An agent can shuffle tiles back and
    forth in a half-empty board racking up reward without building anything.
``empty cells``
    Rewarding empty space directly rewards *not merging into big tiles*, and
    an agent can hold a near-empty board indefinitely early on.
``max tile``
    Paying repeatedly for holding a big tile rewards sitting still. Paying
    once, at the moment the tile is first created, does not -- so that is how
    it is implemented here.
``corner / monotonicity bonuses``
    These hand the agent the strategy instead of letting it find one, which
    defeats the point of the exercise. None are enabled by default.

Shaping terms that *are* offered are implemented as **potential-based
shaping**: the agent receives ``gamma * phi(s') - phi(s)``. Ng, Harada &
Russell (1999) showed this form leaves the optimal policy unchanged -- it can
only change how fast learning gets there, never what it converges to. That
makes the shaped variants safe to experiment with without silently training a
different game.

Set weights in a config file and compare runs; see ``config/experiments/``.
"""

from __future__ import annotations

from engine import board as B

DEFAULT_REWARD = {
    # The game's own score from merges. This is the whole reward by default.
    "score": 1.0,
    # Paid ONCE, when a tile larger than any seen this game first appears.
    # Value is `new_tile_value * milestone`.
    "milestone": 0.0,
    # Subtracted once when the game ends. Teaches "dying is bad" beyond the
    # mere absence of future reward. Risk: makes values negative early, which
    # slows learning; off by default.
    "game_over_penalty": 0.0,
    # Potential-based shaping on empty-cell count. Policy-invariant.
    "empty_potential": 0.0,
    # Flat bonus per surviving move. NOT potential-based and farmable;
    # provided only so the exploit can be demonstrated. Off by default.
    "survival": 0.0,
    # Discount used by the potential shaping term (match the trainer's gamma).
    "gamma": 1.0,
}


class RewardFunction:
    """Turns a transition into a scalar reward.

    Called as ``fn(state, afterstate, merge_score, max_exp_before)`` and
    returns ``(reward, max_exp_after)``. The trainer threads the running
    maximum exponent through so milestones are paid exactly once per game.
    """

    def __init__(self, cfg: dict | None = None):
        w = dict(DEFAULT_REWARD)
        if cfg:
            unknown = set(cfg) - set(DEFAULT_REWARD)
            if unknown:
                raise ValueError(f"unknown reward keys: {sorted(unknown)}")
            w.update(cfg)
        self.w = w
        self.score = w["score"]
        self.milestone = w["milestone"]
        self.game_over_penalty = w["game_over_penalty"]
        self.empty_potential = w["empty_potential"]
        self.survival = w["survival"]
        self.gamma = w["gamma"]
        # Fast path: the default (pure score) skips all the extra work.
        self.is_pure_score = (
            self.score == 1.0 and self.milestone == 0.0
            and self.empty_potential == 0.0 and self.survival == 0.0)

    def step(self, state: int, afterstate: int, merge_score: int,
             max_exp: int):
        """Reward for one player move. Returns ``(reward, new_max_exp)``."""
        r = self.score * merge_score
        if self.survival:
            r += self.survival
        if self.milestone:
            m = B.max_tile_exp(afterstate)
            if m > max_exp:
                r += self.milestone * (1 << m)
                max_exp = m
        if self.empty_potential:
            # phi(s) = empty_potential * empty_cells(s)
            phi_s = self.empty_potential * B.empty_count(state)
            phi_a = self.empty_potential * B.empty_count(afterstate)
            r += self.gamma * phi_a - phi_s
        return r, max_exp

    def terminal(self) -> float:
        """Extra reward applied when the game ends."""
        return -self.game_over_penalty

    def describe(self) -> dict:
        return dict(self.w)
