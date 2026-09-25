"""TD(0) learning on afterstates -- the algorithm the learning agent uses.

The idea
--------
A move in 2048 has two halves: the player slides (deterministic), then the
game drops a tile (random). The board between the two is the **afterstate**.
Learning values of afterstates rather than states is the single most important
design choice here, because it takes all the randomness out of the thing being
learned: the value of an afterstate is a property of the player's decision
alone. Picking a move is then just

    argmax over legal a of   reward(a) + V(afterstate(a))

with no search and no model of the spawn distribution -- one network
evaluation per legal move, roughly four per turn.

The update
----------
After the game drops its tile we are in a new state, where the agent again
picks its best move, yielding reward ``r'`` and afterstate ``s''``. The
previous afterstate is then corrected towards what actually happened:

    V(s') <- V(s') + alpha * [ r' + gamma * V(s'') - V(s') ]

and at the end of a game the target is just the terminal reward (0 by
default), which is what teaches the agent that dying forfeits all future
score. Because the value function is linear in the n-tuple features, applying
that correction means adding the same small delta to each of the 32 weights
the afterstate activates; ``alpha`` is divided by 32 so the total change to
V(s') is the intended one.

Exploration
-----------
None, by default. This looks wrong and is not: the tile spawns inject plenty
of stochasticity, every game visits a different part of the state space, and
epsilon-greedy is known to *hurt* on 2048 because a single random move late in
a game can destroy a structure that took hundreds of moves to build. An
``epsilon`` knob exists for experiments and is on-policy when used.

Playing the trained policy
--------------------------
``V`` estimates the return of whatever reward the run was trained on, with
that run's discount, so the only policy it describes is the one training
followed: ``argmax r(a) + gamma * V(afterstate(a))`` with the *configured*
reward -- shaping terms included, and with milestone bonuses tracked through
the game exactly as training tracks them. :func:`best_move` is that one scorer.
Training, the trainer's periodic evaluation, experiments and the loaded
:class:`~agents.learned.LearnedAgent` all go through it (via
:class:`GreedyPolicy`), so they cannot drift apart. With the default reward
(pure merge score, gamma 1) it is exactly ``merge score + V(afterstate)``.
"""

from __future__ import annotations

from random import Random

from engine import board as B
from .reward import RewardFunction

_MOVE = B.move
_SPAWN = B.random_spawn
_NEG_INF = float("-inf")


def best_move(b: int, max_exp: int, value, gamma: float,
              reward: RewardFunction, move=_MOVE):
    """The move the value function prefers from ``b``, scored as training does.

    Each legal move scores ``r + gamma * value(afterstate)``, where ``r`` is
    ``reward.step(...)`` -- or just the merge score when the reward is pure
    score, which is the same number computed faster. ``max_exp`` is the
    largest tile exponent this game has produced so far, which milestone
    rewards need. Ties go to the first move in UP, RIGHT, DOWN, LEFT order.

    Returns ``(v, (action, afterstate, value, r, merge_score))`` for the best
    move, or ``(-inf, None)`` when no move is legal.
    """
    pure = reward.is_pure_score
    best_v = _NEG_INF
    best = None
    for a in (0, 1, 2, 3):
        nb, gained, moved = move(b, a)
        if not moved:
            continue
        val = value(nb)
        r = gained if pure else reward.step(b, nb, gained, max_exp)[0]
        v = r + gamma * val
        if v > best_v:
            best_v = v
            best = (a, nb, val, r, gained)
    return best_v, best


class GreedyPolicy:
    """Training's choice of move, with no exploration and no learning.

    Stateful per game only because milestone rewards are: a bonus is paid the
    first time a game produces a new largest tile, so the policy remembers the
    largest one so far, exactly as training does. Call :meth:`new_game` before
    each game, and play the move :meth:`act` returns.
    """

    def __init__(self, net, reward: RewardFunction | None = None,
                 gamma: float = 1.0):
        self.net = net
        self.reward = reward or RewardFunction()
        self.gamma = float(gamma)
        self.max_exp = 0

    def new_game(self) -> None:
        self.max_exp = 0

    def act(self, b: int) -> int:
        """The move to play from ``b``, or -1 when there is none."""
        _, best = best_move(b, self.max_exp, self.net.value, self.gamma,
                            self.reward)
        if best is None:
            return -1
        a, nb, _, _, gained = best
        if not self.reward.is_pure_score:
            self.max_exp = self.reward.step(b, nb, gained, self.max_exp)[1]
        return a


class TDLearner:
    """Plays 2048 with an n-tuple value function, optionally learning as it goes.

    Parameters
    ----------
    net:
        An :class:`~training.ntuple.NTupleNetwork`.
    reward:
        A :class:`~training.reward.RewardFunction` (default: pure merge score).
    alpha:
        Total learning rate. Divided internally by the number of activated
        weights so it means "how far V(s') moves toward its target".
    gamma:
        Discount. 1.0 is correct for 2048 -- the game is finite and undiscounted
        score is exactly what we want to maximise.
    epsilon:
        Probability of a uniformly random legal move (on-policy when non-zero).
    """

    def __init__(self, net, reward: RewardFunction | None = None,
                 alpha: float = 0.1, gamma: float = 1.0,
                 epsilon: float = 0.0):
        self.net = net
        self.reward = reward or RewardFunction()
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)
        # One update touches n_variants weights; divide so that `alpha` is the
        # change applied to V(s') as a whole.
        self.alpha_per_weight = float(alpha) / self.net.n_variants

    # -- greedy policy (no learning) --------------------------------------
    def best_action(self, b: int, max_exp: int = 0) -> int:
        """The move the current value function prefers from one position.

        ``max_exp`` is the largest tile exponent the game has produced so far
        (it only matters for milestone rewards). To play whole games, use
        :meth:`policy`, which keeps track of it.
        """
        _, best = best_move(b, max_exp, self.net.value, self.gamma,
                            self.reward)
        return best[0] if best is not None else -1

    def policy(self) -> GreedyPolicy:
        """This learner's greedy policy: how it plays when it is not learning."""
        return GreedyPolicy(self.net, self.reward, self.gamma)

    # -- one game ----------------------------------------------------------
    def play_game(self, rng: Random, learn: bool = True):
        """Play one game start to finish.

        Returns ``(score, moves, max_tile)``. When ``learn`` is True the
        network is updated online, once per move.
        """
        net = self.net
        value = net.value
        update = net.update_scaled
        alpha = self.alpha_per_weight
        gamma = self.gamma
        rf = self.reward
        pure = rf.is_pure_score
        eps = self.epsilon
        move = _MOVE
        spawn = _SPAWN
        choose = best_move

        b = B.new_game(rng)
        score = 0
        moves = 0
        max_exp = 0
        prev_after = -1
        prev_v = 0.0

        while True:
            # ---- choose a move -------------------------------------------
            best_v = _NEG_INF
            best_after = -1
            best_val = 0.0
            best_r = 0.0
            best_gain = 0

            if eps and rng.random() < eps:
                legal = []
                for a in (0, 1, 2, 3):
                    nb, gained, moved = move(b, a)
                    if moved:
                        legal.append((nb, gained))
                if legal:
                    best_after, best_gain = legal[rng.randrange(len(legal))]
                    best_val = value(best_after)
                    best_r = (best_gain if pure else
                              rf.step(b, best_after, best_gain, max_exp)[0])
                    best_v = best_r + gamma * best_val
            else:
                # The same scorer every evaluation of this policy uses.
                v, best = choose(b, max_exp, value, gamma, rf, move)
                if best is not None:
                    best_v = v
                    _, best_after, best_val, best_r, best_gain = best

            # ---- terminal? -----------------------------------------------
            # No legal move exists exactly when the game is over, so this
            # doubles as the game-over test and costs nothing extra.
            if best_after < 0:
                if learn and prev_after >= 0:
                    update(prev_after, rf.terminal() - prev_v, alpha)
                break

            # ---- learn ---------------------------------------------------
            if learn and prev_after >= 0:
                # TD error: (r' + gamma V(s'')) - V(s')
                update(prev_after, best_v - prev_v, alpha)

            if not pure:
                _, max_exp = rf.step(b, best_after, best_gain, max_exp)

            prev_after = best_after
            prev_v = best_val

            score += best_gain
            moves += 1
            b = spawn(best_after, rng)

        return score, moves, B.max_tile(b)
