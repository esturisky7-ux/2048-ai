"""Engine throughput measurements.

These are *speed* measurements of the machine running them, and say nothing
about how well any agent plays. Keep them mentally separate from evaluation
results, which measure an agent on a fixed set of seeded games.

The functions live here rather than in ``tests/`` so that both the command-line
benchmark (``python3 tests/benchmark_engine.py``) and the control center's
Benchmark page run exactly the same code and cannot drift apart.
"""

from __future__ import annotations

import time
import timeit
from pathlib import Path
from random import Random

from . import board as B

ROOT = Path(__file__).resolve().parent.parent


def random_rollout(rng: Random):
    """One full game of uniformly random legal moves. Returns (score, moves)."""
    b = B.new_game(rng)
    score = moves = 0
    move = B.move
    spawn = B.random_spawn
    choice = rng.choice
    ACTIONS = B.ACTIONS
    while True:
        legal = [a for a in ACTIONS if move(b, a)[2]]
        if not legal:
            return score, moves
        nb, gained, _ = move(b, choice(legal))
        b = spawn(nb, rng)
        score += gained
        moves += 1


def random_rollout_fast(rng: Random):
    """Same, but tries actions in a random order instead of listing them all.

    Avoids computing all four moves per step, which is the dominant cost of the
    naive loop.
    """
    b = B.new_game(rng)
    score = moves = 0
    move = B.move
    spawn = B.random_spawn
    shuffle = rng.shuffle
    order = [0, 1, 2, 3]
    while True:
        shuffle(order)
        for a in order:
            nb, gained, moved = move(b, a)
            if moved:
                b = spawn(nb, rng)
                score += gained
                moves += 1
                break
        else:
            return score, moves


def bench_rollouts(fn, n_games: int, seed: int = 12345) -> dict:
    """Run ``n_games`` rollouts and report throughput."""
    rng = Random(seed)
    t0 = time.perf_counter()
    total_moves = 0
    total_score = 0
    best = 0
    for _ in range(n_games):
        s, m = fn(rng)
        total_moves += m
        total_score += s
        if s > best:
            best = s
    dt = time.perf_counter() - t0
    return {
        "games": n_games,
        "seconds": dt,
        "games_per_second": n_games / dt if dt else 0.0,
        "moves_per_second": total_moves / dt if dt else 0.0,
        "moves": total_moves,
        "mean_score": total_score / n_games if n_games else 0.0,
        "best_score": best,
    }


# Primitives timed one call at a time, to show where the time actually goes.
MICRO_CASES = (
    ("move LEFT", "B.move(b, 3)"),
    ("move UP (1 transpose)", "B.move(b, 0)"),
    ("move_no_score LEFT", "B.move_no_score(b, 3)"),
    ("move_no_score UP (1 transpose)", "B.move_no_score(b, 0)"),
    ("transpose", "B.transpose(b)"),
    ("empty_count", "B.empty_count(b)"),
    ("max_tile_exp", "B.max_tile_exp(b)"),
    ("is_game_over", "B.is_game_over(b)"),
    ("legal_actions", "B.legal_actions(b)"),
    ("random_spawn", "B.random_spawn(0, rng)"),
)


def micro(number: int = 200000) -> list[dict]:
    """Per-call cost of each engine primitive, in microseconds."""
    setup = ("import sys; sys.path.insert(0, %r)\n"
             "from engine import board as B\n"
             "from random import Random\n"
             "rng = Random(1)\n"
             "b = B.from_list([2,4,8,16, 32,64,128,256, 2,4,8,16, 32,64,2,4])\n"
             % str(ROOT))
    out = []
    for name, stmt in MICRO_CASES:
        t = timeit.timeit(stmt, setup=setup, number=number)
        out.append({"name": name, "microseconds": t / number * 1e6})
    return out
