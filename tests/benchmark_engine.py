"""Measure raw simulator throughput.

Run before optimising anything: `python3 tests/benchmark_engine.py`.
Reports moves/second and games/second for a uniformly-random player, which is
the floor on how fast any agent can possibly run.
"""
import os
import sys
import time
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import board as B  # noqa: E402


def random_rollout(rng):
    """One full game, random legal moves. Returns (score, moves)."""
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


def random_rollout_fast(rng):
    """Same, but tries actions in a random order instead of listing them all.

    Avoids computing all four moves per step, which is the dominant cost of
    the naive loop.
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


def bench(fn, n_games, label):
    rng = Random(12345)
    t0 = time.perf_counter()
    total_moves = 0
    total_score = 0
    best = 0
    for _ in range(n_games):
        s, m = fn(rng)
        total_moves += m
        total_score += s
        best = max(best, s)
    dt = time.perf_counter() - t0
    print(f"{label:28s} {n_games:6d} games  {dt:6.2f}s  "
          f"{n_games/dt:8.1f} games/s  {total_moves/dt:10.0f} moves/s  "
          f"avg score {total_score/n_games:7.1f}")
    return total_moves / dt


def micro():
    """Per-primitive timing, to see where the time actually goes."""
    import timeit
    setup = ("import sys; sys.path.insert(0, %r)\n"
             "from engine import board as B\n"
             "from random import Random\n"
             "rng = Random(1)\n"
             "b = B.from_list([2,4,8,16, 32,64,128,256, 2,4,8,16, 32,64,2,4])\n"
             % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cases = [
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
    ]
    print("\n-- per-call cost --")
    for name, stmt in cases:
        n = 200000
        t = timeit.timeit(stmt, setup=setup, number=n)
        print(f"  {name:26s} {t/n*1e6:7.3f} us")


if __name__ == "__main__":
    print("2048 engine benchmark (pure Python bitboard)\n")
    bench(random_rollout, 2000, "random (all legal moves)")
    bench(random_rollout_fast, 2000, "random (try-in-order)")
    micro()
