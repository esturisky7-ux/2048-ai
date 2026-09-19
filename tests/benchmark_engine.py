"""Measure raw simulator throughput.

Run before optimising anything: `python3 tests/benchmark_engine.py`.
Reports moves/second and games/second for a uniformly-random player, which is
the floor on how fast any agent can possibly run.

The measurements themselves live in :mod:`engine.benchmark` so that this
command and the control center's Benchmark page run identical code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import benchmark as bench  # noqa: E402


def show(result: dict, label: str) -> None:
    print(f"{label:28s} {result['games']:6d} games  {result['seconds']:6.2f}s  "
          f"{result['games_per_second']:8.1f} games/s  "
          f"{result['moves_per_second']:10.0f} moves/s  "
          f"avg score {result['mean_score']:7.1f}")


if __name__ == "__main__":
    print("2048 engine benchmark (pure Python bitboard)\n")
    show(bench.bench_rollouts(bench.random_rollout, 2000),
         "random (all legal moves)")
    show(bench.bench_rollouts(bench.random_rollout_fast, 2000),
         "random (try-in-order)")
    print("\n-- per-call cost --")
    for row in bench.micro():
        print(f"  {row['name']:26s} {row['microseconds']:7.3f} us")
