#!/usr/bin/env python3
"""Run controlled experiments and compare their results.

Examples (write ``python`` instead of ``python3`` on Windows):

    python3 experiment.py --list
    python3 experiment.py --run reward-milestone --games 5000
    python3 experiment.py --run-all --games 3000
    python3 experiment.py --compare

Results (including the exact config used) land in data/experiments/.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.runner import (run_experiment, list_experiments,      # noqa: E402
                                load_results, comparison_table, result_path)
from training.checkpoint import python_command                         # noqa: E402
from version import version_string                                     # noqa: E402


def main() -> int:
    py = python_command()
    p = argparse.ArgumentParser(
        prog=f"{py} experiment.py",
        description="Run controlled 2048 training/agent experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.replace("python3 ", f"{py} "))
    p.add_argument("--list", action="store_true", help="show available experiments")
    p.add_argument("--run", metavar="NAME", help="run one experiment")
    p.add_argument("--run-all", action="store_true", help="run every experiment in turn")
    p.add_argument("--compare", action="store_true", help="table of saved results")
    p.add_argument("--games", type=int, default=None, help="override training games")
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fresh", action="store_true",
                   help="delete any existing run for the experiment first")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-experiment training output")
    p.add_argument("--version", action="version", version=version_string())
    a = p.parse_args()

    if a.list or not (a.run or a.run_all or a.compare):
        exps = list_experiments()
        if not exps:
            print("no experiments in config/experiments/")
            return 0
        print(f"{'name':<26}{'kind':<8}{'result':<9}description")
        print("-" * 96)
        for e in exps:
            print(f"{e['name']:<26}{e['kind']:<8}"
                  f"{'yes' if e['has_result'] else '-':<9}{e['description']}")
        print(f"\nrun one with:  {py} experiment.py --run <name> --games 5000")
        return 0

    if a.compare:
        print(comparison_table(load_results()))
        return 0

    names = [e["name"] for e in list_experiments()] if a.run_all else [a.run]
    for i, name in enumerate(names, 1):
        print(f"\n=== [{i}/{len(names)}] experiment: {name} ===")
        try:
            res = run_experiment(name, games=a.games, eval_games=a.eval_games,
                                 workers=a.workers, fresh=a.fresh, quiet=a.quiet)
        except FileNotFoundError as e:
            print(f"  error: {e}", file=sys.stderr)
            return 1
        ev = res.get("evaluation", {})
        if ev.get("games"):
            print(f"  -> mean {ev['mean_score']:,.0f} "
                  f"[{ev['ci95_mean'][0]:,.0f}, {ev['ci95_mean'][1]:,.0f}]  "
                  f"median {ev['median_score']:,.0f}  "
                  f"best tile {ev['highest_tile']:,}")
        print(f"  saved {os.path.relpath(result_path(name))}")

    print()
    print(comparison_table(load_results()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
