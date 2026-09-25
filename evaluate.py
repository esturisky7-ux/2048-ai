#!/usr/bin/env python3
"""Evaluate and compare 2048 agents with a fixed, repeatable procedure.

Examples (write ``python`` instead of ``python3`` on Windows):

    python3 evaluate.py --games 1000
    python3 evaluate.py --agent heuristic --games 2000
    python3 evaluate.py --compare random heuristic learned --games 500
    python3 evaluate.py --agent expectimax --depth 3 --prob-cutoff 5e-3 --games 50
    python3 evaluate.py --agent learned --checkpoint checkpoints/default/snapshots/games-000100000.f32

Every agent sees the *same* seeded games, so differences between agents are
differences in play, not luck. Results are appended to
``data/<run>/evaluations.jsonl`` unless ``--no-save`` is given.

A run's current weights can only be evaluated while nothing is training it:
they would change from one game to the next. Evaluate a snapshot instead
(``train.py --snapshot-every N`` takes them), or stop training first.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.learned import live_run_for                    # noqa: E402
from agents.registry import make_agent, AGENT_NAMES        # noqa: E402
from evaluation.evaluator import evaluate, format_report   # noqa: E402
from training.checkpoint import (Run, display_path,       # noqa: E402
                                 python_command)
from training.runlock import RunBusy, hold_frozen          # noqa: E402
from version import version_string                         # noqa: E402


def build_agent(name: str, args):
    kw = {}
    if name == "expectimax":
        kw.update(depth=args.depth, prob_cutoff=args.prob_cutoff,
                  adaptive=args.adaptive)
    elif name == "learned":
        kw.update(run=args.run, depth=args.depth if args.depth_given else 1)
        if args.checkpoint:
            kw["checkpoint"] = args.checkpoint
        if args.prob_cutoff:
            kw["prob_cutoff"] = args.prob_cutoff
    return make_agent(name, seed=args.agent_seed, **kw)


def comparison_table(results: dict) -> str:
    hdr = (f"{'agent':<14}{'games':>7}{'mean':>11}{'95% CI':>21}"
           f"{'median':>10}{'best':>10}{'tile':>8}{'2048%':>8}{'g/s':>9}")
    lines = [hdr, "-" * len(hdr)]
    for name, r in results.items():
        if not r.get("games"):
            lines.append(f"{name:<14}  (no games)")
            continue
        ci = f"[{r['ci95_mean'][0]:,.0f}, {r['ci95_mean'][1]:,.0f}]"
        lines.append(
            f"{name:<14}{r['games']:>7,}{r['mean_score']:>11,.0f}{ci:>21}"
            f"{r['median_score']:>10,.0f}{r['max_score']:>10,}"
            f"{r['highest_tile']:>8,}"
            f"{r['tile_rates']['2048']['rate']*100:>7.1f}%"
            f"{r['games_per_second']:>9.2f}")
    return "\n".join(lines)


def main() -> int:
    py = python_command()
    p = argparse.ArgumentParser(
        prog=f"{py} evaluate.py",
        description="Evaluate 2048 agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.replace("python3 ", f"{py} "))
    p.add_argument("--games", type=int, default=1000,
                   help="games to play (default 1000)")
    p.add_argument("--agent", default="learned", choices=AGENT_NAMES,
                   help="which agent to evaluate (default learned)")
    p.add_argument("--compare", nargs="+", metavar="AGENT", default=None,
                   help=f"evaluate several agents on identical games "
                        f"({'/'.join(AGENT_NAMES)})")
    p.add_argument("--seed", type=int, default=None,
                   help="evaluation seed (default: the run's config)")
    p.add_argument("--agent-seed", type=int, default=0,
                   help="seed for agents that make random choices")
    p.add_argument("--run", default="default", help="training run to load")
    p.add_argument("--checkpoint", default=None,
                   help="explicit .f32 weight file (e.g. a snapshot)")
    p.add_argument("--depth", type=int, default=2,
                   help="search depth for expectimax / learned")
    p.add_argument("--prob-cutoff", type=float, default=1e-3,
                   help="expectimax: stop expanding lines rarer than this")
    p.add_argument("--adaptive", action="store_true",
                   help="expectimax: search deeper when the board is crowded")
    p.add_argument("--out", default=None, help="write JSON results here")
    p.add_argument("--no-save", action="store_true",
                   help="do not append to the run's evaluations.jsonl")
    p.add_argument("--quiet", action="store_true",
                   help="no progress line while games are running")
    p.add_argument("--version", action="version", version=version_string())
    args = p.parse_args()
    # --depth and --depth=N are both explicit; the learned agent defaults to
    # greedy (depth 1) unless the user actually asked for search.
    args.depth_given = any(a == "--depth" or a.startswith("--depth=")
                           for a in sys.argv)

    run = Run(args.run)
    cfg = run.load_config() or {}
    seed = args.seed if args.seed is not None else \
        cfg.get("evaluation", {}).get("seed", 987654)

    names = args.compare if args.compare else [args.agent]
    for n in names:
        if n not in AGENT_NAMES:
            print(f"error: unknown agent {n!r}; choose from {AGENT_NAMES}",
                  file=sys.stderr)
            return 2

    # Hold a learned agent's current weights still for the whole run -- or
    # refuse before playing a single game, if that run is being trained.
    frozen = contextlib.ExitStack()
    live = [live_run_for(args.run, args.checkpoint)] \
        if "learned" in names else []
    try:
        frozen.enter_context(hold_frozen(live, purpose="evaluation, "
                                                        "evaluate.py"))
    except RunBusy as e:
        print(f"error: {e}\n"
              f"       A snapshot looks like:  {py} evaluate.py --checkpoint "
              f"{display_path(Run(e.run).snapshot_path(50000))}",
              file=sys.stderr)
        return 1

    results = {}
    with frozen:
        for name in names:
            try:
                agent = build_agent(name, args)
            except FileNotFoundError as e:
                print(f"error: {e}\n"
                      f"       Train an agent first, for example:\n"
                      f"           {py} train.py --games 20000",
                      file=sys.stderr)
                return 1

            # Only draw the carriage-return progress line on a real terminal;
            # piped or redirected output would otherwise collect every update.
            show_progress = sys.stdout.isatty() and not args.quiet

            def progress(done, total, _n=name):
                if show_progress:
                    print(f"\r  {_n}: {done}/{total} games", end="",
                          flush=True)

            try:
                res = evaluate(agent, games=args.games, seed=seed,
                               progress=progress if show_progress else None)
            except KeyboardInterrupt:
                if show_progress:
                    print("\r" + " " * 40, end="\r")
                print("interrupted; no results written", file=sys.stderr)
                return 130
            finally:
                if hasattr(agent, "close"):
                    agent.close()
            if show_progress:
                print("\r" + " " * 40, end="\r")
            results[name] = res
            if not args.compare:
                print(format_report(res))

    if args.compare:
        print(f"\nsame {args.games} seeded games for every agent (seed {seed})\n")
        print(comparison_table(results))

    payload = {"timestamp": time.time(), "seed": seed, "games": args.games,
               "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")
    if not args.no_save and not args.compare and args.agent == "learned":
        run.create_dirs()
        with open(run.eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(results["learned"], separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
