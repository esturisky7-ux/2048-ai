#!/usr/bin/env python3
"""Train the 2048 learning agent.

Examples (write ``python`` instead of ``python3`` on Windows):

    python3 train.py --games 100000        train 100k games, then stop
    python3 train.py --resume              continue where you left off
    python3 train.py --resume --workers 2  use two CPU cores
    python3 train.py --list-runs           what has been trained so far
    python3 train.py --config baseline --run baseline

Training checkpoints periodically and always saves on exit, so Ctrl-C is a
safe way to stop. See README.md for the full story.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.config import load_config, list_experiments   # noqa: E402
from training.checkpoint import Run, list_runs, python_command   # noqa: E402
from training.ntuple import TUPLE_SETS                      # noqa: E402
from version import version_string                          # noqa: E402


def build_overrides(args) -> dict:
    """CLI flags -> a config fragment layered on top of the file config."""
    o: dict = {"learning": {}, "training": {}, "reward": {}}
    if args.alpha is not None:
        o["learning"]["alpha"] = args.alpha
    if args.alpha_decay is not None:
        o["learning"]["alpha_decay"] = args.alpha_decay
    if args.epsilon is not None:
        o["learning"]["epsilon"] = args.epsilon
    if args.gamma is not None:
        o["learning"]["gamma"] = args.gamma
    if args.seed is not None:
        o["training"]["seed"] = args.seed
    if args.checkpoint_every is not None:
        o["training"]["checkpoint_every"] = args.checkpoint_every
    if args.report_every is not None:
        o["training"]["report_every"] = args.report_every
    if args.eval_every is not None:
        o["training"]["eval_every"] = args.eval_every
    if args.eval_games is not None:
        o["training"]["eval_games"] = args.eval_games
    if args.snapshot_every is not None:
        o["training"]["snapshot_every"] = args.snapshot_every
    if args.tuple_set is not None:
        o["tuple_set"] = args.tuple_set
    return {k: v for k, v in o.items() if v or k not in ("learning", "training", "reward")}


def self_check() -> int:
    """Play a handful of games end to end and report that the install works.

    Deliberately uses the smallest network and a throwaway run name, so it
    finishes in seconds on any machine and leaves nothing behind.
    """
    import shutil
    import tempfile
    from training.config import load_config as _load

    print(version_string())
    print(f"project root  {Path(__file__).resolve().parent}")
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10 or newer is required "
              f"(this is {sys.version.split()[0]})", file=sys.stderr)
        return 1

    from engine import board as B
    b = B.from_list([2, 2, 4, 4] + [0] * 12)
    nb, gained, moved = B.move(b, B.LEFT)
    ok = moved and gained == 12 and B.to_list(nb)[:4] == [4, 8, 0, 0]
    print(f"engine        {'ok' if ok else 'FAILED'}  "
          f"(merge rules, {B.empty_count(0)} empty cells on a blank board)")
    if not ok:
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="2048-check-"))
    try:
        from training import checkpoint as CP
        real_cp, real_data = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        CP.CHECKPOINT_ROOT = tmp / "checkpoints"
        CP.DATA_ROOT = tmp / "data"
        try:
            cfg = _load(overrides={"run": "selfcheck", "tuple_set": "8x4",
                                   "training": {"report_every": 0,
                                                "checkpoint_every": 0,
                                                "eval_every": 0}})
            from training.trainer import Trainer, worker_start_method
            t = Trainer("selfcheck", cfg, quiet=True)
            t.train(n_games=20)
            print(f"training      ok  (20 games, best score "
                  f"{t.all_time.best_score:,})")
            print(f"checkpoint    ok  (wrote {t.run.meta_path.name} and "
                  f"weights.f32)")
            print(f"workers       start method '{worker_start_method()}'")
        finally:
            CP.CHECKPOINT_ROOT, CP.DATA_ROOT = real_cp, real_data
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    py = python_command()
    print(f"\neverything works. Start real training with:\n"
          f"    {py} train.py --games 20000")
    return 0


def main() -> int:
    cores = os.cpu_count() or 1
    py = python_command()
    p = argparse.ArgumentParser(
        prog=f"{py} train.py",
        description="Train a 2048 agent by temporal-difference learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.replace("python3 ", f"{py} "))
    p.add_argument("--games", type=int, default=0,
                   help="games to play this session (0 = until Ctrl-C)")
    p.add_argument("--resume", action="store_true",
                   help="continue an existing run from its checkpoint")
    p.add_argument("--run", default=None,
                   help="run name (default: 'default', or the config's)")
    p.add_argument("--config", default=None,
                   help="config file, or a name from config/experiments/")
    p.add_argument("--workers", type=int, default=1,
                   help=f"training processes (this machine has {cores} "
                        f"CPU core{'s' if cores != 1 else ''})")
    p.add_argument("--tuple-set", default=None, choices=sorted(TUPLE_SETS),
                   help="n-tuple network shape (new runs only)")
    p.add_argument("--alpha", type=float, default=None, help="learning rate")
    p.add_argument("--alpha-decay", type=float, default=None,
                   help="geometric decay factor per alpha_decay_every games")
    p.add_argument("--epsilon", type=float, default=None,
                   help="random-move probability (default 0)")
    p.add_argument("--gamma", type=float, default=None, help="discount")
    p.add_argument("--seed", type=int, default=None, help="tile-spawn seed")
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--report-every", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=None,
                   help="freeze a copy of the weights every N games (0 = off)")
    p.add_argument("--eval-every", type=int, default=None,
                   help="run the fixed evaluation every N games (0 = off)")
    p.add_argument("--eval-games", type=int, default=None)
    p.add_argument("--evaluate-now", action="store_true",
                   help="evaluate the current checkpoint and exit")
    p.add_argument("--list-runs", action="store_true",
                   help="show every run that has a checkpoint, then exit")
    p.add_argument("--list-experiments", action="store_true",
                   help="show the configs in config/experiments/, then exit")
    p.add_argument("--check", action="store_true",
                   help="verify the install can train, then exit")
    p.add_argument("--quiet", action="store_true",
                   help="suppress the periodic progress lines")
    p.add_argument("--version", action="version", version=version_string())
    args = p.parse_args()

    if args.check:
        return self_check()

    if args.list_runs:
        runs = list_runs()
        if not runs:
            print("no runs yet")
            return 0
        print(f"{'run':<24}{'games':>12}  {'best score':>11}  "
              f"{'best tile':>9}  tuple set")
        for r in runs:
            print(f"{r['name']:<24}{r['games']:>12,}  {r['best_score']:>11,}  "
                  f"{r['best_tile']:>9,}  {r['tuple_set']}")
        return 0

    if args.list_experiments:
        names = list_experiments()
        print("\n".join(names) if names else "no experiment configs")
        return 0

    if args.workers > cores:
        print(f"note: this machine has {cores} CPU core"
              f"{'s' if cores != 1 else ''}; --workers {args.workers} "
              f"will oversubscribe it and is likely to be slower.")

    overrides = build_overrides(args)
    if args.run:
        overrides["run"] = args.run

    if args.resume:
        run_name = args.run or (load_config(args.config).get("run", "default")
                                if args.config else "default")
        if not Run(run_name).exists():
            print(f"error: no checkpoint for run {run_name!r}. "
                  f"Start one with:  {py} train.py --games 20000",
                  file=sys.stderr)
            return 1
        cfg_over = overrides
    else:
        cfg = load_config(args.config, overrides)
        run_name = cfg["run"]
        cfg_over = cfg

    from training.trainer import Trainer
    trainer = Trainer(run_name, cfg_over, resume=args.resume, quiet=args.quiet)

    if args.evaluate_now:
        trainer.eval_games = args.eval_games or trainer.eval_games
        trainer.evaluate_now()
        trainer.save(note="manual evaluation")
        return 0

    trainer.train(n_games=args.games, workers=max(1, args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
