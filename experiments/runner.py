"""Controlled experiments: vary one thing, measure, keep the config with the result.

An experiment is a JSON file in ``config/experiments/``. Two kinds:

``"kind": "train"``  (the default)
    Train a fresh run with the given config for N games, then evaluate it with
    the fixed evaluation procedure. Use this to compare reward functions,
    learning rates, network shapes and other training parameters.

``"kind": "agent"``
    Skip training and just evaluate an agent configuration. Use this to
    compare search depths and hand-written evaluation-function weights.

Every result is written to ``data/experiments/<name>.json`` **with the full
resolved config inside it**, so months later it is still possible to say
exactly what produced a number. Seeds are fixed by default so two experiments
differ only in what you changed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from training.config import load_config, resolve_path, EXPERIMENT_DIR
from training.checkpoint import Run, atomic_write_json

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "experiments"


def result_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def load_experiment(name: str) -> dict:
    with open(resolve_path(name), encoding="utf-8") as f:
        return json.load(f)


def list_experiments() -> list:
    exp_dir = Path(EXPERIMENT_DIR)
    if not exp_dir.is_dir():
        return []
    out = []
    for path in sorted(exp_dir.iterdir()):
        if path.suffix != ".json":
            continue
        name = path.stem
        try:
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, json.JSONDecodeError):
            spec = {}
        out.append({
            "name": name,
            "kind": spec.get("kind", "train"),
            "description": spec.get("description", ""),
            "has_result": result_path(name).exists(),
        })
    return out


def run_experiment(name: str, games: int | None = None,
                   eval_games: int | None = None, workers: int = 1,
                   fresh: bool = False, quiet: bool = False) -> dict:
    """Run one experiment end to end and save its result."""
    spec = load_experiment(name)
    kind = spec.get("kind", "train")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if kind == "agent":
        res = _run_agent_experiment(spec, eval_games, quiet)
        payload = {
            "experiment": name, "kind": kind,
            "description": spec.get("description", ""),
            "config": spec, "evaluation": res,
            "started_at": started, "finished_at": time.time(),
        }
    else:
        payload = _run_train_experiment(name, spec, games, eval_games,
                                        workers, fresh, quiet)
        payload["started_at"] = started
        payload["finished_at"] = time.time()

    atomic_write_json(result_path(name), payload)
    return payload


def _run_agent_experiment(spec: dict, eval_games, quiet) -> dict:
    from agents.registry import make_agent
    from evaluation.evaluator import evaluate

    a = dict(spec.get("agent") or {})
    agent_name = a.pop("name", "expectimax")
    n = eval_games or spec.get("eval_games", 200)
    seed = spec.get("eval_seed", 987654)
    agent = make_agent(agent_name, seed=spec.get("agent_seed", 0), **a)
    if not quiet:
        print(f"  evaluating {agent_name} over {n} games (seed {seed})...")
    res = evaluate(agent, games=n, seed=seed)
    if hasattr(agent, "close"):
        agent.close()
    return res


def _run_train_experiment(name, spec, games, eval_games, workers,
                          fresh, quiet) -> dict:
    import shutil
    from training.trainer import Trainer
    from evaluation.evaluator import evaluate

    cfg = load_config(overrides={k: v for k, v in spec.items()
                                 if k not in ("kind", "description",
                                              "games", "eval_games",
                                              "eval_seed", "agent",
                                              "agent_seed")})
    run_name = spec.get("run", f"exp-{name}")
    cfg["run"] = run_name
    n_games = games or spec.get("games", 5000)
    n_eval = eval_games or spec.get("eval_games", 200)

    run = Run(run_name)
    if fresh and run.exists():
        shutil.rmtree(run.dir, ignore_errors=True)
        shutil.rmtree(run.data_dir, ignore_errors=True)

    resume = run.exists()
    trainer = Trainer(run_name, cfg if not resume else None,
                      resume=resume, quiet=quiet)
    trainer.eval_every = 0          # the experiment evaluates once, at the end
    trainer.train(n_games=n_games, workers=workers)

    class _Frozen:
        name = "learned"
        def __init__(self, learner, g):
            self._l, self._g = learner, g
        def act(self, b):
            return self._l.best_action(b)
        def new_game(self):
            pass
        def describe(self):
            return {"name": "learned", "games_trained": self._g}

    if not quiet:
        print(f"  evaluating over {n_eval} fixed games...")
    res = evaluate(_Frozen(trainer.learner, trainer.game_index),
                   games=n_eval, seed=cfg["evaluation"]["seed"])
    meta = run.load_meta() or {}
    return {
        "experiment": name, "kind": "train",
        "description": spec.get("description", ""),
        "run": run_name,
        "games_trained": trainer.game_index,
        "config": cfg,
        "training": meta.get("all_time", {}),
        "evaluation": res,
    }


def load_results() -> list:
    if not RESULTS_DIR.is_dir():
        return []
    out = []
    for path in sorted(RESULTS_DIR.iterdir()):
        if path.suffix == ".json":
            try:
                with open(path, encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def comparison_table(results: list) -> str:
    rows = []
    for r in results:
        ev = r.get("evaluation") or {}
        if not ev.get("games"):
            continue
        rows.append((
            r.get("experiment", "?"),
            r.get("games_trained", 0),
            ev.get("mean_score", 0),
            ev.get("ci95_mean", [0, 0]),
            ev.get("median_score", 0),
            ev.get("highest_tile", 0),
            ev.get("tile_rates", {}).get("2048", {}).get("rate", 0) * 100,
            ev.get("games", 0),
        ))
    rows.sort(key=lambda t: t[2], reverse=True)
    hdr = (f"{'experiment':<26}{'trained':>10}{'eval mean':>12}"
           f"{'95% CI':>21}{'median':>10}{'tile':>8}{'2048%':>8}{'n':>6}")
    out = [hdr, "-" * len(hdr)]
    for name, trained, mean, ci, med, tile, r2048, n in rows:
        out.append(f"{name:<26}{trained:>10,}{mean:>12,.0f}"
                   f"{f'[{ci[0]:,.0f}, {ci[1]:,.0f}]':>21}"
                   f"{med:>10,.0f}{tile:>8,}{r2048:>7.1f}%{n:>6,}")
    return "\n".join(out) if rows else "no experiment results yet"
