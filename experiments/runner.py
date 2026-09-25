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
from training.checkpoint import DATA_ROOT, Run, atomic_write_json

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(DATA_ROOT) / "experiments"


def result_path(name: str) -> Path:
    return RESULTS_DIR / f"{name}.json"


def experiment_run_name(name: str, spec: dict | None = None) -> str | None:
    """The run a training experiment trains, or ``None`` for an agent one."""
    spec = load_experiment(name) if spec is None else spec
    if spec.get("kind", "train") != "train":
        return None
    return spec.get("run", f"exp-{name}")


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
    from training.runlock import hold_frozen

    a = dict(spec.get("agent") or {})
    agent_name = a.pop("name", "expectimax")
    n = eval_games or spec.get("eval_games", 200)
    seed = spec.get("eval_seed", 987654)
    agent = make_agent(agent_name, seed=spec.get("agent_seed", 0), **a)
    try:
        if not quiet:
            print(f"  evaluating {agent_name} over {n} games (seed {seed})...")
        with hold_frozen([getattr(agent, "live_run", None)],
                         purpose="evaluation, experiment"):
            res = evaluate(agent, games=n, seed=seed)
    finally:
        if hasattr(agent, "close"):
            agent.close()
    return res


def _run_train_experiment(name, spec, games, eval_games, workers,
                          fresh, quiet) -> dict:
    import shutil
    from training.trainer import PolicyAgent, Trainer
    from evaluation.evaluator import evaluate

    cfg = load_config(overrides={k: v for k, v in spec.items()
                                 if k not in ("kind", "description",
                                              "games", "eval_games",
                                              "eval_seed", "agent",
                                              "agent_seed")})
    run_name = experiment_run_name(name, spec)
    cfg["run"] = run_name
    n_games = games or spec.get("games", 5000)
    n_eval = eval_games or spec.get("eval_games", 200)

    run = Run(run_name)
    # Own the run before looking at it: a --fresh reset must never delete a
    # run another process is training, and nothing may start training it
    # between the reset and this experiment's own trainer taking over.
    lock = run.lock(purpose=f"experiment {name}").acquire()
    try:
        if fresh and run.exists():
            shutil.rmtree(run.dir, ignore_errors=True)
            shutil.rmtree(run.data_dir, ignore_errors=True)

        resume = run.exists()
        trainer = Trainer(run_name, cfg if not resume else None,
                          resume=resume, quiet=quiet, lock=lock)
        try:
            trainer.eval_every = 0      # the experiment evaluates once, at the end
            trainer.train(n_games=n_games, workers=workers)
            if trainer.stop_requested:
                # Stopped part-way: a result for fewer games than the
                # experiment specifies would be misleading, so there is none.
                raise KeyboardInterrupt(
                    f"experiment {name} was stopped at game "
                    f"{trainer.game_index:,}; no result was written")
            if not quiet:
                print(f"  evaluating over {n_eval} fixed games...")
            res = evaluate(PolicyAgent(trainer.learner.policy(),
                                       trainer.game_index),
                           games=n_eval, seed=cfg["evaluation"]["seed"])
            games_trained = trainer.game_index
        finally:
            trainer.close()
    finally:
        lock.release()
    meta = run.load_meta() or {}
    return {
        "experiment": name, "kind": "train",
        "description": spec.get("description", ""),
        "run": run_name,
        "games_trained": games_trained,
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
