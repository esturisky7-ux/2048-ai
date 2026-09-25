"""Job worker: ``python -m dashboard.runner <spec.json>``.

The job manager launches this module as a child process for everything except
training, which keeps using ``train.py`` so the browser and the command line
drive exactly the same tested code path.

A spec is a JSON file written by the server. The runner validates it again on
arrival -- the server has already validated the browser's input, but a job
worker that trusts its input is a job worker that will eventually be handed
something surprising -- then writes two files next to it:

``<job>.progress.json``   overwritten periodically; what the UI polls
``<job>.result.json``     written once, at the end; the job's real output

Everything here is a thin wrapper around the same functions the CLI uses
(``evaluation.evaluator``, ``experiments.runner``, ``engine.benchmark``), so
there is exactly one implementation of each measurement in the project.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.registry import make_agent, AGENT_NAMES        # noqa: E402
from evaluation.evaluator import evaluate, game_seed       # noqa: E402
from training.checkpoint import Run                        # noqa: E402
from training.runlock import hold_frozen                   # noqa: E402

# Progress is written at most this often. Evaluation calls back after every
# game, and rewriting a file thousands of times a second would cost more than
# the games do.
PROGRESS_INTERVAL = 0.35


class ProgressWriter:
    """Throttled, atomic-ish progress publishing."""

    def __init__(self, path: Path):
        self.path = path
        self._last = 0.0
        self._payload: dict = {}

    def set(self, force: bool = False, **fields) -> None:
        self._payload.update(fields)
        now = time.time()
        if not force and now - self._last < PROGRESS_INTERVAL:
            return
        self._last = now
        self._payload["updated_at"] = now
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._payload, f)
            os.replace(tmp, self.path)
        except OSError:
            pass          # progress is advisory; never fail a job over it


def _write_result(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _build_agent(spec: dict):
    """Construct an agent from a validated spec fragment."""
    name = spec.get("agent", "learned")
    if name not in AGENT_NAMES:
        raise ValueError(f"unknown agent {name!r}")
    kw = {}
    if name == "expectimax":
        kw["depth"] = int(spec.get("depth", 2))
        kw["prob_cutoff"] = float(spec.get("prob_cutoff", 1e-3))
        kw["adaptive"] = bool(spec.get("adaptive", False))
    elif name == "learned":
        kw["run"] = spec.get("run", "default")
        kw["depth"] = int(spec.get("depth", 1))
        if spec.get("checkpoint"):
            kw["checkpoint"] = spec["checkpoint"]
    return make_agent(name, seed=int(spec.get("agent_seed", 0)), **kw)


def _live_run(spec: dict) -> str | None:
    """The run whose current weights this agent spec plays, if any."""
    if spec.get("agent", "learned") != "learned":
        return None
    from agents.learned import live_run_for
    return live_run_for(spec.get("run", "default"), spec.get("checkpoint"))


def _label_for(spec: dict) -> str:
    name = spec.get("agent", "learned")
    depth = int(spec.get("depth", 1 if name == "learned" else 2))
    if name in ("learned", "expectimax") and depth > 1:
        return f"{name} d{depth}"
    return name


# ---------------------------------------------------------------------------
# Job types
# ---------------------------------------------------------------------------
def run_evaluation(spec: dict, progress: ProgressWriter) -> dict:
    """Fixed-seed evaluation of one agent."""
    games = int(spec["games"])
    seed = int(spec["seed"])
    agent = _build_agent(spec)
    label = _label_for(spec)
    progress.set(force=True, phase="evaluating", agent=label,
                 done=0, total=games)

    def on_progress(done, total):
        progress.set(done=done, total=total)

    try:
        # Current weights stay frozen for the whole evaluation, or it is
        # refused because the run is being trained.
        with hold_frozen([getattr(agent, "live_run", None)],
                         purpose="evaluation, control center"):
            res = evaluate(agent, games=games, seed=seed,
                           progress=on_progress)
    finally:
        if hasattr(agent, "close"):
            agent.close()

    res["label"] = label
    res["agent_spec"] = spec
    progress.set(force=True, phase="done", done=games, total=games)

    # Append to the run's evaluation log when this evaluated a run's current
    # weights, so the dashboard's evaluation series picks it up exactly as a
    # command-line evaluation would.
    if spec.get("agent") == "learned" and not spec.get("checkpoint") \
            and spec.get("save", True):
        run = Run(spec.get("run", "default"))
        run.create_dirs()
        with open(run.eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, separators=(",", ":"), default=str) + "\n")
    return res


def run_comparison(spec: dict, progress: ProgressWriter) -> dict:
    """Evaluate several agents over the identical set of seeded games."""
    games = int(spec["games"])
    seed = int(spec["seed"])
    agents = spec["agents"]
    results = {}
    total_units = games * len(agents)
    done_units = 0

    # Every run whose current weights take part stays frozen for the whole
    # comparison -- or the comparison is refused before any game is played,
    # not after the first few agents have already been measured.
    with hold_frozen([_live_run(sub) for sub in agents],
                     purpose="comparison, control center"):
        for index, sub in enumerate(agents):
            label = _label_for(sub)
            progress.set(force=True, phase="evaluating", agent=label,
                         agent_index=index, agent_count=len(agents),
                         done=done_units, total=total_units)
            agent = _build_agent(sub)
            base = done_units

            def on_progress(done, _total, _base=base):
                progress.set(done=_base + done, total=total_units)

            t0 = time.perf_counter()
            try:
                res = evaluate(agent, games=games, seed=seed,
                               progress=on_progress)
            finally:
                if hasattr(agent, "close"):
                    agent.close()
            elapsed = time.perf_counter() - t0
            res["label"] = label
            res["agent_spec"] = sub
            # Mean wall-clock time per decision, which is what makes a search
            # agent feel slow. mean_moves is decisions per game.
            moves = res.get("mean_moves", 0) * res.get("games", 0)
            res["ms_per_decision"] = (elapsed / moves * 1000.0) \
                if moves else 0.0
            results[label] = res
            done_units += games

    progress.set(force=True, phase="done", done=total_units,
                 total=total_units)
    return {"seed": seed, "games": games, "results": results,
            "timestamp": time.time()}


def run_benchmark(spec: dict, progress: ProgressWriter) -> dict:
    """Throughput of this machine. Not a statement about agent strength."""
    from engine import benchmark as bench
    from engine import board as B
    from random import Random

    scale = float(spec.get("scale", 1.0))
    steps = 5
    out: dict = {"timestamp": time.time(), "machine": machine_info()}

    progress.set(force=True, phase="engine: random rollouts", done=0,
                 total=steps)
    out["random_all_moves"] = bench.bench_rollouts(
        bench.random_rollout, max(50, int(600 * scale)))

    progress.set(force=True, phase="engine: fast rollouts", done=1,
                 total=steps)
    out["random_try_in_order"] = bench.bench_rollouts(
        bench.random_rollout_fast, max(50, int(1200 * scale)))

    progress.set(force=True, phase="engine: per-call costs", done=2,
                 total=steps)
    out["primitives"] = bench.micro(number=max(20000, int(120000 * scale)))

    # Agent decision rate: how many moves per second each agent can choose.
    progress.set(force=True, phase="agents: decision rate", done=3,
                 total=steps)
    out["agents"] = []
    rng = Random(4242)
    boards = []
    b = B.new_game(rng)
    for _ in range(400):
        legal = B.legal_actions(b)
        if not legal:
            b = B.new_game(rng)
            continue
        boards.append(b)
        nb, _, _ = B.move(b, legal[0])
        b = B.random_spawn(nb, rng)

    for name in ("random", "heuristic", "expectimax", "learned"):
        budget = {"random": 4000, "heuristic": 1500,
                  "expectimax": 40, "learned": 800}[name]
        n = max(10, int(budget * scale))
        try:
            agent = _build_agent({"agent": name,
                                  "run": spec.get("run", "default")})
        except Exception as e:
            out["agents"].append({"agent": name, "available": False,
                                  "reason": str(e)})
            continue
        try:
            sample = [boards[i % len(boards)] for i in range(n)]
            t0 = time.perf_counter()
            for bd in sample:
                agent.act(bd)
            dt = time.perf_counter() - t0
        finally:
            if hasattr(agent, "close"):
                agent.close()
        out["agents"].append({
            "agent": name, "available": True, "decisions": n,
            "seconds": dt,
            "decisions_per_second": n / dt if dt else 0.0,
            "ms_per_decision": dt / n * 1000.0 if n else 0.0,
        })

    # Training throughput: real TD updates on a throwaway network, so this
    # measures the actual learning loop rather than a proxy for it.
    progress.set(force=True, phase="training: moves/sec", done=4, total=steps)
    try:
        from training.ntuple import NTupleNetwork
        from training.learner import TDLearner
        from training.reward import RewardFunction
        net = NTupleNetwork("8x4")          # 2 MB, anonymous memory
        try:
            learner = TDLearner(net, RewardFunction(), alpha=0.1)
            n_games = max(5, int(120 * scale))
            rng = Random(7)
            t0 = time.perf_counter()
            moves = 0
            for i in range(n_games):
                _, m, _ = learner.play_game(Random(game_seed(99, i)))
                moves += m
            dt = time.perf_counter() - t0
            out["training"] = {
                "tuple_set": "8x4", "games": n_games, "moves": moves,
                "seconds": dt,
                "games_per_second": n_games / dt if dt else 0.0,
                "moves_per_second": moves / dt if dt else 0.0,
            }
        finally:
            net.close()
    except Exception as e:
        out["training"] = {"error": str(e)}

    progress.set(force=True, phase="done", done=steps, total=steps)
    return out


def run_experiment_job(spec: dict, progress: ProgressWriter) -> dict:
    """Run a shipped or custom experiment config end to end."""
    from experiments.runner import run_experiment
    name = spec["name"]
    progress.set(force=True, phase=f"running experiment {name}",
                 done=0, total=1)
    res = run_experiment(
        name,
        games=spec.get("games"),
        eval_games=spec.get("eval_games"),
        workers=int(spec.get("workers", 1)),
        fresh=bool(spec.get("fresh", False)),
        quiet=True,
    )
    progress.set(force=True, phase="done", done=1, total=1)
    return res


def machine_info() -> dict:
    """Describe the machine, degrading gracefully when a metric is missing."""
    info = {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    # A readable CPU model where the OS offers one without extra tooling.
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
    except OSError:
        pass
    if not info.get("cpu_model"):
        info["cpu_model"] = info["processor"] or info["machine"]
    try:
        info["total_ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * \
            os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        info["total_ram_bytes"] = None
    return info


HANDLERS = {
    "evaluation": run_evaluation,
    "comparison": run_comparison,
    "benchmark": run_benchmark,
    "experiment": run_experiment_job,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m dashboard.runner <spec.json>", file=sys.stderr)
        return 2
    spec_path = Path(argv[1])
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    job_type = spec.get("type")
    handler = HANDLERS.get(job_type)
    if handler is None:
        print(f"unknown job type {job_type!r}", file=sys.stderr)
        return 2

    progress = ProgressWriter(Path(spec["progress_path"]))
    result_path = Path(spec["result_path"])
    try:
        result = handler(spec, progress)
    except KeyboardInterrupt:
        progress.set(force=True, phase="cancelled")
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        progress.set(force=True, phase="failed", error=str(e))
        import traceback
        traceback.print_exc()
        # Last, so it is the line the job list reports as the reason.
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    _write_result(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
