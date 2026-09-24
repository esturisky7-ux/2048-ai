"""The control center's HTTP API: routing, validation and payload building.

Everything the browser can ask for lives here. :mod:`dashboard.server` owns the
socket and the request/response plumbing; this module owns *what the answers
are*, which keeps the transport thin and makes the whole API testable without
opening a port.

Two rules shape the code below.

**Nothing from the browser reaches a shell or a filesystem path.** Job commands
are built as explicit argument lists from validated values, and checkpoints are
addressed by identifier — :mod:`dashboard.store` is the only place that turns
an identifier into a path.

**Nothing slow happens here.** Training and evaluation are handed to the job
manager, which supervises them as subprocesses. A handler in this module should
always return in milliseconds.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.registry import AGENT_NAMES                      # noqa: E402
from training.checkpoint import Run, list_runs, read_json    # noqa: E402
from training.config import load_config, list_experiments    # noqa: E402
from training.ntuple import TUPLE_SETS                       # noqa: E402
from training.stats import AllTimeStats, History             # noqa: E402
from version import __version__                              # noqa: E402

from . import store
from .events import LOG
from .games import AIGameSession, HumanGameSession, SESSIONS
from .jobs import MANAGER, JobConflict
from .store import InvalidName

# A run's training process may die without clearing its status file; a status
# older than this is treated as stale rather than reported as live training.
STALE_AFTER = 15.0        # the trainer heartbeats every 3 s

DIRECTIONS = {"up": 0, "right": 1, "down": 2, "left": 3}


class ApiError(Exception):
    """A request that should become a 4xx with a readable message."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _int(data: dict, key: str, default: int | None = None,
         lo: int = 0, hi: int = 10 ** 9, required: bool = False) -> int:
    if key not in data or data[key] in (None, ""):
        if required:
            raise ApiError(f"{key} is required")
        return default
    try:
        v = int(data[key])
    except (TypeError, ValueError):
        raise ApiError(f"{key} must be a whole number")
    if not lo <= v <= hi:
        raise ApiError(f"{key} must be between {lo} and {hi}")
    return v


def _float(data: dict, key: str, default: float | None = None,
           lo: float = 0.0, hi: float = 1e9) -> float:
    if key not in data or data[key] in (None, ""):
        return default
    try:
        v = float(data[key])
    except (TypeError, ValueError):
        raise ApiError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise ApiError(f"{key} must be between {lo} and {hi}")
    return v


def _agent(name, field: str = "agent") -> str:
    if name not in AGENT_NAMES:
        raise ApiError(f"{field} must be one of {', '.join(AGENT_NAMES)}")
    return name


def _run_name(data: dict, key: str = "run", default: str = "default") -> str:
    """Resolve a run name, telling "absent" apart from "explicitly empty".

    A missing key means "use the default run", which is what every read
    endpoint wants. A key that is present but blank is a mistake — silently
    training the default run because someone cleared a text field would be a
    genuinely bad surprise — so it is rejected.
    """
    if key not in data or data[key] is None:
        name = default
    else:
        name = data[key]
    try:
        return store.validate_run_name(name)
    except InvalidName as e:
        raise ApiError(str(e))


def pid_alive(pid: int) -> bool:
    """Is that process still around?

    ``os.kill(pid, 0)`` is the usual POSIX probe, but on Windows ``os.kill``
    calls ``TerminateProcess`` for any signal other than the two console
    events — it would *kill* the trainer instead of asking after it. Windows
    therefore gets a read-only handle-open probe via ``ctypes`` instead, and if
    even that is unavailable the run is reported as alive and the staleness
    timer alone decides.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                     False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by somebody else
    except (OSError, TypeError):
        return False
    return True


# ---------------------------------------------------------------------------
# History and status payloads
# ---------------------------------------------------------------------------
def downsample(rows: list, target: int = 500) -> list:
    """Thin a history to about ``target`` points, always keeping the last one."""
    n = len(rows)
    if n <= target:
        return rows
    step = n / target
    out = [rows[int(i * step)] for i in range(target)]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def history_series(run: Run, limit: int = 500) -> dict:
    """Flatten history.jsonl into parallel arrays the charts can plot."""
    rows = downsample(History(run.history_path).read(), limit)
    out = {"games": [], "mean_score": [], "median_score": [], "max_tile": [],
           "rate_512": [], "rate_1024": [], "rate_2048": [], "rate_4096": [],
           "rate_8192": [], "games_per_second": [], "moves_per_second": [],
           "wall_time": [], "alpha": []}
    for r in rows:
        roll = r.get("rolling") or {}
        if not roll.get("games"):
            continue
        rates = roll.get("tile_rates", {})
        out["games"].append(r.get("games", 0))
        out["mean_score"].append(round(roll.get("mean_score", 0), 1))
        out["median_score"].append(round(roll.get("median_score", 0), 1))
        out["max_tile"].append(roll.get("max_tile", 0))
        for m in (512, 1024, 2048, 4096, 8192):
            out[f"rate_{m}"].append(round(rates.get(str(m), 0) * 100, 2))
        out["games_per_second"].append(round(r.get("games_per_second", 0), 2))
        out["moves_per_second"].append(round(r.get("moves_per_second", 0), 1))
        out["wall_time"].append(r.get("wall_time", 0))
        out["alpha"].append(r.get("alpha", 0))
    return out


def eval_series(run: Run, limit: int = 200) -> dict:
    rows = store.list_evaluations(run.name, limit)
    return {
        "games_trained": [(r.get("games_trained")
                           or (r.get("agent") or {}).get("games_trained", 0))
                          for r in rows],
        "mean_score": [round(r.get("mean_score", 0), 1) for r in rows],
        "ci_low": [round(r.get("ci95_mean", [0, 0])[0], 1) for r in rows],
        "ci_high": [round(r.get("ci95_mean", [0, 0])[1], 1) for r in rows],
        "highest_tile": [r.get("highest_tile", 0) for r in rows],
        "games": [r.get("games", 0) for r in rows],
        "rate_2048": [round(_rate(r, "2048") * 100, 2) for r in rows],
        "rate_4096": [round(_rate(r, "4096") * 100, 2) for r in rows],
    }


def _rate(record: dict, tile: str) -> float:
    entry = (record.get("tile_rates") or {}).get(tile)
    if isinstance(entry, dict):
        return entry.get("rate", 0.0)
    return entry or 0.0


def _training_view(run_name: str) -> dict:
    """What the trainer for this run is doing right now, if anything."""
    run = Run(run_name)
    status = read_json(run.status_path, {}) or {}
    job = MANAGER.current_training()
    job_for_run = job if (job and job.params.get("run") == run_name) else None

    running = bool(status.get("running"))
    updated = status.get("updated_at", 0)
    if running:
        fresh = (time.time() - updated) < STALE_AFTER
        if not (fresh and pid_alive(status.get("pid", -1))):
            running = False

    # A job we launched is authoritative about its own state: it may be
    # starting up (no status file yet) or stopping (status still says running).
    state = "STOPPED"
    if job_for_run is not None:
        state = {"RUNNING": "TRAINING", "STOPPING": "STOPPING",
                 "QUEUED": "STARTING"}.get(job_for_run.state, "STOPPED")
    elif running:
        state = "TRAINING"          # started from the command line

    return {
        "state": state,
        "running": state in ("TRAINING", "STARTING", "STOPPING"),
        "external": running and job_for_run is None,
        "status": status,
        "job": job_for_run.to_dict() if job_for_run else None,
        "stale": bool(status.get("running")) and not running,
    }


def status_payload(run_name: str) -> dict:
    """Everything the Overview page needs, in one request."""
    run = Run(run_name)
    meta = run.load_meta() or {}
    cfg = run.load_config() or {}
    tv = _training_view(run_name)
    status = tv["status"]
    running = tv["running"]

    at_raw = meta.get("all_time") or {}
    at = AllTimeStats(at_raw).summary() if at_raw else {}

    games = meta.get("games", 0)
    last_eval = meta.get("last_eval")
    if running:
        games = max(games, status.get("games", 0))
        last_eval = status.get("last_eval") or last_eval
        live_at = status.get("all_time")
        if live_at and live_at.get("games", 0) >= at.get("games", 0):
            at = live_at

    rolling = (status.get("rolling") if running else None) \
        or meta.get("rolling", {})

    # meta.json only records an evaluation when the *trainer* ran one (via
    # --eval-every). Evaluations started from the control center are appended
    # to the run's evaluation log instead, so fall back to that — otherwise
    # the Overview would report "not evaluated yet" right after you evaluated.
    if not last_eval:
        recent = store.list_evaluations(run_name, limit=1)
        if recent:
            ev = recent[-1]
            last_eval = {
                "games_trained": (ev.get("games_trained")
                                  or (ev.get("agent") or {}).get(
                                      "games_trained")),
                "eval_games": ev.get("games"),
                "mean_score": ev.get("mean_score"),
                "median_score": ev.get("median_score"),
                "ci95_mean": ev.get("ci95_mean"),
                "highest_tile": ev.get("highest_tile"),
                "tile_rates": {k: (v["rate"] if isinstance(v, dict) else v)
                               for k, v in (ev.get("tile_rates") or {}).items()},
                "timestamp": ev.get("timestamp"),
            }

    # Schedules: how far the run is from its next checkpoint and evaluation.
    training_cfg = cfg.get("training", {})
    ck_every = training_cfg.get("checkpoint_every") or 0
    ev_every = training_cfg.get("eval_every") or 0
    schedule = {
        "checkpoint_every": ck_every,
        "eval_every": ev_every,
        "games_to_checkpoint": (ck_every - games % ck_every) if ck_every else None,
        "games_to_eval": (ev_every - games % ev_every) if ev_every else None,
    }

    gps = status.get("games_per_second", 0.0) if running else 0.0
    # The trainer's smoothed rate only exists after its first report interval.
    # Until then, derive one from the session counters it is already
    # publishing, so the UI shows a real number instead of a dash for the
    # first few seconds. Still measured data, just a coarser average.
    mps = status.get("moves_per_second", 0.0) if running else 0.0
    if running:
        session_seconds = status.get("session_seconds", 0.0)
        if session_seconds > 0.5:
            if not gps and status.get("session_games"):
                gps = status["session_games"] / session_seconds
            if not mps and status.get("session_moves"):
                mps = status["session_moves"] / session_seconds
    job = tv["job"]
    target = (job or {}).get("params", {}).get("target_games") or 0
    eta = None
    if running and gps > 0 and target:
        remaining = max(0, target - games)
        eta = remaining / gps
    if schedule["games_to_checkpoint"] and gps > 0:
        schedule["seconds_to_checkpoint"] = schedule["games_to_checkpoint"] / gps
    if schedule["games_to_eval"] and gps > 0:
        schedule["seconds_to_eval"] = schedule["games_to_eval"] / gps

    # The headline state also reflects other work, so the UI can say what the
    # system as a whole is doing.
    ai_state = tv["state"]
    if ai_state == "STOPPED":
        if MANAGER.active("evaluation") or MANAGER.active("comparison"):
            ai_state = "EVALUATING"
        elif MANAGER.active("benchmark"):
            ai_state = "BENCHMARKING"
        elif MANAGER.active("experiment"):
            ai_state = "EXPERIMENTING"
    failed = [j for j in MANAGER.recent(6) if j.state == "FAILED"]

    return {
        "run": run_name,
        "exists": run.exists(),
        "state": ai_state,
        "training": tv,
        "running": running,
        "pid": status.get("pid") if running else None,
        "games": games,
        "session_games": status.get("session_games", 0) if running else 0,
        "session_seconds": status.get("session_seconds", 0.0) if running else 0.0,
        "workers": (job or {}).get("params", {}).get("workers"),
        "target_games": target or None,
        "eta_seconds": eta,
        "tuple_set": meta.get("tuple_set"),
        "alpha": status.get("alpha") if running else meta.get("alpha"),
        "games_per_second": gps,
        "moves_per_second": mps,
        "total_train_seconds": at.get("train_seconds", 0.0),
        "mean_score": at.get("mean_score", 0.0),
        "checkpoint_saved_at": meta.get("saved_at"),
        "checkpoint_saved_iso": meta.get("saved_at_iso"),
        "schedule": schedule,
        "all_time": at,
        "rolling": rolling,
        "last_eval": last_eval,
        "disk_bytes": run.size_on_disk(),
        "runs": list_runs(),
        "has_any_run": bool(list_runs()),
        "jobs": [j.to_dict(with_progress=True) for j in MANAGER.active()],
        "recent_failures": [j.to_dict(with_progress=False) for j in failed],
        "version": __version__,
        "server_time": time.time(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _training_argv(p: dict) -> list[str]:
    """Build the exact ``train.py`` command line for a validated request."""
    argv = [sys.executable, str(ROOT / "train.py"), "--run", p["run"]]
    if p["resume"]:
        argv.append("--resume")
    elif p.get("tuple_set"):
        argv += ["--tuple-set", p["tuple_set"]]
    argv += ["--games", str(p["games"])]
    argv += ["--workers", str(p["workers"])]
    for flag, key in (("--alpha", "alpha"), ("--alpha-decay", "alpha_decay"),
                      ("--epsilon", "epsilon"), ("--gamma", "gamma")):
        if p.get(key) is not None:
            argv += [flag, repr(float(p[key]))]
    for flag, key in (("--seed", "seed"),
                      ("--report-every", "report_every"),
                      ("--checkpoint-every", "checkpoint_every"),
                      ("--snapshot-every", "snapshot_every"),
                      ("--eval-every", "eval_every"),
                      ("--eval-games", "eval_games")):
        if p.get(key) is not None:
            argv += [flag, str(int(p[key]))]
    return argv


def start_training(data: dict) -> dict:
    run_name = _run_name(data)
    resume = bool(data.get("resume"))
    run = Run(run_name)
    exists = run.exists()

    # Checked before anything else: a run that is already training has not
    # necessarily written its first checkpoint yet, and "no checkpoint to
    # resume" would be a confusing way to say "that is already running".
    busy = MANAGER.active_with_key(f"run:{run_name}")
    if busy is not None:
        raise ApiError(
            f"{busy.type} job {busy.id} is already working on "
            f"'run:{run_name}'. Stop it first.", status=409)
    # The job manager only knows its own jobs. A trainer started from a
    # terminal, or left behind by a server that was killed, is just as live,
    # and a second one would interleave updates into the same weight file.
    external = _training_view(run_name)
    if external["external"]:
        raise ApiError(
            f"run '{run_name}' is already being trained by another process "
            f"(pid {external['status'].get('pid')}). Stop that first — the "
            f"Stop button works on it too.", status=409)

    if resume and not exists:
        raise ApiError(f"run '{run_name}' has no checkpoint to resume")
    if not resume and exists:
        raise ApiError(
            f"run '{run_name}' already exists — resume it, or choose another "
            f"name")

    cores = os.cpu_count() or 1
    params = {
        "run": run_name,
        "resume": resume,
        "games": _int(data, "games", 0, lo=0, hi=100_000_000),
        "workers": _int(data, "workers", 1, lo=1, hi=max(64, cores)),
        "alpha": _float(data, "alpha", None, lo=1e-6, hi=10.0),
        "alpha_decay": _float(data, "alpha_decay", None, lo=0.0001, hi=1.0),
        "epsilon": _float(data, "epsilon", None, lo=0.0, hi=1.0),
        "gamma": _float(data, "gamma", None, lo=0.0, hi=1.0),
        "seed": _int(data, "seed", None, lo=0, hi=2 ** 63 - 1),
        "report_every": _int(data, "report_every", None, lo=0, hi=10 ** 7),
        "checkpoint_every": _int(data, "checkpoint_every", None, lo=0,
                                 hi=10 ** 7),
        "snapshot_every": _int(data, "snapshot_every", None, lo=0, hi=10 ** 7),
        "eval_every": _int(data, "eval_every", None, lo=0, hi=10 ** 7),
        "eval_games": _int(data, "eval_games", None, lo=1, hi=100000),
    }
    if not resume:
        ts = data.get("tuple_set") or "4x6"
        if ts not in TUPLE_SETS:
            raise ApiError(f"tuple_set must be one of "
                           f"{', '.join(sorted(TUPLE_SETS))}")
        params["tuple_set"] = ts

    if params["workers"] > cores:
        LOG.add("warn", f"training with {params['workers']} workers on "
                        f"{cores} CPU cores may be slower", run=run_name)

    meta = run.load_meta() or {}
    already = meta.get("games", 0) if resume else 0
    params["target_games"] = (already + params["games"]) if params["games"] \
        else 0

    argv = _training_argv(params)
    label = (f"{'resume' if resume else 'new run'} '{run_name}' — "
             f"{params['games'] or 'continuous'} games, "
             f"{params['workers']} worker(s)")
    try:
        job = MANAGER.submit("training", label, argv, params=params,
                             exclusive_key=f"run:{run_name}")
    except JobConflict as e:
        raise ApiError(str(e), status=409)
    LOG.add("info", f"training {'resumed' if resume else 'started'}: "
                    f"{run_name}", job_id=job.id,
            games=params["games"], workers=params["workers"])
    return {"ok": True, "job": job.to_dict()}


def stop_training(data: dict) -> dict:
    job_id = data.get("job_id")
    if job_id:
        job = MANAGER.get(str(job_id))
        if job is None:
            raise ApiError("no such job", status=404)
    else:
        job = MANAGER.current_training()
        if job is None:
            if "run" in data:
                return _stop_external_training(_run_name(data))
            raise ApiError("no training job is running", status=409)
    if not MANAGER.stop(job.id):
        raise ApiError("that job is not running", status=409)
    return {"ok": True, "job": job.to_dict(), "message":
            "stopping after the current game; the checkpoint will be saved"}


def _stop_external_training(run_name: str) -> dict:
    """Stop a trainer this server did not launch, the way Ctrl-C would.

    That is a ``train.py`` started from a terminal, or one left running by a
    control center that was killed. Its PID comes from the run's own status
    heartbeat, and is only signalled while that heartbeat is fresh and the
    process is alive -- the same test that reports it as training at all.
    """
    tv = _training_view(run_name)
    if not tv["external"]:
        raise ApiError("no training job is running", status=409)
    pid = tv["status"].get("pid")
    # An experiment job trains its run itself; stop it as the job it is, so
    # it is recorded as cancelled rather than failed.
    for job in MANAGER.active():
        if job.pid == pid and MANAGER.stop(job.id):
            return {"ok": True, "job": job.to_dict(), "message":
                    "stopping after the current game; the checkpoint will "
                    "be saved"}
    if os.name == "nt":
        # There is no way to deliver Ctrl-C to another console's process.
        raise ApiError(
            f"run '{run_name}' is being trained by process {pid}, which this "
            f"control center did not start. Press Ctrl-C in its window.",
            status=409)
    import signal
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        raise ApiError("that training process has already exited", status=409)
    except PermissionError:
        raise ApiError(f"not allowed to stop process {pid}", status=403)
    LOG.add("info", f"asked training process {pid} to stop", run=run_name)
    return {"ok": True, "job": None, "pid": pid, "message":
            "stopping after the current game; the checkpoint will be saved"}


# ---------------------------------------------------------------------------
# Jobs backed by dashboard.runner
# ---------------------------------------------------------------------------
def _submit_runner(job_type: str, label: str, spec: dict,
                   exclusive_key: str | None = None):
    from .jobs import JOB_DIR
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{job_type}-{int(time.time() * 1000)}"
    spec_path = JOB_DIR / f"{stamp}.spec.json"
    progress_path = JOB_DIR / f"{stamp}.progress.json"
    result_path = JOB_DIR / f"{stamp}.result.json"
    spec = dict(spec)
    spec.update({"type": job_type,
                 "progress_path": str(progress_path),
                 "result_path": str(result_path)})
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)

    argv = [sys.executable, "-m", "dashboard.runner", str(spec_path)]
    try:
        job = MANAGER.submit(job_type, label, argv, params=spec,
                             exclusive_key=exclusive_key)
    except JobConflict as e:
        raise ApiError(str(e), status=409)
    # Point the job at the runner's files so progress and result are found.
    job._progress_path = progress_path
    job._result_path = result_path
    return job


def _agent_spec(data: dict, prefix: str = "") -> dict:
    """Validate one agent description from the browser."""
    name = _agent(data.get(prefix + "agent", "learned"))
    spec = {"agent": name}
    if name == "learned":
        cid = data.get(prefix + "checkpoint")
        if cid:
            try:
                store.checkpoint_path(cid)          # existence + containment
            except InvalidName as e:
                raise ApiError(str(e))
            except FileNotFoundError as e:
                raise ApiError(str(e), status=404)
            run_name, _snap = store.parse_checkpoint_id(cid)
            spec["run"] = run_name
            spec["checkpoint"] = str(store.checkpoint_path(cid))
            spec["checkpoint_id"] = cid
        else:
            spec["run"] = _run_name(data, prefix + "run")
            if not Run(spec["run"]).exists():
                raise ApiError(
                    f"run '{spec['run']}' has not been trained yet",
                    status=404)
        spec["depth"] = _int(data, prefix + "depth", 1, lo=1, hi=3)
    elif name == "expectimax":
        spec["depth"] = _int(data, prefix + "depth", 2, lo=1, hi=4)
        spec["prob_cutoff"] = _float(data, prefix + "prob_cutoff", 1e-3,
                                     lo=1e-6, hi=1.0)
        spec["adaptive"] = bool(data.get(prefix + "adaptive"))
    spec["agent_seed"] = _int(data, prefix + "agent_seed", 0, lo=0,
                              hi=2 ** 31 - 1)
    return spec


def start_evaluation(data: dict) -> dict:
    spec = _agent_spec(data)
    spec["games"] = _int(data, "games", 200, lo=1, hi=200000)
    spec["seed"] = _int(data, "seed", 987654, lo=0, hi=2 ** 63 - 1)
    spec["save"] = bool(data.get("save", True))
    label = f"evaluate {spec['agent']} × {spec['games']} games"
    job = _submit_runner("evaluation", label, spec)
    LOG.add("info", label, job_id=job.id)
    return {"ok": True, "job": job.to_dict()}


def start_comparison(data: dict) -> dict:
    raw = data.get("agents")
    if not isinstance(raw, list) or not raw:
        raise ApiError("agents must be a non-empty list")
    if len(raw) > 6:
        raise ApiError("compare at most 6 agents at a time")
    agents = []
    for item in raw:
        if isinstance(item, str):
            item = {"agent": item}
        if not isinstance(item, dict):
            raise ApiError("each agent must be a name or an object")
        agents.append(_agent_spec(item))
    spec = {
        "agents": agents,
        "games": _int(data, "games", 100, lo=1, hi=50000),
        "seed": _int(data, "seed", 987654, lo=0, hi=2 ** 63 - 1),
    }
    label = (f"compare {', '.join(a['agent'] for a in agents)} × "
             f"{spec['games']} games")
    job = _submit_runner("comparison", label, spec)
    LOG.add("info", label, job_id=job.id)
    return {"ok": True, "job": job.to_dict()}


def start_benchmark(data: dict) -> dict:
    spec = {"scale": _float(data, "scale", 1.0, lo=0.05, hi=10.0),
            "run": _run_name(data)}
    job = _submit_runner("benchmark", "benchmark this machine", spec,
                         exclusive_key="benchmark")
    LOG.add("info", "benchmark started", job_id=job.id)
    return {"ok": True, "job": job.to_dict()}


def start_experiment(data: dict) -> dict:
    name = data.get("name")
    available = list_experiments()
    if name not in available:
        raise ApiError(f"unknown experiment {name!r}", status=404)
    spec = {
        "name": name,
        "games": _int(data, "games", None, lo=1, hi=10_000_000),
        "eval_games": _int(data, "eval_games", None, lo=1, hi=100000),
        "workers": _int(data, "workers", 1, lo=1, hi=64),
        "fresh": bool(data.get("fresh")),
    }
    job = _submit_runner("experiment", f"experiment {name}", spec,
                         exclusive_key=f"experiment:{name}")
    LOG.add("info", f"experiment {name} started", job_id=job.id)
    return {"ok": True, "job": job.to_dict()}


def _finish_job_side_effects(job) -> None:
    """Persist results the moment a job that produces them completes."""
    if job.state != "COMPLETED" or not job.result:
        return
    if job.type == "comparison" and not job.params.get("_saved"):
        store.save_comparison(job.result)
        job.params["_saved"] = True
        LOG.add("info", "comparison finished", job_id=job.id)
    elif job.type == "benchmark" and not job.params.get("_saved"):
        store.save_benchmark(job.result)
        job.params["_saved"] = True
        LOG.add("info", "benchmark finished", job_id=job.id)
    elif job.type == "evaluation" and not job.params.get("_logged"):
        job.params["_logged"] = True
        LOG.add("info",
                f"evaluation finished: mean "
                f"{job.result.get('mean_score', 0):,.0f} over "
                f"{job.result.get('games', 0)} games", job_id=job.id)


def job_payload(job) -> dict:
    _finish_job_side_effects(job)
    return job.to_dict()


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
def _build_agent_for_session(spec: dict):
    from agents.registry import make_agent
    name = spec["agent"]
    kw = {}
    if name == "expectimax":
        kw.update(depth=spec.get("depth", 2),
                  prob_cutoff=spec.get("prob_cutoff", 1e-3),
                  adaptive=spec.get("adaptive", False))
    elif name == "learned":
        kw.update(run=spec.get("run", "default"), depth=spec.get("depth", 1))
        if spec.get("checkpoint"):
            kw["checkpoint"] = spec["checkpoint"]
    return make_agent(name, seed=None, **kw)


def _session_label(spec: dict) -> str:
    name = spec["agent"]
    depth = spec.get("depth", 1)
    if name == "learned":
        return "learned" if depth == 1 else f"learned+search d{depth}"
    if name == "expectimax":
        return f"expectimax d{depth}"
    return name


def start_ai_game(data: dict) -> dict:
    spec = _agent_spec(data)
    seed = _int(data, "seed", None, lo=0, hi=2 ** 31 - 1)
    try:
        agent = _build_agent_for_session(spec)
    except FileNotFoundError as e:
        raise ApiError(str(e), status=404)
    session = AIGameSession(agent, _session_label(spec), seed=seed)
    try:
        SESSIONS.add(session)
    except RuntimeError as e:
        raise ApiError(str(e), status=409)
    session.start()
    return {"ok": True, "session": session.snapshot(0, limit=1)}


def start_human_game(data: dict) -> dict:
    seed = _int(data, "seed", None, lo=0, hi=2 ** 31 - 1)
    session = HumanGameSession(seed=seed)
    try:
        SESSIONS.add(session)
    except RuntimeError as e:
        raise ApiError(str(e), status=409)
    return {"ok": True, "session": session.state()}


def _session(session_id: str):
    s = SESSIONS.get(session_id)
    if s is None:
        raise ApiError("no such game session", status=404)
    return s


def game_state(session_id: str, since: int = 0) -> dict:
    s = _session(session_id)
    if isinstance(s, AIGameSession):
        return s.snapshot(max(0, since))
    return s.state()


def game_move(session_id: str, data: dict) -> dict:
    s = _session(session_id)
    if not isinstance(s, HumanGameSession):
        raise ApiError("only human games accept moves")
    direction = data.get("direction")
    if direction not in DIRECTIONS:
        raise ApiError("direction must be up, down, left or right")
    return s.move(DIRECTIONS[direction])


def game_control(session_id: str, data: dict) -> dict:
    s = _session(session_id)
    action = data.get("action")
    if action not in ("pause", "resume", "step", "stop"):
        raise ApiError("action must be pause, resume, step or stop")
    if isinstance(s, AIGameSession):
        getattr(s, action)()
        if action == "stop":
            SESSIONS.drop(session_id)
            return {"ok": True, "stopped": True}
        return {"ok": True, "session": s.snapshot(0, limit=1)}
    if action == "stop":
        SESSIONS.drop(session_id)
        return {"ok": True, "stopped": True}
    raise ApiError("human games support only 'stop'")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def handle_get(path: str, query: dict) -> dict:
    """Dispatch a GET. Returns the JSON payload, or raises ApiError."""
    one = lambda k, d=None: (query.get(k, [d]) or [d])[0]   # noqa: E731

    if path == "/api/status":
        return status_payload(_run_name({"run": one("run")}))
    if path == "/api/history":
        run = Run(_run_name({"run": one("run")}))
        limit = max(10, min(int(one("limit", "500") or 500), 5000))
        return {"history": history_series(run, limit),
                "evaluations": eval_series(run)}
    if path == "/api/runs":
        return {"runs": list_runs(), "default_config": load_config(),
                "tuple_sets": sorted(TUPLE_SETS)}
    if path == "/api/agents":
        return {"agents": list(AGENT_NAMES)}
    if path == "/api/training":
        return _training_view(_run_name({"run": one("run")}))
    if path == "/api/jobs":
        limit = max(1, min(int(one("limit", "25") or 25), 100))
        jt = one("type")
        return {"jobs": [job_payload(j) for j in MANAGER.recent(limit, jt)],
                "active": [job_payload(j) for j in MANAGER.active()]}
    if path.startswith("/api/jobs/"):
        rest = path[len("/api/jobs/"):]
        job_id, _, tail = rest.partition("/")
        job = MANAGER.get(job_id)
        if job is None:
            raise ApiError("no such job", status=404)
        if tail == "log":
            lines = max(1, min(int(one("lines", "200") or 200), 2000))
            return {"id": job.id, "lines": job.tail(lines)}
        if tail:
            raise ApiError("no such job endpoint", status=404)
        return job_payload(job)
    if path == "/api/evaluations":
        run_name = _run_name({"run": one("run")})
        limit = max(1, min(int(one("limit", "200") or 200), 1000))
        return {"run": run_name,
                "evaluations": store.list_evaluations(run_name, limit)}
    if path == "/api/comparisons":
        return {"comparisons": store.list_comparisons()}
    if path == "/api/benchmarks":
        return {"benchmarks": store.list_benchmarks()}
    if path == "/api/checkpoints":
        return {"checkpoints": store.list_checkpoints()}
    if path == "/api/experiments":
        from experiments.runner import list_experiments as detailed
        from experiments.runner import load_results
        return {"experiments": detailed(), "results": load_results()}
    if path == "/api/system":
        from .sysinfo import system_info
        return system_info(MANAGER)
    if path == "/api/logs":
        limit = max(1, min(int(one("limit", "100") or 100), 1000))
        return {"events": LOG.recent(limit, one("level"),
                                     int(one("since", "0") or 0))}
    if path == "/api/settings":
        return {"settings": store.load_settings(),
                "defaults": store.default_settings()}
    if path == "/api/sessions":
        return {"sessions": SESSIONS.describe()}
    if path.startswith("/api/game/"):
        rest = path[len("/api/game/"):]
        session_id, _, tail = rest.partition("/")
        if tail in ("", "state"):
            return game_state(session_id, int(one("since", "0") or 0))
        raise ApiError("no such game endpoint", status=404)
    raise ApiError(f"no route {path}", status=404)


def handle_post(path: str, data: dict) -> dict:
    """Dispatch a POST. ``data`` is the decoded JSON body."""
    if path == "/api/training/start":
        return start_training(data)
    if path == "/api/training/resume":
        return start_training({**data, "resume": True})
    if path == "/api/training/stop":
        return stop_training(data)
    if path == "/api/evaluate":
        return start_evaluation(data)
    if path == "/api/compare":
        return start_comparison(data)
    if path == "/api/benchmark":
        return start_benchmark(data)
    if path == "/api/experiments/run":
        return start_experiment(data)
    if path == "/api/settings":
        return {"ok": True, "settings": store.save_settings(
            data.get("settings", data))}
    if path == "/api/logs/clear":
        LOG.clear()
        LOG.add("info", "log cleared from the control center")
        return {"ok": True}

    if path == "/api/checkpoints/label":
        cid = data.get("id")
        try:
            store.set_label(cid, data.get("label"))
        except InvalidName as e:
            raise ApiError(str(e))
        return {"ok": True}
    if path == "/api/checkpoints/delete":
        cid = data.get("id")
        if not data.get("confirm"):
            raise ApiError("confirm must be true to delete a checkpoint")
        try:
            run_name, snap = store.parse_checkpoint_id(cid)
        except InvalidName as e:
            raise ApiError(str(e))
        if MANAGER.active_with_key(f"run:{run_name}"):
            raise ApiError(
                f"run '{run_name}' is busy; stop its job before deleting",
                status=409)
        try:
            result = store.delete_checkpoint(cid)
        except FileNotFoundError:
            raise ApiError("no such checkpoint", status=404)
        LOG.add("warn", f"deleted checkpoint {cid}")
        return {"ok": True, **result}

    if path == "/api/game/ai/start":
        return start_ai_game(data)
    if path == "/api/game/human/start":
        return start_human_game(data)
    if path.startswith("/api/game/"):
        rest = path[len("/api/game/"):]
        session_id, _, tail = rest.partition("/")
        if tail == "move":
            return game_move(session_id, data)
        if tail == "control":
            return game_control(session_id, data)
        raise ApiError("no such game endpoint", status=404)

    if path.startswith("/api/jobs/") and path.endswith("/stop"):
        job_id = path[len("/api/jobs/"):-len("/stop")]
        if not MANAGER.stop(job_id):
            raise ApiError("that job is not running", status=409)
        return {"ok": True}

    raise ApiError(f"no route {path}", status=404)
