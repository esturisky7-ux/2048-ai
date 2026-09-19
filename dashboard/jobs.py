"""Job manager: long-running work as supervised subprocesses.

The control center must never run training or evaluation inside an HTTP
request handler -- a request that takes two hours is not a request. Everything
slow is therefore a **job**: a child process launched with an explicit argument
list, tracked by this module, and reported on through the API.

Why subprocesses rather than threads
------------------------------------
Three reasons, in order of importance:

1. *The CPython GIL.* Training is pure Python compute. A training thread inside
   the server would fight the server for the interpreter lock and make the UI
   crawl while also slowing training down.
2. *Interruptibility.* A subprocess can be signalled and will run its own
   ``finally`` block, which is exactly how ``train.py`` already checkpoints on
   Ctrl-C. A wedged thread cannot be killed at all.
3. *Blast radius.* A crash in a job cannot take the server down with it.

Graceful stop
-------------
Stopping means "ask politely, then insist": deliver the platform's interrupt,
wait for the child to save its checkpoint and exit, and only escalate to
``terminate()``/``kill()`` if it ignores that. ``train.py`` handles the
interrupt by finishing the current game, writing a checkpoint and flushing
statistics, so a browser-initiated stop is exactly as safe as Ctrl-C.

Conflicting work
----------------
Two training jobs writing the same weight file through separate processes would
interleave their updates and corrupt the run's accounting. A job therefore
declares an *exclusive key* (for training, the run name); starting a second job
with a live exclusive key is refused with a clear error rather than allowed to
race.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Where a job's scratch files live: its spec, progress, result and captured
# output. Under data/ so .gitignore already excludes it.
JOB_DIR = ROOT / "data" / "jobs"

# How long a stopping job gets to save and exit before we escalate. Training
# finishes the game in progress first, and a strong agent's game can be several
# thousand moves, so this is deliberately generous.
GRACE_SECONDS = 90.0
# After escalation, how long terminate() gets before kill().
TERMINATE_SECONDS = 10.0

# Job states. QUEUED exists for completeness; jobs currently start immediately.
QUEUED = "QUEUED"
RUNNING = "RUNNING"
STOPPING = "STOPPING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

TERMINAL_STATES = (COMPLETED, FAILED, CANCELLED)

_ids = itertools.count(1)


def _now() -> float:
    return time.time()


def _popen_kwargs() -> dict:
    """Platform flags that make a child interruptible without killing us.

    On Windows a console control event goes to the whole process *group*, so a
    child must be put in its own group or stopping one job would interrupt the
    server and every other job with it. On POSIX the child gets its own process
    group for the same reason.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _interrupt(proc: subprocess.Popen) -> None:
    """Deliver the platform's "please stop" signal to a child.

    POSIX gets SIGINT, which Python turns into KeyboardInterrupt and which
    ``train.py`` handles by checkpointing. Windows has no SIGINT delivery
    between processes; the nearest equivalent is a console Ctrl-Break event,
    which the trainer also handles.
    """
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except (ProcessLookupError, OSError, ValueError):
        pass          # already gone, or the OS refused; the reaper will notice


class Job:
    """One unit of supervised background work."""

    __slots__ = ("id", "type", "label", "params", "state", "created_at",
                 "started_at", "ended_at", "pid", "error", "result",
                 "exclusive_key", "_proc", "_log_path", "_progress_path",
                 "_result_path", "_stop_requested_at", "_progress_cache",
                 "_progress_mtime")

    def __init__(self, job_type: str, label: str, params: dict,
                 exclusive_key: str | None = None):
        self.id = f"job-{next(_ids):06d}"
        self.type = job_type
        self.label = label
        self.params = params
        self.exclusive_key = exclusive_key
        self.state = QUEUED
        self.created_at = _now()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.pid: int | None = None
        self.error: str | None = None
        self.result = None
        self._proc: subprocess.Popen | None = None
        self._stop_requested_at: float | None = None
        base = JOB_DIR / self.id
        self._log_path = base.with_suffix(".log")
        self._progress_path = base.with_suffix(".progress.json")
        self._result_path = base.with_suffix(".result.json")
        self._progress_cache: dict = {}
        self._progress_mtime = 0.0

    # -- progress ----------------------------------------------------------
    def progress(self) -> dict:
        """Whatever the child last reported. Cached on the file's mtime."""
        try:
            st = self._progress_path.stat()
        except OSError:
            return self._progress_cache
        if st.st_mtime != self._progress_mtime:
            try:
                with open(self._progress_path, encoding="utf-8") as f:
                    self._progress_cache = json.load(f)
                self._progress_mtime = st.st_mtime
            except (OSError, json.JSONDecodeError):
                pass      # a half-written file; keep the previous snapshot
        return self._progress_cache

    def tail(self, lines: int = 200) -> list[str]:
        """Last N lines of the child's captured output."""
        try:
            with open(self._log_path, encoding="utf-8", errors="replace") as f:
                return f.read().splitlines()[-lines:]
        except OSError:
            return []

    def duration(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.ended_at or _now()) - self.started_at

    def to_dict(self, with_progress: bool = True) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "state": self.state,
            "params": self.params,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration(),
            "pid": self.pid,
            "error": self.error,
            "exclusive_key": self.exclusive_key,
        }
        if with_progress:
            d["progress"] = self.progress()
            d["result"] = self.result
        return d


class JobConflict(Exception):
    """Raised when a job would collide with one already running."""


class JobManager:
    """Owns every job, its subprocess, and the reaper thread watching them."""

    def __init__(self, keep: int = 60):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.keep = keep
        self.lock = threading.RLock()
        self._reaper: threading.Thread | None = None
        self._shutdown = threading.Event()
        self.on_event = None          # set by the server to record log events

    # -- lifecycle ---------------------------------------------------------
    def start_reaper(self) -> None:
        if self._reaper is not None:
            return
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="job-reaper")
        self._reaper.start()

    def _emit(self, level: str, message: str, **fields) -> None:
        if self.on_event:
            try:
                self.on_event(level, message, **fields)
            except Exception:
                pass      # logging must never break job control

    def _reap_loop(self) -> None:
        while not self._shutdown.wait(0.4):
            try:
                self._reap_once()
            except Exception:
                pass

    def _reap_once(self) -> None:
        with self.lock:
            live = [j for j in self.jobs.values()
                    if j.state in (RUNNING, STOPPING)]
        for job in live:
            proc = job._proc
            if proc is None:
                continue
            code = proc.poll()
            if code is None:
                # Still alive. If we asked it to stop a while ago, escalate.
                if job.state == STOPPING and job._stop_requested_at:
                    waited = _now() - job._stop_requested_at
                    if waited > GRACE_SECONDS + TERMINATE_SECONDS:
                        proc.kill()
                    elif waited > GRACE_SECONDS:
                        proc.terminate()
                continue
            self._finish(job, code)

    def _finish(self, job: Job, code: int) -> None:
        with self.lock:
            if job.state in TERMINAL_STATES:
                return
            job.ended_at = _now()
            # A stopped job exiting non-zero is normal: an interrupted process
            # reports 130 (or whatever the platform uses) even though it saved
            # correctly, so the state we asked for wins over the exit code.
            if job.state == STOPPING:
                job.state = CANCELLED
            elif code == 0:
                job.state = COMPLETED
            else:
                job.state = FAILED
                tail = [ln for ln in job.tail(40) if ln.strip()]
                job.error = tail[-1] if tail else f"exited with code {code}"
            job.result = self._read_result(job)
        level = "error" if job.state == FAILED else "info"
        self._emit(level, f"{job.type} job {job.state.lower()}",
                   job_id=job.id, label=job.label, error=job.error)

    @staticmethod
    def _read_result(job: Job):
        try:
            with open(job._result_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # -- submitting --------------------------------------------------------
    def submit(self, job_type: str, label: str, argv: list[str],
               params: dict | None = None, exclusive_key: str | None = None,
               env: dict | None = None) -> Job:
        """Launch ``argv`` as a supervised job.

        ``argv`` is always an explicit list and is never passed through a
        shell, so nothing the browser sends can become a command.
        """
        params = params or {}
        with self.lock:
            if exclusive_key:
                clash = self.active_with_key(exclusive_key)
                if clash is not None:
                    raise JobConflict(
                        f"{clash.type} job {clash.id} is already working on "
                        f"'{exclusive_key}'. Stop it first.")
            job = Job(job_type, label, params, exclusive_key)
            JOB_DIR.mkdir(parents=True, exist_ok=True)

            child_env = dict(os.environ)
            child_env["PYTHONUNBUFFERED"] = "1"
            if env:
                child_env.update(env)

            # Popen duplicates the descriptor into the child, so the parent
            # closes its own copy immediately. Holding it would leak one file
            # handle per job for the lifetime of the server.
            log = open(job._log_path, "w", encoding="utf-8", buffering=1)
            try:
                proc = subprocess.Popen(
                    argv, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=child_env,
                    text=True, **_popen_kwargs())
            except OSError as e:
                job.state = FAILED
                job.error = f"could not start: {e}"
                job.ended_at = _now()
                self._register(job)
                raise
            finally:
                log.close()
            job._proc = proc
            job.pid = proc.pid
            job.state = RUNNING
            job.started_at = _now()
            self._register(job)
        self._emit("info", f"{job_type} job started",
                   job_id=job.id, label=label, pid=job.pid)
        self.start_reaper()
        return job

    def _register(self, job: Job) -> None:
        self.jobs[job.id] = job
        self.order.append(job.id)
        # Trim finished jobs, oldest first, so a long session does not grow
        # without bound. Live jobs are never trimmed.
        while len(self.order) > self.keep:
            for i, jid in enumerate(self.order):
                if self.jobs[jid].state in TERMINAL_STATES:
                    self.order.pop(i)
                    self.jobs.pop(jid, None)
                    break
            else:
                break

    # -- queries -----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def active_with_key(self, key: str) -> Job | None:
        for jid in reversed(self.order):
            j = self.jobs.get(jid)
            if j and j.exclusive_key == key and j.state in (RUNNING, STOPPING,
                                                            QUEUED):
                return j
        return None

    def active(self, job_type: str | None = None) -> list[Job]:
        with self.lock:
            return [self.jobs[j] for j in self.order
                    if self.jobs[j].state in (RUNNING, STOPPING, QUEUED)
                    and (job_type is None or self.jobs[j].type == job_type)]

    def recent(self, limit: int = 25, job_type: str | None = None) -> list[Job]:
        with self.lock:
            out = [self.jobs[j] for j in reversed(self.order)
                   if job_type is None or self.jobs[j].type == job_type]
        return out[:limit]

    def current_training(self) -> Job | None:
        jobs = self.active("training")
        return jobs[0] if jobs else None

    # -- stopping ----------------------------------------------------------
    def stop(self, job_id: str) -> bool:
        """Ask a job to stop gracefully. Returns False if it was not running."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.state not in (RUNNING, QUEUED):
                return False
            job.state = STOPPING
            job._stop_requested_at = _now()
            proc = job._proc
        if proc is not None:
            _interrupt(proc)
        self._emit("info", f"{job.type} job stopping", job_id=job.id,
                   label=job.label)
        return True

    def stop_all(self, wait: float = GRACE_SECONDS) -> None:
        """Stop every live job and wait for them. Used on server shutdown."""
        live = self.active()
        for job in live:
            self.stop(job.id)
        if not live:
            return
        deadline = _now() + wait
        while _now() < deadline:
            self._reap_once()
            if not self.active():
                return
            time.sleep(0.25)
        # Anything still alive after the grace period is terminated, then
        # killed. Training has already had its chance to checkpoint.
        for job in self.active():
            proc = job._proc
            if proc and proc.poll() is None:
                proc.terminate()
        time.sleep(1.0)
        for job in self.active():
            proc = job._proc
            if proc and proc.poll() is None:
                proc.kill()
        self._reap_once()

    def shutdown(self) -> None:
        self._shutdown.set()


# The single manager the server uses.
MANAGER = JobManager()
