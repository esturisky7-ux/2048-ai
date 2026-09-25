"""Training orchestration: the loop, statistics, checkpoints and workers.

Responsibilities kept here rather than in ``learner.py`` so the learning
algorithm stays readable on its own:

* own the run: hold its lock (:mod:`training.runlock`) from before the first
  file is touched until the trainer is closed
* drive games and feed results into rolling / all-time statistics
* checkpoint on a game-count interval, and always on exit
* snapshot and evaluate on their own intervals, with the weights holding still
* append history rows so progress can be graphed afterwards
* publish a small status file the dashboard polls
* shut down cleanly on Ctrl-C, saving before it exits
* optionally fan out across processes, and fail loudly if one of them dies

Determinism. Each game's tile spawns come from ``Random(game_seed(seed, i))``,
so with one worker a run is exactly reproducible and resuming continues the
same sequence. With more than one worker every index is still played exactly
once -- including across Ctrl-C and ``--resume`` -- but weight updates
interleave between processes, so results are reproducible only up to that
interleaving. That is the cost of Hogwild-style parallelism and is called out
in the README.

Portability. Workers share weights through a memory map of one file, which
every supported OS keeps coherent between processes, so ``--workers`` behaves
the same everywhere. Only the way the processes are *started* differs: Linux
forks (the child inherits the already-built lookup tables for free), while
Windows and macOS spawn a fresh interpreter, which costs a few seconds of
table building per worker at startup and nothing thereafter.
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import time
import traceback
from random import Random

from evaluation.evaluator import game_seed
from .checkpoint import Run, display_path
from .learner import TDLearner
from .ntuple import NTupleNetwork
from .reward import RewardFunction
from .stats import AllTimeStats, History, RollingStats

# Results are shipped from workers to the parent in batches; per-game IPC
# would cost more than a game does.
WORKER_BATCH = 16
# How often a worker pushes its dirty weight pages to disk. Flushing a 268 MB
# mapping every batch is far more msync traffic than durability needs.
WORKER_FLUSH_EVERY = 512
# The dashboard decides "is training alive?" from how recently the status file
# was touched. Reporting intervals are measured in games, and a strong agent
# plays only a few games a minute, so status gets its own wall-clock heartbeat.
STATUS_HEARTBEAT = 3.0
# Workers that have finished their games get this long to exit on their own;
# after a failure the survivors get the shorter one. Then terminate(), and
# after that kill().
WORKER_EXIT_GRACE = 30.0
WORKER_ABORT_GRACE = 3.0
WORKER_KILL_GRACE = 5.0
# A worker found dead before saying it had finished gets this long for its
# last messages (usually the traceback of what killed it) to arrive.
WORKER_LAST_WORDS = 1.0
# "No end": the stop line of a segment that trains until Ctrl-C.
_NO_END = 1 << 62


class TrainingError(RuntimeError):
    """Training could not play the games it was asked to.

    ``details`` holds a worker's traceback when there is one.
    """

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.details = details


def worker_start_method() -> str:
    """How worker processes should be created on this platform.

    ``fork`` is used on Linux because the child inherits the parent's already
    built move tables and its open weight mapping, so a worker starts
    instantly and costs no extra memory. It is deliberately *not* used on
    macOS: fork there is only safe in a process that has never touched the
    system frameworks, and CPython has defaulted away from it since 3.8.
    Windows has no fork at all. Both fall back to ``spawn``, where the worker
    maps the same weight file itself and gets the same shared memory.

    Set ``AI2048_START_METHOD`` to override, which is mainly how the spawn
    path gets exercised on a Linux machine.
    """
    import multiprocessing as mp
    methods = mp.get_all_start_methods()
    forced = os.environ.get("AI2048_START_METHOD")
    if forced:
        if forced not in methods:
            raise ValueError(
                f"AI2048_START_METHOD={forced!r} is not available here; "
                f"choose from {methods}")
        return forced
    if sys.platform.startswith("linux") and "fork" in methods:
        return "fork"
    return "spawn" if "spawn" in methods else methods[0]


def alpha_at(cfg: dict, games: int) -> float:
    """Learning rate for a given game count (optional geometric decay)."""
    lc = cfg["learning"]
    a = lc["alpha"]
    decay = lc.get("alpha_decay", 1.0)
    every = lc.get("alpha_decay_every", 0) or 0
    if decay != 1.0 and every > 0:
        a *= decay ** (games // every)
    return max(a, lc.get("alpha_min", 0.0))


class Trainer:
    """Owns one training run, from construction until :meth:`close`.

    Constructing a trainer takes the run's exclusive lock before any of the
    run's files is read or opened for writing, and raises
    :class:`training.runlock.RunBusy` if another process holds the run. Pass
    ``lock`` to hand over a lock the caller already holds (an experiment that
    resets the run first does this), in which case the caller releases it.
    """

    def __init__(self, run_name: str = "default", config: dict | None = None,
                 resume: bool = False, quiet: bool = False, lock=None,
                 purpose: str = "training"):
        self.run = Run(run_name)
        self.quiet = quiet
        self.stop_requested = False
        self.net = None
        self._owns_lock = lock is None
        self.lock = lock if lock is not None else \
            self.run.lock(purpose=purpose).acquire()
        try:
            self._setup(config, resume)
        except BaseException:
            self.close()
            raise

    def _setup(self, config: dict | None, resume: bool) -> None:
        from .config import load_config

        if resume:
            if not self.run.exists():
                raise FileNotFoundError(
                    f"cannot resume: no checkpoint at {self.run.meta_path}")
            saved = self.run.load_config() or {}
            # Saved config wins over defaults, explicit overrides win over both.
            from .config import deep_merge
            self.config = deep_merge(load_config(), saved)
            if config:
                self.config = deep_merge(self.config, config)
            meta = self.run.load_meta() or {}
            self.game_index = meta.get("games", 0)
            self.all_time = AllTimeStats(meta.get("all_time"))
        else:
            self.config = config or load_config()
            meta = None
            self.game_index = 0
            self.all_time = AllTimeStats()

        self.run.create_dirs()
        self.tuple_set = self.config["tuple_set"]
        if resume and meta and meta.get("tuple_set") != self.tuple_set:
            raise ValueError(
                f"checkpoint was trained with tuple set "
                f"{meta.get('tuple_set')!r} but config says "
                f"{self.tuple_set!r}; use a different --run name")

        self.net = NTupleNetwork(self.tuple_set, path=self.run.weights_path)
        self.reward = RewardFunction(self.config.get("reward"))
        lc = self.config["learning"]
        self.learner = TDLearner(
            self.net, self.reward, alpha=alpha_at(self.config, self.game_index),
            gamma=lc.get("gamma", 1.0), epsilon=lc.get("epsilon", 0.0))

        tc = self.config["training"]
        self.rolling = RollingStats(tc.get("rolling_window", 1000))
        self.history = History(self.run.history_path)
        self.seed = tc.get("seed", 12345)
        self.report_every = tc.get("report_every", 200)
        self.checkpoint_every = tc.get("checkpoint_every", 2000)
        self.snapshot_every = tc.get("snapshot_every", 0)
        self.eval_every = tc.get("eval_every", 0)
        self.eval_games = tc.get("eval_games", 200)

        self.run.save_config(self.config)
        # session_t0 marks when this session began and never moves.
        # _banked_at marks when cumulative train_seconds was last updated;
        # only that one advances on a checkpoint.
        self.session_t0 = time.perf_counter()
        self._banked_at = self.session_t0
        self.session_games = 0
        self.session_moves = 0
        self.last_eval = (self.run.load_meta() or {}).get("last_eval") \
            if resume else None
        self._last_report_t = self.session_t0
        # Must start from the resumed game index, not 0, or the first rate
        # after --resume counts every game the run has ever played.
        self._last_report_games = self.game_index
        self._last_report_moves = 0
        # Results arrive in batches from workers, so the raw interval rate is
        # bursty; an EMA gives a number that means something at a glance.
        self._gps_ema = 0.0
        self._mps_ema = 0.0
        self._last_status_t = 0.0

    def close(self) -> None:
        """Unmap the weights and give up the run. Safe to call twice."""
        if self.net is not None:
            try:
                self.net.close()
            except (BufferError, ValueError, OSError):
                pass
            self.net = None
        if self.lock is not None:
            if self._owns_lock:
                self.lock.release()
            self.lock = None

    # -- signals -----------------------------------------------------------
    def install_signal_handlers(self) -> dict:
        """Turn an interrupt into a clean checkpoint instead of lost work.

        Ctrl-C reaches Python as SIGINT on every OS. Windows also has
        Ctrl-Break (SIGBREAK), which is the only interrupt another process can
        deliver there, so it is handled the same way; SIGTERM does not exist
        on Windows and is registered only where it does.

        Returns the handlers that were replaced, for
        :meth:`restore_signal_handlers`.
        """
        def handler(signum, frame):
            if self.stop_requested:
                self._log("\nsecond interrupt -- exiting immediately")
                sys.exit(130)
            self.stop_requested = True
            self._log("\nstopping after the current game; "
                      "checkpointing (Ctrl-C again to force quit)")
        previous = {}
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, handler)
            except (ValueError, OSError):
                pass        # not the main thread, or not supported here
        return previous

    @staticmethod
    def restore_signal_handlers(previous: dict) -> None:
        """Put back what :meth:`install_signal_handlers` replaced, so a
        process that goes on after training (an experiment evaluating its
        result) is interruptible the ordinary way again."""
        for sig, old in previous.items():
            try:
                signal.signal(sig, old if old is not None else signal.SIG_DFL)
            except (ValueError, OSError, TypeError):
                pass

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)

    # -- checkpointing -----------------------------------------------------
    def save(self, note: str = "") -> None:
        now = time.perf_counter()
        self.all_time.train_seconds += now - self._banked_at
        self._banked_at = now
        self.net.flush()
        self.run.save_meta({
            "games": self.game_index,
            "tuple_set": self.tuple_set,
            "all_time": self.all_time.dump(),
            "rolling": self.rolling.summary(),
            "alpha": self.learner.alpha,
            "seed": self.seed,
            "last_eval": self.last_eval,
            "note": note,
        })

    def take_snapshot(self) -> None:
        """Freeze the weights as they stand after exactly ``game_index`` games.

        Called only at a boundary where nothing is writing the weights: between
        games with one worker, and with every worker stopped with several.
        """
        games = self.game_index
        if self.run.snapshot_path(games).exists():
            # Snapshots are immutable. This one was taken when an earlier
            # session passed the same game count (it may since have crashed
            # and resumed from an older checkpoint); it is not rewritten.
            self._log(f"  snapshot for game {games:,} already exists; "
                      f"kept as it is")
            return
        self.net.flush()
        path = self.run.snapshot(games)
        self._log(f"  snapshot -> {display_path(path)}")

    # -- reporting ---------------------------------------------------------
    def _rate(self):
        """Smoothed (games/s, moves/s) plus the session average games/s."""
        now = time.perf_counter()
        dt = now - self._last_report_t
        dg = self.game_index - self._last_report_games
        dm = self.session_moves - self._last_report_moves
        self._last_report_t = now
        self._last_report_games = self.game_index
        self._last_report_moves = self.session_moves
        if dt > 0:
            gps, mps = dg / dt, dm / dt
            k = 0.3
            self._gps_ema = gps if self._gps_ema == 0 else \
                k * gps + (1 - k) * self._gps_ema
            self._mps_ema = mps if self._mps_ema == 0 else \
                k * mps + (1 - k) * self._mps_ema
        session_dt = now - self.session_t0
        session_gps = (self.session_games / session_dt) if session_dt > 0 else 0.0
        return self._gps_ema, self._mps_ema, session_gps

    def report(self) -> None:
        gps, mps, session_gps = self._rate()
        roll = self.rolling.summary()
        row = {
            "games": self.game_index,
            "session_games": self.session_games,
            "games_per_second": gps,
            "moves_per_second": mps,
            "session_games_per_second": session_gps,
            "alpha": self.learner.alpha,
            "rolling": roll,
            "all_time": self.all_time.summary(),
        }
        self.history.append(row)
        self._last_status_t = time.perf_counter()
        self.write_status(running=True, gps=gps, mps=mps)
        if roll.get("games"):
            r = roll["tile_rates"]
            self._log(
                f"games {self.game_index:>9,} | {gps:5.1f} g/s "
                f"{mps:6.0f} mv/s | a={self.learner.alpha:.4f} | "
                f"mean {roll['mean_score']:>8,.0f} "
                f"med {roll['median_score']:>8,.0f} "
                f"best {self.all_time.best_score:>8,} | "
                f"tile {roll['max_tile']:>5,} | "
                f"512 {r['512']*100:5.1f}% "
                f"1k {r['1024']*100:5.1f}% "
                f"2k {r['2048']*100:5.1f}% "
                f"4k {r['4096']*100:5.1f}%")

    def maybe_write_status(self) -> None:
        """Heartbeat the status file on a timer, independent of game counts."""
        now = time.perf_counter()
        if now - self._last_status_t >= STATUS_HEARTBEAT:
            self._last_status_t = now
            self.write_status(running=True, gps=self._gps_ema,
                              mps=self._mps_ema)

    def write_status(self, running: bool, gps: float = 0.0,
                     mps: float = 0.0) -> None:
        self.run.write_status({
            "run": self.run.name,
            "running": running,
            "pid": os.getpid(),
            "games": self.game_index,
            "session_games": self.session_games,
            "games_per_second": gps,
            "moves_per_second": mps,
            "session_moves": self.session_moves,
            "session_seconds": time.perf_counter() - self.session_t0,
            "total_train_seconds": (self.all_time.train_seconds
                                    + (time.perf_counter() - self._banked_at)),
            "alpha": self.learner.alpha,
            "tuple_set": self.tuple_set,
            "rolling": self.rolling.summary(),
            "all_time": self.all_time.summary(),
            "last_eval": self.last_eval,
            "updated_at": time.time(),
        })

    # -- periodic evaluation ----------------------------------------------
    def maybe_evaluate(self) -> None:
        if not self.eval_every or self.game_index % self.eval_every:
            return
        self.evaluate_now()

    def evaluate_now(self) -> None:
        """Run the fixed evaluation procedure on the policy as it stands.

        Only ever called while nothing writes the weights (see
        :meth:`_at_boundary`), so the policy really is frozen. An evaluation
        cut short by Ctrl-C is not recorded: a partial result is not
        comparable with the full-length ones around it.
        """
        from evaluation.evaluator import evaluate
        if self.stop_requested:
            self._log("  evaluation skipped: stopping")
            return
        self._log(f"  evaluating ({self.eval_games} fixed games)...")
        res = evaluate(PolicyAgent(self.learner.policy(), self.game_index),
                       games=self.eval_games,
                       seed=self.config["evaluation"]["seed"],
                       progress=lambda done, total: self.maybe_write_status(),
                       stop_flag=lambda: self.stop_requested)
        if res.get("games", 0) < self.eval_games:
            self._log(f"  evaluation interrupted after {res.get('games', 0)} "
                      f"of {self.eval_games} games; not recorded")
            return
        res["games_trained"] = self.game_index
        self.last_eval = {
            "games_trained": self.game_index,
            "eval_games": res["games"],
            "mean_score": res["mean_score"],
            "median_score": res["median_score"],
            "ci95_mean": res["ci95_mean"],
            "highest_tile": res["highest_tile"],
            "tile_rates": {k: v["rate"] for k, v in res["tile_rates"].items()},
            "timestamp": res["timestamp"],
        }
        with open(self.run.eval_path, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(res, separators=(",", ":")) + "\n")
        self.save(note="evaluation")
        self._log(
            f"  eval @ {self.game_index:,} games: mean "
            f"{res['mean_score']:,.0f} "
            f"[{res['ci95_mean'][0]:,.0f}, {res['ci95_mean'][1]:,.0f}] "
            f"median {res['median_score']:,.0f} "
            f"best tile {res['highest_tile']:,}")

    # -- main loops --------------------------------------------------------
    def train(self, n_games: int = 0, workers: int = 1) -> None:
        """Play ``n_games`` more games (0 = until interrupted).

        Raises :class:`TrainingError` if a finite target could not be
        reached -- a worker process failed, for instance. A stop asked for with
        Ctrl-C is not an error. Either way the checkpoint is saved first.
        """
        previous_handlers = self.install_signal_handlers()
        target = self.game_index + n_games if n_games else 0
        self._log(
            f"run '{self.run.name}'  tuple set {self.tuple_set} "
            f"({self.net.n_weights:,} weights, {self.net.nbytes()/1e6:.0f} MB)\n"
            f"starting at game {self.game_index:,}"
            + (f", training to {target:,}" if target else ", until Ctrl-C")
            + f"  |  alpha {self.learner.alpha:g}, {workers} worker(s)")
        try:
            if workers > 1:
                self._train_parallel(target, workers)
            else:
                self._train_serial(target)
            if target and self.game_index < target and not self.stop_requested:
                raise TrainingError(
                    f"training ended at game {self.game_index:,}, short of "
                    f"its target of {target:,}")
        finally:
            self.save(note="final")
            self.write_status(running=False)
            self._log(
                f"\nstopped at {self.game_index:,} games "
                f"({self.session_games:,} this session). "
                f"checkpoint saved to {display_path(self.run.meta_path)}")
            self.restore_signal_handlers(previous_handlers)

    def _record_game(self, index: int, score: int, moves: int,
                     tile: int) -> None:
        """Count one finished game (``index`` is its seed index)."""
        self.game_index += 1
        self.session_games += 1
        self.session_moves += moves
        self.rolling.add(score, moves, tile)
        self.all_time.add(score, moves, tile)

    def _after_game(self) -> None:
        """Bookkeeping due after a game; safe while other workers still play."""
        gi = self.game_index
        if gi % 64 == 0:
            self.learner.set_alpha(alpha_at(self.config, gi))
        if self.report_every and gi % self.report_every == 0:
            self.report()
        if self.checkpoint_every and gi % self.checkpoint_every == 0:
            self.save()

    def _at_boundary(self) -> None:
        """Work that needs the weights to hold still: snapshots, evaluation.

        Each has its own schedule, independent of checkpointing. When several
        fall on the same game they share one pass: one snapshot, then the
        evaluation.
        """
        gi = self.game_index
        if self.snapshot_every and gi and gi % self.snapshot_every == 0:
            self.take_snapshot()
        self.maybe_evaluate()

    def _next_boundary(self, target: int) -> int:
        """The next game count at which the weights must hold still (0: none)."""
        gi = self.game_index
        ends = [target] if target else []
        for every in (self.eval_every, self.snapshot_every):
            if every:
                ends.append((gi // every + 1) * every)
        return min(ends) if ends else 0

    def _train_serial(self, target: int) -> None:
        learner = self.learner
        while not self.stop_requested:
            if target and self.game_index >= target:
                break
            index = self.game_index
            rng = Random(game_seed(self.seed, index))
            score, moves, tile = learner.play_game(rng)
            self._record_game(index, score, moves, tile)
            self.maybe_write_status()
            self._after_game()
            self._at_boundary()

    def _train_parallel(self, target: int, workers: int) -> None:
        """Hogwild training in segments, with a clean boundary between them.

        Workers play disjoint, interleaved game indices and all write the one
        shared weight table. Snapshots and evaluations need that table to hold
        still, so training runs in segments that end exactly at the next such
        point: every worker plays its share of the games before it, reports,
        and exits; the boundary work then runs with no writer alive; and a
        fresh set of workers carries on from the very next index. Checkpoints
        still happen mid-segment, as results arrive, as they always have.
        """
        import multiprocessing as mp

        method = worker_start_method()
        ctx = mp.get_context(method)
        if method != "fork":
            self._log(f"  starting {workers} worker(s) with '{method}'; "
                      f"each builds its lookup tables once (a few seconds)")
        while not self.stop_requested:
            if target and self.game_index >= target:
                break
            _Segment(self, ctx, method, workers,
                     self._next_boundary(target)).run()
            self._at_boundary()


class PolicyAgent:
    """A trained policy as an agent the evaluator can play.

    Used for the trainer's periodic evaluation and an experiment's final one,
    which evaluate the in-memory network directly.
    """

    name = "learned"

    def __init__(self, policy, games_trained: int):
        self.policy = policy
        self.games_trained = games_trained

    def new_game(self) -> None:
        self.policy.new_game()

    def act(self, board: int) -> int:
        return self.policy.act(board)

    def describe(self) -> dict:
        return {"name": "learned", "games_trained": self.games_trained}


def _describe_exit(code) -> tuple[str, str]:
    """How a process ended, and a hint about why: ``("with exit code 3",
    "")`` or ``("after being killed by SIGKILL", " (... out of memory?)")``."""
    if code is None:
        return "without an exit code", ""
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        hint = " (the system may have run out of memory)" \
            if name == "SIGKILL" else ""
        return f"after being killed by {name}", hint
    return f"with exit code {code}", ""


class _Segment:
    """One set of worker processes playing the game indices [start, end).

    Worker ``w`` of ``n`` plays ``start + w``, ``start + w + n``, ... in order,
    so the workers between them cover every index exactly once. Before
    playing an index a worker *claims* it, under a lock shared with the
    parent, and it never claims one at or past ``stop_at``. That line starts
    at the segment's end and moves only when training is asked to stop:
    see :meth:`_draw_stop_line`.

    The parent counts results as they arrive, checks that each worker's
    indices arrive in order with none missing, and watches for workers that
    fail. A worker that raises reports its traceback before it exits; one
    that dies outright is found by its exit code. Either way the others are
    stopped, and :class:`TrainingError` says which worker failed and how.
    """

    def __init__(self, trainer: Trainer, ctx, method: str, workers: int,
                 end: int):
        self.t = trainer
        self.ctx = ctx
        self.start = trainer.game_index
        self.end = end
        self.n = max(1, min(workers, end - self.start)) if end else workers
        # Under fork a worker inherits the live network; under spawn nothing
        # is inherited and an mmap cannot be pickled, so the worker is told
        # where the weight file is and maps it itself.
        self.net_spec = trainer.net if method == "fork" else \
            (trainer.tuple_set, str(trainer.run.weights_path))
        self.forked_from = os.getpid() if method == "fork" else None
        self.q = ctx.Queue(maxsize=256)
        self.claim = ctx.Lock()
        self.claimed = ctx.RawArray("q", [-1] * self.n)
        self.stop_at = ctx.RawValue("q", end if end else _NO_END)
        self.next_index = [self.start + w for w in range(self.n)]
        self.finished = [False] * self.n
        self.procs: list = []
        self.failure: str | None = None
        self.details: str | None = None
        self.stopping = False

    # -- the segment -------------------------------------------------------
    def run(self) -> None:
        t = self.t
        try:
            for w in range(self.n):
                p = self.ctx.Process(
                    target=_worker_main, name=f"train-worker-{w}",
                    args=(w, self.n, self.net_spec, t.config, t.seed,
                          self.start, self.q, self.claim, self.claimed,
                          self.stop_at, self.forked_from),
                    daemon=True)
                p.start()
                self.procs.append(p)
            self._supervise()
        finally:
            self._shut_down(aborting=self.failure is not None
                            or sys.exc_info()[0] is not None)
        if self.failure is None and t.game_index != self.stop_at.value:
            self._fail(f"the workers played {t.game_index - self.start:,} "
                       f"games where {self.stop_at.value - self.start:,} "
                       f"were due")
        if self.failure is not None:
            raise TrainingError(
                f"{self.failure}. Training stopped at game "
                f"{t.game_index:,}.", self.details)

    def _fail(self, why: str, details: str | None = None) -> None:
        if self.failure is None:
            self.failure = why
            self.details = details

    def _supervise(self) -> None:
        t = self.t
        # The parent only aggregates; it never plays, so Ctrl-C is responsive.
        while not all(self.finished) and self.failure is None:
            if t.stop_requested and not self.stopping:
                self.stopping = True
                self._draw_stop_line()
            try:
                msg = self.q.get(timeout=0.5)
            except queue.Empty:
                self._check_for_dead_workers()
                t.maybe_write_status()
                continue
            self._handle(msg)
            t.maybe_write_status()

    def _handle(self, msg) -> None:
        kind, w = msg[0], msg[1]
        if kind == "games":
            for index, score, moves, tile in msg[2]:
                if index != self.next_index[w]:
                    self._fail(f"worker {w} of {self.n} reported game "
                               f"{index:,} when game {self.next_index[w]:,} "
                               f"was due")
                    return
                self.next_index[w] += self.n
                self.t._record_game(index, score, moves, tile)
                self.t._after_game()
        elif kind == "done":
            self.finished[w] = True
            if msg[2] != self.next_index[w]:
                self._fail(f"worker {w} of {self.n} stopped before game "
                           f"{msg[2]:,}, but its results end before game "
                           f"{self.next_index[w]:,}")
        elif kind == "error":
            tb = (msg[2] or "").strip()
            last = tb.splitlines()[-1] if tb else "unknown error"
            self._fail(f"worker {w} of {self.n} failed: {last}", tb)

    def _check_for_dead_workers(self) -> None:
        dead = [w for w, p in enumerate(self.procs)
                if not self.finished[w] and not p.is_alive()]
        if not dead:
            return
        # A worker's last messages can still be in flight when it exits:
        # its "done", or the traceback of whatever killed it.
        deadline = time.monotonic() + WORKER_LAST_WORDS
        while self.failure is None and time.monotonic() < deadline \
                and not all(self.finished[w] for w in dead):
            try:
                self._handle(self.q.get(timeout=0.1))
            except queue.Empty:
                pass
        for w in dead:
            if not self.finished[w]:
                how, hint = _describe_exit(self.procs[w].exitcode)
                self._fail(f"worker {w} of {self.n} exited {how} before "
                           f"finishing its games{hint}")

    def _draw_stop_line(self) -> None:
        """Choose where a stop lands, so that the games played stay a prefix.

        Every index below the line gets played and none at or above it:
        setting the line just past the highest index any worker has claimed
        lets the games in flight finish, lets a worker that is behind catch up
        on its indices below the line, and then stops them all. The run ends
        with exactly the games [0, line) played, so --resume carries on
        without playing any game twice or skipping one.
        """
        if not self.claim.acquire(timeout=10):
            self._fail("a worker stopped responding while holding the "
                       "claim lock")
            return
        try:
            top = max(self.claimed[:]) if self.n else -1
            line = max(self.start, top + 1)
            if line < self.stop_at.value:
                self.stop_at.value = line
        finally:
            self.claim.release()

    def _shut_down(self, aborting: bool) -> None:
        """Let the workers exit (stopping them first when aborting), then make
        sure every one of them is gone: terminate, kill, and join each."""
        if aborting:
            # Nobody claims another game; each worker stops after its current
            # one. A worker that died holding the lock cannot block this.
            got = self.claim.acquire(timeout=1)
            self.stop_at.value = 0
            if got:
                self.claim.release()
        deadline = time.monotonic() + (WORKER_ABORT_GRACE if aborting
                                       else WORKER_EXIT_GRACE)
        while any(p.is_alive() for p in self.procs) \
                and time.monotonic() < deadline:
            # Keep draining: a worker blocked on a full queue cannot exit.
            try:
                msg = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                break
            if not aborting:
                self._handle(msg)
        forced = set()
        for w, p in enumerate(self.procs):
            if p.is_alive():
                forced.add(w)
                p.terminate()
        for p in self.procs:
            p.join(WORKER_KILL_GRACE)
            if p.is_alive():
                p.kill()
                p.join()
        if not aborting:
            for w, p in enumerate(self.procs):
                if w in forced:
                    # Every game it played was counted; only its exit hung.
                    self.t._log(f"  worker {w} of {self.n} finished its "
                                f"games but did not exit; it was stopped")
                elif p.exitcode != 0:
                    how, hint = _describe_exit(p.exitcode)
                    self._fail(f"worker {w} of {self.n} exited {how} after "
                               f"its last game{hint}")
        try:
            self.q.close()
        except (OSError, ValueError):
            pass


def _open_worker_network(net_spec):
    """Get the shared weight table inside a worker. Returns ``(net, mine)``.

    Under ``fork`` the parent's network object arrives intact and must not be
    closed by the child. Under ``spawn`` the worker receives only
    ``(tuple_set, weights_path)`` and maps that file itself; mapping the same
    file from two processes yields the same shared pages on Linux, macOS and
    Windows alike, so the Hogwild scheme is unchanged.
    """
    if isinstance(net_spec, tuple):
        tuple_set, path = net_spec
        return NTupleNetwork(tuple_set, path=path), True
    return net_spec, False


def _parent_watch(forked_from: int | None):
    """A zero-argument test for "is the trainer that started me alive?".

    Under ``fork`` the answer is the parent pid: an orphan is re-parented, so
    ``os.getppid()`` stops matching the trainer's pid, which the trainer
    recorded before forking (a worker that asked for its parent's pid only
    once it was running could already be an orphan, and would never notice).
    multiprocessing's own ``parent_process().is_alive()`` cannot be trusted
    there, because every later sibling inherits the trainer's end of the pipe
    it watches. Everywhere else -- ``spawn``, and all of Windows -- it can.
    """
    if forked_from is not None:
        return lambda: os.getppid() == forked_from
    import multiprocessing as mp
    parent = mp.parent_process()
    return parent.is_alive if parent is not None else (lambda: True)


def _worker_main(wid, n_workers, net_spec, config, seed, start_index,
                 out_q, claim, claimed, stop_at, forked_from=None):
    """Child process: play seeded games and update the shared weights.

    The weights are one memory-mapped file shared by every worker, so they all
    write into the *same* memory. Updates are not locked (Hogwild): n-tuple
    updates touch a handful of the millions of weights, collisions are rare,
    and the lost-update noise is small next to the TD error itself.

    Messages to the parent: ``("games", wid, [(index, score, moves, tile),
    ...])``, then ``("done", wid, next_index)``, or ``("error", wid,
    traceback)`` if anything goes wrong.
    """
    # The parent handles interrupts and says when to stop. One aimed at the
    # whole process group -- Ctrl-C in a terminal, or the control center's
    # Ctrl-Break on Windows -- must not kill a worker half-way through a game.
    # SIGTERM, on the other hand, is how the parent ends a worker that will
    # not stop, so it keeps its default action: a forked worker would
    # otherwise have inherited the trainer's own "stop after this game"
    # handler, and shrugged it off.
    for name, action in (("SIGINT", signal.SIG_IGN),
                         ("SIGBREAK", signal.SIG_IGN),
                         ("SIGTERM", signal.SIG_DFL)):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, action)
            except (ValueError, OSError):
                pass
    parent_alive = _parent_watch(forked_from)

    def send(msg) -> bool:
        while True:
            try:
                out_q.put(msg, timeout=1.0)
                return True
            except queue.Full:
                if not parent_alive():
                    return False

    net, mine = None, False
    try:
        net, mine = _open_worker_network(net_spec)
        reward = RewardFunction(config.get("reward"))
        lc = config["learning"]
        learner = TDLearner(net, reward, alpha=alpha_at(config, start_index),
                            gamma=lc.get("gamma", 1.0),
                            epsilon=lc.get("epsilon", 0.0))
        idx = start_index + wid
        batch = []
        played = 0
        while True:
            with claim:
                if idx >= stop_at.value:
                    break
                claimed[wid] = idx
            if not parent_alive():
                # Nobody is left to count these games, and a new trainer may
                # already own the run: stop writing it, and do not wait for
                # queued messages that nobody will read.
                out_q.cancel_join_thread()
                return
            score, moves, tile = learner.play_game(Random(game_seed(seed, idx)))
            batch.append((idx, score, moves, tile))
            idx += n_workers
            played += 1
            if played % 64 == 0:
                learner.set_alpha(alpha_at(config, idx))
            if len(batch) >= WORKER_BATCH:
                if not send(("games", wid, batch)):
                    out_q.cancel_join_thread()
                    return
                batch = []
            if played % WORKER_FLUSH_EVERY == 0:
                net.flush()
        net.flush()
        if (batch and not send(("games", wid, batch))) \
                or not send(("done", wid, idx)):
            out_q.cancel_join_thread()
    except BaseException:
        try:
            out_q.put(("error", wid, traceback.format_exc()), timeout=5)
        except Exception:
            pass
        raise
    finally:
        if mine and net is not None:
            try:
                net.close()
            except Exception:
                pass
