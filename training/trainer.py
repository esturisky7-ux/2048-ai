"""Training orchestration: the loop, statistics, checkpoints and workers.

Responsibilities kept here rather than in ``learner.py`` so the learning
algorithm stays readable on its own:

* drive games and feed results into rolling / all-time statistics
* checkpoint on a game-count interval, and always on exit
* append history rows so progress can be graphed afterwards
* publish a small status file the dashboard polls
* shut down cleanly on Ctrl-C, saving before it exits
* optionally fan out across processes

Determinism. Each game's tile spawns come from ``Random(game_seed(seed, i))``,
so with one worker a run is exactly reproducible and resuming continues the
same sequence. With more than one worker the games themselves are still
seeded, but weight updates interleave between processes, so results are
reproducible only up to that interleaving. That is the cost of Hogwild-style
parallelism and is called out in the README.

Portability. Workers share weights through a memory map of one file, which
every supported OS keeps coherent between processes, so ``--workers`` behaves
the same everywhere. Only the way the processes are *started* differs: Linux
forks (the child inherits the already-built lookup tables for free), while
Windows and macOS spawn a fresh interpreter, which costs a few seconds of
table building per worker at startup and nothing thereafter.
"""

from __future__ import annotations

import os
import signal
import sys
import time
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
    """Owns one training run."""

    def __init__(self, run_name: str = "default", config: dict | None = None,
                 resume: bool = False, quiet: bool = False):
        from .config import load_config

        self.run = Run(run_name)
        self.quiet = quiet
        self.stop_requested = False

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

    # -- signals -----------------------------------------------------------
    def install_signal_handlers(self) -> None:
        """Turn an interrupt into a clean checkpoint instead of lost work.

        Ctrl-C reaches Python as SIGINT on every OS. Windows also has
        Ctrl-Break (SIGBREAK), which is the only interrupt another process can
        deliver there, so it is handled the same way; SIGTERM does not exist
        on Windows and is registered only where it does.
        """
        def handler(signum, frame):
            if self.stop_requested:
                self._log("\nsecond interrupt -- exiting immediately")
                sys.exit(130)
            self.stop_requested = True
            self._log("\nstopping after the current game; "
                      "checkpointing (Ctrl-C again to force quit)")
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass        # not the main thread, or not supported here

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
        if self.snapshot_every and self.game_index \
                and self.game_index % self.snapshot_every == 0:
            path = self.run.snapshot(self.game_index)
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
        """Freeze the policy and run the fixed evaluation procedure."""
        from evaluation.evaluator import evaluate
        self._log(f"  evaluating ({self.eval_games} fixed games)...")

        class _Frozen:
            name = "learned"
            def __init__(self, learner, games):
                self._l = learner
                self._g = games
            def act(self, b):
                return self._l.best_action(b)
            def new_game(self):
                pass
            def describe(self):
                return {"name": "learned", "games_trained": self._g}

        res = evaluate(_Frozen(self.learner, self.game_index),
                       games=self.eval_games,
                       seed=self.config["evaluation"]["seed"],
                       stop_flag=lambda: self.stop_requested)
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
        """Play ``n_games`` more games (0 = until interrupted)."""
        self.install_signal_handlers()
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
        finally:
            self.save(note="final")
            self.write_status(running=False)
            self._log(
                f"\nstopped at {self.game_index:,} games "
                f"({self.session_games:,} this session). "
                f"checkpoint saved to {display_path(self.run.meta_path)}")

    def _train_serial(self, target: int) -> None:
        learner = self.learner
        cfg = self.config
        rolling = self.rolling
        all_time = self.all_time
        while not self.stop_requested:
            if target and self.game_index >= target:
                break
            rng = Random(game_seed(self.seed, self.game_index))
            score, moves, tile = learner.play_game(rng)
            self.game_index += 1
            self.session_games += 1
            self.session_moves += moves
            rolling.add(score, moves, tile)
            all_time.add(score, moves, tile)

            self.maybe_write_status()
            if self.game_index % 64 == 0:
                learner.set_alpha(alpha_at(cfg, self.game_index))
            if self.report_every and self.game_index % self.report_every == 0:
                self.report()
            if self.checkpoint_every \
                    and self.game_index % self.checkpoint_every == 0:
                self.save()
            self.maybe_evaluate()

    def _train_parallel(self, target: int, workers: int) -> None:
        import multiprocessing as mp

        method = worker_start_method()
        ctx = mp.get_context(method)
        # With fork the worker inherits the live network object. With spawn
        # nothing is inherited and an mmap cannot be pickled, so the worker is
        # told where the weight file is and maps it itself.
        net_spec = self.net if method == "fork" else \
            (self.tuple_set, str(self.run.weights_path))
        if method != "fork":
            self._log(f"  starting {workers} worker(s) with '{method}'; "
                      f"each builds its lookup tables once (a few seconds)")
        q = ctx.Queue(maxsize=256)
        stop = ctx.Event()
        procs = []
        start_index = self.game_index
        for wid in range(workers):
            p = ctx.Process(target=_worker_main,
                            args=(wid, workers, net_spec, self.config,
                                  self.seed, start_index, target, q, stop),
                            daemon=True)
            p.start()
            procs.append(p)

        # The parent only aggregates; it never plays, so Ctrl-C is responsive.
        finished = 0
        try:
            while finished < workers:
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    if self.stop_requested:
                        stop.set()
                    if not any(p.is_alive() for p in procs):
                        break
                    self.maybe_write_status()
                    continue
                if item is None:
                    finished += 1
                    continue
                for score, moves, tile in item:
                    self.game_index += 1
                    self.session_games += 1
                    self.session_moves += moves
                    self.rolling.add(score, moves, tile)
                    self.all_time.add(score, moves, tile)
                    if self.report_every \
                            and self.game_index % self.report_every == 0:
                        self.report()
                    if self.checkpoint_every \
                            and self.game_index % self.checkpoint_every == 0:
                        self.save()
                self.maybe_write_status()
                if self.stop_requested:
                    stop.set()
        finally:
            stop.set()
            for p in procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()


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


def _worker_main(wid, n_workers, net_spec, config, seed, start_index, target,
                 out_q, stop_ev):
    """Child process: play seeded games and update the shared weights.

    The weights are one memory-mapped file shared by every worker, so they all
    write into the *same* memory. Updates are not locked (Hogwild): n-tuple
    updates touch a handful of the millions of weights, collisions are rare,
    and the lost-update noise is small next to the TD error itself.
    """
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)   # parent handles Ctrl-C

    net, mine = _open_worker_network(net_spec)
    reward = RewardFunction(config.get("reward"))
    lc = config["learning"]
    learner = TDLearner(net, reward, alpha=alpha_at(config, start_index),
                        gamma=lc.get("gamma", 1.0),
                        epsilon=lc.get("epsilon", 0.0))
    idx = start_index + wid
    batch = []
    played = 0
    try:
        while not stop_ev.is_set():
            if target and idx >= target:
                break
            rng = Random(game_seed(seed, idx))
            batch.append(learner.play_game(rng))
            idx += n_workers
            played += 1
            if played % 64 == 0:
                learner.set_alpha(alpha_at(config, idx))
            if len(batch) >= WORKER_BATCH:
                out_q.put(batch)
                batch = []
            if played % WORKER_FLUSH_EVERY == 0:
                net.flush()
        if batch:
            out_q.put(batch)
    finally:
        try:
            net.flush()
            out_q.put(None)
        except Exception:
            pass
        if mine:
            try:
                net.close()
            except Exception:
                pass
