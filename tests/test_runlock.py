"""Run ownership: one writer per run, enforced by the operating system.

Covers the run lock itself (``training/runlock.py``) and everything that
relies on it -- across real processes, because that is the only place the
failures it prevents can happen:

* two trainers on one run (command line, control center, experiment)
* deleting or resetting a run while something trains it
* a crashed or killed trainer, which must never strand its run
* evaluating a run's *current* weights, which must be refused while it is
  being trained, and must keep training out until the evaluation is done

Everything runs in a private ``AI2048_HOME``, so no real run is touched.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dashboard import api, store                              # noqa: E402
from dashboard import jobs as J                               # noqa: E402
from dashboard.jobs import MANAGER                            # noqa: E402
from training import checkpoint as CP                         # noqa: E402
from training.runlock import RunBusy, busy_message            # noqa: E402

PY = sys.executable
WINDOWS = os.name == "nt"

# Holds a run's lock in a process of its own until its stdin is closed:
# "some other process holds this run", with no trainer or status file at all.
HOLDER = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from training.checkpoint import Run\n"
    "lock = Run(sys.argv[2]).lock(shared=sys.argv[3] == 'shared',\n"
    "                             purpose='test holder').acquire()\n"
    "print('held', flush=True)\n"
    "sys.stdin.read()\n"
)


def popen_interruptible(args, **kw):
    """A child this process can later interrupt the way Ctrl-C would."""
    if WINDOWS:
        kw["creationflags"] = kw.get("creationflags", 0) | \
            subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([PY] + args, cwd=ROOT, **kw)


def interrupt(proc):
    proc.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)


def wait_for(predicate, timeout=60.0, interval=0.1, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")


class HomeTest(unittest.TestCase):
    """A private AI2048_HOME per class, reused by its tests (the n-tuple
    lookup tables are cached there, so only the first trainer builds them).
    In-process roots point at it too, and the job manager is stubbed so an
    API call can never launch a real job by accident."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="2048lock-")
        cls.env = dict(os.environ, AI2048_HOME=cls.home, PYTHONUNBUFFERED="1")
        cls.env.pop("AI2048_START_METHOD", None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.home, ignore_errors=True)

    def setUp(self):
        self.procs = []
        self.submitted = []
        root = Path(self.home)
        self._saved = (CP.CHECKPOINT_ROOT, CP.DATA_ROOT, store.CHECKPOINT_ROOT,
                       store.DATA_ROOT, store.LABELS_PATH, MANAGER.submit,
                       J.JOB_DIR)
        CP.CHECKPOINT_ROOT = store.CHECKPOINT_ROOT = root / "checkpoints"
        CP.DATA_ROOT = store.DATA_ROOT = root / "data"
        store.LABELS_PATH = CP.DATA_ROOT / "checkpoint-labels.json"
        J.JOB_DIR = root / "jobs"
        MANAGER.submit = self._capture_submit

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            try:
                p.communicate(timeout=30)
            except (subprocess.TimeoutExpired, ValueError):
                pass
        (CP.CHECKPOINT_ROOT, CP.DATA_ROOT, store.CHECKPOINT_ROOT,
         store.DATA_ROOT, store.LABELS_PATH, MANAGER.submit,
         J.JOB_DIR) = self._saved

    def _capture_submit(self, job_type, label, argv, params=None,
                        exclusive_key=None, env=None):
        """Record what the API would launch, then refuse to launch it."""
        self.submitted.append({"type": job_type, "argv": argv,
                               "exclusive_key": exclusive_key})
        raise J.JobConflict("(test) the API validated this request")

    # -- helpers -----------------------------------------------------------
    def log_path(self, name):
        return os.path.join(self.home, f"{name}-{len(self.procs)}.log")

    def spawn(self, args, name="proc", **kw):
        log = open(self.log_path(name), "w", encoding="utf-8")
        try:
            p = popen_interruptible(args, stdout=log, stderr=subprocess.STDOUT,
                                    env=self.env, **kw)
        finally:
            log.close()
        p.log = log.name
        self.procs.append(p)
        return p

    def output(self, proc):
        with open(proc.log, encoding="utf-8", errors="replace") as f:
            return f.read()

    def run_cli(self, *args, timeout=300):
        return subprocess.run([PY, *args], cwd=ROOT, env=self.env,
                              capture_output=True, text=True, timeout=timeout)

    def train(self, run, games=20, *extra):
        r = self.run_cli("train.py", "--run", run, "--tuple-set", "8x4",
                         "--games", str(games), "--report-every", "0",
                         "--eval-every", "0", "--checkpoint-every", "0",
                         *extra)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def start_trainer(self, run, *extra):
        """A trainer that runs until interrupted, once it holds the run."""
        p = self.spawn(["train.py", "--run", run, "--tuple-set", "8x4",
                        "--games", "0", "--report-every", "0",
                        "--eval-every", "0", "--checkpoint-every", "50",
                        *extra], name=f"train-{run}")
        self.wait_until_held_by(run, p)
        return p

    def wait_until_held_by(self, run, proc, timeout=60):
        def held():
            if proc.poll() is not None:
                raise AssertionError(f"process exited early:\n"
                                     f"{self.output(proc)}")
            mode, note = CP.Run(run).lock_holder()
            return mode == "exclusive" and (note or {}).get("pid") == proc.pid
        wait_for(held, timeout, what=f"process {proc.pid} to hold {run}")

    def hold(self, run, shared=False):
        """Hold a run's lock from a separate, non-training process."""
        p = subprocess.Popen(
            [PY, "-c", HOLDER, ROOT, run, "shared" if shared else "x"],
            cwd=ROOT, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        line = p.stdout.readline()
        if line.strip() != "held":
            p.kill()
            self.fail(f"the lock holder failed:\n{line}{p.stdout.read()}")
        return p

    def release(self, holder):
        holder.stdin.close()
        holder.wait(timeout=30)
        holder.stdout.close()
        # Reaped: tearDown's communicate() would read the closed pipe, which
        # on Windows fails in a reader thread.
        self.procs.remove(holder)

    def stop(self, proc, timeout=120):
        interrupt(proc)
        proc.wait(timeout=timeout)
        return self.output(proc)

    def assert_nobody_writes(self, run, quiet=3.0):
        """No process -- an orphaned worker, say -- still writes the run.

        A worker that was mid-game when its trainer died may finish that one
        game first; after that, nothing may change.
        """
        path = CP.Run(run).weights_path
        time.sleep(1.0)
        before = path.read_bytes()
        time.sleep(quiet)
        self.assertEqual(path.read_bytes(), before,
                         f"something is still writing run {run}")

    def assert_free(self, run, timeout=15):
        wait_for(lambda: CP.Run(run).lock_holder()[0] is None, timeout,
                 what=f"run {run} to be free")


# ---------------------------------------------------------------------------
# The lock itself
# ---------------------------------------------------------------------------
class TestRunLockSemantics(HomeTest):
    def test_exclusive_excludes_exclusive(self):
        run = CP.Run("sem-x")
        a = run.lock(purpose="first").acquire()
        try:
            with self.assertRaises(RunBusy) as ctx:
                run.lock(purpose="second").acquire(wait=0.2)
            self.assertEqual(ctx.exception.mode, "exclusive")
            self.assertEqual(ctx.exception.holder["pid"], os.getpid())
            self.assertIn("already in use by process", str(ctx.exception))
            self.assertIn("first", str(ctx.exception))
        finally:
            a.release()

    def test_shared_holders_coexist_but_exclude_a_writer(self):
        run = CP.Run("sem-s")
        a = run.lock(shared=True).acquire()
        b = run.lock(shared=True).acquire(wait=0.2)
        try:
            with self.assertRaises(RunBusy) as ctx:
                run.lock().acquire(wait=0.2)
            self.assertEqual(ctx.exception.mode, "shared")
            self.assertIn("evaluation of its current weights",
                          str(ctx.exception))
        finally:
            a.release()
            b.release()
        run.lock().acquire(wait=0.2).release()

    def test_a_writer_excludes_readers_and_says_why(self):
        run = CP.Run("sem-w")
        with run.lock(purpose="training, test").acquire():
            with self.assertRaises(RunBusy) as ctx:
                run.lock(shared=True).acquire(wait=0.2)
        msg = str(ctx.exception)
        self.assertIn("being trained by process", msg)
        self.assertIn("snapshots", msg)

    def test_release_frees_the_run_and_is_idempotent(self):
        run = CP.Run("sem-r")
        lock = run.lock().acquire()
        self.assertTrue(lock.held)
        lock.release()
        lock.release()
        self.assertFalse(lock.held)
        self.assertEqual(run.lock_holder(), (None, None))
        run.lock().acquire(wait=0.2).release()

    def test_the_lock_lives_outside_the_run_it_protects(self):
        run = CP.Run("sem-d")
        run.create_dirs()
        with run.lock().acquire():
            self.assertFalse(str(run.lock_path).startswith(str(run.dir)))
            shutil.rmtree(run.dir)
            self.assertEqual(run.lock_holder()[0], "exclusive")
        self.assertEqual(run.lock_holder(), (None, None))

    def test_listing_runs_ignores_the_lock_directory(self):
        with CP.Run("sem-list").lock().acquire():
            self.assertEqual(CP.list_runs(), [])

    def test_trainer_holds_its_run_until_closed(self):
        from training.config import load_config
        from training.trainer import Trainer
        t = Trainer("sem-t", load_config(overrides={"tuple_set": "8x4"}),
                    quiet=True)
        try:
            mode, note = t.run.lock_holder()
            self.assertEqual(mode, "exclusive")
            self.assertEqual(note["pid"], os.getpid())
        finally:
            t.close()
        self.assertEqual(t.run.lock_holder(), (None, None))

    def test_trainer_releases_its_run_when_setting_up_fails(self):
        from training.config import load_config
        from training.trainer import Trainer
        t = Trainer("sem-f", load_config(overrides={"tuple_set": "8x4"}),
                    quiet=True)
        t.train(n_games=2)
        t.close()
        with self.assertRaises(ValueError):          # tuple set mismatch
            Trainer("sem-f", {"tuple_set": "4x5"}, resume=True, quiet=True)
        self.assertEqual(CP.Run("sem-f").lock_holder(), (None, None))

    def test_busy_messages_are_actionable(self):
        note = {"pid": 4321, "purpose": "training, train.py",
                "since": time.time()}
        msg = busy_message("r", False, "exclusive", note)
        self.assertIn("process 4321 (training, train.py", msg)
        self.assertIn("stop that one first", msg)
        self.assertIn("snapshots", busy_message("r", True, "exclusive", note))
        self.assertIn("another process", busy_message("r", False,
                                                      "exclusive", None))


# ---------------------------------------------------------------------------
# Trainers, deletions and experiments in separate processes
# ---------------------------------------------------------------------------
class TestRunOwnershipAcrossProcesses(HomeTest):
    def test_a_second_trainer_is_refused_and_ctrl_c_releases_the_run(self):
        run = "own-cli"
        first = self.start_trainer(run)
        second = self.run_cli("train.py", "--run", run, "--tuple-set", "8x4",
                              "--games", "5", timeout=120)
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertIn(f"already in use by process {first.pid}", second.stderr)
        self.assertIn("training, train.py", second.stderr)
        self.assertIsNone(first.poll(), "the first trainer was disturbed")

        out = self.stop(first)
        self.assertEqual(first.returncode, 0, out)
        self.assertIn("checkpoint saved", out)
        self.assert_free(run)
        games = CP.Run(run).load_meta()["games"]
        self.train(run, 5, "--resume")
        self.assertEqual(CP.Run(run).load_meta()["games"], games + 5)

    def test_unrelated_runs_train_at_the_same_time(self):
        procs = [self.spawn(["train.py", "--run", name, "--tuple-set", "8x4",
                             "--games", "40", "--report-every", "0",
                             "--eval-every", "0"], name=name)
                 for name in ("own-a", "own-b")]
        for p in procs:
            p.wait(timeout=300)
            self.assertEqual(p.returncode, 0, self.output(p))
        for name in ("own-a", "own-b"):
            self.assertEqual(CP.Run(name).load_meta()["games"], 40)
            self.assert_free(name)

    def test_a_killed_trainer_does_not_strand_its_run(self):
        run = "own-kill"
        self.train(run, 5)
        p = self.start_trainer(run, "--resume")
        p.kill()
        p.wait(timeout=30)
        self.assert_free(run)
        self.train(run, 5, "--resume")

    def test_a_killed_multi_worker_trainer_does_not_strand_its_run(self):
        """Its workers notice the trainer is gone and stop writing the run.

        Forked workers share their parent's lock, so they also hold the run
        until they have gone. Killed twice per start method: while the
        workers may still be starting up -- the moment a worker is most
        likely to miss that it has been orphaned -- and once they are playing.
        """
        import multiprocessing as mp
        for method in [m for m in ("fork", "spawn")
                       if m in mp.get_all_start_methods()]:
            for when in ("starting", "playing"):
                with self.subTest(start_method=method, killed=when):
                    run = f"own-kill-{method}-{when}"
                    self.env["AI2048_START_METHOD"] = method
                    try:
                        self.train(run, 5)
                        p = self.start_trainer(run, "--resume",
                                               "--workers", "2")
                        if when == "playing":
                            status = CP.Run(run).status_path

                            def playing():
                                st = CP.read_json(status) or {}
                                return st.get("pid") == p.pid and \
                                    st.get("session_games", 0) > 0
                            wait_for(playing, 90, what="the workers to play")
                        p.kill()
                        p.wait(timeout=30)
                        self.assert_free(run, timeout=60)
                        self.assert_nobody_writes(run)
                        self.train(run, 5, "--resume")
                    finally:
                        self.env.pop("AI2048_START_METHOD", None)

    def test_deleting_a_run_trained_elsewhere_is_refused(self):
        run = "own-del"
        trainer = self.start_trainer(run)
        wait_for(CP.Run(run).meta_path.exists, 60,
                 what=f"{run}'s first checkpoint")
        with self.assertRaises(api.ApiError) as ctx:
            api.handle_post("/api/checkpoints/delete",
                            {"id": run, "confirm": True})
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn(f"process {trainer.pid}", ctx.exception.message)
        self.assertTrue(CP.Run(run).weights_path.exists())
        self.assertIsNone(trainer.poll())

        out = self.stop(trainer)
        self.assertEqual(trainer.returncode, 0, out)
        result = api.handle_post("/api/checkpoints/delete",
                                 {"id": run, "confirm": True})
        self.assertEqual(result["kind"], "run")
        self.assertFalse(CP.Run(run).dir.exists())

    def test_the_control_center_refuses_a_run_held_by_any_process(self):
        """Not a trainer, no status file: only the lock says it is busy."""
        run = "own-api"
        self.train(run, 5)
        holder = self.hold(run)
        self.assertFalse(CP.Run(run).status_path.exists() and CP.read_json(
            CP.Run(run).status_path).get("running"))
        with self.assertRaises(api.ApiError) as ctx:
            api.start_training({"run": run, "resume": True, "games": 5})
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn(f"process {holder.pid}", ctx.exception.message)
        self.assertEqual(self.submitted, [])
        self.release(holder)
        with self.assertRaises(api.ApiError):          # (test) submit stub
            api.start_training({"run": run, "resume": True, "games": 5})
        self.assertEqual(len(self.submitted), 1)

    def test_a_control_center_training_job_fails_cleanly_against_a_cli_one(
            self):
        run = "own-job"
        trainer = self.start_trainer(run)
        mgr = J.JobManager(keep=5)
        saved_dir, J.JOB_DIR = J.JOB_DIR, Path(self.home) / "jobs"
        try:
            argv = api._training_argv({"run": run, "resume": False,
                                       "tuple_set": "8x4", "games": 5,
                                       "workers": 1})
            job = mgr.submit("training", "resume", argv,
                             params={"run": run}, env=self.env)
            wait_for(lambda: (mgr._reap_once() or
                              job.state in J.TERMINAL_STATES), 120,
                     what="the job to end")
            self.assertEqual(job.state, J.FAILED, job.tail())
            self.assertIn(f"already in use by process {trainer.pid}",
                          job.error)
        finally:
            mgr.stop_all(wait=10)
            mgr.shutdown()
            J.JOB_DIR = saved_dir
        self.assertIsNone(trainer.poll())

    def test_a_fresh_experiment_cannot_reset_a_run_being_trained(self):
        run = "own-exp"
        trainer = self.start_trainer(run)
        # Holding the lock comes before the trainer's first file; wait for a
        # checkpoint, so a reset would have something to destroy.
        wait_for(CP.Run(run).meta_path.exists, 60,
                 what=f"{run}'s first checkpoint")
        weights = CP.Run(run).weights_path
        inode = os.stat(weights).st_ino
        spec = os.path.join(self.home, "exp-own.json")
        with open(spec, "w", encoding="utf-8") as f:
            json.dump({"description": "test", "kind": "train", "games": 5,
                       "eval_games": 2, "run": run, "tuple_set": "8x4"}, f)
        r = self.run_cli("experiment.py", "--run", spec, "--fresh",
                         "--quiet", timeout=120)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn(f"already in use by process {trainer.pid}", r.stderr)
        self.assertEqual(os.stat(weights).st_ino, inode,
                         "the run was reset under its trainer")
        self.assertTrue(CP.Run(run).meta_path.exists())
        self.assertIsNone(trainer.poll())

    def test_a_dashboard_experiment_is_keyed_by_the_run_it_trains(self):
        from experiments.runner import experiment_run_name
        name = "alpha-high"
        run = experiment_run_name(name)
        self.assertEqual(run, "exp-alpha-high")
        holder = self.hold(run)
        with self.assertRaises(api.ApiError) as ctx:
            api.start_experiment({"name": name, "games": 5})
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn(f"process {holder.pid}", ctx.exception.message)
        self.release(holder)
        with self.assertRaises(api.ApiError):          # (test) submit stub
            api.start_experiment({"name": name, "games": 5})
        self.assertEqual(self.submitted[-1]["exclusive_key"], f"run:{run}")
        # Agent experiments train nothing and keep their own key.
        with self.assertRaises(api.ApiError):
            api.start_experiment({"name": "search-depth-1"})
        self.assertEqual(self.submitted[-1]["exclusive_key"],
                         "experiment:search-depth-1")


# ---------------------------------------------------------------------------
# Fixed evaluation of a run's current weights
# ---------------------------------------------------------------------------
class TestFrozenEvaluation(HomeTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = subprocess.run(
            [PY, "train.py", "--run", "frozen", "--tuple-set", "8x4",
             "--games", "40", "--snapshot-every", "20", "--report-every", "0",
             "--eval-every", "0"], cwd=ROOT, env=cls.env,
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout + r.stderr

    def test_a_read_only_view_of_current_weights_still_changes(self):
        """Why any of this is needed: the reader sees the writer's updates."""
        from training.ntuple import NTupleNetwork
        path = str(CP.Run("frozen").weights_path)
        reader = NTupleNetwork("8x4", path=path, readonly=True)
        writer = NTupleNetwork("8x4", path=path)
        try:
            board = 0x0000_0000_0000_4321
            before = reader.value(board)
            writer.update(board, 1.0)
            self.assertNotEqual(reader.value(board), before)
            writer.update(board, -1.0)
        finally:
            writer.close()
            reader.close()

    def test_the_api_refuses_live_weights_but_accepts_snapshots(self):
        holder = self.hold("frozen")
        snap = "frozen:games-000000020.f32"
        try:
            with self.assertRaises(api.ApiError) as ctx:
                api.start_evaluation({"agent": "learned", "run": "frozen",
                                      "games": 5})
            self.assertEqual(ctx.exception.status, 409)
            self.assertIn("being trained", ctx.exception.message)
            self.assertIn("snapshots", ctx.exception.message)
            # The current weights named by checkpoint id are just as live.
            with self.assertRaises(api.ApiError) as ctx:
                api.start_evaluation({"agent": "learned", "checkpoint": "frozen",
                                      "games": 5})
            self.assertEqual(ctx.exception.status, 409)
            with self.assertRaises(api.ApiError) as ctx:
                api.start_comparison({"agents": ["random", {
                    "agent": "learned", "run": "frozen"}], "games": 5})
            self.assertEqual(ctx.exception.status, 409)
            self.assertEqual(self.submitted, [])

            with self.assertRaises(api.ApiError):      # (test) submit stub
                api.start_evaluation({"agent": "learned", "checkpoint": snap,
                                      "games": 5})
            with self.assertRaises(api.ApiError):
                api.start_comparison({"agents": ["random", {
                    "agent": "learned", "checkpoint": snap}], "games": 5})
            self.assertEqual([s["type"] for s in self.submitted],
                             ["evaluation", "comparison"])
        finally:
            self.release(holder)

    def test_the_api_refuses_while_a_training_job_is_starting(self):
        """A job launched from here may not hold the lock yet."""
        job = J.Job("training", "starting", {"run": "frozen"},
                    exclusive_key="run:frozen")
        job.state = J.RUNNING
        MANAGER.jobs[job.id] = job
        MANAGER.order.append(job.id)
        try:
            with self.assertRaises(api.ApiError) as ctx:
                api.start_evaluation({"agent": "learned", "run": "frozen",
                                      "games": 5})
            self.assertEqual(ctx.exception.status, 409)
            self.assertIn(job.id, ctx.exception.message)
        finally:
            MANAGER.jobs.pop(job.id, None)
            MANAGER.order.remove(job.id)

    def test_evaluate_py_refuses_live_weights_but_accepts_snapshots(self):
        holder = self.hold("frozen")
        try:
            for args in (["--agent", "learned"],
                         ["--compare", "random", "learned"]):
                r = self.run_cli("evaluate.py", "--run", "frozen", "--games",
                                 "3", "--no-save", "--quiet", *args)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn("being trained", r.stderr)
                self.assertIn("--checkpoint", r.stderr)
                self.assertNotIn("mean score", r.stdout)   # nothing was played
            snap = str(CP.Run("frozen").snapshot_path(20))
            r = self.run_cli("evaluate.py", "--agent", "learned",
                             "--checkpoint", snap, "--games", "3",
                             "--no-save", "--quiet")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("mean score", r.stdout)
        finally:
            self.release(holder)

    def test_an_evaluation_keeps_training_out_until_it_is_done(self):
        """The whole contract, end to end, through the dashboard's runner."""
        from agents.learned import LearnedAgent
        from evaluation.evaluator import evaluate

        run = CP.Run("frozen")
        reference = os.path.join(self.home, "reference.f32")
        shutil.copyfile(run.weights_path, reference)
        jobs = Path(self.home) / "runner"
        jobs.mkdir(exist_ok=True)
        spec_path = jobs / "eval.spec.json"
        spec = {"type": "evaluation", "agent": "learned", "run": "frozen",
                "depth": 1, "games": 400, "seed": 5, "agent_seed": 0,
                "save": True, "progress_path": str(jobs / "eval.progress.json"),
                "result_path": str(jobs / "eval.result.json")}
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        before = run.eval_path.read_text(encoding="utf-8") \
            if run.eval_path.exists() else ""

        evaluator = self.spawn(["-m", "dashboard.runner", str(spec_path)],
                               name="runner")
        wait_for(lambda: run.lock_holder()[0] == "shared" or
                 evaluator.poll() is not None, 60,
                 what="the evaluation to hold the run")
        self.assertIsNone(evaluator.poll(), "the evaluation ended too soon "
                          "to test anything; give it more games")
        refused = self.run_cli("train.py", "--run", "frozen", "--resume",
                               "--games", "5", timeout=120)
        self.assertEqual(refused.returncode, 1,
                         refused.stdout + refused.stderr)
        self.assertIn("in use by an evaluation", refused.stderr)
        self.assertIsNone(evaluator.poll(),
                          "training was only refused because the "
                          "evaluation had already finished")

        evaluator.wait(timeout=300)
        self.assertEqual(evaluator.returncode, 0, self.output(evaluator))
        with open(spec["result_path"], encoding="utf-8") as f:
            result = json.load(f)

        # Exactly what the weights as they stood beforehand score.
        agent = LearnedAgent(run="frozen", checkpoint=reference)
        try:
            frozen = evaluate(agent, games=400, seed=5)
        finally:
            agent.close()
        for key in ("games", "mean_score", "median_score", "max_score",
                    "highest_tile", "tile_histogram", "mean_moves"):
            self.assertEqual(result[key], frozen[key], key)

        # Training resumes afterwards, and what was recorded stays recorded.
        record = run.eval_path.read_text(encoding="utf-8")
        self.assertTrue(record.startswith(before))
        self.assertEqual(len(record.splitlines()),
                         len(before.splitlines()) + 1)
        self.train("frozen", 10, "--resume")
        self.assertEqual(run.eval_path.read_text(encoding="utf-8"), record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
