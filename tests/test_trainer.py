"""The trainer's boundaries: evaluations, snapshots and failures.

Snapshots and evaluations need the weights to hold still, which with several
workers means stopping every one of them first; a worker that fails must fail
the whole run, loudly. These tests watch that happen in a fresh process per
case (``train_driver.py``), once per multiprocessing start method this
platform has, so the ``spawn`` path Windows and macOS use is exercised on
Linux as well.
"""

import hashlib
import json
import multiprocessing as mp
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

from training import checkpoint as CP                         # noqa: E402

PY = sys.executable
WINDOWS = os.name == "nt"
DRIVER = os.path.join(ROOT, "tests", "train_driver.py")
METHODS = [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()]


def sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class DriverTest(unittest.TestCase):
    """A private AI2048_HOME per class; each test uses its own run names."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="2048trainer-")
        cls.env = dict(os.environ, AI2048_HOME=cls.home, PYTHONUNBUFFERED="1")
        cls.env.pop("AI2048_START_METHOD", None)
        cls.env.pop("DRIVER_FAULT", None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.home, ignore_errors=True)

    def setUp(self):
        self._roots = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        CP.CHECKPOINT_ROOT = Path(self.home) / "checkpoints"
        CP.DATA_ROOT = Path(self.home) / "data"

    def tearDown(self):
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = self._roots

    def environment(self, method=None, fault=None):
        env = dict(self.env)
        if method:
            env["AI2048_START_METHOD"] = method
        if fault:
            env["DRIVER_FAULT"] = fault
        return env

    def observe(self, run, sessions, method=None):
        r = subprocess.run([PY, DRIVER, "observe",
                            json.dumps({"run": run, "sessions": sessions})],
                           cwd=ROOT, env=self.environment(method),
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def train(self, run, *args, method=None, fault=None, driver=False,
              expect=0):
        cmd = [DRIVER, "cli"] if driver else ["train.py"]
        r = subprocess.run([PY, *cmd, "--run", run, "--tuple-set", "8x4",
                            "--report-every", "0", *args],
                           cwd=ROOT, env=self.environment(method, fault),
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, expect, r.stdout + r.stderr)
        return r

    def snapshots(self, run):
        return [int(Path(p).stem.split("-")[1])
                for p in CP.Run(run).list_snapshots()]

    def evaluations(self, run):
        path = CP.Run(run).eval_path
        if not path.exists():
            return []
        return [json.loads(line)["games_trained"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]


# ---------------------------------------------------------------------------
# Periodic evaluation
# ---------------------------------------------------------------------------
class TestParallelEvaluation(DriverTest):
    def test_workers_stop_for_each_evaluation_at_exactly_its_game(self):
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"eval-{method}"
                seen = self.observe(run, [{"games": 40, "workers": 3,
                                           "eval_every": 12}], method)
                self.assertEqual([e["games"] for e in seen["evals"]],
                                 [12, 24, 36])
                for e in seen["evals"]:
                    self.assertEqual(e["workers_alive"], 0, e)
                    self.assertEqual(e["before"], e["after"],
                                     "the weights changed during an "
                                     "evaluation")
                self.assertEqual(self.evaluations(run), [12, 24, 36])
                self.assertEqual(seen["sessions"][0]["game_index"], 40)
                self.assertEqual(CP.Run(run).load_meta()["games"], 40)
                self.assertEqual(sorted(seen["indices"]), list(range(40)))

    def test_one_worker_evaluates_on_the_same_schedule(self):
        seen = self.observe("eval-serial", [{"games": 40, "workers": 1,
                                             "eval_every": 12}])
        self.assertEqual([e["games"] for e in seen["evals"]], [12, 24, 36])
        self.assertEqual(seen["indices"], list(range(40)))

    def test_every_game_index_is_played_once_across_stop_and_resume(self):
        """A stop lands on a line every worker reaches, so the games played
        are always exactly [0, n): nothing repeated, nothing skipped."""
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"prefix-{method}"
                seen = self.observe(run, [
                    {"games": 0, "workers": 3, "stop_after": 20},
                    {"games": 0, "workers": 2, "resume": True,
                     "stop_after": 7},
                    {"games": 13, "workers": 3, "resume": True},
                ], method)
                first, second, third = seen["sessions"]
                self.assertTrue(first["stopped"] and second["stopped"])
                self.assertFalse(third["stopped"])
                self.assertEqual(third["game_index"],
                                 second["game_index"] + 13)
                total = third["game_index"]
                self.assertEqual(len(seen["indices"]), total,
                                 "a game index was played twice")
                self.assertEqual(sorted(seen["indices"]), list(range(total)))
                self.assertEqual(CP.Run(run).load_meta()["games"], total)

    def test_an_evaluation_cut_short_by_a_stop_is_not_recorded(self):
        """A partial evaluation is not comparable with full-length ones."""
        for when, stop in (("before", {"stop_after": 5}),
                           ("during", {"stop_in_eval": 3})):
            with self.subTest(stopped=when):
                run = f"eval-stop-{when}"
                seen = self.observe(run, [{"games": 0, "workers": 1,
                                           "eval_every": 5, "eval_games": 400,
                                           **stop}])
                self.assertEqual([e["games"] for e in seen["evals"]], [5])
                self.assertEqual(self.evaluations(run), [])
                meta = CP.Run(run).load_meta()
                self.assertEqual(meta["games"], 5)
                self.assertIsNone(meta["last_eval"])


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------
class TestWorkerFailures(DriverTest):
    ARGS = ("--games", "300", "--workers", "2", "--eval-every", "0",
            "--checkpoint-every", "0")

    def check_failed(self, run, r, reason):
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        self.assertIn(reason, lines[-1], "the reason is not the last line")
        self.assertTrue(lines[-1].startswith("error: "), lines[-1])
        self.assertIn("Training stopped at game", lines[-1])
        self.assertIn("checkpoint saved", r.stdout)
        meta = CP.Run(run).load_meta()
        self.assertLess(meta["games"], 300)
        self.assertEqual(CP.Run(run).lock_holder(), (None, None))
        self.assertFalse(CP.read_json(CP.Run(run).status_path)["running"])
        # The healthy worker was stopped too: nothing writes the run now.
        weights = CP.Run(run).weights_path.read_bytes()
        time.sleep(1.5)
        self.assertEqual(CP.Run(run).weights_path.read_bytes(), weights,
                         "a worker outlived its failed trainer")

    def test_a_worker_exception_fails_the_run(self):
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"fail-raise-{method}"
                r = self.train(run, *self.ARGS, method=method,
                               fault="raise:1:4", driver=True, expect=1)
                self.check_failed(run, r, "worker 1 of 2 failed: "
                                          "ValueError: simulated worker "
                                          "failure")
                # The worker's own traceback is in the log too.
                self.assertIn('raise ValueError("simulated worker failure")',
                              r.stderr)

    def test_a_worker_that_exits_fails_the_run(self):
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"fail-exit-{method}"
                r = self.train(run, *self.ARGS, method=method,
                               fault="exit:1:4", driver=True, expect=1)
                self.check_failed(run, r, "worker 1 of 2 exited with exit "
                                          "code 3 before finishing its games")

    @unittest.skipIf(WINDOWS, "SIGKILL is a POSIX signal")
    def test_a_worker_killed_by_a_signal_fails_the_run(self):
        run = "fail-kill"
        r = self.train(run, *self.ARGS, fault="kill:0:4", driver=True,
                       expect=1)
        self.check_failed(run, r, "worker 0 of 2 exited after being killed "
                                  "by SIGKILL")
        self.assertIn("out of memory", r.stderr)

    def test_a_failed_worker_fails_the_control_center_job(self):
        from dashboard import jobs as J
        saved, J.JOB_DIR = J.JOB_DIR, Path(self.home) / "jobs"
        mgr = J.JobManager(keep=5)
        try:
            env = {"AI2048_HOME": self.home, "DRIVER_FAULT": "raise:1:4"}
            job = mgr.submit("training", "doomed",
                             [PY, DRIVER, "cli", "--run", "fail-job",
                              "--tuple-set", "8x4", *self.ARGS],
                             params={"run": "fail-job"}, env=env)
            deadline = time.time() + 300
            while job.state not in J.TERMINAL_STATES and \
                    time.time() < deadline:
                mgr._reap_once()
                time.sleep(0.2)
            self.assertEqual(job.state, J.FAILED, "\n".join(job.tail()))
            self.assertIn("worker 1 of 2 failed: ValueError: simulated "
                          "worker failure", job.error)
            self.assertIn("Traceback", "\n".join(job.tail(200)))
        finally:
            mgr.stop_all(wait=10)
            mgr.shutdown()
            J.JOB_DIR = saved

    def test_ctrl_c_still_stops_several_workers_cleanly(self):
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"ctrl-c-{method}"
                kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} \
                    if WINDOWS else {}
                p = subprocess.Popen(
                    [PY, "train.py", "--run", run, "--tuple-set", "8x4",
                     "--games", "0", "--workers", "2", "--report-every", "0",
                     "--eval-every", "0", "--checkpoint-every", "0"],
                    cwd=ROOT, env=self.environment(method),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, **kw)
                try:
                    status = CP.Run(run).status_path
                    deadline = time.time() + 120
                    while time.time() < deadline and not (
                            CP.read_json(status) or {}).get("games"):
                        time.sleep(0.2)
                    p.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS
                                  else signal.SIGINT)
                    out, _ = p.communicate(timeout=120)
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.communicate()
                self.assertEqual(p.returncode, 0, out)
                self.assertIn("stopping after the current game", out)
                self.assertIn("checkpoint saved", out)
                self.assertNotIn("error", out.lower())
                self.assertGreater(CP.Run(run).load_meta()["games"], 0)
                self.assertEqual(CP.Run(run).lock_holder(), (None, None))


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
class TestSnapshots(DriverTest):
    def test_snapshots_keep_their_own_schedule(self):
        self.train("snap-odd", "--games", "12", "--snapshot-every", "2",
                   "--checkpoint-every", "3", "--eval-every", "0")
        self.assertEqual(self.snapshots("snap-odd"), [2, 4, 6, 8, 10, 12])

    def test_snapshots_without_periodic_checkpoints(self):
        self.train("snap-nock", "--games", "12", "--snapshot-every", "2",
                   "--checkpoint-every", "0", "--eval-every", "0")
        self.assertEqual(self.snapshots("snap-nock"), [2, 4, 6, 8, 10, 12])

    def test_coincident_boundaries_take_one_snapshot_each(self):
        r = self.train("snap-same", "--games", "12", "--snapshot-every", "4",
                       "--checkpoint-every", "4", "--eval-every", "4",
                       "--eval-games", "2")
        self.assertEqual(self.snapshots("snap-same"), [4, 8, 12])
        self.assertEqual(r.stdout.count("snapshot ->"), 3, r.stdout)
        self.assertNotIn("already exists", r.stdout)
        self.assertEqual(self.evaluations("snap-same"), [4, 8, 12])

    def test_a_resumed_run_continues_the_schedule_without_repeating(self):
        run = "snap-resume"
        self.train(run, "--games", "5", "--snapshot-every", "2",
                   "--checkpoint-every", "0", "--eval-every", "0")
        self.assertEqual(self.snapshots(run), [2, 4])
        r = self.train(run, "--resume", "--games", "5")
        self.assertEqual(self.snapshots(run), [2, 4, 6, 8, 10])
        # Resuming exactly on a boundary does not take that snapshot again.
        stamp = os.stat(CP.Run(run).snapshot_path(10)).st_mtime_ns
        r = self.train(run, "--resume", "--games", "2")
        self.assertEqual(self.snapshots(run), [2, 4, 6, 8, 10, 12])
        self.assertNotIn("already exists", r.stdout)
        self.assertEqual(os.stat(CP.Run(run).snapshot_path(10)).st_mtime_ns,
                         stamp)

    def test_multi_worker_snapshots_are_taken_with_no_writer(self):
        for method in METHODS:
            with self.subTest(start_method=method):
                run = f"snap-workers-{method}"
                seen = self.observe(run, [{"games": 24, "workers": 2,
                                           "snapshot_every": 8}], method)
                self.assertEqual([s["games"] for s in seen["snapshots"]],
                                 [8, 16, 24])
                for s in seen["snapshots"]:
                    self.assertEqual(s["workers_alive"], 0, s)
                    self.assertEqual(s["weights"], s["snapshot"],
                                     "the snapshot is not the weights as "
                                     "they stood at its boundary")
                self.assertEqual(self.snapshots(run), [8, 16, 24])
                self.assertEqual(sorted(seen["indices"]), list(range(24)))

    def test_snapshots_never_change_after_training_resumes(self):
        for workers in ("1", "2"):
            with self.subTest(workers=workers):
                run = f"snap-frozen-{workers}"
                self.train(run, "--games", "8", "--workers", workers,
                           "--snapshot-every", "4", "--eval-every", "0")
                before = {g: sha(CP.Run(run).snapshot_path(g))
                          for g in (4, 8)}
                self.train(run, "--resume", "--games", "30",
                           "--workers", workers)
                after = {g: sha(CP.Run(run).snapshot_path(g))
                         for g in (4, 8)}
                self.assertEqual(before, after)
                self.assertNotEqual(sha(CP.Run(run).weights_path),
                                    before[8], "training did not continue")

    def test_a_run_rewound_by_a_crash_keeps_its_snapshot(self):
        """Weights run ahead of the last checkpoint: after a crash the run
        resumes from an earlier game count, reaches a snapshot's game again
        with different weights, and must leave the snapshot alone."""
        run = "snap-crash"
        self.train(run, "--games", "8", "--snapshot-every", "6",
                   "--checkpoint-every", "4", "--eval-every", "0")
        original = sha(CP.Run(run).snapshot_path(6))
        meta_path = CP.Run(run).meta_path
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["games"] = 4                     # the last checkpoint before it
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        r = self.train(run, "--resume", "--games", "4")
        self.assertIn("snapshot for game 6 already exists", r.stdout)
        self.assertEqual(sha(CP.Run(run).snapshot_path(6)), original)
        self.assertEqual(self.snapshots(run), [6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
