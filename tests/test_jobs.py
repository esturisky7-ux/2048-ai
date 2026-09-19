"""Job manager tests: the thing that keeps long work out of HTTP handlers.

These use short-lived real subprocesses rather than mocks, because the whole
point of the job manager is how it behaves around process boundaries —
starting, signalling, reaping and refusing to double-book. A mock process
cannot fail in the ways a real one can.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import jobs as J                               # noqa: E402


def wait_until(predicate, timeout=25.0, interval=0.1):
    """Poll until a predicate holds, driving the reaper as we go."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        J.MANAGER._reap_once()
        if predicate():
            return True
        time.sleep(interval)
    return False


class JobManagerTest(unittest.TestCase):
    """Each test gets its own manager and job directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="2048jobs-")
        self._job_dir = J.JOB_DIR
        J.JOB_DIR = Path(self.tmp)
        self.mgr = J.JobManager(keep=10)
        self._real = J.MANAGER
        J.MANAGER = self.mgr

    def tearDown(self):
        self.mgr.stop_all(wait=8)
        self.mgr.shutdown()
        J.MANAGER = self._real
        J.JOB_DIR = self._job_dir
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def python(self, code, *args):
        return [sys.executable, "-c", code, *args]


class TestLifecycle(JobManagerTest):
    def test_successful_job_reaches_completed(self):
        job = self.mgr.submit("test", "prints and exits",
                              self.python("print('hello')"))
        self.assertEqual(job.state, J.RUNNING)
        self.assertIsNotNone(job.pid)
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))
        self.assertEqual(job.state, J.COMPLETED)
        self.assertIsNotNone(job.ended_at)
        self.assertGreater(job.duration(), 0)
        self.assertIn("hello", "\n".join(job.tail()))

    def test_failing_job_reaches_failed_and_keeps_the_error(self):
        job = self.mgr.submit("test", "raises",
                              self.python("raise SystemExit('boom')"))
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))
        self.assertEqual(job.state, J.FAILED)
        self.assertTrue(job.error)
        self.assertIn("boom", "\n".join(job.tail()))

    def test_unlaunchable_command_fails_cleanly(self):
        with self.assertRaises(OSError):
            self.mgr.submit("test", "nonexistent",
                            [os.path.join(self.tmp, "definitely-not-a-program")])
        # The failed job is still recorded, so the UI can explain it.
        jobs = self.mgr.recent(5)
        self.assertTrue(jobs)
        self.assertEqual(jobs[0].state, J.FAILED)

    def test_job_dict_is_json_safe(self):
        import json
        job = self.mgr.submit("test", "quick", self.python("pass"))
        wait_until(lambda: job.state in J.TERMINAL_STATES)
        json.dumps(job.to_dict())          # must not raise


class TestStopping(JobManagerTest):
    """The graceful-stop path, which is what protects a training run."""

    # A child that catches the interrupt, saves, and exits — exactly the
    # shape train.py has.
    SAVER = (
        "import signal, sys, time, os\n"
        "stop = {'v': False}\n"
        "def handler(sig, frame):\n"
        "    stop['v'] = True\n"
        "for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):\n"
        "    sig = getattr(signal, name, None)\n"
        "    if sig is not None:\n"
        "        try: signal.signal(sig, handler)\n"
        "        except Exception: pass\n"
        "print('ready', flush=True)\n"
        "while not stop['v']:\n"
        "    time.sleep(0.05)\n"
        "open(sys.argv[1], 'w').write('saved')\n"
        "print('checkpoint saved', flush=True)\n"
    )

    def test_stop_lets_the_child_save_before_exiting(self):
        marker = os.path.join(self.tmp, "saved.txt")
        job = self.mgr.submit("training", "saver",
                              self.python(self.SAVER, marker))
        self.assertTrue(wait_until(lambda: "ready" in "\n".join(job.tail())))

        self.assertTrue(self.mgr.stop(job.id))
        self.assertEqual(job.state, J.STOPPING)
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))

        # Stopped on request is CANCELLED, not FAILED, whatever exit code the
        # platform produced.
        self.assertEqual(job.state, J.CANCELLED)
        self.assertTrue(os.path.exists(marker),
                        "the child was killed before it could save")
        self.assertEqual(open(marker, encoding="utf-8").read(), "saved")

    def test_stop_is_idempotent_and_safe_on_dead_jobs(self):
        job = self.mgr.submit("test", "quick", self.python("pass"))
        wait_until(lambda: job.state in J.TERMINAL_STATES)
        self.assertFalse(self.mgr.stop(job.id))
        self.assertFalse(self.mgr.stop("job-does-not-exist"))

    def test_stop_all_stops_everything(self):
        marker = os.path.join(self.tmp, "a.txt")
        marker2 = os.path.join(self.tmp, "b.txt")
        self.mgr.submit("training", "a", self.python(self.SAVER, marker))
        self.mgr.submit("training", "b", self.python(self.SAVER, marker2),
                        exclusive_key="other")
        self.assertTrue(wait_until(lambda: len(self.mgr.active()) == 2))
        self.mgr.stop_all(wait=20)
        self.assertEqual(self.mgr.active(), [])

    def test_an_unresponsive_child_is_escalated(self):
        """A child that ignores the interrupt must not block shutdown."""
        ignorer = (
            "import signal, time\n"
            "for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):\n"
            "    sig = getattr(signal, name, None)\n"
            "    if sig is not None:\n"
            "        try: signal.signal(sig, signal.SIG_IGN)\n"
            "        except Exception: pass\n"
            "print('ready', flush=True)\n"
            "while True: time.sleep(0.05)\n"
        )
        job = self.mgr.submit("test", "ignores signals", self.python(ignorer))
        self.assertTrue(wait_until(lambda: "ready" in "\n".join(job.tail())))
        # Shorten the grace period so the test does not take 90 seconds.
        real_grace = J.GRACE_SECONDS
        J.GRACE_SECONDS = 1.0
        try:
            self.mgr.stop(job.id)
            self.assertTrue(wait_until(
                lambda: job.state in J.TERMINAL_STATES, timeout=30))
        finally:
            J.GRACE_SECONDS = real_grace


class TestConflicts(JobManagerTest):
    def test_two_jobs_cannot_share_an_exclusive_key(self):
        a = self.mgr.submit("training", "first", self.python(
            "import time; time.sleep(30)"), exclusive_key="run:x")
        with self.assertRaises(J.JobConflict) as ctx:
            self.mgr.submit("training", "second", self.python("pass"),
                            exclusive_key="run:x")
        self.assertIn(a.id, str(ctx.exception))

    def test_different_keys_run_concurrently(self):
        self.mgr.submit("training", "x", self.python("import time; time.sleep(5)"),
                        exclusive_key="run:x")
        self.mgr.submit("training", "y", self.python("import time; time.sleep(5)"),
                        exclusive_key="run:y")
        self.assertEqual(len(self.mgr.active()), 2)

    def test_key_frees_up_once_the_job_ends(self):
        job = self.mgr.submit("training", "brief", self.python("pass"),
                              exclusive_key="run:z")
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))
        self.assertIsNone(self.mgr.active_with_key("run:z"))
        self.mgr.submit("training", "again", self.python("pass"),
                        exclusive_key="run:z")       # must not raise


class TestBookkeeping(JobManagerTest):
    def test_finished_jobs_are_trimmed_but_live_ones_are_not(self):
        mgr = J.JobManager(keep=3)
        try:
            live = mgr.submit("test", "live",
                              self.python("import time; time.sleep(20)"))
            for i in range(6):
                j = mgr.submit("test", f"quick{i}", self.python("pass"))
                wait_until(lambda j=j: j.state in J.TERMINAL_STATES)
                mgr._reap_once()
            self.assertIn(live.id, mgr.jobs, "a running job was trimmed away")
            self.assertLessEqual(len(mgr.jobs), 6)
        finally:
            mgr.stop_all(wait=8)
            mgr.shutdown()

    def test_progress_is_read_from_the_child_and_cached(self):
        import json
        job = self.mgr.submit("test", "writes progress", self.python("pass"))
        with open(job._progress_path, "w", encoding="utf-8") as f:
            json.dump({"done": 7, "total": 10, "phase": "working"}, f)
        self.assertEqual(job.progress()["done"], 7)
        # Corrupt file: keep the last good snapshot rather than crashing.
        time.sleep(0.01)
        with open(job._progress_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        os.utime(job._progress_path, (time.time() + 1, time.time() + 1))
        self.assertEqual(job.progress()["done"], 7)

    def test_events_are_emitted_for_state_changes(self):
        seen = []
        self.mgr.on_event = lambda level, msg, **f: seen.append((level, msg, f))
        job = self.mgr.submit("test", "quick", self.python("pass"))
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))
        self.assertTrue(any("started" in m for _, m, _ in seen))
        self.assertTrue(any("completed" in m for _, m, _ in seen))

    def test_a_broken_event_hook_cannot_break_job_control(self):
        def explode(*a, **k):
            raise RuntimeError("logging is down")
        self.mgr.on_event = explode
        job = self.mgr.submit("test", "quick", self.python("pass"))
        self.assertTrue(wait_until(lambda: job.state in J.TERMINAL_STATES))
        self.assertEqual(job.state, J.COMPLETED)


class TestPlatform(unittest.TestCase):
    def test_popen_flags_isolate_the_child_process_group(self):
        """Stopping one job must not signal the server or its siblings."""
        kwargs = J._popen_kwargs()
        if os.name == "nt":
            import subprocess
            self.assertEqual(kwargs.get("creationflags"),
                             subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.assertTrue(kwargs.get("start_new_session"))

    def test_interrupt_tolerates_a_dead_process(self):
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        J._interrupt(proc)             # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
