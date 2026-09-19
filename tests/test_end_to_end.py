"""End-to-end test: drive the real command-line entry points.

Everything else tests modules in isolation. This drives ``train.py``,
``evaluate.py``, ``experiment.py`` and ``server.py`` as a user would, in
subprocesses, and checks the things that actually matter across a restart:

* training writes a checkpoint and history
* stopping and resuming continues from the saved game count
* SIGINT (Ctrl-C) still saves before exiting
* evaluation runs against the checkpoint and is reproducible
* the dashboard serves its pages, reports the run, and streams a live game

It uses its own run name and tuple set and cleans up after itself, so it never
disturbs a real training run.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "e2e-test"
PY = sys.executable
WINDOWS = os.name == "nt"


def run(args, timeout=600, **kw):
    return subprocess.run([PY] + args, cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout, **kw)


def popen_interruptible(args, **kw):
    """Start a child that this process can later interrupt gracefully.

    POSIX can simply send SIGINT. Windows cannot: ``send_signal(SIGINT)``
    raises there, and the only interrupt deliverable to another process is a
    console control event, which needs the child in its own process group.
    """
    if WINDOWS:
        kw["creationflags"] = kw.get("creationflags", 0) | \
            subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([PY] + args, cwd=ROOT, **kw)


def interrupt(proc):
    """Ask ``proc`` to stop the way a user pressing Ctrl-C would."""
    if WINDOWS:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_json(url, payload, timeout=30):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 # The control center refuses mutating requests without it.
                 "X-2048-Request": "1"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())


def cleanup():
    shutil.rmtree(os.path.join(ROOT, "checkpoints", RUN), ignore_errors=True)
    shutil.rmtree(os.path.join(ROOT, "data", RUN), ignore_errors=True)
    # the experiment step writes a real result file; don't leave it behind
    for f in ("search-depth-1.json",):
        p = os.path.join(ROOT, "data", "experiments", f)
        if os.path.exists(p):
            os.remove(p)


class TestEndToEnd(unittest.TestCase):
    """Ordered: each step builds on the previous one's checkpoint."""

    @classmethod
    def setUpClass(cls):
        cleanup()
        cls.ck = os.path.join(ROOT, "checkpoints", RUN)
        cls.data = os.path.join(ROOT, "data", RUN)

    @classmethod
    def tearDownClass(cls):
        cleanup()

    def meta(self):
        with open(os.path.join(self.ck, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_01_train_creates_a_checkpoint(self):
        r = run(["train.py", "--run", RUN, "--tuple-set", "8x4",
                 "--games", "120", "--report-every", "60",
                 "--checkpoint-every", "60", "--eval-every", "0"])
        self.assertEqual(r.returncode, 0, r.stderr)
        for f in ("meta.json", "config.json", "weights.f32"):
            self.assertTrue(os.path.exists(os.path.join(self.ck, f)), f)
        m = self.meta()
        self.assertEqual(m["games"], 120)
        self.assertEqual(m["tuple_set"], "8x4")
        self.assertGreater(m["all_time"]["best_score"], 0)
        # history rows were written for graphing
        with open(os.path.join(self.data, "history.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("rolling", rows[-1])

    def test_02_resume_continues_from_the_checkpoint(self):
        before = self.meta()["games"]
        r = run(["train.py", "--run", RUN, "--resume", "--games", "60",
                 "--report-every", "60"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.meta()["games"], before + 60)
        self.assertIn(f"starting at game {before:,}", r.stdout)

    def test_03_resume_rejects_a_conflicting_tuple_set(self):
        r = run(["train.py", "--run", RUN, "--resume", "--games", "10",
                 "--tuple-set", "4x5"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("tuple set", (r.stderr + r.stdout))

    def test_04_resume_without_a_checkpoint_fails_cleanly(self):
        r = run(["train.py", "--run", "definitely-not-a-run", "--resume",
                 "--games", "10"])
        self.assertEqual(r.returncode, 1)
        self.assertIn("no checkpoint", r.stderr)

    def test_05_ctrl_c_saves_before_exiting(self):
        """SIGINT must checkpoint, not discard the session's work."""
        before = self.meta()["games"]
        p = popen_interruptible(
            ["train.py", "--run", RUN, "--resume", "--games", "0",
             "--report-every", "25", "--checkpoint-every", "100000"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                time.sleep(2)
                # wait until it has definitely played past the last checkpoint
                if os.path.exists(os.path.join(self.data, "history.jsonl")):
                    break
            time.sleep(6)
            interrupt(p)
            out, _ = p.communicate(timeout=120)
        finally:
            if p.poll() is None:
                p.kill()
                p.communicate()
        self.assertIn("stopping after the current game", out)
        self.assertIn("checkpoint saved", out)
        self.assertGreater(self.meta()["games"], before,
                           "interrupted session lost its games")

    def test_05b_two_workers_train_the_same_checkpoint(self):
        """--workers must add games to one shared weight file, on any OS.

        Run once per start method this platform supports, so the spawn path
        that Windows and macOS use is exercised even on Linux.
        """
        import multiprocessing as mp
        methods = [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()]
        self.assertTrue(methods)
        for method in methods:
            with self.subTest(start_method=method):
                before = self.meta()["games"]
                env = dict(os.environ, AI2048_START_METHOD=method)
                r = run(["train.py", "--run", RUN, "--resume", "--games", "40",
                         "--workers", "2", "--report-every", "40"],
                        timeout=900, env=env)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(self.meta()["games"], before + 40)
                self.assertGreater(
                    os.path.getsize(os.path.join(self.ck, "weights.f32")), 0)

    def test_06_evaluate_is_reproducible(self):
        def results_only(text):
            # Drop the wall-clock line: elapsed time and games/s legitimately
            # vary between runs, everything else must not.
            return [ln for ln in text.splitlines()
                    if not ln.lstrip().startswith("games  ")]

        a = run(["evaluate.py", "--agent", "learned", "--run", RUN,
                 "--games", "25", "--seed", "4242", "--no-save", "--quiet"])
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertIn("mean score", a.stdout)
        b = run(["evaluate.py", "--agent", "learned", "--run", RUN,
                 "--games", "25", "--seed", "4242", "--no-save", "--quiet"])
        self.assertEqual(results_only(a.stdout), results_only(b.stdout),
                         "same seed must replay identically")

        # a different seed must produce a different set of games
        c = run(["evaluate.py", "--agent", "learned", "--run", RUN,
                 "--games", "25", "--seed", "9999", "--no-save", "--quiet"])
        self.assertNotEqual(results_only(a.stdout), results_only(c.stdout))

    def test_07_evaluate_writes_json(self):
        out = os.path.join(ROOT, "data", RUN, "e2e.json")
        r = run(["evaluate.py", "--agent", "learned", "--run", RUN,
                 "--games", "20", "--seed", "7", "--out", out, "--no-save",
                 "--quiet"])
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        res = d["results"]["learned"]
        self.assertEqual(res["games"], 20)
        self.assertIn("ci95_mean", res)
        self.assertIn("tile_rates", res)

    def test_08_compare_agents_on_identical_games(self):
        r = run(["evaluate.py", "--compare", "random", "heuristic", "learned",
                 "--run", RUN, "--games", "12", "--seed", "5", "--quiet"])
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ("random", "heuristic", "learned"):
            self.assertIn(name, r.stdout)

    def test_09_experiment_runs_and_saves_its_config(self):
        r = run(["experiment.py", "--run", "search-depth-1",
                 "--eval-games", "4"], timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr)
        p = os.path.join(ROOT, "data", "experiments", "search-depth-1.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        self.assertIn("config", d)          # config stored next to the result
        self.assertEqual(d["config"]["agent"]["depth"], 1)
        self.assertEqual(d["evaluation"]["games"], 4)

    def test_10_control_center_serves_and_streams(self):
        """The web control center sees the run this module just trained.

        The CLI and the browser drive the same files, so a run created with
        ``train.py`` must be visible and playable from the server without any
        extra step.
        """
        port = free_port()
        p = popen_interruptible(["server.py", "--port", str(port), "--quiet"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(60):
                try:
                    urllib.request.urlopen(base + "/", timeout=2).read()
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                self.fail("the control center did not start")

            # the single-page app and its assets
            for path in ("/", "/static/css/app.css", "/static/js/core.js"):
                with urllib.request.urlopen(base + path, timeout=5) as r:
                    self.assertEqual(r.status, 200, path)

            # status reflects the run trained by the CLI earlier in this module
            st = get_json(f"{base}/api/status?run={RUN}")
            self.assertTrue(st["exists"])
            self.assertFalse(st["training"]["running"])   # no trainer alive
            self.assertGreater(st["games"], 0)
            self.assertEqual(st["tuple_set"], "8x4")

            # history is graphable
            h = get_json(f"{base}/api/history?run={RUN}")
            self.assertIn("history", h)
            self.assertGreater(len(h["history"]["games"]), 0)

            # and the checkpoint the CLI wrote is listed
            cps = get_json(f"{base}/api/checkpoints")["checkpoints"]
            self.assertTrue([c for c in cps if c["run"] == RUN])

            # a live game streams real frames from that checkpoint
            started = post_json(f"{base}/api/game/ai/start",
                                {"agent": "learned", "run": RUN, "depth": 1})
            self.assertTrue(started.get("ok"), started)
            sid = started["session"]["id"]
            frames = []
            for _ in range(40):
                snap = get_json(f"{base}/api/game/{sid}?since={len(frames)}")
                frames.extend(snap.get("frames", []))
                if len(frames) > 12 or snap.get("done"):
                    break
                time.sleep(0.25)
            self.assertGreater(len(frames), 5, "no frames streamed")
            f0 = frames[0]
            self.assertEqual(len(f0["board"]), 4)
            self.assertEqual(len(f0["board"][0]), 4)
            self.assertGreaterEqual(frames[-1]["score"], f0["score"])
            post_json(f"{base}/api/game/{sid}/control", {"action": "stop"})

            # bad input is rejected, not a 500
            self.assertIn("error", post_json(f"{base}/api/game/ai/start",
                                             {"agent": "nope"}))
        finally:
            interrupt(p)
            try:
                p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
