"""End-to-end: drive the control center the way the browser does.

The other new modules test pieces. This one starts ``server.py`` as a real
subprocess and performs the workflow a person actually performs — train, watch
it progress, stop it, resume, evaluate, watch a game, play a game, look at the
results — then restarts the server and checks the work is still there.

It is the slowest module in the suite and the one most worth having: every
other test could pass while the product as a whole was broken.
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
PY = sys.executable
WINDOWS = os.name == "nt"
RUN = "cc-e2e"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def popen_interruptible(args, **kw):
    """Start a child this process can later interrupt gracefully."""
    if WINDOWS:
        kw["creationflags"] = kw.get("creationflags", 0) | \
            subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([PY] + args, cwd=ROOT, **kw)


def interrupt(proc):
    if WINDOWS:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGINT)


def cleanup():
    for path in (os.path.join(ROOT, "checkpoints", RUN),
                 os.path.join(ROOT, "data", RUN)):
        shutil.rmtree(path, ignore_errors=True)


class ControlCenterTest(unittest.TestCase):
    """Ordered: each step builds on the last. The server runs for them all."""

    @classmethod
    def setUpClass(cls):
        cleanup()
        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.proc = cls.start_server()

    @classmethod
    def tearDownClass(cls):
        cls.stop_server(cls.proc)
        cleanup()

    @classmethod
    def start_server(cls):
        proc = popen_interruptible(
            ["server.py", "--port", str(cls.port), "--quiet"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 45
        while time.time() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"server exited early:\n{proc.stdout.read()}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/",
                                       timeout=2).read()
                return proc
            except Exception:
                time.sleep(0.4)
        proc.kill()
        raise AssertionError("the control center did not start")

    @classmethod
    def stop_server(cls, proc):
        """Interrupt the server and return everything it printed."""
        if proc.poll() is not None:
            try:
                return proc.stdout.read() if proc.stdout else ""
            except ValueError:
                return ""
        interrupt(proc)
        try:
            out, _ = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        return out or ""

    # -- request helpers ---------------------------------------------------
    def get(self, path, **params):
        url = self.base + path
        if params:
            from urllib.parse import urlencode
            url += "?" + urlencode(params)
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())

    def post(self, path, payload=None):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json",
                     "X-2048-Request": "1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def wait_for(self, predicate, timeout=120, interval=0.5, what="condition"):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = self.get("/api/status", run=RUN)
            if predicate(last):
                return last
            time.sleep(interval)
        self.fail(f"timed out waiting for {what}; last status: "
                  f"state={last and last.get('state')} "
                  f"games={last and last.get('games')}")

    # -- the workflow ------------------------------------------------------
    def test_01_the_app_shell_loads(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            html = r.read().decode()
        self.assertIn("2048 AI Control Center", html)
        self.assertIn("/static/js/core.js", html)

    def test_02_a_fresh_install_is_reported_honestly(self):
        status = self.get("/api/status", run=RUN)
        self.assertFalse(status["exists"])
        self.assertEqual(status["state"], "STOPPED")
        self.assertEqual(status["games"], 0)

    def test_03_start_training_from_the_browser(self):
        code, payload = self.post("/api/training/start", {
            "run": RUN, "games": 600, "workers": 1, "tuple_set": "8x4",
            "report_every": 100, "checkpoint_every": 200, "eval_every": 0,
        })
        self.assertEqual(code, 200, payload)
        self.assertEqual(payload["job"]["type"], "training")
        self.assertEqual(payload["job"]["state"], "RUNNING")
        self.__class__.train_job = payload["job"]["id"]

    def test_04_progress_updates_without_a_reload(self):
        status = self.wait_for(lambda s: s["games"] > 0, timeout=90,
                               what="the first games to be played")
        self.assertEqual(status["state"], "TRAINING")
        self.assertTrue(status["training"]["running"])
        rated = self.wait_for(lambda s: s["games_per_second"] > 0,
                              timeout=60, what="a throughput reading")
        self.assertGreater(rated["games_per_second"], 0)
        self.assertGreater(rated["moves_per_second"], 0)
        self.assertEqual(status["workers"], 1)
        self.assertEqual(status["target_games"], 600)
        self.assertIsNotNone(status["pid"])

        # The numbers must actually move, not just be present.
        first = status["games"]
        later = self.wait_for(lambda s: s["games"] > first, timeout=60,
                              what="the game counter to advance")
        self.assertGreater(later["games"], first)

    def test_05_a_second_job_on_the_same_run_is_refused(self):
        code, payload = self.post("/api/training/resume",
                                  {"run": RUN, "games": 100})
        self.assertEqual(code, 409)
        self.assertIn("already working on", payload["error"])

    def test_06_stopping_is_graceful_and_saves(self):
        before = self.get("/api/status", run=RUN)["games"]
        code, payload = self.post("/api/training/stop", {})
        self.assertEqual(code, 200, payload)
        self.assertIn("checkpoint", payload["message"])

        status = self.wait_for(lambda s: not s["training"]["running"],
                               timeout=120, what="training to stop")
        self.assertEqual(status["state"], "STOPPED")

        meta_path = os.path.join(ROOT, "checkpoints", RUN, "meta.json")
        self.assertTrue(os.path.exists(meta_path), "no checkpoint was written")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.assertGreaterEqual(meta["games"], before,
                                "the stopped run lost games")
        self.assertEqual(meta["tuple_set"], "8x4")
        self.__class__.games_after_stop = meta["games"]

        job = self.get(f"/api/jobs/{self.train_job}")
        self.assertEqual(job["state"], "CANCELLED")

    def test_07_resume_continues_from_the_checkpoint(self):
        before = self.games_after_stop
        code, payload = self.post("/api/training/resume",
                                  {"run": RUN, "games": 120, "workers": 1})
        self.assertEqual(code, 200, payload)
        status = self.wait_for(lambda s: s["games"] >= before + 120
                               or not s["training"]["running"],
                               timeout=180, what="the resumed run to finish")
        self.assertGreaterEqual(status["games"], before,
                                "resuming restarted from zero")
        self.wait_for(lambda s: not s["training"]["running"], timeout=120,
                      what="the resumed job to end")

    def test_08_evaluate_the_trained_agent(self):
        code, payload = self.post("/api/evaluate", {
            "agent": "learned", "run": RUN, "games": 25, "seed": 4242,
        })
        self.assertEqual(code, 200, payload)
        job_id = payload["job"]["id"]

        deadline = time.time() + 180
        job = None
        while time.time() < deadline:
            job = self.get(f"/api/jobs/{job_id}")
            if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.6)
        self.assertEqual(job["state"], "COMPLETED", job.get("error"))

        res = job["result"]
        self.assertEqual(res["games"], 25)
        self.assertEqual(res["seed"], 4242)
        self.assertGreater(res["mean_score"], 0)
        self.assertEqual(len(res["ci95_mean"]), 2)
        self.assertLess(res["ci95_mean"][0], res["mean_score"])
        self.assertGreater(res["ci95_mean"][1], res["mean_score"])
        self.assertIn("2048", res["tile_rates"])

        # It is recorded for the run, so the charts and history can show it.
        evals = self.get("/api/evaluations", run=RUN)["evaluations"]
        self.assertTrue(evals)
        self.assertEqual(evals[-1]["games"], 25)

    def test_09_watch_the_trained_agent_play(self):
        code, payload = self.post("/api/game/ai/start",
                                  {"agent": "learned", "run": RUN, "seed": 5})
        self.assertEqual(code, 200, payload)
        sid = payload["session"]["id"]
        try:
            deadline = time.time() + 45
            snap = None
            while time.time() < deadline:
                snap = self.get(f"/api/game/{sid}", since=0)
                if snap["total"] > 6:
                    break
                time.sleep(0.3)
            self.assertGreater(snap["total"], 6, "no frames were streamed")
            frame = snap["frames"][0]
            self.assertEqual(len(frame["board"]), 4)
            self.assertEqual(len(frame["board"][0]), 4)
            self.assertGreaterEqual(snap["score"], 0)
            self.assertGreater(snap["mean_decision_ms"], 0)

            code, _ = self.post(f"/api/game/{sid}/control", {"action": "pause"})
            self.assertEqual(code, 200)
            code, _ = self.post(f"/api/game/{sid}/control", {"action": "step"})
            self.assertEqual(code, 200)
            code, _ = self.post(f"/api/game/{sid}/control", {"action": "resume"})
            self.assertEqual(code, 200)
        finally:
            self.post(f"/api/game/{sid}/control", {"action": "stop"})

    def test_10_play_a_human_game(self):
        code, payload = self.post("/api/game/human/start", {"seed": 99})
        self.assertEqual(code, 200, payload)
        sid = payload["session"]["id"]
        try:
            moves = 0
            for direction in ["left", "up", "right", "down"] * 5:
                code, state = self.post(f"/api/game/{sid}/move",
                                        {"direction": direction})
                self.assertEqual(code, 200)
                if state.get("moved"):
                    moves += 1
                if state["game_over"]:
                    break
            self.assertGreater(moves, 0, "no move was ever legal")
            self.assertGreaterEqual(state["score"], 0)
            self.assertEqual(state["moves"], moves)
            # The engine, not the browser, decides what is legal.
            self.assertTrue(set(state["legal"]) <=
                            {"up", "down", "left", "right"})
        finally:
            self.post(f"/api/game/{sid}/control", {"action": "stop"})

    def test_11_statistics_and_history_are_available(self):
        status = self.get("/api/status", run=RUN)
        self.assertGreater(status["games"], 0)
        self.assertGreater(status["all_time"]["best_score"], 0)
        self.assertIn("tile_rates", status["all_time"])

        hist = self.get("/api/history", run=RUN)
        self.assertGreater(len(hist["history"]["games"]), 0)
        # Every series must be the same length or the charts would misalign.
        n = len(hist["history"]["games"])
        for key, values in hist["history"].items():
            self.assertEqual(len(values), n, f"series {key} is ragged")
        self.assertGreater(len(hist["evaluations"]["mean_score"]), 0)

    def test_12_checkpoints_are_listed_and_labelled(self):
        cps = self.get("/api/checkpoints")["checkpoints"]
        mine = [c for c in cps if c["run"] == RUN]
        self.assertTrue(mine, "the trained run is not listed")
        entry = mine[0]
        self.assertEqual(entry["kind"], "current")
        self.assertGreater(entry["games"], 0)
        self.assertGreater(entry["size_bytes"], 0)
        self.assertEqual(entry["tuple_set"], "8x4")
        self.assertIsNotNone(entry["eval"])

        code, _ = self.post("/api/checkpoints/label",
                            {"id": RUN, "label": "end to end"})
        self.assertEqual(code, 200)
        cps = self.get("/api/checkpoints")["checkpoints"]
        entry = [c for c in cps if c["id"] == RUN][0]
        self.assertEqual(entry["label"], "end to end")

    def test_13_run_a_benchmark(self):
        code, payload = self.post("/api/benchmark", {"scale": 0.05,
                                                     "run": RUN})
        self.assertEqual(code, 200, payload)
        job_id = payload["job"]["id"]
        deadline = time.time() + 240
        job = None
        while time.time() < deadline:
            job = self.get(f"/api/jobs/{job_id}")
            if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(1.0)
        self.assertEqual(job["state"], "COMPLETED", job.get("error"))

        res = job["result"]
        self.assertGreater(res["random_try_in_order"]["moves_per_second"], 0)
        self.assertGreater(len(res["primitives"]), 3)
        self.assertTrue(res["machine"]["platform"])
        self.assertTrue(any(a["agent"] == "random" and a["available"]
                            for a in res["agents"]))
        saved = self.get("/api/benchmarks")["benchmarks"]
        self.assertTrue(saved, "the benchmark result was not saved")

    def test_14_system_and_logs_report_real_values(self):
        sysinfo = self.get("/api/system")
        self.assertTrue(sysinfo["os"]["system"])
        self.assertGreater(sysinfo["cpu"]["count"], 0)
        self.assertGreater(sysinfo["server"]["pid"], 0)
        self.assertGreater(sysinfo["server"]["uptime_seconds"], 0)
        self.assertIn(sysinfo["server"]["worker_start_method"],
                      ("fork", "spawn", "forkserver", "unknown"))
        self.assertGreaterEqual(sysinfo["storage"]["checkpoints_bytes"], 0)

        events = self.get("/api/logs", limit=200)["events"]
        self.assertTrue(events)
        messages = " ".join(e["message"] for e in events)
        self.assertIn("training", messages)
        # Nothing sensitive is ever logged.
        for event in events:
            blob = json.dumps(event).lower()
            for secret in ("password", "token", "api_key", "secret",
                           "authorization"):
                self.assertNotIn(secret, blob)

    def test_15_state_survives_a_server_restart(self):
        before = self.get("/api/status", run=RUN)
        evals_before = len(self.get("/api/evaluations", run=RUN)["evaluations"])

        self.stop_server(self.proc)

        self.__class__.proc = self.start_server()

        after = self.get("/api/status", run=RUN)
        self.assertEqual(after["games"], before["games"])
        self.assertEqual(after["all_time"]["best_score"],
                         before["all_time"]["best_score"])
        self.assertEqual(after["tuple_set"], "8x4")
        self.assertEqual(
            len(self.get("/api/evaluations", run=RUN)["evaluations"]),
            evals_before)

        entry = [c for c in self.get("/api/checkpoints")["checkpoints"]
                 if c["id"] == RUN][0]
        self.assertEqual(entry["label"], "end to end")
        # Recent events were reloaded from disk, not lost.
        self.assertTrue(self.get("/api/logs", limit=50)["events"])

    def test_16_deleting_a_checkpoint_needs_confirmation_and_works(self):
        code, payload = self.post("/api/checkpoints/delete", {"id": RUN})
        self.assertEqual(code, 400)
        self.assertIn("confirm", payload["error"])
        self.assertTrue(os.path.exists(
            os.path.join(ROOT, "checkpoints", RUN)))

        code, payload = self.post("/api/checkpoints/delete",
                                  {"id": RUN, "confirm": True})
        self.assertEqual(code, 200, payload)
        self.assertFalse(os.path.exists(
            os.path.join(ROOT, "checkpoints", RUN)))
        self.assertFalse(os.path.exists(os.path.join(ROOT, "data", RUN)))

    def test_17_the_server_shuts_down_cleanly(self):
        out = self.stop_server(self.proc)
        self.assertIn("done", out.lower(),
                      f"the shutdown message never appeared:\n{out}")
        self.assertIn("saved", out.lower())
        self.assertEqual(self.proc.poll(), 0,
                         f"non-zero exit from the server:\n{out}")
        # Restart so tearDownClass has something well-defined to stop.
        self.__class__.proc = self.start_server()


if __name__ == "__main__":
    unittest.main(verbosity=2)
