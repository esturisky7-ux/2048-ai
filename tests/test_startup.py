"""Starting, stopping and restarting the control center.

Every other test stops the server with Ctrl-C. People do not: they close the
terminal window, press an editor's stop button, run the start command twice,
or come back to a trainer that outlived an earlier server. Each of those used
to leave something behind -- a training process nobody could stop from the
browser, a second trainer writing the same weights, or an "address already in
use" error -- and this module keeps them fixed.
"""

import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WINDOWS = os.name == "nt"
RUNS = ("startup-sigterm", "startup-sighup", "startup-external")

# Loopback only, so never through a proxy that happens to be configured.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cleanup():
    for run in RUNS:
        for path in (os.path.join(ROOT, "checkpoints", run),
                     os.path.join(ROOT, "data", run)):
            shutil.rmtree(path, ignore_errors=True)


def pid_gone(pid, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.2)
    return False


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def status_of(run):
    return read_json(os.path.join(ROOT, "data", run, "status.json"))


def wait_for_games(run, timeout=60.0):
    """Wait until the run's trainer is past start-up and playing games."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = status_of(run)
        if st.get("running") and st.get("games", 0) > 0:
            return st
        time.sleep(0.2)
    raise AssertionError(f"training for {run!r} never got going")


class ServerProcess:
    """``server.py`` on a free port, as a subprocess."""

    def __init__(self, port):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [PY, "server.py", "--port", str(port), "--quiet"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 45
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"server exited early:\n{self.proc.stdout.read()}")
            try:
                OPENER.open(self.base + "/", timeout=2).read()
                return
            except Exception:
                time.sleep(0.3)
        self.proc.kill()
        raise AssertionError("the control center did not start")

    def post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "X-2048-Request": "1"}, method="POST")
        try:
            with OPENER.open(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, timeout=120):
        try:
            out, _ = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, _ = self.proc.communicate()
            raise AssertionError(f"server did not exit:\n{out}")
        return out

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.communicate()


def cli(*args, timeout=150):
    return subprocess.run([PY, "server.py", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


class StartStopRestartTest(unittest.TestCase):
    """--stop, --restart and starting twice, on every platform."""

    def setUp(self):
        self.port = free_port()
        self.server = ServerProcess(self.port)

    def tearDown(self):
        self.server.kill()

    def test_starting_twice_points_at_the_running_one(self):
        r = cli("--port", str(self.port))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("already running", r.stdout)
        self.assertIn(f"http://127.0.0.1:{self.port}/", r.stdout)
        self.assertIsNone(self.server.proc.poll(), "the first one must live on")

    def test_stop_then_stop_again(self):
        r = cli("--port", str(self.port), "--stop")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("stopped", r.stdout)
        out = self.server.wait()
        self.assertEqual(self.server.proc.returncode, 0, out)
        self.assertIn("training data is saved", out)

        r = cli("--port", str(self.port), "--stop")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("no control center is running", r.stdout)

    def test_restart_replaces_the_running_one(self):
        new = subprocess.Popen(
            [PY, "server.py", "--port", str(self.port), "--quiet",
             "--restart"], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        try:
            self.server.wait()
            deadline = time.time() + 45
            while time.time() < deadline:
                try:
                    OPENER.open(self.server.base + "/", timeout=2).read()
                    break
                except Exception:
                    if new.poll() is not None:
                        self.fail(f"--restart exited:\n{new.stdout.read()}")
                    time.sleep(0.3)
            else:
                self.fail("the restarted control center never came up")
        finally:
            r = cli("--port", str(self.port), "--stop")
            try:
                new.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                new.kill()
                new.communicate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_shutdown_route_is_not_reachable_cross_site(self):
        req = urllib.request.Request(
            self.server.base + "/api/shutdown", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            OPENER.open(req, timeout=10)
        self.assertEqual(cm.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            OPENER.open(self.server.base + "/api/shutdown", timeout=10)
        self.assertEqual(cm.exception.code, 404)
        self.assertIsNone(self.server.proc.poll())


class OtherProgramOnPortTest(unittest.TestCase):
    """A port held by something else is reported, and never touched."""

    def test_other_program(self):
        class Quiet(http.server.BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.send_response(200)
                self.end_headers()

            do_GET = do_HEAD

            def log_message(self, *a):
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Quiet)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            r = cli("--port", str(port), "--stop")
            self.assertEqual(r.returncode, 1)
            self.assertIn("another program", r.stderr)

            # Windows lets a second SO_REUSEADDR socket share the port, so
            # there the bind itself would succeed.
            if not WINDOWS:
                r = cli("--port", str(port), "--quiet")
                self.assertEqual(r.returncode, 1)
                self.assertIn("Another program is using that port", r.stderr)
        finally:
            httpd.shutdown()
            httpd.server_close()


@unittest.skipIf(WINDOWS, "SIGTERM and SIGHUP are POSIX signals")
class SignalShutdownTest(unittest.TestCase):
    """Closing the window or killing the server still saves training."""

    @classmethod
    def setUpClass(cls):
        cleanup()

    @classmethod
    def tearDownClass(cls):
        cleanup()

    def check(self, sig, run):
        server = ServerProcess(free_port())
        try:
            code, body = server.post("/api/training/start", {
                "run": run, "tuple_set": "8x4", "games": 0,
                "checkpoint_every": 100000})
            self.assertEqual(code, 200, body)
            trainer = wait_for_games(run)["pid"]

            server.proc.send_signal(sig)
            out = server.wait()
        finally:
            server.kill()
        self.assertEqual(server.proc.returncode, 0, out)
        self.assertIn("training data is saved", out)
        self.assertTrue(pid_gone(trainer), "the trainer was left running")
        self.assertFalse(status_of(run).get("running"))
        meta = read_json(os.path.join(ROOT, "checkpoints", run, "meta.json"))
        self.assertGreater(meta.get("games", 0), 0,
                           "no checkpoint was written on the way out")

    def test_sigterm(self):
        self.check(signal.SIGTERM, "startup-sigterm")

    def test_sighup(self):
        self.check(signal.SIGHUP, "startup-sighup")


@unittest.skipIf(WINDOWS, "a console's Ctrl-C cannot be sent from outside it")
class ExternalTrainerTest(unittest.TestCase):
    """A trainer this server did not start: stoppable, and not doubled."""

    RUN = "startup-external"

    def setUp(self):
        cleanup()
        self.trainer = subprocess.Popen(
            [PY, "train.py", "--run", self.RUN, "--tuple-set", "8x4",
             "--games", "0", "--checkpoint-every", "100"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.kill()
        if self.trainer.poll() is None:
            self.trainer.kill()
            self.trainer.wait()
        cleanup()

    def test_refuses_a_second_trainer_and_can_stop_it(self):
        wait_for_games(self.RUN)
        meta_path = os.path.join(ROOT, "checkpoints", self.RUN, "meta.json")
        deadline = time.time() + 60
        while not os.path.exists(meta_path) and time.time() < deadline:
            time.sleep(0.2)

        self.server = ServerProcess(free_port())
        code, body = self.server.post("/api/training/resume",
                                      {"run": self.RUN, "resume": True})
        self.assertEqual(code, 409, body)
        self.assertIn("already being trained by another process", body["error"])

        code, body = self.server.post("/api/training/stop", {"run": self.RUN})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["pid"], self.trainer.pid)
        self.trainer.wait(timeout=60)
        self.assertFalse(status_of(self.RUN).get("running"))

        # Without a run to look at, the old answer stands.
        code, body = self.server.post("/api/training/stop", {})
        self.assertEqual(code, 409, body)


if __name__ == "__main__":
    unittest.main()
