"""HTTP layer tests: the security boundary, over a real socket.

``test_api.py`` covers what the handlers decide; this module covers what the
transport allows through. The distinction matters, because the interesting
failures here are things a unit test on the handler cannot see: a cross-origin
POST, a traversal encoded in a URL, a response missing its hardening headers.

A real server is started on an ephemeral port so the requests are genuine HTTP
rather than a mocked handler.
"""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import server as S                             # noqa: E402
from dashboard import store                                   # noqa: E402
from dashboard.games import SESSIONS                          # noqa: E402
from training import checkpoint as CP                         # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerTest(unittest.TestCase):
    """Runs one ThreadingHTTPServer for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="2048srv-")
        cls._cp, cls._dr = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        cls._scp, cls._sdr = store.CHECKPOINT_ROOT, store.DATA_ROOT
        root = Path(cls.tmp)
        CP.CHECKPOINT_ROOT = store.CHECKPOINT_ROOT = root / "checkpoints"
        CP.DATA_ROOT = store.DATA_ROOT = root / "data"
        store.SETTINGS_PATH = CP.DATA_ROOT / "ui-settings.json"

        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", cls.port), S.Handler)
        cls.httpd.daemon_threads = True
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        SESSIONS.stop_all()
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = cls._cp, cls._dr
        store.CHECKPOINT_ROOT, store.DATA_ROOT = cls._scp, cls._sdr
        store.SETTINGS_PATH = Path(cls._sdr) / "ui-settings.json"
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- helpers -----------------------------------------------------------
    def get(self, path, timeout=10):
        req = urllib.request.Request(self.base + path)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def post(self, path, payload=None, headers=None, timeout=10):
        data = json.dumps(payload or {}).encode()
        hdrs = {"Content-Type": "application/json"}
        hdrs.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except json.JSONDecodeError:
                return e.code, {}

    def api_post(self, path, payload=None):
        """A POST the way the real front end sends it."""
        return self.post(path, payload, {"X-2048-Request": "1"})


class TestStatic(ServerTest):
    def test_the_app_shell_and_its_assets_are_served(self):
        for path in ("/", "/index.html", "/static/css/app.css",
                     "/static/js/core.js", "/static/js/charts.js",
                     "/static/js/ui.js", "/static/js/shell.js",
                     "/static/js/views/overview.js",
                     "/static/js/views/training.js",
                     "/static/js/views/play.js",
                     "/static/js/views/evaluate.js",
                     "/static/js/views/compare.js",
                     "/static/js/views/experiments.js",
                     "/static/js/views/checkpoints.js",
                     "/static/js/views/benchmarks.js",
                     "/static/js/views/misc.js"):
            status, body, _ = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertGreater(len(body), 20, path)

    def test_every_script_the_shell_references_exists(self):
        """A typo in index.html would break the app but not any other test."""
        import re
        _, body, _ = self.get("/")
        html = body.decode()
        for src in re.findall(r'<script src="([^"]+)"', html):
            status, _, _ = self.get(src)
            self.assertEqual(status, 200, f"{src} referenced but not served")
        for href in re.findall(r'<link rel="stylesheet" href="([^"]+)"', html):
            status, _, _ = self.get(href)
            self.assertEqual(status, 200, f"{href} referenced but not served")

    def test_traversal_out_of_the_static_directory_is_refused(self):
        for path in ("/static/../../train.py",
                     "/static/../server.py",
                     "/static/%2e%2e/%2e%2e/train.py",
                     "/static/..%2f..%2ftrain.py",
                     "/static/css/../../../../etc/passwd",
                     "/static/./../../LICENSE"):
            status, _, _ = self.get(path)
            self.assertEqual(status, 404, path)

    def test_responses_carry_the_hardening_headers(self):
        _, _, headers = self.get("/")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertIn("Content-Security-Policy", headers)
        csp = headers["Content-Security-Policy"]
        self.assertIn("default-src 'self'", csp)
        # No CDN escape hatch: scripts may only come from this origin.
        self.assertIn("script-src 'self'", csp)

    def test_no_cors_headers_are_ever_sent(self):
        """CORS would let any website call this control API."""
        for path in ("/", "/api/status", "/api/system"):
            _, _, headers = self.get(path)
            self.assertNotIn("Access-Control-Allow-Origin", headers, path)


class TestApiOverHttp(ServerTest):
    def test_read_endpoints_return_json(self):
        for path in ("/api/status", "/api/runs", "/api/agents", "/api/system",
                     "/api/checkpoints", "/api/jobs", "/api/logs",
                     "/api/settings", "/api/experiments"):
            status, body, headers = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn("application/json", headers.get("Content-Type", ""))
            json.loads(body)

    def test_unknown_api_route_is_404_json(self):
        status, body, _ = self.get("/api/not-a-thing")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))

    def test_errors_come_back_as_json_not_html(self):
        status, payload = self.api_post("/api/training/start",
                                        {"run": "../evil"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)
        self.assertIn("run names", payload["error"])


class TestRequestGuards(ServerTest):
    """The three things standing between a stray web page and your training."""

    def test_post_without_the_custom_header_is_refused(self):
        status, payload = self.post("/api/training/stop", {})
        self.assertEqual(status, 403)
        self.assertIn("X-2048-Request", payload["error"])

    def test_post_from_a_foreign_origin_is_refused(self):
        status, _ = self.post("/api/training/stop", {},
                              {"X-2048-Request": "1",
                               "Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_post_from_a_loopback_origin_is_allowed_through(self):
        status, payload = self.post(
            "/api/training/stop", {},
            {"X-2048-Request": "1", "Origin": f"http://127.0.0.1:{self.port}"})
        # Reaches the handler, which then reports there is nothing to stop.
        self.assertEqual(status, 409)
        self.assertIn("no training job", payload["error"])

    def test_null_origin_is_refused(self):
        status, _ = self.post("/api/training/stop", {},
                              {"X-2048-Request": "1", "Origin": "null"})
        self.assertEqual(status, 403)

    def test_malformed_json_is_a_400_not_a_crash(self):
        req = urllib.request.Request(
            self.base + "/api/settings", data=b"{not json",
            headers={"Content-Type": "application/json",
                     "X-2048-Request": "1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        self.assertEqual(code, 400)

    def test_a_json_array_body_is_refused(self):
        req = urllib.request.Request(
            self.base + "/api/settings", data=b"[1,2,3]",
            headers={"Content-Type": "application/json",
                     "X-2048-Request": "1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        self.assertEqual(code, 400)

    def test_oversized_bodies_are_rejected(self):
        big = json.dumps({"settings": {"theme": "x" * 2_000_000}}).encode()
        req = urllib.request.Request(
            self.base + "/api/settings", data=big,
            headers={"Content-Type": "application/json",
                     "X-2048-Request": "1"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                code = r.status
        except urllib.error.HTTPError as e:
            code = e.code
        except (urllib.error.URLError, OSError):
            code = 413          # server closed the connection on us; fine
        self.assertIn(code, (413, 400))

    def test_get_is_not_a_way_to_mutate(self):
        """Nothing that changes state may be reachable with a GET."""
        for path in ("/api/training/start", "/api/training/stop",
                     "/api/evaluate", "/api/benchmark",
                     "/api/checkpoints/delete", "/api/game/human/start"):
            status, _, _ = self.get(path)
            self.assertEqual(status, 404, f"{path} answered a GET")


class TestGameOverHttp(ServerTest):
    def test_a_human_game_can_be_played_over_the_api(self):
        status, payload = self.api_post("/api/game/human/start", {"seed": 7})
        self.assertEqual(status, 200)
        sid = payload["session"]["id"]
        try:
            status, state = self.api_post(f"/api/game/{sid}/move",
                                          {"direction": "left"})
            self.assertEqual(status, 200)
            self.assertIn("board", state)
            self.assertEqual(len(state["board"]), 4)

            status, _, _ = self.get(f"/api/game/{sid}")
            self.assertEqual(status, 200)
        finally:
            self.api_post(f"/api/game/{sid}/control", {"action": "stop"})

    def test_unknown_session_ids_are_404(self):
        status, _ = self.api_post("/api/game/not-a-session/move",
                                  {"direction": "left"})
        self.assertEqual(status, 404)

    def test_session_ids_cannot_smuggle_a_path(self):
        for bad in ("../../etc/passwd", "..%2f..%2fetc", "a/b/c"):
            status, _ = self.api_post(f"/api/game/{bad}/move",
                                      {"direction": "left"})
            self.assertIn(status, (404, 400), bad)


class TestStream(ServerTest):
    def test_the_event_stream_sends_status_frames(self):
        req = urllib.request.Request(self.base + "/api/stream")
        with urllib.request.urlopen(req, timeout=15) as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/event-stream", r.headers.get("Content-Type"))
            # An SSE event ends at a blank line; the status payload is larger
            # than any fixed-size read, so accumulate until one is complete.
            buf = b""
            deadline = time.time() + 12
            while b"\n\n" not in buf and time.time() < deadline:
                block = r.read(1024)
                if not block:
                    break
                buf += block
        chunk = buf.decode("utf-8", "replace")
        self.assertIn("event: status", chunk)
        event, _, _rest = chunk.partition("\n\n")
        data_line = [ln for ln in event.splitlines()
                     if ln.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        self.assertIn("state", payload)
        self.assertIn("games", payload)


class TestOriginHelper(unittest.TestCase):
    def test_loopback_origins_are_recognised(self):
        for good in ("http://127.0.0.1:8000", "http://localhost:8000",
                     "http://localhost", "https://127.0.0.1:9",
                     "http://[::1]:8000"):
            self.assertTrue(S._is_loopback_origin(good), good)

    def test_everything_else_is_not(self):
        for bad in ("https://evil.example", "http://127.0.0.1.evil.com",
                    "http://localhost.evil.com", "null", "", None,
                    "http://0.0.0.0:8000", "http://192.168.1.5:8000"):
            self.assertFalse(S._is_loopback_origin(bad), str(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
