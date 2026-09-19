"""The control center's HTTP transport.

A standard-library ``ThreadingHTTPServer`` that serves the single-page front
end, routes ``/api/*`` to :mod:`dashboard.api`, and streams live updates over
Server-Sent Events. No framework, no dependencies, nothing loaded from a CDN.

Because this server can start processes and delete files, its endpoints are
privileged local controls rather than a public read-only API. Three things
enforce that:

* it binds to **127.0.0.1** unless a host is passed explicitly, and warns
  loudly when it is not loopback;
* every mutating request must carry the ``X-2048-Request`` header, which a
  cross-origin page cannot set without a CORS preflight that is never granted
  — so a random website you visit cannot POST to your training server;
* any ``Origin`` header that is present must be a loopback origin.

Long work never happens in a handler. Training, evaluation, comparison,
benchmarking and experiments are all handed to the job manager, which runs
them as supervised subprocesses.
"""

from __future__ import annotations

import json
import os
import posixpath
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from version import __version__                              # noqa: E402

from . import api                                            # noqa: E402
from .api import (ApiError, downsample, eval_series,         # noqa: E402,F401
                  history_series, pid_alive, status_payload)
from .events import LOG                                      # noqa: E402
from .games import SESSIONS                                  # noqa: E402
from .jobs import MANAGER                                    # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Mutating requests must carry this header. Its only job is to be impossible
# to set from a cross-origin form or image tag.
REQUEST_HEADER = "X-2048-Request"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

# Server-Sent Events: how often a stream pushes a status frame, and how many
# streams may be open at once (one per browser tab is the normal case).
SSE_INTERVAL = 1.0
SSE_MAX_CLIENTS = 12
_sse_clients = threading.BoundedSemaphore(SSE_MAX_CLIENTS)
_shutdown = threading.Event()


def _is_loopback_origin(origin: str) -> bool:
    if not origin or origin == "null":
        return False
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


class Handler(BaseHTTPRequestHandler):
    server_version = f"2048ai/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- responses ---------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              cache: str = "no-store", extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        # Defence in depth for a local control surface.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=str).encode(),
                   "application/json; charset=utf-8")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def log_message(self, fmt, *args):
        if os.environ.get("DASHBOARD_VERBOSE"):
            super().log_message(fmt, *args)

    # -- guards ------------------------------------------------------------
    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and not _is_loopback_origin(origin):
            return False
        # A cross-site form post cannot set a custom header.
        return self.headers.get(REQUEST_HEADER) is not None

    # -- routing -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        query = parse_qs(u.query)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/api/stream":
                return self._stream(query)
            if path.startswith("/api/"):
                return self._json(api.handle_get(path, query))
            return self._error(404, f"no route {path}")
        except ApiError as e:
            return self._error(e.status, e.message)
        except BrokenPipeError:
            pass
        except Exception as e:
            LOG.add("error", f"GET {path} failed: {type(e).__name__}: {e}")
            return self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        path = u.path
        try:
            if not path.startswith("/api/"):
                return self._error(404, f"no route {path}")
            if not self._origin_ok():
                return self._error(
                    403, "this endpoint only accepts same-origin requests "
                         f"carrying the {REQUEST_HEADER} header")
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                return self._error(413, "request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._error(400, "invalid JSON body")
            if not isinstance(data, dict):
                return self._error(400, "body must be a JSON object")
            return self._json(api.handle_post(path, data))
        except ApiError as e:
            return self._error(e.status, e.message)
        except BrokenPipeError:
            pass
        except Exception as e:
            LOG.add("error", f"POST {path} failed: {type(e).__name__}: {e}")
            return self._error(500, f"{type(e).__name__}: {e}")

    # -- server-sent events ------------------------------------------------
    def _stream(self, query: dict):
        """Push a status frame every second until the client goes away.

        Cheaper and simpler than websockets, needs no dependency, and
        reconnects on its own in every browser that supports EventSource.
        """
        run = (query.get("run", ["default"]) or ["default"])[0]
        if not _sse_clients.acquire(blocking=False):
            return self._error(503, "too many live connections")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            last_seq = 0
            while not _shutdown.is_set():
                try:
                    payload = api.status_payload(run)
                except ApiError:
                    payload = {"error": "unknown run", "run": run}
                self._sse("status", payload)

                events = LOG.recent(20, since_seq=last_seq)
                if events:
                    last_seq = events[-1].get("seq", last_seq)
                    self._sse("events", {"events": events})

                if _shutdown.wait(SSE_INTERVAL):
                    break
            self._sse("bye", {"reason": "server shutting down"})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass          # the tab was closed; entirely normal
        finally:
            _sse_clients.release()

    def _sse(self, event: str, payload: dict) -> None:
        body = json.dumps(payload, default=str)
        self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode())
        self.wfile.flush()

    # -- static files ------------------------------------------------------
    def _static(self, rel: str):
        # URL paths are always '/'-separated, so they are split as POSIX and
        # rejoined with pathlib rather than pasted into a filesystem path.
        rel = posixpath.normpath("/" + rel).lstrip("/")
        base = STATIC_DIR.resolve()
        full = base.joinpath(*[part for part in rel.split("/") if part])
        try:
            resolved = full.resolve()
            resolved.relative_to(base)
        except (OSError, ValueError):
            return self._error(404, "not found")
        if not resolved.is_file():
            return self._error(404, "not found")
        ctype = {".html": "text/html; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8",
                 ".json": "application/json; charset=utf-8",
                 ".svg": "image/svg+xml",
                 ".png": "image/png",
                 ".webmanifest": "application/manifest+json",
                 }.get(resolved.suffix, "application/octet-stream")
        with open(resolved, "rb") as f:
            body = f.read()
        # The shell is revalidated every load; hashed assets do not exist here,
        # so everything is no-store to avoid serving a stale UI after an update.
        return self._send(200, body, ctype)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def _banner(host: str, port: int) -> None:
    import platform
    from training.checkpoint import list_runs

    runs = list_runs()
    if runs:
        best = max(runs, key=lambda r: r.get("games", 0))
        ckpt = (f"Loaded — run '{best['name']}', "
                f"{best.get('games', 0):,} games trained")
    else:
        ckpt = "None yet — the dashboard will offer to train one"
    training = "Stopped"
    for info in runs:
        from training.checkpoint import Run, read_json
        st = read_json(Run(info["name"]).status_path, {}) or {}
        if st.get("running") and pid_alive(st.get("pid", -1)) and \
                (time.time() - st.get("updated_at", 0)) < api.STALE_AFTER:
            training = f"Running — run '{info['name']}' (started elsewhere)"
            break

    lines = [
        "",
        "  2048 AI Control Center",
        f"  Dashboard:     http://{host}:{port}",
        f"  Platform:      {platform.system()} {platform.machine()}",
        f"  Python:        {platform.python_version()}",
        f"  AI checkpoint: {ckpt}",
        f"  Training:      {training}",
        "",
        "  Press Ctrl+C to stop the server.",
        "",
    ]
    # Flushed explicitly so the banner still appears when output is piped or
    # redirected, which is how supervisors and the test suite run it.
    print("\n".join(lines), flush=True)


def serve(host: str = "127.0.0.1", port: int = 8000,
          open_browser: bool = False, banner: bool = True):
    loopback = host in LOOPBACK_HOSTS
    if not loopback:
        print(f"WARNING: binding to {host}, which is not loopback.\n"
              f"         This control center can start processes and delete\n"
              f"         files, and it has no authentication. Only do this on\n"
              f"         a network you fully trust — an SSH tunnel is safer:\n"
              f"             ssh -L {port}:127.0.0.1:{port} you@this-machine",
              file=sys.stderr)

    # Give training the CPU; the dashboard is never the important workload.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass

    LOG.load()
    LOG.add("info", f"control center started on http://{host}:{port}",
            version=__version__)
    MANAGER.on_event = lambda level, message, **f: LOG.add(level, message, **f)
    MANAGER.start_reaper()

    # Windows delivers Ctrl-Break as SIGBREAK, whose default action kills the
    # process outright; route it to the same clean shutdown as Ctrl-C.
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        try:
            signal.signal(sigbreak, _raise_keyboard_interrupt)
        except (ValueError, OSError):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    if banner:
        _banner(host, port)
    else:
        print(f"2048 AI control center -> {url}")
    if open_browser:
        threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutting down...")
    finally:
        shutdown(httpd)


def shutdown(httpd=None) -> None:
    """Stop accepting work, let jobs save, then close.

    A training job is asked to stop the same way Ctrl-C asks it to: it finishes
    the current game and writes a checkpoint. The server waits for that rather
    than exiting and leaving the work unsaved.
    """
    _shutdown.set()
    live = MANAGER.active()
    if live:
        print(f"  waiting for {len(live)} job(s) to save and exit...")
        for job in live:
            print(f"    {job.type}: {job.label}")
    MANAGER.stop_all()
    MANAGER.shutdown()
    SESSIONS.stop_all()
    LOG.add("info", "control center stopped")
    if httpd is not None:
        try:
            httpd.shutdown()
        except Exception:
            pass
        httpd.server_close()
    print("  done. training data is saved.")
