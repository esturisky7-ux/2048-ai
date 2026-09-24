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
# POST here (with the header above) to stop the server gracefully; this is
# what ``server.py --stop`` and ``--restart`` use.
SHUTDOWN_PATH = "/api/shutdown"

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
            if path == SHUTDOWN_PATH:
                return self._shutdown_requested()
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

    def _shutdown_requested(self):
        """``server.py --stop``: shut down exactly as Ctrl-C would.

        The reply goes out first; ``serve_forever`` then returns in the main
        thread, whose ``finally`` runs the usual graceful :func:`shutdown`.
        """
        self._json({"ok": True, "pid": os.getpid(), "message":
                    "shutting down; running jobs save before they exit"})
        LOG.add("info", "shutdown requested over HTTP")
        threading.Thread(target=self.server.shutdown, daemon=True,
                         name="shutdown-request").start()

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


# Signals that must stop the server as cleanly as Ctrl-C does: closing the
# terminal window (SIGHUP), ``kill``, an editor's stop button or logging out
# (SIGTERM), and Ctrl-Break on Windows (SIGBREAK). Their default action ends
# the process on the spot and skips shutdown(), and a training job -- which
# runs in its own session so it can be stopped on its own -- would then carry
# on training with no server left to stop it.
_STOP_SIGNALS = ("SIGTERM", "SIGHUP", "SIGBREAK")


def _ignore_signal(signum, frame):
    pass


def _raise_keyboard_interrupt(signum, frame):
    # Only the first one counts: a second hang-up must not cut short the wait
    # for training to save. (A Python-level no-op rather than SIG_IGN, which
    # would be inherited by any child started afterwards.)
    for name in _STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _ignore_signal)
            except (ValueError, OSError):
                pass
    raise KeyboardInterrupt


def _say(text: str = "", err: bool = False) -> None:
    """``print`` that survives a terminal which has gone away.

    Once the window is closed, writing to it raises; shutdown has to carry on
    regardless, because that is exactly when training is saving.
    """
    try:
        print(text, file=sys.stderr if err else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


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

    # Bind first, so a port that is already taken fails before anything is
    # started or logged -- not after an "started" entry in the event log.
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

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

    for name in _STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _raise_keyboard_interrupt)
            except (ValueError, OSError):
                pass      # not the main thread (tests), or not allowed here

    url = f"http://{host}:{port}/"
    if banner:
        _banner(host, port)
    else:
        print(f"2048 AI control center -> {url}")
    if open_browser:
        threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        httpd.serve_forever()
        _say("\n  shutting down (asked to by --stop or --restart)...")
    except KeyboardInterrupt:
        _say("\n  shutting down...")
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
        _say(f"  waiting for {len(live)} job(s) to save and exit...")
        for job in live:
            _say(f"    {job.type}: {job.label}")
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
    _say("  done. training data is saved.")


# ---------------------------------------------------------------------------
# A control center that is already running: the start-up check, --stop and
# --restart. http.client is used directly, never urllib, so an HTTP proxy set
# in the environment is not consulted for what is always a loopback address.
# ---------------------------------------------------------------------------
def _client_host(host: str) -> str:
    """Where to connect to reach a server that was bound to ``host``."""
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "::1"
    return host.strip("[]")


# A listener on this machine completes a connection at once, so a connection
# that fails *for any reason* within this long means nobody is listening.
# (Windows takes up to ~2 s to report a refused loopback connection, and a
# timeout must not be mistaken for "something is there".)
_CONNECT_TIMEOUT = 1.0


def probe(host: str, port: int, timeout: float = 5.0) -> str | None:
    """What is listening on ``host:port``?

    ``"control-center"`` if it is one of these servers, ``"other"`` if it is
    some other program, and ``None`` if nothing is listening at all.
    """
    import http.client
    conn = http.client.HTTPConnection(_client_host(host), port,
                                      timeout=_CONNECT_TIMEOUT)
    try:
        try:
            conn.connect()
        except OSError:
            return None
        conn.sock.settimeout(timeout)
        conn.request("HEAD", "/")
        resp = conn.getresponse()
        server = resp.getheader("Server") or ""
    except (OSError, http.client.HTTPException):
        return "other"       # it accepted the connection but did not answer
    finally:
        conn.close()
    return "control-center" if server.startswith("2048ai/") else "other"


def _port_closed(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((_client_host(host), port),
                                      timeout=_CONNECT_TIMEOUT):
            return False
    except OSError:
        return True


def stop_running(host: str, port: int, wait: float | None = None) -> bool:
    """Ask the control center on ``host:port`` to shut down, and wait for it.

    It shuts down exactly as it does on Ctrl-C, so running training finishes
    its game and writes a checkpoint first. Returns True once nothing listens
    on the port any more (including when nothing was running), False if the
    port belongs to some other program or the server did not stop in time.
    """
    import http.client
    from .jobs import GRACE_SECONDS, TERMINATE_SECONDS
    if wait is None:
        wait = GRACE_SECONDS + TERMINATE_SECONDS + 20

    state = probe(host, port)
    if state is None:
        _say(f"  no control center is running on port {port}.")
        return True
    if state == "other":
        _say(f"  port {port} is used by another program, not the 2048 AI "
             f"control center; leaving it alone.", err=True)
        return False

    conn = http.client.HTTPConnection(_client_host(host), port, timeout=10)
    try:
        conn.request("POST", SHUTDOWN_PATH, body=b"{}",
                     headers={REQUEST_HEADER: "1",
                              "Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        status = resp.status
    except (OSError, http.client.HTTPException) as e:
        _say(f"  could not ask the control center to stop: {e}", err=True)
        return False
    finally:
        conn.close()
    if status != 200:
        # A copy started before --stop existed has no shutdown route.
        _say(f"  the control center on port {port} did not accept the stop "
             f"request (HTTP {status}).\n"
             f"  Press Ctrl+C in the terminal it is running in instead.",
             err=True)
        return False

    _say(f"  stopping the control center on port {port}; "
         f"running jobs save first...")
    started = time.time()
    next_note = started + 10
    while time.time() - started < wait:
        if _port_closed(host, port):
            _say("  stopped.")
            return True
        if time.time() >= next_note:
            _say("  still waiting for running jobs to save...")
            next_note += 10
        time.sleep(0.25)
    _say(f"  it is still running after {wait:.0f} seconds.", err=True)
    return False
