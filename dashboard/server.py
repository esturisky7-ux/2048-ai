"""Lightweight dashboard server: stdlib ``http.server``, no framework.

Binds to 127.0.0.1 by default. The bind address is only changed by an explicit
``--host`` flag, and passing anything other than a loopback address prints a
warning, because this server has no authentication and is not built to face a
network.

The server never talks to the training process. It reads the run's files
(``status.json``, ``history.jsonl``, ``evaluations.jsonl``) and, for live
games, opens the weights read-only and plays its own game in a throttled
background thread.
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

from agents.registry import make_agent, AGENT_NAMES            # noqa: E402
from training.checkpoint import Run, list_runs, read_json      # noqa: E402
from training.stats import AllTimeStats                        # noqa: E402
from dashboard.live import LiveGameManager                     # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MANAGER = LiveGameManager()

# A run's training process may die without clearing its status file; treat a
# status older than this as stale rather than reporting phantom training.
STALE_AFTER = 15.0   # trainer heartbeats every 3s


def pid_alive(pid: int) -> bool:
    """Is that process still around?

    ``os.kill(pid, 0)`` is the usual POSIX probe, but on Windows ``os.kill``
    calls ``TerminateProcess`` for any signal other than the two console
    events -- it would *kill* the trainer instead of asking after it. Windows
    therefore gets a read-only handle-open probe via ``ctypes`` instead, and
    if even that is unavailable the run is reported as alive and the staleness
    timer alone decides.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                     False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by somebody else
    except (OSError, TypeError):
        return False
    return True


def downsample(rows: list, target: int = 500) -> list:
    """Thin a history to about ``target`` points, always keeping the last one."""
    n = len(rows)
    if n <= target:
        return rows
    step = n / target
    out = [rows[int(i * step)] for i in range(target)]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def history_series(run: Run, limit: int = 500) -> dict:
    """Flatten history.jsonl into parallel arrays the charts can plot."""
    from training.stats import History
    rows = downsample(History(run.history_path).read(), limit)
    out = {"games": [], "mean_score": [], "median_score": [], "max_tile": [],
           "rate_512": [], "rate_1024": [], "rate_2048": [], "rate_4096": [],
           "games_per_second": [], "wall_time": [], "alpha": []}
    for r in rows:
        roll = r.get("rolling") or {}
        if not roll.get("games"):
            continue
        rates = roll.get("tile_rates", {})
        out["games"].append(r.get("games", 0))
        out["mean_score"].append(round(roll.get("mean_score", 0), 1))
        out["median_score"].append(round(roll.get("median_score", 0), 1))
        out["max_tile"].append(roll.get("max_tile", 0))
        out["rate_512"].append(round(rates.get("512", 0) * 100, 2))
        out["rate_1024"].append(round(rates.get("1024", 0) * 100, 2))
        out["rate_2048"].append(round(rates.get("2048", 0) * 100, 2))
        out["rate_4096"].append(round(rates.get("4096", 0) * 100, 2))
        out["games_per_second"].append(round(r.get("games_per_second", 0), 2))
        out["wall_time"].append(r.get("wall_time", 0))
        out["alpha"].append(r.get("alpha", 0))
    return out


def eval_series(run: Run, limit: int = 200) -> dict:
    rows = []
    if run.eval_path.exists():
        with open(run.eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows = rows[-limit:]
    return {
        "games_trained": [r.get("games_trained", 0) for r in rows],
        "mean_score": [round(r.get("mean_score", 0), 1) for r in rows],
        "ci_low": [round(r.get("ci95_mean", [0, 0])[0], 1) for r in rows],
        "ci_high": [round(r.get("ci95_mean", [0, 0])[1], 1) for r in rows],
        "highest_tile": [r.get("highest_tile", 0) for r in rows],
        "rate_2048": [round(r.get("tile_rates", {}).get("2048", {}).get("rate", 0) * 100, 2)
                      for r in rows],
    }


def status_payload(run_name: str) -> dict:
    run = Run(run_name)
    meta = run.load_meta() or {}
    status = read_json(run.status_path, {}) or {}

    running = bool(status.get("running"))
    updated = status.get("updated_at", 0)
    if running:
        fresh = (time.time() - updated) < STALE_AFTER
        if not (fresh and pid_alive(status.get("pid", -1))):
            running = False

    # meta stores the raw counters (dump()); the UI wants derived figures
    # (mean score, tile rates), so rehydrate and summarise.
    at_raw = meta.get("all_time") or {}
    at = AllTimeStats(at_raw).summary() if at_raw else {}

    # While training is live the status file is fresher than the checkpoint
    # (written every report interval rather than every checkpoint interval),
    # so prefer it for the fast-moving fields.
    games = meta.get("games", 0)
    last_eval = meta.get("last_eval")
    if running:
        games = max(games, status.get("games", 0))
        last_eval = status.get("last_eval") or last_eval
        live_at = status.get("all_time")
        if live_at and live_at.get("games", 0) >= at.get("games", 0):
            at = live_at
    return {
        "run": run_name,
        "exists": run.exists(),
        "running": running,
        "pid": status.get("pid") if running else None,
        "status_age": (time.time() - updated) if updated else None,
        "games": games,
        "tuple_set": meta.get("tuple_set"),
        "alpha": meta.get("alpha"),
        "games_per_second": status.get("games_per_second", 0.0),
        "moves_per_second": status.get("moves_per_second", 0.0),
        "session_seconds": status.get("session_seconds", 0.0),
        "session_games": status.get("session_games", 0),
        "total_train_seconds": at.get("train_seconds", 0.0),
        "mean_score": at.get("mean_score", 0.0),
        "checkpoint_saved_at": meta.get("saved_at"),
        "checkpoint_saved_iso": meta.get("saved_at_iso"),
        "checkpoint_path": os.path.relpath(run.meta_path, ROOT),
        "all_time": at,
        "rolling": (status.get("rolling") if running else None)
                   or meta.get("rolling", {}),
        "last_eval": last_eval,
        "disk_bytes": run.size_on_disk(),
        "runs": list_runs(),
        "server_time": time.time(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "2048lab/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str,
              cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        # Defence in depth for a local tool: no framing, no sniffing.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def log_message(self, fmt, *args):
        if os.environ.get("DASHBOARD_VERBOSE"):
            super().log_message(fmt, *args)

    # -- routing -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/status":
                return self._json(status_payload(q.get("run", ["default"])[0]))
            if path == "/api/history":
                run = Run(q.get("run", ["default"])[0])
                limit = int(q.get("limit", ["500"])[0])
                return self._json({"history": history_series(run, limit),
                                   "evaluations": eval_series(run)})
            if path == "/api/runs":
                return self._json({"runs": list_runs()})
            if path == "/api/watch/state":
                since = int(q.get("since", ["0"])[0])
                return self._json(MANAGER.snapshot(since))
            if path == "/api/agents":
                return self._json({"agents": list(AGENT_NAMES)})
            return self._error(404, f"no route {path}")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._error(400, "invalid JSON body")

            if u.path == "/api/watch/start":
                return self._start_watch(data)
            if u.path == "/api/watch/stop":
                MANAGER.stop()
                return self._json({"ok": True})
            return self._error(404, f"no route {u.path}")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._error(500, f"{type(e).__name__}: {e}")

    # -- actions -----------------------------------------------------------
    def _start_watch(self, data: dict):
        name = str(data.get("agent", "learned"))
        if name not in AGENT_NAMES:
            return self._error(400, f"unknown agent {name!r}")
        run_name = str(data.get("run", "default"))
        kwargs = {}
        label = name
        if name == "expectimax":
            depth = max(1, min(4, int(data.get("depth", 2))))
            kwargs.update(depth=depth, adaptive=bool(data.get("adaptive", False)),
                          prob_cutoff=float(data.get("prob_cutoff", 1e-3)))
            label = f"expectimax d{depth}"
        elif name == "learned":
            depth = max(1, min(3, int(data.get("depth", 1))))
            kwargs.update(run=run_name, depth=depth)
            label = "learned" if depth == 1 else f"learned+search d{depth}"
        try:
            agent = make_agent(name, seed=None, **kwargs)
        except FileNotFoundError as e:
            return self._error(400, str(e))
        seed = data.get("seed")
        game = MANAGER.start(agent, seed=int(seed) if seed is not None else None,
                             label=label)
        return self._json({"ok": True, "agent": label, "seed": game.seed})

    # -- static files ------------------------------------------------------
    def _static(self, rel: str):
        # Normalise and confine to STATIC_DIR: no traversal out of the folder.
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
                 ".svg": "image/svg+xml"}.get(resolved.suffix,
                                              "application/octet-stream")
        with open(resolved, "rb") as f:
            return self._send(200, f.read(), ctype)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False):
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback:
        print(f"WARNING: binding to {host}, which is not loopback.\n"
              f"         This dashboard has no authentication. Only do this on\n"
              f"         a network you trust.", file=sys.stderr)
    # Give the trainer the CPU; the dashboard is never the important workload.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass

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
    print(f"2048 AI dashboard -> {url}")
    print("press Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        MANAGER.stop()
        httpd.server_close()
