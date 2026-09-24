#!/usr/bin/env python3
"""Launch the 2048 AI Control Center.

Examples (write ``python`` or ``py`` instead of ``python3`` on Windows):

    python3 server.py               then open http://127.0.0.1:8000
    python3 server.py --open        start it and open a browser
    python3 server.py --port 8080   if something already uses port 8000
    python3 server.py --stop        stop the one that is running
    python3 server.py --restart     stop the one that is running, start again

On Linux and macOS, ``./start.sh`` and ``./stop.sh`` wrap these.

This is the main way to use the project: training, evaluation, experiments,
checkpoints and the game viewer are all driven from the browser. The
command-line tools (``train.py``, ``evaluate.py``, ``experiment.py``) remain
available for scripting and headless use.

Binds to localhost only unless --host is given explicitly. There is no
authentication, so do not expose it to a network you do not control; use an
SSH tunnel instead.
"""

import argparse
import errno
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.server import probe, serve, stop_running  # noqa: E402
from training.checkpoint import python_command            # noqa: E402
from version import version_string                        # noqa: E402


def main() -> int:
    py = python_command()
    p = argparse.ArgumentParser(
        prog=f"{py} server.py",
        description="Serve the 2048 AI Control Center.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.replace("python3 ", f"{py} "))
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default localhost; do not expose publicly)")
    p.add_argument("--port", type=int, default=8000,
                   help="TCP port to listen on (default 8000)")
    p.add_argument("--open", action="store_true", help="open a browser window")
    p.add_argument("--quiet", action="store_true",
                   help="one-line startup output instead of the banner")
    ctl = p.add_mutually_exclusive_group()
    ctl.add_argument("--stop", action="store_true",
                     help="stop the control center running on --port "
                          "(running training saves first), then exit")
    ctl.add_argument("--restart", action="store_true",
                     help="stop the control center running on --port, if "
                          "any, then start a fresh one")
    p.add_argument("--version", action="version", version=version_string())
    a = p.parse_args()
    url = f"http://{a.host}:{a.port}/"

    if a.stop or a.restart:
        if not stop_running(a.host, a.port):
            return 1
        if a.stop:
            return 0
    elif probe(a.host, a.port) == "control-center":
        # Starting twice is the most common mistake there is; answer it
        # instead of failing with "address already in use".
        print(f"The 2048 AI control center is already running at {url}\n"
              f"  Open that address in your browser. Run this command again\n"
              f"  with --restart to restart it, or with --stop to stop it.")
        if a.open:
            __import__("webbrowser").open(url)
        return 0

    try:
        serve(a.host, a.port, a.open, banner=not a.quiet)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES):
            why = ("Another program is using that port. Stop it, or"
                   if e.errno == errno.EADDRINUSE else
                   "This system does not allow listening on that port, so")
            print(f"error: cannot listen on {a.host}:{a.port} ({e.strerror}).\n"
                  f"       {why}\n"
                  f"       run the control center on a different port:\n"
                  f"           {py} server.py --port {a.port + 1}",
                  file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
