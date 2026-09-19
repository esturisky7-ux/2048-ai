#!/usr/bin/env python3
"""Launch the 2048 AI Control Center.

Examples (write ``python`` or ``py`` instead of ``python3`` on Windows):

    python3 server.py               then open http://127.0.0.1:8000
    python3 server.py --open        start it and open a browser
    python3 server.py --port 8080   if something already uses port 8000

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

from dashboard.server import serve             # noqa: E402
from training.checkpoint import python_command  # noqa: E402
from version import version_string              # noqa: E402


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
    p.add_argument("--version", action="version", version=version_string())
    a = p.parse_args()
    try:
        serve(a.host, a.port, a.open, banner=not a.quiet)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES):
            print(f"error: cannot listen on {a.host}:{a.port} ({e.strerror}).\n"
                  f"       Something else is probably using that port; try:\n"
                  f"           {py} server.py --port {a.port + 1}",
                  file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
