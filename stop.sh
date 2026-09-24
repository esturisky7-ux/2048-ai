#!/usr/bin/env bash
# Stop the 2048 AI Control Center started with ./start.sh (or server.py).
#
#   ./stop.sh                   stop the one on the default port, 8000
#   ./stop.sh --port 8080       stop the one on another port
#
# The shutdown is the same as pressing Ctrl+C in its terminal: running
# training finishes the current game and writes a checkpoint, then the server
# exits. This waits until it has.
set -euo pipefail

cd "$(dirname "$0")"

find_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python python3.13 python3.12 \
                     python3.11 python3.10; do
        [ -n "$candidate" ] || continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
                >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! PY="$(find_python)"; then
    echo "error: Python 3.10 or newer is required, and none was found." >&2
    exit 1
fi

exec "$PY" server.py --stop "$@"
