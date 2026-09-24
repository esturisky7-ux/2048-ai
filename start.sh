#!/usr/bin/env bash
# Start the 2048 AI Control Center -- the web UI and everything behind it --
# with one command. There is nothing else to start: one Python process serves
# the page, the API and the live updates, and launches training and
# evaluation itself when you ask for them in the browser.
#
#   ./start.sh                  start it and open http://127.0.0.1:8000
#   ./start.sh --port 8080      any server.py option is passed through
#   ./start.sh --restart        stop the copy that is running, start again
#   NO_BROWSER=1 ./start.sh     do not open a browser window
#   PYTHON=/path/to/python3 ./start.sh   use a specific Python
#
# Stop it with Ctrl+C here, or ./stop.sh from another terminal. Either way
# running training finishes its game and saves a checkpoint first.
set -euo pipefail

cd "$(dirname "$0")"

# The first Python 3.10+ we can find. PYTHON wins if it is set.
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
    echo "       Install it from https://www.python.org/downloads/ (or with" >&2
    echo "       your package manager), then run ./start.sh again." >&2
    exit 1
fi

# Open a browser unless asked not to, or there is no desktop to open it on
# (a text-mode browser would otherwise take over this terminal).
OPEN=(--open)
if [ -n "${NO_BROWSER:-}" ]; then
    OPEN=()
elif [ "$(uname -s)" = "Linux" ] && [ -z "${DISPLAY:-}" ] \
        && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    OPEN=()
fi

# exec: Python replaces this shell, so Ctrl+C, closing the window and
# ./stop.sh all reach the server directly and it can shut down cleanly.
exec "$PY" server.py ${OPEN[@]+"${OPEN[@]}"} "$@"
