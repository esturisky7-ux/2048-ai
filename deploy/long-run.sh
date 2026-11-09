#!/usr/bin/env bash
# Train one run for as long as it takes, on a Linux machine or VM, in a way
# that is safe to start again after anything: a crash, a reboot, a preempted
# Spot VM, an OOM kill. It is one command that starts the run if it does not
# exist and continues it if it does, using every CPU this user may use.
#
#   ./deploy/long-run.sh                         run "main", until stopped
#   TARGET_GAMES=5000000 ./deploy/long-run.sh    stop by itself at 5M games
#   RUN=big WORKERS=6 ./deploy/long-run.sh       another name, six workers
#   ./deploy/long-run.sh --eval-every 100000     any train.py option follows
#
# Environment:
#   RUN           run name                                     (default main)
#   WORKERS       a number, or "auto" = one per usable CPU     (default auto)
#   TARGET_GAMES  total games for the whole run; unset = no end
#   AI2048_HOME   where checkpoints/ and data/ live (put them on a disk with
#                 room: the default 4x6 network is a 268 MB file, plus 268 MB
#                 for every snapshot)
#   PYTHON        a specific Python 3.10+
#
# Ctrl-C, SIGTERM and a closing terminal all stop it the same safe way: the
# games in flight finish, a checkpoint is written, then it exits. It exits 0
# for that and for reaching TARGET_GAMES, and non-zero if training failed, so
# a supervisor (see 2048-train.service) restarts it only when it should.
# See docs/LONG_RUNS.md.
set -euo pipefail

cd "$(dirname "$0")/.."

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

args=(--run "${RUN:-main}" --workers "${WORKERS:-auto}" --resume-if-exists)
if [ -n "${TARGET_GAMES:-}" ]; then
    args+=(--target-games "$TARGET_GAMES")
fi

# exec: Python replaces this shell, so a signal sent to this process (by
# systemd, by tmux, by kill) reaches the trainer itself and not a wrapper.
exec "$PY" train.py "${args[@]}" "$@"
