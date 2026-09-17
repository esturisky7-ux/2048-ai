"""Live game playback for the dashboard.

The visualisation must never slow training down, so it does not ask the
trainer for anything. It plays its *own* game in a background thread, loading
the learned weights through a **read-only** memory map of the same file the
trainer writes. Reads of a file another process is writing are safe here: the
worst case is that a frame uses a weight mid-update, which changes nothing a
viewer could notice.

Throttling is what keeps it cheap. The thread only stays a small number of
frames ahead of what the browser has consumed, so at 1x speed it computes
about eight moves a second and the rest of the CPU stays with training. The
server process also lowers its own scheduling priority at startup.
"""

from __future__ import annotations

import threading
import time
from random import Random

from engine import board as B

# How far ahead of the viewer the worker is allowed to compute. Small enough
# that slow speeds cost almost nothing, large enough to hide search latency.
BUFFER_AHEAD = 48
MAX_FRAMES = 30000


class LiveGame:
    """One in-progress game being streamed to the browser."""

    def __init__(self, agent, seed: int | None = None, agent_label: str = ""):
        self.agent = agent
        self.agent_label = agent_label or getattr(agent, "name", "agent")
        self.seed = seed if seed is not None else int(time.time() * 1000) % (1 << 30)
        self.rng = Random(self.seed)
        self.frames: list = []
        self.lock = threading.Lock()
        self.viewer_pos = 0
        self.done = False
        self.error: str | None = None
        self.started_at = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- control -----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- worker ------------------------------------------------------------
    def _append(self, board: int, score: int, moves: int, action) -> None:
        with self.lock:
            if len(self.frames) < MAX_FRAMES:
                self.frames.append({
                    "i": len(self.frames),
                    "board": B.board_to_rows(board),
                    "score": score,
                    "moves": moves,
                    "max_tile": B.max_tile(board),
                    "empty": B.empty_count(board),
                    "action": action,
                })

    def _run(self) -> None:
        try:
            if hasattr(self.agent, "new_game"):
                self.agent.new_game()
            b = B.new_game(self.rng)
            score = moves = 0
            self._append(b, score, moves, None)
            while not self._stop.is_set():
                # Stay only a little ahead of the viewer: this is the whole
                # throttle, and it is why slow playback costs no CPU.
                with self.lock:
                    ahead = len(self.frames) - self.viewer_pos
                if ahead > BUFFER_AHEAD:
                    time.sleep(0.05)
                    continue
                if B.is_game_over(b):
                    break
                a = self.agent.act(b)
                nb, gained, moved = B.move(b, a)
                if not moved:
                    legal = B.legal_actions(b)
                    if not legal:
                        break
                    a = legal[0]
                    nb, gained, moved = B.move(b, a)
                b = B.random_spawn(nb, self.rng)
                score += gained
                moves += 1
                self._append(b, score, moves, B.ACTION_NAMES[a])
                if len(self.frames) >= MAX_FRAMES:
                    break
        except Exception as e:                      # keep the server alive
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done = True
            if hasattr(self.agent, "close"):
                try:
                    self.agent.close()
                except Exception:
                    pass

    # -- reading -----------------------------------------------------------
    def snapshot(self, since: int, limit: int = 120) -> dict:
        with self.lock:
            self.viewer_pos = max(self.viewer_pos, since)
            total = len(self.frames)
            chunk = self.frames[since:since + limit]
            last = self.frames[-1] if self.frames else None
        return {
            "agent": self.agent_label,
            "seed": self.seed,
            "frames": chunk,
            "total": total,
            "done": self.done,
            "error": self.error,
            "final_score": last["score"] if last else 0,
            "final_max_tile": last["max_tile"] if last else 0,
            "alive": self.is_alive(),
        }


class LiveGameManager:
    """Holds the one live game the dashboard is showing."""

    def __init__(self):
        self.current: LiveGame | None = None
        self.lock = threading.Lock()

    def start(self, agent, seed=None, label: str = "") -> LiveGame:
        with self.lock:
            if self.current is not None:
                self.current.stop()
            game = LiveGame(agent, seed=seed, agent_label=label)
            self.current = game
        game.start()
        return game

    def stop(self) -> None:
        with self.lock:
            if self.current is not None:
                self.current.stop()

    def snapshot(self, since: int) -> dict:
        with self.lock:
            game = self.current
        if game is None:
            return {"frames": [], "total": 0, "done": True, "idle": True}
        out = game.snapshot(since)
        out["idle"] = False
        return out
