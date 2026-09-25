"""Game sessions: agents playing, and humans playing, through the browser.

Both kinds of game run on **the Python engine**, never on a reimplementation in
JavaScript. That is a correctness decision, not a stylistic one: a second
implementation of 2048's merge rules would eventually disagree with the first,
and then "the AI scored more than you" would be measuring the difference
between two rule sets instead of two players. The browser draws boards and
sends key presses; every rule is applied by :mod:`engine.board`.

AI sessions run a throttled worker thread that stays a little ahead of what the
viewer has *displayed* -- the browser reports its playback cursor with every
poll -- so slow playback costs almost no CPU and training keeps the machine.
Human sessions are pure request/response and hold no thread at all. Sessions
nobody looks at any more are collected on a timer of their own, which stops
their thread and closes their agent.
"""

from __future__ import annotations

import itertools
import threading
import time
from random import Random

from engine import board as B
from engine.game import Game

# How far ahead of the viewer an AI worker may compute. Small enough that slow
# playback is nearly free, large enough to hide a slow agent's latency.
BUFFER_AHEAD = 48
MAX_FRAMES = 30000
# Sessions nobody has touched for this long are collected, so a browser tab
# closed mid-game does not leak a thread.
SESSION_TTL = 900.0
MAX_SESSIONS = 12
# How often the collector looks for such sessions.
JANITOR_INTERVAL = 30.0

_ids = itertools.count(1)


def _board_payload(b: int) -> dict:
    return {
        "board": B.board_to_rows(b),
        "max_tile": B.max_tile(b),
        "empty": B.empty_count(b),
    }


class AIGameSession:
    """One agent playing one game, streamed to the browser frame by frame."""

    kind = "ai"

    def __init__(self, agent, label: str, seed: int | None = None):
        self.id = f"g{next(_ids)}"
        self.agent = agent
        self.label = label
        self.seed = seed if seed is not None else \
            int(time.time() * 1000) % (1 << 30)
        self.rng = Random(self.seed)
        self.frames: list[dict] = []
        self.lock = threading.Lock()
        self.viewer_pos = 0
        self.done = False
        self.error: str | None = None
        self.created_at = time.time()
        self.touched_at = self.created_at
        self.paused = False
        self.decision_total = 0.0
        self.decision_count = 0
        self.last_decision_ms = 0.0
        self._step_once = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._close_lock = threading.Lock()

    # -- control -----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"ai-game-{self.id}")
        self._thread.start()

    def stop(self) -> None:
        """End the session: the thread exits after the move it is choosing,
        and the agent is closed (right away, if no thread is running)."""
        self._stop.set()
        self._step_once.set()
        if not self.is_alive():
            self._close_agent()

    def _close_agent(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if hasattr(self.agent, "close"):
            try:
                self.agent.close()
            except Exception:
                pass

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._step_once.set()

    def step(self) -> None:
        """Compute exactly one more move while paused."""
        self.paused = True
        self._step_once.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- worker ------------------------------------------------------------
    def _append(self, board: int, score: int, moves: int, action,
                decision_ms: float) -> None:
        with self.lock:
            if len(self.frames) < MAX_FRAMES:
                frame = _board_payload(board)
                frame.update({"i": len(self.frames), "score": score,
                              "moves": moves, "action": action,
                              "decision_ms": round(decision_ms, 3)})
                self.frames.append(frame)

    def _run(self) -> None:
        try:
            if hasattr(self.agent, "new_game"):
                self.agent.new_game()
            b = B.new_game(self.rng)
            score = moves = 0
            self._append(b, score, moves, None, 0.0)
            while not self._stop.is_set():
                if self.paused:
                    # Sleep until resumed or asked for a single step.
                    if not self._step_once.wait(0.2):
                        continue
                    self._step_once.clear()
                    if self._stop.is_set():
                        break
                else:
                    with self.lock:
                        ahead = len(self.frames) - self.viewer_pos
                    if ahead > BUFFER_AHEAD:
                        time.sleep(0.05)
                        continue
                if B.is_game_over(b):
                    break
                t0 = time.perf_counter()
                a = self.agent.act(b)
                decision_ms = (time.perf_counter() - t0) * 1000.0
                self.decision_total += decision_ms
                self.decision_count += 1
                self.last_decision_ms = decision_ms
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
                self._append(b, score, moves, B.ACTION_NAMES[a], decision_ms)
                if len(self.frames) >= MAX_FRAMES:
                    break
        except Exception as e:                      # keep the server alive
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done = True
            self._close_agent()

    # -- reading -----------------------------------------------------------
    def snapshot(self, since: int, limit: int = 160,
                 cursor: int | None = None) -> dict:
        """Frames from ``since`` on, and what the viewer has reached.

        ``cursor`` is the frame the viewer is *displaying*, which is what the
        worker stays ``BUFFER_AHEAD`` frames ahead of. A viewer that fetches
        frames faster than it plays them must say so, or slow playback would
        compute the whole game up front. Without a cursor, fetched frames
        count as displayed, as they did before cursors existed.
        """
        self.touched_at = time.time()
        with self.lock:
            self.viewer_pos = max(self.viewer_pos,
                                  since if cursor is None else cursor)
            total = len(self.frames)
            chunk = self.frames[since:since + limit]
            last = self.frames[-1] if self.frames else None
        mean_ms = (self.decision_total / self.decision_count
                   if self.decision_count else 0.0)
        return {
            "id": self.id,
            "kind": self.kind,
            "agent": self.label,
            "seed": self.seed,
            "frames": chunk,
            "total": total,
            "done": self.done,
            "paused": self.paused,
            "error": self.error,
            "alive": self.is_alive(),
            "score": last["score"] if last else 0,
            "moves": last["moves"] if last else 0,
            "max_tile": last["max_tile"] if last else 0,
            "mean_decision_ms": round(mean_ms, 3),
            "last_decision_ms": round(self.last_decision_ms, 3),
            "elapsed": time.time() - self.created_at,
        }


class HumanGameSession:
    """A game a person plays, with the rules applied on the Python side."""

    kind = "human"

    def __init__(self, seed: int | None = None):
        self.id = f"h{next(_ids)}"
        self.seed = seed if seed is not None else \
            int(time.time() * 1000) % (1 << 30)
        self.game = Game(seed=self.seed)
        self.created_at = time.time()
        self.touched_at = self.created_at
        self.ended_at: float | None = None
        self.lock = threading.Lock()

    def move(self, action: int) -> dict:
        """Apply one move. Illegal moves are a no-op, reported as such."""
        self.touched_at = time.time()
        with self.lock:
            if self.game.game_over:
                return self.state(moved=False)
            _, _, moved = self.game.step(action)
            if self.game.game_over and self.ended_at is None:
                self.ended_at = time.time()
            return self.state(moved=moved)

    def state(self, moved: bool | None = None) -> dict:
        self.touched_at = time.time()
        g = self.game
        payload = _board_payload(g.board)
        payload.update({
            "id": self.id,
            "kind": self.kind,
            "seed": self.seed,
            "score": g.score,
            "moves": g.moves,
            "game_over": g.game_over,
            "legal": [B.ACTION_NAMES[a] for a in g.legal_actions()],
            "elapsed": (self.ended_at or time.time()) - self.created_at,
        })
        if moved is not None:
            payload["moved"] = moved
        return payload

    def stop(self) -> None:
        """Present so the manager can treat both session kinds alike."""


class SessionManager:
    """Holds the live game sessions, with a cap and a time-to-live.

    Expired sessions are collected every ``janitor_interval`` seconds by a
    background thread, started with the first session, so an abandoned game
    -- a tab closed mid-game, or left paused -- is cleaned up even if no other
    game is ever started.
    """

    def __init__(self, janitor_interval: float | None = None):
        self.sessions: dict[str, object] = {}
        self.lock = threading.Lock()
        self.janitor_interval = janitor_interval
        self._janitor: threading.Thread | None = None
        self._janitor_stop = threading.Event()

    def _ensure_janitor(self) -> None:
        # One that was told to stop may not have exited yet; it has its own
        # stop event, so a new one can start beside it.
        if self._janitor is not None and self._janitor.is_alive() \
                and not self._janitor_stop.is_set():
            return
        self._janitor_stop = threading.Event()
        self._janitor = threading.Thread(
            target=self._janitor_loop, args=(self._janitor_stop,),
            daemon=True, name="game-janitor")
        self._janitor.start()

    def _janitor_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.janitor_interval or JANITOR_INTERVAL):
            try:
                self.collect()
            except Exception:
                pass          # a failed sweep must not end the sweeping

    def collect(self) -> list[str]:
        """Drop every session untouched for ``SESSION_TTL`` seconds."""
        with self.lock:
            return self._collect_locked()

    def add(self, session) -> object:
        with self.lock:
            self._ensure_janitor()
            self._collect_locked()
            if len(self.sessions) >= MAX_SESSIONS:
                # Drop the oldest finished session to make room; if they are
                # all live, refuse rather than starting unbounded threads.
                victim = None
                for sid, s in self.sessions.items():
                    if getattr(s, "done", False) or \
                            getattr(getattr(s, "game", None), "game_over",
                                    False):
                        victim = sid
                        break
                if victim is None:
                    raise RuntimeError(
                        "too many games running; stop one first")
                self._drop_locked(victim)
            self.sessions[session.id] = session
        return session

    def get(self, session_id: str):
        with self.lock:
            return self.sessions.get(session_id)

    def stop(self, session_id: str) -> bool:
        with self.lock:
            s = self.sessions.get(session_id)
            if s is None:
                return False
            try:
                s.stop()
            except Exception:
                pass
            return True

    def drop(self, session_id: str) -> None:
        with self.lock:
            self._drop_locked(session_id)

    def _drop_locked(self, session_id: str) -> None:
        s = self.sessions.pop(session_id, None)
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass

    def _collect_locked(self) -> list[str]:
        now = time.time()
        stale = [s for s, obj in self.sessions.items()
                 if now - getattr(obj, "touched_at", now) > SESSION_TTL]
        for sid in stale:
            self._drop_locked(sid)
        return stale

    def stop_all(self) -> None:
        with self.lock:
            self._janitor_stop.set()
            for sid in list(self.sessions):
                self._drop_locked(sid)

    def describe(self) -> list[dict]:
        with self.lock:
            out = []
            for s in self.sessions.values():
                out.append({
                    "id": s.id, "kind": s.kind,
                    "age": time.time() - s.created_at,
                    "done": bool(getattr(s, "done", False)
                                 or getattr(getattr(s, "game", None),
                                            "game_over", False)),
                })
            return out


SESSIONS = SessionManager()
