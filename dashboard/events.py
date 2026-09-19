"""A small event log: what the system did, and when.

The Logs page needs an answer to "what has been happening?" that does not
involve tailing four terminals. Events are appended to a JSONL file and kept in
a bounded in-memory ring so the API can answer without re-reading the file.

Deliberately *not* a general logging sink: nothing here captures environment
variables, command lines with credentials in them, or arbitrary tracebacks from
untrusted sources. Events carry a level, a message and a few explicit fields.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "events.jsonl"

LEVELS = ("debug", "info", "warn", "error")
# Kept in memory so /api/logs never has to read the whole file.
RING = 1000
# The file is trimmed when it passes this, so a long-lived server cannot fill
# a disk with its own log.
MAX_BYTES = 2_000_000


class EventLog:
    def __init__(self, path: Path = LOG_PATH, ring: int = RING):
        self.path = path
        self.events: deque = deque(maxlen=ring)
        self.lock = threading.Lock()
        self._seq = 0
        self._loaded = False

    def load(self) -> None:
        """Read the tail of the file so restarts keep recent history."""
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()[-self.events.maxlen:]
        except OSError:
            return
        with self.lock:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue          # tolerate a torn final line
                self._seq += 1
                ev["seq"] = self._seq
                self.events.append(ev)

    def add(self, level: str, message: str, **fields) -> dict:
        if level not in LEVELS:
            level = "info"
        ev = {"ts": time.time(), "level": level, "message": str(message)}
        for k, v in fields.items():
            if v is None:
                continue
            # Only simple values; no objects, no tracebacks by accident.
            if isinstance(v, (str, int, float, bool)):
                ev[k] = v
            else:
                ev[k] = str(v)
        with self.lock:
            self._seq += 1
            ev["seq"] = self._seq
            self.events.append(ev)
        self._append_file(ev)
        return ev

    def _append_file(self, ev: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                self._trim()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        except OSError:
            pass          # logging must never break the thing being logged

    def _trim(self) -> None:
        """Keep the most recent half of the file."""
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            keep = lines[len(lines) // 2:]
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(keep)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def recent(self, limit: int = 100, level: str | None = None,
               since_seq: int = 0) -> list[dict]:
        limit = max(1, min(int(limit), self.events.maxlen))
        with self.lock:
            rows = list(self.events)
        if since_seq:
            rows = [e for e in rows if e.get("seq", 0) > since_seq]
        if level and level in LEVELS:
            wanted = LEVELS[LEVELS.index(level):]
            rows = [e for e in rows if e.get("level") in wanted]
        return rows[-limit:]

    def clear(self) -> None:
        with self.lock:
            self.events.clear()
        try:
            self.path.unlink()
        except OSError:
            pass


LOG = EventLog()
