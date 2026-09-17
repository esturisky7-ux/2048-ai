"""Training statistics: rolling window, all-time totals, and history on disk.

Three layers, because they answer different questions:

``RollingStats``
    A fixed-size deque of recent game results. Answers "is it improving right
    now?" -- the all-time average lags badly once a run is long.
``AllTimeStats``
    Cheap running aggregates plus records. Never needs the raw history.
``History``
    An append-only JSONL file, one row per reporting interval, so the
    dashboard can graph progress over the whole run and nothing is lost on a
    crash.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

# Tile milestones tracked as achievement rates.
MILESTONES = (512, 1024, 2048, 4096, 8192)


def _percentile(sorted_vals, q: float):
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class RollingStats:
    """Summary of the most recent ``window`` games."""

    def __init__(self, window: int = 1000):
        self.window = window
        self.scores: deque = deque(maxlen=window)
        self.moves: deque = deque(maxlen=window)
        self.max_tiles: deque = deque(maxlen=window)

    def add(self, score: int, moves: int, max_tile: int) -> None:
        self.scores.append(score)
        self.moves.append(moves)
        self.max_tiles.append(max_tile)

    def __len__(self) -> int:
        return len(self.scores)

    def summary(self) -> dict:
        n = len(self.scores)
        if n == 0:
            return {"games": 0}
        s = sorted(self.scores)
        rates = {}
        for m in MILESTONES:
            rates[str(m)] = sum(1 for t in self.max_tiles if t >= m) / n
        return {
            "games": n,
            "mean_score": sum(self.scores) / n,
            "median_score": _percentile(s, 0.5),
            "p10_score": _percentile(s, 0.10),
            "p90_score": _percentile(s, 0.90),
            "best_score": s[-1],
            "mean_moves": sum(self.moves) / n,
            "max_tile": max(self.max_tiles),
            "median_max_tile": sorted(self.max_tiles)[n // 2],
            "tile_rates": rates,
        }


class AllTimeStats:
    """Running totals and records for the whole training run."""

    def __init__(self, state: dict | None = None):
        self.games = 0
        self.moves = 0
        self.score_sum = 0
        self.best_score = 0
        self.best_tile = 0
        self.longest_game = 0
        self.best_score_game = 0       # game number the record was set on
        self.best_tile_game = 0
        self.tile_counts = {m: 0 for m in MILESTONES}
        self.train_seconds = 0.0
        if state:
            self.load(state)

    def add(self, score: int, moves: int, max_tile: int) -> None:
        self.games += 1
        self.moves += moves
        self.score_sum += score
        if score > self.best_score:
            self.best_score = score
            self.best_score_game = self.games
        if max_tile > self.best_tile:
            self.best_tile = max_tile
            self.best_tile_game = self.games
        if moves > self.longest_game:
            self.longest_game = moves
        for m in MILESTONES:
            if max_tile >= m:
                self.tile_counts[m] += 1

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.games if self.games else 0.0

    def summary(self) -> dict:
        g = self.games or 1
        return {
            "games": self.games,
            "moves": self.moves,
            "mean_score": self.score_sum / g if self.games else 0.0,
            "best_score": self.best_score,
            "best_score_game": self.best_score_game,
            "best_tile": self.best_tile,
            "best_tile_game": self.best_tile_game,
            "longest_game": self.longest_game,
            "train_seconds": self.train_seconds,
            "tile_rates": {str(m): (self.tile_counts[m] / g if self.games else 0.0)
                           for m in MILESTONES},
            "tile_counts": {str(m): self.tile_counts[m] for m in MILESTONES},
        }

    def dump(self) -> dict:
        return {
            "games": self.games, "moves": self.moves,
            "score_sum": self.score_sum, "best_score": self.best_score,
            "best_tile": self.best_tile, "longest_game": self.longest_game,
            "best_score_game": self.best_score_game,
            "best_tile_game": self.best_tile_game,
            "tile_counts": {str(k): v for k, v in self.tile_counts.items()},
            "train_seconds": self.train_seconds,
        }

    def load(self, d: dict) -> None:
        self.games = d.get("games", 0)
        self.moves = d.get("moves", 0)
        self.score_sum = d.get("score_sum", 0)
        self.best_score = d.get("best_score", 0)
        self.best_tile = d.get("best_tile", 0)
        self.longest_game = d.get("longest_game", 0)
        self.best_score_game = d.get("best_score_game", 0)
        self.best_tile_game = d.get("best_tile_game", 0)
        self.train_seconds = d.get("train_seconds", 0.0)
        tc = d.get("tile_counts", {})
        self.tile_counts = {m: tc.get(str(m), 0) for m in MILESTONES}


class History:
    """Append-only JSONL log of periodic training snapshots."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.resolve().parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        row = dict(row)
        row.setdefault("wall_time", time.time())
        # Append-and-flush: a crash loses at most the row being written.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()

    def read(self, limit: int | None = None) -> list:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue      # tolerate a torn final line after a crash
        return rows[-limit:] if limit else rows
