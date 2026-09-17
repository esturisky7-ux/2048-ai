"""Run directories, checkpointing and resume.

A *run* is one training experiment. Everything belonging to it lives under two
directories named after it:

    checkpoints/<run>/weights.f32     memory-mapped float32 weights
    checkpoints/<run>/meta.json       games played, stats, RNG state, config
    checkpoints/<run>/config.json     the configuration the run started with
    checkpoints/<run>/snapshots/      optional frozen copies of the weights
    data/<run>/history.jsonl          periodic training snapshots (for graphs)
    data/<run>/evaluations.jsonl      fixed-procedure evaluation results

Crash safety has two halves. The weights are a memory-mapped file, so the
kernel is already writing them back as training proceeds; a checkpoint only
has to ``flush()`` the dirty pages. The metadata is small and is written with
the write-temp-then-``os.replace`` dance, which is atomic on Linux, so
``meta.json`` is never observed half-written. Worst case after a hard power
loss you resume from the last completed checkpoint interval.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

# Paths are derived from this file's location with pathlib, so they carry the
# right separator on every OS and nothing depends on the working directory.
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = ROOT / "checkpoints"
DATA_ROOT = ROOT / "data"


def python_command() -> str:
    """The interpreter name to print in user-facing hints.

    The Windows installers put ``python`` (and the ``py`` launcher) on PATH,
    while Linux and macOS reserve ``python`` for whatever is installed --
    often nothing -- and expect ``python3``. Printing the wrong one is the
    single most common way a first run fails, so error messages ask here.
    """
    return "python" if os.name == "nt" else "python3"


def display_path(path, start=None) -> str:
    """A short path for printing, falling back to the absolute one.

    ``os.path.relpath`` raises on Windows when the two paths sit on different
    drives -- which happens as soon as somebody runs the project from ``D:``
    with a working directory on ``C:``, or points a run at another volume.
    A progress message is never worth crashing a training run over, so an
    unrelatable path is simply printed in full.
    """
    try:
        return os.path.relpath(path, start) if start else os.path.relpath(path)
    except (ValueError, OSError):
        return os.fspath(path)


def atomic_write_json(path, obj) -> None:
    """Write JSON so readers never see a partial file.

    ``os.replace`` is atomic on POSIX and on Windows, but Windows also refuses
    to replace a file another process happens to have open -- which the
    dashboard briefly does every poll. A few quick retries turn that race into
    a delay instead of a lost checkpoint.
    """
    path = os.fspath(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                  # Windows: reader has it open
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


class Run:
    """Paths and metadata for one training run."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.dir = Path(CHECKPOINT_ROOT) / name
        self.data_dir = Path(DATA_ROOT) / name
        self.weights_path = self.dir / "weights.f32"
        self.meta_path = self.dir / "meta.json"
        self.config_path = self.dir / "config.json"
        self.snapshot_dir = self.dir / "snapshots"
        self.history_path = self.data_dir / "history.jsonl"
        self.eval_path = self.data_dir / "evaluations.jsonl"
        self.status_path = self.data_dir / "status.json"

    # -- lifecycle ---------------------------------------------------------
    def exists(self) -> bool:
        return self.meta_path.exists()

    def create_dirs(self) -> None:
        """Make every directory this run writes to. Safe to call repeatedly."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict) -> None:
        self.create_dirs()
        atomic_write_json(self.config_path, config)

    def load_config(self):
        return read_json(self.config_path)

    def save_meta(self, meta: dict) -> None:
        meta = dict(meta)
        meta["saved_at"] = time.time()
        meta["saved_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(self.meta_path, meta)

    def load_meta(self):
        return read_json(self.meta_path)

    def write_status(self, status: dict) -> None:
        """Small file the dashboard polls; safe to lose."""
        try:
            atomic_write_json(self.status_path, status)
        except OSError:
            pass

    # -- snapshots ---------------------------------------------------------
    def snapshot(self, games: int) -> str:
        """Freeze a copy of the weights so runs can be compared later."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        dst = self.snapshot_dir / f"games-{games:09d}.f32"
        shutil.copyfile(self.weights_path, dst)
        return str(dst)

    def list_snapshots(self):
        if not self.snapshot_dir.is_dir():
            return []
        return sorted(str(f) for f in self.snapshot_dir.iterdir()
                      if f.suffix == ".f32")

    def size_on_disk(self) -> int:
        total = 0
        for base in (self.dir, self.data_dir):
            for f in Path(base).rglob("*"):
                try:
                    if f.is_file():
                        total += f.stat().st_size
                except OSError:
                    pass
        return total


def list_runs():
    """Every run that has a saved checkpoint, newest first."""
    root = Path(CHECKPOINT_ROOT)
    if not root.is_dir():
        return []
    out = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        r = Run(name)
        if r.exists():
            meta = r.load_meta() or {}
            out.append({
                "name": name,
                "games": meta.get("games", 0),
                "saved_at": meta.get("saved_at", 0),
                "tuple_set": meta.get("tuple_set", "?"),
                "best_score": meta.get("all_time", {}).get("best_score", 0),
                "best_tile": meta.get("all_time", {}).get("best_tile", 0),
            })
    out.sort(key=lambda d: d["saved_at"], reverse=True)
    return out
