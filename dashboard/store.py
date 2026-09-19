"""Checkpoint inventory, saved results and UI settings.

Three small stores that the API reads and writes, kept together because they
share one rule: **the browser never supplies a filesystem path.** It supplies
an identifier, and this module is the only thing that turns an identifier into
a path — inside the project's own directories, verified afterwards. That keeps
"read a checkpoint" from ever becoming "read any file on this machine".
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from training.checkpoint import (CHECKPOINT_ROOT, DATA_ROOT, Run,  # noqa: F401
                                 atomic_write_json, list_runs, read_json)

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = Path(DATA_ROOT) / "ui-settings.json"
LABELS_PATH = Path(DATA_ROOT) / "checkpoint-labels.json"
COMPARISON_DIR = Path(DATA_ROOT) / "comparisons"
BENCHMARK_DIR = Path(DATA_ROOT) / "benchmarks"

# A run name has to be safe as a single path component on every platform, so
# it is restricted rather than sanitised: anything outside this is rejected.
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SNAPSHOT_RE = re.compile(r"^games-\d{9}\.f32$")
# Windows refuses these names regardless of extension.
RESERVED_NAMES = {"con", "prn", "aux", "nul",
                  *(f"com{i}" for i in range(1, 10)),
                  *(f"lpt{i}" for i in range(1, 10))}


class InvalidName(ValueError):
    """A run name or checkpoint id that will not be turned into a path."""


def validate_run_name(name: str) -> str:
    if not isinstance(name, str) or not RUN_NAME_RE.match(name):
        raise InvalidName(
            "run names may use letters, digits, dot, dash and underscore "
            "(1-64 characters) and must start with a letter or digit")
    if name.lower() in RESERVED_NAMES or name in (".", ".."):
        raise InvalidName(f"{name!r} is a reserved name")
    return name


def _inside(path: Path, parent: Path) -> bool:
    """True when ``path`` really is under ``parent`` after resolution."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def parse_checkpoint_id(cid: str) -> tuple[str, str | None]:
    """``"default"`` or ``"default:games-000050000.f32"`` -> (run, snapshot)."""
    if not isinstance(cid, str) or not cid:
        raise InvalidName("missing checkpoint id")
    run, sep, snap = cid.partition(":")
    validate_run_name(run)
    if not sep:
        return run, None
    if not SNAPSHOT_RE.match(snap):
        raise InvalidName(f"{snap!r} is not a snapshot name")
    return run, snap


def checkpoint_path(cid: str) -> Path:
    """Resolve a checkpoint id to a weight file, or raise.

    The id is parsed, rebuilt into a path under ``checkpoints/``, and then
    verified to actually live there. Both halves matter: the pattern stops the
    obvious traversal, and the containment check stops anything clever with
    symlinks or platform-specific path quirks.
    """
    run, snap = parse_checkpoint_id(cid)
    r = Run(run)
    path = r.weights_path if snap is None else (r.snapshot_dir / snap)
    root = Path(CHECKPOINT_ROOT)
    if not _inside(path, root):
        raise InvalidName("checkpoint is outside the checkpoint directory")
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {cid!r}")
    return path


def _labels() -> dict:
    return read_json(LABELS_PATH, {}) or {}


def set_label(cid: str, label: str | None) -> None:
    parse_checkpoint_id(cid)                       # validate before storing
    data = _labels()
    if label:
        data[cid] = str(label)[:80]
    else:
        data.pop(cid, None)
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    atomic_write_json(LABELS_PATH, data)


def _latest_eval_for(run_name: str) -> dict | None:
    run = Run(run_name)
    if not run.eval_path.exists():
        return None
    last = None
    try:
        with open(run.eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return None
    return last


def list_checkpoints() -> list[dict]:
    """Every run's current weights plus every frozen snapshot."""
    labels = _labels()
    out = []
    for info in list_runs():
        name = info["name"]
        run = Run(name)
        meta = run.load_meta() or {}
        cfg = run.load_config() or {}
        ev = _latest_eval_for(name)
        try:
            size = run.weights_path.stat().st_size
        except OSError:
            size = 0
        all_time = meta.get("all_time") or {}
        out.append({
            "id": name,
            "run": name,
            "kind": "current",
            "label": labels.get(name),
            "games": meta.get("games", 0),
            "tuple_set": meta.get("tuple_set"),
            "saved_at": meta.get("saved_at"),
            "saved_at_iso": meta.get("saved_at_iso"),
            "size_bytes": size,
            "best_score": all_time.get("best_score", 0),
            "best_tile": all_time.get("best_tile", 0),
            "alpha": meta.get("alpha"),
            "learning": cfg.get("learning"),
            "eval": {
                "mean_score": ev.get("mean_score"),
                "games": ev.get("games"),
                "ci95_mean": ev.get("ci95_mean"),
                "highest_tile": ev.get("highest_tile"),
                "games_trained": (ev.get("games_trained")
                                  or (ev.get("agent") or {}).get(
                                      "games_trained")),
            } if ev else None,
        })
        for snap in run.list_snapshots():
            p = Path(snap)
            cid = f"{name}:{p.name}"
            try:
                size = p.stat().st_size
                mtime = p.stat().st_mtime
            except OSError:
                size, mtime = 0, None
            digits = "".join(ch for ch in p.stem if ch.isdigit())
            out.append({
                "id": cid,
                "run": name,
                "kind": "snapshot",
                "label": labels.get(cid),
                "games": int(digits) if digits else 0,
                "tuple_set": meta.get("tuple_set"),
                "saved_at": mtime,
                "size_bytes": size,
                "best_score": None,
                "best_tile": None,
                "alpha": None,
                "learning": cfg.get("learning"),
                "eval": None,
            })
    out.sort(key=lambda d: (d["run"], d["kind"] != "current", -d["games"]))
    return out


def delete_checkpoint(cid: str) -> dict:
    """Delete a snapshot, or an entire run. Never touches anything else."""
    run_name, snap = parse_checkpoint_id(cid)
    if snap is not None:
        path = checkpoint_path(cid)
        path.unlink()
        return {"deleted": cid, "kind": "snapshot"}

    import shutil
    run = Run(run_name)
    root_ck, root_dt = Path(CHECKPOINT_ROOT), Path(DATA_ROOT)
    removed = []
    for path, parent in ((run.dir, root_ck), (run.data_dir, root_dt)):
        if path.exists() and _inside(path, parent):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path.name))
    set_label(cid, None)
    return {"deleted": cid, "kind": "run", "removed": removed}


# ---------------------------------------------------------------------------
# Saved comparison and benchmark results
# ---------------------------------------------------------------------------
def _save_json_result(directory: Path, payload: dict, prefix: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}-{int(time.time() * 1000)}.json"
    atomic_write_json(directory / name, payload)
    return name


def save_comparison(payload: dict) -> str:
    return _save_json_result(COMPARISON_DIR, payload, "cmp")


def save_benchmark(payload: dict) -> str:
    return _save_json_result(BENCHMARK_DIR, payload, "bench")


def _load_results(directory: Path, limit: int) -> list[dict]:
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.json"), reverse=True)[:limit]
    out = []
    for f in files:
        data = read_json(f)
        if isinstance(data, dict):
            data["_file"] = f.name
            out.append(data)
    return out


def list_comparisons(limit: int = 20) -> list[dict]:
    return _load_results(COMPARISON_DIR, limit)


def list_benchmarks(limit: int = 20) -> list[dict]:
    return _load_results(BENCHMARK_DIR, limit)


def list_evaluations(run_name: str, limit: int = 200) -> list[dict]:
    """Every fixed-seed evaluation recorded for a run, oldest first."""
    validate_run_name(run_name)
    run = Run(run_name)
    rows: list[dict] = []
    if not run.eval_path.exists():
        return rows
    try:
        with open(run.eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


# ---------------------------------------------------------------------------
# UI settings
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "theme": "dark",                 # dark | light | system
    "refresh_ms": 2000,              # dashboard poll interval
    "playback_speed": 1,             # default live-game speed multiplier
    "eval_games": 200,               # default evaluation length
    "workers": 1,                    # default worker count (overridden below)
    "confirm_destructive": True,     # ask before deleting a checkpoint
    "show_tooltips": True,           # ML term explanations
    "compact_numbers": True,
}

# Only these keys are accepted from the browser, with these types.
SETTING_TYPES = {
    "theme": str, "refresh_ms": int, "playback_speed": float,
    "eval_games": int, "workers": int, "confirm_destructive": bool,
    "show_tooltips": bool, "compact_numbers": bool,
}


def default_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    # Suggest half the cores by default: enough to be faster, not so much that
    # the machine becomes unusable while training.
    cores = os.cpu_count() or 1
    s["workers"] = max(1, min(cores - 1, cores // 2)) or 1
    return s


def load_settings() -> dict:
    s = default_settings()
    saved = read_json(SETTINGS_PATH, {}) or {}
    for k, v in saved.items():
        if k in SETTING_TYPES:
            s[k] = v
    return s


def save_settings(patch: dict) -> dict:
    s = load_settings()
    for k, v in (patch or {}).items():
        want = SETTING_TYPES.get(k)
        if want is None:
            continue                               # unknown key: ignored
        try:
            if want is bool:
                s[k] = bool(v)
            elif want is int:
                s[k] = int(v)
            elif want is float:
                s[k] = float(v)
            else:
                s[k] = str(v)[:40]
        except (TypeError, ValueError):
            continue
    s["theme"] = s["theme"] if s["theme"] in ("dark", "light", "system") \
        else "dark"
    s["refresh_ms"] = max(500, min(int(s["refresh_ms"]), 30000))
    s["eval_games"] = max(1, min(int(s["eval_games"]), 100000))
    s["workers"] = max(1, min(int(s["workers"]), 64))
    s["playback_speed"] = max(0.0, min(float(s["playback_speed"]), 100.0))
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    atomic_write_json(SETTINGS_PATH, s)
    return s
