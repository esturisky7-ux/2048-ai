"""Configuration loading and merging.

Precedence, lowest to highest: ``config/default.json`` -> a named config file
-> the config saved with the run being resumed -> command-line overrides.
Resolved configs are saved next to the checkpoint so an experiment's exact
settings can always be recovered from its results.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

# Every path is derived from this file's own location, so the project runs
# from any working directory and needs no environment variables. pathlib
# builds them with the right separator for the host OS.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.json"
EXPERIMENT_DIR = ROOT / "config" / "experiments"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_default() -> dict:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(name: str) -> str:
    """Accept a path, or the bare name of a file in config/experiments/."""
    if os.path.exists(name):
        return name
    cand = EXPERIMENT_DIR / name
    for p in (cand, cand.with_name(cand.name + ".json")):
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"config {name!r} not found (looked in {EXPERIMENT_DIR})")


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = load_default()
    if path:
        with open(resolve_path(path), encoding="utf-8") as f:
            cfg = deep_merge(cfg, json.load(f))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def list_experiments():
    if not EXPERIMENT_DIR.is_dir():
        return []
    return sorted(f.stem for f in EXPERIMENT_DIR.iterdir()
                  if f.suffix == ".json")
