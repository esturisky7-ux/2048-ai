"""Install the optional offline demo shipped in the demo release archive.

No downloads, archive extraction, user-supplied paths, or overwrites. The fixed
bundle is verified before an atomic directory rename under the usual run lock.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .checkpoint import Run, atomic_write_json

BUNDLE = Path(__file__).resolve().parent.parent / "demo"
RUN_NAME = "demo-v1"
WEIGHT_BYTES = 8 * (16 ** 4) * 4


def status() -> dict:
    return {"available": (BUNDLE / "weights.f32.gz").is_file(),
            "run": RUN_NAME,
            "release_url": "https://github.com/esturisky7-ux/2048-ai/releases"}


def install() -> dict:
    """Install a verified copy, or reuse this demo's original snapshot."""
    if not status()["available"]:
        raise FileNotFoundError(
            "The optional demo is in the demo edition release ZIP. "
            "Download it from the project's Releases page and run it locally.")
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    if manifest["tuple_set"] != "8x4" or manifest["weight_bytes"] != WEIGHT_BYTES:
        raise ValueError("unsupported demo format")
    games = manifest["games_trained"]
    if type(games) is not int or not 1 <= games <= 10**9:
        raise ValueError("invalid demo game count")
    run = Run(RUN_NAME)
    snap_name = f"games-{games:09d}.f32"
    checkpoint_id = f"{RUN_NAME}:{snap_name}"
    with run.lock(purpose="install pretrained demo"):
        if run.dir.exists():
            snapshot = run.snapshot_dir / snap_name
            if (run.dir / "demo-origin.json").is_file() and snapshot.is_file():
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["sha256"]:
                    return {"run": RUN_NAME, "checkpoint": checkpoint_id,
                            "installed": False}
            raise FileExistsError(
                f"Run {RUN_NAME!r} already exists; it was left unchanged. "
                "Rename or remove it yourself before installing the demo.")
        # Read at most the expected size plus one, even if the bundle is corrupt.
        with gzip.open(BUNDLE / "weights.f32.gz", "rb") as f:
            weights = f.read(WEIGHT_BYTES + 1)
        if len(weights) != WEIGHT_BYTES or hashlib.sha256(weights).hexdigest() != manifest["sha256"]:
            raise ValueError("demo checkpoint checksum or size mismatch")
        stage = Path(tempfile.mkdtemp(prefix=".demo-", dir=run.dir.parent))
        try:
            (stage / "weights.f32").write_bytes(weights)
            (stage / "snapshots").mkdir()
            (stage / "snapshots" / snap_name).write_bytes(weights)
            config = manifest["config"]
            atomic_write_json(stage / "config.json", {**config, "run": RUN_NAME})
            atomic_write_json(stage / "meta.json", {
                "games": games, "tuple_set": "8x4",
                "alpha": config["learning"]["alpha"],
                "demo": True,
            })
            atomic_write_json(stage / "demo-origin.json", manifest)
            os.rename(stage, run.dir)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {"run": RUN_NAME, "checkpoint": checkpoint_id, "installed": True}
