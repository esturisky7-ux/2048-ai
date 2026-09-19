"""System and diagnostics facts for the System page.

Every probe here is best-effort. A metric that cannot be obtained on this
platform returns ``None`` rather than raising, because a diagnostics page that
crashes on the platform you are trying to diagnose is worse than useless.
Nothing in here reads environment variables, credentials or anything outside
the project directory.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from version import __version__                              # noqa: E402

SERVER_STARTED_AT = time.time()


def _total_ram() -> int | None:
    """Physical RAM in bytes, or None where it is not cheaply available."""
    try:                                      # Linux, most Unixes
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=3)
            if out.returncode == 0:
                return int(out.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            pass
    return None


def _available_ram() -> int | None:
    try:                                      # Linux
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    return None


def _cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8",
                      errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=3)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown"


def _git_commit() -> dict | None:
    """Short commit and branch, when the project is a git checkout."""
    if not (ROOT / ".git").exists():
        return None
    git = shutil.which("git")
    if not git:
        return None
    info = {}
    for key, args in (("commit", ["rev-parse", "--short", "HEAD"]),
                      ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            out = subprocess.run([git, "-C", str(ROOT)] + args,
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                info[key] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return info or None
    return info or None


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _disk_free() -> int | None:
    try:
        return shutil.disk_usage(ROOT).free
    except OSError:
        return None


def start_method() -> str:
    from training.trainer import worker_start_method
    try:
        return worker_start_method()
    except Exception:
        return "unknown"


def system_info(manager=None) -> dict:
    """Everything the System page shows."""
    from training.checkpoint import CHECKPOINT_ROOT, DATA_ROOT

    jobs = []
    if manager is not None:
        for job in manager.active():
            jobs.append({"id": job.id, "type": job.type, "pid": job.pid,
                         "label": job.label, "state": job.state})

    return {
        "project": {
            "version": __version__,
            "root": str(ROOT),
            "git": _git_commit(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "bits": sys.maxsize.bit_length() + 1,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "model": _cpu_model(),
            "count": os.cpu_count(),
        },
        "memory": {
            "total_bytes": _total_ram(),
            "available_bytes": _available_ram(),
        },
        "storage": {
            "checkpoints_bytes": _dir_size(Path(CHECKPOINT_ROOT)),
            "data_bytes": _dir_size(Path(DATA_ROOT)),
            "free_bytes": _disk_free(),
        },
        "server": {
            "pid": os.getpid(),
            "uptime_seconds": time.time() - SERVER_STARTED_AT,
            "started_at": SERVER_STARTED_AT,
            "worker_start_method": start_method(),
        },
        "jobs": jobs,
    }
