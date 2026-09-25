"""Run locks: which process may write a run, enforced by the operating system.

A run's weights are one memory-mapped file that a trainer writes directly.
Two trainers on one run would interleave their updates into it while each kept
its own game count, so the checkpoint would describe neither of them; deleting
or resetting a run underneath a live trainer is just as bad. The control
center's job list and the trainer's status heartbeat can *suggest* that a run
is busy, but only for processes they know about, and only once those have
started reporting. The guarantee comes from here instead: an advisory lock on a
small file, taken through the operating system and held on an open file.

* **Exclusive** -- training, and deleting or resetting a run. One holder.
* **Shared** -- evaluating a run's *current* weights, which must not change
  while the evaluation plays. Any number of holders, never alongside an
  exclusive one.

The kernel drops a lock the moment the process holding it exits, however it
exits -- normally, on Ctrl-C, through an exception, or killed outright -- so a
crash can never strand a run and there is no stale lock to clean up. ``flock``
provides this on Linux and macOS, ``LockFileEx`` on Windows. A worker forked by
a trainer shares its parent's lock, so the run stays locked for as long as any
of them could still be writing it.

Lock files live in ``checkpoints/.locks/``, outside the run directories, so
deleting a run never deletes the lock that is protecting the deletion. Each is
empty apart from a short note from the current exclusive holder -- its process
id and what it is doing -- which turns "busy" into an error that says what to
stop.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

# How long an acquisition keeps trying before it reports the run as busy. A
# real holder keeps a run for minutes or hours, so this changes no answer; it
# rides over the instant in which another process is only *checking* whether
# the run is free (see ``inspect``).
DEFAULT_WAIT = 2.0
_RETRY_INTERVAL = 0.05
_NOTE_LIMIT = 4096
_OPEN_FLAGS = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)


class RunBusy(RuntimeError):
    """The run is locked by another process in a conflicting mode.

    ``mode`` is what was found: ``"exclusive"`` (a trainer, or a deletion) or
    ``"shared"`` (evaluations of the current weights). ``holder`` is the
    exclusive holder's note, when it left one.
    """

    def __init__(self, message: str, run: str, mode: str | None,
                 holder: dict | None = None):
        super().__init__(message)
        self.run = run
        self.mode = mode
        self.holder = holder


# ---------------------------------------------------------------------------
# Platform layer: try a non-blocking lock on an open descriptor, and undo it.
# ---------------------------------------------------------------------------
if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _ERROR_LOCK_VIOLATION = 33
    # Windows byte-range locks are mandatory: nobody else may read a locked
    # byte. The lock therefore covers one byte far past the end of the note
    # that other processes read to find out who holds the run.
    _LOCK_OFFSET = 1 << 30

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p),
                    ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    # A private loader, so these prototypes never change ``ctypes.windll``
    # for other code in the same process.
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _LockFileEx = _kernel32.LockFileEx
    _LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                            wintypes.DWORD, wintypes.DWORD,
                            ctypes.POINTER(_Overlapped))
    _LockFileEx.restype = wintypes.BOOL
    _UnlockFileEx = _kernel32.UnlockFileEx
    _UnlockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                              wintypes.DWORD, ctypes.POINTER(_Overlapped))
    _UnlockFileEx.restype = wintypes.BOOL

    def _region() -> "_Overlapped":
        ov = _Overlapped()
        ov.Offset = _LOCK_OFFSET & 0xFFFFFFFF
        ov.OffsetHigh = _LOCK_OFFSET >> 32
        return ov

    def _try_lock(fd: int, shared: bool) -> bool:
        flags = _LOCKFILE_FAIL_IMMEDIATELY
        if not shared:
            flags |= _LOCKFILE_EXCLUSIVE_LOCK
        ov = _region()
        if _LockFileEx(msvcrt.get_osfhandle(fd), flags, 0, 1, 0,
                       ctypes.byref(ov)):
            return True
        err = ctypes.get_last_error()
        if err == _ERROR_LOCK_VIOLATION:
            return False
        raise ctypes.WinError(err)

    def _unlock(fd: int) -> None:
        ov = _region()
        _UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(ov))

else:
    import errno
    import fcntl

    def _try_lock(fd: int, shared: bool) -> bool:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def _read_note(path) -> dict | None:
    try:
        with open(path, "rb") as f:
            data = f.read(_NOTE_LIMIT)
        note = json.loads(data.decode("utf-8")) if data.strip() else None
    except (OSError, ValueError):
        return None
    return note if isinstance(note, dict) else None


def _write_note(fd: int, note: dict | None) -> None:
    """Replace the holder's note. A courtesy for error messages: the lock is
    what matters, so failing to write the note never fails the lock."""
    data = json.dumps(note).encode("utf-8") if note else b""
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        if data:
            os.write(fd, data)
    except OSError:
        pass


def inspect(path) -> tuple[str | None, dict | None]:
    """Who holds the lock at ``path`` right now, without keeping it.

    Returns ``(mode, note)``: mode is ``None`` when the run is free,
    ``"shared"`` when only evaluations hold it, and ``"exclusive"`` when a
    trainer or a deletion does, in which case ``note`` says who (if known).

    The answer can be stale as soon as it is returned, so it is only good for
    friendly early refusals and error messages. Anything that must not race
    has to *acquire* the lock instead.
    """
    try:
        fd = os.open(path, _OPEN_FLAGS, 0o644)
    except FileNotFoundError:
        return None, None          # no lock directory yet: nothing holds it
    try:
        if _try_lock(fd, shared=False):
            _unlock(fd)
            return None, None
        if _try_lock(fd, shared=True):
            _unlock(fd)
            return "shared", None
        return "exclusive", _read_note(path)
    finally:
        os.close(fd)


def describe_holder(note: dict | None) -> str:
    """``"process 1234 (training, since 14:02:11)"``, or a generic phrase."""
    if not note or not note.get("pid"):
        return "another process"
    detail = []
    if note.get("purpose"):
        detail.append(str(note["purpose"]))
    if note.get("since"):
        try:
            detail.append("since " + time.strftime(
                "%H:%M:%S", time.localtime(float(note["since"]))))
        except (TypeError, ValueError, OverflowError, OSError):
            pass
    return f"process {note['pid']}" + (f" ({', '.join(detail)})"
                                       if detail else "")


def busy_message(run: str, want_shared: bool, mode: str | None,
                 note: dict | None) -> str:
    """The sentence a refused caller shows, worded for what it wanted."""
    if want_shared:
        return (f"run '{run}' is being trained by {describe_holder(note)}, so "
                f"its current weights keep changing and an evaluation of them "
                f"would not measure one fixed policy. Evaluate one of its "
                f"snapshots instead, or wait until training stops.")
    if mode == "shared":
        return (f"run '{run}' is in use by an evaluation of its current "
                f"weights. Wait for the evaluation to finish (or stop it), "
                f"then try again.")
    return (f"run '{run}' is already in use by {describe_holder(note)}. Only "
            f"one process may train, reset or delete a run at a time: stop "
            f"that one first (Ctrl-C in its terminal, or Stop in the control "
            f"center), or use a different run name.")


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------
class RunLock:
    """A lock on one run, held from :meth:`acquire` until :meth:`release`.

    Usually made by :meth:`training.checkpoint.Run.lock`. Works as a context
    manager too::

        with Run("default").lock(purpose="deleting the run").acquire():
            ...
    """

    def __init__(self, path, run: str, shared: bool = False,
                 purpose: str = ""):
        self.path = Path(path)
        self.run = run
        self.shared = bool(shared)
        self.purpose = purpose
        self._fd: int | None = None
        self._pid: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None and self._pid == os.getpid()

    def acquire(self, wait: float = DEFAULT_WAIT) -> "RunLock":
        """Take the lock, or raise :class:`RunBusy` after ``wait`` seconds."""
        if self._fd is not None:
            raise RuntimeError(f"the lock on run '{self.run}' is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, _OPEN_FLAGS, 0o644)
        try:
            deadline = time.monotonic() + max(0.0, wait)
            while not _try_lock(fd, self.shared):
                if time.monotonic() >= deadline:
                    mode, note = inspect(self.path)
                    raise RunBusy(busy_message(self.run, self.shared, mode,
                                               note),
                                  self.run, mode, note)
                time.sleep(_RETRY_INTERVAL)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._pid = os.getpid()
        if not self.shared:
            _write_note(fd, {"pid": self._pid, "run": self.run,
                             "purpose": self.purpose, "since": time.time()})
        return self

    def release(self) -> None:
        """Give the lock up. Safe to call twice, or if it was never taken."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        if self._pid != os.getpid():
            # A forked worker's copy of its parent's lock. Unlocking it here
            # would unlock it for the parent too, which still owns the run.
            return
        try:
            if not self.shared:
                _write_note(fd, None)
            _unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> "RunLock":
        if self._fd is None:
            self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "held" if self.held else "free"
        mode = "shared" if self.shared else "exclusive"
        return f"<RunLock {self.run!r} {mode} {state}>"


@contextmanager
def hold_frozen(runs, purpose: str = "evaluation"):
    """Keep these runs' current weights from changing until the block ends.

    ``runs`` names the runs whose *current* weights are about to be
    evaluated -- a learned agent's ``live_run`` -- and may contain ``None``
    for agents that play no such thing (snapshots, which never change, and
    agents without learned weights). For each run this holds a *shared* lock
    for the whole block, so the evaluation cannot start while the run is
    being trained -- it raises :class:`RunBusy` instead -- and training cannot
    start until the evaluation is done.
    """
    from .checkpoint import Run
    locks = []
    try:
        for run in sorted({r for r in runs if r}):
            locks.append(Run(run).lock(shared=True, purpose=purpose).acquire())
        yield
    finally:
        for lock in reversed(locks):
            lock.release()
