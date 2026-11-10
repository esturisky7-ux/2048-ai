"""Running unattended on a Linux server or VM.

A long run is started once and left alone, so what matters is what happens
when something goes wrong with nobody watching: the SSH session drops, the
disk fills, the machine reboots. These tests cover the pieces that make that
survivable --

* the terminal hanging up (SIGHUP) saves and stops instead of vanishing, but
  leaves a run started under ``nohup`` running, and never mistakes the
  repeated hangups a terminal sends for "exit now";
* output that has nowhere to go does not kill training;
* the weight file's disk space can be reserved up front, so a full disk is an
  error message at start-up and not a SIGBUS hours later;
* ``--workers auto`` counts the CPUs the process may use, not the machine's;
* ``--resume-if-exists`` and ``--target-games`` make one command that is safe
  to run again and again, which is what a supervisor needs;
* the launcher script and the systemd unit in ``deploy/``.
"""

import contextlib
import errno
import io
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as TRAIN                                          # noqa: E402
from training import checkpoint as CP                          # noqa: E402
from training.ntuple import NTupleNetwork                      # noqa: E402
from training.trainer import Trainer                           # noqa: E402

PY = sys.executable
WINDOWS = os.name == "nt"
POSIX_ONLY = unittest.skipIf(WINDOWS, "needs POSIX signals and process groups")


class HomeTest(unittest.TestCase):
    """A private AI2048_HOME per class, for subprocesses and in-process use."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="2048longrun-")
        cls.env = dict(os.environ, AI2048_HOME=cls.home, PYTHONUNBUFFERED="1")
        cls.env.pop("AI2048_START_METHOD", None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.home, ignore_errors=True)

    def setUp(self):
        self._roots = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        CP.CHECKPOINT_ROOT = Path(self.home) / "checkpoints"
        CP.DATA_ROOT = Path(self.home) / "data"

    def tearDown(self):
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = self._roots

    def games_on_disk(self, run) -> int:
        return (CP.Run(run).load_meta() or {}).get("games", 0)

    def cli(self, *args, expect=0):
        r = subprocess.run([PY, "train.py", *args], cwd=ROOT, env=self.env,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, expect, r.stdout + r.stderr)
        return r


# ---------------------------------------------------------------------------
# CPUs and --workers auto
# ---------------------------------------------------------------------------
class TestAvailableCpus(unittest.TestCase):
    def test_follows_the_affinity_mask_not_the_machine(self):
        with mock.patch.object(os, "sched_getaffinity", create=True,
                               return_value={0, 1, 2}):
            self.assertEqual(CP.available_cpus(), 3)

    def test_falls_back_to_the_machine_count_without_an_affinity_call(self):
        for failure in (OSError, AttributeError):
            with self.subTest(failure=failure.__name__), \
                    mock.patch.object(os, "sched_getaffinity", create=True,
                                      side_effect=failure):
                self.assertEqual(CP.available_cpus(), os.cpu_count() or 1)

    def test_an_empty_mask_is_not_zero_workers(self):
        with mock.patch.object(os, "sched_getaffinity", create=True,
                               return_value=set()):
            self.assertGreaterEqual(CP.available_cpus(), 1)

    def test_is_at_least_one_and_never_more_than_the_machine(self):
        n = CP.available_cpus()
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, os.cpu_count() or 1)


class TestWorkersArgument(HomeTest):
    def test_parses_a_count_or_auto(self):
        self.assertEqual(TRAIN.workers_arg("3"), 3)
        self.assertEqual(TRAIN.workers_arg("auto"), "auto")
        self.assertEqual(TRAIN.workers_arg(" AUTO "), "auto")

    def test_rejects_anything_else_with_a_useful_message(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            TRAIN.workers_arg("many")
        self.assertIn("auto", str(cm.exception))

    def test_auto_trains_with_one_worker_per_available_cpu(self):
        r = self.cli("--run", "auto-workers", "--tuple-set", "8x4",
                     "--games", "16", "--workers", "auto",
                     "--report-every", "0", "--eval-every", "0")
        self.assertIn(f"{CP.available_cpus()} worker(s)", r.stdout)
        self.assertEqual(self.games_on_disk("auto-workers"), 16)


# ---------------------------------------------------------------------------
# One command that is safe to run again: --resume-if-exists, --target-games
# ---------------------------------------------------------------------------
class TestResumeAndTarget(HomeTest):
    QUIET = ("--report-every", "0", "--eval-every", "0")

    def test_starts_a_new_run_then_continues_it_toward_a_total(self):
        run = "target"
        self.cli("--run", run, "--tuple-set", "8x4", "--resume-if-exists",
                 "--target-games", "30", *self.QUIET)
        self.assertEqual(self.games_on_disk(run), 30)

        # Again, same command: already there, so nothing happens and the
        # weights are not touched.
        before = (CP.Run(run).weights_path).read_bytes()
        r = self.cli("--run", run, "--tuple-set", "8x4", "--resume-if-exists",
                     "--target-games", "30", *self.QUIET)
        self.assertIn("nothing to do", r.stdout)
        self.assertEqual(self.games_on_disk(run), 30)
        self.assertEqual(CP.Run(run).weights_path.read_bytes(), before)

        # A bigger total continues the same run from where it stopped.
        r = self.cli("--run", run, "--resume-if-exists",
                     "--target-games", "50", *self.QUIET)
        self.assertIn("starting at game 30", r.stdout)
        self.assertEqual(self.games_on_disk(run), 50)

    def test_resume_if_exists_works_with_a_per_session_game_count_too(self):
        run = "per-session"
        for expected in (10, 20):
            self.cli("--run", run, "--tuple-set", "8x4", "--resume-if-exists",
                     "--games", "10", *self.QUIET)
            self.assertEqual(self.games_on_disk(run), expected)

    def test_target_games_counts_the_whole_run_not_the_session(self):
        run = "total"
        self.cli("--run", run, "--tuple-set", "8x4", "--games", "25",
                 *self.QUIET)
        r = self.cli("--run", run, "--resume", "--target-games", "40",
                     *self.QUIET)
        self.assertIn("starting at game 25", r.stdout)
        self.assertEqual(self.games_on_disk(run), 40)

    def test_games_and_target_games_are_alternatives(self):
        r = self.cli("--run", "both", "--games", "5", "--target-games", "5",
                     expect=2)
        self.assertIn("use one of them", r.stderr)

    def test_resume_still_refuses_a_run_that_does_not_exist(self):
        r = self.cli("--run", "never-trained", "--resume", expect=1)
        self.assertIn("no checkpoint", r.stderr)


# ---------------------------------------------------------------------------
# The terminal going away
# ---------------------------------------------------------------------------
@POSIX_ONLY
class TestHangup(HomeTest):
    """What the kernel does when an SSH session drops: SIGHUP, to the whole
    foreground process group, usually more than once."""

    def start(self, run, workers=1, ignoring=False):
        cmd = [PY, "train.py", "--run", run, "--tuple-set", "8x4",
               "--games", "0", "--workers", str(workers),
               "--report-every", "0", "--eval-every", "0",
               "--checkpoint-every", "0"]
        if ignoring:
            # What nohup does: SIGHUP is ignored before the program starts,
            # and an ignored signal stays ignored across exec.
            cmd = ["sh", "-c", 'trap "" HUP; exec "$@"', "sh"] + cmd
        p = subprocess.Popen(cmd, cwd=ROOT, env=self.env,
                             start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        self.addCleanup(self.reap, p)
        return p

    @staticmethod
    def reap(p):
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
        p.communicate()

    def games_reported(self, run) -> int:
        return (CP.read_json(CP.Run(run).status_path) or {}).get("games", 0)

    def wait_until(self, run, games, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.games_reported(run) >= games:
                return
            time.sleep(0.1)
        self.fail(f"{run} never reached {games} games")

    def hang_up(self, p, times=1):
        for _ in range(times):
            os.killpg(p.pid, signal.SIGHUP)
            time.sleep(0.02)

    def assert_saved_and_stopped(self, run, p, out):
        self.assertEqual(p.returncode, 0, out)
        self.assertIn("terminal closed (SIGHUP)", out)
        self.assertNotIn("Traceback", out)
        self.assertGreater(self.games_on_disk(run), 0,
                           "no checkpoint was written on the way out")
        self.assertFalse((CP.read_json(CP.Run(run).status_path) or {})
                         .get("running"))
        self.assertEqual(CP.Run(run).lock_holder(), (None, None))

    def test_a_hangup_saves_and_stops_the_run(self):
        for workers in (1, 2):
            with self.subTest(workers=workers):
                run = f"hup-{workers}"
                p = self.start(run, workers)
                self.wait_until(run, 100)
                self.hang_up(p)
                out, _ = p.communicate(timeout=120)
                self.assert_saved_and_stopped(run, p, out)

    def test_repeated_hangups_are_not_a_force_quit(self):
        run = "hup-repeated"
        p = self.start(run, workers=2)
        self.wait_until(run, 100)
        self.hang_up(p, times=4)
        out, _ = p.communicate(timeout=120)
        self.assertNotIn("exiting immediately", out)
        self.assert_saved_and_stopped(run, p, out)

    def test_a_run_started_under_nohup_outlives_the_hangup(self):
        run = "hup-nohup"
        p = self.start(run, workers=2, ignoring=True)
        self.wait_until(run, 100)
        self.hang_up(p, times=2)
        seen = self.games_reported(run)
        # Still training afterwards: more games than when the hangup came.
        self.wait_until(run, seen + 100, timeout=60)
        self.assertIsNone(p.poll(), "the run stopped; nohup asked it not to")
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=120)
        self.assertEqual(p.returncode, 0, out)
        self.assertNotIn("SIGHUP", out)
        self.assertIn("stopping after the current game", out)
        self.assertGreater(self.games_on_disk(run), seen)


class TestOutputGoingAway(HomeTest):
    def test_a_closed_pipe_does_not_stop_training(self):
        run = "closed-pipe"
        p = subprocess.Popen(
            [PY, "train.py", "--run", run, "--tuple-set", "8x4",
             "--games", "400", "--report-every", "20", "--eval-every", "0"],
            cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        try:
            p.stdout.readline()           # the banner; then nobody is reading
            p.stdout.close()
            p.wait(timeout=120)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
        self.assertEqual(p.returncode, 0)
        self.assertEqual(self.games_on_disk(run), 400)

    def test_log_goes_quiet_instead_of_raising(self):
        class Dead(io.StringIO):
            def write(self, s):
                raise OSError(errno.EIO, "Input/output error")

        trainer = Trainer.__new__(Trainer)
        trainer.quiet = False
        with contextlib.redirect_stdout(Dead()):
            trainer._log("one")           # must not raise
            self.assertTrue(trainer.quiet)
            trainer._log("two")           # and now does not even try


# ---------------------------------------------------------------------------
# Reserving the weight file's disk space
# ---------------------------------------------------------------------------
def allocated(path) -> int:
    return os.stat(path).st_blocks * 512


@unittest.skipIf(WINDOWS, "st_blocks is POSIX")
class TestPreallocation(HomeTest):
    def path(self, name):
        return str(Path(self.home) / name)

    def net(self, name, **kw):
        net = NTupleNetwork("8x4", path=self.path(name), **kw)
        self.addCleanup(net.close)
        return net

    @unittest.skipUnless(hasattr(os, "posix_fallocate"), "Linux only")
    def test_reserves_every_block_of_the_file(self):
        net = self.net("reserved.f32", preallocate=True)
        self.assertGreaterEqual(allocated(net.path), net.nbytes())

    @unittest.skipUnless(hasattr(os, "posix_fallocate"), "Linux only")
    def test_default_is_still_a_sparse_file(self):
        net = self.net("sparse.f32")
        if allocated(net.path) >= net.nbytes():
            self.skipTest("this filesystem has no sparse files")
        self.assertLess(allocated(net.path), net.nbytes())

    @unittest.skipUnless(hasattr(os, "posix_fallocate"), "Linux only")
    def test_leaves_weights_that_are_already_there_alone(self):
        net = self.net("existing.f32")
        net.weights[7] = 1.5
        net.weights[net.n_weights - 1] = -2.25
        net.flush()
        net.close()
        again = NTupleNetwork("8x4", path=net.path, preallocate=True)
        self.addCleanup(again.close)
        self.assertEqual(again.weights[7], 1.5)
        self.assertEqual(again.weights[again.n_weights - 1], -2.25)

    def test_a_full_disk_is_an_ordinary_error_naming_the_problem(self):
        with mock.patch.object(os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(OSError) as cm:
                NTupleNetwork("8x4", path=self.path("full.f32"),
                              preallocate=True)
        self.assertEqual(cm.exception.errno, errno.ENOSPC)
        self.assertIn("free disk space", str(cm.exception))
        self.assertIn("AI2048_HOME", str(cm.exception))

    def test_a_filesystem_that_cannot_reserve_just_stays_sparse(self):
        for code in (errno.EOPNOTSUPP, errno.ENOSYS):
            with self.subTest(errno=errno.errorcode[code]), \
                    mock.patch.object(os, "posix_fallocate", create=True,
                                      side_effect=OSError(code, "no")):
                net = self.net(f"unsupported-{code}.f32", preallocate=True)
                net.weights[3] = 4.0
                self.assertEqual(net.weights[3], 4.0)

    def test_an_unexpected_failure_is_not_swallowed(self):
        with mock.patch.object(os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.EIO, "disk error")):
            with self.assertRaises(OSError) as cm:
                NTupleNetwork("8x4", path=self.path("broken.f32"),
                              preallocate=True)
        self.assertEqual(cm.exception.errno, errno.EIO)

    def test_the_trainer_refuses_to_start_and_lets_go_of_the_run(self):
        from training.config import load_config
        cfg = load_config(overrides={"run": "no-room", "tuple_set": "8x4"})
        with mock.patch.object(os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(OSError):
                Trainer("no-room", cfg, quiet=True, preallocate=True)
        self.assertEqual(CP.Run("no-room").lock_holder(), (None, None))

    def test_train_py_reports_a_full_disk_without_a_traceback(self):
        err = io.StringIO()
        argv = ["train.py", "--run", "no-room-cli", "--tuple-set", "8x4",
                "--games", "1"]
        with mock.patch.object(os, "posix_fallocate", create=True,
                               side_effect=OSError(errno.ENOSPC, "full")), \
                mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stderr(err):
            code = TRAIN.main()
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_a_status_heartbeat_is_never_fsynced_but_a_checkpoint_is(self):
        run = CP.Run("fsync-check")
        run.create_dirs()
        with mock.patch.object(os, "fsync") as fsync:
            run.write_status({"running": True})
            self.assertEqual(fsync.call_count, 0)
            run.save_meta({"games": 1})
            self.assertEqual(fsync.call_count, 1)
        self.assertEqual(CP.read_json(run.status_path), {"running": True})


# ---------------------------------------------------------------------------
# deploy/: the launcher and the systemd unit
# ---------------------------------------------------------------------------
LAUNCHER = os.path.join(ROOT, "deploy", "long-run.sh")
UNIT = os.path.join(ROOT, "deploy", "2048-train.service")


@unittest.skipIf(WINDOWS or not shutil.which("bash"), "needs bash")
class TestLauncher(HomeTest):
    def launch(self, **env):
        e = dict(self.env, **env)
        return subprocess.run(
            ["bash", LAUNCHER, "--tuple-set", "8x4", "--report-every", "0",
             "--eval-every", "0"],
            cwd=ROOT, env=e, capture_output=True, text=True, timeout=300)

    def test_is_valid_bash_and_executable(self):
        r = subprocess.run(["bash", "-n", LAUNCHER], capture_output=True,
                           text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.access(LAUNCHER, os.X_OK),
                        "deploy/long-run.sh lost its executable bit")

    def test_starts_continues_and_stops_at_the_target(self):
        r = self.launch(RUN="launcher", WORKERS="2", TARGET_GAMES="24")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.games_on_disk("launcher"), 24)

        r = self.launch(RUN="launcher", WORKERS="2", TARGET_GAMES="24")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("nothing to do", r.stdout)

        r = self.launch(RUN="launcher", TARGET_GAMES="40")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("starting at game 24", r.stdout)
        self.assertEqual(self.games_on_disk("launcher"), 40)

    def test_a_bad_worker_count_fails_instead_of_being_ignored(self):
        r = self.launch(RUN="launcher-bad", WORKERS="lots", TARGET_GAMES="5")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--workers", r.stderr)


class TestSystemdUnit(unittest.TestCase):
    def setUp(self):
        with open(UNIT, encoding="utf-8") as f:
            self.text = f.read()
        self.settings = [line for line in self.text.splitlines()
                         if line.strip() and not line.lstrip().startswith(
                             ("#", ";"))]

    def value(self, key):
        found = [line.split("=", 1)[1] for line in self.settings
                 if line.startswith(key + "=")]
        self.assertTrue(found, f"{key} is not set")
        return found[-1]

    def test_stops_the_trainer_alone_so_it_can_stop_its_workers(self):
        # Workers keep SIGTERM's default action; if systemd signalled the whole
        # control group they would die mid-game and the stop would "fail".
        self.assertEqual(self.value("KillMode"), "mixed")
        self.assertEqual(self.value("KillSignal"), "SIGTERM")

    def test_finishes_stopping_inside_a_spot_vms_shutdown_window(self):
        seconds = int(re.fullmatch(r"(\d+)s?", self.value("TimeoutStopSec"))
                      .group(1))
        self.assertLessEqual(seconds, 30)

    def test_restarts_after_failures_but_not_after_a_clean_stop(self):
        self.assertEqual(self.value("Restart"), "on-failure")

    def test_no_setting_has_a_trailing_comment(self):
        # systemd would take "# note" as part of the value.
        for line in self.settings:
            if "=" in line:
                self.assertNotRegex(line, r"\s#", line)

    def test_runs_the_launcher_with_the_placeholders_the_docs_fill_in(self):
        self.assertIn("@USER@", self.text)
        self.assertIn("@DIR@", self.text)
        self.assertTrue(self.value("ExecStart").startswith("@DIR@/deploy/"
                                                           "long-run.sh"))


if __name__ == "__main__":
    unittest.main()
