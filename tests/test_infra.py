"""Tests for checkpointing, config merging, statistics, evaluation and the
dashboard API -- the parts that must not lose data or lie about numbers."""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import checkpoint as CP                            # noqa: E402
from training.config import deep_merge, load_config              # noqa: E402
from training.stats import (AllTimeStats, RollingStats, History,  # noqa: E402
                            _percentile)
from training.trainer import alpha_at                             # noqa: E402
from evaluation.evaluator import (wilson_interval, game_seed,     # noqa: E402
                                  summarise, format_report)


class TempRoots(unittest.TestCase):
    """Redirect checkpoint/data roots so tests never touch real runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="2048infra-")
        self._cp, self._dr = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        CP.CHECKPOINT_ROOT = os.path.join(self.tmp, "checkpoints")
        CP.DATA_ROOT = os.path.join(self.tmp, "data")

    def tearDown(self):
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = self._cp, self._dr
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAtomicWrite(TempRoots):
    def test_replaces_atomically_and_leaves_no_temp(self):
        p = os.path.join(self.tmp, "x.json")
        CP.atomic_write_json(p, {"a": 1})
        CP.atomic_write_json(p, {"a": 2})
        self.assertEqual(CP.read_json(p), {"a": 2})
        self.assertFalse(os.path.exists(p + ".tmp"))

    def test_read_json_tolerates_corruption(self):
        p = os.path.join(self.tmp, "bad.json")
        with open(p, "w") as f:
            f.write("{not json")
        self.assertIsNone(CP.read_json(p))
        self.assertEqual(CP.read_json(p, default={}), {})

    def test_read_json_missing_file(self):
        self.assertIsNone(CP.read_json(os.path.join(self.tmp, "nope.json")))


class TestRun(TempRoots):
    def test_meta_roundtrip_and_exists(self):
        r = CP.Run("t1")
        self.assertFalse(r.exists())
        r.create_dirs()
        r.save_meta({"games": 500, "tuple_set": "4x5"})
        self.assertTrue(r.exists())
        m = r.load_meta()
        self.assertEqual(m["games"], 500)
        self.assertIn("saved_at", m)
        self.assertIn("saved_at_iso", m)

    def test_config_roundtrip(self):
        r = CP.Run("t2")
        cfg = {"tuple_set": "8x4", "learning": {"alpha": 0.2}}
        r.save_config(cfg)
        self.assertEqual(r.load_config(), cfg)

    def test_list_runs_sorted_by_recency(self):
        import time
        a = CP.Run("older"); a.create_dirs(); a.save_meta({"games": 1})
        time.sleep(0.02)
        b = CP.Run("newer"); b.create_dirs(); b.save_meta({"games": 2})
        names = [r["name"] for r in CP.list_runs()]
        self.assertEqual(names[0], "newer")
        self.assertIn("older", names)

    def test_status_write_is_non_fatal(self):
        r = CP.Run("t3")
        r.create_dirs()
        r.write_status({"running": True})
        self.assertTrue(CP.read_json(r.status_path)["running"])

    def test_snapshot_copies_weights(self):
        r = CP.Run("t4")
        r.create_dirs()
        with open(r.weights_path, "wb") as f:
            f.write(b"\x01\x02\x03\x04")
        dst = r.snapshot(1234)
        self.assertTrue(os.path.exists(dst))
        self.assertIn("games-000001234.f32", dst)
        self.assertEqual(len(r.list_snapshots()), 1)

    def test_size_on_disk(self):
        r = CP.Run("t5")
        r.create_dirs()
        with open(r.weights_path, "wb") as f:
            f.write(b"x" * 2048)
        self.assertGreaterEqual(r.size_on_disk(), 2048)


class TestConfig(unittest.TestCase):
    def test_deep_merge_is_recursive(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        got = deep_merge(base, {"a": {"y": 9}, "c": 4})
        self.assertEqual(got, {"a": {"x": 1, "y": 9}, "b": 3, "c": 4})

    def test_deep_merge_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        self.assertEqual(base["a"]["x"], 1)

    def test_default_config_has_required_sections(self):
        cfg = load_config()
        for k in ("run", "tuple_set", "learning", "reward", "training",
                  "evaluation"):
            self.assertIn(k, cfg)
        self.assertEqual(cfg["reward"]["score"], 1.0)
        # every shaping term must default to off
        for k in ("milestone", "game_over_penalty", "empty_potential",
                  "survival"):
            self.assertEqual(cfg["reward"][k], 0.0)

    def test_overrides_apply(self):
        cfg = load_config(overrides={"learning": {"alpha": 0.42}})
        self.assertEqual(cfg["learning"]["alpha"], 0.42)
        self.assertEqual(cfg["learning"]["gamma"], 1.0)

    def test_every_experiment_config_is_valid(self):
        """Each shipped experiment must build a legal reward/evaluator config."""
        from training.config import EXPERIMENT_DIR
        from training.reward import RewardFunction
        from agents.heuristic import BoardEvaluator, DEFAULT_WEIGHTS
        from training.ntuple import TUPLE_SETS
        for fn in sorted(os.listdir(EXPERIMENT_DIR)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(EXPERIMENT_DIR, fn), encoding="utf-8") as f:
                spec = json.load(f)
            self.assertIn("description", spec, fn)
            if spec.get("kind") == "agent":
                w = (spec.get("agent") or {}).get("weights")
                if w:
                    self.assertFalse(set(w) - set(DEFAULT_WEIGHTS), fn)
            else:
                RewardFunction(spec.get("reward"))     # raises on bad keys
                if "tuple_set" in spec:
                    self.assertIn(spec["tuple_set"], TUPLE_SETS, fn)


class TestAlphaSchedule(unittest.TestCase):
    def test_constant_by_default(self):
        cfg = load_config()
        self.assertEqual(alpha_at(cfg, 0), alpha_at(cfg, 10 ** 6))

    def test_geometric_decay_with_floor(self):
        cfg = load_config(overrides={"learning": {
            "alpha": 0.2, "alpha_decay": 0.5,
            "alpha_decay_every": 1000, "alpha_min": 0.05}})
        self.assertAlmostEqual(alpha_at(cfg, 0), 0.2)
        self.assertAlmostEqual(alpha_at(cfg, 1000), 0.1)
        self.assertAlmostEqual(alpha_at(cfg, 2000), 0.05)
        self.assertAlmostEqual(alpha_at(cfg, 10 ** 6), 0.05)   # floor holds


class TestStats(unittest.TestCase):
    def test_percentile_matches_known_values(self):
        v = [1, 2, 3, 4, 5]
        self.assertEqual(_percentile(v, 0.0), 1)
        self.assertEqual(_percentile(v, 0.5), 3)
        self.assertEqual(_percentile(v, 1.0), 5)
        self.assertAlmostEqual(_percentile(v, 0.25), 2.0)
        self.assertEqual(_percentile([], 0.5), 0.0)
        self.assertEqual(_percentile([7], 0.5), 7.0)

    def test_rolling_window_evicts(self):
        r = RollingStats(3)
        for i in range(10):
            r.add(i * 100, i, 2 ** (i + 1))
        self.assertEqual(len(r), 3)
        self.assertEqual(r.summary()["mean_score"], (700 + 800 + 900) / 3)

    def test_rolling_empty(self):
        self.assertEqual(RollingStats(5).summary(), {"games": 0})

    def test_tile_rates_are_cumulative(self):
        r = RollingStats(10)
        r.add(100, 10, 2048)
        r.add(100, 10, 256)
        s = r.summary()
        self.assertAlmostEqual(s["tile_rates"]["512"], 0.5)
        self.assertAlmostEqual(s["tile_rates"]["2048"], 0.5)
        self.assertAlmostEqual(s["tile_rates"]["4096"], 0.0)

    def test_all_time_records_and_roundtrip(self):
        a = AllTimeStats()
        a.add(100, 10, 64)
        a.add(9000, 900, 2048)
        a.add(50, 5, 32)
        self.assertEqual(a.games, 3)
        self.assertEqual(a.best_score, 9000)
        self.assertEqual(a.best_score_game, 2)
        self.assertEqual(a.best_tile, 2048)
        self.assertEqual(a.longest_game, 900)
        b = AllTimeStats(a.dump())
        self.assertEqual(b.dump(), a.dump())
        self.assertAlmostEqual(b.summary()["mean_score"], (100 + 9000 + 50) / 3)

    def test_history_append_and_tolerate_torn_line(self):
        d = tempfile.mkdtemp(prefix="2048hist-")
        try:
            p = os.path.join(d, "h.jsonl")
            h = History(p)
            h.append({"games": 1})
            h.append({"games": 2})
            with open(p, "a") as f:
                f.write('{"games": 3, "trunc')      # simulate a crash mid-write
            rows = h.read()
            self.assertEqual([r["games"] for r in rows], [1, 2])
            self.assertIn("wall_time", rows[0])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestEvaluatorStats(unittest.TestCase):
    def test_wilson_bounds_are_sane(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        lo, hi = wilson_interval(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0 < hi < 0.06)
        lo, hi = wilson_interval(100, 100)
        self.assertAlmostEqual(hi, 1.0, places=12)   # exact 1 up to fp noise
        self.assertTrue(0.94 < lo < 1.0)
        lo, hi = wilson_interval(50, 100)
        self.assertTrue(lo < 0.5 < hi)
        # more data must tighten the interval
        w1 = wilson_interval(50, 100)
        w2 = wilson_interval(500, 1000)
        self.assertLess(w2[1] - w2[0], w1[1] - w1[0])

    def test_game_seed_deterministic_and_spread(self):
        self.assertEqual(game_seed(1, 1), game_seed(1, 1))
        self.assertNotEqual(game_seed(1, 1), game_seed(1, 2))
        self.assertNotEqual(game_seed(1, 1), game_seed(2, 1))
        self.assertEqual(len({game_seed(7, i) for i in range(1000)}), 1000)

    def test_summarise_numbers(self):
        scores = [100, 200, 300, 400, 500]
        res = summarise(scores, [10] * 5, [64, 128, 512, 1024, 2048],
                        1.0, 42, {"name": "x"})
        self.assertEqual(res["games"], 5)
        self.assertEqual(res["mean_score"], 300)
        self.assertEqual(res["median_score"], 300)
        self.assertEqual(res["min_score"], 100)
        self.assertEqual(res["max_score"], 500)
        self.assertEqual(res["highest_tile"], 2048)
        # sample std dev of 100..500 is 158.11
        self.assertAlmostEqual(res["std_score"], 158.113883, places=4)
        self.assertAlmostEqual(res["stderr_score"],
                               158.113883 / math.sqrt(5), places=4)
        lo, hi = res["ci95_mean"]
        self.assertLess(lo, 300)
        self.assertGreater(hi, 300)
        self.assertAlmostEqual(res["tile_rates"]["512"]["rate"], 0.6)
        self.assertAlmostEqual(res["tile_rates"]["2048"]["rate"], 0.2)

    def test_summarise_empty(self):
        self.assertEqual(summarise([], [], [], 0.0, 1, {})["games"], 0)

    def test_format_report_runs(self):
        res = summarise([100, 200], [5, 6], [64, 128], 0.5, 1, {"name": "x"})
        self.assertIn("mean score", format_report(res))
        self.assertEqual(format_report({"games": 0}), "no games played")


class TestPortability(unittest.TestCase):
    """The parts that differ between operating systems, checked on all of them."""

    def test_paths_are_built_for_this_os(self):
        """Nothing is hardcoded to a POSIX layout; Run paths are absolute."""
        from pathlib import Path
        r = CP.Run("portability")
        for p in (r.dir, r.weights_path, r.meta_path, r.history_path):
            self.assertIsInstance(p, Path)
            self.assertTrue(p.is_absolute(), p)
        # the separator is whatever this OS uses
        self.assertIn(os.sep, str(r.weights_path))
        self.assertTrue(str(r.weights_path).endswith("weights.f32"))

    def test_python_command_matches_the_platform(self):
        cmd = CP.python_command()
        self.assertEqual(cmd, "python" if os.name == "nt" else "python3")

    def test_worker_start_method_is_supported_here(self):
        import multiprocessing as mp
        from training.trainer import worker_start_method
        method = worker_start_method()
        self.assertIn(method, mp.get_all_start_methods())
        if os.name == "nt":
            self.assertEqual(method, "spawn", "Windows has no fork")

    def test_worker_network_spec_is_picklable(self):
        """Spawn pickles worker arguments; a live mmap never survives that."""
        import pickle
        from training.trainer import _open_worker_network
        spec = ("8x4", str(os.path.join(self.__class__.__name__, "w.f32")))
        pickle.loads(pickle.dumps(spec))        # must not raise
        with tempfile.TemporaryDirectory(prefix="2048spec-") as d:
            path = os.path.join(d, "weights.f32")
            net, mine = _open_worker_network(("8x4", path))
            try:
                self.assertTrue(mine, "a spawned worker owns its own mapping")
                self.assertEqual(net.tuple_set_name, "8x4")
            finally:
                net.close()

    def test_pid_alive_answers_correctly_here(self):
        from dashboard import server as DS
        self.assertTrue(DS.pid_alive(os.getpid()))
        self.assertFalse(DS.pid_alive(-1))
        self.assertFalse(DS.pid_alive(None))
        self.assertFalse(DS.pid_alive("not a pid"))

    def test_pid_alive_never_calls_os_kill_on_windows(self):
        """On Windows os.kill(pid, 0) *terminates* the process it asks about."""
        import unittest.mock as mock
        from dashboard import server as DS

        def forbidden(*a, **k):
            raise AssertionError("os.kill must not be used on Windows")

        with mock.patch.object(DS.os, "name", "nt"), \
                mock.patch.object(DS.os, "kill", forbidden):
            # ctypes.windll does not exist off Windows, so the helper falls
            # back to "assume alive" rather than signalling anything.
            self.assertTrue(DS.pid_alive(os.getpid()))
            self.assertFalse(DS.pid_alive(0))

    def test_display_path_survives_an_unrelatable_path(self):
        """On Windows relpath raises across drive letters; printing must not."""
        import unittest.mock as mock
        here = os.path.abspath(os.curdir)
        self.assertEqual(CP.display_path(os.path.join(here, "a", "b"), here),
                         os.path.join("a", "b"))

        def cross_drive(*a, **k):
            raise ValueError("path is on mount 'C:', start on mount 'D:'")

        with mock.patch.object(CP.os.path, "relpath", cross_drive):
            got = CP.display_path(os.path.join(here, "a", "b"))
        self.assertEqual(got, os.path.join(here, "a", "b"))

    def test_atomic_write_retries_when_the_target_is_locked(self):
        """Windows refuses os.replace while a reader has the file open."""
        import unittest.mock as mock
        with tempfile.TemporaryDirectory(prefix="2048lock-") as d:
            path = os.path.join(d, "meta.json")
            calls = []
            real = os.replace

            def flaky(src, dst):
                calls.append(dst)
                if len(calls) < 3:
                    raise PermissionError(32, "file in use")
                return real(src, dst)

            with mock.patch.object(os, "replace", flaky):
                CP.atomic_write_json(path, {"games": 7})
            self.assertEqual(CP.read_json(path), {"games": 7})
            self.assertEqual(len(calls), 3)
            self.assertFalse(os.path.exists(path + ".tmp"))


class TestDashboardAPI(TempRoots):
    def test_downsample_keeps_endpoints(self):
        from dashboard.server import downsample
        rows = list(range(1000))
        out = downsample(rows, 100)
        self.assertLessEqual(len(out), 101)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[-1], 999)
        self.assertEqual(downsample([1, 2, 3], 100), [1, 2, 3])

    def test_history_series_shapes_match(self):
        from dashboard.server import history_series
        r = CP.Run("dash")
        r.create_dirs()
        h = History(r.history_path)
        for i in range(5):
            h.append({"games": i * 100, "games_per_second": 10.0, "alpha": 0.1,
                      "rolling": {"games": 100, "mean_score": 1000 + i,
                                  "median_score": 900, "max_tile": 512,
                                  "tile_rates": {"512": 0.5, "1024": 0.1,
                                                 "2048": 0.0, "4096": 0.0}}})
        s = history_series(r)
        n = len(s["games"])
        self.assertEqual(n, 5)
        for k, v in s.items():
            self.assertEqual(len(v), n, f"series {k} has the wrong length")

    def test_history_series_skips_empty_rows(self):
        from dashboard.server import history_series
        r = CP.Run("dash2")
        r.create_dirs()
        History(r.history_path).append({"games": 0, "rolling": {"games": 0}})
        self.assertEqual(history_series(r)["games"], [])

    def test_status_payload_reports_stale_process_as_stopped(self):
        import time
        from dashboard.server import status_payload
        r = CP.Run("stale")
        r.create_dirs()
        r.save_meta({"games": 10, "tuple_set": "8x4",
                     "all_time": AllTimeStats().dump()})
        # a status claiming to run, but written long ago by a dead pid
        r.write_status({"running": True, "pid": 999999,
                        "updated_at": time.time() - 9999})
        self.assertFalse(status_payload("stale")["running"])

    def test_status_payload_derives_rates(self):
        from dashboard.server import status_payload
        a = AllTimeStats()
        a.add(1000, 100, 2048)
        a.add(10, 5, 64)
        r = CP.Run("rates")
        r.create_dirs()
        r.save_meta({"games": 2, "tuple_set": "8x4", "all_time": a.dump()})
        p = status_payload("rates")
        self.assertAlmostEqual(p["all_time"]["tile_rates"]["2048"], 0.5)
        self.assertAlmostEqual(p["all_time"]["mean_score"], 505.0)

    def test_status_payload_for_missing_run(self):
        from dashboard.server import status_payload
        p = status_payload("never-existed")
        self.assertFalse(p["exists"])
        self.assertFalse(p["running"])
        self.assertEqual(p["games"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
