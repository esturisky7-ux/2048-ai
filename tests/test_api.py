"""API tests: routing, validation, and the security boundary.

These drive :mod:`dashboard.api` directly rather than over HTTP, so they are
fast and can assert on the exact error a bad request produces. The HTTP layer
itself — CSRF headers, path traversal, streaming — is covered by
``test_server.py``, and the whole stack is exercised end to end by
``test_end_to_end.py``.

The validation tests matter more than they look: this API can start processes
and delete files, so "what does it do with input it did not expect" is the
question that decides whether a stray fetch from another browser tab can ruin
someone's training run.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import api                                    # noqa: E402
from dashboard import store                                  # noqa: E402
from dashboard.api import ApiError                           # noqa: E402
from dashboard.games import SESSIONS                         # noqa: E402
from dashboard.jobs import MANAGER                           # noqa: E402
from training import checkpoint as CP                        # noqa: E402


class SandboxedRoots(unittest.TestCase):
    """Point the checkpoint and data roots at a temporary directory.

    Two layers, because monkeypatching a module global only sandboxes *this*
    process. A job launches a child, and a child re-imports everything fresh:

    1. the module globals are patched, for anything running in-process;
    2. ``AI2048_HOME`` is set, which a child does inherit;
    3. and the job manager is stubbed outright, because an API unit test has
       no business starting a real training process at all.

    Layer 3 is the one that matters. An earlier version of this file had only
    layer 1, and a validation test that accidentally passed validation
    launched ``train.py`` against the developer's real checkpoint directory.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="2048api-")
        self._cp, self._dr = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        self._scp, self._sdr = store.CHECKPOINT_ROOT, store.DATA_ROOT
        self._home = os.environ.get("AI2048_HOME")
        os.environ["AI2048_HOME"] = self.tmp
        self._submit = MANAGER.submit
        MANAGER.submit = self._refuse_to_launch
        root = Path(self.tmp)
        CP.CHECKPOINT_ROOT = store.CHECKPOINT_ROOT = root / "checkpoints"
        CP.DATA_ROOT = store.DATA_ROOT = root / "data"
        store.SETTINGS_PATH = CP.DATA_ROOT / "ui-settings.json"
        store.LABELS_PATH = CP.DATA_ROOT / "checkpoint-labels.json"
        store.COMPARISON_DIR = CP.DATA_ROOT / "comparisons"
        store.BENCHMARK_DIR = CP.DATA_ROOT / "benchmarks"

    @staticmethod
    def _refuse_to_launch(job_type, label, argv, **kwargs):
        raise AssertionError(
            f"an API unit test tried to launch a real subprocess: {argv}. "
            f"End-to-end behaviour belongs in test_control_center.py.")

    def tearDown(self):
        MANAGER.submit = self._submit
        if self._home is None:
            os.environ.pop("AI2048_HOME", None)
        else:
            os.environ["AI2048_HOME"] = self._home
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = self._cp, self._dr
        store.CHECKPOINT_ROOT, store.DATA_ROOT = self._scp, self._sdr
        store.SETTINGS_PATH = Path(self._sdr) / "ui-settings.json"
        store.LABELS_PATH = Path(self._sdr) / "checkpoint-labels.json"
        store.COMPARISON_DIR = Path(self._sdr) / "comparisons"
        store.BENCHMARK_DIR = Path(self._sdr) / "benchmarks"
        SESSIONS.stop_all()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, name="default", games=100, tuple_set="8x4"):
        from training.ntuple import NTupleNetwork
        run = CP.Run(name)
        run.create_dirs()
        NTupleNetwork(tuple_set, path=str(run.weights_path)).close()
        run.save_config({"tuple_set": tuple_set, "run": name,
                         "learning": {"alpha": 0.1},
                         "training": {"checkpoint_every": 2000,
                                      "eval_every": 20000}})
        run.save_meta({"games": games, "tuple_set": tuple_set,
                       "all_time": {"games": games, "best_score": 4242,
                                    "best_tile": 512, "score_sum": games * 100},
                       "alpha": 0.1})
        return run


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class TestRouting(SandboxedRoots):
    def test_unknown_routes_are_404(self):
        for path in ("/api/nope", "/api/game/", "/api/jobs/x/nope"):
            with self.assertRaises(ApiError) as ctx:
                api.handle_get(path, {})
            self.assertEqual(ctx.exception.status, 404, path)
        with self.assertRaises(ApiError) as ctx:
            api.handle_post("/api/nope", {})
        self.assertEqual(ctx.exception.status, 404)

    def test_core_get_endpoints_answer(self):
        self.make_run()
        for path in ("/api/status", "/api/runs", "/api/agents", "/api/training",
                     "/api/jobs", "/api/checkpoints", "/api/experiments",
                     "/api/system", "/api/logs", "/api/settings",
                     "/api/sessions", "/api/comparisons", "/api/benchmarks",
                     "/api/evaluations", "/api/history"):
            payload = api.handle_get(path, {})
            self.assertIsInstance(payload, dict, path)
            json.dumps(payload, default=str)        # must be serialisable

    def test_status_reports_a_missing_run_without_failing(self):
        p = api.handle_get("/api/status", {"run": ["never-trained"]})
        self.assertFalse(p["exists"])
        self.assertEqual(p["state"], "STOPPED")
        self.assertEqual(p["games"], 0)

    def test_status_lists_the_runs_once_per_payload(self):
        """Every open tab builds one of these a second."""
        from unittest import mock
        self.make_run("a")
        self.make_run("b")
        calls = []

        def counting():
            calls.append(1)
            return CP.list_runs()
        with mock.patch.object(api, "list_runs", counting):
            p = api.status_payload("a")
        self.assertEqual(len(calls), 1)
        self.assertEqual({r["name"] for r in p["runs"]}, {"a", "b"})
        self.assertTrue(p["has_any_run"])

    def test_evaluations_are_read_newest_first_and_only_as_needed(self):
        run = self.make_run("evals")
        rows = [{"games": 10, "mean_score": float(i)} for i in range(6)]
        with open(run.eval_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps(row) + "\n")
                if i == 2:
                    f.write('{"torn": \n\n')        # a crash mid-write
        means = lambda got: [r["mean_score"] for r in got]    # noqa: E731
        self.assertEqual(means(store.list_evaluations("evals", 2)), [4.0, 5.0])
        self.assertEqual(means(store.list_evaluations("evals", 4)),
                         [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(means(store.list_evaluations("evals", 100)),
                         [float(i) for i in range(6)])
        self.assertEqual(api.status_payload("evals")["last_eval"]
                         ["mean_score"], 5.0)

    def test_status_has_the_fields_the_overview_needs(self):
        self.make_run(games=5000)
        p = api.handle_get("/api/status", {})
        for key in ("state", "games", "session_games", "workers", "alpha",
                    "games_per_second", "moves_per_second", "schedule",
                    "all_time", "rolling", "tuple_set", "total_train_seconds",
                    "checkpoint_saved_at", "has_any_run", "jobs", "version"):
            self.assertIn(key, p, key)
        self.assertEqual(p["schedule"]["checkpoint_every"], 2000)
        # 5000 games into a 2000-game checkpoint cycle => 1000 games to go.
        self.assertEqual(p["schedule"]["games_to_checkpoint"], 1000)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class TestValidation(SandboxedRoots):
    def test_run_names_that_are_not_safe_path_components_are_refused(self):
        for bad in ("../evil", "a/b", "a\\b", "", " ", "a b", "..",
                    "x" * 65, "con", "NUL", ".hidden", "a\x00b"):
            with self.assertRaises(ApiError, msg=bad):
                api.start_training({"run": bad, "games": 10})

    def test_good_run_names_are_accepted(self):
        for good in ("default", "my-run", "run_2", "a.b", "X1"):
            self.assertEqual(store.validate_run_name(good), good)

    def test_numeric_fields_are_range_checked(self):
        self.make_run("r1")
        for payload, word in (
            ({"run": "r1", "resume": True, "games": -5}, "between"),
            ({"run": "r1", "resume": True, "games": "many"}, "whole number"),
            ({"run": "r1", "resume": True, "workers": 0}, "between"),
            ({"run": "r1", "resume": True, "workers": 9999}, "between"),
            ({"run": "r1", "resume": True, "alpha": 50}, "between"),
            ({"run": "r1", "resume": True, "epsilon": 2}, "between"),
        ):
            with self.assertRaises(ApiError, msg=str(payload)) as ctx:
                api.start_training(payload)
            self.assertIn(word, ctx.exception.message)

    def test_tuple_set_must_be_known(self):
        with self.assertRaises(ApiError) as ctx:
            api.start_training({"run": "new", "games": 10, "tuple_set": "9x9"})
        self.assertIn("tuple_set", ctx.exception.message)

    def test_resume_requires_an_existing_run(self):
        with self.assertRaises(ApiError) as ctx:
            api.start_training({"run": "ghost", "resume": True})
        self.assertIn("no checkpoint", ctx.exception.message)

    def test_new_run_refuses_to_clobber_an_existing_one(self):
        self.make_run("taken")
        with self.assertRaises(ApiError) as ctx:
            api.start_training({"run": "taken", "games": 10})
        self.assertIn("already exists", ctx.exception.message)

    def test_agent_names_are_an_allowlist(self):
        for bad in ("gpt", "", None, "learned; rm -rf /", 42):
            with self.assertRaises(ApiError):
                api.start_evaluation({"agent": bad, "games": 5})

    def test_evaluating_an_untrained_run_is_a_clear_404(self):
        with self.assertRaises(ApiError) as ctx:
            api.start_evaluation({"agent": "learned", "run": "ghost",
                                  "games": 5})
        self.assertEqual(ctx.exception.status, 404)
        self.assertIn("not been trained", ctx.exception.message)

    def test_comparison_requires_a_sensible_agent_list(self):
        for bad in ({}, {"agents": []}, {"agents": "random"},
                    {"agents": [1, 2]}, {"agents": ["a"] * 7}):
            with self.assertRaises(ApiError, msg=str(bad)):
                api.start_comparison(bad)

    def test_unknown_experiment_is_404(self):
        with self.assertRaises(ApiError) as ctx:
            api.start_experiment({"name": "does-not-exist"})
        self.assertEqual(ctx.exception.status, 404)


# ---------------------------------------------------------------------------
# Checkpoint identifiers: the file-system boundary
# ---------------------------------------------------------------------------
class TestSandboxIsRealYo(SandboxedRoots):
    """The sandbox has to actually sandbox, or these tests damage real data."""

    def test_the_manager_refuses_to_launch_from_a_unit_test(self):
        with self.assertRaises(AssertionError):
            MANAGER.submit("training", "nope", ["echo", "hi"])

    def test_a_valid_training_request_never_reaches_a_subprocess(self):
        self.make_run("existing")
        with self.assertRaises(AssertionError):
            api.start_training({"run": "existing", "resume": True,
                                "games": 10})

    def test_roots_point_inside_the_temporary_directory(self):
        for root in (CP.CHECKPOINT_ROOT, CP.DATA_ROOT,
                     store.CHECKPOINT_ROOT, store.DATA_ROOT):
            self.assertTrue(str(root).startswith(self.tmp), str(root))
        self.assertEqual(os.environ["AI2048_HOME"], self.tmp)


class TestCheckpointIds(SandboxedRoots):
    TRAVERSALS = [
        "../../etc/passwd", "/etc/passwd", "..", "../default",
        "default:../../../etc/passwd", "default:../weights.f32",
        "default:/etc/passwd", "default:weights.f32", "default:..",
        "default:games-1.f32", "a/b", "a\\b", "",
        "default:games-000000001.f32/../../x",
    ]

    def test_traversal_attempts_never_produce_a_path(self):
        self.make_run()
        for bad in self.TRAVERSALS:
            with self.assertRaises((store.InvalidName, FileNotFoundError),
                                   msg=bad):
                store.checkpoint_path(bad)

    def test_delete_refuses_every_traversal(self):
        self.make_run()
        for bad in self.TRAVERSALS:
            with self.assertRaises(ApiError, msg=bad):
                api.handle_post("/api/checkpoints/delete",
                                {"id": bad, "confirm": True})

    def test_a_valid_id_resolves_inside_the_checkpoint_directory(self):
        run = self.make_run("real")
        path = store.checkpoint_path("real")
        self.assertTrue(path.exists())
        self.assertTrue(str(path.resolve()).startswith(
            str(Path(CP.CHECKPOINT_ROOT).resolve())))
        self.assertEqual(path, run.weights_path)

    def test_delete_requires_explicit_confirmation(self):
        self.make_run("doomed")
        with self.assertRaises(ApiError) as ctx:
            api.handle_post("/api/checkpoints/delete", {"id": "doomed"})
        self.assertIn("confirm", ctx.exception.message)
        self.assertTrue(CP.Run("doomed").exists())

    def test_delete_removes_only_that_run(self):
        self.make_run("keep")
        self.make_run("doomed")
        api.handle_post("/api/checkpoints/delete",
                        {"id": "doomed", "confirm": True})
        self.assertFalse(CP.Run("doomed").exists())
        self.assertTrue(CP.Run("keep").exists())

    def test_labels_are_validated_and_bounded(self):
        self.make_run("labelled")
        api.handle_post("/api/checkpoints/label",
                        {"id": "labelled", "label": "x" * 500})
        found = [c for c in store.list_checkpoints() if c["id"] == "labelled"]
        self.assertEqual(len(found), 1)
        self.assertLessEqual(len(found[0]["label"]), 80)
        with self.assertRaises(ApiError):
            api.handle_post("/api/checkpoints/label",
                            {"id": "../x", "label": "no"})

    def test_listing_reports_real_metadata(self):
        self.make_run("listed", games=777)
        entry = [c for c in store.list_checkpoints() if c["id"] == "listed"][0]
        self.assertEqual(entry["games"], 777)
        self.assertEqual(entry["tuple_set"], "8x4")
        self.assertEqual(entry["kind"], "current")
        self.assertGreater(entry["size_bytes"], 0)


# ---------------------------------------------------------------------------
# Game sessions
# ---------------------------------------------------------------------------
class TestGameSessions(SandboxedRoots):
    def test_human_game_applies_the_python_rules(self):
        r = api.start_human_game({"seed": 42})
        sid = r["session"]["id"]
        state = r["session"]
        self.assertEqual(state["score"], 0)
        self.assertEqual(state["moves"], 0)
        # Exactly two tiles on a new board.
        filled = sum(1 for v in sum(state["board"], []) if v)
        self.assertEqual(filled, 2)

        moved_any = False
        for direction in ("left", "up", "right", "down"):
            s = api.game_move(sid, {"direction": direction})
            moved_any = moved_any or s["moved"]
            self.assertIn("legal", s)
            self.assertGreaterEqual(s["score"], 0)
        self.assertTrue(moved_any)

    def test_human_game_is_reproducible_from_its_seed(self):
        a = api.start_human_game({"seed": 12345})["session"]
        b = api.start_human_game({"seed": 12345})["session"]
        self.assertEqual(a["board"], b["board"])

    def test_bad_directions_are_refused(self):
        sid = api.start_human_game({})["session"]["id"]
        for bad in ("sideways", "", None, 3, "UP "):
            with self.assertRaises(ApiError, msg=str(bad)):
                api.game_move(sid, {"direction": bad})

    def test_moves_on_an_unknown_session_are_404(self):
        with self.assertRaises(ApiError) as ctx:
            api.game_move("nope", {"direction": "left"})
        self.assertEqual(ctx.exception.status, 404)

    def test_ai_game_runs_and_can_be_controlled(self):
        import time
        r = api.start_ai_game({"agent": "random"})
        sid = r["session"]["id"]
        deadline = time.time() + 10
        while time.time() < deadline:
            snap = api.game_state(sid, 0)
            if snap["total"] > 3:
                break
            time.sleep(0.1)
        self.assertGreater(snap["total"], 3, "the agent produced no frames")
        self.assertEqual(len(snap["frames"][0]["board"]), 4)

        self.assertTrue(api.game_control(sid, {"action": "pause"})
                        ["session"]["paused"])
        api.game_control(sid, {"action": "step"})
        api.game_control(sid, {"action": "resume"})
        api.game_control(sid, {"action": "stop"})
        with self.assertRaises(ApiError):
            api.game_state(sid, 0)

    def test_control_actions_are_an_allowlist(self):
        sid = api.start_ai_game({"agent": "random"})["session"]["id"]
        try:
            for bad in ("delete", "", None, "PAUSE"):
                with self.assertRaises(ApiError, msg=str(bad)):
                    api.game_control(sid, {"action": bad})
        finally:
            api.game_control(sid, {"action": "stop"})

    def test_human_sessions_reject_ai_controls(self):
        sid = api.start_human_game({})["session"]["id"]
        with self.assertRaises(ApiError):
            api.game_control(sid, {"action": "pause"})

    def test_ai_game_on_an_untrained_run_is_a_clear_error(self):
        with self.assertRaises(ApiError) as ctx:
            api.start_ai_game({"agent": "learned", "run": "ghost"})
        self.assertEqual(ctx.exception.status, 404)

    def test_session_cap_is_enforced(self):
        from dashboard import games as G
        made = []
        try:
            for _ in range(G.MAX_SESSIONS + 4):
                try:
                    made.append(api.start_human_game({})["session"]["id"])
                except ApiError as e:
                    self.assertEqual(e.status, 409)
                    break
            self.assertLessEqual(len(SESSIONS.sessions), G.MAX_SESSIONS)
        finally:
            SESSIONS.stop_all()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class TestSettings(SandboxedRoots):
    def test_defaults_suggest_a_polite_worker_count(self):
        d = store.default_settings()
        cores = os.cpu_count() or 1
        self.assertGreaterEqual(d["workers"], 1)
        self.assertLessEqual(d["workers"], max(1, cores - 1),
                             "the default should not consume every core")

    def test_unknown_keys_are_ignored_not_stored(self):
        saved = store.save_settings({"theme": "light", "evil": "payload",
                                     "__proto__": "nope"})
        self.assertEqual(saved["theme"], "light")
        self.assertNotIn("evil", saved)
        self.assertNotIn("__proto__", saved)

    def test_values_are_clamped_to_sane_ranges(self):
        s = store.save_settings({"refresh_ms": 1, "eval_games": 10 ** 9,
                                 "workers": 9999, "playback_speed": -5,
                                 "theme": "rainbow"})
        self.assertGreaterEqual(s["refresh_ms"], 500)
        self.assertLessEqual(s["eval_games"], 100000)
        self.assertLessEqual(s["workers"], 64)
        self.assertGreaterEqual(s["playback_speed"], 0.0)
        self.assertEqual(s["theme"], "dark")

    def test_settings_round_trip_through_disk(self):
        store.save_settings({"eval_games": 321, "theme": "light"})
        loaded = store.load_settings()
        self.assertEqual(loaded["eval_games"], 321)
        self.assertEqual(loaded["theme"], "light")

    def test_bad_types_do_not_raise(self):
        s = store.save_settings({"refresh_ms": "soon", "workers": None,
                                 "confirm_destructive": "yes"})
        self.assertIsInstance(s["refresh_ms"], int)
        self.assertIsInstance(s["workers"], int)
        self.assertIs(s["confirm_destructive"], True)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
class TestEventLog(unittest.TestCase):
    def setUp(self):
        from dashboard.events import EventLog
        self.tmp = tempfile.mkdtemp(prefix="2048log-")
        self.log = EventLog(Path(self.tmp) / "events.jsonl", ring=50)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_events_are_recorded_and_returned_newest_last(self):
        for i in range(5):
            self.log.add("info", f"event {i}")
        rows = self.log.recent(10)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["message"], "event 4")

    def test_level_filtering_includes_more_severe_levels(self):
        self.log.add("debug", "d")
        self.log.add("info", "i")
        self.log.add("warn", "w")
        self.log.add("error", "e")
        self.assertEqual(len(self.log.recent(10, level="warn")), 2)
        self.assertEqual(len(self.log.recent(10, level="error")), 1)
        self.assertEqual(len(self.log.recent(10)), 4)

    def test_only_simple_field_values_are_stored(self):
        ev = self.log.add("info", "x", num=1, flag=True, obj={"a": 1},
                          nothing=None)
        self.assertEqual(ev["num"], 1)
        self.assertIs(ev["flag"], True)
        self.assertIsInstance(ev["obj"], str)
        self.assertNotIn("nothing", ev)

    def test_unknown_levels_fall_back_to_info(self):
        self.assertEqual(self.log.add("catastrophe", "x")["level"], "info")

    def test_ring_is_bounded(self):
        for i in range(120):
            self.log.add("info", str(i))
        self.assertLessEqual(len(self.log.recent(1000)), 50)

    def test_since_seq_returns_only_newer_events(self):
        self.log.add("info", "a")
        marker = self.log.recent(1)[-1]["seq"]
        self.log.add("info", "b")
        rows = self.log.recent(10, since_seq=marker)
        self.assertEqual([r["message"] for r in rows], ["b"])

    def test_a_torn_final_line_does_not_lose_the_file(self):
        from dashboard.events import EventLog
        self.log.add("info", "one")
        self.log.add("info", "two")
        with open(self.log.path, "a", encoding="utf-8") as f:
            f.write('{"ts": 1, "message": "trunc')
        fresh = EventLog(self.log.path, ring=50)
        fresh.load()
        self.assertEqual([r["message"] for r in fresh.recent(10)],
                         ["one", "two"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
