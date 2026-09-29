"""Regression coverage for honest evaluation, exact decisions and offline demo."""
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import board as B
from evaluation.evaluator import evaluate, play_one, InvalidActionError, format_report
from training.learner import GreedyPolicy
from training.reward import RewardFunction
from training import checkpoint as CP, demo
from agents.learned import LearnedAgent
from dashboard.games import AIGameSession


class FirstLegal:
    def act(self, board):
        return B.legal_actions(board)[0]


class BadAgent:
    def __init__(self, action):
        self.action = action
    def act(self, board):
        return self.action


class TestEvaluationIntegrity(unittest.TestCase):
    def test_invalid_actions_fail_by_default(self):
        for action in (-1, 4, None, "left", True, []):
            with self.subTest(action=action), self.assertRaises(InvalidActionError):
                evaluate(BadAgent(action), games=1)

    def test_legal_direction_that_cannot_move_is_invalid(self):
        board = B.from_list([2, 2, 0, 0] + [0] * 12)
        with patch.object(B, "new_game", return_value=board):
            with self.assertRaises(InvalidActionError):
                play_one(BadAgent(B.UP), 1)

    def test_diagnostic_fallback_and_truncation_are_explicit(self):
        result = evaluate(BadAgent(-1), games=2, move_limit=3, strict=False)
        self.assertEqual(result["integrity"]["invalid_actions"], 6)
        self.assertEqual(result["integrity"]["truncated_games"], 2)
        self.assertFalse(result["integrity"]["valid"])
        self.assertIn("WARNING", format_report(result))
        self.assertEqual(len(result["per_game"]), 2)

    def test_healthy_games_pass_and_replay(self):
        result = evaluate(FirstLegal(), games=2, seed=77)
        self.assertTrue(result["integrity"]["valid"])
        for game in result["per_game"]:
            self.assertEqual(play_one(FirstLegal(), int(game["seed"])),
                             (game["score"], game["moves"], game["max_tile"]))

    def test_stop_is_reported_as_incomplete(self):
        result = evaluate(FirstLegal(), games=2, stop_flag=lambda: True)
        self.assertFalse(result["integrity"]["complete"])
        self.assertFalse(result["integrity"]["valid"])
        self.assertEqual(result["per_game"], [])

    def test_cli_truncation_is_a_nonzero_exit(self):
        import subprocess
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "evaluate.py"),
                            "--agent", "random", "--games", "1", "--move-limit", "1",
                            "--no-save", "--quiet"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("WARNING", r.stdout)
        self.assertIn("truncated games       1", r.stdout)

    def test_natural_end_at_limit_is_not_truncated(self):
        with patch.object(B, "is_game_over", return_value=True):
            result = evaluate(FirstLegal(), games=1, move_limit=1)
        self.assertEqual(result["integrity"]["truncated_games"], 0)

    def test_positive_limits_required(self):
        for kwargs in ({"games": 0}, {"move_limit": 0}):
            with self.assertRaises(ValueError):
                evaluate(FirstLegal(), **kwargs)


class Values:
    def value(self, b):
        return float((b * 2654435761) % 997)


class TestExplanations(unittest.TestCase):
    def agent(self, depth):
        agent = object.__new__(LearnedAgent)
        agent.net = Values()
        agent.policy = GreedyPolicy(agent.net, RewardFunction({"milestone": 2}), 0.9)
        agent.depth = depth
        agent.prob_cutoff = 1e-3
        agent._dead = 0
        return agent

    def test_explained_and_normal_policies_match_including_milestones(self):
        for depth in (1, 2):
            normal, explained = self.agent(depth), self.agent(depth)
            rng = Random(3)
            board = B.new_game(rng)
            for _ in range(15):
                action = normal.act(board)
                chosen, info = explained.act_with_explanation(board)
                self.assertEqual(chosen, action)
                self.assertEqual(normal.policy.max_exp, explained.policy.max_exp)
                self.assertEqual(info["board_before"], B.board_to_rows(board))
                self.assertEqual(info["selected"], B.ACTION_NAMES[action])
                self.assertEqual(len(info["candidates"]), len(B.legal_actions(board)))
                for c in info["candidates"]:
                    self.assertAlmostEqual(c["total"], c["reward"] + 0.9*c["future_value"])
                    candidate_action = B.ACTION_NAMES.index(c["action"])
                    self.assertEqual(c["afterstate"], B.board_to_rows(B.move(board, candidate_action)[0]))
                self.assertEqual(max(info["candidates"], key=lambda c:c["total"])["action"], info["selected"])
                board = B.random_spawn(B.move(board, chosen)[0], rng)

    def test_explanation_does_not_score_moves_twice(self):
        net = Values()
        with patch.object(net, "value", wraps=net.value) as value:
            policy = GreedyPolicy(net)
            board = B.new_game(Random(12))
            policy.act_with_explanation(board)
            self.assertEqual(value.call_count, len(B.legal_actions(board)))

    def test_stream_attaches_decision_to_correct_frame(self):
        session = AIGameSession(GreedyPolicy(Values()), "test", seed=12)
        with patch("dashboard.games.MAX_FRAMES", 2):
            session._run()
        before, after = session.snapshot(0)["frames"]
        self.assertNotIn("explanation", before)
        self.assertEqual(after["explanation"]["board_before"], before["board"])
        self.assertEqual(after["explanation"]["selected"], after["action"])
        self.assertEqual(after["moves"], 1)


class TestDemo(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.weights = b'\0' * demo.WEIGHT_BYTES
        manifest = {"tuple_set":"8x4", "weight_bytes":demo.WEIGHT_BYTES,
                    "games_trained":10000, "sha256":hashlib.sha256(self.weights).hexdigest(),
                    "config":{"tuple_set":"8x4", "learning":{"alpha":0.1}}}
        (self.bundle / "manifest.json").write_text(json.dumps(manifest))
        (self.bundle / "weights.f32.gz").write_bytes(gzip.compress(self.weights))
        self.patches = [patch.object(demo, "BUNDLE", self.bundle),
                        patch.object(CP, "CHECKPOINT_ROOT", self.root / "checkpoints"),
                        patch.object(CP, "DATA_ROOT", self.root / "data")]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def test_install_and_repeat_preserve_current_training(self):
        result = demo.install()
        self.assertTrue(result["installed"])
        run = CP.Run(demo.RUN_NAME)
        self.assertEqual(run.load_config()["tuple_set"], "8x4")
        self.assertEqual(run.load_meta()["games"], 10000)
        run.weights_path.write_bytes(b"user changed weights")
        self.assertFalse(demo.install()["installed"])
        self.assertEqual(run.weights_path.read_bytes(), b"user changed weights")
        self.assertEqual((run.snapshot_dir / "games-000010000.f32").read_bytes(), self.weights)

    def test_never_overwrites_existing_run(self):
        run = CP.Run(demo.RUN_NAME)
        run.create_dirs()
        run.weights_path.write_bytes(b"precious")
        with self.assertRaises(FileExistsError): demo.install()
        self.assertEqual(run.weights_path.read_bytes(), b"precious")

    def test_corrupt_bundle_does_not_leave_partial_run(self):
        (self.bundle / "weights.f32.gz").write_bytes(gzip.compress(b"bad"))
        with self.assertRaises(ValueError): demo.install()
        self.assertFalse(CP.Run(demo.RUN_NAME).dir.exists())

    def test_missing_optional_bundle_is_clear(self):
        (self.bundle / "weights.f32.gz").unlink()
        self.assertFalse(demo.status()["available"])
        with self.assertRaisesRegex(FileNotFoundError, "release ZIP"): demo.install()

    def test_oversized_payload_rejected(self):
        (self.bundle / "weights.f32.gz").write_bytes(gzip.compress(self.weights+b'x'))
        with self.assertRaises(ValueError): demo.install()

    def test_busy_run_cannot_be_installed(self):
        from training.runlock import RunBusy
        run = CP.Run(demo.RUN_NAME)
        with run.lock(purpose="another job"):
            with self.assertRaises(RunBusy): demo.install()
        self.assertFalse(run.dir.exists())

    def test_checksum_rejects_same_sized_corruption(self):
        (self.bundle / "weights.f32.gz").write_bytes(gzip.compress(b'x' * demo.WEIGHT_BYTES))
        with self.assertRaisesRegex(ValueError, "checksum"): demo.install()
        self.assertFalse(CP.Run(demo.RUN_NAME).dir.exists())

    def test_api_reports_unavailable_bundle(self):
        from dashboard import api
        (self.bundle / "weights.f32.gz").unlink()
        self.assertFalse(api.handle_get("/api/demo", {})["available"])
        with self.assertRaises(api.ApiError) as raised:
            api.handle_post("/api/demo/install", {})
        self.assertEqual(raised.exception.status, 404)
