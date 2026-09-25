"""The trained policy is one policy, wherever it is played.

A learned value function only describes the policy it was trained with:
``argmax r(a) + gamma * V(afterstate(a))`` with the run's own reward -- shaping
terms, milestone bonuses and discount included. Training, the trainer's
periodic evaluation, experiments and the loaded agent must all score moves
that way, or an evaluation measures a policy nobody trained.

The decision tests use a mock network whose values are set by hand, so each
reward term's effect on the choice is visible. The drift tests play whole
games both ways on real (random) weights and demand identical results.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from random import Random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import board as B                                  # noqa: E402
from evaluation.evaluator import play_one                      # noqa: E402
from training import checkpoint as CP                          # noqa: E402
from training.learner import GreedyPolicy, TDLearner, best_move  # noqa: E402
from training.ntuple import NTupleNetwork                      # noqa: E402
from training.reward import RewardFunction                     # noqa: E402
from training.trainer import PolicyAgent                       # noqa: E402

UP, RIGHT, DOWN, LEFT = B.UP, B.RIGHT, B.DOWN, B.LEFT

# Two 2s side by side in the top row: LEFT and RIGHT merge them into a 4 (and
# score 4); DOWN only slides them to the bottom row; UP does nothing.
BOARD = B.from_list([2, 2, 0, 0] + [0] * 12)
AFTER = {a: B.move(BOARD, a)[0] for a in (RIGHT, DOWN, LEFT)}


class MockNet:
    """A value function set by hand: ``values[afterstate]``, else 0."""

    n_variants = 1

    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, b):
        return self.values.get(b, 0.0)

    def update_scaled(self, b, err, alpha):
        pass


class HashNet(MockNet):
    """Arbitrary but fixed values for every board, so that whole games take
    varied decisions without training anything."""

    def value(self, b):
        return float((b * 2654435761) % 997)


def choice(reward=None, gamma=1.0, values=None, max_exp=1):
    net = MockNet(values)
    return best_move(BOARD, max_exp, net.value, gamma,
                     RewardFunction(reward))[1][0]


class TestScorerDecisions(unittest.TestCase):
    """Each reward term, alone, changes the move exactly as training would."""

    def test_default_is_merge_score_plus_value(self):
        # LEFT and RIGHT score 4 now; DOWN is worth more later.
        self.assertEqual(choice(values={AFTER[DOWN]: 10.0}), DOWN)
        self.assertEqual(choice(values={AFTER[DOWN]: 3.0}), RIGHT)

    def test_gamma_discounts_the_future(self):
        values = {AFTER[DOWN]: 10.0}
        self.assertEqual(choice(values=values), DOWN)
        self.assertEqual(choice(values=values, gamma=0.3), RIGHT)

    def test_empty_potential_rewards_freeing_cells(self):
        values = {AFTER[DOWN]: 10.0}
        self.assertEqual(choice(values=values), DOWN)
        # Merging frees a cell: +10 per empty cell outweighs DOWN's value.
        self.assertEqual(choice(values=values,
                                reward={"empty_potential": 10.0}), RIGHT)

    def test_milestone_pays_for_a_new_largest_tile(self):
        values = {AFTER[DOWN]: 10.0}
        self.assertEqual(choice(values=values), DOWN)
        # Largest tile so far is a 2 (exponent 1): making a 4 pays 5 * 4.
        self.assertEqual(choice(values=values, reward={"milestone": 5.0},
                                max_exp=1), RIGHT)
        # Once the game has made a 4, making another pays nothing.
        self.assertEqual(choice(values=values, reward={"milestone": 5.0},
                                max_exp=2), DOWN)

    def test_survival_is_a_constant_unless_the_score_weight_changes(self):
        values = {AFTER[DOWN]: 3.0}
        self.assertEqual(choice(values=values), RIGHT)
        self.assertEqual(choice(values=values, reward={"survival": 20.0}),
                         RIGHT)
        self.assertEqual(choice(values=values, reward={"score": 0.1,
                                                       "survival": 20.0}),
                         DOWN)

    def test_no_legal_move(self):
        dead = B.from_list([2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2])
        self.assertEqual(best_move(dead, 0, MockNet().value, 1.0,
                                   RewardFunction()), (float("-inf"), None))
        self.assertEqual(GreedyPolicy(MockNet()).act(dead), -1)


class TestPolicyState(unittest.TestCase):
    def test_milestones_are_tracked_through_a_game_as_training_does(self):
        policy = GreedyPolicy(MockNet({AFTER[DOWN]: 10.0}),
                              RewardFunction({"milestone": 5.0}))
        policy.new_game()
        self.assertEqual(policy.max_exp, 0)
        self.assertEqual(policy.act(BOARD), RIGHT)      # makes the first 4
        self.assertEqual(policy.max_exp, 2)
        self.assertEqual(policy.act(BOARD), DOWN)       # a second 4 is no news
        policy.new_game()
        self.assertEqual(policy.max_exp, 0)

    def test_pure_score_never_needs_the_state(self):
        policy = GreedyPolicy(MockNet({AFTER[DOWN]: 10.0}))
        self.assertEqual(policy.act(BOARD), DOWN)
        self.assertEqual(policy.max_exp, 0)

    def test_best_action_is_the_same_scorer(self):
        learner = TDLearner(MockNet({AFTER[DOWN]: 10.0}),
                            RewardFunction({"empty_potential": 10.0}))
        self.assertEqual(learner.best_action(BOARD), RIGHT)
        self.assertEqual(TDLearner(MockNet({AFTER[DOWN]: 10.0}), gamma=0.3)
                         .best_action(BOARD), RIGHT)


CONFIGS = [
    ("default", None, 1.0),
    ("gamma", None, 0.7),
    ("empty potential", {"empty_potential": 6.0, "gamma": 0.9}, 0.9),
    ("milestone", {"milestone": 0.8}, 1.0),
    ("survival", {"score": 0.1, "survival": 20.0}, 1.0),
    ("everything", {"score": 0.5, "milestone": 0.3, "empty_potential": 2.0,
                    "survival": 1.0, "game_over_penalty": 50.0}, 0.95),
]


class Sandboxed(unittest.TestCase):
    """Runs live in a temporary directory, so an agent never picks up the
    configuration of a real run by accident."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="2048policy-")
        cls._roots = CP.CHECKPOINT_ROOT, CP.DATA_ROOT
        CP.CHECKPOINT_ROOT = Path(cls.tmp) / "checkpoints"
        CP.DATA_ROOT = Path(cls.tmp) / "data"

    @classmethod
    def tearDownClass(cls):
        CP.CHECKPOINT_ROOT, CP.DATA_ROOT = cls._roots
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestNoDrift(Sandboxed):
    """Training's own games and every evaluation path agree, game by game."""

    def test_training_and_its_evaluation_play_the_same_games(self):
        """play_game(learn=False) is how training plays; the policy is how the
        trainer's evaluation and an experiment's evaluation play."""
        for name, reward, gamma in CONFIGS:
            with self.subTest(config=name):
                learner = TDLearner(HashNet(), RewardFunction(reward),
                                    gamma=gamma)
                agent = PolicyAgent(learner.policy(), 0)
                for seed in range(4):
                    self.assertEqual(learner.play_game(Random(seed),
                                                       learn=False),
                                     play_one(agent, seed))

    def test_a_loaded_agent_plays_the_runs_own_objective(self):
        """LearnedAgent reads reward and gamma from the run's config."""
        from agents.learned import LearnedAgent
        for i, (name, reward, gamma) in enumerate(CONFIGS):
            with self.subTest(config=name):
                run = CP.Run(f"policy-{i}")
                run.create_dirs()
                net = NTupleNetwork("8x4", path=str(run.weights_path))
                rng = Random(i)
                for _ in range(60000):
                    net.weights[rng.randrange(net.n_weights)] = \
                        rng.uniform(-40, 40)
                net.flush()
                cfg = {"tuple_set": "8x4", "learning": {"gamma": gamma}}
                if reward:
                    cfg["reward"] = reward
                run.save_config(cfg)
                run.save_meta({"games": 1, "tuple_set": "8x4"})
                learner = TDLearner(net, RewardFunction(reward), gamma=gamma)
                agent = LearnedAgent(run=run.name)
                try:
                    self.assertEqual(agent.describe()["gamma"], gamma)
                    for seed in range(3):
                        self.assertEqual(
                            learner.play_game(Random(seed), learn=False),
                            play_one(agent, seed))
                finally:
                    agent.close()
                    net.close()


class TestSearchOnTop(Sandboxed):
    """Depth > 1 searches the training objective at every ply."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.path = os.path.join(cls.tmp, "weights.f32")
        net = NTupleNetwork("8x4", path=cls.path)
        rng = Random(3)
        for _ in range(60000):
            net.weights[rng.randrange(net.n_weights)] = rng.uniform(-40, 40)
        net.flush()
        net.close()
        rng = Random(11)
        cls.boards = []
        while len(cls.boards) < 12:
            b = B.new_game(rng)
            for _ in range(rng.randrange(3, 60)):
                legal = B.legal_actions(b)
                if not legal:
                    break
                b = B.random_spawn(B.move(b, rng.choice(legal))[0], rng)
            if B.legal_actions(b):
                cls.boards.append(b)

    def agent(self, **kw):
        from agents.learned import LearnedAgent
        return LearnedAgent(checkpoint=self.path, tuple_set="8x4", depth=2,
                            prob_cutoff=1e-3, **kw)

    @staticmethod
    def reference(agent, board, reward, gamma, legacy=False):
        """Root move by a from-scratch depth-2 search.

        ``legacy`` is the search as it was: raw merge score between plies,
        undiscounted, whatever the run was trained on.
        """
        rf = RewardFunction(reward)
        value = agent.net.value

        def r_of(b, nb, gained, mx):
            if legacy or rf.is_pure_score:
                return gained, mx
            return rf.step(b, nb, gained, mx)

        g = 1.0 if legacy else gamma

        def chance(after, depth, prob, mx):
            if depth <= 0 or prob < 1e-3:
                return value(after)
            cells = B.spawn_positions(after)
            total = 0.0
            for c in cells:
                for tile, p in ((1, 0.9), (2, 0.1)):
                    p /= len(cells)
                    total += p * best(after | (tile << (4 * c)), depth,
                                      prob * p, mx)
            return total

        def best(b, depth, prob, mx):
            vals = []
            for a in (0, 1, 2, 3):
                nb, gained, moved = B.move(b, a)
                if moved:
                    r, mx2 = r_of(b, nb, gained, mx)
                    vals.append(r + g * chance(nb, depth - 1, prob, mx2))
            return max(vals) if vals else (0.0 if legacy else
                                           (rf.terminal() or 0.0))

        scores = []
        for a in (0, 1, 2, 3):
            nb, gained, moved = B.move(board, a)
            if moved:
                r, mx = r_of(board, nb, gained, 0)
                scores.append((r + g * chance(nb, 1, 1.0, mx), -a, a))
        return max(scores)[2]

    def test_default_search_is_unchanged(self):
        agent = self.agent()
        try:
            for b in self.boards:
                agent.new_game()
                self.assertEqual(agent.act(b),
                                 self.reference(agent, b, None, 1.0,
                                                legacy=True))
        finally:
            agent.close()

    def test_search_optimises_the_trained_objective(self):
        reward = {"milestone": 2.0, "empty_potential": 8.0, "gamma": 0.6}
        agent = self.agent(reward=reward, gamma=0.6)
        differs = 0
        try:
            for b in self.boards:
                agent.new_game()
                got = agent.act(b)
                self.assertEqual(got, self.reference(agent, b, reward, 0.6))
                differs += got != self.reference(agent, b, reward, 0.6,
                                                 legacy=True)
        finally:
            agent.close()
        self.assertGreater(differs, 0, "these boards cannot tell the two "
                                       "objectives apart; pick others")


class TestDescribe(Sandboxed):
    def test_results_record_the_objective(self):
        from agents.learned import LearnedAgent
        path = os.path.join(self.tmp, "weights.f32")
        NTupleNetwork("8x4", path=path).close()
        agent = LearnedAgent(checkpoint=path, tuple_set="8x4",
                             reward={"milestone": 1.0}, gamma=0.9)
        try:
            desc = json.loads(json.dumps(agent.describe()))
        finally:
            agent.close()
        self.assertEqual(desc["gamma"], 0.9)
        self.assertEqual(desc["reward"]["milestone"], 1.0)
        self.assertIsNone(agent.live_run, "a file outside every run is "
                                          "never live")


if __name__ == "__main__":
    unittest.main(verbosity=2)
