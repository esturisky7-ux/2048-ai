"""Tests for the n-tuple network, the reward functions and TD learning."""

import os
import shutil
import sys
import tempfile
import unittest
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import board as B                                      # noqa: E402
from training.ntuple import (NTupleNetwork, TUPLE_SETS,            # noqa: E402
                             symmetry_permutations)
from training.learner import TDLearner                             # noqa: E402
from training.reward import RewardFunction, DEFAULT_REWARD         # noqa: E402

SMALL = "8x4"      # 2 MB: fast to build, fine for correctness


class TestSymmetries(unittest.TestCase):
    def test_eight_distinct_permutations(self):
        perms = symmetry_permutations()
        self.assertEqual(len(perms), 8)
        self.assertEqual(len({tuple(p) for p in perms}), 8)

    def test_each_is_a_bijection(self):
        for p in symmetry_permutations():
            self.assertEqual(sorted(p), list(range(16)))

    def test_group_is_closed_under_composition(self):
        perms = [tuple(p) for p in symmetry_permutations()]
        s = set(perms)
        for a in perms:
            for b in perms:
                self.assertIn(tuple(a[b[i]] for i in range(16)), s)


class TestNTupleNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = NTupleNetwork(SMALL)
        rng = Random(1)
        for _ in range(4000):
            cls.net.weights[rng.randrange(cls.net.n_weights)] = rng.uniform(-3, 3)

    @classmethod
    def tearDownClass(cls):
        cls.net.close()

    def test_generated_value_matches_reference(self):
        rng = Random(2)
        for _ in range(1500):
            b = rng.getrandbits(64)
            self.assertAlmostEqual(self.net.value(b),
                                   self.net.value_reference(b), places=3)

    def test_indices_in_range(self):
        rng = Random(3)
        for _ in range(500):
            b = rng.getrandbits(64)
            for i in self.net.indices(b):
                self.assertTrue(0 <= i < self.net.n_weights)

    def test_activates_exactly_n_variants(self):
        self.assertEqual(len(self.net.indices(0)), self.net.n_variants)
        self.assertEqual(self.net.n_variants, len(self.net.tuples) * 8)

    def test_update_moves_value_by_expected_amount(self):
        """Value change equals d * sum(k^2) over the activated index multiset.

        Not d * 64: when a board has symmetry, several of the 64 variants land
        on the *same* weight, so that weight is incremented once per variant
        AND read once per variant. That is the correct gradient step for a
        linear model whose feature value is an activation count, but it means
        the naive "d times the number of lookups" is wrong.
        """
        from collections import Counter
        rng = Random(5)
        for _ in range(60):
            b = rng.getrandbits(64)
            counts = Counter(self.net.indices(b))
            expected = 0.25 * sum(k * k for k in counts.values())
            before = self.net.value(b)
            self.net.update(b, 0.25)
            self.assertAlmostEqual(self.net.value(b) - before, expected,
                                   places=2)
            self.net.update(b, -0.25)

    def test_weight_sharing_multiplicity_stays_mild_on_real_boards(self):
        """Effective step size is alpha * sum(k^2)/N; keep an eye on it.

        A maximally symmetric board (a perfect checkerboard) collapses the 64
        lookups onto 16 weights, giving a 4x effective learning rate. Boards
        that actually occur in play are nowhere near that, so the default
        alpha of 0.1 is comfortably stable -- but a very large alpha combined
        with a symmetric position is the one place this scheme could diverge.
        """
        from collections import Counter

        def ratio(board):
            c = Counter(self.net.indices(board))
            return sum(k * k for k in c.values()) / self.net.n_variants

        checker = B.from_list([2, 4, 2, 4, 4, 2, 4, 2,
                               2, 4, 2, 4, 4, 2, 4, 2])
        self.assertAlmostEqual(ratio(checker), 4.0, places=6)

        realistic = B.from_list([1024, 512, 64, 8, 256, 128, 32, 4,
                                 16, 8, 4, 2, 8, 4, 2, 0])
        self.assertLess(ratio(realistic), 1.5)

        rng = Random(12)
        worst = max(ratio(rng.getrandbits(64)) for _ in range(400))
        self.assertLess(worst, 2.0)

    def test_update_scaled_equals_update(self):
        rng = Random(6)
        b = rng.getrandbits(64)
        v0 = self.net.value(b)
        self.net.update_scaled(b, 2.0, 0.1)      # d = 0.2
        v1 = self.net.value(b)
        self.net.update(b, -0.2)
        v2 = self.net.value(b)
        self.assertAlmostEqual(v0, v2, places=3)
        self.assertGreater(v1, v0)

    def test_symmetric_boards_share_value(self):
        """Weight sharing means a board and its rotation evaluate the same."""
        net = NTupleNetwork(SMALL)
        rng = Random(7)
        for _ in range(300):
            net.weights[rng.randrange(net.n_weights)] = rng.uniform(-1, 1)
        rng2 = Random(8)
        for _ in range(60):
            b = rng2.getrandbits(64)
            self.assertAlmostEqual(net.value(b),
                                   net.value(B.transpose(b)), places=3)
        net.close()

    def test_all_tuple_sets_build_and_agree(self):
        for name in TUPLE_SETS:
            if name == "4x6":
                continue          # 268 MB; covered by the running default run
            net = NTupleNetwork(name)
            rng = Random(9)
            for _ in range(300):
                net.weights[rng.randrange(net.n_weights)] = rng.uniform(-1, 1)
            for _ in range(200):
                b = rng.getrandbits(64)
                self.assertAlmostEqual(net.value(b), net.value_reference(b),
                                       places=3, msg=name)
            net.close()

    def test_unknown_tuple_set_rejected(self):
        with self.assertRaises(ValueError):
            NTupleNetwork("nonesuch")


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="2048test-")
        self.path = os.path.join(self.dir, "w.f32")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_weights_survive_close_and_reopen(self):
        net = NTupleNetwork(SMALL, path=self.path)
        rng = Random(11)
        boards = [rng.getrandbits(64) for _ in range(40)]
        for b in boards:
            net.update(b, 0.5)
        vals = [net.value(b) for b in boards]
        net.flush()
        net.close()

        net2 = NTupleNetwork(SMALL, path=self.path)
        self.assertEqual([net2.value(b) for b in boards], vals)
        net2.close()

    def test_file_is_created_with_the_right_size(self):
        net = NTupleNetwork(SMALL, path=self.path)
        self.assertEqual(os.path.getsize(self.path), 4 * net.n_weights)
        net.close()

    def test_readonly_map_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            NTupleNetwork(SMALL, path=os.path.join(self.dir, "nope.f32"),
                          readonly=True)

    def test_two_maps_share_one_file(self):
        """This is what makes --workers 2 work without any IPC."""
        a = NTupleNetwork(SMALL, path=self.path)
        b_net = NTupleNetwork(SMALL, path=self.path)
        board = 0x123456789ABCDEF
        a.update(board, 1.0)
        self.assertAlmostEqual(b_net.value(board), a.value(board), places=3)
        a.close()
        b_net.close()


class TestReward(unittest.TestCase):
    def test_default_is_pure_merge_score(self):
        rf = RewardFunction()
        self.assertTrue(rf.is_pure_score)
        for k, v in DEFAULT_REWARD.items():
            if k not in ("score", "gamma"):
                self.assertEqual(v, 0.0, f"{k} should default to 0")

    def test_pure_score_returns_merge_score(self):
        rf = RewardFunction()
        r, _ = rf.step(0, 0, 137, 0)
        self.assertEqual(r, 137)
        self.assertEqual(rf.terminal(), 0.0)

    def test_milestone_paid_once_per_new_maximum(self):
        rf = RewardFunction({"milestone": 1.0})
        board = B.from_list([1024] + [0] * 15)
        r1, m1 = rf.step(0, board, 0, 0)
        self.assertEqual(r1, 1024)
        self.assertEqual(m1, 10)
        r2, m2 = rf.step(0, board, 0, m1)     # same tile again: no second payout
        self.assertEqual(r2, 0)
        self.assertEqual(m2, 10)

    def test_game_over_penalty(self):
        rf = RewardFunction({"game_over_penalty": 250.0})
        self.assertEqual(rf.terminal(), -250.0)

    def test_empty_potential_is_a_difference(self):
        """Potential shaping must telescope: a round trip nets out to zero."""
        rf = RewardFunction({"empty_potential": 3.0, "gamma": 1.0})
        s = B.from_list([2, 2, 0, 0] + [0] * 12)      # 14 empty
        a = B.from_list([4, 0, 0, 0] + [0] * 12)      # 15 empty
        fwd, _ = rf.step(s, a, 0, 0)
        back, _ = rf.step(a, s, 0, 0)
        self.assertAlmostEqual(fwd, 3.0)
        self.assertAlmostEqual(back, -3.0)
        self.assertAlmostEqual(fwd + back, 0.0)

    def test_survival_is_flat_and_farmable(self):
        rf = RewardFunction({"score": 0.0, "survival": 5.0})
        r, _ = rf.step(0, 0, 0, 0)
        self.assertEqual(r, 5.0)     # paid even with no merge at all

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            RewardFunction({"bogus": 1.0})


class TestTDLearner(unittest.TestCase):
    def setUp(self):
        self.net = NTupleNetwork(SMALL)
        self.learner = TDLearner(self.net, alpha=0.1)

    def tearDown(self):
        self.net.close()

    def test_alpha_is_divided_across_activated_weights(self):
        self.assertAlmostEqual(self.learner.alpha_per_weight,
                               0.1 / self.net.n_variants)

    def test_play_game_returns_sane_results(self):
        score, moves, tile = self.learner.play_game(Random(1))
        self.assertGreater(moves, 0)
        self.assertGreaterEqual(score, 0)
        self.assertIn(tile, [1 << i for i in range(1, 16)])

    def test_learning_changes_weights(self):
        before = sum(self.net.weights[i] for i in range(0, self.net.n_weights, 997))
        for s in range(15):
            self.learner.play_game(Random(s))
        after = sum(self.net.weights[i] for i in range(0, self.net.n_weights, 997))
        self.assertNotAlmostEqual(before, after)

    def test_learn_false_leaves_weights_untouched(self):
        for s in range(6):
            self.learner.play_game(Random(s))
        snap = [self.net.weights[i]
                for i in range(0, self.net.n_weights, 1301)]
        for s in range(20, 26):
            self.learner.play_game(Random(s), learn=False)
        after = [self.net.weights[i]
                 for i in range(0, self.net.n_weights, 1301)]
        self.assertEqual(snap, after)

    def test_td_update_converges_towards_its_target(self):
        """Repeated updates must walk the value to the target, not past it.

        Uses the default alpha on a realistic board -- the operating regime
        the trainer actually runs in.
        """
        net = NTupleNetwork(SMALL)
        learner = TDLearner(net, alpha=0.1)
        board = B.from_list([1024, 512, 64, 8, 256, 128, 32, 4,
                             16, 8, 4, 2, 8, 4, 2, 0])
        net.update(board, 100.0)              # start badly over-valued
        target = 0.0
        prev = abs(net.value(board) - target)
        self.assertGreater(prev, 0)
        # Each step multiplies the error by (1 - alpha*sum(k^2)/N) ~= 0.887,
        # so ~150 steps takes a starting error of ~7200 down below 1.
        for _ in range(150):
            err = target - net.value(board)
            learner.net.update_scaled(board, err, learner.alpha_per_weight)
            now = abs(net.value(board) - target)
            self.assertLess(now, prev)        # strictly monotone, no overshoot
            prev = now
        self.assertLess(prev, 1.0)            # and it actually gets there
        net.close()

    def test_terminal_state_value_is_pushed_down_by_real_training(self):
        """After training, positions one move from death score below live ones."""
        net = NTupleNetwork("4x5")
        learner = TDLearner(net, alpha=0.1)
        rng = Random(3)
        for _ in range(120):
            learner.play_game(rng)
        dead_ish = B.from_list([2, 4, 2, 4, 4, 2, 4, 2,
                                2, 4, 2, 4, 4, 2, 4, 8])
        healthy = B.from_list([1024, 512, 256, 128, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0])
        self.assertLess(net.value(dead_ish), net.value(healthy))
        net.close()

    def test_best_action_is_legal(self):
        rng = Random(3)
        for _ in range(300):
            b = 0
            for i in range(16):
                b |= rng.randrange(0, 4) << (4 * i)
            legal = B.legal_actions(b)
            if not legal:
                continue
            self.assertIn(self.learner.best_action(b), legal)

    def test_learning_improves_play(self):
        """The headline claim: TD learning actually makes the agent better."""
        net = NTupleNetwork("4x5")
        learner = TDLearner(net, alpha=0.1)
        rng = Random(42)
        first = [learner.play_game(rng)[0] for _ in range(40)]
        for _ in range(260):
            learner.play_game(rng)
        last = [learner.play_game(rng)[0] for _ in range(40)]
        net.close()
        m0, m1 = sum(first) / len(first), sum(last) / len(last)
        self.assertGreater(m1, m0 * 1.5,
                           f"mean score went {m0:.0f} -> {m1:.0f}; "
                           f"expected clear improvement")

    def test_epsilon_still_plays_legal_moves(self):
        learner = TDLearner(self.net, alpha=0.1, epsilon=0.5)
        score, moves, tile = learner.play_game(Random(9))
        self.assertGreater(moves, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
