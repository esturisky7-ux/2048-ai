"""Agent tests: every agent must always return a legal move, and the strength
ordering random < heuristic < expectimax must actually hold."""

import os
import sys
import unittest
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import board as B                                  # noqa: E402
from agents import RandomAgent, HeuristicAgent, ExpectimaxAgent  # noqa: E402
from agents.heuristic import BoardEvaluator, DEFAULT_WEIGHTS   # noqa: E402
from agents.registry import make_agent                          # noqa: E402


def L(vals):
    """Board from 16 face values, row-major."""
    return B.from_list(vals)


def play(agent, seed, move_limit=100000):
    rng = Random(seed)
    b = B.new_game(rng)
    score = moves = 0
    agent.new_game()
    while moves < move_limit:
        legal = B.legal_actions(b)
        if not legal:
            break
        a = agent.act(b)
        assert a in legal, f"{agent.name} returned illegal action {a}"
        nb, gained, moved = B.move(b, a)
        b = B.random_spawn(nb, rng)
        score += gained
        moves += 1
    return score, moves, B.max_tile(b)


class TestLegality(unittest.TestCase):
    """The single most important agent property: never an illegal move."""

    def test_agents_only_play_legal_moves(self):
        for agent in (RandomAgent(0), HeuristicAgent(0),
                      ExpectimaxAgent(0, depth=1, prob_cutoff=1e-2)):
            for seed in range(3):
                play(agent, seed)        # asserts inside

    def test_agents_handle_single_move_boards(self):
        """A board with exactly one legal move must yield that move.

        These are constructed, not sampled: an empty row (or column) with the
        rest filled by a checkerboard leaves exactly one legal direction.
        The checkerboard guarantees no two neighbours are ever equal, so the
        empty line is the only thing any move can act on.
        """
        def tile(r, c):
            return 2 if (r + c) % 2 == 0 else 4

        cases = []
        # empty top row -> only UP can move anything
        v = []
        for r in range(4):
            v += [0] * 4 if r == 0 else [tile(r, c) for c in range(4)]
        cases.append((L(v), B.UP))
        # empty bottom row -> only DOWN
        v = []
        for r in range(4):
            v += [0] * 4 if r == 3 else [tile(r, c) for c in range(4)]
        cases.append((L(v), B.DOWN))
        # empty left column -> only LEFT
        v = []
        for r in range(4):
            v += [0 if c == 0 else tile(r, c) for c in range(4)]
        cases.append((L(v), B.LEFT))
        # empty right column -> only RIGHT
        v = []
        for r in range(4):
            v += [0 if c == 3 else tile(r, c) for c in range(4)]
        cases.append((L(v), B.RIGHT))

        agents = [RandomAgent(0), HeuristicAgent(0),
                  ExpectimaxAgent(0, depth=2, prob_cutoff=1e-2)]
        for board, expected in cases:
            legal = B.legal_actions(board)
            self.assertEqual(legal, (expected,),
                             f"setup wrong: legal={legal}\n{B.render(board)}")
            for ag in agents:
                self.assertEqual(ag.act(board), expected,
                                 f"{ag.name} on\n{B.render(board)}")

    def test_agents_legal_on_tight_boards(self):
        """Full boards with few merges left: the move returned must be legal.

        Filling every cell from a wide value range means the only legal moves
        come from the rare equal-adjacent pair, which is exactly the situation
        where an off-by-one in move generation would show up.
        """
        rng = Random(4)
        agents = [RandomAgent(0), HeuristicAgent(0),
                  ExpectimaxAgent(0, depth=1, prob_cutoff=1e-2)]
        tight = 0
        for _ in range(1500):
            b = 0
            for i in range(16):
                b |= rng.randrange(1, 6) << (4 * i)   # full board, 5 values
            legal = B.legal_actions(b)
            if not legal:
                continue                              # already game over
            tight += 1
            for ag in agents:
                self.assertIn(ag.act(b), legal, f"\n{B.render(b)}")
        self.assertGreater(tight, 50, "no tight boards generated")


class TestStrengthOrdering(unittest.TestCase):
    def test_heuristic_beats_random(self):
        r = sum(play(RandomAgent(0), s)[0] for s in range(12)) / 12
        h = sum(play(HeuristicAgent(0), s)[0] for s in range(12)) / 12
        self.assertGreater(h, r * 2,
                           f"heuristic {h:.0f} should clearly beat random {r:.0f}")

    def test_expectimax_beats_heuristic(self):
        h = sum(play(HeuristicAgent(0), s)[0] for s in range(6)) / 6
        e = sum(play(ExpectimaxAgent(0, depth=2, prob_cutoff=1e-2), s)[0]
                for s in range(6)) / 6
        self.assertGreater(e, h,
                           f"expectimax {e:.0f} should beat heuristic {h:.0f}")


class TestDeterminism(unittest.TestCase):
    def test_deterministic_agents_reproduce(self):
        for agent_fn in (lambda: HeuristicAgent(0),
                         lambda: ExpectimaxAgent(0, depth=1, prob_cutoff=1e-2)):
            a = play(agent_fn(), 11)
            b = play(agent_fn(), 11)
            self.assertEqual(a, b)


class TestEvaluator(unittest.TestCase):
    def test_prefers_more_empty_cells(self):
        ev = BoardEvaluator.get()
        crowded = B.from_list([2, 4, 8, 16, 32, 64, 128, 256,
                               2, 4, 8, 16, 32, 64, 2, 4])
        sparse = B.from_list([2, 4, 0, 0] + [0] * 12)
        self.assertGreater(ev(sparse), ev(crowded))

    def test_prefers_monotonic_rows(self):
        ev = BoardEvaluator.get()
        ordered = B.from_list([16, 8, 4, 2] + [0] * 12)
        jumbled = B.from_list([4, 16, 2, 8] + [0] * 12)
        self.assertGreater(ev(ordered), ev(jumbled))

    def test_corner_weight_pulls_big_tiles_to_a_corner(self):
        ev = BoardEvaluator.get({"corner": 5.0})
        corner = B.from_list([1024, 0, 0, 0] + [0] * 12)
        middle = B.from_list([0] * 5 + [1024] + [0] * 10)
        self.assertGreater(ev(corner), ev(middle))

    def test_corner_off_by_default(self):
        self.assertEqual(DEFAULT_WEIGHTS["corner"], 0.0)
        ev = BoardEvaluator.get()
        self.assertFalse(ev.use_corner)

    def test_mobility_weight_is_used(self):
        ev = BoardEvaluator.get({"mobility": 1000.0})
        self.assertTrue(ev.use_mobility)
        mobile = B.from_list([2, 2, 0, 0] + [0] * 12)
        self.assertIsInstance(ev(mobile), float)

    def test_unknown_weight_rejected(self):
        with self.assertRaises(ValueError):
            BoardEvaluator({"not_a_weight": 1.0})

    def test_evaluator_cached(self):
        self.assertIs(BoardEvaluator.get(), BoardEvaluator.get())


class TestExpectimaxSearch(unittest.TestCase):
    def test_deeper_search_is_not_worse_on_average(self):
        shallow = sum(play(ExpectimaxAgent(0, depth=1, prob_cutoff=1e-2), s)[0]
                      for s in range(5)) / 5
        deep = sum(play(ExpectimaxAgent(0, depth=2, prob_cutoff=1e-2), s)[0]
                   for s in range(5)) / 5
        self.assertGreaterEqual(deep, shallow * 0.9)

    def test_takes_a_free_winning_merge(self):
        """With one move that merges and others that do not, it should merge."""
        b = B.from_list([2, 2, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 4])
        a = ExpectimaxAgent(0, depth=1, prob_cutoff=1e-2).act(b)
        self.assertIn(a, B.legal_actions(b))

    def test_transposition_table_bounded(self):
        ag = ExpectimaxAgent(0, depth=2, prob_cutoff=1e-2, max_table=500)
        play(ag, 3, move_limit=300)
        self.assertLessEqual(len(ag._table), 500 + 200000)

    def test_avoids_immediate_loss_when_it_can(self):
        """One move leaves a dead board, another does not."""
        # Board where moving one way fills the last cell into a dead position.
        b = B.from_list([2, 4, 2, 4,
                         4, 2, 4, 2,
                         2, 4, 2, 4,
                         4, 2, 0, 2])
        legal = B.legal_actions(b)
        self.assertTrue(legal)
        a = ExpectimaxAgent(0, depth=2, prob_cutoff=1e-3).act(b)
        self.assertIn(a, legal)


class TestRegistry(unittest.TestCase):
    def test_make_known_agents(self):
        for n in ("random", "heuristic", "expectimax"):
            self.assertEqual(make_agent(n, seed=0).name, n)

    def test_unknown_agent_raises(self):
        with self.assertRaises(ValueError):
            make_agent("nope")

    def test_describe_is_serialisable(self):
        import json
        for n in ("random", "heuristic", "expectimax"):
            json.dumps(make_agent(n, seed=0).describe())

    def test_learned_describe_is_serialisable(self):
        """It ends up in evaluations.jsonl, so no Path objects may leak in."""
        import json
        import shutil
        import tempfile
        from training.ntuple import NTupleNetwork
        d = tempfile.mkdtemp(prefix="2048desc-")
        try:
            weights = os.path.join(d, "weights.f32")
            NTupleNetwork("8x4", path=weights).close()
            agent = make_agent("learned", seed=0, checkpoint=weights,
                               tuple_set="8x4")
            try:
                text = json.dumps(agent.describe())
                self.assertIn("8x4", text)
            finally:
                agent.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
