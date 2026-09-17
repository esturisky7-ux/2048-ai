"""Engine correctness tests.

The merge rules are the part of 2048 that implementations most often get
wrong, so they are tested exhaustively rather than by example alone:
``test_all_rows_match_reference`` checks all 65536 possible rows against an
independent reference implementation.
"""

import os
import sys
import unittest
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import board as B          # noqa: E402
from engine.game import Game           # noqa: E402


def L(vals):
    """Build a board from 16 face values, row-major."""
    return B.from_list(vals)


def row(vals):
    """Build a board whose top row is ``vals`` and the rest empty."""
    return L(list(vals) + [0] * 12)


def top_row(b):
    return B.to_list(b)[:4]


class TestMergeRules(unittest.TestCase):
    def test_2222_left(self):
        # 2 2 2 2  ->  4 4 . .   (two separate merges, score 4 + 4)
        nb, sc, moved = B.move(row([2, 2, 2, 2]), B.LEFT)
        self.assertEqual(top_row(nb), [4, 4, 0, 0])
        self.assertEqual(sc, 8)
        self.assertTrue(moved)

    def test_2222_right(self):
        nb, sc, _ = B.move(row([2, 2, 2, 2]), B.RIGHT)
        self.assertEqual(top_row(nb), [0, 0, 4, 4])
        self.assertEqual(sc, 8)

    def test_4488_left(self):
        # 4 4 8 8  ->  8 16 . .   (score 8 + 16)
        nb, sc, _ = B.move(row([4, 4, 8, 8]), B.LEFT)
        self.assertEqual(top_row(nb), [8, 16, 0, 0])
        self.assertEqual(sc, 24)

    def test_no_double_merge(self):
        # 4 4 8 . must give 8 8 . . -- the new 8 must NOT merge with the old 8.
        nb, sc, _ = B.move(row([4, 4, 8, 0]), B.LEFT)
        self.assertEqual(top_row(nb), [8, 8, 0, 0])
        self.assertEqual(sc, 8)

    def test_no_double_merge_chain(self):
        # 2 2 4 . -> 4 4 . . and NOT 8 . . .
        nb, sc, _ = B.move(row([2, 2, 4, 0]), B.LEFT)
        self.assertEqual(top_row(nb), [4, 4, 0, 0])
        self.assertEqual(sc, 4)

    def test_merge_pairs_from_the_moving_edge(self):
        # Sliding left, the leftmost eligible pair merges first:
        # 2 2 2 . -> 4 2 . .
        nb, sc, _ = B.move(row([2, 2, 2, 0]), B.LEFT)
        self.assertEqual(top_row(nb), [4, 2, 0, 0])
        self.assertEqual(sc, 4)
        # Sliding right, the rightmost pair merges first: . 2 2 2 -> . . 2 4
        nb, sc, _ = B.move(row([0, 2, 2, 2]), B.RIGHT)
        self.assertEqual(top_row(nb), [0, 0, 2, 4])
        self.assertEqual(sc, 4)

    def test_gaps_are_compacted(self):
        nb, sc, _ = B.move(row([2, 0, 0, 2]), B.LEFT)
        self.assertEqual(top_row(nb), [4, 0, 0, 0])
        self.assertEqual(sc, 4)
        nb, sc, _ = B.move(row([0, 0, 0, 8]), B.LEFT)
        self.assertEqual(top_row(nb), [8, 0, 0, 0])
        self.assertEqual(sc, 0)

    def test_unequal_neighbours_do_not_merge(self):
        nb, sc, _ = B.move(row([2, 4, 2, 4]), B.LEFT)
        self.assertEqual(top_row(nb), [2, 4, 2, 4])
        self.assertEqual(sc, 0)

    def test_no_move_is_reported(self):
        b = row([2, 4, 8, 16])
        nb, sc, moved = B.move(b, B.LEFT)
        self.assertFalse(moved)
        self.assertEqual(nb, b)
        self.assertEqual(sc, 0)

    def test_32768_cannot_merge(self):
        # 15 is the largest exponent a nibble holds; two 32768s must not merge.
        b = row([32768, 32768, 0, 0])
        nb, sc, moved = B.move(b, B.LEFT)
        self.assertEqual(top_row(nb), [32768, 32768, 0, 0])
        self.assertEqual(sc, 0)
        self.assertFalse(moved)


class TestReferenceImplementation(unittest.TestCase):
    """Cross-check the packed tables against a naive list implementation."""

    @staticmethod
    def ref_slide_left(line):
        """Independent reference: compact, then merge adjacent equal pairs."""
        vals = [v for v in line if v]
        out, score, i = [], 0, 0
        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1] and vals[i] < 15:
                out.append(vals[i] + 1)
                score += 1 << (vals[i] + 1)
                i += 2
            else:
                out.append(vals[i])
                i += 1
        out += [0] * (4 - len(out))
        return out, score

    def test_all_rows_match_reference(self):
        for r in range(65536):
            line = [(r >> (4 * c)) & 0xF for c in range(4)]
            exp, exp_sc = self.ref_slide_left(line)
            got = [(B._ROW_LEFT[r] >> (4 * c)) & 0xF for c in range(4)]
            self.assertEqual(got, exp, f"row {r:04x} slide-left mismatch")
            self.assertEqual(B._SCORE_LEFT[r], exp_sc, f"row {r:04x} score")

            rev = line[::-1]
            rexp, rexp_sc = self.ref_slide_left(rev)
            rexp = rexp[::-1]
            rgot = [(B._ROW_RIGHT[r] >> (4 * c)) & 0xF for c in range(4)]
            self.assertEqual(rgot, rexp, f"row {r:04x} slide-right mismatch")
            self.assertEqual(B._SCORE_RIGHT[r], rexp_sc)

    def test_tile_conservation(self):
        """A slide never creates or destroys tile mass (sum of face values)."""
        rng = Random(1234)
        for _ in range(3000):
            b = 0
            for i in range(16):
                if rng.random() < 0.7:
                    b |= rng.randrange(1, 12) << (4 * i)
            before = sum(B.to_list(b))
            for a in B.ACTIONS:
                nb, _, _ = B.move(b, a)
                self.assertEqual(sum(B.to_list(nb)), before)

    def test_score_equals_merged_tile_values(self):
        """Every merge removes exactly one tile and scores the tile it makes."""
        rng = Random(99)
        for _ in range(2000):
            b = 0
            for i in range(16):
                if rng.random() < 0.6:
                    b |= rng.randrange(1, 11) << (4 * i)
            for a in B.ACTIONS:
                nb, sc, moved = B.move(b, a)
                if not moved:
                    continue
                before = sorted(v for v in B.to_list(b) if v)
                after = sorted(v for v in B.to_list(nb) if v)
                n_merges = len(before) - len(after)
                self.assertGreaterEqual(n_merges, 0)
                # No merges <=> no score.
                self.assertEqual(n_merges == 0, sc == 0)
                # Score is the total face value of the tiles merging created,
                # which must be at least 4 per merge and at most the whole
                # board's mass.
                self.assertGreaterEqual(sc, 4 * n_merges)
                self.assertLessEqual(sc, sum(before))
                # Tile mass is conserved, so the multiset difference is only
                # the merged pairs collapsing into doubled tiles.
                self.assertEqual(sum(before), sum(after))


class TestTranspose(unittest.TestCase):
    def test_against_naive(self):
        rng = Random(7)
        for _ in range(2000):
            b = rng.getrandbits(64)
            flat = B.to_list(b)
            naive = [flat[4 * (i % 4) + i // 4] for i in range(16)]
            self.assertEqual(B.to_list(B.transpose(b)), naive)

    def test_involution(self):
        rng = Random(8)
        for _ in range(500):
            b = rng.getrandbits(64)
            self.assertEqual(B.transpose(B.transpose(b)), b)


class TestDirections(unittest.TestCase):
    def test_up_is_left_on_columns(self):
        # Left column 2,2,2,2 sliding up must give 4,4,.,.
        b = L([2, 0, 0, 0,
               2, 0, 0, 0,
               2, 0, 0, 0,
               2, 0, 0, 0])
        nb, sc, _ = B.move(b, B.UP)
        col = B.to_list(nb)[0::4]
        self.assertEqual(col, [4, 4, 0, 0])
        self.assertEqual(sc, 8)

    def test_down(self):
        b = L([4, 0, 0, 0,
               4, 0, 0, 0,
               8, 0, 0, 0,
               8, 0, 0, 0])
        nb, sc, _ = B.move(b, B.DOWN)
        col = B.to_list(nb)[0::4]
        self.assertEqual(col, [0, 0, 8, 16])
        self.assertEqual(sc, 24)

    def test_up_down_match_transposed_left_right(self):
        rng = Random(21)
        for _ in range(2000):
            b = rng.getrandbits(64)
            up = B.move_no_score(b, B.UP)
            self.assertEqual(up, B.transpose(
                B.move_no_score(B.transpose(b), B.LEFT)))
            dn = B.move_no_score(b, B.DOWN)
            self.assertEqual(dn, B.transpose(
                B.move_no_score(B.transpose(b), B.RIGHT)))


class TestQueries(unittest.TestCase):
    def test_empty_count(self):
        self.assertEqual(B.empty_count(0), 16)
        self.assertEqual(B.empty_count(row([2, 2, 2, 2])), 12)
        full = L([2, 4, 8, 16] * 4)
        self.assertEqual(B.empty_count(full), 0)

    def test_empty_count_matches_naive(self):
        rng = Random(3)
        for _ in range(2000):
            b = rng.getrandbits(64)
            self.assertEqual(B.empty_count(b),
                             sum(1 for v in B.to_list(b) if v == 0))

    def test_max_tile(self):
        self.assertEqual(B.max_tile(0), 0)
        self.assertEqual(B.max_tile(row([2, 4, 8, 16])), 16)
        self.assertEqual(B.max_tile(L([0] * 15 + [2048])), 2048)

    def test_game_over_detection(self):
        # Checkerboard of alternating values: full and no equal neighbours.
        dead = L([2, 4, 2, 4,
                  4, 2, 4, 2,
                  2, 4, 2, 4,
                  4, 2, 4, 2])
        self.assertTrue(B.is_game_over(dead))
        self.assertEqual(B.legal_actions(dead), ())

        # Same board but with one merge available horizontally.
        alive = L([2, 2, 4, 8,
                   4, 8, 2, 4,
                   2, 4, 8, 2,
                   4, 2, 4, 8])
        self.assertFalse(B.is_game_over(alive))

        # Vertical-only merge: the transposed check must catch it.
        vert = L([2, 4, 2, 4,
                  2, 2, 4, 2,
                  4, 4, 2, 4,
                  2, 2, 4, 2])
        self.assertFalse(B.is_game_over(vert))

        self.assertFalse(B.is_game_over(0))

    def test_game_over_matches_legal_actions(self):
        rng = Random(11)
        for _ in range(4000):
            b = 0
            for i in range(16):
                b |= rng.randrange(0, 5) << (4 * i)
            self.assertEqual(B.is_game_over(b), len(B.legal_actions(b)) == 0)

    def test_legal_actions(self):
        b = row([2, 2, 0, 0])
        acts = set(B.legal_actions(b))
        self.assertIn(B.LEFT, acts)
        self.assertIn(B.RIGHT, acts)
        self.assertIn(B.DOWN, acts)
        self.assertNotIn(B.UP, acts)   # already at the top, nothing to merge


class TestSpawning(unittest.TestCase):
    def test_new_game_has_two_tiles(self):
        for s in range(50):
            b = B.new_game(Random(s))
            tiles = [v for v in B.to_list(b) if v]
            self.assertEqual(len(tiles), 2)
            self.assertTrue(all(v in (2, 4) for v in tiles))

    def test_spawn_only_fills_empty_cells(self):
        rng = Random(5)
        b = L([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 0, 0, 0, 0, 0])
        occupied_before = [v for v in B.to_list(b)[:11]]
        nb = B.random_spawn(b, rng)
        self.assertEqual(B.to_list(nb)[:11], occupied_before)
        self.assertEqual(B.empty_count(nb), B.empty_count(b) - 1)

    def test_spawn_on_full_board_is_noop(self):
        full = L([2, 4, 8, 16] * 4)
        self.assertEqual(B.random_spawn(full, Random(0)), full)

    def test_spawn_distribution(self):
        """~10% fours, and every empty cell reachable roughly uniformly."""
        rng = Random(2024)
        b = 0
        fours = 0
        cells = [0] * 16
        N = 40000
        for _ in range(N):
            nb = B.random_spawn(b, rng)
            flat = B.to_list(nb)
            idx = next(i for i, v in enumerate(flat) if v)
            cells[idx] += 1
            if flat[idx] == 4:
                fours += 1
        self.assertAlmostEqual(fours / N, 0.10, delta=0.01)
        for c in cells:
            self.assertAlmostEqual(c / N, 1 / 16, delta=0.006)

    def test_spawn_positions(self):
        b = L([2, 0, 0, 4] + [0] * 12)
        self.assertEqual(B.spawn_positions(b),
                         [1, 2] + list(range(4, 16)))


class TestRoundTrip(unittest.TestCase):
    def test_to_from_list(self):
        rng = Random(17)
        for _ in range(500):
            vals = [0 if rng.random() < 0.3 else 1 << rng.randrange(1, 16)
                    for _ in range(16)]
            self.assertEqual(B.to_list(B.from_list(vals)), vals)

    def test_from_list_rejects_non_powers_of_two(self):
        with self.assertRaises(ValueError):
            B.from_list([3] + [0] * 15)


class TestGame(unittest.TestCase):
    def test_reproducible_with_seed(self):
        def play(seed):
            g = Game(seed=seed)
            r = Random(seed)
            while not g.game_over:
                acts = g.legal_actions()
                g.step(r.choice(acts))
            return g.score, g.board, g.moves
        self.assertEqual(play(42), play(42))

    def test_illegal_step_is_a_noop(self):
        g = Game(seed=1)
        g.board = row([2, 4, 8, 16])
        g.score = 100
        before = g.board
        reward, done, moved = g.step(B.LEFT)
        self.assertFalse(moved)
        self.assertEqual(g.board, before)
        self.assertEqual(g.score, 100)
        self.assertEqual(reward, 0)

    def test_score_accumulates(self):
        g = Game(seed=3)
        g.board = row([2, 2, 0, 0])
        g.score = 0
        reward, _, moved = g.step(B.LEFT)
        self.assertTrue(moved)
        self.assertEqual(reward, 4)
        self.assertEqual(g.score, 4)

    def test_step_spawns_a_tile(self):
        g = Game(seed=4)
        g.board = row([2, 2, 0, 0])
        g.step(B.LEFT)
        # 2 2 . .  ->  4 . . .  plus one spawned tile == 2 tiles
        self.assertEqual(len([v for v in B.to_list(g.board) if v]), 2)

    def test_games_terminate(self):
        rng = Random(0)
        for s in range(20):
            g = Game(seed=s)
            steps = 0
            while not g.game_over and steps < 100000:
                g.step(rng.choice(g.legal_actions()))
                steps += 1
            self.assertTrue(g.game_over)
            self.assertEqual(B.legal_actions(g.board), ())

    def test_reset(self):
        g = Game(seed=9)
        while not g.game_over:
            g.step(Random(0).choice(g.legal_actions()))
        g.reset()
        self.assertFalse(g.game_over)
        self.assertEqual(g.score, 0)
        self.assertEqual(g.moves, 0)
        self.assertEqual(len([v for v in B.to_list(g.board) if v]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
