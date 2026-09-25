"""Live AI games: computed only as fast as they are watched, and cleaned up.

The server plays an AI game in a thread that stays ``BUFFER_AHEAD`` frames
ahead of what the viewer has *displayed*. The viewer fetches frames faster
than slow playback shows them, so it reports its playback cursor; treating
fetched frames as watched made 0.25x playback compute a whole game up front.
A counting agent makes every decision visible.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import board as B                                 # noqa: E402
from dashboard import api                                     # noqa: E402
from dashboard import games as G                              # noqa: E402

# One initial frame, BUFFER_AHEAD + 1 ahead of the cursor, and at most one
# decision already under way when the worker checks the buffer.
SLACK = 3


class CountingAgent:
    """Always the first legal move (so games are long), counting decisions."""

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.closed = 0
        self.delay = delay

    def new_game(self):
        pass

    def act(self, board):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return B.legal_actions(board)[0]

    def close(self):
        self.closed += 1


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.sessions = []

    def tearDown(self):
        for s in self.sessions:
            s.stop()

    def start(self, **kw):
        agent = CountingAgent(**kw)
        s = G.AIGameSession(agent, "counting", seed=7)
        self.sessions.append(s)
        s.start()
        return s, agent

    def settle(self, agent, quiet=0.3):
        """Wait until the worker has stopped deciding."""
        last = -1
        while agent.calls != last:
            last = agent.calls
            time.sleep(quiet)
        return agent.calls


class TestThrottle(SessionTest):
    def test_fetching_without_watching_computes_only_the_buffer(self):
        s, agent = self.start()
        fetched = 0
        for _ in range(25):                  # what 0.25x playback's polls did
            snap = s.snapshot(fetched, cursor=0)
            fetched += len(snap["frames"])
            time.sleep(0.02)
        self.assertLessEqual(self.settle(agent), G.BUFFER_AHEAD + SLACK)
        self.assertLessEqual(fetched, G.BUFFER_AHEAD + SLACK + 1)

    def test_slow_playback_stays_just_ahead_of_the_viewer(self):
        s, agent = self.start()
        fetched = cursor = 0
        for _ in range(60):                  # one frame shown per poll
            snap = s.snapshot(fetched, cursor=cursor)
            fetched += len(snap["frames"])
            cursor = min(cursor + 1, fetched - 1)
            time.sleep(0.01)
            self.assertLessEqual(agent.calls,
                                 cursor + G.BUFFER_AHEAD + SLACK)
        self.assertLessEqual(self.settle(agent),
                             cursor + G.BUFFER_AHEAD + SLACK)
        self.assertGreater(agent.calls, cursor)       # but it is ahead

    def test_maximum_speed_keeps_up_to_the_end(self):
        s, agent = self.start()
        fetched = 0
        deadline = time.time() + 30
        snap = {"done": False}
        while not snap["done"] and time.time() < deadline:
            snap = s.snapshot(fetched, cursor=fetched)
            fetched += len(snap["frames"])
            time.sleep(0.01)
        self.assertTrue(snap["done"], "the game never finished")
        self.assertEqual(fetched, snap["total"])
        self.assertEqual(agent.calls, snap["total"] - 1)

    def test_a_cursor_never_moves_backwards(self):
        s, agent = self.start()
        wait_for(lambda: len(s.frames) > 30)
        s.snapshot(0, cursor=30)
        s.snapshot(0, cursor=0)
        self.assertEqual(s.viewer_pos, 30)

    def test_without_a_cursor_fetched_frames_count_as_watched(self):
        """What a client from before cursors existed (or curl) still gets."""
        s, agent = self.start()
        fetched = since = 0
        wait_for(lambda: agent.calls > G.BUFFER_AHEAD)
        for _ in range(5):
            since = fetched
            fetched += len(s.snapshot(since)["frames"])
            time.sleep(0.05)
        self.assertEqual(s.viewer_pos, since)
        self.assertGreater(since, G.BUFFER_AHEAD)


class TestControls(SessionTest):
    def test_pause_stops_computing_and_step_computes_exactly_one(self):
        s, agent = self.start()
        wait_for(lambda: agent.calls >= 3)
        s.pause()
        paused_at = self.settle(agent)
        s.snapshot(0, cursor=len(s.frames))      # viewer caught up
        time.sleep(0.4)
        self.assertEqual(agent.calls, paused_at)
        for n in (1, 2, 3):
            s.step()
            self.assertTrue(wait_for(lambda: agent.calls == paused_at + n))
            time.sleep(0.25)
            self.assertEqual(agent.calls, paused_at + n)
        s.resume()
        self.assertTrue(wait_for(lambda: agent.calls > paused_at + 3))

    def test_stop_ends_the_thread_and_closes_the_agent(self):
        s, agent = self.start()
        wait_for(lambda: agent.calls >= 3)
        s.stop()
        self.assertTrue(wait_for(lambda: not s.is_alive()))
        self.assertEqual(agent.closed, 1)
        s.stop()                                 # twice is harmless
        self.assertEqual(agent.closed, 1)

    def test_a_session_that_never_started_still_closes_its_agent(self):
        agent = CountingAgent()
        s = G.AIGameSession(agent, "counting", seed=1)
        s.stop()
        self.assertEqual(agent.closed, 1)


class TestCollection(unittest.TestCase):
    def setUp(self):
        self._ttl = G.SESSION_TTL
        G.SESSION_TTL = 0.6
        self.mgr = G.SessionManager(janitor_interval=0.1)

    def tearDown(self):
        G.SESSION_TTL = self._ttl
        self.mgr.stop_all()

    def add(self, **kw):
        agent = CountingAgent(**kw)
        s = G.AIGameSession(agent, "counting", seed=3)
        self.mgr.add(s)
        s.start()
        return s, agent

    def test_an_abandoned_paused_game_is_collected_on_its_own(self):
        s, agent = self.add()
        wait_for(lambda: agent.calls >= 2)
        s.pause()
        # Nobody polls it again, and no other session is ever started.
        self.assertTrue(wait_for(lambda: s.id not in self.mgr.sessions, 5))
        self.assertTrue(wait_for(lambda: not s.is_alive(), 5))
        self.assertEqual(agent.closed, 1)

    def test_an_abandoned_playing_game_is_collected_too(self):
        s, agent = self.add(delay=0.01)
        self.assertTrue(wait_for(lambda: s.id not in self.mgr.sessions, 5))
        self.assertTrue(wait_for(lambda: not s.is_alive(), 5))
        self.assertEqual(agent.closed, 1)

    def test_a_watched_game_is_not_collected(self):
        s, agent = self.add()
        stop = threading.Event()

        def viewer():
            while not stop.is_set():
                s.snapshot(0, limit=1, cursor=0)
                time.sleep(0.05)
        t = threading.Thread(target=viewer, daemon=True)
        t.start()
        try:
            time.sleep(1.0)
            self.assertIn(s.id, self.mgr.sessions)
            self.assertTrue(s.is_alive())
        finally:
            stop.set()
            t.join()

    def test_the_collector_is_stopped_with_the_sessions(self):
        self.add()
        janitor = self.mgr._janitor
        self.mgr.stop_all()
        self.assertTrue(wait_for(lambda: not janitor.is_alive(), 5))
        self.assertEqual(self.mgr.sessions, {})

    def test_a_new_session_right_after_stop_all_is_still_collected(self):
        self.add()
        self.mgr.stop_all()
        s, agent = self.add()                # before the old collector exits
        self.assertTrue(wait_for(lambda: s.id not in self.mgr.sessions, 5))
        self.assertEqual(agent.closed, 1)


class TestApi(unittest.TestCase):
    def tearDown(self):
        G.SESSIONS.stop_all()

    def test_the_state_endpoint_takes_a_cursor(self):
        sid = api.start_ai_game({"agent": "random", "seed": 5})["session"]["id"]
        session = G.SESSIONS.get(sid)
        wait_for(lambda: len(session.frames) > 10)
        snap = api.handle_get(f"/api/game/{sid}",
                              {"since": ["0"], "cursor": ["4"]})
        self.assertGreater(len(snap["frames"]), 4)
        self.assertEqual(session.viewer_pos, 4)
        for bad in ("x", "1.5"):
            with self.assertRaises(api.ApiError) as ctx:
                api.handle_get(f"/api/game/{sid}", {"cursor": [bad]})
            self.assertEqual(ctx.exception.status, 400)
        # A negative position counts as zero, as a negative "since" always did.
        snap = api.handle_get(f"/api/game/{sid}",
                              {"since": ["-5"], "cursor": ["-1"]})
        self.assertEqual(snap["frames"][0]["i"], 0)
        self.assertEqual(session.viewer_pos, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
