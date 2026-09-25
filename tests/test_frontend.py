"""Browser-side regressions, tested without a browser or a JavaScript stack.

The control center's front end is plain scripts with no build step and no
test framework, and this project adds none. So:

* the Checkpoints page's Watch button is checked at the source level (the
  snapshot id must travel in the route, be selected once the Play page's
  options load, and reach ``/api/game/ai/start``), and its server half --
  a game started from a snapshot id plays that snapshot -- for real;
* the live-update fallback in ``core.js`` is exercised for real under Node.js
  when it is installed (GitHub's runners have it), with fake timers, a fake
  ``EventSource`` and a fake ``fetch`` and nothing else: Node's own ``vm``
  module, no packages. Without Node that test is skipped.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

JS = Path(ROOT) / "dashboard" / "static" / "js"


def source(name: str) -> str:
    return (JS / name).read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i:text.index(end, i)]


# ---------------------------------------------------------------------------
# Watch a snapshot from the Checkpoints page
# ---------------------------------------------------------------------------
class TestSnapshotWatch(unittest.TestCase):
    def test_the_watch_button_puts_a_snapshot_in_the_route(self):
        handler = between(source("views/checkpoints.js"),
                          "watch.onclick", "const evaluate")
        self.assertIn("App.setRun(c.run)", handler)
        self.assertRegex(handler, r'c\.kind === "snapshot"')
        self.assertRegex(handler,
                         r"play/watch/\$\{encodeURIComponent\(c\.id\)\}")
        # The current weights keep the plain route, and so the old default.
        self.assertRegex(handler, r':\s*"play/watch"')

    def test_the_play_page_selects_the_snapshot_and_plays_it(self):
        play = source("views/play.js")
        # The id comes from the route...
        self.assertRegex(play, r"this\.mountWatch\(args\[1\] \? "
                               r"safeDecode\(args\[1\]\)")
        # ...is selected only once the listed options have loaded, and only
        # if it is one of them...
        after = between(play, "this.loadCheckpointOptions(ckptSel).then(",
                        "\n  },")
        self.assertIn("[...ckptSel.options].some((o) => o.value === "
                      "checkpoint)", after)
        self.assertIn("ckptSel.value = checkpoint", after)
        # ...and is what the start request sends.
        start = between(play, "const start = async", "startBtn.onclick")
        self.assertIn("payload.checkpoint = ckptSel.value", start)
        self.assertIn('API.post("/api/game/ai/start", payload)', start)

    def test_a_game_started_from_a_snapshot_id_plays_that_snapshot(self):
        from dashboard import api, games, store
        from training import checkpoint as CP
        from training.ntuple import NTupleNetwork
        tmp = tempfile.mkdtemp(prefix="2048watch-")
        saved = (CP.CHECKPOINT_ROOT, CP.DATA_ROOT, store.CHECKPOINT_ROOT)
        CP.CHECKPOINT_ROOT = store.CHECKPOINT_ROOT = Path(tmp) / "checkpoints"
        CP.DATA_ROOT = Path(tmp) / "data"
        try:
            run = CP.Run("watched")
            run.create_dirs()
            NTupleNetwork("8x4", path=str(run.weights_path)).close()
            run.save_meta({"games": 30, "tuple_set": "8x4"})
            run.snapshot(20)
            r = api.start_ai_game({"agent": "learned",
                                   "checkpoint": "watched:games-000000020.f32"})
            agent = games.SESSIONS.get(r["session"]["id"]).agent
            self.assertEqual(Path(agent.checkpoint_path),
                             run.snapshot_path(20))
            self.assertEqual(agent.games_trained, 20)
            self.assertIsNone(agent.live_run)
        finally:
            games.SESSIONS.stop_all()
            CP.CHECKPOINT_ROOT, CP.DATA_ROOT, store.CHECKPOINT_ROOT = saved
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Live updates: server-sent events, with polling only while they are down
# ---------------------------------------------------------------------------
# Loads core.js into a bare context and drives App.startLive() through the
# stream's life. Prints one JSON object of observations.
HARNESS = r"""
const vm = require("vm");
const fs = require("fs");

let now = 0, nextId = 1, fetches = 0, hangFetch = null;
const timers = new Map();
const setTimeoutFake = (fn, ms) => { const id = nextId++; timers.set(id, { at: now + ms, fn }); return id; };
const clearTimeoutFake = (id) => { timers.delete(id); };
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); };
async function advance(ms) {
  const end = now + ms;
  for (;;) {
    const due = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
    if (!due) break;
    timers.delete(due[0]); now = due[1].at; due[1].fn(); await flush();
  }
  now = end;
}
const pollTimers = () => timers.size;

async function fetchFake() {
  fetches++;
  if (hangFetch) await hangFetch.promise;
  return { ok: true, status: 200, statusText: "OK",
           json: async () => ({ state: "STOPPED", run: "default", jobs: [] }) };
}
class FakeES {
  constructor(url) { this.url = url; this.readyState = 0; this.handlers = {}; FakeES.all.push(this); }
  addEventListener(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); }
  close() { this.readyState = 2; }
  emit(type, data) { for (const fn of this.handlers[type] || []) fn({ data: JSON.stringify(data) }); }
}
FakeES.all = [];

const noop = () => {};
const context = {
  console, URL, URLSearchParams, setImmediate,
  setTimeout: setTimeoutFake, clearTimeout: clearTimeoutFake, fetch: fetchFake,
  EventSource: FakeES,
  location: { search: "", origin: "http://127.0.0.1", hash: "" },
  document: { addEventListener: noop, dispatchEvent: noop, querySelector: () => null },
  localStorage: { getItem: () => null, setItem: noop },
  CustomEvent: class { constructor(type, o) { this.type = type; this.detail = o && o.detail; } },
  matchMedia: () => ({ matches: false, addEventListener: noop }),
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8") + "\n;globalThis.App = App;", context);
const App = context.App;
const statuses = [];
App.setStatus = (s) => { if (!(s && s.error)) statuses.push(s); };

(async () => {
  const out = {};
  const polled = async (ms) => { const before = fetches; await advance(ms); return fetches - before; };

  App.startLive();
  const es = FakeES.all[0];
  out.healthy = await polled(20000);                       // SSE up: no polling

  // The stream drops, and the error fires again before the first poll has
  // even come back: exactly when an unguarded loop gets started twice.
  es.onerror(); es.onerror(); es.onerror(); await flush();
  out.dropped_timers = pollTimers();
  out.dropped = await polled(20000);                        // one loop: ~10 polls

  es.onopen();                                              // reconnected
  out.reconnected = await polled(20000);

  es.onerror(); await flush();
  es.emit("status", { error: "unknown run" });              // not a valid status
  out.after_error_status = await polled(10000);
  es.emit("status", { state: "STOPPED", run: "default" });  // a valid one
  out.after_valid_status = await polled(20000);

  for (let i = 0; i < 25; i++) {                            // flapping
    es.onerror(); await advance(300); es.onopen(); await flush();
  }
  es.onerror(); await flush();
  out.flapping_timers = pollTimers();
  out.after_flapping = await polled(20000);

  es.readyState = 2; es.onerror(); await flush();           // gives up for good
  out.permanent = await polled(20000);

  // A tick waiting on the network when the stream comes back must not
  // schedule another one.
  let release; hangFetch = { promise: new Promise((r) => { release = r; }) };
  App.stopLive(); App.startLive();
  const es2 = FakeES.all[FakeES.all.length - 1];
  es2.onerror(); await flush();                             // tick in flight
  es2.onopen(); hangFetch = null; release(); await flush();
  out.in_flight_timers = pollTimers();
  out.in_flight = await polled(20000);

  out.old_stream_closed = es.readyState === 2;
  out.stale_stream_error = (() => { const before = pollTimers(); es.onerror(); return pollTimers() - before; })();
  await flush();
  out.after_stale_error = await polled(20000);
  out.statuses = statuses.length;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class TestLiveUpdateFallback(unittest.TestCase):
    """core.js polls only while the event stream is down, and never twice."""

    @classmethod
    def setUpClass(cls):
        r = subprocess.run(["node", "-e", HARNESS, str(JS / "core.js")],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise AssertionError(r.stdout + r.stderr)
        cls.seen = json.loads(r.stdout.strip().splitlines()[-1])

    def test_no_polling_while_the_stream_is_healthy(self):
        self.assertEqual(self.seen["healthy"], 0)

    def test_a_dropped_stream_polls_once_per_interval(self):
        self.assertLessEqual(self.seen["dropped_timers"], 1)
        # 20 s at the default 2 s interval: ten polls, not thirty.
        self.assertTrue(9 <= self.seen["dropped"] <= 11, self.seen)

    def test_polling_stops_when_the_stream_reconnects(self):
        self.assertEqual(self.seen["reconnected"], 0)

    def test_a_valid_status_event_stops_polling_but_an_error_does_not(self):
        self.assertGreater(self.seen["after_error_status"], 0)
        self.assertEqual(self.seen["after_valid_status"], 0)

    def test_flapping_never_stacks_timers(self):
        self.assertLessEqual(self.seen["flapping_timers"], 1)
        self.assertTrue(9 <= self.seen["after_flapping"] <= 11, self.seen)

    def test_a_stream_that_gives_up_for_good_keeps_polling(self):
        self.assertTrue(9 <= self.seen["permanent"] <= 11, self.seen)

    def test_a_tick_in_flight_when_polling_stops_schedules_nothing(self):
        self.assertEqual(self.seen["in_flight_timers"], 0)
        self.assertEqual(self.seen["in_flight"], 0)

    def test_a_replaced_stream_cannot_restart_polling(self):
        self.assertTrue(self.seen["old_stream_closed"])
        self.assertEqual(self.seen["stale_stream_error"], 0)
        self.assertEqual(self.seen["after_stale_error"], 0)


class TestLiveUpdateSource(unittest.TestCase):
    """The same guarantees, checked where Node.js is not available."""

    def test_polling_is_guarded_and_cancellable(self):
        core = source("core.js")
        start = between(core, "  startPolling() {", "\n  },")
        self.assertIn("if (this._polling) return;", start)
        self.assertIn("if (gen !== this._pollGen) return;", start)
        stop = between(core, "  stopPolling() {", "\n  },")
        self.assertIn("this._pollGen++", stop)
        self.assertIn("this.stopPolling()", between(core, "  stopLive() {",
                                                     "\n  },"))
        live = between(core, "  startLive() {", "\n  },")
        self.assertRegex(live, r"es\.onopen = .*this\.stopPolling\(\)")
        self.assertRegex(live, r"status && !status\.error\) "
                               r"this\.stopPolling\(\)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
