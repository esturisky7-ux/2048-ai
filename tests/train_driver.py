"""Test driver: run the trainer in a fresh process, observed or sabotaged.

The trainer's multi-worker behaviour can only be tested honestly in a process
of its own: workers are forked or spawned from it, and a spawned worker
re-imports its parent's ``__main__``. This script is that process. Not a test
module itself (unittest discovery only collects ``test*.py``); the tests in
``test_trainer.py`` and ``test_runlock.py`` run it.

    python tests/train_driver.py cli [train.py arguments...]
        train.py's own main(), so its exit code and messages are the real ones

    python tests/train_driver.py observe '<json spec>'
        run training sessions in-process with instruments attached, and print
        one JSON line describing what happened

``DRIVER_FAULT=mode:worker:games`` sabotages one worker after it has played
that many games: ``raise`` raises an exception, ``exit`` calls ``os._exit(3)``
and ``kill`` sends itself SIGKILL. The fault lives in this file, so it reaches
a spawned worker too: the worker re-imports this module, finds the saboteur
by name, and nothing in the project's own code needs a test hook.
"""

import hashlib
import json
import os
import signal
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import training.trainer as T                                   # noqa: E402

REAL_WORKER = T._worker_main
FAULT = os.environ.get("DRIVER_FAULT", "")


def sabotaged_worker(wid, *args):
    mode, who, after = FAULT.split(":")
    if wid == int(who):
        from training import learner as L
        real_play = L.TDLearner.play_game
        played = [0]

        def play_game(self, rng, learn=True):
            played[0] += 1
            if played[0] > int(after):
                if mode == "raise":
                    raise ValueError("simulated worker failure")
                if mode == "exit":
                    os._exit(3)
                if mode == "kill":
                    os.kill(os.getpid(), signal.SIGKILL)
            return real_play(self, rng, learn)

        L.TDLearner.play_game = play_game
    return REAL_WORKER(wid, *args)


def digest(buf) -> str:
    return hashlib.sha256(bytes(buf)).hexdigest()


def observe(spec: dict) -> dict:
    """Run ``spec["sessions"]`` against one run and report what happened.

    Each session is ``{"games", "workers", "resume", "stop_after",
    "stop_in_eval", ...}``: ``stop_after`` asks to stop (as Ctrl-C does) once
    the session has played that many games, ``stop_in_eval`` once an
    evaluation has started that many; extra keys are training config
    overrides. Recorded: every game index the
    parent counted, and for each evaluation and snapshot the game count, how
    many worker processes were alive, and weight checksums around it.
    """
    import multiprocessing as mp
    from training.config import load_config

    seen = {"indices": [], "evals": [], "snapshots": [], "sessions": []}
    real_record = T.Trainer._record_game
    real_eval = T.Trainer.evaluate_now
    real_snap = T.Trainer.take_snapshot

    def record_game(self, index, score, moves, tile):
        seen["indices"].append(index)
        real_record(self, index, score, moves, tile)
        stop_after = getattr(self, "_driver_stop_after", 0)
        if stop_after and self.session_games >= stop_after:
            self.stop_requested = True        # what Ctrl-C does

    def evaluate_now(self):
        before = digest(self.net.weights)
        alive = len(mp.active_children())
        stop_in_eval = getattr(self, "_driver_stop_in_eval", 0)
        if stop_in_eval:
            # Ctrl-C part-way through the evaluation.
            from evaluation import evaluator as E
            real_play, played = E.play_one, [0]

            def play_one(*a, **k):
                played[0] += 1
                if played[0] == stop_in_eval:
                    self.stop_requested = True
                return real_play(*a, **k)
            E.play_one = play_one
        try:
            real_eval(self)
        finally:
            if stop_in_eval:
                E.play_one = real_play
        seen["evals"].append({"games": self.game_index, "workers_alive": alive,
                              "before": before,
                              "after": digest(self.net.weights)})

    def take_snapshot(self):
        alive = len(mp.active_children())
        weights = digest(self.net.weights)
        real_snap(self)
        with open(self.run.snapshot_path(self.game_index), "rb") as f:
            snap = digest(f.read())
        seen["snapshots"].append({"games": self.game_index,
                                  "workers_alive": alive,
                                  "weights": weights, "snapshot": snap})

    T.Trainer._record_game = record_game
    T.Trainer.evaluate_now = evaluate_now
    T.Trainer.take_snapshot = take_snapshot

    run = spec.get("run", "driver")
    for session in spec["sessions"]:
        session = dict(session)
        games = session.pop("games")
        workers = session.pop("workers", 1)
        resume = session.pop("resume", False)
        stop_after = session.pop("stop_after", 0)
        stop_in_eval = session.pop("stop_in_eval", 0)
        training = {"report_every": 0, "checkpoint_every": 0,
                    "eval_every": 0, "eval_games": 2}
        training.update(session)
        overrides = {"training": training}
        if resume:
            t = T.Trainer(run, overrides, resume=True, quiet=True)
        else:
            t = T.Trainer(run, load_config(overrides={
                "run": run, "tuple_set": "8x4", **overrides}), quiet=True)
        t._driver_stop_after = stop_after
        t._driver_stop_in_eval = stop_in_eval
        try:
            t.train(n_games=games, workers=workers)
            seen["sessions"].append({"game_index": t.game_index,
                                     "stopped": t.stop_requested})
        finally:
            t.close()
    return seen


def main() -> int:
    if FAULT:
        T._worker_main = sabotaged_worker
    mode = sys.argv[1]
    if mode == "cli":
        import train
        sys.argv = ["train.py"] + sys.argv[2:]
        return train.main()
    if mode == "observe":
        print(json.dumps(observe(json.loads(sys.argv[2]))), flush=True)
        return 0
    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
