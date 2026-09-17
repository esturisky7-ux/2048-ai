# Contributing

Thanks for looking. Bug reports, questions and pull requests are all welcome.

## Before you start

Run this and paste the output into any issue you open — it identifies the
version, Python, platform and multiprocessing mode in one line each:

```bash
python3 train.py --check
```

(On Windows: `py train.py --check`.)

## Reporting a bug

Useful bug reports say:

1. what you ran, exactly — the full command line;
2. what happened, including the traceback if there is one;
3. what you expected instead;
4. the output of `python3 train.py --check`.

If it involves training results, the run's `checkpoints/<run>/config.json` and
game count matter too, because behaviour depends on both.

## The ground rules

**Zero dependencies.** The project uses only the Python 3 standard library, and
that is a feature rather than an accident: anyone can clone it and run it with
no install step, on any machine, offline. A pull request that adds a dependency
needs to make a strong case for why the standard library genuinely cannot do
the job.

**Performance claims need measurements.** The engine and the training loop have
been optimised by profiling, and the comments record the before/after numbers
(for example `3.62 µs → 2.29 µs`). If you change something on the hot path,
include the measurement — `python3 tests/benchmark_engine.py` is the standard
one — and say what machine it was measured on.

**Correctness before speed.** Several optimisations here exist because they
were verified to produce bit-identical results first. Keep that order.

**Do not break checkpoint compatibility silently.** The weight-file format is
just a flat float32 array whose length is determined by the tuple set, so it is
easy to break by changing a tuple definition. If a change would make existing
checkpoints unreadable or meaningless, say so prominently in the pull request.

**Cross-platform.** The project runs on Linux, Windows and macOS, and CI tests
all three on every push. Things to watch for:

- Build paths from the source file's location with `pathlib`; never hardcode a
  separator, a home directory or a working directory.
- Open text files with `encoding="utf-8"` explicitly.
- Do not assume `fork`; anything touching multiprocessing has to work under
  `spawn` too, which means arguments must be picklable and entry points must be
  guarded with `if __name__ == "__main__":`.
- `os.kill(pid, 0)` is not a liveness probe on Windows — it terminates the
  process.
- Do not assume a shell, a POSIX signal, or a command-line tool exists.

The `TestPortability` class in `tests/test_infra.py` covers several of these.

## Running the tests

```bash
python3 -m unittest discover -s tests          # everything, ~65 s
python3 -m unittest tests.test_engine          # one module
python3 tests/benchmark_engine.py              # throughput benchmark
```

All 141 must pass before a pull request is ready. The end-to-end module drives
the real command-line entry points as subprocesses, so it is slower than the
rest and worth running last.

To exercise the multiprocessing path the other platforms use:

```bash
AI2048_START_METHOD=spawn python3 -m unittest tests.test_end_to_end
```

## New tests

Anything that changes behaviour needs a test. The existing suite is a decent
model for what "a good test here" looks like:

- prefer checking against an independent reference implementation over
  checking against a hardcoded expected value (`test_engine.py` verifies all
  65,536 possible rows this way);
- test the property, not the implementation — "training improves play" and
  "TD updates converge without overshoot" are real tests in
  `test_learning.py`;
- test the failure modes too: torn files, corrupt JSON, stale status, illegal
  moves on constructed boards.

## Code style

Match the surrounding code. In practice that means:

- standard library only, no formatter enforced, ~79 column lines;
- module docstrings that explain *why*, not just what — this codebase leans
  heavily on them, and that is deliberate;
- comments on non-obvious performance tricks, with the measurement;
- no unnecessary abstraction on the hot path.

## Pull requests

Small and focused is better than large and sweeping. Describe what changed and
why, include measurements for anything performance-related, and note explicitly
if behaviour or file formats change.

## License

By contributing you agree that your contribution is licensed under the
[MIT License](LICENSE), the same as the rest of the project.
