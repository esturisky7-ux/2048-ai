# Command reference

Every command in one place, with the Windows equivalent where it differs.

## Which Python command?

| Platform | Use | Notes |
|---|---|---|
| Linux | `python3` | `python` often does not exist |
| macOS | `python3` | same |
| Windows | `py` | the launcher installed by python.org; `python` also works, `python3` usually does not |

Everything below is written with `python3`. On Windows, substitute `py`:

```powershell
py train.py --games 1000
```

Nothing else about the commands changes between platforms — same flags, same
output, same file layout.

---

## Getting help

```bash
python3 train.py --help
python3 evaluate.py --help
python3 experiment.py --help
python3 server.py --help
```

Version and platform, which is the first thing to include in a bug report:

```bash
python3 train.py --version
```

Verify the installation actually works (takes about two seconds):

```bash
python3 train.py --check
```

---

## Training

### Start a new run

```bash
python3 train.py --games 1000                  # a quick first taste
python3 train.py --games 20000                 # a serious run
python3 train.py                               # train until Ctrl-C
```

```powershell
py train.py --games 20000
```

### Train a specified number of games

```bash
python3 train.py --games 50000
```

The count is *this session's* games when resuming, not a total target.

### Resume

```bash
python3 train.py --resume                      # continue the 'default' run
python3 train.py --resume --games 20000        # 20k more, then stop
python3 train.py --resume --run bigrun         # continue a named run
```

```powershell
py train.py --resume
```

### One worker (the default)

```bash
python3 train.py --resume --workers 1
```

### Two workers

```bash
python3 train.py --resume --workers 2
```

Use at most one worker per CPU core. `train.py` prints a warning if you ask for
more than the machine has.

### Named runs and network shapes

```bash
python3 train.py --games 20000 --run experiment-a
python3 train.py --games 20000 --tuple-set 4x5     # 17 MB instead of 268 MB
python3 train.py --games 20000 --tuple-set 8x4     # 2 MB, fast, weaker
python3 train.py --list-runs                       # what has been trained
```

The network shape can only be chosen for a *new* run; resuming with a different
one is rejected rather than silently corrupting the weights.

### Learning parameters

```bash
python3 train.py --games 20000 --alpha 0.05            # learning rate
python3 train.py --games 20000 --alpha-decay 0.5       # geometric decay
python3 train.py --games 20000 --epsilon 0.01          # some exploration
python3 train.py --games 20000 --gamma 0.99            # discount
python3 train.py --games 20000 --seed 777              # tile-spawn seed
```

### Reporting, checkpointing, snapshots and periodic evaluation

```bash
python3 train.py --games 20000 --report-every 500      # progress line interval
python3 train.py --games 20000 --checkpoint-every 5000
python3 train.py --games 20000 --snapshot-every 10000  # freeze weight copies
python3 train.py --games 20000 --eval-every 5000 --eval-games 200
python3 train.py --games 20000 --quiet                 # no progress lines
```

Snapshots and evaluations keep their own schedules, independent of
`--checkpoint-every`, and work with any `--workers`: the workers stop at each
point so the weights hold still for it.

### Evaluate the current checkpoint and stop

```bash
python3 train.py --evaluate-now --eval-games 500
```

### Use a config file

```bash
python3 train.py --config baseline --run baseline
python3 train.py --config config/experiments/alpha-low.json
python3 train.py --list-experiments
```

### Stop training

Press **Ctrl-C**. Training finishes the current game, writes a checkpoint and
exits. Press Ctrl-C twice to quit immediately (the last partial interval is
then lost, but the previous checkpoint is intact).

On Windows, Ctrl-C in the training window does the same thing.

---

## Evaluating

### The learned agent

```bash
python3 evaluate.py --games 200
python3 evaluate.py --games 1000                 # tighter confidence intervals
python3 evaluate.py --games 200 --run bigrun
```

### A specific agent

```bash
python3 evaluate.py --agent random     --games 2000
python3 evaluate.py --agent heuristic  --games 1000
python3 evaluate.py --agent expectimax --games 50
python3 evaluate.py --agent learned    --games 200
```

### Comparing agents on identical games

```bash
python3 evaluate.py --compare random heuristic learned --games 200
python3 evaluate.py --compare random heuristic expectimax learned --games 100
```

Every agent plays the *same* seeded games, so the comparison measures play, not
luck. Note that expectimax is slow — 100 games takes over a minute.

### Search settings

```bash
python3 evaluate.py --agent expectimax --depth 3 --prob-cutoff 5e-3 --games 50
python3 evaluate.py --agent expectimax --adaptive --games 50
python3 evaluate.py --agent learned --depth 2 --games 50   # search + learning
```

### Evaluating a snapshot

```bash
python3 evaluate.py --agent learned \
    --checkpoint checkpoints/default/snapshots/games-000050000.f32
```

```powershell
py evaluate.py --agent learned --checkpoint checkpoints\default\snapshots\games-000050000.f32
```

PowerShell accepts forward slashes in paths too, so the Linux form also works.

A run's *current* weights cannot be evaluated while that run is being trained
(they would change during the evaluation); `evaluate.py` says so and exits
with status 1. Snapshots can be evaluated at any time.

### Saving results

```bash
python3 evaluate.py --games 500 --out results.json
python3 evaluate.py --games 500 --no-save          # don't append to the run log
python3 evaluate.py --games 500 --quiet            # no progress line
python3 evaluate.py --games 500 --seed 4242        # a different game set
```

---

## Experiments

```bash
python3 experiment.py --list                            # what is available
python3 experiment.py --run baseline --games 5000
python3 experiment.py --run reward-milestone --games 5000
python3 experiment.py --run search-depth-2              # agent-only, no training
python3 experiment.py --run-all --games 3000            # everything, in turn
python3 experiment.py --compare                         # table of saved results
python3 experiment.py --run baseline --games 5000 --fresh   # discard any prior run
python3 experiment.py --run baseline --games 5000 --workers 2
```

Results are written to `data/experiments/<name>.json`, each containing the full
resolved config that produced it.

---

## The control center

**This is the main way to use the project.** One command, then everything
happens in the browser.

```bash
./start.sh                            # start it and open http://127.0.0.1:8000
./stop.sh                             # stop it (from any terminal)
./start.sh --restart                  # stop the running one, start again
```

`start.sh` finds a Python 3.10+ and runs `server.py --open`; any `server.py`
option can be added to it. The same commands without the wrapper:

```bash
python3 server.py                     # http://127.0.0.1:8000
python3 server.py --open              # and open a browser
python3 server.py --port 8080         # if 8000 is taken
python3 server.py --quiet             # one line instead of the banner
python3 server.py --stop              # stop the running one
python3 server.py --restart           # stop the running one, start again
```

```powershell
py server.py --open
py server.py --stop
```

Starting it while it is already running does not fail: it prints the address
of the running one and exits.

Stop it with Ctrl-C, `--stop` / `./stop.sh`, or by closing its terminal
window; `kill` and an editor's stop button work too. Every one of these is
graceful: running jobs are asked to stop the way Ctrl-C asks them to, and the
server waits for training to write its checkpoint before exiting.

From the browser you can train, stop, resume, evaluate, compare agents, run
experiments, manage checkpoints, watch the AI play, play yourself, benchmark
the machine and read diagnostics — none of which needs a terminal.

The server binds to **localhost only**. `--host` exists but exposes a server
that can start processes and delete files; it prints a warning, which you
should take seriously. Use a tunnel instead:

```bash
ssh -L 8000:127.0.0.1:8000 you@the-machine
```

### Deep links

```
http://127.0.0.1:8000/#/training
http://127.0.0.1:8000/#/evaluate
http://127.0.0.1:8000/?watch=learned&depth=1&speed=5#/play/watch
http://127.0.0.1:8000/?watch=expectimax&depth=3&speed=1#/play/watch
http://127.0.0.1:8000/?watch=learned&seed=1&speed=0#/play/watch
```

| Parameter | Values |
|---|---|
| `watch` | `learned`, `expectimax`, `heuristic`, `random` |
| `depth` | search depth (`learned` and `expectimax`) |
| `speed` | `0.25`, `0.5`, `1`, `2`, `5`, `10`, or `0` for maximum |
| `seed` | replays exactly the same game |
| `live=poll` | poll instead of using the event stream |

Pages are at `#/overview`, `#/training`, `#/play`, `#/evaluate`, `#/compare`,
`#/experiments`, `#/checkpoints`, `#/benchmarks`, `#/logs`, `#/system` and
`#/settings`.

### Keyboard shortcuts

| Keys | Action |
|---|---|
| `g` then `o` `t` `p` `e` `c` `k` `l` `s` | Jump to a page |
| Arrow keys / `WASD` | Move, in a human game |
| `Space` | Pause or resume a watched game |
| `?` | Show the shortcut list |

### The HTTP API

The browser talks to a small JSON API, which is also usable from `curl`.
Mutating requests need the `X-2048-Request` header — a cross-origin page
cannot set it, which is the CSRF defence.

```bash
curl -s "http://127.0.0.1:8000/api/status?run=default"

curl -s -X POST http://127.0.0.1:8000/api/training/start \
     -H "Content-Type: application/json" -H "X-2048-Request: 1" \
     -d '{"run":"my-run","games":20000,"workers":2,"tuple_set":"4x6"}'

curl -s -X POST http://127.0.0.1:8000/api/training/stop \
     -H "Content-Type: application/json" -H "X-2048-Request: 1" -d '{}'
```

| Method and path | Purpose |
|---|---|
| `GET /api/status?run=` | Everything the Overview shows |
| `GET /api/history?run=` | Chart series, plus the evaluation series |
| `GET /api/stream?run=` | Server-Sent Events: status and log frames |
| `GET /api/runs` · `GET /api/agents` | What exists |
| `GET /api/training?run=` | Just the training view |
| `POST /api/training/start` · `/resume` · `/stop` | Control training |
| `GET /api/jobs` · `/api/jobs/{id}` · `/api/jobs/{id}/log` | Job state |
| `POST /api/jobs/{id}/stop` | Stop any job |
| `POST /api/evaluate` · `GET /api/evaluations?run=` | Fixed-seed evaluation |
| `POST /api/compare` · `GET /api/comparisons` | Agent comparison |
| `POST /api/benchmark` · `GET /api/benchmarks` | Machine throughput |
| `GET /api/experiments` · `POST /api/experiments/run` | The experiment lab |
| `GET /api/checkpoints` · `POST /api/checkpoints/label` · `/delete` | Checkpoints |
| `POST /api/game/ai/start` · `POST /api/game/human/start` | Start a game |
| `GET /api/game/{id}?since=&cursor=` · `POST /api/game/{id}/move` · `/control` | Play it (`cursor`: the frame on screen, which the server stays a few dozen frames ahead of) |
| `GET /api/system` · `GET /api/logs` | Diagnostics |
| `GET /api/settings` · `POST /api/settings` | UI preferences |
| `POST /api/shutdown` | Stop the server gracefully (what `--stop` uses) |

### Environment variables

| Variable | Effect |
|---|---|
| `AI2048_HOME` | Where `checkpoints/` and `data/` live. Defaults to the project directory — useful for keeping 268 MB weight files off a small system disk. |
| `AI2048_START_METHOD` | Force the multiprocessing start method (`fork` or `spawn`); mainly for exercising the Windows/macOS path on Linux. |
| `DASHBOARD_VERBOSE` | Log every HTTP request. |

```bash
AI2048_HOME=/data/2048 python3 server.py
```

---

## Tests and benchmarks

```bash
python3 -m unittest discover -s tests             # all 251 tests (~2 min)
python3 -m unittest discover -s tests -v          # verbose
python3 -m unittest discover -s tests -q          # quiet
python3 -m unittest tests.test_engine             # one module
python3 -m unittest tests.test_learning
python3 -m unittest tests.test_end_to_end         # drives the real CLIs
python3 -m unittest tests.test_control_center     # drives the web interface
python3 -m unittest tests.test_api                # API routing and validation
python3 -m unittest tests.test_server             # HTTP and security boundary
python3 -m unittest tests.test_jobs               # the job manager
python3 tests/benchmark_engine.py                 # throughput benchmark
```

```powershell
py -m unittest discover -s tests
py tests\benchmark_engine.py
```

Force the other multiprocessing start method (useful for reproducing a
Windows/macOS issue on Linux):

```bash
AI2048_START_METHOD=spawn python3 -m unittest tests.test_end_to_end
```

```powershell
$env:AI2048_START_METHOD = "spawn"
py -m unittest tests.test_end_to_end
Remove-Item Env:\AI2048_START_METHOD
```

---

## Housekeeping

### Where everything lives

```
checkpoints/<run>/     weights, metadata, snapshots
data/<run>/            history, evaluations, status
data/tables/           rebuildable index caches
data/experiments/      experiment results
```

### Start a run over

```bash
rm -rf checkpoints/default data/default
```

```powershell
Remove-Item -Recurse -Force checkpoints\default, data\default
```

### Reclaim disk

```bash
rm -rf data/tables                       # rebuildable, a few seconds
rm -rf checkpoints/*/snapshots           # frozen weight copies
```

```powershell
Remove-Item -Recurse -Force data\tables
```

### Verbose dashboard logging

```bash
DASHBOARD_VERBOSE=1 python3 server.py
```

```powershell
$env:DASHBOARD_VERBOSE = "1"
py server.py
```

---

## A typical session

One command, then the browser — this is the recommended path:

```bash
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
./start.sh
```

(`py server.py --open` on Windows.)

The command-line equivalent, for scripting and headless machines:

```bash
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
python3 train.py --check                                  # confirm it works
python3 train.py --games 20000                            # train (~50 min on a slow laptop)
python3 evaluate.py --games 200                           # measure it
python3 evaluate.py --compare random heuristic learned --games 200
python3 train.py --resume --workers 2                     # keep going, faster
```
