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

## The dashboard

```bash
python3 server.py                     # http://127.0.0.1:8000/
python3 server.py --open              # and open a browser
python3 server.py --port 8080         # if 8000 is taken
```

```powershell
py server.py --open
```

Stop it with Ctrl-C.

The dashboard binds to **localhost only**. `--host 0.0.0.0` exists but exposes
an unauthenticated server to your network; it prints a warning, which you should
take seriously.

### Deep links

```
http://127.0.0.1:8000/?run=bigrun
http://127.0.0.1:8000/?watch=learned&depth=1&speed=5
http://127.0.0.1:8000/?watch=expectimax&depth=3&speed=1
http://127.0.0.1:8000/?watch=learned&seed=1&speed=0
```

| Parameter | Values |
|---|---|
| `run` | any run name from `--list-runs` |
| `watch` | `learned`, `expectimax`, `heuristic`, `random` |
| `depth` | search depth (`learned` and `expectimax`) |
| `speed` | `0.25`, `0.5`, `1`, `2`, `5`, or `0` for maximum |
| `seed` | replays exactly the same game |

---

## Tests and benchmarks

```bash
python3 -m unittest discover -s tests             # all 141 tests (~65 s)
python3 -m unittest discover -s tests -v          # verbose
python3 -m unittest discover -s tests -q          # quiet
python3 -m unittest tests.test_engine             # one module
python3 -m unittest tests.test_learning
python3 -m unittest tests.test_end_to_end         # drives the real CLIs
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

```bash
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
python3 train.py --check                                  # confirm it works
python3 train.py --games 20000                            # train (~50 min on a slow laptop)
python3 evaluate.py --games 200                           # measure it
python3 evaluate.py --compare random heuristic learned --games 200
python3 server.py --open                                  # watch it play
python3 train.py --resume --workers 2                     # keep going, faster
```
