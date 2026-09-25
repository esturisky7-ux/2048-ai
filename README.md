<div align="center">

# 2048 AI

**Train a 2048 agent from self-play, evaluate it honestly, and watch it learn in
a live local dashboard.**

[![tests](https://github.com/esturisky7-ux/2048-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/esturisky7-ux/2048-ai/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-6f42c1)](#platform-support)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](#requirements)

**TD(0) n-tuple learning · Fixed-seed evaluation · Pure Python standard library**

**📖 [Read the Wiki](https://github.com/esturisky7-ux/2048-ai/wiki)** — task-oriented
guides for installation, training, evaluation, the control center and troubleshooting

</div>

![The 2048 AI Control Center showing a live training run: status, records, achievement rates and six interactive charts](docs/images/control-center.png)

<p align="center"><sub>The control center during a live two-worker training run — every number on this page is read from the running system.</sub></p>

> **Recorded evaluation:** after 66,000 self-play training games, the learned
> agent averaged **55,626** over 200 fixed-seed evaluation games, reached the
> **2048 tile in 90.5%** of them, and reached **8192**. [See the complete
> evaluation report.](#evaluating-an-agent)

---

## Quick Start

```bash
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
./start.sh
```

That starts the control center and opens **<http://127.0.0.1:8000>** in your
browser. It is the only command you need: there is no separate front end to
build or start, no database and nothing to install. Training, stopping and
resuming it, evaluation, agent comparisons, experiments, checkpoints, watching
the AI play, playing yourself, benchmarks and diagnostics all happen in the
browser.

```
  2048 AI Control Center
  Dashboard:     http://127.0.0.1:8000
  Platform:      Linux x86_64
  Python:        3.12.3
  AI checkpoint: None yet — the dashboard will offer to train one
  Training:      Stopped

  Press Ctrl+C to stop the server.
```

**To stop it,** press **Ctrl+C** in that terminal, or run `./stop.sh` from
another one. Running training always finishes its game and saves first.

On **Windows**, run the server directly (`py` is the Python launcher):

```powershell
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
py server.py --open
```

**First time?** The control center notices there is no trained agent and offers
to start one. A thousand games takes about a minute and already reaches the 512
tile; it suggests a worker count from your CPU without claiming the whole
machine.

The command line has not gone anywhere — see
[The command line](#the-command-line) — but you no longer need it.

---

## Table of contents

**Getting it running**

- [Requirements](#requirements)
- [First-time setup](#first-time-setup)
- [Starting the application](#starting-the-application)
- [Stopping and restarting](#stopping-and-restarting)
- [Ports](#ports)
- [Troubleshooting](#troubleshooting)

**The project**

- [Overview](#overview)
- [The control center](#the-control-center)
- [Features](#features)
- [How it works](#how-it-works)
- [The AI: TD(0) afterstate n-tuple learning](#the-ai-td0-afterstate-n-tuple-learning)
- [The four agents](#the-four-agents)
- [Project architecture](#project-architecture)
- [Starting a new training run](#starting-a-new-training-run)
- [Resuming training](#resuming-training)
- [Using multiple workers](#using-multiple-workers)
- [Evaluating an agent](#evaluating-an-agent)
- [Watching the AI play](#watching-the-ai-play)
- [The command line](#the-command-line)
- [Running experiments](#running-experiments)
- [Running tests](#running-tests)
- [Understanding checkpoints](#understanding-checkpoints)
- [Performance and benchmarks](#performance-and-benchmarks)
- [Platform support](#platform-support)
- [Project structure](#project-structure)
- [Security and networking](#security-and-networking)
- [Git workflow](#git-workflow)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

| What | Version | Why |
|---|---|---|
| **Python** | **3.10 or newer** (3.10, 3.12 and 3.13 are tested) | Runs everything: the web server, the UI's API, training and evaluation |
| **A web browser** | any current one | The control center is a web page at <http://127.0.0.1:8000> |
| **Git** | any | To clone and update the repository. Optional: GitHub's **Code ▸ Download ZIP** works too |
| Disk space | ~300 MB free | The default network's weight file (268 MB; created sparse on Linux/macOS, so it only grows as used) |

That is the complete list. In particular you do **not** need:

- **Node.js or npm** — the front end is plain HTML, CSS and JavaScript served
  by the Python server. There is no `package.json`, no build step and no
  separate front-end server.
- **pip packages or a virtual environment** — only the Python standard library
  is used, so there is no `requirements.txt` and nothing to `pip install`. A
  virtual environment does no harm if you prefer one.
- **A GPU, Docker, a database or any environment variables.**

Check your Python:

```bash
python3 --version        # Linux and macOS
py --version             # Windows
```

If it is missing or older than 3.10:

- **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install python3 git`
- **macOS:** the installer from <https://www.python.org/downloads/macos/>, or
  `brew install python git`. (macOS ships an old or stub `python3`; if the
  command opens a developer-tools prompt, install one of these.)
- **Windows:** the installer from <https://www.python.org/downloads/windows/>
  — tick **“Add python.exe to PATH”** — and Git from
  <https://git-scm.com/downloads>. Use `py` rather than `python3`, which may
  open the Microsoft Store instead.

---

## First-time setup

**Linux and macOS:**

```bash
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
python3 train.py --check      # optional: verifies Python and the engine, ~2 seconds
./start.sh
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/esturisky7-ux/2048-ai.git
cd 2048-ai
py train.py --check           # optional
py server.py --open
```

There is no install step: no `pip install`, no `npm install`, nothing to
build. The `checkpoints/` and `data/` folders are created the first time you
train; they are deliberately not in Git (see
[Understanding checkpoints](#understanding-checkpoints)).

Then, in the browser, press **Start your first training run**. You do not need
to train from the terminal first.

Platform notes:

- **macOS and Windows** start multi-worker training with `spawn` rather than
  `fork` (forking is not safe on macOS once system frameworks are loaded, and
  does not exist on Windows). The only visible difference is that each worker
  spends a few seconds building its lookup tables at start-up.
- **Windows:** the weight file is not sparse on NTFS, so the default network
  really does occupy 268 MB from the first run. Choose the `4x5` (17 MB) or
  `8x4` (2 MB) network on the Training page if that matters.

---

## Starting the application

```bash
./start.sh
```

This is the one command to use. It:

1. changes into the project folder, so it works from any directory
   (`~/2048-ai/start.sh` is fine);
2. finds a Python 3.10 or newer (`python3`, `python`, or `python3.1x`);
3. runs **`server.py`**, the control center, which serves the web page, its
   JSON API and the live-update stream from **one process on one port**;
4. opens <http://127.0.0.1:8000> in your browser once the server is listening.

Training, evaluation, comparisons, benchmarks and experiments are **not**
separate things to start. You start them from the browser, and the server
runs each one as a supervised background process.

Options — anything `server.py` accepts can be added:

```bash
./start.sh --port 8080         # use another port
./start.sh --restart           # stop the one that is running, start fresh
./start.sh --quiet             # one line of output instead of the banner
NO_BROWSER=1 ./start.sh        # do not open a browser window
PYTHON=python3.12 ./start.sh   # use a particular Python
```

Running `./start.sh` while it is already running is harmless: it prints the
address of the running one and exits.

**Without the script** (Windows, or if you prefer), run the server directly —
this is exactly what `start.sh` does:

```bash
python3 server.py --open       # Linux / macOS
py server.py --open            # Windows
```

**Only one terminal is needed.** Older versions of these instructions ran
`train.py` in one terminal and the dashboard in a second; that still works,
but it is no longer necessary.

---

## Stopping and restarting

| To… | Linux / macOS | Windows |
|---|---|---|
| **Stop** | **Ctrl+C** in the server's terminal, or `./stop.sh` from any terminal | **Ctrl+C**, or `py server.py --stop` |
| **Restart** | `./start.sh --restart` | `py server.py --restart` |
| **Stop one on another port** | `./stop.sh --port 8080` | `py server.py --stop --port 8080` |

Every way of stopping is safe. The server stops accepting work, asks any
running job to stop the way Ctrl+C would, **waits for training to finish its
current game and write a checkpoint**, and only then exits. That can take a
few seconds with a strong agent, whose games are long. `./stop.sh` waits for
all of it and then prints `stopped.`

Closing the terminal window, `kill <pid>`, an editor's stop button and logging
out are all handled the same way — training is never left running on its own.

Press Ctrl+C a second time only if you really want to quit without waiting.

**Restart after every `git pull`** that changes Python files. The page is
re-read on every load, but the server keeps running the code it started with
until it is restarted.

Training resumes from its checkpoint: after a restart, open **Training** and
choose **Continue**.

---

## Ports

| Port | Bound to | Used for |
|---|---|---|
| **8000** (TCP) | `127.0.0.1` only | Everything: the web page (`/`), static files (`/static/…`), the JSON API (`/api/…`) and live updates (`/api/stream`, Server-Sent Events) |

That is the only port. Nothing else listens: training and evaluation jobs are
local background processes that open no ports, and the front end talks to the
server on the same address it was loaded from — so there is no CORS setup, no
API URL to configure, and no second port for a front-end dev server.

Change it with `--port` (`./start.sh --port 8080`), and pass the same
`--port` to `./stop.sh`. The server binds to your own machine only; see
[Security and networking](#security-and-networking) before changing `--host`.

Optional environment variables (none are required):

| Variable | Effect |
|---|---|
| `NO_BROWSER=1` | `start.sh` does not open a browser |
| `PYTHON=…` | `start.sh` uses this Python |
| `AI2048_HOME=…` | Keep checkpoints, run data and caches somewhere other than the project folder |
| `DASHBOARD_VERBOSE=1` | Log every HTTP request to the terminal |

---

## Troubleshooting

### Starting and stopping

**`The 2048 AI control center is already running at http://127.0.0.1:8000/`.**
Not an error: one is already running (perhaps in another terminal window, or
one you forgot about). Open that address. To replace it with a fresh one — for
example after a `git pull` — run `./start.sh --restart`.

**The page looks broken or out of date after `git pull`.**
The web page is read from disk on every load, but the Python server keeps
running the code it started with, so an old server behind a new page can
disagree. Restart it with `./start.sh --restart` (`py server.py --restart` on
Windows), then reload the page with Ctrl+Shift+R (Cmd+Shift+R on macOS).

**`Another program is using that port` / `Address already in use`.**
Something other than the control center has port 8000. Either stop that
program, or use another port for both commands:
`./start.sh --port 8080` and later `./stop.sh --port 8080`. To see what holds
the port: `lsof -i :8000` (Linux/macOS) or `netstat -ano | findstr :8000`
(Windows).

**`./start.sh: Permission denied`.**
The execute bit was lost (for example by copying the folder or unzipping a
download). Restore it with `chmod +x start.sh stop.sh`, or run
`bash start.sh`.

**`/usr/bin/env: 'bash\r': No such file or directory`.**
The script was saved with Windows line endings. The repository's
`.gitattributes` prevents this for fresh clones; to repair an existing copy
run `sed -i 's/\r$//' start.sh stop.sh` (on macOS: `sed -i '' 's/\r$//' start.sh stop.sh`).

**`error: Python 3.10 or newer is required, and none was found`.**
`start.sh` looks for `python3`, `python` and `python3.10`–`python3.13`.
Install a newer Python (see [Requirements](#requirements)) or point it at one
explicitly: `PYTHON=/path/to/python3.12 ./start.sh`.

**No browser window opened.**
`start.sh` opens one only when there is a desktop to open it on — not over
SSH, for example. Open <http://127.0.0.1:8000> yourself. `NO_BROWSER=1
./start.sh` turns it off on purpose.

**Training says "started outside this control center".**
That run is being trained by a process this server did not launch: a
`train.py` you started in a terminal, or one left behind by an older version
of the control center that was closed without stopping it. **Stop** still
works on it (it sends the same Ctrl-C a terminal would, so it saves a
checkpoint). Resuming the same run is refused until it has stopped, because two
trainers writing one weight file would corrupt it. On Windows, stop it with
Ctrl+C in its own window.

**"run 'NAME' is already in use by process 1234".**
Every trainer — from a terminal, the control center or an experiment — holds
its run's lock (in `checkpoints/.locks/`) for as long as it runs, so a second
trainer, a deletion or an experiment's `--fresh` reset of that run is refused.
Stop the process named in the message, or use another run name. The lock goes
away the moment its process exits, even if it crashed, so there is never a
stale lock to remove.

**Training keeps running after I closed the terminal.**
Fixed: closing the window, `kill`, an editor's stop button and logging out now
all stop training cleanly, the same as Ctrl+C. If you have a trainer left over
from before this fix, use **Stop** as described above, or `kill -INT <pid>`
(the pid is shown on the Training page).

### Everything else

**`python3: command not found` (Windows).**
Use `py` or `python`. `python3` is a Unix convention. If none of them work,
Python is not on PATH — reinstall it with “Add python.exe to PATH” ticked.

**`python` opens the Microsoft Store.**
That is Windows' placeholder for a missing Python. Install the real one from
<https://www.python.org/downloads/windows/> and use `py`.

**`SyntaxError` on start-up.**
You are on Python 3.9 or older. Check with `python3 --version`; 3.10 is the
minimum.

**`no trained weights at ... (run 'default' has not been trained yet)`.**
Nothing has been trained yet, or you are pointing at the wrong run. Train
something first (`python3 train.py --games 1000`) or pass `--run NAME`.

**The dashboard says “no training process running” while training is running.**
The dashboard decides from a heartbeat file and the trainer's PID. Check they
are looking at the same run — the run selector is in the top right, and the URL
carries `?run=NAME`.

**Training is slower than the numbers in [Performance](#performance-and-benchmarks).**
That is expected as the agent improves: better play means much longer games.
Watch moves/sec instead of games/sec. If moves/sec is also low, check nothing
else is using the CPU, and try `--workers N` up to your core count.

**Training seems stuck at a low score.**
Give it games. A few thousand games is early; the 2048 rate typically crosses
50% somewhere around 10,000–15,000 games with the default settings.

**I want to start over.**
Delete the run's two directories: `checkpoints/<run>/` and `data/<run>/`. Or
just train under a new `--run` name.

**Disk filled up.**
`data/tables/` holds rebuildable index caches (22–34 MB each) and
`checkpoints/*/snapshots/` holds frozen weight copies (268 MB each with the
default network). Both are safe to delete; the tables rebuild themselves.

**Something else.**
Run `python3 train.py --check` and include its output — it reports the version,
Python, platform and worker start method — when opening an issue.

---

## Overview

2048 is a sliding-tile game: every move slides the whole board one way, equal
tiles merge, and the game drops a new tile in a random empty cell. It is easy
to play badly and genuinely hard to play well.

This repository is a laboratory for making a computer play it well, and for
being able to *prove* that one version plays better than another:

- a **bitboard engine** that simulates ~140,000 moves a second in pure Python;
- four **agents** of increasing strength, from uniform random to a learned
  value function;
- a **reinforcement-learning trainer** that improves by self-play and
  checkpoints so a run survives Ctrl-C, a crash or a power cut;
- an **evaluation harness** that replays the same seeded games for every agent
  and reports confidence intervals, so differences are measured, not claimed;
- an **experiment framework** that stores the exact config alongside each
  result;
- a **local web dashboard** with live charts and a game viewer.

After a couple of hours of training on a slow laptop, the learned agent reaches
the 2048 tile in about 89% of games and has produced the 8192 tile.

---

## The control center

`./start.sh` (or `python3 server.py`) starts a local web application that drives all of it. It is
the primary interface; the command-line tools remain for scripting and headless
use.

| Page | What you do there |
|---|---|
| **Overview** | Live status, the current run, records, achievement rates, and seven interactive charts. Hover any chart for the exact value and the game number it came from. |
| **Training** | Start a new run with sensible defaults (basic and advanced settings), or continue an existing one — "train 10,000 more", a custom amount, or continuously. Live throughput, ETA, and a **graceful stop** that finishes the current game and writes a checkpoint. |
| **Play** | Watch any agent play, with pause, single-step, restart and speeds from 0.25× to maximum. Play 2048 yourself with the arrow keys, WASD, swipe or on-screen buttons. Or take the AI on in **You vs AI**. |
| **Evaluate** | Run a fixed-seed evaluation on any agent or saved checkpoint, watch its progress, and get the mean with a 95% confidence interval, percentiles and Wilson intervals for every tile rate. Export JSON or CSV. |
| **Compare** | Several agents over the identical seeded games, with the confidence intervals drawn so overlap is visible rather than hidden. |
| **Experiments** | Run any of the 18 shipped configurations and compare the results, with the sample size shown next to every number. |
| **Checkpoints** | Every saved agent with its games, evaluation score, size and configuration. Watch, evaluate, resume, label, or delete with confirmation. |
| **Benchmarks** | Engine throughput, per-agent decision rate and real training moves/second, with the CPU and Python that produced them. |
| **Logs / System** | Recent events, and platform, versions, storage and running job PIDs. |

<p align="center">
  <img src="docs/images/play.png" width="49%" alt="Watching the trained agent play, with pause, step and speed controls">
  <img src="docs/images/training.png" width="49%" alt="The training page during a live two-worker run">
</p>

Nothing in the interface is mocked. Every number is read from the running
system, and pages that have no data say so rather than inventing any.

**How it works.** The browser talks to a small JSON API. Anything slow —
training, evaluation, comparison, benchmarking, experiments — is handed to a
**job manager** that runs it as a supervised subprocess, so a long job never
blocks the web server and a crash in one cannot take the server down.
Live updates arrive over Server-Sent Events, with polling as a fallback.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#the-control-center) has the
detail.

**Still zero dependencies.** No React, no npm, no Flask, no Docker. The server
is `http.server`; the front end is hand-written HTML, CSS and JavaScript that
loads nothing from a CDN.

### Keyboard shortcuts

`g` then `o`/`t`/`p`/`e`/`c`/`k`/`l`/`s` jumps between pages, arrow keys or
WASD play a human game, `Space` pauses a watched game, and `?` lists them all.

---

## Features

| | |
|---|---|
| **Browser-first** | `./start.sh`, then do everything at `http://127.0.0.1:8000`. |
| **Zero dependencies** | Python 3.10+ standard library only. No install step, no npm. |
| **Fast pure-Python engine** | 64-bit bitboards, precomputed row tables, ~140k moves/s. |
| **Learns from self-play** | TD(0) on afterstates with an n-tuple value network. |
| **Crash-safe checkpoints** | Memory-mapped weights, atomic metadata, Ctrl-C always saves. |
| **Resume any run** | `--resume` continues from the exact game count. |
| **Multi-core training** | `--workers N` shares one weight table across processes. |
| **Honest evaluation** | Identical seeded games per agent, 95% CIs, Wilson intervals for tile rates. |
| **Experiment framework** | 18 shipped configs; each result stores the config that produced it. |
| **Supervised jobs** | Training and evaluation run as subprocesses with real state, progress and a graceful stop. |
| **Play it yourself** | Human play and You-vs-AI, on the same Python engine the AI uses. |
| **Localhost-only** | Binds to 127.0.0.1, refuses cross-origin requests, never takes a filesystem path from the browser. |
| **Tested** | 337 tests, run on Linux, Windows and macOS by CI. |

---

## How it works

One move, end to end:

```
board  ──►  agent picks a direction  ──►  afterstate  ──►  game drops a random tile  ──►  next board
                     ▲                        │
                     └──── learning ──────────┘
```

The **afterstate** — the board after the slide but *before* the random tile
appears — is the pivot the whole design turns on. It is the only board the
agent is fully responsible for, so it is the thing worth learning the value of.

```
2048 board
    ↓
bitboard engine          16 nibbles in one 64-bit int; moves are table lookups
    ↓
agent chooses move       argmax over legal a of  reward(a) + V(afterstate(a))
    ↓
afterstate               the board the agent's decision actually produced
    ↓
environment spawns a random tile
    ↓
next state
    ↓
TD update                V(previous afterstate) ← nudged towards what happened
    ↓
n-tuple weights          67 million float32 values in a memory-mapped file
    ↓
checkpoint               flush dirty pages + atomic meta.json
```

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) walks through each stage in
detail, including why a bitboard, why precomputed tables, and how workers
share weights.

---

## The AI: TD(0) afterstate n-tuple learning

You do not need a machine-learning background to follow this. Here is the whole
idea in four steps.

### 1. Give every board a score guess

The agent keeps a function `V(board)` that guesses *how much more score this
position is worth from here on*. At the start it guesses zero for everything,
because it knows nothing.

### 2. Choose moves by that guess

For each of the (at most) four legal moves, the agent computes the board that
move produces and asks:

```
value of this move  =  points the merge scores right now  +  V(resulting board)
```

and plays the best one. No search, no lookahead — four table lookups per turn.
That is why it runs hundreds of times faster than a search-based player.

### 3. Correct the guess with what actually happened

After the move, the game drops a tile and the agent picks its next move, which
is worth `r' + V(s'')`. That is a better estimate of the previous position's
value than the old guess was, because it contains one extra step of real
experience. So nudge the old guess towards it:

```
V(s')  ←  V(s')  +  alpha · [ r' + gamma·V(s'')  −  V(s') ]
                              └──────── the TD error ────────┘
```

`alpha` (0.1 by default) is how far to move: all the way would throw away
everything learned so far, zero would learn nothing. `gamma` is 1.0, because
2048 is finite and a point earned late is worth exactly as much as a point
earned early.

When the game ends the target is simply 0 — there is no future score after
death. That single rule is what teaches the agent that surviving matters,
without anyone telling it so.

This is **TD(0)**: temporal-difference learning with a one-step lookahead. The
*afterstate* part is the refinement that makes it work so well here — the agent
learns values of boards it fully controls, so the random tile spawn never
pollutes the thing being learned.

### 4. Represent V with an n-tuple network, not a neural network

`V` cannot be a table of every board — there are far too many. It is instead a
sum over a handful of small cell patterns (**tuples**):

```
 0  1  2  3         tuple A: cells 0,1,2,3,4,5      (a 6-cell strip)
 4  5  6  7         tuple B: cells 4,5,6,7,8,9
 8  9 10 11         tuple C: cells 0,1,2,4,5,6      (a 2×3 block)
12 13 14 15         tuple D: cells 4,5,6,8,9,10
```

Each tuple has its own lookup table indexed by the exact tile values in those
cells. `V(board)` is the sum of one lookup per tuple, and learning adds the same
small delta to each of those entries. Every tuple is also shared across the 8
symmetries of a square (4 rotations × 2 reflections), because a pattern learned
in one corner should apply in all of them — this multiplies the effective
training data by eight.

The default network is 4 tuples × 16<sup>6</sup> entries = **67 million
weights**, evaluated with 32 array reads.

**Why not a neural network?** Because on this problem an n-tuple network is
both faster and stronger. Evaluating a board is a few dozen memory reads
instead of matrix multiplications; there is no autograd, no BLAS, no GPU, and
no training instability. It is also what the strongest published 2048 agents
actually use (Szubert & Jaśkowski 2014; Wu et al. 2014). A small MLP on the
same CPU budget would be roughly two orders of magnitude slower per update and
would play worse.

**Why no exploration (epsilon = 0)?** This looks like a mistake and is not. The
random tile spawns already push the agent into new situations constantly, and a
single random move late in a game can destroy a structure that took a thousand
moves to build. An `--epsilon` knob exists for experiments; the default is off
deliberately.

**What is the reward?** The game's own merge score, and nothing else. The sum of
merge scores over a game *is* the final score, so maximising expected return is
literally maximising expected score. Alternative shaping terms are available in
`config/experiments/` — and `training/reward.py` explains which of them are
farmable exploits and why.

---

## The four agents

All four take a board and return a direction. They differ only in how they
decide. The numbers below are measured over the *same* 100 seeded games
(see [Performance](#performance-and-benchmarks)).

### `random` — the floor

Picks uniformly among legal moves. Exists so every other number has something
to be compared against.

> mean score **1,025** · never reaches 512 · 724 games/s

### `heuristic` — hand-written judgement, no lookahead

Scores each resulting board with a formula a human wrote: empty cells and
available merges are good; disordered rows, scattered tile sizes and heavy
boards are bad. Plays the best-scoring move. Every feature computable from one
row of four cells is precomputed into a 65,536-entry table, so evaluating a
board costs a transpose and twelve array lookups.

> mean score **4,604** · reaches 1,024 at best · 98 games/s

### `expectimax` — search that models the dice

Builds a game tree that alternates between *max* nodes (the player chooses) and
*chance* nodes (the game drops a 2 with probability 0.9 or a 4 with probability
0.1, uniformly over empty cells), and averages properly over the randomness.
Leaves are scored with the heuristic evaluator. Depth is bounded, unlikely
branches are cut off by a probability threshold, and repeated positions are
memoised.

This is the classic strong 2048 algorithm. Its weakness here is cost: the tree
grows explosively as the board fills.

> mean score **14,584** · 12% reach 2048 · **1.4 games/s**

### `learned` — the trained value function

The TD-trained n-tuple network described above, playing greedily: one network
evaluation per legal move. No search at all.

> mean score **54,045** · **89% reach 2048** · best game 123,748 with an 8192 tile · 5.5 games/s

It scores **3.7× more than depth-2 expectimax while running 3.9× faster**,
because a search that must examine thousands of positions can never compete
with a lookup that has already absorbed millions of games of experience.
`--depth 2` on the learned agent puts a search on top of the learned values:
stronger still, and much slower. Like the plain policy, the search scores every
move with the reward and discount the run was trained with (by default, the
merge score and a discount of 1).

---

## Project architecture

```
engine/       bitboard core and a small stateful Game wrapper
agents/       random, heuristic, expectimax, learned + a name→agent registry
training/     n-tuple network, TD learner, reward functions, stats, checkpoints,
              and the trainer that drives the loop
evaluation/   the fixed, seeded evaluation procedure and its statistics
experiments/  runs a config, evaluates it, stores config+result together
dashboard/    stdlib HTTP server, JSON API, static front end, live game thread
config/       default.json plus 18 experiment configs
tests/        337 tests: unit, integration, API, end-to-end, plus a benchmark
docs/         architecture and command reference
```

Four command-line entry points sit on top: `train.py`, `evaluate.py`,
`experiment.py`, `server.py`.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture and
[`docs/COMMANDS.md`](docs/COMMANDS.md) for a command reference.

---

## Starting a new training run

```bash
python3 train.py --games 20000
```

Progress lines look like this:

```
games       600 |  28.5 g/s  12178 mv/s | a=0.1000 | mean    6,702 med    6,498 best   17,284 | tile 1,024 | 512  65.8% 1k  13.7% 2k   0.0% 4k   0.0%
```

which reads as: total games played, games and moves per second, current learning
rate, mean and median score over the last 1,000 games, the best score ever, the
best tile in the window, and how often each milestone tile was reached recently.

Useful options:

```bash
python3 train.py --games 50000 --run bigrun      # name the run
python3 train.py --tuple-set 4x5                 # smaller, lighter network
python3 train.py --alpha 0.05                    # learning rate
python3 train.py --eval-every 10000              # periodic fixed evaluation
python3 train.py --snapshot-every 25000          # freeze weights for later
python3 train.py --list-runs                     # what exists already
```

`--games 0` (the default) trains until you stop it. **Ctrl-C is always safe**:
it finishes the current game, writes a checkpoint and exits. Press it twice to
quit immediately.

---

## Resuming training

```bash
python3 train.py --resume
```

That continues the run named `default` from its exact saved game count, with
its saved configuration. For a named run:

```bash
python3 train.py --resume --run bigrun --games 20000
```

Resuming restores the game counter, all-time records, the learning-rate
schedule and the tile statistics — the report line picks up exactly where it
left off. A run trained with one network shape refuses to resume with another,
rather than silently producing nonsense.

---

## Using multiple workers

```bash
python3 train.py --resume --workers 2
```

Workers are separate processes that all play games and all update **one shared
weight table**, which lives in a memory-mapped file. There is no message passing
for the weights themselves and no lock: updates touch a few dozen of tens of
millions of entries, so collisions are rare and the occasional lost update is
noise next to the TD error itself. (This is the Hogwild! scheme.)

Measured on the development laptop (2 cores), fresh 4×6 run, 600 games:

| workers | games/s | moves/s | speed-up |
|---|---|---|---|
| 1 | 28.5 | 12,178 | — |
| 2 | 53.7 | 21,593 | **1.8×** |

Use at most one worker per CPU core; `train.py` warns if you ask for more.

`--eval-every` and `--snapshot-every` work with any number of workers: at each
of those points every worker finishes its share of the games, the evaluation or
snapshot runs on weights that hold still, and a fresh set of workers carries
on. Every game index is played exactly once, even across Ctrl-C and
`--resume`. If a worker fails, the whole run fails with the reason — the
others are stopped, the checkpoint is saved, `train.py` exits non-zero and the
control center shows the job as failed.

One caveat worth knowing: with a single worker a run is bit-for-bit
reproducible from its seed. With several, the games are still seeded but the
weight updates interleave between processes, so a rerun is statistically
identical rather than literally identical.

---

## Evaluating an agent

Training statistics are a moving target — the agent changes while they are
being collected. Evaluation instead freezes the policy and replays **the same
seeded games** every time, so two evaluations differ only because the agents
differ.

```bash
python3 evaluate.py --games 200
```

```
agent            learned  (depth 1)  [66,000 games trained]
games            200   seed 987654   37.6s   (5.32 games/s)

mean score         55,626.2   95% CI [52,207, 59,045]
median score       59,012.0
std dev            24,669.3
min / max             8,752 / 129,032
p25 / p75 / p95  36,053 / 72,151 / 92,321
mean moves          2,645.8   (longest 5,378)
highest tile          8,192   median 4,096

tile             rate      95% CI
  1,024         99.00%   [96.43%, 99.73%]
  2,048         90.50%   [85.64%, 93.83%]
  4,096         57.00%   [50.07%, 63.67%]
  8,192          2.50%   [ 1.07%,  5.72%]
```

Compare agents on identical games:

```bash
python3 evaluate.py --compare random heuristic expectimax learned --games 100
```

Other things worth knowing:

```bash
python3 evaluate.py --agent expectimax --depth 3 --prob-cutoff 5e-3 --games 50
python3 evaluate.py --agent learned --depth 2 --games 50     # search on top of learning
python3 evaluate.py --agent learned --checkpoint checkpoints/default/snapshots/games-000050000.f32
python3 evaluate.py --games 500 --out results.json --no-save
```

Percentages come with **Wilson score intervals** rather than the usual
normal approximation, because they stay correct near 0% and 100%; means get a
standard normal-approximation interval.

A run's current weights cannot be evaluated while that run is being trained:
they would change from one game to the next, so the result would not describe
any one agent. Evaluate a snapshot instead (`--snapshot-every` takes them, from
the command line or the control center), or stop training first. While an
evaluation of the current weights runs, training that run waits for it.

---

## Watching the AI play

**Play ▸ Watch the AI** in the control center. Pick an agent, press **Start
game**, and watch it move.

![Watching the trained agent, with score, move count, max tile and per-move decision time](docs/images/play.png)

| Control | What it does |
|---|---|
| Agent | Learned, expectimax, heuristic or random |
| Search depth | For learned and expectimax; depth 1 is the plain trained policy |
| Checkpoint | Play with a frozen snapshot instead of the current weights |
| Speed | 0.25×, 0.5×, 1×, 2×, 5×, 10× or maximum |
| Pause / Step / Restart / Stop | Including single-stepping one move at a time |
| **Watch current AI** | One click: load the latest checkpoint and go |

The game is played by the server in a throttled background thread that stays
only a few dozen moves ahead of what your browser is showing, so slow
playback costs almost no CPU and training keeps the machine. **Watch** on a
snapshot in the Checkpoints page opens this view with that snapshot selected.

A particular game can be linked directly:

```
http://127.0.0.1:8000/?watch=learned&depth=1&speed=5&seed=1#/play/watch
```

| Parameter | Meaning |
|---|---|
| `watch` | `learned`, `expectimax`, `heuristic` or `random` |
| `depth` | search depth, for `expectimax` and `learned` |
| `speed` | `0.25`, `0.5`, `1`, `2`, `5`, `10`, or `0` for maximum |
| `seed` | replays exactly the same game, move for move |
| `live=poll` | use polling instead of the event stream |

### Play it yourself

**Play ▸ Play yourself** gives you the board, with arrow keys, WASD, swipe or
on-screen buttons. **You vs AI** lets you play a seeded game and then watches
the trained agent play the same one.

Both use the **Python engine**, not a second implementation in JavaScript —
your game follows exactly the rules the AI trains on. (One game each is for
fun, not a measurement; [Evaluate](#evaluating-an-agent) is how agents are
actually compared.)

---

## The command line

The web interface is the *preferred* way to use the project, not the only one.
Every command-line tool still works exactly as before, which is what you want
for scripting, automation, headless machines and CI:

```bash
python3 train.py --resume --games 20000 --workers 2
python3 evaluate.py --compare random heuristic learned --games 200
python3 experiment.py --run baseline --games 5000
python3 tests/benchmark_engine.py
python3 train.py --check
```

Both interfaces drive the same code and the same files, so they compose: start
a run from the terminal and watch it in the browser, or start one in the
browser and inspect its checkpoint with `train.py --list-runs`. The control
center notices a run started elsewhere and says so rather than pretending it
owns it.

`docs/COMMANDS.md` is the full reference. One addition worth knowing:

```bash
AI2048_HOME=/somewhere/else python3 server.py
```

moves everything the project writes — checkpoints, run data, caches — somewhere
other than the project directory, which is useful when 268 MB weight files do
not belong on your system disk.

---

## Running experiments

An experiment is a JSON file in `config/experiments/`. Running one trains (or
just evaluates) with that configuration and stores **the result together with
the exact config that produced it**, so a number is still interpretable months
later.

```bash
python3 experiment.py --list                            # what is available
python3 experiment.py --run reward-milestone --games 5000
python3 experiment.py --run-all --games 3000            # everything, in turn
python3 experiment.py --compare                         # table of saved results
```

The 18 shipped experiments cover learning rates, exploration, reward shaping
(including two deliberately farmable rewards, as demonstrations of what goes
wrong), network shapes and search depths.

Results land in `data/experiments/<name>.json`.

---

## Running tests

```bash
python3 -m unittest discover -s tests            # everything (~90 s)
python3 -m unittest discover -s tests -v         # verbose
python3 -m unittest tests.test_engine            # one module
python3 tests/benchmark_engine.py                # throughput benchmark
```

On Windows use `py -m unittest discover -s tests`.

**337 tests.** What they actually check:

- **`test_engine.py`** — merge rules including the awkward cases (`2 2 2 2` →
  `4 4 . .`, `4 4 8 8` → `8 16 . .`), that a freshly merged tile cannot merge
  again in the same move, tile-mass conservation, game-over detection against
  independent legal-move enumeration, and the spawn distribution. It verifies
  **all 65,536 possible rows** against a reference implementation written a
  different way.
- **`test_agents.py`** — every agent always returns a *legal* move, including
  on constructed boards with exactly one legal move; the strength ordering
  random < heuristic < expectimax genuinely holds; deterministic agents
  reproduce exactly.
- **`test_learning.py`** — the generated `value`/`update` code matches a naive
  reference; the symmetry group really is a group; weights survive close and
  reopen; two memory maps of one file stay in sync (the mechanism `--workers`
  depends on); potential-based shaping telescopes to zero; TD updates converge
  without overshoot; **and that training measurably improves play**.
- **`test_infra.py`** — atomic writes leave no partial files; corrupt JSON is
  tolerated; a torn final history line does not lose the rest; percentiles,
  Wilson intervals and summary statistics are numerically correct; every
  shipped experiment config is valid; a stale status file is reported as
  stopped rather than phantom training; and the portability guarantees
  (path handling, worker start method, the Windows-safe liveness probe).
- **`test_end_to_end.py`** — drives `train.py`, `evaluate.py`, `experiment.py`
  and `server.py` as real subprocesses: checkpoint, resume from the right game
  count, **Ctrl-C saves before exiting**, two workers train one shared
  checkpoint under *both* process start methods, evaluation is reproducible,
  and a run created from the command line is visible and playable in the
  control center.
- **`test_jobs.py`** — the job manager: jobs reach the right terminal state, a
  **stopped job gets to save before it exits**, an unresponsive child is
  escalated rather than hanging shutdown, two jobs cannot work on one run, and
  finished jobs are trimmed while live ones are not.
- **`test_api.py`** — every route; run names that are not safe path components;
  numeric ranges; agent allowlists; **checkpoint identifiers that try to become
  file paths**; game sessions and their controls; settings clamping. Its
  sandbox is itself tested, because an earlier version of it launched real
  training against a real checkpoint directory.
- **`test_server.py`** — the HTTP boundary over a real socket: path traversal,
  the CSRF header, cross-origin rejection, hardening headers, no CORS, mutating
  routes unreachable by GET, oversized and malformed bodies, and the event
  stream.
- **`test_control_center.py`** — the whole product, start to finish: start the
  server, train, watch progress update, stop gracefully, resume, evaluate,
  watch the agent play, play a human game, read the statistics, list
  checkpoints, benchmark, **restart the server and confirm everything is still
  there**, then delete a checkpoint and shut down cleanly.
- **`test_runlock.py`** — one writer per run, across real processes: a second
  trainer (terminal, control center or experiment) is refused, a run being
  trained elsewhere cannot be deleted or reset, **a killed trainer never
  strands its run** (with one worker or several, killed while they start or
  while they play), and a run's current weights cannot be evaluated while it
  trains, while an evaluation keeps training out until it is done.
- **`test_trainer.py`** — with several workers, evaluations and snapshots
  happen at exactly their game counts with no worker running, every game index
  is played once across stops and resumes, and a worker that raises, exits or
  is killed fails the run loudly (from the command line and as a control-center
  job); snapshots keep their own schedule and never change afterwards.
- **`test_policy.py`** — training, its evaluations and the loaded agent choose
  identical moves for every reward configuration, and search on top of the
  learned values optimises the same objective.
- **`test_games.py`** — a watched game is computed only as fast as it is shown,
  pause and step behave, and abandoned games are cleaned up on their own.
- **`test_frontend.py`** — the Checkpoints page's Watch button plays the
  snapshot it was pressed on, and the live-update fallback polls only while
  the event stream is down (run under Node.js where it is installed).
- **`test_startup.py`** — the ways people really start and stop it: starting a
  second copy points at the first instead of failing, `--stop` and
  `--restart`, a port held by another program is left alone, **SIGTERM and
  SIGHUP (closing the terminal) still save training and leave no trainer
  behind**, and a trainer the server did not start can be stopped from the
  browser but is never joined by a second one.

---

## Understanding checkpoints

A **run** is one training experiment. Everything belonging to it lives under
two directories named after it:

```
checkpoints/<run>/weights.f32     memory-mapped float32 weights (the agent)
checkpoints/<run>/meta.json       games played, records, learning rate, config
checkpoints/<run>/config.json     the configuration the run started with
checkpoints/<run>/snapshots/      optional frozen copies of the weights
data/<run>/history.jsonl          one row per report interval, for the graphs
data/<run>/evaluations.jsonl      every fixed evaluation ever run
data/<run>/status.json            small heartbeat file the dashboard polls
```

`checkpoints/.locks/` holds one small file per run, which the process training
(or deleting, or evaluating) a run locks while it works on it; see
[Troubleshooting](#troubleshooting).

Crash safety has two halves. The weights are a memory-mapped file, so the
operating system is already writing them back as training proceeds; a
checkpoint only has to flush the dirty pages. The metadata is small and written
with the write-temp-then-rename dance, which is atomic, so `meta.json` is never
observed half-written. The worst case after a hard power loss is that you
resume from the last completed checkpoint interval (2,000 games by default).

**These directories are not in Git**, and that is deliberate: the default
network's weight file is 268 MB, well past GitHub's 100 MB limit, and it is
pure derived data. The repository is fully usable without any checkpoint — you
train your own, which is the point.

If you want to share a trained agent, attach the `.f32` file to a **GitHub
Release** rather than committing it (Releases allow files up to 2 GB), and note
the tuple set it was trained with. Then:

```bash
python3 evaluate.py --agent learned --checkpoint path/to/downloaded.f32
```

Snapshots (`--snapshot-every N`) let you compare an agent against its own
younger self on identical games, which is the cleanest way to see improvement.
They follow their own schedule, whatever `--checkpoint-every` is set to, and a
snapshot is never modified once it has been written.

---

## Performance and benchmarks

**All numbers below were measured on the development machine** — a Dell
Latitude 3300: Intel Celeron 3865U at 1.80 GHz, **2 cores**, ~8 GB RAM,
integrated graphics, **no dedicated GPU**, Python 3.12.3 on Linux. This is
modest hardware by any standard; most machines will be faster. Your figures
will differ with CPU, Python version, operating system, worker count and — for
anything involving the learned agent — how much it has trained, because a
stronger agent plays much longer games.

### Engine

```
random play, full games        1,200 games/s      141,475 moves/s
```

Per-call costs (`python3 tests/benchmark_engine.py`):

```
move LEFT                 1.46 us      is_game_over      3.25 us
move UP (1 transpose)     2.70 us      legal_actions     3.77 us
transpose                 1.15 us      random_spawn      1.61 us
empty_count               0.74 us      max_tile_exp      0.78 us
```

Optimisation was driven by profiling, not guesswork: folding the second
transpose into the vertical-move tables (3.62 → 2.29 µs), building spawn
positions from row tables (2.82 → 1.63 µs) and testing game-over with row
tables instead of full moves (5.58 → 3.25 µs) together took random play from
887 to ~1,200 games/s. Each change was verified to produce bit-identical
results before being kept.

### Agents

100 identical seeded games each, learned agent after 66,000 training games
(`python3 evaluate.py --compare random heuristic expectimax learned --games 100`):

```
agent           games       mean               95% CI    median      best    tile   2048%      g/s
--------------------------------------------------------------------------------------------------
random            100      1,025         [917, 1,133]       912     3,060     256    0.0%   724.39
heuristic         100      4,604       [4,164, 5,044]     4,248    13,696   1,024    0.0%    98.12
expectimax        100     14,584     [13,454, 15,714]    15,400    26,952   2,048   12.0%     1.40
learned           100     54,045     [49,225, 58,865]    52,176   123,748   8,192   89.0%     5.45
```

Expectimax strength against speed (6 games per row):

```
depth  adaptive  cutoff   mean score   games/s
  1       no      1e-3         5,563     89.0
  2       no      1e-3        11,061      1.71   <- default
  3       no      1e-2        13,997      0.18
  3       no      5e-3        19,735      0.13
  1      yes      1e-3        16,143      0.29
  2      yes      1e-2        15,857      0.10
```

### Training progress

Measured during the 66,000-game run shown in the screenshot, default 4×6
network:

| games trained | rolling mean score | best tile seen | 2048 rate |
|---|---|---|---|
| 1,000 | 7,858 | 2,048 | 0.1% |
| 5,000 | 15,653 | 2,048 | 14.3% |
| 10,000 | 22,019 | 4,096 | 37.8% |
| 20,000 | 30,665 | 4,096 | 63.9% |
| 30,000 | 37,801 | 8,192 | 75.0% |
| 50,000 | 49,138 | 8,192 | 83.6% |
| 66,000 | 55,208 | 8,192 | 87.5% |

The rolling 2048 rate passed 50% at about 13,500 games, roughly **22 minutes**
into that run on this laptop; the whole 66,000 games took about **2.5 hours** of
active training. The run had not stopped improving when it was stopped.

### Throughput

```
fresh 4x6 run, 1 worker:   ~28 games/s    ~12,200 moves/s
fresh 4x6 run, 2 workers:  ~54 games/s    ~21,600 moves/s
strong agent, 1 worker:    ~5-6 games/s   ~14,600 moves/s
```

Games per second **falls as the agent improves** — a game that reaches 4096
takes several thousand moves instead of a few hundred. Moves per second is the
honest measure of throughput, and it stays roughly flat at 12,000–15,000 per
core.

### Other hardware, for scale

The same engine benchmark on GitHub's CI runners, which is a useful reminder of
how much the hardware matters:

```
Dell Latitude 3300 (Celeron 3865U)     1,200 games/s     141,475 moves/s
GitHub windows-latest runner           2,639 games/s     310,971 moves/s
GitHub macos-latest runner (arm64)     3,743 games/s     441,027 moves/s
```

Training throughput scales similarly, so the "~50 minutes for 20,000 games"
figure above is close to a worst case rather than a typical one.

### Memory and disk

| Network | Weights | File size |
|---|---|---|
| `4x6` (default) | 67.1 M | 268 MB |
| `4x5` | 4.2 M | 17 MB |
| `8x4` | 0.5 M | 2 MB |

Plus roughly 60 MB of engine and index lookup tables per process. On Linux and
macOS the weight file is created sparse, so it only occupies what has actually
been written; on Windows it occupies its full size immediately.

---

## Platform support

| Platform | Status |
|---|---|
| **Linux** | Developed and hand-tested here: the complete test suite, long real training runs, and a from-scratch clone-and-run. Also `ubuntu-latest` in CI. |
| **Windows** | Reviewed for Windows specifically, and the **complete test suite passes on `windows-latest`** in CI on every push — including the Ctrl-Break interrupt path and two-worker training. Not hand-tested on physical Windows hardware. |
| **macOS** | Same: reviewed, and the **complete test suite passes on `macos-latest`** (Apple silicon) in CI on every push. Not hand-tested on physical Apple hardware. |

CI covers Python 3.10, 3.12 and 3.13 on each of the three.

The `tests` badge at the top of this page shows the current state of all three;
the [Actions tab](../../actions) has the per-platform detail.

What was specifically made portable rather than assumed:

- **Paths.** Every directory is derived from the source file's own location
  with `pathlib`, so nothing depends on the working directory, on `$HOME`, or
  on `/` as a separator. There are no hardcoded paths anywhere in the code.
- **Multiprocessing.** Workers use `fork` on Linux and `spawn` on Windows and
  macOS; on `spawn` they receive the weight file's path and map it themselves,
  which gives the same shared memory, because every one of the three operating
  systems keeps mappings of the same file coherent between processes. All
  entry points are `if __name__ == "__main__":` guarded, which `spawn`
  requires. Both code paths are exercised by the test suite on every platform
  (`AI2048_START_METHOD` forces one).
- **Process liveness.** The dashboard checks whether a trainer is still alive.
  The POSIX idiom for this is `os.kill(pid, 0)` — on Windows that would
  *terminate* the training process, so Windows gets a read-only handle probe
  instead.
- **Interrupts.** Ctrl-C saves and exits cleanly everywhere. Windows also
  delivers Ctrl-Break, which is handled identically; `SIGTERM` is registered
  only where it exists.
- **Text files.** Every JSON and JSONL file is read and written as UTF-8
  explicitly, rather than inheriting a locale-dependent default.
- **Atomic writes.** `os.replace` is atomic on all three, but Windows refuses
  to replace a file another process has open — which the dashboard briefly does
  on every poll — so the checkpoint writer retries briefly instead of failing.

Known differences that are documented rather than papered over:

- `spawn` costs each worker a few seconds of lookup-table building at start-up.
  Single-worker training is identical everywhere.
- Weight files are sparse on Linux/macOS and fully allocated on Windows.
- The dashboard lowers its own priority with `os.nice` on POSIX; there is no
  equivalent call on Windows, so it simply does not.

---

## Project structure

```
2048-ai/
├── train.py                   train an agent            (--help for options)
├── evaluate.py                evaluate and compare agents
├── experiment.py              run controlled experiments
├── server.py                  launch the dashboard (--stop, --restart)
├── start.sh                   one-command start: finds Python, runs server.py
├── stop.sh                    stop the dashboard gracefully from any terminal
├── version.py                 version string, reported by --version
│
├── engine/
│   ├── board.py               bitboard core: move tables, spawn, queries
│   ├── benchmark.py           throughput measurements, shared by CLI and UI
│   └── game.py                small stateful wrapper used outside training
│
├── agents/
│   ├── base.py                Agent interface + RandomAgent
│   ├── heuristic.py           hand-written evaluator + one-ply agent
│   ├── expectimax.py          probability-cut expectimax search
│   ├── learned.py             wrapper around a trained n-tuple network
│   └── registry.py            name → agent factory
│
├── training/
│   ├── ntuple.py              n-tuple network: tables, mmap, generated code
│   ├── learner.py             TD(0) afterstate learning
│   ├── reward.py              reward functions and why most of them are traps
│   ├── stats.py               rolling / all-time statistics, history file
│   ├── checkpoint.py          run directories, atomic saves, resume
│   ├── config.py              config loading and merging
│   └── trainer.py             the training loop, workers, reporting
│
├── evaluation/evaluator.py    fixed seeded evaluation + confidence intervals
├── experiments/runner.py      experiment execution and comparison tables
│
├── dashboard/                 the control center
│   ├── server.py              HTTP transport, SSE, static files, guards
│   ├── api.py                 routing, validation and payload building
│   ├── jobs.py                job manager: supervised subprocesses
│   ├── runner.py              job worker for evaluation/compare/benchmark
│   ├── games.py               AI and human game sessions (Python engine)
│   ├── store.py               checkpoint ids, saved results, UI settings
│   ├── sysinfo.py             platform diagnostics, degrading gracefully
│   ├── events.py              the event log behind the Logs page
│   └── static/                HTML, CSS and JS — no CDN, no framework, no build
│
├── config/
│   ├── default.json           the default configuration
│   └── experiments/           18 experiment configs
│
├── tests/                     337 tests + the engine benchmark
├── docs/
│   ├── ARCHITECTURE.md        how the pieces fit together, and why
│   ├── COMMANDS.md            command reference, with Windows equivalents
│   └── images/                documentation screenshots
│
├── checkpoints/               created on first run — not in Git
└── data/                      created on first run — not in Git
```

---

## Security and networking

This project makes **no network connections at all**. It downloads nothing,
uploads nothing, and contacts no telemetry service. The dashboard's front end
loads no CDN scripts, fonts or stylesheets.

The one thing that listens is the control center, and it binds to
**127.0.0.1** — your own machine only — unless you explicitly pass `--host`.

That default matters more than it used to. This server can **start processes,
delete files and shut itself down**, so its endpoints are privileged local controls rather than
a read-only dashboard. It has no authentication, and three things enforce the
boundary instead:

- **Loopback by default.** A non-loopback `--host` prints a warning that
  recommends an SSH tunnel instead. Believe it.
- **A required custom header.** Every mutating request must carry
  `X-2048-Request`, which a cross-origin page cannot set without a CORS
  preflight that is never granted — so a random website you visit cannot POST
  to your training server. Any `Origin` header present must be loopback.
- **No filesystem path ever comes from the browser.** Checkpoints are
  addressed by identifier, validated against a strict pattern, rebuilt into a
  path, and then verified to still be inside the checkpoint directory. Run
  names must be safe single path components on every platform.

Inputs are validated throughout: agent names and job actions are allowlists,
numbers are range-checked, search depth is clamped, malformed or oversized
bodies get a 400 or 413, and static serving is confined to `dashboard/static/`
with traversal rejected. Job commands are built as explicit argument lists —
nothing is ever passed through a shell. Responses carry
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, a
`Content-Security-Policy` that allows scripts only from this origin, and no
CORS headers at all.

For remote access, do not expose the port. Forward it:

```bash
ssh -L 8000:127.0.0.1:8000 you@the-machine
```

The repository contains no credentials, keys or tokens, and `.gitignore`
carries patterns for the usual secret filenames so one cannot be committed by
accident.

---

## Git workflow

The repository has one long-lived branch, **`main`**, on GitHub at
`origin`. Everything below is run from inside the project folder.

### The everyday commands

```bash
git status
```

Shows which files have changed, which of those are staged (queued for the next
commit), which are new and untracked, and which branch you are on. Run it
before and after everything else; it is always safe.

```bash
git pull
```

Downloads the latest commits from GitHub and merges them into your current
branch, so you are working on the newest version. Do this before you start
changing things, and again before you push.

```bash
git diff
```

Shows exactly what you have changed, line by line, that is **not staged yet**.
`git diff --staged` shows what **is** staged, i.e. what the next commit will
contain. Press `q` to leave the viewer.

```bash
git add .
```

Stages every changed and new file in the project folder for the next commit.
It is safe here because `.gitignore` already excludes checkpoints, training
data, caches, logs and editor files — but check `git status` first. To stage
only some files, name them: `git add README.md start.sh`.

```bash
git commit -m "Describe the change"
```

Records the staged changes as one snapshot in your local history, with a
message saying what changed and why. Nothing leaves your computer yet.

```bash
git push
```

Uploads your new commits to GitHub so they are backed up and visible there.
The first time you push a new branch, use `git push -u origin <branch>`; after
that, plain `git push` is enough.

```bash
git log --oneline
```

Lists the commits on the current branch, newest first, one line each: a short
id and the message. Add `-10` to see only the last ten. Press `q` to leave.

```bash
git restore <file>
```

Throws away your uncommitted changes to that file and puts back the last
committed version. **This cannot be undone**, so look at `git diff <file>`
first. (`git restore --staged <file>` is harmless: it only un-stages the file
and keeps your edits.)

### The normal workflow when you change something

```bash
git status                            # 1. start from a known state
git pull                              # 2. get the latest version first
#    ... edit files ...
./start.sh --restart                  # 3. run it and check your change works
python3 -m unittest discover -s tests # 4. make sure nothing else broke
git status                            # 5. see what changed
git diff                              # 6. read your own changes
git add .                             # 7. stage them
git commit -m "Describe the change"   # 8. record them
git push                              # 9. upload them
```

Why each step:

1. **`git status`** — confirms you are on the branch you think you are, and
   that there is nothing half-finished from last time mixed in.
2. **`git pull`** — starting from the newest code avoids conflicts later.
3. **`./start.sh --restart`** — the running server keeps the code it started
   with, so restart it to test what you actually changed.
4. **Tests** — about 90 seconds, and they cover the things that are easy to
   break by accident (see [Running tests](#running-tests)).
5. **`git status`** — the list of files that will go into the commit. If
   `checkpoints/`, `data/`, a `.f32` file or anything large appears, stop:
   it should be ignored, not committed.
6. **`git diff`** — catches debugging leftovers and accidental edits before
   they become permanent.
7. **`git add .`** — chooses what goes into the commit.
8. **`git commit`** — one commit per logical change, with a message that says
   what and why ("Fix restart leaving training running", not "updates").
9. **`git push`** — publishes the commit. If it is rejected because GitHub has
   newer commits, run `git pull`, then `git push` again.

After **any** `git pull` that brings in Python changes, restart the control
center (`./start.sh --restart`) so it runs the new code.

### Working on a branch

For anything bigger than a small fix, a branch keeps `main` working while you
experiment:

```bash
git switch main && git pull           # start from the latest main
git switch -c my-change               # create a branch and move onto it
#    ... edit, test, add, commit as above, as often as you like ...
git push -u origin my-change          # upload the branch (first time)
```

Then open a **pull request** from `my-change` into `main` on GitHub, check the
diff and the CI tests there, and merge it. Afterwards:

```bash
git switch main                       # back to main
git pull                              # bring in the merged change
git branch -d my-change               # delete the finished local branch
```

`git branch` lists your branches, with `*` next to the current one.

### What never goes into Git

`.gitignore` keeps these out, and it is worth knowing why:

- **`checkpoints/`** — trained weights. The default network is 268 MB, and
  GitHub rejects any file over 100 MB. To share a trained agent, attach the
  `.f32` file to a GitHub **Release** instead.
- **`data/`** — training history, evaluations, job logs and rebuildable
  lookup tables: specific to your machine and regenerated as you use it.
- **`__pycache__/`, `.venv/`, test caches, `*.log`, `nohup.out`** — generated
  files.
- **`.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`** — editor and
  operating-system clutter.
- **`.env`, keys and anything named like a credential** — the project needs no
  secrets, so none should ever be committed by accident.

If a file that should be ignored was committed earlier, `git rm --cached
<file>` stops tracking it without deleting your copy; commit that change.

---

## Contributing

Bug reports, questions and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). The short version: keep the zero-dependency
rule, run `python3 -m unittest discover -s tests` before you push, and if you
change something for performance, include the measurement.

---

## License

[MIT](LICENSE) — do what you like with it, keep the copyright notice, no
warranty.

---

## References

The learning method here follows the line of work that made n-tuple networks
the standard approach for 2048:

- M. Szubert and W. Jaśkowski, *Temporal Difference Learning of N-Tuple
  Networks for the Game 2048* (IEEE CIG, 2014) — the afterstate TD formulation
  used here.
- K.-H. Yeh, I-C. Wu, C.-H. Hsueh, C.-C. Chang, C.-C. Liang and H. Chiang,
  *Multi-Stage Temporal Difference Learning for 2048-like Games* (2016).
- A. Y. Ng, D. Harada and S. Russell, *Policy Invariance Under Reward
  Transformations* (ICML, 1999) — why the optional shaping terms cannot change
  the optimal policy.
- B. Recht, C. Re, S. Wright and F. Niu, *Hogwild!: A Lock-Free Approach to
  Parallelizing Stochastic Gradient Descent* (NeurIPS, 2011) — the basis for
  `--workers`.
