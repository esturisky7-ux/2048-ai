# Architecture

How the pieces fit together, and why each one is built the way it is.

This document assumes you can read Python but not that you know anything about
reinforcement learning or 2048 solvers.

- [The pipeline](#the-pipeline)
- [The bitboard](#the-bitboard)
- [Precomputed row tables](#precomputed-row-tables)
- [Agents](#agents)
- [The n-tuple network](#the-n-tuple-network)
- [Why not a neural network](#why-not-a-neural-network)
- [TD(0) on afterstates](#td0-on-afterstates)
- [Rewards](#rewards)
- [Checkpoints and crash safety](#checkpoints-and-crash-safety)
- [Multiprocessing and worker training](#multiprocessing-and-worker-training)
- [Evaluation](#evaluation)
- [The dashboard](#the-dashboard)
- [Module map](#module-map)

---

## The pipeline

Every single move in training goes through this path:

```
                         2048 board
                              │
                              ▼
                     ┌─────────────────┐
                     │ bitboard engine │   engine/board.py
                     └─────────────────┘   one 64-bit int; moves are 4 table lookups
                              │
                              ▼
                    ┌───────────────────┐
                    │ agent chooses a   │  training/learner.py
                    │ move              │  argmax over legal a of r(a) + V(afterstate(a))
                    └───────────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │afterstate│      the board after the slide,
                        └──────────┘      before the game reacts
                              │
                              ▼
            ┌───────────────────────────────────┐
            │ environment spawns a random tile  │  engine/board.py: random_spawn()
            │ (90% a 2, 10% a 4, uniform cell)  │
            └───────────────────────────────────┘
                              │
                              ▼
                         next state
                              │
                              ▼
                     ┌─────────────────┐
                     │   TD update     │   training/learner.py
                     └─────────────────┘   V(s') ← V(s') + α·[r' + γ·V(s'') − V(s')]
                              │
                              ▼
                   ┌─────────────────────┐
                   │   n-tuple weights   │  training/ntuple.py
                   └─────────────────────┘  67M float32 in a memory-mapped file
                              │
                              ▼
                      ┌──────────────┐
                      │  checkpoint  │     training/checkpoint.py
                      └──────────────┘     flush dirty pages + atomic meta.json
```

The loop runs roughly 12,000–15,000 times a second per core. Every design
decision below exists because of that number: anything on this path that
allocates an object, looks up an attribute or calls an unnecessary function
costs real training throughput.

---

## The bitboard

A 2048 board is 16 cells, each holding a power of two up to 32768, or nothing.
Storing the **exponent** rather than the value means each cell fits in 4 bits:

```
0 → empty    1 → 2     2 → 4     3 → 8    ...   11 → 2048
12 → 4096    13 → 8192   14 → 16384   15 → 32768
```

Sixteen cells × 4 bits = **64 bits**, so an entire board is a single Python
`int`. Cell `i` (row-major, `i = 4·row + col`) occupies bits `[4i, 4i+3]`.

```
 cell indices          bit layout
 0  1  2  3            row 0 = bits  0..15
 4  5  6  7            row 1 = bits 16..31
 8  9 10 11            row 2 = bits 32..47
12 13 14 15            row 3 = bits 48..63
```

Why this matters:

- **No allocation.** Copying a board is copying an integer. A 4×4 list of lists
  would allocate five objects per move and dominate the profile.
- **Whole rows are indices.** A row is 16 contiguous bits — a number from 0 to
  65535 — which can be used *directly* as an array index. That is what makes
  the next section possible.
- **Cheap queries.** "Is the game over?", "how many empty cells?", "what is the
  largest tile?" all become a handful of shifts and table lookups.

Everything in `engine/board.py` is a free function taking and returning `int`.
There is no board class on the hot path, no rendering and no I/O, on purpose.

---

## Precomputed row tables

Because a row is a 16-bit number, we can precompute *every possible row* — all
65,536 of them — exactly once at import, and then never think about merge
rules again.

`engine/board.py` builds, for every row value:

| Table | Contents |
|---|---|
| `_ROW_LEFT`, `_ROW_RIGHT` | the row after sliding left / right |
| `_SCORE_LEFT`, `_SCORE_RIGHT` | the merge score that slide produces |
| `_L[r]`, `_R[r]` | the slid row **pre-shifted** into row position `r` |
| `_U[c]`, `_D[c]` | the slid column written straight back into column `c` |
| `_EC[r]` | the empty cell indices in that row, as absolute board indices |
| `_EMPTY`, `_MAXEXP` | empty count and largest exponent in that row |

A horizontal move becomes four lookups and three ORs:

```python
nb = _L0[r0] | _L1[r1] | _L2[r2] | _L3[r3]
sc = _SCORE_LEFT[r0] + _SCORE_LEFT[r1] + _SCORE_LEFT[r2] + _SCORE_LEFT[r3]
```

Vertical moves need the board transposed so columns become rows. The obvious
implementation transposes, slides, and transposes back — two transposes. The
`_U`/`_D` tables **fold the un-transpose into the table itself**: entry
`_U[c][row]` scatters the slid line's nibbles straight into column `c` of a
normally-oriented board. That removed one transpose per vertical move and was
measured at 3.62 µs → 2.29 µs. It costs about 4 MB of tables, which is the
trade this project makes repeatedly: memory is cheap, Python bytecode is not.

Three properties come free from this design:

1. **One source of truth for the rules.** `_slide_line_left()` implements merge
   semantics once, in the most obvious way possible, and runs 65,536 times at
   import and never again. The subtle rule — a tile formed by a merge cannot
   merge again in the same move — lives in exactly one place.
2. **Testability.** The test suite verifies all 65,536 rows against an
   independently written reference implementation. There is no "most rows
   probably work".
3. **Speed without C.** ~140,000 moves/second in pure CPython on a 1.8 GHz
   Celeron.

The same idea appears twice more: the heuristic evaluator precomputes every
line-local feature into a 65,536-entry table, and the n-tuple network
precomputes row→index tables (below).

---

## Agents

An agent is anything with `act(board) -> direction`. They take a raw `int`, not
a game object, so the same agent works in training, evaluation and the
dashboard with no adapters.

| Agent | Strategy | Cost per move |
|---|---|---|
| `random` | uniform over legal moves | one `legal_actions` call |
| `heuristic` | best static evaluation of the immediate afterstate | 4 × (move + 12 lookups) |
| `expectimax` | search a max/chance tree, static evaluation at leaves | thousands of nodes |
| `learned` | best `reward + V(afterstate)` using the trained network | 4 × (move + 32 lookups) |

`expectimax` models the game's randomness explicitly: *max* nodes are the
player's choice, *chance* nodes average over "a 2 with probability 0.9 or a 4
with probability 0.1, uniformly over empty cells". Its cost is bounded three
ways — a depth limit, a probability cutoff that abandons lines rarer than a
threshold, and a transposition table for positions reachable by several move
orders.

The probability cutoff is the interesting one: it makes the effective depth
*adaptive for free*. A chance node with ten empty cells splits probability
twenty ways, so lines die off quickly on an open board and survive longer on a
crowded one, which is exactly where accuracy matters.

---

## The n-tuple network

The learned agent needs `V(board)`, an estimate of remaining score. There are
far too many boards to tabulate, so `V` is a **sum of lookups over small cell
patterns**.

A *tuple* is a set of board cells. The default `4x6` set uses four 6-cell
patterns:

```
 0  1  2  3      A = (0,1,2,3,4,5)     ■ ■ ■ ■     two horizontal strips
 4  5  6  7      B = (4,5,6,7,8,9)     ■ ■ · ·
 8  9 10 11      C = (0,1,2,4,5,6)     · · · ·     and two 2×3 blocks
12 13 14 15      D = (4,5,6,8,9,10)    · · · ·
```

For a 6-cell tuple, the lookup index packs six 4-bit exponents into 24 bits, so
that tuple's table has 16<sup>6</sup> = 16.7 M entries. Four tuples gives
**67 million float32 weights = 268 MB**.

```
V(board) = Σ over tuples t, Σ over the 8 symmetries s:  W[ index(t, s, board) ]
```

Three implementation details do the heavy lifting:

**Symmetry sharing.** The board has no intrinsic orientation, so a pattern
learned in one corner should transfer to the other three. Every tuple is
evaluated under all 8 symmetries of the square (4 rotations × 2 reflections)
into the *same* weight table. Effective training data ×8, at no memory cost.
4 tuples × 8 symmetries = 32 lookups per evaluation.

**Row-partial index tables.** Building an index the obvious way costs a
shift-and-mask per cell. Instead, for each (tuple, symmetry) variant we
precompute a table per board row that maps that row's 16-bit value to the bits
of the index it contributes. An index is then one array read per row the
pattern touches, OR'd together. The tuple's base offset is folded into the
first table so the generated code needs no addition at all. These tables take a
few seconds to build and are cached in `data/tables/idx-*.bin`, so a second
process (a worker, the evaluator, the dashboard) pays ~0.2 s instead of ~4 s.

**Generated straight-line code.** `value` and `update` are *emitted as Python
source* — fully unrolled over all 32 variants with every table bound as a local
— then compiled with `exec`. An interpreted loop over the variants is far
slower. The generated source is kept on the instance as `.source`, so it stays
inspectable:

```python
>>> from training.ntuple import NTupleNetwork
>>> print(NTupleNetwork("8x4").source[:200])
```

**Storage.** Weights live in a memory-mapped file of float32. That single
choice buys three things at once: several processes share one copy (so
`--workers` needs no IPC for weights), checkpointing only has to flush dirty
pages rather than serialise 268 MB, and a crash leaves almost-current weights
already on disk.

Three network shapes ship:

| Set | Tuples | Weights | Size | Use |
|---|---|---|---|---|
| `4x6` | 4 × 6 cells | 67.1 M | 268 MB | default; strongest |
| `4x5` | 4 × 5 cells | 4.2 M | 17 MB | much lighter, a little weaker |
| `8x4` | 8 × 4 cells | 0.5 M | 2 MB | fast experiments and tests |

---

## Why not a neural network

The honest answer is that on this problem an n-tuple network is both faster and
stronger, so a neural network would be a worse tool chosen for fashion.

- **Cost per update.** A network evaluation here is 32 array reads; a learning
  update is 32 additions. Even a small MLP needs matrix multiplies, an
  activation, a backward pass and an optimiser step — on a GPU-less dual-core
  CPU that is roughly two orders of magnitude more work per update. Since
  learning quality is driven by *how many updates you can afford*, this is not
  a small difference.
- **The structure fits.** 2048's value function is dominated by local
  configurations: a monotone strip along an edge, a big tile wedged in a
  corner, a pair ready to merge. An n-tuple network is exactly a linear model
  over an enormous sparse one-hot encoding of those local patterns. A dense
  network would have to *discover* that structure; here it is given.
- **Stability.** A linear model with a fixed feature map cannot diverge the way
  a bootstrapped nonlinear approximator can. There is no learning-rate warm-up,
  no target network, no replay buffer, no gradient clipping — none of the
  machinery that exists to keep deep RL from falling over.
- **It is what actually works.** The strongest published 2048 agents use
  n-tuple networks (Szubert & Jaśkowski 2014; Wu et al. 2014), not deep nets.

The trade is memory: 268 MB of weights for a problem a neural network could
address with a few hundred kilobytes of parameters. On a laptop with 8 GB of
RAM that is an easy trade, and the file is created sparse so it only grows as
positions are actually visited.

---

## TD(0) on afterstates

A move in 2048 has two halves:

```
state s  ──player slides──►  afterstate s'  ──game drops a tile──►  next state s''
         (deterministic)                      (random)
```

**Learning the value of `s'` rather than `s` is the single most important design
decision in this project.** The afterstate is entirely determined by the
player's choice, so its value is a property of the decision alone — all the
randomness has been pushed out of the thing being learned. Choosing a move is
then just

```
argmax over legal a of   reward(a) + V(afterstate(a))
```

with **no search and no model of the spawn distribution**: one network
evaluation per legal move, about four per turn. A state-value formulation would
have to average over every possible tile spawn to compare two moves, which is
both slower and noisier.

The update, after the game drops its tile and the agent picks its next move
(giving reward `r'` and afterstate `s''`):

```
V(s')  ←  V(s')  +  α · [ r' + γ·V(s'')  −  V(s') ]
```

and at the end of a game the target is just the terminal reward — 0 by default.
That is what teaches the agent that dying forfeits all future score; nobody has
to encode "survive" as a goal.

Because `V` is linear in the n-tuple features, applying that correction means
adding the same small delta to each of the 32 activated weights. `α` is divided
by 32 internally, so the configured `alpha` means "how far `V(s')` moves
towards its target" rather than "how much each weight changes".

`γ = 1.0` is correct here, not a simplification: 2048 is a finite episodic game
and undiscounted score is exactly what we want to maximise.

**Exploration is off by default.** The tile spawns already inject plenty of
stochasticity, every game visits a different part of the state space, and
ε-greedy is known to *hurt* on 2048 — a single random move late in a game can
destroy a structure that took a thousand moves to build. The `epsilon` knob
exists for experiments and is on-policy when used.

---

## Rewards

The default reward is the **merge score and nothing else**, because the sum of
merge scores over a game is exactly the game's final score. Maximising expected
return is then literally maximising expected score, with no proxy and no
mismatch.

`training/reward.py` documents why every other tempting signal is a trap:

| Signal | Failure mode |
|---|---|
| per-move survival bonus | rewards stalling: shuffle tiles forever, farm reward |
| empty-cell bonus | rewards *not merging*; an agent can hold an empty board |
| holding a big tile | rewards sitting still (so milestones are paid **once**, on creation) |
| corner / monotonicity bonuses | hands the agent the strategy instead of letting it find one |

The shaping terms that *are* offered are implemented as **potential-based
shaping** — the agent receives `γ·φ(s') − φ(s)`. Ng, Harada & Russell (1999)
proved this form leaves the optimal policy unchanged: it can only change how
fast learning gets there, never what it converges to. That makes the shaped
variants safe to experiment with without silently training a different game.
The test suite checks that the shaping telescopes to zero over a round trip.

---

## Checkpoints and crash safety

```
checkpoints/<run>/weights.f32     memory-mapped float32 weights
checkpoints/<run>/meta.json       games, records, RNG seed, alpha, last eval
checkpoints/<run>/config.json     the configuration the run started with
checkpoints/<run>/snapshots/      frozen weight copies (--snapshot-every)
data/<run>/history.jsonl          one row per report interval, for graphs
data/<run>/evaluations.jsonl      every fixed evaluation
data/<run>/status.json            heartbeat the dashboard polls
```

Two different mechanisms, because the two kinds of data have different needs:

**Weights: memory-mapped, flushed.** The kernel is already writing dirty pages
back as training proceeds. A checkpoint calls `flush()`, which only moves pages
that actually changed — nothing serialises 268 MB. After a hard power loss the
file on disk is almost current.

**Metadata: atomic replace.** `meta.json` is small, so it is written to a
temporary file, `fsync`'d, and then `os.replace`'d over the real one. That
rename is atomic on every supported OS, so a reader never sees a half-written
file. On Windows the replace can fail transiently if a reader has the file
open, so the writer retries briefly rather than losing the checkpoint.

`history.jsonl` is append-only and flushed per row: a crash loses at most the
row being written, and the reader tolerates a torn final line rather than
discarding the file.

Ctrl-C is not an interruption to be survived but a supported way to stop: the
handler sets a flag, the loop finishes the current game, and the `finally`
block checkpoints. A second Ctrl-C exits immediately.

---

## Multiprocessing and worker training

`--workers N` starts N processes that all play games and all update **one
shared weight table**.

```
   parent process                     worker 0            worker 1
   ┌──────────────┐                 ┌──────────┐        ┌──────────┐
   │ aggregates   │◄── results ─────│ plays    │        │ plays    │
   │ stats,       │     (batched)   │ games,   │        │ games,   │
   │ checkpoints, │◄────────────────│ updates  │        │ updates  │
   │ reports      │                 └────┬─────┘        └────┬─────┘
   └──────────────┘                      │                   │
          │                              ▼                   ▼
          └────────────────► checkpoints/<run>/weights.f32 ◄──┘
                              one mmap, shared by all
```

**The weights are never sent between processes.** Every process maps the same
file, and every operating system this project supports guarantees that mappings
of one file are coherent across processes. Writing a weight in one worker is
immediately visible in the other.

**No locking.** This is the Hogwild! scheme: an update touches ~32 of tens of
millions of weights, so two workers colliding on the same weight in the same
instant is rare, and the occasional lost update is small noise next to the TD
error itself. Locking would cost far more than it saves.

Only *results* go over a queue, batched 16 games at a time — per-game IPC would
cost more than a game does. The parent never plays, so it stays responsive to
Ctrl-C and can checkpoint on schedule.

**Start methods.** Linux uses `fork`: the worker inherits the parent's already
built move tables and open mapping, so it starts instantly and costs no extra
memory. Windows has no `fork`, and on macOS forking is unsafe once system
frameworks are loaded, so both use `spawn` — a fresh interpreter that is handed
`(tuple_set, weights_path)` and maps the file itself. The result is identical
Hogwild semantics; the only cost is a few seconds of table building per worker
at start-up. Set `AI2048_START_METHOD=spawn` to exercise that path on Linux;
the test suite does exactly this so both paths are tested everywhere.

Because `spawn` re-imports the main module in the child, every entry point is
guarded with `if __name__ == "__main__":`.

**Determinism.** Each game's tile spawns come from `Random(game_seed(seed, i))`,
so with one worker a run is bit-for-bit reproducible and resuming continues the
same sequence. With several workers the games are still seeded, but weight
updates interleave between processes, so reruns are statistically rather than
literally identical. That is the price of Hogwild, and it is stated rather than
hidden.

---

## Evaluation

Training statistics answer "is it improving right now?" but they are a moving
target: the agent changes while the numbers are being collected, and online
updates add noise.

Evaluation is deliberately separate. It **freezes the policy** and replays the
**same seeded games** every time, so two evaluations differ only because the
agents differ. `game_seed(base, i)` is a deterministic mixing function, so game
`i` of an evaluation is always the same game.

Every figure comes with an interval:

- the **mean** gets a normal-approximation confidence interval — scores are
  skewed, but the mean of hundreds of them is not, so the CLT applies;
- **tile achievement rates** are proportions, so they get **Wilson score
  intervals**, which stay correct near 0% and 100% where the usual normal
  approximation produces bounds outside [0, 1].

`evaluate.py --compare a b c` runs every agent over the identical game set, so
differences between agents are differences in play and not luck.

---

## The dashboard

`http.server` from the standard library, no framework, no dependencies, and a
front end of hand-written HTML/CSS/JS that loads nothing from the internet —
including the charts, which are drawn on `<canvas>` directly.

```
browser ──HTTP──► dashboard/server.py ──reads──► data/<run>/status.json
                         │                       data/<run>/history.jsonl
                         │                       data/<run>/evaluations.jsonl
                         │                       checkpoints/<run>/meta.json
                         │
                         └──► dashboard/live.py ──► weights.f32 (READ-ONLY)
                                    plays its own game in a thread
```

**The dashboard never talks to the trainer.** It only reads files the trainer
writes, and for live games it opens the weights through a *read-only* mapping,
so nothing it does can disturb or corrupt a run in progress.

The live game is played in a background thread that stays only ~48 frames ahead
of what the browser has consumed. That buffer *is* the throttle: at 1× playback
it computes about eight moves a second and leaves the rest of the CPU to
training. The server process also lowers its own scheduling priority at
start-up where the OS supports it.

Liveness detection: the trainer heartbeats `status.json` every three seconds; a
status older than fifteen seconds, or one whose PID no longer exists, is
reported as stopped rather than as phantom training. The PID check is a
read-only handle probe on Windows, because the POSIX `os.kill(pid, 0)` idiom
would *terminate* the trainer there.

---

## Module map

| Module | Responsibility |
|---|---|
| `engine/board.py` | bitboard primitives: move tables, spawn, queries. No I/O. |
| `engine/game.py` | small stateful wrapper for code that is not the hot loop |
| `agents/base.py` | the `Agent` interface and `RandomAgent` |
| `agents/heuristic.py` | precomputed static evaluator + one-ply agent |
| `agents/expectimax.py` | max/chance search with probability cutoff |
| `agents/learned.py` | read-only wrapper around a trained network |
| `agents/registry.py` | `make_agent(name, **kw)` used by every entry point |
| `training/ntuple.py` | the network: tuples, symmetries, index tables, mmap, codegen |
| `training/learner.py` | TD(0) afterstate learning; the innermost loop |
| `training/reward.py` | reward functions and their failure modes |
| `training/stats.py` | rolling window, all-time aggregates, history file |
| `training/checkpoint.py` | run directories, atomic writes, resume |
| `training/config.py` | layered configuration (defaults → file → CLI) |
| `training/trainer.py` | the loop, reporting, checkpoint scheduling, workers |
| `evaluation/evaluator.py` | fixed seeded evaluation and its statistics |
| `experiments/runner.py` | run a config, evaluate, store config + result |
| `dashboard/server.py` | HTTP server and JSON API |
| `dashboard/live.py` | throttled live-game thread |

Dependency direction is strictly one way: `engine` knows about nothing,
`agents` and `training` know about `engine`, `evaluation` knows about `engine`,
and `dashboard` and the entry points sit on top of all of it. Nothing in
`engine/` imports anything from the project.
