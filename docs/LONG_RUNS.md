# Long runs on a Linux server or VM

How to start a training run that lasts hours or days on a Linux machine — a
Google Compute Engine VM, any cloud instance, a box in a closet — and leave it
alone. Nothing here needs more than the standard library and a shell.

## The short version

```bash
git clone <this repository> && cd 2048-ai
python3 train.py --check                       # does it work here?

# Start (or, after any interruption, continue) a run on every CPU:
./deploy/long-run.sh
```

`deploy/long-run.sh` is one command that **starts the run if it does not exist
and continues it if it does**, so it is safe to run again after anything. It
uses every CPU the user may use (`--workers auto`), and stops by itself if you
give it a total:

```bash
TARGET_GAMES=5000000 ./deploy/long-run.sh
RUN=big WORKERS=6 ./deploy/long-run.sh --eval-every 100000
```

Anything after the script is passed to `train.py`. Run it somewhere that
outlives your SSH session — one of the three ways [below](#starting-it).

## Choosing the machine

Training is pure Python and CPU-bound, so the thing to buy is **CPUs**.

- **It scales with workers.** On the 4-core machine this was developed on, a
  fresh default (4×6) run played 39,600 moves/s with one worker and 148,900
  with four — 3.8×. Workers share one weight table, so adding one costs almost no
  memory: about 5 MB each. (A strong agent plays fewer, much longer games, so
  compare moves/s rather than games/s.)
- **Memory is not the constraint.** The weights are one 268 MB file shared by
  every worker, plus roughly 60 MB of lookup tables. A 32-worker run needs well
  under a gigabyte.
- **Disk:** keep about 1 GB free, plus 268 MB for each snapshot you ask for.
  The weight file's space is reserved when training starts (see below), so a
  full disk refuses to start the run rather than crashing it later.
- **Put the data on a persistent disk.** `AI2048_HOME` moves `checkpoints/` and
  `data/` somewhere else. On a cloud VM use a persistent disk, not a local SSD:
  local SSD contents are lost when the VM is stopped, which is what happens to
  a preempted Spot VM.

`--workers auto` counts the CPUs the process is *allowed* to run on
(`os.sched_getaffinity`), not the machine's, so it behaves under `taskset`, a
systemd `CPUAffinity=` and container CPU sets. Where a cloud vCPU is a
hyperthread rather than a whole core, expect less than linear scaling once the
workers outnumber the physical cores.

## Starting it

Pick one. All of them stop the same safe way.

**tmux** (or `screen`) — simplest. The run lives in the terminal multiplexer,
not in your SSH session:

```bash
tmux new -s train
./deploy/long-run.sh
# detach with Ctrl-b d; come back later with: tmux attach -t train
```

**nohup** — no multiplexer needed:

```bash
mkdir -p logs
nohup ./deploy/long-run.sh >> logs/main.log 2>&1 &
tail -f logs/main.log
```

**systemd** — the right choice for something that must survive a reboot or a
preemption, because an enabled unit starts again on boot and carries on from
the last checkpoint:

```bash
sed -e "s|@USER@|$USER|g" -e "s|@DIR@|$PWD|g" deploy/2048-train.service \
    | sudo tee /etc/systemd/system/2048-train.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now 2048-train

journalctl -u 2048-train -f          # progress lines
sudo systemctl stop 2048-train       # stop safely: checkpoint, then exit
```

Edit the unit first (`sudo systemctl edit --full 2048-train`, or the file
before installing it) to set the run name, a `TARGET_GAMES`, an `AI2048_HOME`,
or the options after `long-run.sh`. Leave `KillMode=mixed` alone: workers
deliberately keep SIGTERM's default action, so signalling the whole group would
kill them mid-game and report the stop as a failure.

## When something goes wrong

| What happens | What the run does |
|---|---|
| Your SSH session drops, or the terminal closes (SIGHUP) | Finishes the games in flight, writes a checkpoint, exits. Under `nohup`, tmux or systemd the hangup never reaches it and it carries on. |
| `kill`, `systemctl stop`, a VM shutdown, a Spot preemption (SIGTERM) | The same safe stop. Plan for it to take a few seconds — well inside the roughly 30 seconds a Spot VM gets. |
| Ctrl-C | The same safe stop. A second Ctrl-C quits at once. |
| A worker is killed (out of memory) or training fails | The run stops with the reason on the last line and a non-zero exit status. systemd restarts it after 30 seconds. |
| Power loss, host failure | The weights are a memory-mapped file the kernel writes back as it goes, so most of the learning is on disk; the game counter and statistics come from the last checkpoint (every 2,000 games by default) and the games since are played again. |
| The disk is full | Refused when training starts, with a message. It cannot happen mid-run for the weights, whose space is already reserved. |

Because the safe stop is the same in every case, **there is no wrong way to
stop it** — and `./deploy/long-run.sh` is always the way to start it again.

## Settings worth choosing for a long run

```bash
./deploy/long-run.sh --eval-every 100000 --snapshot-every 500000
```

- **`--eval-every`** (default 20,000 games). A periodic evaluation plays its
  games on one core while every worker waits, so on a many-CPU machine it is
  the largest serial cost in the run. Make it rare — the progress lines already
  show how training is going — and use `--eval-games` to shorten it.
- **`--snapshot-every`** freezes a copy of the weights (268 MB each for the
  default network). A handful, spread across the run, is a cheap safety net and
  lets you compare how the agent played at different points.
- **`--checkpoint-every`** (default 2,000 games) is how much a crash can cost.
  Raise it only if you are short of write bandwidth.
- **`TARGET_GAMES`** / `--target-games N` is a total for the whole run, however
  many sessions it takes; `--games N` counts only the current one.

## Watching it

```bash
python3 train.py --list-runs          # games, best score and tile per run
tail -n 3 logs/main.log               # or journalctl -u 2048-train -n 3
```

The control center can watch a run that was started from the command line. It
reads the same files, so give it the same `AI2048_HOME`, and reach it over an
SSH tunnel because it listens on the machine's loopback address only:

```bash
# on your laptop
gcloud compute ssh VM_NAME --zone ZONE -- -L 8000:localhost:8000
# on the VM, in a second session
NO_BROWSER=1 ./start.sh
# then open http://localhost:8000 on your laptop
```

## Getting the result off the VM

The trained agent is `checkpoints/<run>/weights.f32` plus `meta.json` and
`config.json` beside it; snapshots are under `snapshots/`.

```bash
gcloud compute scp --zone ZONE \
    VM_NAME:2048-ai/checkpoints/main/weights.f32 ./weights.f32
# or into a bucket the VM may write to:
gcloud storage cp checkpoints/main/weights.f32 gs://YOUR_BUCKET/
```

Stop the run (or copy a snapshot) first if you need a copy that is exactly
consistent: the live file changes while workers train.

## Spot VMs

A Spot VM is cheaper and can be preempted at any time. That is fine here: the
preemption arrives as a shutdown, which the run treats as the safe stop above,
and with the systemd unit enabled the run resumes by itself when the VM is
started again. A preempted Spot VM is stopped rather than restarted for you
(unless it was created with the DELETE termination action), so start it again:

```bash
gcloud compute instances start VM_NAME --zone ZONE
```

Keep `AI2048_HOME` on the boot disk or another persistent disk for this to
work.
