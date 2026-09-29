# Offline demonstration agent v1

The optional demo edition release ZIP contains `weights.f32.gz`. Source clones
keep only this manifest, evaluation, and documentation. The application never
downloads weights: the user explicitly installs a local copy with **Try a
Trained Agent**. The copy is checked by byte count and SHA-256, installed under
an exclusive run lock, and never overwrites an existing run. Watching the demo
uses an immutable snapshot even if its current weights are later trained.

- Architecture: 8×4 n-tuple network, 524,288 float32 weights (2 MiB unpacked).
- Training: 10,000 games, seed 204801, one worker, default reward and learning
  rate, no search, from the source revision recorded in `manifest.json`.
- Evaluation: 200 games with held-out base seed 9102048, depth 1.
- Mean score: 31,323.2; 95% normal-approximation CI: [29,258, 33,388].
- 2048 achievement: 69.5%; Wilson 95% CI: [62.8%, 75.5%].
- 4096 achievement: 13.5%; highest tile: 4096.
- Integrity: zero illegal actions, zero truncated games.

These results describe this small demonstration, not the larger 4×6 agent in
the main README. Raw per-game results and configuration are included for audit
and replay. Per-game 64-bit seeds are stored as decimal strings to preserve
precision in browser exports; pass `int(seed)` when replaying in Python. Evaluation timing is machine-dependent. The checkpoint was produced
from this project's own self-play; it is distributed under the repository's MIT
license. No external training data was used.

To reproduce, run the training command in `manifest.json`, then:

```sh
python3 evaluate.py --run demo-v1 --agent learned --depth 1 \
  --games 200 --seed 9102048 --no-save --out demo/evaluation.json
```

The demo ZIP includes the full application. Extract it and run
`python3 server.py --open` (Windows: `py server.py --open`).
