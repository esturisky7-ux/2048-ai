"""Fixed-procedure evaluation, kept deliberately separate from training.

Training statistics are a moving target: the agent changes while the numbers
are being collected, exploration and online updates add noise, and the rolling
average lags. Evaluation instead freezes the policy and replays **the same set
of seeded games** every time, so two evaluations differ only because the agent
differs. That is what makes "did this experiment help?" answerable.

Every reported figure comes with an interval:

* the mean gets a normal-approximation confidence interval (n is large and
  scores are an average, so the CLT applies even though scores are skewed);
* tile achievement rates are proportions, so they get Wilson score intervals,
  which stay sensible near 0% and 100% where the normal approximation does not.

"Freezes the policy" needs care for a learned agent playing a run's *current*
weights: those are a memory map shared with any trainer of the run, so a
read-only view of them still changes as the trainer writes. Callers make the
freeze real with :func:`training.runlock.hold_frozen`, which holds the run's
shared lock for the whole evaluation.
"""

from __future__ import annotations

import math
import time
from random import Random

from engine import board as B

MILESTONES = (128, 256, 512, 1024, 2048, 4096, 8192)


def _percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] * (1 - (pos - lo)) + sorted_vals[hi] * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion. Correct near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def game_seed(base_seed: int, index: int) -> int:
    """Deterministic per-game seed. The same (base, index) always replays."""
    return (base_seed * 6364136223846793005 + index * 1442695040888963407) \
        & 0xFFFFFFFFFFFFFFFF


class InvalidActionError(ValueError):
    """An evaluation agent returned an action that is not legal."""


def play_one(agent, seed: int, move_limit: int = 200000, *,
             strict: bool = True, diagnostics: dict | None = None):
    """Play one seeded game; reject illegal moves unless diagnostic mode is explicit."""
    if move_limit < 1:
        raise ValueError("move_limit must be positive")
    invalid = 0
    rng = Random(seed)
    b = B.new_game(rng)
    score = moves = 0
    if hasattr(agent, "new_game"):
        agent.new_game()
    while moves < move_limit:
        legal = B.legal_actions(b)
        if not legal:
            break
        a = agent.act(b)
        if type(a) is not int or a not in legal:
            invalid += 1
            if strict:
                raise InvalidActionError(
                    f"illegal action {a!r} at move {moves}, seed {seed}")
            a = legal[0]
        nb, gained, moved = B.move(b, a)
        b = B.random_spawn(nb, rng)
        score += gained
        moves += 1
    if diagnostics is not None:
        diagnostics.update(invalid_actions=invalid,
                           truncated=not B.is_game_over(b))
    return score, moves, B.max_tile(b)


def evaluate(agent, games: int = 1000, seed: int = 987654,
             progress=None, move_limit: int = 200000,
             stop_flag=None, *, strict: bool = True) -> dict:
    """Run ``games`` seeded games and summarise.

    ``progress`` is an optional callback ``(done, total)``.
    ``stop_flag`` is an optional zero-arg callable; evaluation stops early
    (reporting what it has) when it returns True.
    """
    if games < 1 or move_limit < 1:
        raise ValueError("games and move_limit must be positive")
    scores, moves_list, tiles, per_game = [], [], [], []
    t0 = time.perf_counter()
    for i in range(games):
        if stop_flag is not None and stop_flag():
            break
        details = {}
        seed_i = game_seed(seed, i)
        s, m, t = play_one(agent, seed_i, move_limit, strict=strict,
                           diagnostics=details)
        # A 64-bit seed must survive browser JSON parsing without rounding.
        per_game.append({"index": i, "seed": str(seed_i), "score": s,
                         "moves": m, "max_tile": t, **details})
        scores.append(s)
        moves_list.append(m)
        tiles.append(t)
        if progress and (i + 1) % max(1, games // 20) == 0:
            progress(i + 1, games)
    elapsed = time.perf_counter() - t0
    result = summarise(scores, moves_list, tiles, elapsed, seed,
                       getattr(agent, "describe", lambda: {"name": "?"})())
    invalid = sum(g["invalid_actions"] for g in per_game)
    truncated = sum(g["truncated"] for g in per_game)
    result.update(per_game=per_game, requested_games=games, move_limit=move_limit,
                  integrity={"strict": strict, "invalid_actions": invalid,
                             "truncated_games": truncated,
                             "complete": len(per_game) == games,
                             "valid": not invalid and not truncated
                                      and len(per_game) == games})
    return result


def summarise(scores, moves_list, tiles, elapsed: float, seed: int,
              agent_desc: dict) -> dict:
    n = len(scores)
    if n == 0:
        return {"games": 0, "agent": agent_desc, "seed": seed}
    s = sorted(scores)
    mean = sum(scores) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    else:
        var = 0.0
    std = math.sqrt(var)
    stderr = std / math.sqrt(n)
    ci = (mean - 1.96 * stderr, mean + 1.96 * stderr)

    rates = {}
    for m in MILESTONES:
        k = sum(1 for t in tiles if t >= m)
        lo, hi = wilson_interval(k, n)
        rates[str(m)] = {"rate": k / n, "count": k, "ci95": [lo, hi]}

    tile_hist: dict[str, int] = {}
    for t in tiles:
        tile_hist[str(t)] = tile_hist.get(str(t), 0) + 1

    return {
        "agent": agent_desc,
        "games": n,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "games_per_second": n / elapsed if elapsed > 0 else 0.0,
        "mean_score": mean,
        "median_score": _percentile(s, 0.5),
        "std_score": std,
        "stderr_score": stderr,
        "ci95_mean": [ci[0], ci[1]],
        "min_score": s[0],
        "max_score": s[-1],
        "p25_score": _percentile(s, 0.25),
        "p75_score": _percentile(s, 0.75),
        "p95_score": _percentile(s, 0.95),
        "mean_moves": sum(moves_list) / n,
        "median_moves": _percentile(sorted(moves_list), 0.5),
        "max_moves": max(moves_list),
        "highest_tile": max(tiles),
        "median_tile": sorted(tiles)[n // 2],
        "tile_rates": rates,
        "tile_histogram": tile_hist,
        "timestamp": time.time(),
    }


def format_report(res: dict) -> str:
    """Human-readable summary for the terminal."""
    if not res.get("games"):
        return "no games played"
    a = res.get("agent", {})
    lines = [
        f"agent            {a.get('name','?')}"
        + (f"  (depth {a['depth']})" if a.get("depth") else "")
        + (f"  [{a['games_trained']:,} games trained]"
           if a.get("games_trained") else ""),
        f"games            {res['games']:,}   seed {res['seed']}"
        f"   {res['elapsed_seconds']:.1f}s"
        f"   ({res['games_per_second']:.2f} games/s)",
        "",
        f"mean score       {res['mean_score']:10,.1f}"
        f"   95% CI [{res['ci95_mean'][0]:,.0f}, {res['ci95_mean'][1]:,.0f}]",
        f"median score     {res['median_score']:10,.1f}",
        f"std dev          {res['std_score']:10,.1f}",
        f"min / max        {res['min_score']:10,} / {res['max_score']:,}",
        f"p25 / p75 / p95  {res['p25_score']:,.0f} / {res['p75_score']:,.0f}"
        f" / {res['p95_score']:,.0f}",
        f"mean moves       {res['mean_moves']:10,.1f}"
        f"   (longest {res['max_moves']:,})",
        f"highest tile     {res['highest_tile']:10,}"
        f"   median {res['median_tile']:,}",
        "",
        "tile             rate      95% CI",
    ]
    for m in MILESTONES:
        r = res["tile_rates"][str(m)]
        if r["count"] == 0 and m > 512:
            continue
        lines.append(
            f"  {m:<6,}       {r['rate']*100:6.2f}%   "
            f"[{r['ci95'][0]*100:5.2f}%, {r['ci95'][1]*100:5.2f}%]")
    integrity = res.get("integrity", {})
    if integrity:
        lines.extend(["", "evaluation integrity  " +
                      ("PASS" if integrity["valid"] else "WARNING"),
                      f"invalid actions       {integrity['invalid_actions']}",
                      f"truncated games       {integrity['truncated_games']}"])
        if not integrity["complete"]:
            lines.append("incomplete evaluation: stopped before requested game count")
    return "\n".join(lines)
