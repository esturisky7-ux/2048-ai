/* 2048 AI Laboratory dashboard.
   No framework, no build step: fetch + canvas. Charts are drawn by hand
   because four line charts do not justify shipping a charting library. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "—"
    : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

function duration(sec) {
  if (!sec || sec < 0) return "—";
  sec = Math.floor(sec);
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}
function ago(ts) {
  if (!ts) return "—";
  return duration(Date.now() / 1000 - ts) + " ago";
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* Axis labels have to stay readable for both game counts (0 .. 1,000,000)
   and elapsed hours (0 .. 0.05), so pick precision from the magnitude. */
function formatAxis(v, unit) {
  const suffix = unit || "";
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toFixed(a >= 100000 ? 0 : 1) + "k" + suffix;
  if (a >= 10 || v === 0) return v.toFixed(0) + suffix;
  if (a >= 1) return v.toFixed(1) + suffix;
  return v.toFixed(2) + suffix;
}

/* ------------------------------------------------------------------ state */
const state = {
  run: "default",
  frames: [],          // frames received from the server
  cursor: 0,           // index of the frame currently displayed
  fetched: 0,          // how many frames we have pulled
  playing: false,
  done: false,
  speed: 1,
  timer: null,
  poller: null,
  lastStatus: null,
};

/* ------------------------------------------------------------------ charts */
function drawChart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 48, r: 10, t: 10, b: 22 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const lines = series.filter((s) => s.y && s.y.length);
  if (!lines.length) {
    ctx.fillStyle = css("--fg-faint"); ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("no data yet", w / 2, h / 2);
    return;
  }

  let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity;
  for (const s of lines) {
    for (let i = 0; i < s.y.length; i++) {
      const x = s.x[i];
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (s.y[i] > ymax) ymax = s.y[i];
      if (opts.allowNegative && s.y[i] < ymin) ymin = s.y[i];
    }
  }
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax <= ymin) ymax = ymin + 1;
  if (opts.percent) ymax = Math.min(100, Math.max(ymax * 1.15, 5));
  else ymax *= 1.1;

  const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * plotW;
  const Y = (v) => pad.t + plotH - ((v - ymin) / (ymax - ymin)) * plotH;

  // grid + y labels
  ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  ctx.fillStyle = css("--fg-faint");
  ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "right";
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = ymin + ((ymax - ymin) * i) / ticks;
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    let lab;
    if (opts.tileScale) lab = v >= 1 ? fmt(Math.pow(2, Math.round(v))) : "0";
    else if (opts.percent) lab = v.toFixed(0) + "%";
    else lab = v >= 10000 ? (v / 1000).toFixed(0) + "k" : fmt(v);
    ctx.fillText(lab, pad.l - 6, y + 3);
  }
  // x labels
  ctx.textAlign = "center";
  for (let i = 0; i <= 2; i++) {
    const v = xmin + ((xmax - xmin) * i) / 2;
    ctx.fillText(formatAxis(v, opts.xUnit), X(v), h - 7);
  }

  for (const s of lines) {
    if (s.band) {   // confidence band
      ctx.fillStyle = s.color + "22";
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) ctx[i ? "lineTo" : "moveTo"](X(s.x[i]), Y(s.band[0][i]));
      for (let i = s.x.length - 1; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.band[1][i]));
      ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.8;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.y.length; i++) {
      const px = X(s.x[i]), py = Y(s.y[i]);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (s.points) {
      ctx.fillStyle = s.color;
      for (let i = 0; i < s.y.length; i++) {
        ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), 2.4, 0, 7); ctx.fill();
      }
    }
  }
  ctx.setLineDash([]);
  // legend
  if (opts.legend) {
    ctx.font = "10px system-ui"; ctx.textAlign = "left";
    let lx = pad.l + 4;
    for (const s of lines) {
      if (!s.label) continue;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, pad.t + 2, 8, 2.5);
      ctx.fillStyle = css("--fg-dim");
      ctx.fillText(s.label, lx + 12, pad.t + 6);
      lx += 14 + ctx.measureText(s.label).width + 12;
    }
  }
}

/* ------------------------------------------------------------------ status */
async function refreshStatus() {
  let d;
  try {
    d = await (await fetch(`/api/status?run=${encodeURIComponent(state.run)}`)).json();
  } catch (e) { return; }
  state.lastStatus = d;

  const pill = $("train-pill");
  pill.className = "pill " + (d.running ? "pill-run" : "pill-idle");
  pill.querySelector("span").textContent = d.running ? "Training" : "Stopped";

  $("s-games").textContent = fmt(d.games);
  $("s-gps").textContent = d.running ? fmt(d.games_per_second, 1) : "—";
  $("s-mps").textContent = d.running && d.moves_per_second
    ? fmt(d.moves_per_second, 0) + " moves/s" : "";
  $("s-session").textContent = d.running ? duration(d.session_seconds) : "—";
  $("s-session-games").textContent = d.running ? fmt(d.session_games) + " games" : "";
  $("s-traintime").textContent = duration(d.total_train_seconds);
  $("s-ckpt").textContent = d.checkpoint_saved_at ? ago(d.checkpoint_saved_at) : "none";
  $("s-ckpt-path").textContent = d.exists
    ? `${d.tuple_set || "?"} · ${(d.disk_bytes / 1e6).toFixed(0)} MB` : "no checkpoint";

  const ev = d.last_eval;
  $("s-eval").textContent = ev ? fmt(ev.mean_score, 0) : "—";
  $("s-eval-note").textContent = ev
    ? `mean @ ${fmt(ev.games_trained)} games (${ev.eval_games} eval games)` : "not evaluated yet";

  const at = d.all_time || {};
  $("r-score").textContent = fmt(at.best_score || 0);
  $("r-score-at").textContent = at.best_score_game ? `at game ${fmt(at.best_score_game)}` : "";
  $("r-tile").textContent = fmt(at.best_tile || 0);
  $("r-tile-at").textContent = at.best_tile_game ? `at game ${fmt(at.best_tile_game)}` : "";
  $("r-long").textContent = fmt(at.longest_game || 0);
  $("r-games").textContent = fmt(at.games || 0);
  $("r-moves").textContent = at.moves ? fmt(at.moves) + " moves total" : "";
  $("rec-note").textContent = at.games
    ? `all-time mean ${fmt(at.mean_score, 0)}` : "";

  const rates = (at.tile_rates) || {};
  $("rate-bars").innerHTML = ["512", "1024", "2048", "4096", "8192"].map((k) => {
    const pct = (rates[k] || 0) * 100;
    const n = (at.tile_counts || {})[k] || 0;
    return `<div class="rate"><div class="rl"><span>${fmt(k)}</span>
      <b>${pct.toFixed(pct < 1 && pct > 0 ? 2 : 1)}%</b></div>
      <div class="bar"><i style="width:${Math.min(100, pct)}%"></i></div>
      <div class="rl" style="margin:5px 0 0"><span style="font-size:10px">${fmt(n)} games</span></div></div>`;
  }).join("");

  const sel = $("run-select");
  const names = (d.runs || []).map((r) => r.name);
  if (!names.includes(state.run)) names.unshift(state.run);
  if (sel.dataset.names !== names.join(",")) {
    sel.dataset.names = names.join(",");
    sel.innerHTML = names.map((n) => `<option ${n === state.run ? "selected" : ""}>${n}</option>`).join("");
  }
  $("status-note").textContent = d.running
    ? `pid ${d.pid || ""} · alpha ${d.alpha ?? "—"}`
    : (d.exists ? "no training process running" : "no checkpoint for this run yet");
  $("foot-info").textContent =
    `run "${d.run}" · ${fmt(d.games)} games · ${d.tuple_set || "—"} network`;
}

/* ------------------------------------------------------------------ history */
async function refreshHistory() {
  let d;
  try {
    d = await (await fetch(`/api/history?run=${encodeURIComponent(state.run)}`)).json();
  } catch (e) { return; }
  const h = d.history, ev = d.evaluations;
  const has = h.games && h.games.length;
  $("chart-empty").classList.toggle("hidden", !!has);

  const A = css("--accent"), B = css("--accent-2"), OK = css("--ok"), W = css("--warn");

  drawChart($("c-score"), [
    { x: h.games, y: h.mean_score, color: A, label: "mean (rolling)" },
    { x: h.games, y: h.median_score, color: B, label: "median (rolling)", width: 1.2 },
    ev.games_trained.length
      ? { x: ev.games_trained, y: ev.mean_score, color: OK, label: "evaluation",
          points: true, dash: [4, 3],
          band: [ev.ci_low, ev.ci_high] }
      : {},
  ], { legend: true });

  drawChart($("c-tile"), [
    { x: h.games, y: (h.max_tile || []).map((v) => (v > 0 ? Math.log2(v) : 0)),
      color: A, label: "best tile in window" },
  ], { tileScale: true, legend: true });

  drawChart($("c-rates"), [
    { x: h.games, y: h.rate_512, color: css("--fg-faint"), label: "512", width: 1.2 },
    { x: h.games, y: h.rate_1024, color: B, label: "1024", width: 1.4 },
    { x: h.games, y: h.rate_2048, color: A, label: "2048", width: 2.1 },
    { x: h.games, y: h.rate_4096, color: W, label: "4096", width: 1.6 },
  ], { percent: true, legend: true });

  const t0 = h.wall_time && h.wall_time.length ? h.wall_time[0] : 0;
  drawChart($("c-games"), [
    { x: (h.wall_time || []).map((t) => (t - t0) / 3600), y: h.games, color: B,
      label: "cumulative games vs elapsed hours" },
  ], { legend: true, xUnit: "h" });
}

/* ------------------------------------------------------------------ board */
function renderBoard(frame) {
  const el = $("board");
  if (!el.children.length) {
    el.innerHTML = Array.from({ length: 16 }, () => '<div class="cell v0"></div>').join("");
  }
  const cells = el.children;
  const flat = frame ? frame.board.flat() : new Array(16).fill(0);
  for (let i = 0; i < 16; i++) {
    const v = flat[i];
    const c = cells[i];
    const cls = "cell " + (v === 0 ? "v0" : v <= 2048 ? "v" + v : "big");
    if (c.className !== cls) c.className = cls;
    const txt = v ? String(v) : "";
    if (c.textContent !== txt) c.textContent = txt;
  }
  $("g-score").textContent = fmt(frame ? frame.score : 0);
  $("g-moves").textContent = fmt(frame ? frame.moves : 0);
  $("g-tile").textContent = fmt(frame ? frame.max_tile : 0);
  $("g-action").textContent = frame && frame.action ? frame.action : "—";
}

/* ------------------------------------------------------------------ live game */
function speedDelay() {
  // 1x is a comfortable ~8 moves/second.
  return state.speed === 0 ? 0 : 125 / state.speed;
}

function tick() {
  if (!state.playing) return;
  if (state.cursor < state.frames.length - 1) {
    state.cursor++;
    renderBoard(state.frames[state.cursor]);
  } else if (state.done) {
    finishPlayback();
    return;
  }
  const d = speedDelay();
  if (d === 0) {
    // Maximum: jump to the newest frame we have, then wait for more.
    state.cursor = state.frames.length - 1;
    renderBoard(state.frames[state.cursor]);
    state.timer = setTimeout(tick, 16);
  } else {
    state.timer = setTimeout(tick, d);
  }
}

function finishPlayback() {
  state.playing = false;
  clearTimeout(state.timer);
  clearInterval(state.poller);
  $("watch-btn").disabled = false;
  $("stop-btn").disabled = true;
  const last = state.frames[state.frames.length - 1];
  $("live-status").textContent = last
    ? `game over — score ${fmt(last.score)}, ${fmt(last.moves)} moves, best tile ${fmt(last.max_tile)}`
    : "stopped";
}

async function pollFrames() {
  if (!state.playing) return;
  try {
    const d = await (await fetch(`/api/watch/state?since=${state.fetched}`)).json();
    if (d.error) { $("live-status").textContent = "error: " + d.error; finishPlayback(); return; }
    if (d.frames && d.frames.length) {
      state.frames.push(...d.frames);
      state.fetched += d.frames.length;
    }
    state.done = !!d.done;
    const behind = state.frames.length - 1 - state.cursor;
    $("live-status").textContent =
      `${d.agent || ""} · seed ${d.seed ?? "—"} · frame ${fmt(state.cursor)}/${fmt(d.total)}` +
      (state.done ? " · finished" : ` · buffer ${behind}`);
  } catch (e) { /* transient; next poll will retry */ }
}

async function startWatch(seed = null) {
  const agent = $("agent-select").value;
  const depth = parseInt($("depth-select").value, 10);
  clearTimeout(state.timer); clearInterval(state.poller);
  Object.assign(state, { frames: [], cursor: 0, fetched: 0, done: false, playing: false });
  renderBoard(null);
  $("live-status").textContent = "starting…";
  $("watch-btn").disabled = true;

  let res;
  try {
    res = await (await fetch("/api/watch/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        seed === null ? { agent, depth, run: state.run }
                      : { agent, depth, run: state.run, seed }),
    })).json();
  } catch (e) {
    $("live-status").textContent = "could not reach the server";
    $("watch-btn").disabled = false; return;
  }
  if (res.error) {
    $("live-status").textContent = res.error;
    $("watch-btn").disabled = false; return;
  }
  state.playing = true;
  $("stop-btn").disabled = false;
  await pollFrames();
  state.poller = setInterval(pollFrames, 400);
  tick();
}

async function stopWatch() {
  try { await fetch("/api/watch/stop", { method: "POST" }); } catch (e) { }
  finishPlayback();
}

/* ------------------------------------------------------------------ wiring */
function updateDepthVisibility(setDefault = false) {
  const a = $("agent-select").value;
  const show = a === "expectimax" || a === "learned";
  $("depth-field").style.visibility = show ? "visible" : "hidden";
  // Depth 1 IS the learned policy; expectimax needs 2 to be worth watching.
  if (setDefault) $("depth-select").value = a === "learned" ? "1" : "2";
  const hints = {
    learned: "Depth 1 is the trained policy itself. Higher depths search on top of it: stronger, much slower.",
    expectimax: "Classic search with a hand-written evaluator. Depth 3 is strong but plays only a few moves per second on this CPU.",
    heuristic: "One-ply greedy using the hand-written evaluator. No lookahead.",
    random: "Uniformly random legal moves — the performance floor.",
  };
  $("live-hint").textContent = hints[a] || "";
}

async function pickInitialRun() {
  // ?run=NAME wins; otherwise use "default" if it has a checkpoint, else the
  // most recently saved run, so a fresh visit lands on something with data.
  const fromUrl = new URLSearchParams(location.search).get("run");
  if (fromUrl) { state.run = fromUrl; return; }
  try {
    const d = await (await fetch("/api/runs")).json();
    const runs = d.runs || [];
    if (!runs.length) return;
    state.run = runs.some((r) => r.name === "default") ? "default" : runs[0].name;
  } catch (e) { /* keep the default */ }
}

async function init() {
  renderBoard(null);
  $("agent-select").addEventListener("change", () => updateDepthVisibility(true));
  $("speed-select").addEventListener("change", (e) => {
    state.speed = parseFloat(e.target.value);
    if (state.playing) { clearTimeout(state.timer); tick(); }
  });
  $("watch-btn").addEventListener("click", () => startWatch());
  $("stop-btn").addEventListener("click", stopWatch);
  $("run-select").addEventListener("change", (e) => {
    state.run = e.target.value;
    const u = new URL(location.href);
    u.searchParams.set("run", state.run);
    history.replaceState(null, "", u);
    refreshStatus(); refreshHistory();
  });
  $("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) { }
    refreshHistory();
  });
  try {
    const t = localStorage.getItem("theme");
    if (t) document.documentElement.setAttribute("data-theme", t);
  } catch (e) { }

  updateDepthVisibility(true);
  await pickInitialRun();
  refreshStatus(); refreshHistory();

  // ?watch=learned[&depth=2][&speed=5][&seed=1] starts a game straight away,
  // so a particular view can be bookmarked or opened on a second screen.
  // With a seed the game is replayed move for move, which is how the
  // screenshot in the README was taken.
  const qp = new URLSearchParams(location.search);
  const w = qp.get("watch");
  if (w && [...$("agent-select").options].some((o) => o.value === w)) {
    $("agent-select").value = w;
    if (qp.get("depth")) $("depth-select").value = qp.get("depth");
    if (qp.get("speed")) {
      $("speed-select").value = qp.get("speed");
      state.speed = parseFloat(qp.get("speed"));
    }
    updateDepthVisibility();
    const seed = qp.get("seed");
    startWatch(seed === null ? null : parseInt(seed, 10));
  }
  setInterval(refreshStatus, 2000);
  setInterval(refreshHistory, 15000);
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(refreshHistory, 200); });
}
document.addEventListener("DOMContentLoaded", init);
