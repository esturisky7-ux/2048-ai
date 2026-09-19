/* Play: watch an agent, play yourself, or put the two side by side.
 *
 * Every rule is applied by the Python engine. The browser sends "left" and
 * draws whatever comes back — there is deliberately no second implementation
 * of 2048 in JavaScript that could disagree with the first.
 */

const SPEEDS = [
  { value: 0.25, label: "0.25×" }, { value: 0.5, label: "0.5×" },
  { value: 1, label: "1×" }, { value: 2, label: "2×" },
  { value: 5, label: "5×" }, { value: 10, label: "10×" },
  { value: 0, label: "Maximum" },
];

App.views.play = {
  title: "Play",
  subtitle: "Watch the AI, play yourself, or compare the two",

  mount(root, { args }) {
    this.root = root;
    this.tab = args[0] || "watch";
    this.cleanups = [];
    const tabs = el("div", { class: "btn-row", style: "margin-bottom:18px" });
    for (const [id, label] of [["watch", "Watch the AI"], ["human", "Play yourself"],
                               ["versus", "You vs AI"]]) {
      const b = el("button", { class: `btn ${this.tab === id ? "btn-primary" : ""}` }, label);
      b.onclick = () => App.go(`play/${id}`);
      tabs.append(b);
    }
    root.append(tabs);
    this.body = el("div");
    root.append(this.body);
    if (this.tab === "human") this.mountHuman();
    else if (this.tab === "versus") this.mountVersus();
    else this.mountWatch();
  },

  unmount() {
    for (const fn of this.cleanups || []) { try { fn(); } catch (_) { } }
    this.cleanups = [];
  },

  /* ================================================================ WATCH */
  mountWatch() {
    const state = this.watch = {
      session: null, frames: [], cursor: 0, fetched: 0, playing: false,
      speed: App.settings.playback_speed || 1, done: false,
      timer: null, poller: null,
    };

    const agentSel = selectInput("w-agent", [
      { value: "learned", label: "Learned AI (your trained agent)" },
      { value: "expectimax", label: "Expectimax (classic search)" },
      { value: "heuristic", label: "Heuristic (hand-written rules)" },
      { value: "random", label: "Random (the floor)" },
    ], "learned");
    const depthSel = selectInput("w-depth", [
      { value: 1, label: "1 — the trained policy itself" },
      { value: 2, label: "2 — search on top (slower)" },
      { value: 3, label: "3 — deep search (much slower)" },
    ], 1);
    const speedSel = selectInput("w-speed", SPEEDS, state.speed);
    const seedIn = el("input", { type: "number", id: "w-seed", placeholder: "random",
      min: 0, step: 1 });
    const ckptSel = selectInput("w-ckpt", [{ value: "", label: "Latest checkpoint" }], "");

    const depthField = field("Search depth", depthSel, "",
      "Depth 1 is the learned policy. Higher depths add an expectimax " +
      "search on top: stronger, and much slower.");
    const ckptField = field("Checkpoint", ckptSel, "", "Which saved weights to play with.");

    const syncAgent = () => {
      const a = agentSel.value;
      depthField.style.display = (a === "learned" || a === "expectimax") ? "" : "none";
      ckptField.style.display = a === "learned" ? "" : "none";
      if (a === "expectimax") {
        depthSel.innerHTML = "";
        for (const d of [1, 2, 3, 4]) {
          depthSel.append(el("option", { value: d, selected: d === 2 },
            `${d}${d === 2 ? " — default" : d >= 3 ? " — slow" : ""}`));
        }
      } else if (a === "learned" && depthSel.options.length === 4) {
        depthSel.innerHTML = "";
        for (const [v, l] of [[1, "1 — the trained policy itself"],
                              [2, "2 — search on top (slower)"],
                              [3, "3 — deep search (much slower)"]]) {
          depthSel.append(el("option", { value: v, selected: v === 1 }, l));
        }
      }
    };
    agentSel.onchange = syncAgent;

    const startBtn = el("button", { class: "btn btn-primary" }, "Start game");
    const currentBtn = el("button", { class: "btn" }, "Watch current AI");
    const pauseBtn = el("button", { class: "btn" }, "Pause");
    const stepBtn = el("button", { class: "btn" }, "Step");
    const restartBtn = el("button", { class: "btn" }, "Restart");
    const stopBtn = el("button", { class: "btn btn-danger" }, "Stop");
    pauseBtn.disabled = stepBtn.disabled = restartBtn.disabled = stopBtn.disabled = true;

    const boardNode = el("div");
    const board = new Board(boardNode);
    const stats = el("div", { class: "board-stats" });
    const statusLine = el("div", { class: "faint mono", style: "font-size:12px" }, "idle");
    const hint = el("div", { class: "hint" });

    const paintStats = (snap) => {
      const f = state.frames[state.cursor];
      stats.innerHTML = "";
      stats.append(
        el("div", {}, el("label", {}, "Score"), el("b", {}, F.n(f ? f.score : 0))),
        el("div", {}, el("label", {}, "Moves"), el("b", {}, F.n(f ? f.moves : 0))),
        el("div", {}, el("label", {}, "Max tile"), el("b", {}, F.n(f ? f.max_tile : 0))),
        el("div", {}, el("label", {}, "Decision"),
          el("b", {}, f && f.decision_ms ? `${f.decision_ms.toFixed(1)}ms` : "—")));
      if (snap) {
        statusLine.textContent =
          `${snap.agent} · seed ${snap.seed} · frame ${F.n(state.cursor)}/${F.n(snap.total)}` +
          (snap.done ? " · finished" : ` · buffer ${Math.max(0, snap.total - 1 - state.cursor)}`) +
          (snap.mean_decision_ms ? ` · avg ${snap.mean_decision_ms.toFixed(1)}ms/move` : "");
      }
    };

    const delay = () => {
      const s = Number(speedSel.value);
      return s === 0 ? 16 : 260 / s;
    };

    const tick = () => {
      if (!state.playing) return;
      if (state.cursor < state.frames.length - 1) {
        state.cursor++;
        board.render(state.frames[state.cursor].board);
        paintStats(state.lastSnap);
      } else if (state.done) {
        finish();
        return;
      }
      state.timer = setTimeout(tick, delay());
    };

    const finish = () => {
      state.playing = false;
      clearTimeout(state.timer);
      clearInterval(state.poller);
      pauseBtn.disabled = stepBtn.disabled = true;
      startBtn.disabled = currentBtn.disabled = false;
      const last = state.frames[state.frames.length - 1];
      if (last) {
        statusLine.textContent =
          `game over — score ${F.n(last.score)}, ${F.n(last.moves)} moves, best tile ${F.n(last.max_tile)}`;
        if (last.max_tile >= 2048) {
          Toast.ok("Reached " + F.n(last.max_tile), `final score ${F.n(last.score)}`);
        }
      }
    };

    const poll = async () => {
      if (!state.session) return;
      try {
        const snap = await API.get(`/api/game/${state.session}`, { since: state.fetched });
        state.lastSnap = snap;
        if (snap.error) { Toast.error("Game error", snap.error); finish(); return; }
        if (snap.frames && snap.frames.length) {
          state.frames.push(...snap.frames);
          state.fetched += snap.frames.length;
        }
        state.done = !!snap.done;
        paintStats(snap);
      } catch (_) { /* transient; the next poll retries */ }
    };

    const start = async (opts = {}) => {
      this.stopWatch();
      state.frames = []; state.cursor = 0; state.fetched = 0;
      state.done = false; state.playing = false;
      board.clear();
      startBtn.disabled = currentBtn.disabled = true;
      statusLine.textContent = "starting…";
      const payload = {
        agent: opts.agent || agentSel.value,
        depth: Number(depthSel.value) || 1,
        run: App.run,
      };
      if (payload.agent === "learned" && ckptSel.value) payload.checkpoint = ckptSel.value;
      if (seedIn.value !== "") payload.seed = Number(seedIn.value);
      try {
        const r = await API.post("/api/game/ai/start", payload);
        state.session = r.session.id;
        state.playing = true;
        pauseBtn.disabled = stepBtn.disabled = false;
        restartBtn.disabled = stopBtn.disabled = false;
        pauseBtn.textContent = "Pause";
        await poll();
        state.poller = setInterval(poll, 420);
        tick();
      } catch (e) {
        Toast.error("Could not start the game", e.message);
        statusLine.textContent = e.message;
        startBtn.disabled = currentBtn.disabled = false;
        if (e.status === 404) {
          hint.textContent = "Train an agent first, or pick one that needs no training.";
        }
      }
    };

    startBtn.onclick = () => start();
    currentBtn.onclick = () => {
      agentSel.value = "learned";
      ckptSel.value = "";
      depthSel.value = "1";
      syncAgent();
      start({ agent: "learned" });
    };
    restartBtn.onclick = () => start();
    pauseBtn.onclick = async () => {
      if (!state.session) return;
      if (state.playing) {
        state.playing = false;
        clearTimeout(state.timer);
        pauseBtn.textContent = "Resume";
        await API.post(`/api/game/${state.session}/control`, { action: "pause" }).catch(() => { });
      } else {
        state.playing = true;
        pauseBtn.textContent = "Pause";
        await API.post(`/api/game/${state.session}/control`, { action: "resume" }).catch(() => { });
        tick();
      }
    };
    stepBtn.onclick = async () => {
      if (!state.session) return;
      state.playing = false;
      clearTimeout(state.timer);
      pauseBtn.textContent = "Resume";
      await API.post(`/api/game/${state.session}/control`, { action: "step" }).catch(() => { });
      setTimeout(async () => {
        await poll();
        if (state.cursor < state.frames.length - 1) {
          state.cursor++;
          board.render(state.frames[state.cursor].board);
          paintStats(state.lastSnap);
        }
      }, 140);
    };
    stopBtn.onclick = () => { this.stopWatch(); finish(); statusLine.textContent = "stopped"; };
    speedSel.onchange = () => {
      if (state.playing) { clearTimeout(state.timer); tick(); }
    };

    // Space toggles pause, which is the one shortcut worth having here.
    const onKey = (e) => {
      if (e.code === "Space" && state.session && !pauseBtn.disabled) {
        e.preventDefault();
        pauseBtn.click();
      }
    };
    document.addEventListener("keydown", onKey);
    this.cleanups.push(() => document.removeEventListener("keydown", onKey));
    this.cleanups.push(() => this.stopWatch());

    syncAgent();
    paintStats(null);
    board.render(null);

    // Deep links: ?watch=learned&depth=1&speed=5&seed=1 starts a game on
    // load, so a particular view can be bookmarked or opened on a second
    // screen. Carried over from the original dashboard.
    const qp = new URLSearchParams(location.search);
    const wanted = qp.get("watch");
    if (wanted && [...agentSel.options].some((o) => o.value === wanted)) {
      agentSel.value = wanted;
      if (qp.get("depth")) depthSel.value = qp.get("depth");
      if (qp.get("speed") !== null) speedSel.value = qp.get("speed");
      if (qp.get("seed") !== null) seedIn.value = qp.get("seed");
      syncAgent();
      start();
    }

    this.body.innerHTML = "";
    this.body.append(el("div", { class: "grid split-main" },
      el("section", { class: "panel" },
        el("div", { class: "panel-head" },
          el("h3", {}, "Live game"),
          el("span", { class: "note" },
            "played by the server in a throttled thread — training is unaffected")),
        el("div", { class: "board-wrap" }, boardNode, stats, statusLine)),
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "Agent")),
        el("div", { class: "form-grid", style: "grid-template-columns:1fr" },
          field("Agent", agentSel),
          depthField, ckptField,
          field("Playback speed", speedSel),
          field("Seed", seedIn, "leave blank for a random game",
            "The same seed replays exactly the same game.")),
        el("div", { class: "btn-row", style: "margin-top:16px" }, startBtn, currentBtn),
        el("div", { class: "btn-row", style: "margin-top:8px" },
          pauseBtn, stepBtn, restartBtn, stopBtn),
        hint)));

    this.loadCheckpointOptions(ckptSel);
  },

  stopWatch() {
    const s = this.watch;
    if (!s) return;
    clearTimeout(s.timer); clearInterval(s.poller);
    s.playing = false;
    if (s.session) {
      API.post(`/api/game/${s.session}/control`, { action: "stop" }).catch(() => { });
      s.session = null;
    }
  },

  async loadCheckpointOptions(sel) {
    try {
      const data = await API.get("/api/checkpoints");
      const forRun = (data.checkpoints || []).filter((c) => c.run === App.run);
      for (const c of forRun) {
        if (c.kind === "current") continue;
        sel.append(el("option", { value: c.id },
          `${c.label || "snapshot"} — ${F.compact(c.games)} games`));
      }
    } catch (_) { /* the selector just stays at "latest" */ }
  },

  /* ================================================================ HUMAN */
  mountHuman() {
    const boardNode = el("div");
    const board = new Board(boardNode);
    const stats = el("div", { class: "board-stats" });
    const msg = el("div", { class: "faint", style: "font-size:13px;min-height:20px" });
    let session = null, busy = false, over = false;

    const paint = (s) => {
      board.render(s.board);
      stats.innerHTML = "";
      stats.append(
        el("div", {}, el("label", {}, "Score"), el("b", {}, F.n(s.score))),
        el("div", {}, el("label", {}, "Moves"), el("b", {}, F.n(s.moves))),
        el("div", {}, el("label", {}, "Max tile"), el("b", {}, F.n(s.max_tile))),
        el("div", {}, el("label", {}, "Time"), el("b", {}, F.dur(s.elapsed))));
      over = s.game_over;
      msg.textContent = s.game_over
        ? `Game over — ${F.n(s.score)} points, ${F.n(s.moves)} moves, best tile ${F.n(s.max_tile)}.`
        : "";
      if (s.game_over && !this._announced) {
        this._announced = true;
        Toast.show("Game over", `${F.n(s.score)} points · best tile ${F.n(s.max_tile)}`,
          s.max_tile >= 2048 ? "ok" : "info");
      }
    };

    const move = async (direction) => {
      if (!session || busy || over) return;
      busy = true;
      try { paint(await API.post(`/api/game/${session}/move`, { direction })); }
      catch (e) { Toast.error("Move failed", e.message); }
      finally { busy = false; }
    };

    const newGame = async () => {
      this._announced = false;
      try {
        const r = await API.post("/api/game/human/start", {});
        session = r.session.id;
        over = false;
        paint(r.session);
        msg.textContent = "";
      } catch (e) { Toast.error("Could not start a game", e.message); }
    };

    const onKey = (e) => {
      const keys = {
        ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
        w: "up", s: "down", a: "left", d: "right",
        W: "up", S: "down", A: "left", D: "right",
      };
      const dir = keys[e.key];
      if (!dir) return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) return;
      e.preventDefault();
      move(dir);
    };
    document.addEventListener("keydown", onKey);
    this.cleanups.push(() => document.removeEventListener("keydown", onKey));
    this.cleanups.push(() => {
      if (session) API.post(`/api/game/${session}/control`, { action: "stop" }).catch(() => { });
    });

    // Touch: a swipe of at least 28px in the dominant axis.
    let sx = 0, sy = 0;
    boardNode.addEventListener("pointerdown", (e) => { sx = e.clientX; sy = e.clientY; });
    boardNode.addEventListener("pointerup", (e) => {
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 28) return;
      move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : (dy > 0 ? "down" : "up"));
    });

    const dpad = el("div", { class: "dpad" },
      el("button", { class: "btn up", "aria-label": "Up" }, "↑"),
      el("button", { class: "btn left", "aria-label": "Left" }, "←"),
      el("button", { class: "btn down", "aria-label": "Down" }, "↓"),
      el("button", { class: "btn right", "aria-label": "Right" }, "→"));
    const dirs = ["up", "left", "down", "right"];
    [...dpad.children].forEach((b, i) => { b.onclick = () => move(dirs[i]); });

    const again = el("button", { class: "btn btn-primary" }, "Play again");
    again.onclick = newGame;

    this.body.innerHTML = "";
    this.body.append(el("div", { class: "grid split-main" },
      el("section", { class: "panel" },
        el("div", { class: "panel-head" },
          el("h3", {}, "Your game"),
          el("span", { class: "note" }, "same engine and rules the AI uses")),
        el("div", { class: "board-wrap" }, boardNode, stats, msg)),
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "Controls")),
        el("dl", { class: "kv" },
          el("dt", {}, el("kbd", {}, "↑ ↓ ← →")), el("dd", {}, "Move"),
          el("dt", {}, el("kbd", {}, "W A S D")), el("dd", {}, "Move"),
          el("dt", {}, "Swipe"), el("dd", {}, "Move (touch screens)")),
        el("div", { style: "margin-top:16px" }, dpad),
        el("div", { class: "btn-row", style: "margin-top:16px" }, again),
        el("div", { class: "faint", style: "margin-top:14px;font-size:12px" },
          "Every move is applied by the Python engine, so your game follows " +
          "exactly the rules the AI trains on."))));
    newGame();
  },

  /* =============================================================== VERSUS */
  mountVersus() {
    const st = this.versus = { seed: null, human: null, ai: null, aiSession: null };

    const humanBoardNode = el("div");
    const humanBoard = new Board(humanBoardNode);
    const aiBoardNode = el("div");
    const aiBoard = new Board(aiBoardNode);
    const humanStats = el("div", { class: "board-stats" });
    const aiStats = el("div", { class: "board-stats" });
    const verdict = el("div", { class: "panel", style: "display:none" });
    const note = el("div", { class: "faint", style: "font-size:12.5px" });

    let session = null, busy = false, humanOver = false;

    const paintHuman = (s) => {
      humanBoard.render(s.board);
      humanStats.innerHTML = "";
      humanStats.append(
        el("div", {}, el("label", {}, "Score"), el("b", {}, F.n(s.score))),
        el("div", {}, el("label", {}, "Moves"), el("b", {}, F.n(s.moves))),
        el("div", {}, el("label", {}, "Max tile"), el("b", {}, F.n(s.max_tile))),
        el("div", {}, el("label", {}, "Time"), el("b", {}, F.dur(s.elapsed))));
      if (s.game_over && !humanOver) {
        humanOver = true;
        st.human = { score: s.score, moves: s.moves, max_tile: s.max_tile, elapsed: s.elapsed };
        note.textContent = "Your game is done. Now the AI plays the same seeded game…";
        runAI();
      }
    };

    const move = async (direction) => {
      if (!session || busy || humanOver) return;
      busy = true;
      try { paintHuman(await API.post(`/api/game/${session}/move`, { direction })); }
      catch (e) { Toast.error("Move failed", e.message); }
      finally { busy = false; }
    };

    const runAI = async () => {
      try {
        const r = await API.post("/api/game/ai/start",
          { agent: "learned", run: App.run, depth: 1, seed: st.seed });
        st.aiSession = r.session.id;
      } catch (e) {
        note.textContent = "";
        Toast.error("The AI could not play", e.message);
        showVerdict();
        return;
      }
      let fetched = 0;
      const frames = [];
      const poll = async () => {
        if (!st.aiSession) return;
        let snap;
        try { snap = await API.get(`/api/game/${st.aiSession}`, { since: fetched }); }
        catch (_) { setTimeout(poll, 500); return; }
        if (snap.frames?.length) { frames.push(...snap.frames); fetched += snap.frames.length; }
        const last = frames[frames.length - 1];
        if (last) {
          aiBoard.render(last.board);
          aiStats.innerHTML = "";
          aiStats.append(
            el("div", {}, el("label", {}, "Score"), el("b", {}, F.n(last.score))),
            el("div", {}, el("label", {}, "Moves"), el("b", {}, F.n(last.moves))),
            el("div", {}, el("label", {}, "Max tile"), el("b", {}, F.n(last.max_tile))),
            el("div", {}, el("label", {}, "Time"), el("b", {}, F.dur(snap.elapsed))));
        }
        if (snap.done) {
          st.ai = last ? { score: last.score, moves: last.moves,
                           max_tile: last.max_tile, elapsed: snap.elapsed } : null;
          note.textContent = "";
          showVerdict();
          return;
        }
        setTimeout(poll, 260);
      };
      poll();
    };

    const showVerdict = () => {
      if (!st.human) return;
      const a = st.human, b = st.ai;
      const line = (label, mine, theirs, fmt = F.n) => el("tr", {},
        el("td", {}, label),
        el("td", { class: "num", style: mine > (theirs ?? -1) ? "color:var(--ok);font-weight:650" : "" }, fmt(mine)),
        el("td", { class: "num", style: b && theirs > mine ? "color:var(--ok);font-weight:650" : "" },
          b ? fmt(theirs) : "—"));
      verdict.style.display = "";
      verdict.innerHTML = "";
      verdict.append(
        el("div", { class: "panel-head" }, el("h3", {}, "You vs AI")),
        el("div", { class: "table-wrap" },
          el("table", { class: "data" },
            el("thead", {}, el("tr", {},
              el("th", {}, ""), el("th", { class: "num" }, "You"),
              el("th", { class: "num" }, "Learned AI"))),
            el("tbody", {},
              line("Score", a.score, b?.score),
              line("Highest tile", a.max_tile, b?.max_tile),
              line("Moves", a.moves, b?.moves),
              line("Duration", a.elapsed, b?.elapsed, (v) => F.dur(v))))),
        el("div", { class: "faint", style: "margin-top:12px;font-size:12.5px" },
          "One game each, from the same starting seed. 2048 scores vary " +
          "enormously game to game, so this is for fun — it is not a " +
          "measurement. The ",
          el("a", { href: "#/evaluate" }, "Evaluate"),
          " page compares agents properly, over hundreds of seeded games with " +
          "confidence intervals."));
    };

    const start = async () => {
      st.human = st.ai = null;
      humanOver = false;
      verdict.style.display = "none";
      if (st.aiSession) {
        API.post(`/api/game/${st.aiSession}/control`, { action: "stop" }).catch(() => { });
        st.aiSession = null;
      }
      aiBoard.clear();
      aiStats.innerHTML = "";
      st.seed = Math.floor(Math.random() * 1e9);
      try {
        const r = await API.post("/api/game/human/start", { seed: st.seed });
        session = r.session.id;
        paintHuman(r.session);
        note.textContent = "Play your game. When it ends, the AI plays the same seed.";
      } catch (e) { Toast.error("Could not start", e.message); }
    };

    const onKey = (e) => {
      const keys = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left",
                     ArrowRight: "right", w: "up", s: "down", a: "left", d: "right" };
      const dir = keys[e.key];
      if (!dir) return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag)) return;
      e.preventDefault();
      move(dir);
    };
    document.addEventListener("keydown", onKey);
    this.cleanups.push(() => document.removeEventListener("keydown", onKey));
    this.cleanups.push(() => {
      if (session) API.post(`/api/game/${session}/control`, { action: "stop" }).catch(() => { });
      if (st.aiSession) API.post(`/api/game/${st.aiSession}/control`, { action: "stop" }).catch(() => { });
    });

    const again = el("button", { class: "btn btn-primary" }, "New match");
    again.onclick = start;

    this.body.innerHTML = "";
    this.body.append(
      el("div", { class: "grid cols-2" },
        el("section", { class: "panel" },
          el("div", { class: "panel-head" }, el("h3", {}, "You")),
          el("div", { class: "board-wrap" }, humanBoardNode, humanStats)),
        el("section", { class: "panel" },
          el("div", { class: "panel-head" }, el("h3", {}, "Learned AI")),
          el("div", { class: "board-wrap" }, aiBoardNode, aiStats))),
      el("section", { class: "panel", style: "margin-top:16px" },
        el("div", { class: "btn-row" }, again, note)),
      verdict);
    start();
  },
};
