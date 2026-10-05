/* Play: watch an agent, play yourself, or put the two side by side.
 *
 * Every rule is applied by the Python engine. The browser sends "left" and
 * draws whatever comes back — there is deliberately no second implementation
 * of 2048 in JavaScript that could disagree with the first.
 */

/* A route segment, decoded; a malformed one decodes to nothing at all. */
function safeDecode(part) {
  try { return decodeURIComponent(part); } catch (_) { return ""; }
}

const SPEEDS = [
  { value: 0.25, label: "0.25×" }, { value: 0.5, label: "0.5×" },
  { value: 1, label: "1×" }, { value: 2, label: "2×" },
  { value: 5, label: "5×" }, { value: 10, label: "10×" },
  { value: 0, label: "Maximum" },
];

const MOVE_KEYS = {
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  w: "up", s: "down", a: "left", d: "right",
  W: "up", S: "down", A: "left", D: "right",
};

/* Arrow keys and WASD, ignored while typing in a form control. */
function moveKeyHandler(move) {
  return (e) => {
    const dir = MOVE_KEYS[e.key];
    if (!dir || e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag)) return;
    if (e.target.closest?.(".tabs-list, .toggle-group, .chart")) return;
    if (document.querySelector(".modal-scrim")) return;
    e.preventDefault();
    move(dir);
  };
}

/* Touch: a swipe of at least 28px in the dominant axis. */
function bindSwipe(node, move) {
  let sx = 0, sy = 0;
  node.addEventListener("pointerdown", (e) => { sx = e.clientX; sy = e.clientY; });
  node.addEventListener("pointerup", (e) => {
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 28) return;
    move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : (dy > 0 ? "down" : "up"));
  });
}

App.views.play = {
  title: "Play",
  subtitle: "Watch the AI or play yourself",

  mount(root, { args }) {
    this.root = root;
    this.tab = ["watch", "human"].includes(args[0]) ? args[0] : "watch";
    this.cleanups = [];
    const tabList = tabs([["watch", "Watch the AI"], ["human", "Play Yourself"]],
                         this.tab, (id, viaKeyboard) => {
      // Arrow keys move between tabs, so keep focus there; after a click it
      // is released, and the arrow keys go back to playing.
      this._focusTab = viaKeyboard;
      App.go(`play/${id}`);
    }, "Play mode");
    root.append(el("div", { class: "inline" }, tabList));
    if (this._focusTab) {
      this._focusTab = false;
      tabList.querySelector('[aria-selected="true"]')?.focus();
    }
    this.body = el("div", { class: "stack stack-16" });
    root.append(this.body);
    if (this.tab === "human") this.mountHuman();
    else this.mountWatch(args[1] ? safeDecode(args[1]) : "");
  },

  unmount() {
    for (const fn of this.cleanups || []) { try { fn(); } catch (_) { } }
    this.cleanups = [];
  },

  /* ================================================================ WATCH */
  /* `checkpoint`: a snapshot id from the route (#/play/watch/<id>), as the
   * Checkpoints page's Watch button sends it. It is selected only if the
   * server lists it for the current run; the server checks it again when the
   * game starts. */
  mountWatch(checkpoint = "") {
    const state = this.watch = {
      session: null, frames: [], cursor: 0, fetched: 0, playing: false,
      speed: App.settings.playback_speed ?? 1, done: false,
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
      depthField.hidden = !(a === "learned" || a === "expectimax");
      ckptField.hidden = a !== "learned";
      if (a === "expectimax") {
        depthSel.replaceChildren(...[1, 2, 3, 4].map((d) => el("option",
          { value: d, selected: d === 2 },
          `${d}${d === 2 ? " — default" : d >= 3 ? " — slow" : ""}`)));
      } else if (a === "learned" && depthSel.options.length === 4) {
        depthSel.replaceChildren(...[[1, "1 — the trained policy itself"],
                                     [2, "2 — search on top (slower)"],
                                     [3, "3 — deep search (much slower)"]].map(([v, l]) =>
          el("option", { value: v, selected: v === 1 }, l)));
      }
    };
    agentSel.onchange = syncAgent;

    const startBtn = button("Start Game", { size: "sm", icon: "play" });
    const currentBtn = button("Watch Current AI", { variant: "outline", size: "sm", icon: "eye" });
    const pauseBtn = button("Pause", { variant: "outline", size: "sm", icon: "pause" });
    const stepBtn = button("Step", { variant: "outline", size: "sm", icon: "step-forward" });
    const restartBtn = button("Restart", { variant: "outline", size: "sm", icon: "rotate-ccw" });
    const stopBtn = button("Stop", { variant: "outline", size: "sm", icon: "square" });
    pauseBtn.disabled = stepBtn.disabled = restartBtn.disabled = stopBtn.disabled = true;
    const setPaused = (paused) => setButton(pauseBtn, paused ? "Resume" : "Pause",
                                            paused ? "play" : "pause");

    const boardNode = el("div");
    const board = new Board(boardNode, { max: 440 });
    const stats = el("div", { style: "width:100%" });
    const statusLine = el("div", { class: "status-line", role: "status" }, "idle");
    const hint = el("div", { class: "note" });
    const explanationBody = el("div", { class: "stack stack-16" });
    let explanationFrame = null;
    const paintExplanation = (frame) => {
      if (frame === explanationFrame) return;
      explanationFrame = frame;
      const info = frame?.explanation;
      explanationBody.replaceChildren();
      if (!info) {
        explanationBody.append(el("p", { class: "note" },
          "Start a learned agent, then pause or step to inspect its decisions."));
        return;
      }
      const previewNode = el("div");
      const preview = new Board(previewNode, { max: 240 });
      const caption = el("div", { class: "note", role: "status" },
        "Board before this move");
      preview.render(info.board_before);
      const rows = info.candidates.map((c) => {
        const chosen = c.action === info.selected;
        const inspect = button(c.action + (chosen ? " · chosen" : ""), {
          variant: chosen ? "default" : "outline", size: "sm",
          onClick: () => {
            preview.render(c.afterstate);
            caption.textContent = `${c.action}: after sliding, before the random tile spawns`;
          },
        });
        return el("tr", {}, el("td", {}, inspect), el("td", { class: "num" }, F.n(c.reward, 1)),
          el("td", { class: "num" }, F.n(c.future_value, 1)), el("td", { class: "num" }, F.n(c.total, 1)));
      });
      explanationBody.append(
        el("p", { class: "note" }, `Move ${frame.moves}: ${info.selected}. ` +
          `Total = reward + ${info.gamma} × estimated future value (depth ${info.depth}). ` +
          "Model estimates, not guaranteed scores. Ties prefer Up, Right, Down, Left."),
        table(["Move / preview", ["Reward", { num: true }],
          ["Future", { num: true }], ["Total", { num: true }]], rows),
        el("div", { class: "stack stack-8", style: "align-items:center" }, previewNode, caption));
    };

    const paintStats = (snap) => {
      const f = state.frames[state.cursor];
      paintExplanation(f);
      stats.replaceChildren(miniStats([
        ["Score", F.n(f ? f.score : 0)],
        ["Moves", F.n(f ? f.moves : 0)],
        ["Max tile", F.n(f ? f.max_tile : 0)],
        ["Decision", f && f.decision_ms ? `${f.decision_ms.toFixed(1)}ms` : "—"],
      ]));
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
      setPaused(false);
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
        // `cursor` is the frame on screen, which is what the server stays a
        // few dozen frames ahead of. Fetching is not watching: at 0.25x the
        // buffer here fills long before the frames are shown.
        const snap = await API.get(`/api/game/${state.session}`,
          { since: state.fetched, cursor: state.cursor });
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
      paintStats(null);
      startBtn.disabled = currentBtn.disabled = true;
      statusLine.textContent = "starting…";
      hint.textContent = "";
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
        setPaused(false);
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
        setPaused(true);
        await API.post(`/api/game/${state.session}/control`, { action: "pause" }).catch(() => { });
      } else {
        state.playing = true;
        setPaused(false);
        await API.post(`/api/game/${state.session}/control`, { action: "resume" }).catch(() => { });
        tick();
      }
    };
    stepBtn.onclick = async () => {
      if (!state.session) return;
      state.playing = false;
      clearTimeout(state.timer);
      setPaused(true);
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
    stopBtn.onclick = () => {
      this.stopWatch();
      finish();
      restartBtn.disabled = stopBtn.disabled = true;
      statusLine.textContent = "stopped";
    };
    speedSel.onchange = () => {
      if (state.playing) { clearTimeout(state.timer); tick(); }
    };

    // Space toggles pause, which is the one shortcut worth having here —
    // even with a button focused, so Space right after "Start Game" pauses
    // rather than starting the game over.
    const onKey = (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (e.code === "Space" && state.session && !pauseBtn.disabled &&
          !["input", "textarea", "select"].includes(tag) &&
          !document.querySelector(".modal-scrim")) {
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

    this.body.replaceChildren(el("div", { class: "row play" },
      el("div", { class: "col-main" }, card(
        cardHeader("Live Game",
          "Played by the server in a throttled thread — training is unaffected."),
        cardContent(el("div", { class: "stack stack-16", style: "align-items:center" },
          boardNode, stats, statusLine)))),
      el("div", { class: "col-side" }, card(
        cardHeader("Agent", "Choose who plays, with which weights and how fast."),
        cardContent(el("div", { class: "stack" },
          demoButton(),
          field("Agent", agentSel),
          depthField, ckptField,
          field("Playback speed", speedSel),
          field("Seed", seedIn, "leave blank for a random game",
            "The same seed replays exactly the same game."),
          el("div", { class: "stack stack-8" },
            el("div", { class: "inline" }, startBtn, currentBtn),
            el("div", { class: "inline" }, pauseBtn, stepBtn, restartBtn, stopBtn)),
          hint))))));

    this.body.append(card(cardHeader("Why This Move?",
      "Inspect the decision that produced the displayed frame. Select a move to preview it."),
      cardContent(explanationBody)));

    // Deep links: ?watch=learned&depth=1&speed=5&seed=1 starts a game on
    // load, so a particular view can be bookmarked or opened on a second
    // screen. Carried over from the original dashboard.
    const qp = new URLSearchParams(location.search);
    const wanted = qp.get("watch");
    if (wanted && [...agentSel.options].some((o) => o.value === wanted)) {
      agentSel.value = wanted;
      syncAgent();
      if (qp.get("depth")) depthSel.value = qp.get("depth");
      if (qp.get("speed") !== null) speedSel.value = qp.get("speed");
      if (qp.get("seed") !== null) seedIn.value = qp.get("seed");
      start();
    }

    this.loadCheckpointOptions(ckptSel).then(() => {
      if (!checkpoint || state.session) return;
      if ([...ckptSel.options].some((o) => o.value === checkpoint)) {
        agentSel.value = "learned";
        syncAgent();
        ckptSel.value = checkpoint;
        hint.textContent = "Snapshot selected. Press Start Game to watch it play.";
        if (App.demoCheckpoint === checkpoint) {
          delete App.demoCheckpoint;
          start();
        }
      } else {
        hint.textContent = `Checkpoint ${checkpoint} is not available for run ` +
          `“${App.run}”; playing with the latest weights instead.`;
      }
    });
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
    const board = new Board(boardNode, { max: 440 });
    const stats = el("div", { style: "width:100%" });
    const msg = el("div", { class: "status-line", role: "status" });
    let session = null, busy = false, over = false;

    const paint = (s) => {
      board.render(s.board);
      stats.replaceChildren(miniStats([
        ["Score", F.n(s.score)], ["Moves", F.n(s.moves)],
        ["Max tile", F.n(s.max_tile)], ["Time", F.dur(s.elapsed)],
      ]));
      over = s.game_over;
      msg.textContent = s.game_over
        ? `game over — ${F.n(s.score)} points, ${F.n(s.moves)} moves, best tile ${F.n(s.max_tile)}`
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
      if (session) API.post(`/api/game/${session}/control`, { action: "stop" }).catch(() => { });
      try {
        const r = await API.post("/api/game/human/start", {});
        session = r.session.id;
        over = false;
        paint(r.session);
        msg.textContent = "";
      } catch (e) { Toast.error("Could not start a game", e.message); }
    };

    const onKey = moveKeyHandler(move);
    document.addEventListener("keydown", onKey);
    this.cleanups.push(() => document.removeEventListener("keydown", onKey));
    this.cleanups.push(() => {
      if (session) API.post(`/api/game/${session}/control`, { action: "stop" }).catch(() => { });
    });
    bindSwipe(boardNode, move);

    const dirs = [["up", "arrow-up", "Up"], ["left", "arrow-left", "Left"],
                  ["down", "arrow-down", "Down"], ["right", "arrow-right", "Right"]];
    const dpad = el("div", { class: "dpad", role: "group", "aria-label": "Move" },
      ...dirs.map(([dir, ic, label]) => {
        const b = button(label, { variant: "outline", size: "icon", icon: ic, title: label });
        b.classList.add(dir);
        b.onclick = () => move(dir);
        return b;
      }));

    const k = (t) => el("kbd", {}, t);
    this.body.replaceChildren(el("div", { class: "row play" },
      el("div", { class: "col-main" }, card(
        cardHeader("Your Game", "Same engine and rules the AI uses."),
        cardContent(el("div", { class: "stack stack-16", style: "align-items:center" },
          boardNode, stats, msg)))),
      el("div", { class: "col-side" }, card(
        cardHeader("Controls"),
        cardContent(el("div", { class: "stack" },
          el("dl", { class: "keys" },
            el("dt", {}, el("span", { class: "kbd-group" }, k("↑"), k("↓"), k("←"), k("→"))),
            el("dd", {}, "Move"),
            el("dt", {}, el("span", { class: "kbd-group" }, k("W"), k("A"), k("S"), k("D"))),
            el("dd", {}, "Move"),
            el("dt", {}, k("Swipe")), el("dd", {}, "Move, on a touch screen")),
          dpad,
          el("div", { class: "inline" },
            button("New Game", { variant: "outline", size: "sm", icon: "rotate-ccw",
                                 onClick: newGame })),
          el("div", { class: "footnote" },
            "Every move is applied by the Python engine, so your game follows " +
            "exactly the rules the AI trains on.")))))));
    newGame();
  },
};
