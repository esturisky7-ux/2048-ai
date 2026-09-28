/* Training: start a run, continue one, stop safely, watch it live. */

const QUICK_AMOUNTS = [1000, 10000, 50000, 100000];

App.views.training = {
  title: "Training",
  subtitle: "Start, continue and supervise training runs",

  mount(root, { actions }) {
    this.root = root;
    this._sig = {};
    this._gen = (this._gen || 0) + 1;
    actions.append(runPicker());
    this.live = card();
    this.continueCard = card();
    this.newCard = card();
    this.historyCard = card();
    this.live.hidden = this.continueCard.hidden = true;
    root.append(this.live, this.continueCard, this.newCard, this.historyCard);
    this.buildNew();
    this.loadJobs();
    if (App.status) this.onStatus(App.status);
  },

  unmount() { clearTimeout(this._jobTimer); this._gen = (this._gen || 0) + 1; },

  onStatus(s) {
    this.paintLive(s);
    this.paintContinue(s);
    // Keep the "new run" defaults sensible as runs appear.
    const nameInput = $("#t-run");
    if (nameInput && !nameInput.dataset.touched && !s.has_any_run) {
      nameInput.value = "default";
    }
  },

  /* ------------------------------------------------------- live section */
  paintLive(s) {
    const t = s.training || {};
    if (!t.running) { this.live.hidden = true; return; }
    const st = t.status || {};
    const job = t.job;
    const sch = s.schedule || {};
    const target = s.target_games;
    const frac = target ? Math.min(1, s.games / target) : 0;
    const stopping = t.state === "STOPPING";

    const stop = button(stopping ? "Stopping…" : "Stop Gracefully",
      { variant: "outline", size: "sm", icon: "square", disabled: stopping,
        onClick: () => { stop.disabled = true; this.stop(); } });

    const schedule = [
      `Next checkpoint in ${F.n(sch.games_to_checkpoint ?? 0)} games`,
      sch.eval_every ? `next evaluation in ${F.n(sch.games_to_eval ?? 0)} games` : null,
      st.pid ? `pid ${st.pid}` : null,
    ].filter(Boolean).join(" · ");

    fillCard(this.live, {
      title: job ? job.label : `Run “${s.run}”`,
      description: schedule,
      action: [
        stopping ? badge("Stopping", "outline") : badge("Live", "default", { dot: true }),
        badge(`Session ${F.dur(s.session_seconds)}`, "secondary"),
      ],
    },
    el("div", { class: "stack stack-16" },
      stats([
        statTile("Games", F.n(s.games), target ? `of ${F.n(target)}` : "continuous"),
        statTile("Games / sec", F.n(s.games_per_second, 1)),
        statTile("Moves / sec", F.compact(s.moves_per_second)),
        statTile("Session games", F.n(s.session_games)),
        statTile("Recent mean", F.compact((s.rolling || {}).mean_score),
          `last ${F.n((s.rolling || {}).games || 0)} games`),
        statTile("Best tile (recent)", F.n((s.rolling || {}).max_tile)),
        statTile("Learning rate", s.alpha ? Number(s.alpha).toFixed(4) : "—"),
        statTile("ETA", s.eta_seconds ? F.dur(s.eta_seconds) : "—",
          target ? "at the current rate" : "unlimited run"),
      ]),
      target ? el("div", { class: "stack stack-8" },
        progressBar(frac),
        el("div", { class: "footnote" }, `${F.pct(frac * 100, 1)} of this session's target`)) : null,
      el("div", { class: "inline inline-12" }, stop,
        el("span", { class: "note" },
          t.external
            ? "Started outside this control center (a terminal, or an earlier server). " +
              "Stopping still finishes the current game and writes a checkpoint."
            : "Stopping finishes the current game, writes a checkpoint and flushes " +
              "statistics — nothing is lost."))));
  },

  async stop() {
    try {
      const r = await API.post("/api/training/stop", { run: App.run });
      Toast.show("Stopping training", r.message, "warn");
      App.refreshNow();
    } catch (e) { Toast.error("Could not stop", e.message); }
  },

  /* --------------------------------------------------- continue section */
  paintContinue(s) {
    if (!s.exists || s.training?.running) {
      this.continueCard.hidden = true;
      this._sig.cont = null;
      return;
    }
    // Rebuilt only when the run changes, so typed values survive status ticks.
    const sig = `${s.run}|${s.games}|${s.tuple_set}`;
    if (this._sig.cont === sig) return;
    this._sig.cont = sig;

    const workers = numberInput("t-cont-workers", App.settings.workers, { min: 1, max: 64 });
    const custom = numberInput("t-cont-custom", 25000, { min: 1, step: 1000 });
    const go = (games) => async () => {
      try {
        await API.post("/api/training/resume", {
          run: s.run, games, workers: Number(workers.value) || 1,
        });
        Toast.ok("Training started", games ? `${F.n(games)} more games` : "continuous");
        App.refreshNow();
        this.loadJobs();
      } catch (e) { Toast.error("Could not start training", e.message); }
    };

    const quick = el("div", { class: "inline", style: "justify-content:center" },
      button("Resume Training", { icon: "play", onClick: go(0),
        title: "Train continuously until you stop it" }),
      ...QUICK_AMOUNTS.map((n) => button(`Train ${compactNumber(n)} More`,
        { variant: "outline", onClick: go(n) })));

    this.continueCard.replaceChildren(cardContent(emptyState("pause", "Not Training",
      `Run “${s.run}” has ${F.n(s.games)} games on a ${s.tuple_set || "?"} network. ` +
      "Resume it, or start a new run below.",
      el("div", { class: "stack stack-16", style: "width:100%;align-items:center" },
        quick,
        el("div", { class: "form-grid", style: "--min:180px;width:100%;text-align:left" },
          field("Custom amount", el("div", { class: "inline", style: "flex-wrap:nowrap" }, custom,
            button("Train This Many", { variant: "outline",
              onClick: () => go(Math.max(1, Number(custom.value) || 1))() }))),
          field("Workers", workers,
            `this machine reports ${navigator.hardwareConcurrency || "?"} logical cores`)),
        el("div", { class: "footnote" },
          "Continuing keeps the run's saved configuration, game counter and records.")))));
    this.continueCard.hidden = false;
  },

  /* -------------------------------------------------------- new run form */
  buildNew() {
    const cores = navigator.hardwareConcurrency || 2;
    const suggested = Math.max(1, Math.min(cores - 1, Math.floor(cores / 2))) || 1;

    const name = el("input", { type: "text", id: "t-run", value: "", maxlength: 64,
      placeholder: "my-run", autocomplete: "off" });
    name.oninput = () => { name.dataset.touched = "1"; };
    const games = numberInput("t-games", 20000, { min: 0, step: 1000 });
    const unlimited = checkbox("Train continuously", { id: "t-unlimited" });
    unlimited.input.onchange = () => { games.disabled = unlimited.input.checked; };
    const workers = numberInput("t-workers", suggested, { min: 1, max: 64 });
    const tupleSet = selectInput("t-tuple", [
      { value: "4x6", label: "4x6 — strongest (268 MB)" },
      { value: "4x5", label: "4x5 — lighter (17 MB)" },
      { value: "8x4", label: "8x4 — tiny and quick (2 MB)" },
    ], "4x6");

    const alpha = numberInput("t-alpha", 0.1, { min: 0.0001, max: 1, step: 0.005 });
    const evalEvery = numberInput("t-eval-every", 20000, { min: 0, step: 1000 });
    const evalGames = numberInput("t-eval-games", 200, { min: 1, step: 50 });
    const ckptEvery = numberInput("t-ckpt-every", 2000, { min: 0, step: 500 });
    const reportEvery = numberInput("t-report-every", 200, { min: 0, step: 100 });
    const snapEvery = numberInput("t-snap-every", 0, { min: 0, step: 5000 });
    const seed = numberInput("t-seed", 12345, { min: 0 });
    const epsilon = numberInput("t-epsilon", 0, { min: 0, max: 1, step: 0.01 });
    const gamma = numberInput("t-gamma", 1.0, { min: 0, max: 1, step: 0.01 });
    const decay = numberInput("t-decay", 1.0, { min: 0.0001, max: 1, step: 0.05 });

    const start = button("Start Training", { icon: "play" });
    start.onclick = async () => {
      if (!name.value.trim()) {
        Toast.warn("Name the run", "Give the run a name so you can find it later.");
        name.focus();
        return;
      }
      const payload = {
        run: name.value.trim(),
        games: unlimited.input.checked ? 0 : Math.max(0, Number(games.value) || 0),
        workers: Number(workers.value) || 1,
        tuple_set: tupleSet.value,
        alpha: Number(alpha.value),
        eval_every: Number(evalEvery.value),
        eval_games: Number(evalGames.value),
        checkpoint_every: Number(ckptEvery.value),
        report_every: Number(reportEvery.value),
        snapshot_every: Number(snapEvery.value),
        seed: Number(seed.value),
        epsilon: Number(epsilon.value),
        gamma: Number(gamma.value),
        alpha_decay: Number(decay.value),
      };
      start.disabled = true;
      try {
        await API.post("/api/training/start", payload);
        Toast.ok("Training started", `run “${payload.run}”`);
        App.setRun(payload.run);
        App.refreshNow();
        this.loadJobs();
      } catch (e) {
        Toast.error("Could not start training", e.message);
      } finally { start.disabled = false; }
    };

    fillCard(this.newCard, {
      title: "Start a New Run",
      description: "A fresh agent that knows nothing — takes about a minute to reach 512.",
    },
    el("div", { class: "stack" },
      formGrid([
        field("Run name", name, "letters, digits, dot, dash, underscore",
          "Each run has its own weights, statistics and history."),
        el("div", { class: "field" },
          el("label", { class: "field-label", for: "t-games" }, "Games"),
          games, unlimited.node,
          el("div", { class: "field-description" }, "this session's games")),
        field("Workers", workers,
          `${cores} logical cores detected — ${suggested} suggested`,
          "Each worker is a process playing games and updating the same " +
          "shared weight file. More than one per core is usually slower."),
        field("Network", tupleSet, "cannot be changed later",
          "The n-tuple network's shape. Bigger is stronger and uses more disk."),
      ]),
      collapsible("Advanced Settings", formGrid([
        field("Learning rate (alpha)", alpha, "default 0.1",
          "How far each update moves the value estimate."),
        field("Learning-rate decay", decay, "1.0 = no decay"),
        field("Exploration (epsilon)", epsilon, "default 0 — off deliberately",
          "Probability of a random move. 2048's random tiles already " +
          "explore for you; random moves mostly destroy structure."),
        field("Discount (gamma)", gamma, "1.0 is correct for 2048"),
        field("Random seed", seed, "tile spawns"),
        field("Evaluate every", evalEvery, "games · 0 = off",
          "Runs the fixed-seed evaluation during training so progress is " +
          "measured, not just observed."),
        field("Evaluation games", evalGames, "per evaluation"),
        field("Checkpoint every", ckptEvery, "games"),
        field("Report every", reportEvery, "games · chart resolution"),
        field("Snapshot every", snapEvery, "games · 0 = off",
          "Freezes a full copy of the weights so you can compare the agent " +
          "with its younger self."),
      ], 180)),
      el("div", { class: "inline" }, start)));
  },

  /* ----------------------------------------------------------- job list */
  async loadJobs() {
    clearTimeout(this._jobTimer);
    const gen = this._gen;
    let data;
    try { data = await API.get("/api/jobs", { type: "training", limit: 12 }); }
    catch (_) { data = null; }
    if (gen !== this._gen) return;          // the page was left meanwhile
    clearTimeout(this._jobTimer);
    this._jobTimer = setTimeout(() => this.loadJobs(), 6000);
    if (!data) return;
    const jobs = data.jobs || [];
    if (!jobs.length) {
      fillCard(this.historyCard, { title: "Recent Training Jobs" },
        el("div", { class: "note" }, "Training runs started from this page appear here."));
      return;
    }
    fillCard(this.historyCard, { title: "Recent Training Jobs" },
      table(["State", "Job", ["Duration", { num: true }], ["Started", { num: true }], "Error"],
        jobs.map((j) => el("tr", {},
          el("td", {}, jobStateBadge(j.state)),
          el("td", {}, j.label),
          el("td", { class: "num" }, F.dur(j.duration)),
          el("td", { class: "num muted" }, j.started_at ? F.date(j.started_at) : "—"),
          el("td", { class: "muted wrap" }, j.error || "")))));
  },

  onRunChange() { App.route(); },
};
