/* Overview: what the AI is doing right now, and how good it has become. */

App.views.overview = {
  title: "Overview",
  subtitle: "Live status, records and training progress",

  mount(root, { actions }) {
    this.root = root;
    this.actions = actions;
    this.charts = {};
    this.mode = null;
    actions.append(runPicker());
    // Status may not have arrived yet on a cold load, so the layout is chosen
    // in render() and re-chosen if the answer changes. Rendering the normal
    // dashboard first and never switching was the original bug: a brand new
    // install saw a wall of dashes instead of the welcome screen.
    this.render(App.status);
  },

  /* Pick a layout for the state we are actually in. */
  render(status) {
    const wanted = !status ? "loading"
      : (status.has_any_run ? "dashboard" : "welcome");
    if (wanted === this.mode) return;
    this.mode = wanted;
    this.root.innerHTML = "";
    this.charts = {};
    if (wanted === "loading") {
      this.root.append(el("section", { class: "panel" },
        el("div", { class: "skeleton", style: "height:90px" }, "loading")));
      return;
    }
    if (wanted === "welcome") {
      this.root.append(this.welcome());
      return;
    }
    this.buildDashboard();
  },

  buildDashboard() {
    const root = this.root;
    this.statusPanel = el("section", { class: "panel hero" });
    this.runPanel = el("section", { class: "panel" });
    this.perfPanel = el("section", { class: "panel" });
    this.chartsPanel = el("section", { class: "panel" });
    this.jobsPanel = el("section", { class: "panel", style: "display:none" });

    root.append(this.statusPanel, this.jobsPanel,
      el("div", { class: "grid split-main" }, this.runPanel, this.perfPanel),
      this.chartsPanel);

    this.buildCharts();
    this.loadHistory();
  },

  unmount() { clearTimeout(this._historyTimer); },

  welcome() {
    const start = el("button", { class: "btn btn-primary btn-lg" },
      "Start your first training run");
    start.onclick = () => App.go("training");
    const watch = el("button", { class: "btn btn-lg" }, "Watch a built-in agent");
    watch.onclick = () => App.go("play");
    return el("section", { class: "panel hero", style: "padding:34px" },
      el("h2", {}, "No trained AI yet"),
      el("p", {},
        "Nothing has been trained on this machine. Training teaches an agent " +
        "to play 2048 by playing against itself — a thousand games takes about " +
        "a minute and already reaches the 512 tile. Everything runs locally; " +
        "nothing is uploaded anywhere."),
      el("div", { class: "btn-row" }, start, watch),
      el("div", { class: "faint", style: "margin-top:18px;font-size:12.5px" },
        "You can also watch the random, heuristic and expectimax agents right " +
        "now — they need no training."));
  },

  /* ------------------------------------------------------------- status */
  onStatus(s) {
    this.render(s);
    if (this.mode !== "dashboard") return;
    this.paintStatus(s);
    this.paintRun(s);
    this.paintPerf(s);
    this.paintJobs(s);
    // History is much heavier than status; refresh it on a slower cadence
    // while training, and once otherwise.
    const now = Date.now();
    if (s.training?.running && now - (this._lastHistory || 0) > 12000) {
      this.loadHistory();
    }
  },

  paintStatus(s) {
    const t = s.training || {};
    const stateInfo = {
      TRAINING: ["live", "Training", "The agent is playing and learning right now."],
      STARTING: ["busy", "Starting", "The training process is spinning up."],
      STOPPING: ["warn", "Stopping", "Finishing the current game, then saving a checkpoint."],
      EVALUATING: ["busy", "Evaluating", "Measuring an agent on a fixed set of seeded games."],
      BENCHMARKING: ["busy", "Benchmarking", "Measuring this machine's throughput."],
      EXPERIMENTING: ["busy", "Running an experiment", "Training a variant and evaluating it."],
      STOPPED: ["idle", "Idle", "Nothing is running. The saved checkpoint is ready to use."],
    }[s.state] || ["idle", "Idle", ""];

    const buttons = el("div", { class: "btn-row" });
    if (t.running) {
      const stop = el("button", { class: "btn btn-danger" }, "Stop training");
      stop.disabled = t.state === "STOPPING";
      stop.onclick = async () => {
        stop.disabled = true;
        try {
          const r = await API.post("/api/training/stop", { run: App.run });
          Toast.show("Stopping training", r.message, "warn");
          App.refreshNow();
        } catch (e) { Toast.error("Could not stop", e.message); stop.disabled = false; }
      };
      buttons.append(stop);
      if (t.external) {
        buttons.append(el("span", { class: "faint", style: "font-size:12px" },
          "started outside this control center — stopping it still saves a checkpoint"));
      }
    } else {
      const resume = el("button", { class: "btn btn-primary" },
        s.exists ? "Resume training" : "Start training");
      resume.onclick = () => App.go("training");
      buttons.append(resume);
    }
    const watch = el("button", { class: "btn" }, "Watch it play");
    watch.onclick = () => App.go("play");
    const evaluate = el("button", { class: "btn" }, "Evaluate");
    evaluate.onclick = () => App.go("evaluate");
    buttons.append(watch, evaluate);

    this.statusPanel.innerHTML = "";
    this.statusPanel.append(
      el("div", { style: "display:flex;gap:14px;align-items:center;flex-wrap:wrap" },
        el("span", { class: `pill ${stateInfo[0]}` }, el("i"), el("span", {}, stateInfo[1].toUpperCase())),
        el("div", { style: "flex:1;min-width:200px" },
          el("h2", { style: "margin:0;font-size:17px" }, `Run “${s.run}”`),
          el("div", { class: "dim", style: "font-size:13px" }, stateInfo[2]))),
      el("div", { style: "margin-top:16px" }, buttons));

    if (s.recent_failures && s.recent_failures.length) {
      const f = s.recent_failures[0];
      const view = el("button", { class: "btn btn-sm" }, "See logs");
      view.onclick = () => App.go("logs");
      this.statusPanel.append(el("div", {
        class: "job-card", style: "margin-top:16px;border-color:var(--danger)"
      },
        el("span", { class: "pill bad" }, el("i"), el("span", {}, "FAILED")),
        el("div", { class: "grow" },
          el("div", { class: "title" }, f.label),
          el("div", { class: "meta" }, f.error || "no error recorded")),
        view));
    }
  },

  paintRun(s) {
    const sch = s.schedule || {};
    const rows = [
      ["Total games", F.n(s.games), s.target_games ? `target ${F.compact(s.target_games)}` : null],
      ["Session games", F.n(s.session_games), s.session_seconds ? F.dur(s.session_seconds) : null],
      ["Training time", F.dur(s.total_train_seconds), "all sessions"],
      ["Games / sec", s.games_per_second ? F.n(s.games_per_second, 1) : "—",
        s.moves_per_second ? `${F.compact(s.moves_per_second)} moves/s` : "idle"],
      ["Workers", s.workers ? F.n(s.workers) : (s.training?.running ? "1" : "—"),
        s.training?.external ? "external process" : null],
      ["Learning rate", s.alpha !== null && s.alpha !== undefined ? Number(s.alpha).toFixed(4) : "—",
        "alpha"],
      ["Network", s.tuple_set || "—", "n-tuple shape"],
      ["Checkpoint", F.ago(s.checkpoint_saved_at),
        sch.games_to_checkpoint !== null && sch.games_to_checkpoint !== undefined
          ? `next in ${F.n(sch.games_to_checkpoint)} games` : null],
    ];
    if (sch.eval_every) {
      rows.push(["Next evaluation", `${F.n(sch.games_to_eval)} games`,
        sch.seconds_to_eval ? `~${F.dur(sch.seconds_to_eval)}` : `every ${F.compact(sch.eval_every)}`]);
    }
    if (s.eta_seconds) {
      rows.push(["Finishes in", F.dur(s.eta_seconds), "at the current rate"]);
    }

    this.runPanel.innerHTML = "";
    this.runPanel.append(
      el("div", { class: "panel-head" }, el("h3", {}, "Current run")),
      el("div", { class: "stats" },
        ...rows.map(([l, v, sub]) => statTile(l, v, sub))));
  },

  paintPerf(s) {
    const at = s.all_time || {};
    const roll = s.rolling || {};
    const ev = s.last_eval || {};
    this.perfPanel.innerHTML = "";
    add(this.perfPanel,
      el("div", { class: "panel-head" },
        el("h3", {}, "Performance"),
        tooltip("Rolling numbers come from recent training games and move as " +
                "the agent learns. Evaluation numbers come from a frozen agent " +
                "replaying a fixed set of seeded games — those are the " +
                "comparable ones.")),
      el("div", { class: "stats" },
        statTile("Recent average", F.compact(roll.mean_score),
          roll.games ? `last ${F.n(roll.games)} games` : "training statistic"),
        statTile("Evaluation average", ev.mean_score ? F.compact(ev.mean_score) : "—",
          ev.eval_games ? `${F.n(ev.eval_games)} fixed games` : "not evaluated yet",
          "accent"),
        statTile("Median (recent)", F.compact(roll.median_score)),
        statTile("Best score ever", F.compact(at.best_score),
          at.best_score_game ? `game ${F.compact(at.best_score_game)}` : null, "good"),
        statTile("Highest tile ever", F.n(at.best_tile),
          at.best_tile_game ? `game ${F.compact(at.best_tile_game)}` : null, "good"),
        statTile("Longest game", F.n(at.longest_game), "moves")),
      el("div", { class: "panel-head", style: "margin-top:18px" },
        el("h3", {}, "Achievement rates"),
        el("span", { class: "note" }, `${F.compact(at.games)} games · ${F.compact(at.moves)} moves`)),
      rateBars(at.tile_rates, at.tile_counts));
  },

  paintJobs(s) {
    const jobs = (s.jobs || []).filter((j) => j.type !== "training");
    if (!jobs.length) { this.jobsPanel.style.display = "none"; return; }
    this.jobsPanel.style.display = "";
    this.jobsPanel.innerHTML = "";
    this.jobsPanel.append(el("div", { class: "panel-head" },
      el("h3", {}, "Running now")));
    for (const j of jobs) this.jobsPanel.append(jobCard(j, { onStop: () => App.refreshNow() }));
  },

  /* ------------------------------------------------------------- charts */
  buildCharts() {
    this.chartsPanel.innerHTML = "";
    this.chartsPanel.append(el("div", { class: "panel-head" },
      el("h3", {}, "Training progress"),
      el("span", { class: "note" },
        "hover any chart for the exact value and game number")));
    const grid = el("div", { class: "grid cols-2" });
    this.charts.score = chartCard("Average score", [],
      { legend: true, xUnit: "games", xLabel: "games trained" });
    this.charts.rates = chartCard("Tile achievement rate", [],
      { legend: true, percent: true, xUnit: "games", xLabel: "games trained" });
    this.charts.tile = chartCard("Highest tile reached", [],
      { tileScale: true, xUnit: "games", xLabel: "games trained" });
    this.charts.speed = chartCard("Training speed", [],
      { legend: true, xUnit: "games", xLabel: "games trained" });
    this.charts.evalc = chartCard("Evaluation score (fixed seeded games)", [],
      { legend: true, xUnit: "games", xLabel: "games trained" });
    this.charts.evalrate = chartCard("Evaluation tile rates (fixed seeded games)", [],
      { legend: true, percent: true, xUnit: "games", xLabel: "games trained" });
    this.charts.games = chartCard("Games played over time", [],
      { xUnit: "hours", xLabel: "hours" });
    grid.append(this.charts.score, this.charts.rates,
      this.charts.evalc, this.charts.evalrate,
      this.charts.tile, this.charts.speed, this.charts.games);
    this.chartsPanel.append(grid);
  },

  async loadHistory() {
    this._lastHistory = Date.now();
    let data;
    try { data = await API.get("/api/history", { run: App.run }); }
    catch (_) { return; }
    if (!this.charts.score) return;
    const h = data.history || {};
    const e = data.evaluations || {};
    const g = h.games || [];

    updateChartCard(this.charts.score, [
      { x: g, y: h.mean_score, color: SERIES_COLORS.mean, label: "mean (rolling)" },
      { x: g, y: h.median_score, color: SERIES_COLORS.median, label: "median (rolling)" },
      { x: e.games_trained || [], y: e.mean_score || [], color: SERIES_COLORS.eval,
        label: "evaluation", points: true, dash: [4, 3] },
    ]);
    updateChartCard(this.charts.rates, [
      { x: g, y: h.rate_512, color: SERIES_COLORS.r512, label: "512" },
      { x: g, y: h.rate_1024, color: SERIES_COLORS.r1024, label: "1024" },
      { x: g, y: h.rate_2048, color: SERIES_COLORS.r2048, label: "2048" },
      { x: g, y: h.rate_4096, color: SERIES_COLORS.r4096, label: "4096" },
      { x: g, y: h.rate_8192, color: SERIES_COLORS.r8192, label: "8192" },
    ]);
    updateChartCard(this.charts.tile, [
      { x: g, y: (h.max_tile || []).map((v) => (v > 0 ? Math.log2(v) : 0)),
        color: SERIES_COLORS.tile, label: "best tile in window" },
    ]);
    updateChartCard(this.charts.speed, [
      { x: g, y: h.games_per_second, color: SERIES_COLORS.games, label: "games/s" },
      { x: g, y: (h.moves_per_second || []).map((v) => v / 1000),
        color: SERIES_COLORS.speed, label: "moves/s (thousands)" },
    ]);
    const evx = e.games_trained || [];
    // Score and percentage need separate axes; sharing one would flatten the
    // rates into the baseline.
    updateChartCard(this.charts.evalc, [
      { x: evx, y: e.mean_score || [], color: SERIES_COLORS.eval,
        label: "mean score (95% CI)", points: true,
        band: (e.ci_low && e.ci_low.length === evx.length) ? [e.ci_low, e.ci_high] : null },
    ], { emptyText: "no evaluations yet — run one from the Evaluate page" });
    updateChartCard(this.charts.evalrate, [
      { x: evx, y: e.rate_2048 || [], color: SERIES_COLORS.r2048, label: "2048", points: true },
      { x: evx, y: e.rate_4096 || [], color: SERIES_COLORS.r4096, label: "4096", points: true },
    ], { emptyText: "no evaluations yet" });

    // Wall-clock elapsed would include every hour the machine sat idle
    // between sessions, which turns a multi-day run into one vertical line.
    // Accumulate only the gaps short enough to have been actual training.
    const walls = h.wall_time || [];
    const GAP = 600;                       // 10 minutes ends a session
    let elapsed = 0;
    const hours = walls.map((t, i) => {
      if (i > 0) {
        const dt = t - walls[i - 1];
        if (dt > 0 && dt < GAP) elapsed += dt;
      }
      return elapsed / 3600;
    });
    updateChartCard(this.charts.games, [
      { x: hours, y: g, color: SERIES_COLORS.games, label: "cumulative games" },
    ], { xLabel: "training hours" });
  },

  onRunChange() { App.route(); },
};
