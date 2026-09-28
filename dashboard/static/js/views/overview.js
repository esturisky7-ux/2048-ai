/* Overview: what the AI is doing right now, and how good it has become. */

/* The Training Progress chart's metrics: one chart, switched in place. */
const OVERVIEW_METRICS = [
  { id: "score", label: "Score", desc: "Rolling mean and median score per game." },
  { id: "tiles", label: "Tile Rates", desc: "Share of recent games that reached each tile." },
  { id: "evaluation", label: "Evaluation",
    desc: "Mean score on fixed seeded games, with the 95% confidence interval." },
  { id: "evalRates", label: "Eval Rates",
    desc: "Share of the fixed evaluation games that reached 2048 and 4096." },
  { id: "best", label: "Highest Tile", desc: "Highest tile reached in each reporting window." },
  { id: "speed", label: "Speed",
    desc: "Training throughput: games per second and thousands of moves per second." },
  { id: "games", label: "Games", desc: "Cumulative games played over training hours." },
];

App.views.overview = {
  title: "Overview",
  subtitle: "Live status, records and training progress",

  mount(root, { actions }) {
    this.root = root;
    this.mode = null;
    this.data = null;
    this._sig = {};
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
    this.root.replaceChildren();
    this.chart = null;
    this._sig = {};
    if (wanted === "loading") {
      this.root.append(card(cardContent(
        el("div", { class: "skeleton", style: "height:56px" }, "loading"))));
      return;
    }
    if (wanted === "welcome") {
      this.root.append(this.welcome());
      return;
    }
    this.buildDashboard();
  },

  buildDashboard() {
    this.statusCard = card();
    this.jobsCard = card();
    this.jobsCard.hidden = true;
    this.runCard = card();
    this.perfCard = card();
    this.chartCard = card();

    this.root.append(this.statusCard, this.jobsCard,
      el("div", { class: "row" },
        el("div", { class: "col-main" }, this.runCard),
        el("div", { class: "col-side" }, this.perfCard)),
      this.chartCard);

    this.buildChart();
    this.loadHistory();
  },

  unmount() { clearTimeout(this._historyTimer); },

  welcome() {
    const start = button("Start Your First Training Run", { icon: "play",
      onClick: () => App.go("training") });
    const watch = button("Watch a Built-in Agent", { variant: "outline", icon: "eye",
      onClick: () => App.go("play") });
    return card(cardContent(
      emptyState("gamepad-2", "No Trained AI Yet",
        "Nothing has been trained on this machine. Training teaches an agent " +
        "to play 2048 by playing against itself — a thousand games takes about " +
        "a minute and already reaches the 512 tile. Everything runs locally; " +
        "nothing is uploaded anywhere.",
        el("div", { class: "stack stack-16", style: "align-items:center" },
          el("div", { class: "inline", style: "justify-content:center" }, start, watch),
          el("div", { class: "footnote" },
            "You can also watch the random, heuristic and expectimax agents right " +
            "now — they need no training.")))));
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

  /* Rebuild a card only when what it shows has changed, so buttons are not
   * swapped out from under the pointer every second. */
  changed(key, value) {
    const sig = JSON.stringify(value);
    if (this._sig[key] === sig) return false;
    this._sig[key] = sig;
    return true;
  },

  paintStatus(s) {
    const t = s.training || {};
    const failure = (s.recent_failures || [])[0];
    const idle = s.exists
      ? `Not training. The last checkpoint was saved ${F.ago(s.checkpoint_saved_at)}.`
      : "Not training. This run has no checkpoint yet.";
    const [iconName, text] = {
      TRAINING: ["activity", "The agent is playing and learning right now."],
      STARTING: ["activity", "The training process is spinning up."],
      STOPPING: ["square", "Finishing the current game, then saving a checkpoint."],
      EVALUATING: ["clipboard-check", "Measuring an agent on a fixed set of seeded games."],
      BENCHMARKING: ["gauge", "Measuring this machine's throughput."],
      EXPERIMENTING: ["flask-conical", "Training a variant and evaluating it."],
      STOPPED: ["pause", idle],
    }[s.state] || ["pause", idle];
    if (!this.changed("status", [s.run, s.state, t.running, t.state, t.external,
                                 s.exists, failure?.id])) {
      this._statusText.textContent = text;       // "saved 12m 3s ago" ticks on
      return;
    }
    const { variant, label } = statusBadge(s.state);

    const buttons = el("div", { class: "inline" });
    if (t.running) {
      const stop = button("Stop Training", { variant: "outline", size: "sm", icon: "square",
                                             disabled: t.state === "STOPPING" });
      stop.onclick = async () => {
        stop.disabled = true;
        try {
          const r = await API.post("/api/training/stop", { run: App.run });
          Toast.show("Stopping training", r.message, "warn");
          App.refreshNow();
        } catch (e) { Toast.error("Could not stop", e.message); stop.disabled = false; }
      };
      buttons.append(stop);
    } else {
      buttons.append(button(s.exists ? "Resume Training" : "Start Training",
        { size: "sm", icon: "play", onClick: () => App.go("training") }));
    }
    buttons.append(
      button("Watch It Play", { variant: "outline", size: "sm", onClick: () => App.go("play") }),
      button("Evaluate", { variant: "outline", size: "sm", onClick: () => App.go("evaluate") }));

    const content = el("div", { class: "stack stack-16" },
      el("div", { class: "banner" },
        el("div", { class: "banner-main" },
          el("div", { class: "icon-tile" }, icon(iconName, 18)),
          el("div", { class: "banner-text" },
            el("div", { class: "inline" },
              el("div", { class: "banner-title" }, `Run “${s.run}”`),
              badge(label, variant)),
            this._statusText = el("div", { class: "card-description" }, text),
            t.running && t.external ? el("div", { class: "footnote" },
              "Started outside this control center — stopping it still saves a checkpoint.")
              : null)),
        buttons));

    if (failure) {
      content.append(el("div", { class: "list-row", style: "padding-bottom:0" },
        badge("Failed", "destructive"),
        el("div", { class: "list-main" },
          el("div", { class: "list-title" }, failure.label),
          el("div", { class: "list-meta" }, failure.error || "no error recorded")),
        button("See Logs", { variant: "outline", size: "sm", onClick: () => App.go("logs") })));
    }
    this.statusCard.replaceChildren(cardContent(content));
  },

  paintRun(s) {
    const sch = s.schedule || {};
    const running = !!s.training?.running;
    const tiles = [
      statTile("Total games", F.n(s.games), s.target_games ? `target ${F.compact(s.target_games)}` : ""),
      statTile("Session games", running ? F.n(s.session_games) : "—",
        running ? F.dur(s.session_seconds) : "no active session"),
      statTile("Training time", F.dur(s.total_train_seconds), "all sessions"),
      statTile("Games / sec", s.games_per_second ? F.n(s.games_per_second, 1) : "—",
        s.moves_per_second ? `${F.compact(s.moves_per_second)} moves/s` : "idle"),
      statTile("Workers", s.workers ? F.n(s.workers) : (running ? "1" : "—"),
        s.training?.external ? "external process" : ""),
      statTile("Learning rate", s.alpha !== null && s.alpha !== undefined
        ? Number(s.alpha).toFixed(4) : "—", "alpha"),
      statTile("Network", s.tuple_set || "—", "n-tuple shape"),
      statTile("Checkpoint", F.ago(s.checkpoint_saved_at),
        sch.games_to_checkpoint !== null && sch.games_to_checkpoint !== undefined
          ? `next in ${F.n(sch.games_to_checkpoint)} games` : ""),
    ];
    if (s.eta_seconds) tiles.push(statTile("Finishes in", F.dur(s.eta_seconds), "at the current rate"));

    const description = sch.eval_every
      ? `Next evaluation in ${F.n(sch.games_to_eval)} games` +
        (sch.seconds_to_eval ? ` (~${F.dur(sch.seconds_to_eval)}).` : ` · every ${F.compact(sch.eval_every)} games.`)
      : running ? "Periodic evaluation is off for this run."
      : s.exists ? `${F.n(s.games)} games trained on a ${s.tuple_set || "?"} network.`
      : "This run has not been trained yet.";
    this.runCard.replaceChildren(
      cardHeader("Current Run", description),
      cardContent(stats(tiles)));
  },

  paintPerf(s) {
    const at = s.all_time || {};
    const roll = s.rolling || {};
    const ev = s.last_eval || {};
    this.perfCard.replaceChildren(
      cardHeader("Performance",
        ev.eval_games
          ? `Training numbers move; the evaluation average is measured on ${F.n(ev.eval_games)} fixed games.`
          : "Training numbers move; evaluation averages come from fixed seeded games.",
        tooltip("Rolling numbers come from recent training games and move as " +
                "the agent learns. Evaluation numbers come from a frozen agent " +
                "replaying a fixed set of seeded games — those are the " +
                "comparable ones.")),
      cardContent(el("div", { class: "stack" },
        stats([
          statTile("Recent average", F.compact(roll.mean_score),
            roll.games ? `last ${F.n(roll.games)} games` : "training statistic"),
          statTile("Evaluation average", ev.mean_score ? F.compact(ev.mean_score) : "—",
            ev.eval_games ? `${F.n(ev.eval_games)} fixed games` : "not evaluated yet"),
          statTile("Median (recent)", F.compact(roll.median_score)),
          statTile("Best score ever", F.compact(at.best_score),
            at.best_score_game ? `game ${F.compact(at.best_score_game)}` : ""),
          statTile("Highest tile ever", F.n(at.best_tile),
            at.best_tile_game ? `game ${F.compact(at.best_tile_game)}` : ""),
          statTile("Longest game", F.n(at.longest_game), "moves"),
        ], { cols2: true }),
        el("div", { class: "stack stack-12" },
          sectionHead("Achievement Rates",
            `${F.compact(at.games)} games · ${F.compact(at.moves)} moves`, { split: true }),
          rateBars(at.tile_rates, at.tile_counts)))));
  },

  paintJobs(s) {
    const jobs = (s.jobs || []).filter((j) => j.type !== "training");
    if (!jobs.length) { this.jobsCard.hidden = true; return; }
    fillCard(this.jobsCard, { title: "Running Now",
      description: "Evaluations, comparisons, benchmarks and experiments in progress." },
      el("div", { class: "list" }, ...jobs.map((j) => jobRow(j, { onStop: () => App.refreshNow() }))));
  },

  /* ------------------------------------------------------------- charts */
  buildChart() {
    let metric = "score";
    try { metric = localStorage.getItem("overviewMetric") || metric; } catch (_) { }
    if (!OVERVIEW_METRICS.some((m) => m.id === metric)) metric = "score";
    this.metric = metric;
    this.chartDesc = el("div", { class: "card-description" });
    this.chart = chartView([], { height: 260, xUnit: "games", xLabel: "games trained" });
    const switcher = toggleGroup(OVERVIEW_METRICS.map((m) => [m.id, m.label]), metric, (id) => {
      this.metric = id;
      try { localStorage.setItem("overviewMetric", id); } catch (_) { }
      this.paintChart();
    }, "Chart metric");
    this.chartCard.replaceChildren(
      el("div", { class: "card-header" },
        el("div", { class: "card-heading" },
          el("div", { class: "card-title" }, "Training Progress"), this.chartDesc),
        el("div", { class: "card-action" }, switcher)),
      cardContent(this.chart));
    this.paintChart();
  },

  async loadHistory() {
    this._lastHistory = Date.now();
    let data;
    try { data = await API.get("/api/history", { run: App.run }); }
    catch (_) { return; }
    this.data = data;
    this.paintChart();
  },

  /* Series and axis options for the selected metric. */
  paintChart() {
    if (!this.chart) return;
    const m = OVERVIEW_METRICS.find((x) => x.id === this.metric) || OVERVIEW_METRICS[0];
    this.chartDesc.textContent = m.desc;
    const h = this.data?.history || {};
    const e = this.data?.evaluations || {};
    const g = h.games || [];
    const evx = e.games_trained || [];
    const base = { xUnit: "games", xLabel: "games trained", percent: false, tileScale: false,
                   emptyText: this.data ? "No training history yet" : "Loading…",
                   label: `${m.label} chart` };
    let series = [], opts = base;

    if (m.id === "score") {
      series = [
        { x: g, y: h.mean_score, color: SERIES_COLORS.mean, label: "Mean (rolling)" },
        { x: g, y: h.median_score, color: SERIES_COLORS.median, label: "Median (rolling)" },
      ];
    } else if (m.id === "tiles") {
      series = [512, 1024, 2048, 4096, 8192].map((t) => (
        { x: g, y: h[`rate_${t}`], color: SERIES_COLORS[`r${t}`], label: String(t) }));
      opts = { ...base, percent: true };
    } else if (m.id === "evaluation") {
      // Score and percentage need separate charts; sharing one axis would
      // flatten the rates into the baseline.
      series = [{ x: evx, y: e.mean_score || [], color: SERIES_COLORS.eval, label: "Mean",
                  points: true, bandColor: SERIES_COLORS.evalBand, bandLabel: "95% CI",
                  band: (e.ci_low && e.ci_low.length === evx.length) ? [e.ci_low, e.ci_high] : null }];
      opts = { ...base, emptyText: "No evaluations yet — run one from the Evaluate page" };
    } else if (m.id === "evalRates") {
      series = [
        { x: evx, y: e.rate_2048 || [], color: SERIES_COLORS.r2048, label: "2048", points: true },
        { x: evx, y: e.rate_4096 || [], color: SERIES_COLORS.r4096, label: "4096", points: true },
      ];
      opts = { ...base, percent: true, emptyText: "No evaluations yet" };
    } else if (m.id === "best") {
      series = [{ x: g, y: (h.max_tile || []).map((v) => (v > 0 ? Math.log2(v) : 0)),
                  color: SERIES_COLORS.tile, label: "Highest tile" }];
      opts = { ...base, tileScale: true };
    } else if (m.id === "speed") {
      series = [
        { x: g, y: h.games_per_second, color: SERIES_COLORS.games, label: "Games/s" },
        { x: g, y: (h.moves_per_second || []).map((v) => v / 1000),
          color: SERIES_COLORS.speed, label: "Moves/s (thousands)" },
      ];
    } else if (m.id === "games") {
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
      series = [{ x: hours, y: g, color: SERIES_COLORS.cumulative, label: "Games played" }];
      opts = { ...base, xUnit: "hours", xLabel: "training hours" };
    }
    updateChart(this.chart, series, opts);
  },

  onRunChange() { App.route(); },
};
