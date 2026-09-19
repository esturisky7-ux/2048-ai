/* Training: start a run, continue one, stop safely, watch it live. */

const QUICK_AMOUNTS = [1000, 10000, 50000, 100000];

App.views.training = {
  title: "Training",
  subtitle: "Start, continue and supervise training runs",

  mount(root, { actions }) {
    this.root = root;
    actions.append(runPicker());
    this.live = el("section", { class: "panel", style: "display:none" });
    this.continuePanel = el("section", { class: "panel", style: "display:none" });
    this.newPanel = el("section", { class: "panel" });
    this.historyPanel = el("section", { class: "panel" });
    root.append(this.live, this.continuePanel, this.newPanel, this.historyPanel);
    this.buildNew();
    this.loadJobs();
    if (App.status) this.onStatus(App.status);
  },

  unmount() { clearTimeout(this._jobTimer); },

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
    if (!t.running) { this.live.style.display = "none"; return; }
    this.live.style.display = "";
    const st = t.status || {};
    const job = t.job;
    const sch = s.schedule || {};
    const target = s.target_games;
    const frac = target ? Math.min(1, s.games / target) : 0;

    this.live.innerHTML = "";
    const stop = el("button", { class: "btn btn-danger" },
      t.state === "STOPPING" ? "Stopping…" : "Stop gracefully");
    stop.disabled = t.state === "STOPPING" || t.external;
    stop.onclick = () => this.stop();

    add(this.live,
      el("div", { class: "panel-head" },
        el("span", { class: `pill ${t.state === "STOPPING" ? "warn" : "live"}` },
          el("i"), el("span", {}, t.state === "STOPPING" ? "STOPPING" : "TRAINING • LIVE")),
        el("h3", { style: "margin-left:4px" }, job ? job.label : `run ${s.run}`),
        el("span", { class: "note" }, `session ${F.dur(s.session_seconds)}`)),
      el("div", { class: "stats" },
        statTile("Games", F.n(s.games), target ? `of ${F.n(target)}` : "continuous"),
        statTile("Games / sec", F.n(s.games_per_second, 1)),
        statTile("Moves / sec", F.compact(s.moves_per_second)),
        statTile("Session games", F.n(s.session_games)),
        statTile("Recent mean", F.compact((s.rolling || {}).mean_score),
          `last ${F.n((s.rolling || {}).games || 0)} games`),
        statTile("Best tile (recent)", F.n((s.rolling || {}).max_tile)),
        statTile("Learning rate", s.alpha ? Number(s.alpha).toFixed(4) : "—"),
        statTile("ETA", s.eta_seconds ? F.dur(s.eta_seconds) : "—",
          target ? "at the current rate" : "unlimited run")),
      target ? el("div", { style: "margin-top:14px" },
        el("div", { class: "bar ok" }, el("i", { style: `width:${frac * 100}%` })),
        el("div", { class: "faint", style: "margin-top:6px;font-size:12px" },
          `${F.pct(frac * 100, 1)} of this session's target`)) : null,
      el("div", { class: "btn-row", style: "margin-top:16px" }, stop,
        t.external ? el("span", { class: "faint", style: "font-size:12px" },
          "this run was started from the command line; stop it there") : null),
      el("div", { class: "faint", style: "margin-top:12px;font-size:12px" },
        `Next checkpoint in ${F.n(sch.games_to_checkpoint ?? 0)} games`,
        sch.eval_every ? ` · next evaluation in ${F.n(sch.games_to_eval ?? 0)} games` : "",
        st.pid ? ` · pid ${st.pid}` : ""),
      el("div", { class: "faint", style: "margin-top:6px;font-size:12px" },
        "Stopping finishes the current game, writes a checkpoint and flushes " +
        "statistics — nothing is lost."));
  },

  async stop() {
    try {
      const r = await API.post("/api/training/stop", {});
      Toast.show("Stopping training", r.message, "warn");
      App.refreshNow();
    } catch (e) { Toast.error("Could not stop", e.message); }
  },

  /* --------------------------------------------------- continue section */
  paintContinue(s) {
    if (!s.exists || s.training?.running) {
      this.continuePanel.style.display = "none";
      return;
    }
    this.continuePanel.style.display = "";
    this.continuePanel.innerHTML = "";

    const workers = numberInput("t-cont-workers", App.settings.workers,
      { min: 1, max: 64, style: "max-width:90px" });
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

    const quick = el("div", { class: "btn-row" });
    for (const n of QUICK_AMOUNTS) {
      const b = el("button", { class: "btn" }, `Train ${F.compact(n)} more`);
      b.onclick = go(n);
      quick.append(b);
    }
    const cont = el("button", { class: "btn btn-primary" }, "Train continuously");
    cont.onclick = go(0);
    quick.append(cont);

    const customGo = el("button", { class: "btn" }, "Train this many");
    customGo.onclick = () => go(Math.max(1, Number(custom.value) || 1))();

    this.continuePanel.append(
      el("div", { class: "panel-head" },
        el("h3", {}, `Continue “${s.run}”`),
        el("span", { class: "note" },
          `${F.n(s.games)} games trained · ${s.tuple_set} network`)),
      quick,
      el("div", { class: "form-grid", style: "margin-top:16px;align-items:end" },
        field("Custom amount", el("div", { class: "btn-row" }, custom, customGo)),
        field("Workers", workers,
          `this machine reports ${navigator.hardwareConcurrency || "?"} logical cores`)),
      el("div", { class: "faint", style: "margin-top:12px;font-size:12px" },
        "Continuing keeps the run's saved configuration, game counter and records."));
  },

  /* -------------------------------------------------------- new run form */
  buildNew() {
    const cores = navigator.hardwareConcurrency || 2;
    const suggested = Math.max(1, Math.min(cores - 1, Math.floor(cores / 2))) || 1;

    const name = el("input", { type: "text", id: "t-run", value: "", maxlength: 64,
      placeholder: "my-run", autocomplete: "off" });
    name.oninput = () => { name.dataset.touched = "1"; };
    const games = numberInput("t-games", 20000, { min: 0, step: 1000 });
    const unlimited = el("input", { type: "checkbox", id: "t-unlimited" });
    unlimited.onchange = () => { games.disabled = unlimited.checked; };
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

    const start = el("button", { class: "btn btn-primary btn-lg" }, "Start training");
    start.onclick = async () => {
      if (!name.value.trim()) {
        Toast.warn("Name the run", "Give the run a name so you can find it later.");
        name.focus();
        return;
      }
      const payload = {
        run: name.value.trim(),
        games: unlimited.checked ? 0 : Math.max(0, Number(games.value) || 0),
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

    this.newPanel.innerHTML = "";
    this.newPanel.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Start a new run"),
        el("span", { class: "note" },
          "a fresh agent that knows nothing — takes about a minute to reach 512")),
      el("div", { class: "form-grid" },
        field("Run name", name, "letters, digits, dot, dash, underscore",
          "Each run has its own weights, statistics and history."),
        field("Games", el("div", {},
          games,
          el("label", { class: "check", style: "margin-top:7px" },
            unlimited, el("span", {}, "Train continuously"))),
          "this session's games"),
        field("Workers", workers,
          `${cores} logical cores detected — ${suggested} suggested`,
          "Each worker is a process playing games and updating the same " +
          "shared weight file. More than one per core is usually slower."),
        field("Network", tupleSet, "cannot be changed later",
          "The n-tuple network's shape. Bigger is stronger and uses more disk.")),
      el("details", { class: "advanced" },
        el("summary", {}, "Advanced settings"),
        el("div", { class: "form-grid" },
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
            "with its younger self."))),
      el("div", { class: "btn-row", style: "margin-top:18px" }, start));
  },

  /* ----------------------------------------------------------- job list */
  async loadJobs() {
    let data;
    try { data = await API.get("/api/jobs", { type: "training", limit: 12 }); }
    catch (_) { return; }
    if (!this.historyPanel) return;
    const jobs = data.jobs || [];
    this.historyPanel.innerHTML = "";
    this.historyPanel.append(el("div", { class: "panel-head" },
      el("h3", {}, "Recent training jobs")));
    if (!jobs.length) {
      this.historyPanel.append(el("div", { class: "faint" },
        "Training runs started from this page appear here."));
      return;
    }
    const rows = jobs.map((j) => el("tr", {},
      el("td", {}, el("span", {
        class: `pill ${j.state === "RUNNING" ? "live" :
                j.state === "COMPLETED" ? "idle" :
                j.state === "FAILED" ? "bad" : "warn"}`
      }, el("i"), el("span", {}, j.state))),
      el("td", {}, j.label),
      el("td", { class: "num" }, F.dur(j.duration)),
      el("td", { class: "num faint" }, j.started_at ? F.date(j.started_at) : "—"),
      el("td", { class: "faint" }, j.error || "")));
    this.historyPanel.append(el("div", { class: "table-wrap" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {},
          el("th", {}, "State"), el("th", {}, "Job"),
          el("th", { class: "num" }, "Duration"),
          el("th", { class: "num" }, "Started"), el("th", {}, "Error"))),
        el("tbody", {}, ...rows))));
    this._jobTimer = setTimeout(() => this.loadJobs(), 6000);
  },

  onRunChange() { App.route(); },
};
