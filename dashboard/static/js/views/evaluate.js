/* Evaluate: measure one agent properly, and see the history of measurements. */

App.views.evaluate = {
  title: "Evaluate",
  subtitle: "Freeze an agent and measure it on a fixed set of seeded games",

  mount(root, { actions }) {
    actions.append(runPicker());
    this.form = el("section", { class: "panel" });
    this.progress = el("section", { class: "panel", style: "display:none" });
    this.result = el("section", { class: "panel", style: "display:none" });
    this.history = el("section", { class: "panel" });
    root.append(
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "Why this is separate from training")),
        el("div", { class: "dim", style: "font-size:13.5px;max-width:75ch" },
          "Training statistics move while they are collected — the agent is " +
          "changing after every move. Evaluation freezes the policy and " +
          "replays the same seeded games every time, so two results differ " +
          "only because the agents differ. Means get a normal-approximation " +
          "confidence interval; tile rates get Wilson intervals, which stay " +
          "correct near 0% and 100%.")),
      this.form, this.progress, this.result, this.history);
    this.buildForm();
    this.loadHistory();
    this._onDone = () => this.loadHistory();
    document.addEventListener("job:done", this._onDone);
  },

  unmount() {
    this._stopWatch && this._stopWatch();
    document.removeEventListener("job:done", this._onDone);
  },

  buildForm() {
    const agent = selectInput("e-agent", [
      { value: "learned", label: "Learned AI" },
      { value: "expectimax", label: "Expectimax" },
      { value: "heuristic", label: "Heuristic" },
      { value: "random", label: "Random" },
    ], "learned");
    const ckpt = selectInput("e-ckpt", [{ value: "", label: "Current weights" }], "");
    const games = numberInput("e-games", App.settings.eval_games, { min: 1, max: 200000, step: 50 });
    const seed = numberInput("e-seed", 987654, { min: 0 });
    const depth = selectInput("e-depth", [1, 2, 3], 1);

    const ckptField = field("Checkpoint", ckpt, "",
      "Evaluate a frozen snapshot to compare an agent with its younger self.");
    const depthField = field("Search depth", depth, "1 = no search");
    const sync = () => {
      const a = agent.value;
      ckptField.style.display = a === "learned" ? "" : "none";
      depthField.style.display = (a === "learned" || a === "expectimax") ? "" : "none";
      if (a === "expectimax") {
        depth.innerHTML = "";
        for (const d of [1, 2, 3, 4]) depth.append(el("option", { value: d, selected: d === 2 }, d));
        if (Number(games.value) > 200) games.value = 50;
      } else if (a === "learned") {
        depth.innerHTML = "";
        for (const d of [1, 2, 3]) depth.append(el("option", { value: d, selected: d === 1 }, d));
      }
    };
    agent.onchange = sync;

    const run = el("button", { class: "btn btn-primary btn-lg" }, "Run evaluation");
    run.onclick = async () => {
      run.disabled = true;
      const payload = {
        agent: agent.value,
        games: Math.max(1, Number(games.value) || 1),
        seed: Number(seed.value) || 0,
        depth: Number(depth.value) || 1,
        run: App.run,
      };
      if (agent.value === "learned" && ckpt.value) payload.checkpoint = ckpt.value;
      try {
        const r = await API.post("/api/evaluate", payload);
        this.watchJob(r.job);
      } catch (e) {
        Toast.error("Could not start the evaluation", e.message);
      } finally { run.disabled = false; }
    };

    this.form.innerHTML = "";
    this.form.append(
      el("div", { class: "panel-head" }, el("h3", {}, "Run an evaluation")),
      el("div", { class: "form-grid" },
        field("Agent", agent),
        ckptField,
        field("Games", games, "more games = tighter intervals",
          "200 is usually enough to compare; 1000 for a number you want to quote."),
        field("Seed", seed, "keep this fixed to compare results",
          "Game i of an evaluation is always the same game for a given seed."),
        depthField),
      el("div", { class: "btn-row", style: "margin-top:18px" }, run,
        el("span", { class: "faint", style: "font-size:12px" },
          "expectimax is slow — about 1–2 games/second")));
    sync();
    this.loadCheckpoints(ckpt);
  },

  async loadCheckpoints(sel) {
    try {
      const data = await API.get("/api/checkpoints");
      for (const c of data.checkpoints || []) {
        if (c.kind === "current" && c.run === App.run) continue;
        sel.append(el("option", { value: c.id },
          `${c.run}${c.kind === "snapshot" ? " · snapshot" : ""} — ${F.compact(c.games)} games` +
          (c.label ? ` (${c.label})` : "")));
      }
    } catch (_) { }
  },

  watchJob(job) {
    this.progress.style.display = "";
    this.result.style.display = "none";
    const bar = el("div", { class: "bar blue" }, el("i", { style: "width:0%" }));
    const text = el("div", { class: "faint", style: "margin-top:8px;font-size:12.5px" }, "starting…");
    const stop = el("button", { class: "btn btn-sm btn-danger" }, "Cancel");
    stop.onclick = () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { });
    this.progress.innerHTML = "";
    this.progress.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Evaluating"),
        el("span", { class: "note" }, job.label)),
      bar, text,
      el("div", { class: "btn-row", style: "margin-top:12px" }, stop));

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      const frac = p.total ? (p.done || 0) / p.total : 0;
      bar.firstChild.style.width = `${frac * 100}%`;
      text.textContent = p.total
        ? `${F.n(p.done || 0)} / ${F.n(p.total)} games · ${F.dur(j.duration)}`
        : `${p.phase || j.state} · ${F.dur(j.duration)}`;
      if (j.state === "COMPLETED" && j.result) {
        this.progress.style.display = "none";
        this.showResult(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.style.display = "none";
        if (j.state === "FAILED") Toast.error("Evaluation failed", j.error);
      }
    });
  },

  showResult(res) {
    this.result.style.display = "";
    this.result.innerHTML = "";
    const exportJson = el("button", { class: "btn btn-sm" }, "Export JSON");
    exportJson.onclick = () => downloadFile(
      `evaluation-${res.label || "agent"}-${res.games}.json`,
      JSON.stringify(res, null, 2));
    const exportCsv = el("button", { class: "btn btn-sm" }, "Export CSV");
    exportCsv.onclick = () => {
      const rows = [["metric", "value"],
        ["agent", res.label], ["games", res.games], ["seed", res.seed],
        ["mean_score", res.mean_score], ["median_score", res.median_score],
        ["std_score", res.std_score],
        ["ci95_low", (res.ci95_mean || [])[0]], ["ci95_high", (res.ci95_mean || [])[1]],
        ["min_score", res.min_score], ["max_score", res.max_score],
        ["mean_moves", res.mean_moves], ["highest_tile", res.highest_tile]];
      for (const [tile, r] of Object.entries(res.tile_rates || {})) {
        rows.push([`rate_${tile}`, typeof r === "object" ? r.rate : r]);
      }
      downloadFile(`evaluation-${res.games}.csv`, toCSV(rows), "text/csv");
    };
    this.result.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Result"),
        el("span", { class: "note" }, res.label || ""),
        el("div", { class: "btn-row" }, exportJson, exportCsv)),
      evaluationResult(res));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/evaluations", { run: App.run, limit: 200 }); }
    catch (_) { return; }
    const rows = data.evaluations || [];
    this.history.innerHTML = "";
    this.history.append(el("div", { class: "panel-head" },
      el("h3", {}, `Evaluation history — run “${App.run}”`),
      el("span", { class: "note" }, `${rows.length} recorded`)));
    if (!rows.length) {
      this.history.append(el("div", { class: "faint" },
        "No evaluations recorded for this run yet. Results from this page and " +
        "from periodic evaluation during training both appear here."));
      return;
    }
    const x = rows.map((r) => r.games_trained || (r.agent || {}).games_trained || 0);
    this.history.append(chartCard("Evaluation mean score over training",
      [{ x, y: rows.map((r) => r.mean_score), color: SERIES_COLORS.eval,
         label: "mean", points: true,
         band: [rows.map((r) => (r.ci95_mean || [0, 0])[0]),
                rows.map((r) => (r.ci95_mean || [0, 0])[1])] }],
      { legend: true, xUnit: "games", xLabel: "games trained" }));

    const body = rows.slice().reverse().map((r) => {
      const ci = r.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", { class: "num" }, F.compact(r.games_trained || (r.agent || {}).games_trained || 0)),
        el("td", { class: "num" }, F.n(r.games)),
        el("td", { class: "num" }, F.n(r.mean_score, 0)),
        el("td", { class: "num faint" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(r.median_score, 0)),
        el("td", { class: "num" }, F.pct(_evalRate(r, "2048") * 100, 1)),
        el("td", { class: "num" }, F.n(r.highest_tile)),
        el("td", { class: "faint nowrap" }, F.date(r.timestamp)));
    });
    this.history.append(el("div", { class: "table-wrap", style: "margin-top:16px" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {},
          el("th", { class: "num" }, "Trained"), el("th", { class: "num" }, "n"),
          el("th", { class: "num" }, "Mean"), el("th", { class: "num" }, "95% CI"),
          el("th", { class: "num" }, "Median"), el("th", { class: "num" }, "2048"),
          el("th", { class: "num" }, "Best tile"), el("th", {}, "When"))),
        el("tbody", {}, ...body))));
  },

  onRunChange() { App.route(); },
};

function _evalRate(record, tile) {
  const e = (record.tile_rates || {})[tile];
  if (e && typeof e === "object") return e.rate || 0;
  return e || 0;
}
