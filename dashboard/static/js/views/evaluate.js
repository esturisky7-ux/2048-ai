/* Evaluate: measure one agent properly, and see the history of measurements. */

App.views.evaluate = {
  title: "Evaluate",
  subtitle: "Freeze an agent and measure it on a fixed set of seeded games",

  mount(root, { actions }) {
    actions.append(runPicker());
    this.form = card();
    this.progress = el("div", { hidden: true });
    this.result = card();
    this.result.hidden = true;
    this.history = card();
    root.append(
      card(cardHeader("Why This Is Separate From Training"),
        cardContent(prose(
          "Training statistics move while they are collected — the agent is " +
          "changing after every move. Evaluation freezes the policy and " +
          "replays the same seeded games every time, so two results differ " +
          "only because the agents differ. Means get a normal-approximation " +
          "confidence interval; tile rates get Wilson intervals, which stay " +
          "correct near 0% and 100%."))),
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
      ckptField.hidden = a !== "learned";
      depthField.hidden = !(a === "learned" || a === "expectimax");
      if (a === "expectimax") {
        depth.replaceChildren(...[1, 2, 3, 4].map((d) =>
          el("option", { value: d, selected: d === 2 }, d)));
        if (Number(games.value) > 200) games.value = 50;
      } else if (a === "learned") {
        depth.replaceChildren(...[1, 2, 3].map((d) =>
          el("option", { value: d, selected: d === 1 }, d)));
      }
    };
    agent.onchange = sync;

    const run = button("Run Evaluation", { icon: "play" });
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

    fillCard(this.form, { title: "Run an Evaluation" },
      el("div", { class: "stack" },
        formGrid([
          field("Agent", agent),
          ckptField,
          field("Games", games, "more games = tighter intervals",
            "200 is usually enough to compare; 1000 for a number you want to quote."),
          field("Seed", seed, "keep this fixed to compare results",
            "Game i of an evaluation is always the same game for a given seed."),
          depthField,
        ], 170),
        el("div", { class: "inline inline-12" }, run,
          el("span", { class: "note" }, "Expectimax is slow — about 1–2 games/second."))));
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
    this.result.hidden = true;
    const pc = progressCard("Evaluating", job.label, {
      onCancel: () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { }),
    });
    this.progress.replaceChildren(pc.node);
    this.progress.hidden = false;

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      pc.update(p.total ? (p.done || 0) / p.total : 0, p.total
        ? `${F.n(p.done || 0)} / ${F.n(p.total)} games · ${F.dur(j.duration)}`
        : `${p.phase || j.state} · ${F.dur(j.duration)}`);
      if (j.state === "COMPLETED" && j.result) {
        this.progress.hidden = true;
        this.showResult(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.hidden = true;
        if (j.state === "FAILED") Toast.error("Evaluation failed", j.error);
      }
    });
  },

  showResult(res) {
    const exportJson = () => downloadFile(
      `evaluation-${res.label || "agent"}-${res.games}.json`,
      JSON.stringify(res, null, 2));
    const exportCsv = () => {
      const rows = [["metric", "value"],
        ["agent", res.label], ["games", res.games], ["seed", res.seed],
        ["integrity_valid", res.integrity?.valid],
        ["invalid_actions", res.integrity?.invalid_actions],
        ["truncated_games", res.integrity?.truncated_games],
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
    fillCard(this.result, {
      title: "Result", description: res.label || "",
      action: exportButtons(exportJson, exportCsv),
    }, evaluationResult(res));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/evaluations", { run: App.run, limit: 200 }); }
    catch (_) { return; }
    const rows = data.evaluations || [];
    const head = {
      title: `Evaluation History — Run “${App.run}”`,
      description: "Mean score on fixed seeded games over training.",
      action: badge(`${rows.length} recorded`, "outline"),
    };
    if (!rows.length) {
      fillCard(this.history, head, el("div", { class: "note" },
        "No evaluations recorded for this run yet. Results from this page and " +
        "from periodic evaluation during training both appear here."));
      return;
    }
    const x = rows.map((r) => r.games_trained || (r.agent || {}).games_trained || 0);
    const chart = chartView([{
      x, y: rows.map((r) => r.mean_score), color: SERIES_COLORS.eval, label: "Mean",
      points: true, bandColor: SERIES_COLORS.evalBand, bandLabel: "95% CI",
      band: [rows.map((r) => (r.ci95_mean || [0, 0])[0]),
             rows.map((r) => (r.ci95_mean || [0, 0])[1])],
    }], { height: 220, xUnit: "games", xLabel: "games trained",
          label: "Evaluation mean score over training" });

    const body = rows.slice().reverse().map((r) => {
      const ci = r.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", { class: "num" }, F.compact(r.games_trained || (r.agent || {}).games_trained || 0)),
        el("td", { class: "num" }, F.n(r.games)),
        el("td", { class: "num" }, F.n(r.mean_score, 0)),
        el("td", { class: "num muted" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(r.median_score, 0)),
        el("td", { class: "num" }, F.pct(_evalRate(r, "2048") * 100, 1)),
        el("td", { class: "num" }, F.n(r.highest_tile)),
        el("td", { class: "muted" }, F.date(r.timestamp)));
    });
    fillCard(this.history, head, el("div", { class: "stack" },
      chart,
      table([["Trained", { num: true }], ["N", { num: true }], ["Mean", { num: true }],
             ["95% CI", { num: true }], ["Median", { num: true }], ["2048", { num: true }],
             ["Best tile", { num: true }], "When"], body)));
  },

  onRunChange() { App.route(); },
};

function _evalRate(record, tile) {
  const e = (record.tile_rates || {})[tile];
  if (e && typeof e === "object") return e.rate || 0;
  return e || 0;
}
