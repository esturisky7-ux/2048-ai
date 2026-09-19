/* Experiment Lab: run a shipped configuration, then compare results. */

App.views.experiments = {
  title: "Experiments",
  subtitle: "Change one thing, measure it, keep the config with the result",

  mount(root) {
    this.list = el("section", { class: "panel" });
    this.progress = el("section", { class: "panel", style: "display:none" });
    this.compare = el("section", { class: "panel" });
    root.append(
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "How this works")),
        el("div", { class: "dim", style: "font-size:13.5px;max-width:78ch" },
          "An experiment is a configuration file. Running one trains a fresh " +
          "agent with that configuration (or, for “agent” experiments, just " +
          "evaluates a configuration), then measures it with the fixed-seed " +
          "procedure and stores the result together with the exact config that " +
          "produced it — so a number is still interpretable months later. " +
          "Watch the sample size: two results whose confidence intervals " +
          "overlap have not been shown to differ.")),
      this.list, this.progress, this.compare);
    this.load();
    this._onDone = () => this.load();
    document.addEventListener("job:done", this._onDone);
  },

  unmount() {
    this._stopWatch && this._stopWatch();
    document.removeEventListener("job:done", this._onDone);
  },

  async load() {
    let data;
    try { data = await API.get("/api/experiments"); }
    catch (e) {
      this.list.innerHTML = "";
      this.list.append(el("div", { class: "faint" }, `Could not load: ${e.message}`));
      return;
    }
    this.paintList(data.experiments || []);
    this.paintResults(data.results || []);
  },

  paintList(items) {
    const groups = { train: [], agent: [] };
    for (const e of items) (groups[e.kind] || groups.train).push(e);

    this.list.innerHTML = "";
    this.list.append(el("div", { class: "panel-head" },
      el("h3", {}, "Available experiments"),
      el("span", { class: "note" }, `${items.length} shipped configurations`)));

    for (const [kind, label, blurb] of [
      ["train", "Training experiments", "trains a fresh agent, then evaluates it"],
      ["agent", "Agent experiments", "no training — evaluates a configuration"],
    ]) {
      const rows = groups[kind].map((e) => this.row(e, kind));
      if (!rows.length) continue;
      this.list.append(
        el("div", { class: "panel-head", style: "margin-top:18px" },
          el("h3", {}, label), el("span", { class: "note" }, blurb)),
        el("div", { class: "table-wrap" },
          el("table", { class: "data" },
            el("thead", {}, el("tr", {},
              el("th", {}, "Experiment"), el("th", {}, "What it tests"),
              el("th", { class: "num" }, "Result"), el("th", {}, ""))),
            el("tbody", {}, ...rows))));
    }
  },

  row(e, kind) {
    const games = numberInput("", kind === "train" ? 5000 : 0,
      { min: 0, step: 500, style: "width:92px", title: "training games (0 = config default)" });
    const evalGames = numberInput("", kind === "train" ? 200 : 60,
      { min: 1, step: 20, style: "width:78px", title: "evaluation games" });
    const run = el("button", { class: "btn btn-sm btn-primary" }, "Run");
    run.onclick = async () => {
      run.disabled = true;
      const payload = { name: e.name, eval_games: Number(evalGames.value) || undefined };
      if (kind === "train" && Number(games.value) > 0) payload.games = Number(games.value);
      try {
        const r = await API.post("/api/experiments/run", payload);
        Toast.ok("Experiment started", e.name);
        this.watchJob(r.job);
      } catch (err) {
        Toast.error("Could not start", err.message);
      } finally { run.disabled = false; }
    };
    return el("tr", {},
      el("td", {}, el("div", { style: "font-weight:600" }, e.name)),
      el("td", { class: "dim", style: "max-width:44ch;font-size:12.5px" }, e.description),
      el("td", { class: "num" }, e.has_result
        ? el("span", { class: "tag", style: "color:var(--ok)" }, "yes")
        : el("span", { class: "faint" }, "—")),
      el("td", {}, el("div", { class: "btn-row" },
        kind === "train" ? games : null, evalGames, run)));
  },

  watchJob(job) {
    this.progress.style.display = "";
    const bar = el("div", { class: "bar blue" }, el("i", { style: "width:4%" }));
    const text = el("div", { class: "faint", style: "margin-top:8px;font-size:12.5px" }, "starting…");
    const stop = el("button", { class: "btn btn-sm btn-danger" }, "Cancel");
    stop.onclick = () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { });
    this.progress.innerHTML = "";
    this.progress.append(
      el("div", { class: "panel-head" }, el("h3", {}, "Running"),
        el("span", { class: "note" }, job.label)),
      bar, text, el("div", { class: "btn-row", style: "margin-top:12px" }, stop));

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      text.textContent = `${p.phase || j.state} · ${F.dur(j.duration)}`;
      bar.firstChild.style.width = j.state === "RUNNING" ? "55%" : "100%";
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.style.display = "none";
        if (j.state === "FAILED") Toast.error("Experiment failed", j.error);
        this.load();
      }
    }, 1500);
  },

  paintResults(results) {
    this.compare.innerHTML = "";
    this.compare.append(el("div", { class: "panel-head" },
      el("h3", {}, "Results"),
      el("span", { class: "note" }, "sorted by evaluation mean")));
    const usable = results.filter((r) => (r.evaluation || {}).games);
    if (!usable.length) {
      this.compare.append(el("div", { class: "faint" },
        "No experiment results yet. Run one above — the results are stored in " +
        "data/experiments/ with the config that produced them."));
      return;
    }
    usable.sort((a, b) => (b.evaluation.mean_score || 0) - (a.evaluation.mean_score || 0));

    const maxHi = Math.max(...usable.map((r) => (r.evaluation.ci95_mean || [0, 0])[1]));
    const bars = el("div", {});
    for (const r of usable) {
      const ev = r.evaluation;
      const ci = ev.ci95_mean || [0, 0];
      const thin = ev.games < 30;
      bars.append(el("div", { style: "margin-bottom:12px" },
        el("div", { style: "display:flex;justify-content:space-between;gap:10px;font-size:12.5px;margin-bottom:4px" },
          el("span", {}, r.experiment,
            thin ? el("span", { class: "tag", style: "margin-left:7px;color:var(--warn)",
              title: "very few evaluation games — the interval is wide for a reason" },
              `n=${ev.games}`) : el("span", { class: "faint", style: "margin-left:7px" }, `n=${ev.games}`)),
          el("span", { class: "faint mono" },
            `${F.n(ev.mean_score, 0)}  [${F.compact(ci[0])} – ${F.compact(ci[1])}]`)),
        el("div", { style: "position:relative;height:15px;background:var(--panel-3);border-radius:4px" },
          el("div", { style:
            `position:absolute;left:${(ci[0] / maxHi) * 100}%;width:${((ci[1] - ci[0]) / maxHi) * 100}%;` +
            "top:0;bottom:0;background:var(--accent);opacity:.26;border-radius:4px" }),
          el("div", { style:
            `position:absolute;left:calc(${(ev.mean_score / maxHi) * 100}% - 1px);top:-2px;bottom:-2px;` +
            "width:2px;background:var(--accent)" }))));
    }

    const rows = usable.map((r) => {
      const ev = r.evaluation;
      const ci = ev.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", {}, r.experiment),
        el("td", { class: "num" }, F.compact(r.games_trained || 0)),
        el("td", { class: "num" }, F.n(ev.mean_score, 0)),
        el("td", { class: "num faint nowrap" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(ev.median_score, 0)),
        el("td", { class: "num" }, F.n(ev.highest_tile)),
        el("td", { class: "num" }, F.n(ev.games)));
    });

    this.compare.append(
      bars,
      el("div", { class: "table-wrap", style: "margin-top:16px" },
        el("table", { class: "data" },
          el("thead", {}, el("tr", {},
            el("th", {}, "Experiment"), el("th", { class: "num" }, "Trained"),
            el("th", { class: "num" }, "Eval mean"), el("th", { class: "num" }, "95% CI"),
            el("th", { class: "num" }, "Median"), el("th", { class: "num" }, "Best tile"),
            el("th", { class: "num" }, "n"))),
          el("tbody", {}, ...rows))),
      el("div", { class: "faint", style: "margin-top:12px;font-size:12.5px" },
        "Every result file also contains the fully resolved configuration that " +
        "produced it, so an experiment stays reproducible."));
  },
};
