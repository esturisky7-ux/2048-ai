/* Experiment Lab: run a shipped configuration, then compare results. */

App.views.experiments = {
  title: "Experiments",
  subtitle: "Change one thing, measure it, keep the config with the result",

  mount(root) {
    this.list = card();
    this.progress = el("div", { hidden: true });
    this.results = card();
    root.append(
      card(cardHeader("How This Works"),
        cardContent(prose(
          "An experiment is a configuration file. Running one trains a fresh " +
          "agent with that configuration (or, for “agent” experiments, just " +
          "evaluates a configuration), then measures it with the fixed-seed " +
          "procedure and stores the result together with the exact config that " +
          "produced it — so a number is still interpretable months later. " +
          "Watch the sample size: two results whose confidence intervals " +
          "overlap have not been shown to differ."))),
      this.list, this.progress, this.results);
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
      fillCard(this.list, { title: "Available Experiments" },
        el("div", { class: "note" }, `Could not load: ${e.message}`));
      return;
    }
    this.paintList(data.experiments || []);
    this.paintResults(data.results || []);
  },

  paintList(items) {
    const groups = { train: [], agent: [] };
    for (const e of items) (groups[e.kind] || groups.train).push(e);

    const sections = [];
    for (const [kind, label, blurb] of [
      ["train", "Training experiments", "trains a fresh agent, then evaluates it"],
      ["agent", "Agent experiments", "no training — evaluates a configuration"],
    ]) {
      const rows = groups[kind].map((e) => this.row(e, kind));
      if (!rows.length) continue;
      sections.push(el("div", { class: "stack stack-12" },
        sectionHead(label, blurb),
        table(["Experiment", "What it tests", "Result", ""], rows)));
    }
    fillCard(this.list, {
      title: "Available Experiments",
      description: `${items.length} shipped configurations.`,
    }, el("div", { class: "stack stack-24" }, ...sections));
  },

  row(e, kind) {
    const games = numberInput("", kind === "train" ? 5000 : 0,
      { min: 0, step: 500, style: "width:84px", title: "training games (0 = config default)",
        "aria-label": `${e.name}: training games` });
    const evalGames = numberInput("", kind === "train" ? 200 : 60,
      { min: 1, step: 20, style: "width:72px", title: "evaluation games",
        "aria-label": `${e.name}: evaluation games` });
    const run = button("Run", { size: "sm" });
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
      el("td", { style: "font-weight:500" }, e.name),
      el("td", { class: "desc" }, e.description),
      el("td", {}, e.has_result ? badge("Yes", "secondary") : el("span", { class: "muted" }, "—")),
      el("td", {}, el("div", { class: "inline inline-6", style: "justify-content:flex-end;flex-wrap:nowrap" },
        kind === "train" ? games : null, evalGames, run)));
  },

  watchJob(job) {
    const pc = progressCard("Running", job.label, {
      onCancel: () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { }),
    });
    pc.update(0.04, "starting…");
    this.progress.replaceChildren(pc.node);
    this.progress.hidden = false;

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      // Experiments report phases, not counts, so the bar only says "busy".
      pc.update(j.state === "RUNNING" ? 0.55 : 1, `${p.phase || j.state} · ${F.dur(j.duration)}`);
      if (["COMPLETED", "FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.hidden = true;
        if (j.state === "FAILED") Toast.error("Experiment failed", j.error);
        this.load();
      }
    }, 1500);
  },

  paintResults(results) {
    const head = { title: "Results", description: "Sorted by evaluation mean." };
    const usable = results.filter((r) => (r.evaluation || {}).games);
    if (!usable.length) {
      fillCard(this.results, head, el("div", { class: "note" },
        "No experiment results yet. Run one above — the results are stored in " +
        "data/experiments/ with the config that produced them."));
      return;
    }
    usable.sort((a, b) => (b.evaluation.mean_score || 0) - (a.evaluation.mean_score || 0));

    const bars = ciBars(usable.map((r) => {
      const ev = r.evaluation;
      const ci = ev.ci95_mean || [0, 0];
      const thin = ev.games < 30;
      return {
        name: r.experiment, mean: ev.mean_score, lo: ci[0], hi: ci[1],
        color: "var(--foreground)",
        badge: thin
          ? el("span", { title: "very few evaluation games — the interval is wide for a reason",
                         style: "display:inline-flex" }, badge(`n=${ev.games}`, "destructive"))
          : el("span", { class: "section-note" }, `n=${ev.games}`),
      };
    }));

    const rows = usable.map((r) => {
      const ev = r.evaluation;
      const ci = ev.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", {}, r.experiment),
        el("td", { class: "num" }, r.games_trained ? F.compact(r.games_trained) : "—"),
        el("td", { class: "num" }, F.n(ev.mean_score, 0)),
        el("td", { class: "num muted" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(ev.median_score, 0)),
        el("td", { class: "num" }, F.n(ev.highest_tile)),
        el("td", { class: "num" }, F.n(ev.games)));
    });

    fillCard(this.results, head, el("div", { class: "stack" },
      bars,
      table(["Experiment", ["Trained", { num: true }], ["Eval mean", { num: true }],
             ["95% CI", { num: true }], ["Median", { num: true }], ["Best tile", { num: true }],
             ["n", { num: true }]], rows),
      el("div", { class: "footnote" },
        "Every result file also contains the fully resolved configuration that " +
        "produced it, so an experiment stays reproducible.")));
  },
};
