/* Benchmarks: how fast this machine is. Not how good the AI is. */

App.views.benchmarks = {
  title: "Benchmarks",
  subtitle: "Throughput of this machine — a speed measurement, not a strength one",

  mount(root) {
    this.progress = el("div", { hidden: true });
    this.result = card();
    this.result.hidden = true;
    this.history = card();
    root.append(
      card(cardHeader("Benchmark This Machine"),
        cardContent(el("div", { class: "stack" },
          prose(
            "Measures raw engine throughput, how fast each agent can choose a " +
            "move, and how fast the training loop runs. These numbers describe " +
            "your CPU and Python build. They say nothing about how well any " +
            "agent plays — that is what the ",
            el("a", { href: "#/evaluate" }, "Evaluate"),
            " page is for."),
          this.controls()))),
      this.progress, this.result, this.history);
    this.loadHistory();
  },

  unmount() { this._stopWatch && this._stopWatch(); },

  controls() {
    const scale = selectInput("b-scale", [
      { value: 0.3, label: "Quick (~15 s)" },
      { value: 1, label: "Standard (~45 s)" },
      { value: 2.5, label: "Thorough (~2 min)" },
    ], 1);
    const run = button("Run Benchmark", { icon: "gauge" });
    run.onclick = async () => {
      run.disabled = true;
      try {
        const r = await API.post("/api/benchmark",
          { scale: Number(scale.value), run: App.run });
        this.watchJob(r.job);
      } catch (e) {
        Toast.error("Could not start the benchmark", e.message);
      } finally { run.disabled = false; }
    };
    return el("div", { class: "inline inline-12", style: "align-items:flex-end" },
      el("div", { style: "width:220px;max-width:100%" }, field("Length", scale)), run);
  },

  watchJob(job) {
    this.result.hidden = true;
    const pc = progressCard("Benchmarking", job.label, {
      onCancel: () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { }),
    });
    this.progress.replaceChildren(pc.node);
    this.progress.hidden = false;
    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      const frac = p.total ? (p.done || 0) / p.total : 0;
      pc.update(Math.max(0.04, frac), `${p.phase || j.state} · ${F.dur(j.duration)}`);
      if (j.state === "COMPLETED" && j.result) {
        this.progress.hidden = true;
        this.show(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.hidden = true;
        if (j.state === "FAILED") Toast.error("Benchmark failed", j.error);
      }
    });
  },

  exportButton(res) {
    return exportButtons(() => downloadFile("benchmark.json", JSON.stringify(res, null, 2)));
  },

  /* Stat boxes, the agent table, primitives and the machine it ran on. */
  body(res) {
    const m = res.machine || {};
    const fast = res.random_try_in_order || {};
    const slow = res.random_all_moves || {};
    const tr = res.training || {};
    return el("div", { class: "stack" },
      stats([
        statTile("Engine moves/sec", F.compact(fast.moves_per_second), "random play, try-in-order"),
        statTile("Random games/sec", F.n(fast.games_per_second, 0), "full games"),
        statTile("Engine moves/sec", F.compact(slow.moves_per_second), "random play, all legal moves"),
        statTile("Training moves/sec", tr.moves_per_second ? F.compact(tr.moves_per_second) : "—",
          tr.tuple_set ? `real TD updates, ${tr.tuple_set}` : (tr.error || "")),
      ], { min: 180 }),
      sectionHead("Agent Decision Rate", "how fast each agent chooses one move"),
      table(["Agent", ["Decisions/sec", { num: true }], ["Per decision", { num: true }],
             ["Sampled", { num: true }]],
        (res.agents || []).map((a) => el("tr", {},
          el("td", {}, a.agent),
          el("td", { class: "num" }, a.available ? F.n(a.decisions_per_second, 1) : "—"),
          el("td", { class: "num" }, a.available ? `${a.ms_per_decision.toFixed(3)} ms` : "—"),
          el("td", { class: "num muted" },
            a.available ? F.n(a.decisions) : (a.reason || "unavailable"))))),
      el("div", { class: "auto-grid", style: "--min:320px;gap:24px" },
        el("div", { class: "stack stack-12" },
          sectionHead("Engine Primitives"),
          table(["Operation", ["Per call", { num: true }]],
            (res.primitives || []).map((p) => el("tr", {},
              el("td", { class: "mono", style: "font-size:13px" }, p.name),
              el("td", { class: "num" }, `${p.microseconds.toFixed(3)} µs`))))),
        el("div", { class: "stack stack-12" },
          sectionHead("Machine"),
          kv([
            ["CPU", m.cpu_model || "—"],
            ["Cores", F.n(m.cpu_count)],
            ["RAM", m.total_ram_bytes ? F.bytes(m.total_ram_bytes) : "not reported"],
            ["OS", `${m.platform || "?"} ${m.release || ""} (${m.machine || "?"})`],
            ["Python", m.python || "—"],
            ["Measured", F.date(res.timestamp)],
          ]))));
  },

  show(res) {
    fillCard(this.result, {
      title: "Benchmark Result", description: F.date(res.timestamp),
      action: this.exportButton(res),
    }, this.body(res));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/benchmarks"); } catch (_) { return; }
    const items = data.benchmarks || [];
    const head = {
      title: "Previous Benchmarks",
      description: "Saved so you can see whether a change made the engine faster or slower.",
      action: badge(`${items.length} saved`, "outline"),
    };
    if (!items.length) {
      fillCard(this.history, head, el("div", { class: "note" },
        "Benchmarks you run are saved here."));
      return;
    }
    const list = el("div", { class: "list" });
    for (const b of items) {
      const detail = el("div", { class: "list-detail", hidden: true });
      const open = button("Show", { variant: "outline", size: "sm" });
      open.setAttribute("aria-expanded", "false");
      open.onclick = () => {
        const show = detail.hidden;
        if (show && !detail.firstChild) {
          detail.append(el("div", { class: "inset stack stack-16" },
            el("div", { class: "inline" },
              el("span", { class: "note" }, F.date(b.timestamp)),
              el("span", { class: "spacer" }), ...this.exportButton(b)),
            this.body(b)));
        }
        detail.hidden = !show;
        open.textContent = show ? "Hide" : "Show";
        open.setAttribute("aria-expanded", String(show));
      };
      list.append(
        el("div", { class: "list-row" },
          el("div", { class: "list-main" },
            el("div", { class: "list-title" },
              `${F.compact((b.random_try_in_order || {}).moves_per_second)} moves/s`),
            el("div", { class: "list-meta" },
              `${(b.machine || {}).cpu_model || "?"} · Python ${(b.machine || {}).python || "?"} · ${F.date(b.timestamp)}`)),
          open),
        detail);
    }
    fillCard(this.history, head, list);
  },
};
