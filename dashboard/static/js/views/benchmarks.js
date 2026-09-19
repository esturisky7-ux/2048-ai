/* Benchmarks: how fast this machine is. Not how good the AI is. */

App.views.benchmarks = {
  title: "Benchmarks",
  subtitle: "Throughput of this machine — a speed measurement, not a strength one",

  mount(root) {
    this.panel = el("section", { class: "panel" });
    this.progress = el("section", { class: "panel", style: "display:none" });
    this.result = el("section", { class: "panel", style: "display:none" });
    this.history = el("section", { class: "panel" });
    root.append(
      el("section", { class: "panel hero" },
        el("h2", {}, "Benchmark this machine"),
        el("p", {},
          "Measures raw engine throughput, how fast each agent can choose a " +
          "move, and how fast the training loop runs. These numbers describe " +
          "your CPU and Python build. They say nothing about how well any " +
          "agent plays — that is what the ",
          el("a", { href: "#/evaluate" }, "Evaluate"),
          " page is for."),
        this.controls()),
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
    const run = el("button", { class: "btn btn-primary btn-lg" }, "Run benchmark");
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
    return el("div", { class: "btn-row" },
      el("div", { class: "field", style: "max-width:200px" },
        el("label", {}, "Length"), scale), run);
  },

  watchJob(job) {
    this.progress.style.display = "";
    this.result.style.display = "none";
    const bar = el("div", { class: "bar blue" }, el("i", { style: "width:0%" }));
    const text = el("div", { class: "faint", style: "margin-top:8px;font-size:12.5px" }, "starting…");
    this.progress.innerHTML = "";
    this.progress.append(
      el("div", { class: "panel-head" }, el("h3", {}, "Benchmarking")), bar, text);
    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      const frac = p.total ? (p.done || 0) / p.total : 0;
      bar.firstChild.style.width = `${Math.max(4, frac * 100)}%`;
      text.textContent = `${p.phase || j.state} · ${F.dur(j.duration)}`;
      if (j.state === "COMPLETED" && j.result) {
        this.progress.style.display = "none";
        this.show(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.style.display = "none";
        if (j.state === "FAILED") Toast.error("Benchmark failed", j.error);
      }
    });
  },

  show(res, target) {
    const node = target || this.result;
    node.style.display = "";
    node.innerHTML = "";
    const m = res.machine || {};
    const fast = res.random_try_in_order || {};
    const slow = res.random_all_moves || {};
    const tr = res.training || {};

    const exportJson = el("button", { class: "btn btn-sm" }, "Export JSON");
    exportJson.onclick = () => downloadFile("benchmark.json", JSON.stringify(res, null, 2));

    node.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Benchmark result"),
        el("span", { class: "note" }, F.date(res.timestamp)),
        el("div", { class: "btn-row" }, exportJson)),
      el("div", { class: "stats" },
        statTile("Engine moves/sec", F.compact(fast.moves_per_second),
          "random play, try-in-order", "accent"),
        statTile("Random games/sec", F.n(fast.games_per_second, 0), "full games"),
        statTile("Engine moves/sec", F.compact(slow.moves_per_second),
          "random play, all legal moves"),
        statTile("Training moves/sec", tr.moves_per_second ? F.compact(tr.moves_per_second) : "—",
          tr.tuple_set ? `real TD updates, ${tr.tuple_set}` : (tr.error || ""))),

      el("div", { class: "panel-head", style: "margin-top:20px" },
        el("h3", {}, "Agent decision rate"),
        el("span", { class: "note" }, "how fast each agent chooses one move")),
      el("div", { class: "table-wrap" },
        el("table", { class: "data" },
          el("thead", {}, el("tr", {},
            el("th", {}, "Agent"), el("th", { class: "num" }, "Decisions/sec"),
            el("th", { class: "num" }, "Per decision"), el("th", { class: "num" }, "Sampled"))),
          el("tbody", {}, ...(res.agents || []).map((a) => el("tr", {},
            el("td", {}, a.agent),
            el("td", { class: "num" }, a.available ? F.n(a.decisions_per_second, 1) : "—"),
            el("td", { class: "num" }, a.available ? `${a.ms_per_decision.toFixed(3)} ms` : "—"),
            el("td", { class: "num faint" },
              a.available ? F.n(a.decisions) : (a.reason || "unavailable"))))))),

      el("div", { class: "panel-head", style: "margin-top:20px" },
        el("h3", {}, "Engine primitives")),
      el("div", { class: "table-wrap" },
        el("table", { class: "data" },
          el("thead", {}, el("tr", {},
            el("th", {}, "Operation"), el("th", { class: "num" }, "Per call"))),
          el("tbody", {}, ...(res.primitives || []).map((p) => el("tr", {},
            el("td", { class: "mono" }, p.name),
            el("td", { class: "num" }, `${p.microseconds.toFixed(3)} µs`)))))),

      el("div", { class: "panel-head", style: "margin-top:20px" },
        el("h3", {}, "Machine")),
      el("dl", { class: "kv" },
        el("dt", {}, "CPU"), el("dd", {}, m.cpu_model || "—"),
        el("dt", {}, "Cores"), el("dd", {}, F.n(m.cpu_count)),
        el("dt", {}, "RAM"), el("dd", {}, m.total_ram_bytes ? F.bytes(m.total_ram_bytes) : "not reported"),
        el("dt", {}, "OS"), el("dd", {}, `${m.platform || "?"} ${m.release || ""} (${m.machine || "?"})`),
        el("dt", {}, "Python"), el("dd", {}, m.python || "—"),
        el("dt", {}, "Measured"), el("dd", {}, F.date(res.timestamp))));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/benchmarks"); } catch (_) { return; }
    const items = data.benchmarks || [];
    this.history.innerHTML = "";
    this.history.append(el("div", { class: "panel-head" },
      el("h3", {}, "Previous benchmarks"),
      el("span", { class: "note" }, `${items.length} saved`)));
    if (!items.length) {
      this.history.append(el("div", { class: "faint" },
        "Benchmarks you run are saved here so you can see whether a change " +
        "made the engine faster or slower."));
      return;
    }
    const rows = items.map((b) => {
      const open = el("button", { class: "btn btn-sm" }, "Show");
      const detail = el("div", { style: "display:none" });
      open.onclick = () => {
        if (detail.style.display === "none") {
          detail.style.display = "";
          if (!detail.dataset.built) {
            const p = el("div"); this.show(b, p); detail.append(p);
            detail.dataset.built = "1";
          }
          open.textContent = "Hide";
        } else { detail.style.display = "none"; open.textContent = "Show"; }
      };
      return [el("div", { class: "job-card" },
        el("div", { class: "grow" },
          el("div", { class: "title" },
            `${F.compact((b.random_try_in_order || {}).moves_per_second)} moves/s`),
          el("div", { class: "meta" },
            `${(b.machine || {}).cpu_model || "?"} · Python ${(b.machine || {}).python || "?"} · ${F.date(b.timestamp)}`)),
        open), detail];
    });
    this.history.append(...rows.flat());
  },
};
