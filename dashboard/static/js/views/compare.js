/* Compare: several agents over the identical set of seeded games. */

const COMPARE_AGENTS = [
  { id: "random", label: "Random", note: "the performance floor" },
  { id: "heuristic", label: "Heuristic", note: "hand-written rules, no search" },
  { id: "expectimax", label: "Expectimax", note: "classic search — slow" },
  { id: "learned", label: "Learned AI", note: "your trained agent" },
];
const BAR_COLORS = ["#9aa3b2", "#5ac8fa", "#ff9f43", "#edc22e", "#a78bfa", "#3ddc84"];

App.views.compare = {
  title: "Compare",
  subtitle: "Head-to-head on identical games, with the uncertainty shown",

  mount(root, { actions }) {
    actions.append(runPicker());
    this.form = el("section", { class: "panel" });
    this.progress = el("section", { class: "panel", style: "display:none" });
    this.result = el("section", { class: "panel", style: "display:none" });
    this.history = el("section", { class: "panel" });
    root.append(this.form, this.progress, this.result, this.history);
    this.buildForm();
    this.loadHistory();
  },

  unmount() { this._stopWatch && this._stopWatch(); },

  buildForm() {
    const boxes = {};
    const list = el("div", { class: "form-grid" });
    for (const a of COMPARE_AGENTS) {
      const cb = el("input", { type: "checkbox", checked: a.id !== "expectimax" });
      boxes[a.id] = cb;
      list.append(el("label", { class: "check", style: "align-items:flex-start" },
        cb, el("span", {},
          el("div", {}, a.label),
          el("div", { class: "faint", style: "font-size:11.5px" }, a.note))));
    }
    const games = numberInput("c-games", 100, { min: 1, max: 50000, step: 25 });
    const seed = numberInput("c-seed", 987654, { min: 0 });

    const run = el("button", { class: "btn btn-primary btn-lg" }, "Run comparison");
    run.onclick = async () => {
      const agents = COMPARE_AGENTS.filter((a) => boxes[a.id].checked)
        .map((a) => {
          const spec = { agent: a.id };
          if (a.id === "learned") { spec.run = App.run; spec.depth = 1; }
          if (a.id === "expectimax") spec.depth = 2;
          return spec;
        });
      if (!agents.length) { Toast.warn("Pick at least one agent"); return; }
      run.disabled = true;
      try {
        const r = await API.post("/api/compare", {
          agents, games: Number(games.value) || 100, seed: Number(seed.value) || 0,
        });
        this.watchJob(r.job);
      } catch (e) {
        Toast.error("Could not start the comparison", e.message);
      } finally { run.disabled = false; }
    };

    this.form.innerHTML = "";
    this.form.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Agents"),
        el("span", { class: "note" },
          "every agent plays the same games in the same order")),
      list,
      el("div", { class: "form-grid", style: "margin-top:16px" },
        field("Games each", games, "100 is a good default",
          "Differences smaller than the confidence intervals are not " +
          "differences — raise this until the intervals separate."),
        field("Seed", seed, "fixed set of games")),
      el("div", { class: "btn-row", style: "margin-top:18px" }, run,
        el("span", { class: "faint", style: "font-size:12px" },
          "including expectimax makes this take minutes rather than seconds")));
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
      el("div", { class: "panel-head" }, el("h3", {}, "Comparing"),
        el("span", { class: "note" }, job.label)),
      bar, text, el("div", { class: "btn-row", style: "margin-top:12px" }, stop));

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      const frac = p.total ? (p.done || 0) / p.total : 0;
      bar.firstChild.style.width = `${frac * 100}%`;
      text.textContent = p.agent
        ? `${p.agent} (${(p.agent_index ?? 0) + 1} of ${p.agent_count}) · ` +
          `${F.n(p.done || 0)} / ${F.n(p.total)} games · ${F.dur(j.duration)}`
        : `${p.phase || j.state} · ${F.dur(j.duration)}`;
      if (j.state === "COMPLETED" && j.result) {
        this.progress.style.display = "none";
        this.showResult(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.style.display = "none";
        if (j.state === "FAILED") Toast.error("Comparison failed", j.error);
      }
    });
  },

  showResult(payload, target) {
    const node = target || this.result;
    node.style.display = "";
    node.innerHTML = "";
    const results = payload.results || {};
    const names = Object.keys(results);
    if (!names.length) { node.append(el("div", { class: "faint" }, "no results")); return; }

    const rows = names.map((n, i) => {
      const r = results[n];
      const ci = r.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", {},
          el("span", { class: "sw", style:
            `display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px;background:${BAR_COLORS[i % BAR_COLORS.length]}` }),
          n),
        el("td", { class: "num" }, F.n(r.mean_score, 0)),
        el("td", { class: "num faint nowrap" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(r.median_score, 0)),
        el("td", { class: "num" }, F.n(r.highest_tile)),
        el("td", { class: "num" }, F.pct(_rateOf(r, "2048") * 100, 1)),
        el("td", { class: "num" }, F.pct(_rateOf(r, "4096") * 100, 1)),
        el("td", { class: "num" }, F.n(r.games_per_second, 2)),
        el("td", { class: "num" }, r.ms_per_decision ? `${r.ms_per_decision.toFixed(2)} ms` : "—"));
    });

    // Mean score with the confidence interval drawn, so overlap is visible.
    const maxHi = Math.max(...names.map((n) => (results[n].ci95_mean || [0, 0])[1]));
    const bars = el("div", {});
    names.forEach((n, i) => {
      const r = results[n];
      const ci = r.ci95_mean || [0, 0];
      const color = BAR_COLORS[i % BAR_COLORS.length];
      bars.append(el("div", { style: "margin-bottom:12px" },
        el("div", { style: "display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:4px" },
          el("span", {}, n),
          el("span", { class: "faint mono" },
            `${F.n(r.mean_score, 0)}  [${F.compact(ci[0])} – ${F.compact(ci[1])}]`)),
        el("div", { style: "position:relative;height:16px;background:var(--panel-3);border-radius:4px" },
          el("div", { style:
            `position:absolute;left:${(ci[0] / maxHi) * 100}%;width:${((ci[1] - ci[0]) / maxHi) * 100}%;` +
            `top:0;bottom:0;background:${color};opacity:.28;border-radius:4px` }),
          el("div", { style:
            `position:absolute;left:calc(${(r.mean_score / maxHi) * 100}% - 1px);top:-2px;bottom:-2px;` +
            `width:2px;background:${color}` }))));
    });

    const exportJson = el("button", { class: "btn btn-sm" }, "Export JSON");
    exportJson.onclick = () => downloadFile(`comparison-${payload.games}.json`,
      JSON.stringify(payload, null, 2));
    const exportCsv = el("button", { class: "btn btn-sm" }, "Export CSV");
    exportCsv.onclick = () => {
      const head = ["agent", "games", "mean", "ci_low", "ci_high", "median",
                    "highest_tile", "rate_2048", "rate_4096", "games_per_second",
                    "ms_per_decision"];
      const body = names.map((n) => {
        const r = results[n];
        const ci = r.ci95_mean || [0, 0];
        return [n, r.games, r.mean_score, ci[0], ci[1], r.median_score,
                r.highest_tile, _rateOf(r, "2048"), _rateOf(r, "4096"),
                r.games_per_second, r.ms_per_decision];
      });
      downloadFile(`comparison-${payload.games}.csv`, toCSV([head, ...body]), "text/csv");
    };

    node.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Comparison"),
        el("span", { class: "note" },
          `${F.n(payload.games)} identical games each · seed ${payload.seed}`),
        el("div", { class: "btn-row" }, exportJson, exportCsv)),
      el("div", { class: "table-wrap" },
        el("table", { class: "data" },
          el("thead", {}, el("tr", {},
            el("th", {}, "Agent"), el("th", { class: "num" }, "Mean"),
            el("th", { class: "num" }, "95% CI"), el("th", { class: "num" }, "Median"),
            el("th", { class: "num" }, "Best tile"), el("th", { class: "num" }, "2048"),
            el("th", { class: "num" }, "4096"), el("th", { class: "num" }, "Games/s"),
            el("th", { class: "num" }, "Per move"))),
          el("tbody", {}, ...rows))),
      el("div", { class: "panel-head", style: "margin-top:20px" },
        el("h3", {}, "Mean score with 95% confidence interval")),
      bars,
      el("div", { class: "faint", style: "font-size:12.5px;margin-top:6px" },
        "The shaded band is the confidence interval and the line is the mean. " +
        "Agents whose bands overlap substantially have not been shown to differ."));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/comparisons"); } catch (_) { return; }
    const items = data.comparisons || [];
    this.history.innerHTML = "";
    this.history.append(el("div", { class: "panel-head" },
      el("h3", {}, "Saved comparisons"),
      el("span", { class: "note" }, `${items.length} saved`)));
    if (!items.length) {
      this.history.append(el("div", { class: "faint" },
        "Comparisons you run are saved here automatically."));
      return;
    }
    for (const item of items.slice(0, 8)) {
      const names = Object.keys(item.results || {});
      const open = el("button", { class: "btn btn-sm" }, "Show");
      const detail = el("div", { style: "display:none" });
      open.onclick = () => {
        if (detail.style.display === "none") {
          detail.style.display = "";
          if (!detail.dataset.built) {
            const panel = el("div");
            this.showResult(item, panel);
            detail.append(panel);
            detail.dataset.built = "1";
          }
          open.textContent = "Hide";
        } else { detail.style.display = "none"; open.textContent = "Show"; }
      };
      this.history.append(
        el("div", { class: "job-card" },
          el("div", { class: "grow" },
            el("div", { class: "title" }, names.join(" · ") || "comparison"),
            el("div", { class: "meta" },
              `${F.n(item.games)} games each · seed ${item.seed} · ${F.date(item.timestamp)}`)),
          open),
        detail);
    }
  },

  onRunChange() { App.route(); },
};

function _rateOf(r, tile) {
  const e = (r.tile_rates || {})[tile];
  if (e && typeof e === "object") return e.rate || 0;
  return e || 0;
}
