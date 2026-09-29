/* Compare: several agents over the identical set of seeded games. */

const COMPARE_AGENTS = [
  { id: "random", label: "Random", note: "the performance floor" },
  { id: "heuristic", label: "Heuristic", note: "hand-written rules, no search" },
  { id: "expectimax", label: "Expectimax", note: "classic search — slow" },
  { id: "learned", label: "Learned AI", note: "your trained agent" },
];

App.views.compare = {
  title: "Compare",
  subtitle: "Head-to-head on identical games, with the uncertainty shown",

  mount(root, { actions }) {
    actions.append(runPicker());
    this.form = card();
    this.progress = el("div", { hidden: true });
    this.result = card();
    this.result.hidden = true;
    this.history = card();
    root.append(this.form, this.progress, this.result, this.history);
    this.buildForm();
    this.loadHistory();
  },

  unmount() { this._stopWatch && this._stopWatch(); },

  buildForm() {
    const boxes = {};
    const list = el("div", { class: "auto-grid", style: "--min:200px" });
    for (const a of COMPARE_AGENTS) {
      const cb = el("input", { type: "checkbox", checked: a.id !== "expectimax" });
      boxes[a.id] = cb;
      list.append(el("label", { class: "pick" },
        cb, el("span", { class: "pick-text" },
          el("span", { class: "pick-title" }, a.label),
          el("span", { class: "pick-note" }, a.note))));
    }
    const games = numberInput("c-games", 100, { min: 1, max: 50000, step: 25 });
    const seed = numberInput("c-seed", 987654, { min: 0 });

    const run = button("Run Comparison", { icon: "play" });
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

    fillCard(this.form, {
      title: "Agents", description: "Every agent plays the same games in the same order.",
    },
    el("div", { class: "stack" },
      list,
      formGrid([
        field("Games each", games, "100 is a good default",
          "Differences smaller than the confidence intervals are not " +
          "differences — raise this until the intervals separate."),
        field("Seed", seed, "fixed set of games"),
      ]),
      el("div", { class: "inline inline-12" }, run,
        el("span", { class: "note" },
          "Including expectimax makes this take minutes rather than seconds."))));
  },

  watchJob(job) {
    this.result.hidden = true;
    const pc = progressCard("Comparing", job.label, {
      onCancel: () => API.post(`/api/jobs/${job.id}/stop`, {}).catch(() => { }),
    });
    this.progress.replaceChildren(pc.node);
    this.progress.hidden = false;

    this._stopWatch && this._stopWatch();
    this._stopWatch = Jobs.watch(job.id, (j) => {
      const p = j.progress || {};
      pc.update(p.total ? (p.done || 0) / p.total : 0, p.agent
        ? `${p.agent} (${(p.agent_index ?? 0) + 1} of ${p.agent_count}) · ` +
          `${F.n(p.done || 0)} / ${F.n(p.total)} games · ${F.dur(j.duration)}`
        : `${p.phase || j.state} · ${F.dur(j.duration)}`);
      if (j.state === "COMPLETED" && j.result) {
        this.progress.hidden = true;
        this.showResult(j.result);
        this.loadHistory();
      } else if (["FAILED", "CANCELLED"].includes(j.state)) {
        this.progress.hidden = true;
        if (j.state === "FAILED") Toast.error("Comparison failed", j.error);
      }
    });
  },

  exports(payload) {
    const results = payload.results || {};
    const names = Object.keys(results);
    return exportButtons(
      () => downloadFile(`comparison-${payload.games}.json`, JSON.stringify(payload, null, 2)),
      () => {
        const head = ["agent", "games", "mean", "ci_low", "ci_high", "median",
                      "highest_tile", "rate_2048", "rate_4096", "games_per_second",
                      "ms_per_decision", "integrity_valid", "invalid_actions", "truncated_games"];
        const body = names.map((n) => {
          const r = results[n];
          const ci = r.ci95_mean || [0, 0];
          return [n, r.games, r.mean_score, ci[0], ci[1], r.median_score,
                  r.highest_tile, _rateOf(r, "2048"), _rateOf(r, "4096"),
                  r.games_per_second, r.ms_per_decision, r.integrity?.valid,
                  r.integrity?.invalid_actions, r.integrity?.truncated_games];
        });
        downloadFile(`comparison-${payload.games}.csv`, toCSV([head, ...body]), "text/csv");
      });
  },

  /* The table, the CI bars and the footnote for one comparison. */
  body(payload) {
    const results = payload.results || {};
    const names = Object.keys(results);
    if (!names.length) return el("div", { class: "note" }, "No results.");
    const colors = spreadColors(names.length);

    const rows = names.map((n, i) => {
      const r = results[n];
      const ci = r.ci95_mean || [0, 0];
      return el("tr", {},
        el("td", {}, el("span", { class: "inline", style: "flex-wrap:nowrap" },
          el("span", { class: "swatch", style: `background:${colors[i]}` }), n),
          el("div", { class: "note" }, !r.integrity ? "Legacy: integrity unknown" :
            r.integrity.valid ? "Integrity passed" :
            `Warning: ${r.integrity.invalid_actions} invalid actions, ${r.integrity.truncated_games} truncated games`)),
        el("td", { class: "num" }, F.n(r.mean_score, 0)),
        el("td", { class: "num muted" }, `${F.compact(ci[0])} – ${F.compact(ci[1])}`),
        el("td", { class: "num" }, F.n(r.median_score, 0)),
        el("td", { class: "num" }, F.n(r.highest_tile)),
        el("td", { class: "num" }, F.pct(_rateOf(r, "2048") * 100, 1)),
        el("td", { class: "num" }, F.pct(_rateOf(r, "4096") * 100, 1)),
        el("td", { class: "num" }, F.n(r.games_per_second, 2)),
        el("td", { class: "num" }, r.ms_per_decision ? `${r.ms_per_decision.toFixed(2)} ms` : "—"));
    });

    return el("div", { class: "stack" },
      table(["Agent", ["Mean", { num: true }], ["95% CI", { num: true }],
             ["Median", { num: true }], ["Best tile", { num: true }], ["2048", { num: true }],
             ["4096", { num: true }], ["Games/s", { num: true }], ["Per move", { num: true }]], rows),
      sectionHead("Mean Score With 95% Confidence Interval"),
      ciBars(names.map((n, i) => {
        const r = results[n];
        const ci = r.ci95_mean || [0, 0];
        return { name: n, mean: r.mean_score, lo: ci[0], hi: ci[1], color: colors[i] };
      })),
      el("div", { class: "footnote" },
        "The shaded band is the confidence interval and the line is the mean. " +
        "Agents whose bands overlap substantially have not been shown to differ."));
  },

  showResult(payload) {
    fillCard(this.result, {
      title: "Comparison",
      description: `${F.n(payload.games)} identical games each · seed ${payload.seed}`,
      action: this.exports(payload),
    }, this.body(payload));
  },

  async loadHistory() {
    let data;
    try { data = await API.get("/api/comparisons"); } catch (_) { return; }
    const items = data.comparisons || [];
    const head = {
      title: "Saved Comparisons",
      description: "Comparisons you run are saved here automatically.",
      action: badge(`${items.length} saved`, "outline"),
    };
    if (!items.length) {
      fillCard(this.history, head, el("div", { class: "note" }, "Nothing saved yet."));
      return;
    }
    const list = el("div", { class: "list" });
    for (const item of items.slice(0, 8)) {
      const names = Object.keys(item.results || {});
      const detail = el("div", { class: "list-detail", hidden: true });
      const open = button("Show", { variant: "outline", size: "sm" });
      open.setAttribute("aria-expanded", "false");
      open.onclick = () => {
        const show = detail.hidden;
        if (show && !detail.firstChild) {
          detail.append(el("div", { class: "inset stack stack-16" },
            el("div", { class: "inline" },
              el("span", { class: "note" }, `${F.n(item.games)} identical games each · seed ${item.seed}`),
              el("span", { class: "spacer" }), ...this.exports(item)),
            this.body(item)));
        }
        detail.hidden = !show;
        open.textContent = show ? "Hide" : "Show";
        open.setAttribute("aria-expanded", String(show));
      };
      list.append(
        el("div", { class: "list-row" },
          el("div", { class: "list-main" },
            el("div", { class: "list-title" }, names.join(" · ") || "comparison"),
            el("div", { class: "list-meta" },
              `${F.n(item.games)} games each · seed ${item.seed} · ${F.date(item.timestamp)}`)),
          open),
        detail);
    }
    fillCard(this.history, head, list);
  },

  onRunChange() { App.route(); },
};

function _rateOf(r, tile) {
  const e = (r.tile_rates || {})[tile];
  if (e && typeof e === "object") return e.rate || 0;
  return e || 0;
}
