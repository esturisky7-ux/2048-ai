/* Shared UI pieces: the board renderer, stat tiles, rate bars, run picker.
 *
 * The board renderer is used by three different places (watch the AI, play
 * yourself, human-vs-AI), so it lives here rather than being written three
 * times with three sets of bugs.
 */

/* ------------------------------------------------------------------ board */
/* Tiles are absolutely positioned so they can be transitioned between cells;
 * the grid underneath is just the background. Values come from the server. */
class Board {
  constructor(node) {
    this.node = node;
    this.node.classList.add("board");
    this.tiles = new Map();          // cell index -> {el, value}
    this._grid();
  }

  _grid() {
    this.node.innerHTML = "";
    for (let i = 0; i < 16; i++) this.node.append(el("div", { class: "cell" }));
  }

  /* rows: 4x4 of face values (0 = empty) */
  render(rows, { animate = true } = {}) {
    if (!rows) { this.clear(); return; }
    const flat = rows.flat();
    const present = new Set();

    for (let i = 0; i < 16; i++) {
      const v = flat[i];
      if (!v) continue;
      present.add(i);
      let entry = this.tiles.get(i);
      if (!entry) {
        const tile = el("div", { class: `tile ${F.tileClass(v)}` });
        this.node.append(tile);
        entry = { el: tile, value: null };
        this.tiles.set(i, entry);
        if (animate) tile.classList.add("pop");
        setTimeout(() => tile.classList.remove("pop"), 180);
      }
      if (entry.value !== v) {
        entry.el.className = `tile ${F.tileClass(v)}`;
        entry.el.textContent = v;
        entry.value = v;
        if (animate && entry.value !== null) {
          entry.el.classList.add("pop");
          setTimeout(() => entry.el.classList.remove("pop"), 180);
        }
      }
      this._place(entry.el, i, v);
    }
    for (const [i, entry] of [...this.tiles]) {
      if (!present.has(i)) { entry.el.remove(); this.tiles.delete(i); }
    }
  }

  _place(tile, index, value) {
    const r = Math.floor(index / 4), c = index % 4;
    // Percentages keep the board fully responsive without measuring pixels.
    const gap = 2.6, size = (100 - gap * 5) / 4;
    tile.style.width = `${size}%`;
    tile.style.height = `${size}%`;
    tile.style.left = `${gap + c * (size + gap)}%`;
    tile.style.top = `${gap + r * (size + gap)}%`;
    const digits = String(value).length;
    tile.style.fontSize = `clamp(11px, ${digits >= 5 ? 4.6 : digits >= 4 ? 5.6 : 7}vmin, ${digits >= 5 ? 22 : digits >= 4 ? 27 : 34}px)`;
  }

  clear() {
    for (const [, entry] of this.tiles) entry.el.remove();
    this.tiles.clear();
  }
}

/* ------------------------------------------------------------ small parts */
function statTile(label, value, sub, cls = "") {
  return el("div", { class: `stat ${cls}` },
    el("label", {}, label),
    el("b", {}, value),
    sub ? el("small", {}, sub) : null);
}

function boardStats(pairs) {
  return el("div", { class: "board-stats" },
    ...pairs.map(([l, v]) => el("div", {},
      el("label", {}, l), el("b", {}, v))));
}

const MILESTONES = [512, 1024, 2048, 4096, 8192, 16384];

/* rates: {"512": 0.98, ...} as fractions; counts optional */
function rateBars(rates, counts) {
  const wrap = el("div", {});
  let shown = 0;
  for (const m of MILESTONES) {
    const r = Number((rates || {})[String(m)] || 0);
    // Always show the core four; show larger tiles only once they happen.
    if (m >= 8192 && r <= 0 && shown >= 4) continue;
    shown++;
    const pctv = r * 100;
    wrap.append(el("div", { class: "rate-row" },
      el("span", { class: `tile-chip ${F.tileClass(m)}` }, F.compact(m)),
      el("div", { class: "bar" }, el("i", { style: `width:${Math.min(100, pctv)}%` })),
      el("span", { class: "pct" }, F.pct(pctv, pctv >= 10 ? 0 : 1))));
  }
  if (counts) {
    wrap.append(el("div", { class: "faint", style: "font-size:11px;margin-top:6px" },
      "share of all games this run has played"));
  }
  return wrap;
}

function tooltip(text) {
  if (!App.settings.show_tooltips) return null;
  return el("span", { class: "tip", title: text, "aria-label": text }, "?");
}

function emptyState(icon, title, message, action) {
  return el("div", { class: "panel empty" },
    el("div", { class: "big" }, icon),
    el("h3", {}, title),
    el("div", {}, message),
    action ? el("div", { class: "btn-row", style: "justify-content:center;margin-top:16px" }, action) : null);
}

function field(label, control, hint, tip) {
  return el("div", { class: "field" },
    el("label", {}, label, tip ? tooltip(tip) : null),
    control,
    hint ? el("div", { class: "hint" }, hint) : null);
}

function numberInput(id, value, attrs = {}) {
  return el("input", { type: "number", id, value, ...attrs });
}

function selectInput(id, options, value) {
  const sel = el("select", { id });
  for (const o of options) {
    const opt = typeof o === "string" ? { value: o, label: o } : o;
    sel.append(el("option", { value: opt.value, selected: String(opt.value) === String(value) }, opt.label));
  }
  return sel;
}

/* Run picker used in the top bar. */
function runPicker() {
  const sel = el("select", { id: "run-picker", "aria-label": "Training run" });
  const paint = (status) => {
    const runs = status?.runs || [];
    const names = runs.map((r) => r.name);
    if (!names.includes(App.run)) names.unshift(App.run);
    const same = sel.options.length === names.length &&
      [...sel.options].every((o, i) => o.value === names[i]);
    if (!same) {
      sel.innerHTML = "";
      for (const n of names) sel.append(el("option", { value: n }, n));
    }
    sel.value = App.run;
  };
  sel.onchange = () => App.setRun(sel.value);
  App.onStatus(paint);
  if (App.status) paint(App.status);
  return sel;
}

/* A compact card describing a running job, with a stop button. */
function jobCard(job, { onStop } = {}) {
  const p = job.progress || {};
  const total = p.total || 0;
  const frac = total ? Math.min(1, (p.done || 0) / total) : 0;
  const stop = el("button", { class: "btn btn-sm btn-danger" }, "Stop");
  stop.onclick = async () => {
    stop.disabled = true;
    try {
      await API.post(`/api/jobs/${encodeURIComponent(job.id)}/stop`, {});
      Toast.show("Stopping", job.label, "warn");
      onStop && onStop();
    } catch (e) { Toast.error("Could not stop", e.message); stop.disabled = false; }
  };
  return el("div", { class: "job-card" },
    el("span", { class: `pill ${job.state === "RUNNING" ? "busy" : "warn"}` },
      el("i"), el("span", {}, job.state)),
    el("div", { class: "grow" },
      el("div", { class: "title" }, job.label),
      el("div", { class: "meta" },
        total ? `${F.n(p.done || 0)} / ${F.n(total)} · ` : "",
        p.phase ? p.phase + " · " : "",
        F.dur(job.duration)),
      total ? el("div", { class: "bar slim blue", style: "margin-top:6px" },
        el("i", { style: `width:${frac * 100}%` })) : null),
    ["RUNNING", "QUEUED"].includes(job.state) ? stop : null);
}

/* Renders an evaluation result block (shared by Evaluate and Checkpoints). */
function evaluationResult(res) {
  if (!res || !res.games) return el("div", { class: "faint" }, "no games played");
  const ci = res.ci95_mean || [0, 0];
  const rate = (t) => {
    const e = (res.tile_rates || {})[t];
    if (!e) return null;
    return typeof e === "object" ? e : { rate: e, ci95: null, count: null };
  };
  const stats = el("div", { class: "stats" },
    statTile("Mean score", F.n(res.mean_score, 0),
      `95% CI ${F.compact(ci[0])} – ${F.compact(ci[1])}`, "accent"),
    statTile("Median", F.n(res.median_score, 0)),
    statTile("Std dev", F.n(res.std_score, 0)),
    statTile("Best / worst", `${F.compact(res.max_score)} / ${F.compact(res.min_score)}`),
    statTile("Mean moves", F.n(res.mean_moves, 0), `longest ${F.n(res.max_moves)}`),
    statTile("Highest tile", F.n(res.highest_tile), `median ${F.n(res.median_tile)}`));

  const rows = [];
  for (const m of [512, 1024, 2048, 4096, 8192, 16384]) {
    const r = rate(String(m));
    if (!r) continue;
    if (m >= 8192 && !r.count) continue;
    rows.push(el("tr", {},
      el("td", {}, el("span", { class: `tag ${F.tileClass(m)}` }, F.n(m))),
      el("td", { class: "num" }, F.pct(r.rate * 100, 2)),
      el("td", { class: "num faint" },
        r.ci95 ? `${F.pct(r.ci95[0] * 100, 2)} – ${F.pct(r.ci95[1] * 100, 2)}` : "—"),
      el("td", { class: "num faint" }, r.count !== null && r.count !== undefined ? F.n(r.count) : "—")));
  }

  return el("div", {},
    stats,
    el("div", { class: "panel-head", style: "margin-top:18px" },
      el("h3", {}, "Tile achievement"),
      el("span", { class: "note" }, "Wilson score intervals — correct near 0% and 100%")),
    el("div", { class: "table-wrap" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {},
          el("th", {}, "Tile"), el("th", { class: "num" }, "Rate"),
          el("th", { class: "num" }, "95% CI"), el("th", { class: "num" }, "Games"))),
        el("tbody", {}, ...rows))),
    el("div", { class: "faint", style: "margin-top:12px;font-size:12px" },
      `${F.n(res.games)} games · seed ${res.seed} · `,
      `${F.n(res.elapsed_seconds, 1)}s (${F.n(res.games_per_second, 2)} games/s)`));
}

/* Download helper used by the CSV/JSON export buttons. */
function downloadFile(name, text, type = "application/json") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toCSV(rows) {
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(esc).join(",")).join("\n");
}
