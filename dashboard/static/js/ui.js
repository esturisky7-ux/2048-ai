/* Shared UI pieces: cards, stat boxes, buttons, badges, form fields, tabs,
 * the 2048 board, confidence-interval bars and the run picker.
 *
 * Every view builds its page out of these, so the look lives in one place
 * (with app.css) instead of being re-typed per page. The board renderer is
 * used by three different places (watch the AI, play yourself, human-vs-AI),
 * so it lives here rather than being written three times with three sets of
 * bugs.
 */

/* ------------------------------------------------------------------ board */
/* The grid underneath is only the background; tiles are absolutely placed in
 * percentages of the board so they stay aligned at any size. Tile text is
 * sized in container-query units of the frame, so it scales with the board.
 * Values come from the server. */
class Board {
  constructor(node, { max = 440 } = {}) {
    this.node = node;                  // the frame: sized, and a container
    this.node.classList.add("board-frame");
    this.node.style.setProperty("--board-max", `${max}px`);
    this.grid = el("div", { class: "board", role: "img", "aria-label": "2048 board" });
    this.node.replaceChildren(this.grid);
    this.tiles = new Map();          // cell index -> {el, value}
    for (let i = 0; i < 16; i++) this.grid.append(el("div", { class: "cell" }));
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
        const tile = el("div", { class: "tile" });
        this.grid.append(tile);
        entry = { el: tile, value: null };
        this.tiles.set(i, entry);
        this._place(tile, i);
      }
      if (entry.value !== v) {
        const digits = Math.min(5, String(v).length);
        entry.el.className = `tile ${F.tileClass(v)} d${digits}`;
        entry.el.textContent = v;
        entry.value = v;
        if (animate) {
          entry.el.classList.add("pop");
          setTimeout(() => entry.el.classList.remove("pop"), 180);
        }
      }
    }
    for (const [i, entry] of [...this.tiles]) {
      if (!present.has(i)) { entry.el.remove(); this.tiles.delete(i); }
    }
    this.grid.setAttribute("aria-label",
      `2048 board, highest tile ${Math.max(0, ...flat) || "none"}`);
  }

  _place(tile, index) {
    const r = Math.floor(index / 4), c = index % 4;
    // Padding and gap are both 2.6% of the board; a cell is the rest / 4.
    const gap = 2.6, size = (100 - gap * 5) / 4;
    tile.style.left = `${gap + c * (size + gap)}%`;
    tile.style.top = `${gap + r * (size + gap)}%`;
  }

  clear() {
    for (const [, entry] of this.tiles) entry.el.remove();
    this.tiles.clear();
  }
}

/* ----------------------------------------------------------------- cards */
function card(...children) {
  return el("section", { class: "card" }, ...children);
}

/* Title, optional description, optional action slot top-right. */
function cardHeader(title, description, action) {
  return el("div", { class: "card-header" },
    el("div", { class: "card-heading" },
      el("div", { class: "card-title" }, title),
      description ? el("div", { class: "card-description" }, description) : null),
    action ? el("div", { class: "card-action" }, action) : null);
}

function cardContent(...children) {
  return el("div", { class: "card-content" }, ...children);
}

/* Fill an existing card: header plus content in one go. */
function fillCard(node, { title, description, action } = {}, ...content) {
  node.replaceChildren(
    title ? cardHeader(title, description, action) : null,
    cardContent(...content));
  node.hidden = false;
  return node;
}

/* ---------------------------------------------------------- small parts */
function statTile(label, value, sub) {
  return el("div", { class: "stat" },
    el("div", { class: "stat-label" }, label),
    el("div", { class: "stat-value", title: typeof value === "string" ? value : null }, value),
    el("div", { class: "stat-sub" }, sub || ""));
}

function stats(tiles, { cols2 = false, min } = {}) {
  return el("div", { class: `stats${cols2 ? " cols-2" : ""}`,
                     style: min ? `--min:${min}px` : null }, ...tiles);
}

/* The four little boxes under a game board. */
function miniStats(pairs, max = 440) {
  return el("div", { class: "mini-stats",
                     style: `--n:${pairs.length};--board-max:${max}px` },
    ...pairs.map(([label, value]) => el("div", { class: "mini-stat" },
      el("div", { class: "mini-stat-label" }, label),
      el("div", { class: "mini-stat-value" }, value))));
}

/* A button. variant: default | outline | ghost | destructive;
 * size: default | sm | lg | icon. */
function button(label, { variant = "default", size = "default", icon: iconName,
                         title, onClick, disabled, id, type = "button", ariaLabel } = {}) {
  const iconOnly = size === "icon" || size === "icon-sm";
  const cls = ["btn"];
  if (variant !== "default") cls.push(`btn-${variant}`);
  cls.push(...({ sm: ["btn-sm"], lg: ["btn-lg"], icon: ["btn-icon"],
                 "icon-sm": ["btn-icon", "btn-sm"] }[size] || []));
  const b = el("button", { class: cls.join(" "), type, id, title,
                           "aria-label": ariaLabel || (iconOnly ? (title || label) : null),
                           disabled: !!disabled });
  setButton(b, iconOnly ? "" : label, iconName);
  if (onClick) b.onclick = onClick;
  return b;
}

/* Change a button's label and icon in place (Pause ⇄ Resume). */
function setButton(b, label, iconName) {
  b.replaceChildren(...[iconName ? icon(iconName) : null, label || null].filter(Boolean));
  return b;
}

/* variant: default | secondary | outline | destructive */
function badge(text, variant = "default", { dot = false, title } = {}) {
  return el("span", { class: `badge${variant === "default" ? "" : ` badge-${variant}`}`, title },
    dot ? el("span", { class: "dot" }) : null, text);
}

/* The badge for a job state, used by every job table. */
function jobStateBadge(state) {
  const variant = { RUNNING: "default", QUEUED: "secondary", STOPPING: "outline",
                    COMPLETED: "secondary", FAILED: "destructive",
                    CANCELLED: "outline" }[state] || "outline";
  return badge(F.cap(String(state || "").toLowerCase()), variant);
}

function progressBar(frac) {
  const f = Math.max(0, Math.min(1, frac || 0));
  return el("div", { class: "progress", role: "progressbar", "aria-valuemin": 0,
                     "aria-valuemax": 100, "aria-valuenow": Math.round(f * 100) },
    el("i", { style: `width:${f * 100}%` }));
}

function setProgress(bar, frac) {
  const f = Math.max(0, Math.min(1, frac || 0));
  bar.firstChild.style.width = `${f * 100}%`;
  bar.setAttribute("aria-valuenow", Math.round(f * 100));
}

function sectionHead(title, note, { split = false } = {}) {
  return el("div", { class: `section-head${split ? " split" : ""}` },
    el("div", { class: "section-title" }, title),
    note ? el("span", { class: "section-note" }, note) : null);
}

function prose(...children) {
  return el("p", { class: "prose" }, ...children);
}

function kv(pairs) {
  return el("dl", { class: "kv" }, ...pairs.flatMap(([k, v]) =>
    [el("dt", {}, k), el("dd", {}, v)]));
}

const MILESTONES = [512, 1024, 2048, 4096, 8192, 16384];

/* rates: {"512": 0.98, ...} as fractions; counts optional */
function rateBars(rates, counts) {
  const wrap = el("div", { class: "rates" });
  let shown = 0;
  for (const m of MILESTONES) {
    const r = Number((rates || {})[String(m)] || 0);
    // Always show the core four; show larger tiles only once they happen.
    if (m >= 8192 && r <= 0 && shown >= 4) continue;
    shown++;
    const pctv = r * 100;
    wrap.append(el("div", { class: "rate-row" },
      el("div", { class: `tile-chip ${F.tileClass(m)}` }, F.n(m)),
      progressBar(r),
      el("div", { class: "rate-pct" }, F.pct(pctv, pctv >= 10 || pctv === 0 ? 0 : 1))));
  }
  if (counts) {
    wrap.append(el("div", { class: "footnote" },
      "Share of all games this run has played."));
  }
  return wrap;
}

/* The circle-help icon beside a field label, carrying the explanation. */
function tooltip(text) {
  if (!App.settings.show_tooltips) return null;
  return el("span", { class: "help", title: text, "aria-label": text, role: "note" },
    icon("circle-help", 14));
}

function emptyState(iconName, title, message, action) {
  return el("div", { class: "empty" },
    el("div", { class: "empty-header" },
      el("div", { class: "empty-media" }, icon(iconName, 20)),
      el("div", { class: "empty-title" }, title),
      message ? el("div", { class: "empty-description" }, message) : null),
    action ? el("div", { class: "empty-content" }, action) : null);
}

function field(label, control, hint, tip) {
  return el("div", { class: "field" },
    el("label", { class: "field-label", for: control?.id || null }, label,
      tip ? tooltip(tip) : null),
    control,
    hint ? el("div", { class: "field-description" }, hint) : null);
}

function formGrid(fields, min = 200) {
  return el("div", { class: "form-grid", style: `--min:${min}px` }, ...fields);
}

function numberInput(id, value, attrs = {}) {
  return el("input", { type: "number", id: id || null, value, ...attrs });
}

function selectInput(id, options, value, attrs = {}) {
  const sel = el("select", { id: id || null, ...attrs });
  for (const o of options) {
    const opt = typeof o === "object" ? o : { value: o, label: String(o) };
    sel.append(el("option", { value: opt.value, selected: String(opt.value) === String(value) }, opt.label));
  }
  return sel;
}

function checkbox(label, { checked = false, id, switchStyle = false } = {}) {
  const input = el("input", { type: "checkbox", id: id || null, checked,
                              class: switchStyle ? "switch" : null,
                              role: switchStyle ? "switch" : null });
  const node = el("label", { class: `check${switchStyle ? " switch-row" : ""}` },
    input, el("span", {}, label));
  return { node, input };
}

/* Segmented tabs. items: [[id, label], …]; onSelect(id, viaKeyboard) */
function tabs(items, active, onSelect, label = "Tabs") {
  return el("div", { class: "tabs-list", role: "tablist", "aria-label": label },
    ...items.map(([id, text]) => {
      const t = el("button", { class: "tabs-trigger", role: "tab", type: "button",
                               "aria-selected": String(id === active),
                               tabindex: id === active ? "0" : "-1" }, text);
      t.onclick = () => onSelect(id, false);
      t.onkeydown = (e) => {
        const i = items.findIndex(([x]) => x === id);
        const next = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
        if (next === null) return;
        e.preventDefault();
        onSelect(items[(next + items.length) % items.length][0], true);
      };
      return t;
    }));
}

/* Joined outline toggles, one pressed at a time. items: [[id, label], …] */
function toggleGroup(items, active, onChange, label = "Options") {
  const group = el("div", { class: "toggle-group", role: "group", "aria-label": label });
  const paint = (current) => {
    for (const b of group.children) {
      b.setAttribute("aria-pressed", String(b.dataset.value === current));
    }
  };
  for (const [id, text] of items) {
    const b = el("button", { class: "toggle-item", type: "button", dataset: { value: id } }, text);
    b.onclick = () => { paint(id); onChange(id); };
    group.append(b);
  }
  paint(active);
  return group;
}

/* A bordered box with a ghost trigger that shows and hides its body. */
function collapsible(label, body, { open = false } = {}) {
  const trigger = button(label, { variant: "ghost", size: "sm", icon: "chevron-right" });
  trigger.classList.add("collapsible-trigger");
  const node = el("div", { class: `collapsible${open ? " open" : ""}` },
    el("div", { class: "inline" }, trigger),
    el("div", { class: "collapsible-body" },
      el("div", { class: "collapsible-inner" }, el("div", { class: "collapsible-pad" }, body))));
  trigger.setAttribute("aria-expanded", String(open));
  trigger.onclick = () => {
    const nowOpen = node.classList.toggle("open");
    trigger.setAttribute("aria-expanded", String(nowOpen));
  };
  return node;
}

/* Mean with its 95% confidence interval, drawn so overlap is visible.
 * rows: [{name, mean, lo, hi, color, badge}] */
function ciBars(rows) {
  const maxHi = Math.max(1, ...rows.map((r) => r.hi || 0));
  const pctOf = (v) => `${Math.max(0, Math.min(100, (v / maxHi) * 100)).toFixed(2)}%`;
  return el("div", { class: "ci-list" }, ...rows.map((r) => el("div", { class: "ci-row" },
    el("div", { class: "ci-label" },
      el("span", { class: "ci-name" }, r.name, r.badge || null),
      el("span", { class: "ci-value" },
        `${F.n(r.mean, 0)}  [${F.compact(r.lo)} – ${F.compact(r.hi)}]`)),
    el("div", { class: "ci-track", role: "img",
                "aria-label": `${r.name}: mean ${F.n(r.mean, 0)}, 95% CI ${F.n(r.lo, 0)} to ${F.n(r.hi, 0)}` },
      el("div", { class: "ci-band", style:
        `left:${pctOf(r.lo)};width:${pctOf(Math.max(0, r.hi - r.lo))};background:${r.color}` }),
      el("div", { class: "ci-mean", style: `left:${pctOf(r.mean)};background:${r.color}` })))));
}

/* A standard data table. head: [label | [label, {num:true}]]; rows: <tr>s */
function table(head, rows) {
  return el("div", { class: "table-wrap" },
    el("table", { class: "data" },
      el("thead", {}, el("tr", {}, ...head.map((h) => {
        const [label, o] = Array.isArray(h) ? h : [h, {}];
        return el("th", { class: o.num ? "num" : null }, label);
      }))),
      el("tbody", {}, ...rows)));
}

/* Run picker used in the header on run-scoped pages. */
function runPicker() {
  const sel = el("select", { id: "run-picker", class: "control-sm", "aria-label": "Training run",
                             title: "Training run" });
  const paint = (status) => {
    // The picker is rebuilt on every visit; a detached one stops listening.
    if (!sel.isConnected && sel.dataset.mounted) { off(); return; }
    const runs = status?.runs || [];
    const names = runs.map((r) => r.name);
    if (!names.includes(App.run)) names.unshift(App.run);
    const same = sel.options.length === names.length &&
      [...sel.options].every((o, i) => o.value === names[i]);
    if (!same) {
      sel.replaceChildren(...names.map((n) => el("option", { value: n }, n)));
    }
    sel.value = App.run;
  };
  sel.onchange = () => App.setRun(sel.value);
  const off = App.onStatus(paint);
  if (App.status) paint(App.status);
  requestAnimationFrame(() => { sel.dataset.mounted = "1"; });
  return sel;
}

/* A running job as a list row, with its progress and a Stop button. */
function jobRow(job, { onStop } = {}) {
  const p = job.progress || {};
  const total = p.total || 0;
  const frac = total ? Math.min(1, (p.done || 0) / total) : 0;
  const stop = button("Stop", { variant: "outline", size: "sm", icon: "square" });
  stop.onclick = async () => {
    stop.disabled = true;
    try {
      await API.post(`/api/jobs/${encodeURIComponent(job.id)}/stop`, {});
      Toast.show("Stopping", job.label, "warn");
      onStop && onStop();
    } catch (e) { Toast.error("Could not stop", e.message); stop.disabled = false; }
  };
  return el("div", { class: "list-row" },
    jobStateBadge(job.state),
    el("div", { class: "list-main" },
      el("div", { class: "list-title" }, job.label),
      el("div", { class: "list-meta" },
        total ? `${F.n(p.done || 0)} / ${F.n(total)} · ` : "",
        p.phase ? p.phase + " · " : "",
        F.dur(job.duration)),
      total ? el("div", { style: "margin-top:6px" }, progressBar(frac)) : null),
    ["RUNNING", "QUEUED"].includes(job.state) ? stop : null);
}

/* The card shown while a job runs: progress, a status line and Cancel.
 * Returns {node, update(frac, text)}. */
function progressCard(title, description, { onCancel } = {}) {
  const bar = progressBar(0);
  const text = el("div", { class: "note" }, "Starting…");
  const cancel = onCancel
    ? button("Cancel", { variant: "outline", size: "sm", icon: "square", onClick: onCancel })
    : null;
  const node = card(
    cardHeader(title, description,
      el("span", { class: "badge badge-secondary" }, icon("loader-circle", 12), "Running")),
    cardContent(el("div", { class: "stack stack-12" }, bar,
      el("div", { class: "inline inline-12" }, text, el("span", { class: "spacer" }), cancel))));
  node.querySelector(".icon-loader-circle").classList.add("spinner");
  return {
    node,
    update(frac, message) { setProgress(bar, frac); text.textContent = message; },
  };
}

/* Renders an evaluation result block (shared by Evaluate and Checkpoints). */
function evaluationResult(res) {
  if (!res || !res.games) return el("div", { class: "note" }, "No games played.");
  const ci = res.ci95_mean || [0, 0];
  const rate = (t) => {
    const e = (res.tile_rates || {})[t];
    if (!e) return null;
    return typeof e === "object" ? e : { rate: e, ci95: null, count: null };
  };
  const rows = [];
  for (const m of MILESTONES) {
    const r = rate(String(m));
    if (!r) continue;
    if (m >= 8192 && !r.count) continue;
    rows.push(el("tr", {},
      el("td", {}, el("span", { class: `tile-chip inline ${F.tileClass(m)}` }, F.n(m))),
      el("td", { class: "num" }, F.pct(r.rate * 100, 2)),
      el("td", { class: "num muted" },
        r.ci95 ? `${F.pct(r.ci95[0] * 100, 2)} – ${F.pct(r.ci95[1] * 100, 2)}` : "—"),
      el("td", { class: "num muted" }, r.count !== null && r.count !== undefined ? F.n(r.count) : "—")));
  }

  return el("div", { class: "stack" },
    stats([
      statTile("Mean score", F.n(res.mean_score, 0),
        `95% CI ${F.compact(ci[0])} – ${F.compact(ci[1])}`),
      statTile("Median", F.n(res.median_score, 0)),
      statTile("Std dev", F.n(res.std_score, 0)),
      statTile("Best game", F.n(res.max_score), `worst ${F.n(res.min_score)}`),
      statTile("Mean moves", F.n(res.mean_moves, 0), `longest ${F.n(res.max_moves)}`),
      statTile("Highest tile", F.n(res.highest_tile), `median ${F.n(res.median_tile)}`),
    ]),
    sectionHead("Tile Achievement", "Wilson score intervals — correct near 0% and 100%"),
    table(["Tile", ["Rate", { num: true }], ["95% CI", { num: true }], ["Games", { num: true }]], rows),
    el("div", { class: "footnote" },
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

function exportButtons(onJson, onCsv) {
  return [
    onJson ? button("Export JSON", { variant: "outline", size: "sm", icon: "download", onClick: onJson }) : null,
    onCsv ? button("Export CSV", { variant: "outline", size: "sm", icon: "download", onClick: onCsv }) : null,
  ].filter(Boolean);
}

function toCSV(rows) {
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(esc).join(",")).join("\n");
}
