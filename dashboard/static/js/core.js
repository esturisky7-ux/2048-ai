/* Core: formatting, the API client, live updates, routing and notifications.
 *
 * Plain ES modules-free scripts loaded in order — no bundler, no framework,
 * nothing fetched from a CDN. Views register themselves into App.views and the
 * router calls mount()/unmount() on them.
 */

/* ------------------------------------------------------------- formatting */
const F = {
  n(v, d = 0) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return Number(v).toLocaleString(undefined, {
      minimumFractionDigits: d, maximumFractionDigits: d,
    });
  },
  compact(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const n = Number(v);
    if (!App.settings.compact_numbers) return F.n(n);
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (Math.abs(n) >= 10000) return (n / 1000).toFixed(1) + "k";
    return F.n(n);
  },
  pct(v, d = 1) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return Number(v).toFixed(d) + "%";
  },
  dur(sec) {
    if (!sec && sec !== 0) return "—";
    sec = Math.max(0, Math.floor(sec));
    const d = Math.floor(sec / 86400);
    const h = Math.floor((sec % 86400) / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
  },
  ago(ts) {
    if (!ts) return "never";
    const diff = Date.now() / 1000 - ts;
    if (diff < 0) return "just now";
    return F.dur(diff) + " ago";
  },
  bytes(b) {
    if (b === null || b === undefined) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0, v = Number(b);
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
  },
  time(ts) {
    if (!ts) return "—";
    return new Date(ts * 1000).toLocaleTimeString();
  },
  date(ts) {
    if (!ts) return "—";
    return new Date(ts * 1000).toLocaleString();
  },
  tileClass(v) {
    return v > 32768 ? "t32768" : `t${v}`;
  },
};

/* --------------------------------------------------------------- elements */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}
/* Append children, skipping the nullish ones.
 *
 * Element.append() stringifies null into the literal text "null" — a
 * genuinely surprising way to render `condition ? el(...) : null`. el() already
 * filters its children; this is the same courtesy for an existing node. */
function add(parent, ...children) {
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    parent.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return parent;
}

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ------------------------------------------------------------- API client */
const API = {
  async get(path, params) {
    const url = new URL(path, location.origin);
    for (const [k, v] of Object.entries(params || {})) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    }
    const res = await fetch(url, { headers: { "Accept": "application/json" } });
    const body = await res.json().catch(() => ({ error: res.statusText }));
    if (!res.ok) throw new ApiError(body.error || res.statusText, res.status);
    return body;
  },
  async post(path, data) {
    const res = await fetch(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // The server requires this header. A cross-origin page cannot set it
        // without a preflight, which the server never grants — that is the
        // whole CSRF defence for a control surface that can start processes.
        "X-2048-Request": "1",
      },
      body: JSON.stringify(data || {}),
    });
    const body = await res.json().catch(() => ({ error: res.statusText }));
    if (!res.ok) throw new ApiError(body.error || res.statusText, res.status);
    return body;
  },
};

class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

/* ----------------------------------------------------------- notifications */
const Toast = {
  container: null,
  show(title, message, kind = "info", ms = 5200) {
    if (!this.container) this.container = $("#toasts");
    const close = el("button", { title: "Dismiss", "aria-label": "Dismiss" }, "×");
    const node = el("div", { class: `toast ${kind}`, role: "status" },
      el("div", { class: "t-body" },
        el("div", { class: "t-title" }, title),
        message ? el("div", { class: "t-msg" }, message) : null),
      close);
    close.onclick = () => node.remove();
    this.container.append(node);
    if (ms) setTimeout(() => node.remove(), ms);
    return node;
  },
  ok(t, m) { return this.show(t, m, "ok"); },
  warn(t, m) { return this.show(t, m, "warn"); },
  error(t, m) { return this.show(t, m, "error", 9000); },
};

function confirmDialog(title, message, { danger = false, confirmText = "Confirm" } = {}) {
  return new Promise((resolve) => {
    const scrim = el("div", { class: "modal-scrim" });
    const cancel = el("button", { class: "btn" }, "Cancel");
    const go = el("button", { class: `btn ${danger ? "btn-danger" : "btn-primary"}` }, confirmText);
    const modal = el("div", { class: "modal", role: "dialog", "aria-modal": "true" },
      el("h3", {}, title), el("p", {}, message),
      el("div", { class: "btn-row" }, cancel, go));
    scrim.append(modal);
    const done = (v) => { scrim.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") done(false); if (e.key === "Enter") done(true); };
    cancel.onclick = () => done(false);
    go.onclick = () => done(true);
    scrim.onclick = (e) => { if (e.target === scrim) done(false); };
    document.addEventListener("keydown", onKey);
    document.body.append(scrim);
    go.focus();
  });
}

/* ------------------------------------------------------------------- App */
const App = {
  views: {},
  current: null,
  currentName: null,
  run: "default",
  status: null,
  settings: {
    theme: "dark", refresh_ms: 2000, playback_speed: 1, eval_games: 200,
    workers: 1, confirm_destructive: true, show_tooltips: true,
    compact_numbers: true,
  },
  listeners: new Set(),
  _es: null,
  _pollTimer: null,
  _lastJobStates: new Map(),

  /* -- status distribution ------------------------------------------- */
  onStatus(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },

  setStatus(status) {
    if (status && status.error) return;
    this.status = status;
    this.announceJobs(status);
    this.paintGlobalStatus(status);
    for (const fn of this.listeners) {
      try { fn(status); } catch (e) { console.error(e); }
    }
  },

  /* Tell the user when a job they started finishes, wherever they are. */
  announceJobs(status) {
    const seen = new Set();
    for (const job of [...(status.jobs || []), ...(status.recent_failures || [])]) {
      seen.add(job.id);
      const prev = this._lastJobStates.get(job.id);
      if (prev && prev !== job.state) {
        if (job.state === "COMPLETED") {
          Toast.ok(`${job.type} finished`, job.label);
          document.dispatchEvent(new CustomEvent("job:done", { detail: job }));
        } else if (job.state === "FAILED") {
          Toast.error(`${job.type} failed`, job.error || job.label);
        } else if (job.state === "CANCELLED") {
          Toast.show(`${job.type} stopped`, job.label, "warn");
          document.dispatchEvent(new CustomEvent("job:done", { detail: job }));
        }
      }
      this._lastJobStates.set(job.id, job.state);
    }
    // Forget jobs that have aged out of the server's list.
    for (const id of [...this._lastJobStates.keys()]) {
      if (!seen.has(id)) this._lastJobStates.delete(id);
    }
  },

  paintGlobalStatus(status) {
    const map = {
      TRAINING: ["live", "Training"],
      STARTING: ["busy", "Starting"],
      STOPPING: ["warn", "Stopping"],
      EVALUATING: ["busy", "Evaluating"],
      BENCHMARKING: ["busy", "Benchmarking"],
      EXPERIMENTING: ["busy", "Experimenting"],
      STOPPED: ["idle", "Idle"],
    };
    const [cls, label] = map[status.state] || ["idle", "Idle"];
    const pill = $("#global-status");
    if (pill) {
      pill.className = `pill ${cls}`;
      pill.innerHTML = "";
      pill.append(el("i"), el("span", {}, label));
    }
    const dot = $("#nav-training-dot");
    if (dot) dot.style.display = status.training?.running ? "" : "none";
    const t = $("#topbar-run");
    if (t) t.textContent = status.run;
  },

  /* -- live updates ---------------------------------------------------- */
  startLive() {
    this.stopLive();
    // ?live=poll forces the polling path. Useful when something between the
    // browser and the server buffers event streams (some proxies do), and for
    // headless captures, where an open stream keeps the page from ever
    // reaching an idle state.
    const forcePoll =
      new URLSearchParams(location.search).get("live") === "poll";
    // Server-Sent Events are the primary channel; polling is the fallback for
    // any environment where EventSource is unavailable or the stream drops.
    if (window.EventSource && !forcePoll) {
      try {
        const es = new EventSource(`/api/stream?run=${encodeURIComponent(this.run)}`);
        es.addEventListener("status", (e) => {
          try { this.setStatus(JSON.parse(e.data)); } catch (_) { }
        });
        es.addEventListener("events", (e) => {
          try {
            document.dispatchEvent(new CustomEvent("log:events", {
              detail: JSON.parse(e.data).events || [],
            }));
          } catch (_) { }
        });
        es.onerror = () => {
          // Browser retries on its own; keep a poll going so the UI is never
          // frozen while it does.
          if (!this._pollTimer) this.startPolling();
        };
        this._es = es;
        return;
      } catch (_) { /* fall through to polling */ }
    }
    this.startPolling();
  },

  startPolling() {
    const tick = async () => {
      try { this.setStatus(await API.get("/api/status", { run: this.run })); }
      catch (_) { }
      this._pollTimer = setTimeout(tick, this.settings.refresh_ms);
    };
    tick();
  },

  stopLive() {
    if (this._es) { this._es.close(); this._es = null; }
    if (this._pollTimer) { clearTimeout(this._pollTimer); this._pollTimer = null; }
  },

  async refreshNow() {
    try { this.setStatus(await API.get("/api/status", { run: this.run })); }
    catch (e) { console.error(e); }
  },

  setRun(name) {
    if (name === this.run) return;
    this.run = name;
    localStorage.setItem("run", name);
    this.startLive();
    this.refreshNow();
    if (this.current?.onRunChange) this.current.onRunChange(name);
    else this.route();
  },

  /* -- theme ------------------------------------------------------------ */
  applyTheme(theme) {
    const wanted = theme === "system"
      ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
      : theme;
    document.documentElement.setAttribute("data-theme", wanted);
  },

  /* -- routing ---------------------------------------------------------- */
  route() {
    const hash = location.hash.replace(/^#\/?/, "") || "overview";
    const [name, ...rest] = hash.split("/");
    const view = this.views[name] || this.views.overview;
    if (this.current && this.current.unmount) {
      try { this.current.unmount(); } catch (e) { console.error(e); }
    }
    this.current = view;
    this.currentName = view === this.views[name] ? name : "overview";

    $$(".nav a").forEach((a) => {
      a.classList.toggle("active", a.dataset.view === this.currentName);
      if (a.dataset.view === this.currentName) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    $("#page-title").textContent = view.title || "";
    $("#page-sub").textContent = view.subtitle || "";
    const actions = $("#page-actions");
    actions.innerHTML = "";

    const root = $("#view");
    root.innerHTML = "";
    document.title = `${view.title || "2048 AI"} · 2048 AI Control Center`;
    try {
      view.mount(root, { args: rest, actions });
    } catch (e) {
      console.error(e);
      root.append(el("div", { class: "panel empty" },
        el("h3", {}, "This page failed to load"),
        el("div", { class: "mono" }, String(e && e.message || e))));
    }
    if (this.status) {
      try { view.onStatus && view.onStatus(this.status); } catch (_) { }
    }
    $(".sidebar")?.classList.remove("open");
    $(".scrim")?.remove();
    root.scrollIntoView({ block: "start" });
  },

  go(path) { location.hash = "#/" + path; },
};

/* Views get status pushes automatically. */
App.onStatus((s) => {
  if (App.current && App.current.onStatus) {
    try { App.current.onStatus(s); } catch (e) { console.error(e); }
  }
});

/* ------------------------------------------------------------ job helpers */
const Jobs = {
  /* Poll one job until it leaves a live state, calling onUpdate each time. */
  watch(jobId, onUpdate, interval = 700) {
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      try {
        const job = await API.get(`/api/jobs/${encodeURIComponent(jobId)}`);
        onUpdate(job);
        if (["COMPLETED", "FAILED", "CANCELLED"].includes(job.state)) return;
      } catch (e) {
        onUpdate({ id: jobId, state: "FAILED", error: e.message });
        return;
      }
      setTimeout(tick, interval);
    };
    tick();
    return () => { stopped = true; };
  },

  progressBar(job) {
    const p = job.progress || {};
    const total = p.total || 0;
    const done = p.done || 0;
    const frac = total ? Math.min(1, done / total) : 0;
    const wrap = el("div", {},
      el("div", { class: "bar blue" }, el("i", { style: `width:${frac * 100}%` })),
      el("div", { class: "meta", style: "margin-top:5px" },
        total ? `${F.n(done)} / ${F.n(total)}` : (p.phase || job.state)));
    return wrap;
  },
};

/* ----------------------------------------------------------------- boot */
async function boot() {
  App.run = localStorage.getItem("run") || "default";

  try {
    const s = await API.get("/api/settings");
    App.settings = { ...App.settings, ...s.settings };
  } catch (_) { /* defaults are fine */ }
  App.applyTheme(App.settings.theme);
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if (App.settings.theme === "system") App.applyTheme("system");
  });

  buildNav();
  window.addEventListener("hashchange", () => App.route());
  App.route();
  App.startLive();
  await App.refreshNow();

  // If nothing has ever been trained, land on the welcome flow rather than an
  // overview full of dashes.
  if (App.status && !App.status.has_any_run && !location.hash) {
    App.go("training");
  }
  installShortcuts();
}

const NAV = [
  { group: "Monitor", items: [
    { id: "overview", label: "Overview", icon: "◧" },
    { id: "training", label: "Training", icon: "▶", dot: true },
    { id: "play", label: "Play", icon: "◆" },
  ] },
  { group: "Measure", items: [
    { id: "evaluate", label: "Evaluate", icon: "✓" },
    { id: "compare", label: "Compare", icon: "⇄" },
    { id: "experiments", label: "Experiments", icon: "⚗" },
    { id: "benchmarks", label: "Benchmarks", icon: "◷" },
  ] },
  { group: "Manage", items: [
    { id: "checkpoints", label: "Checkpoints", icon: "▤" },
    { id: "logs", label: "Logs", icon: "≡" },
    { id: "system", label: "System", icon: "⚙" },
    { id: "settings", label: "Settings", icon: "⚙" },
  ] },
];

function buildNav() {
  const nav = $(".nav");
  nav.innerHTML = "";
  for (const section of NAV) {
    const g = el("div", { class: "nav-group" }, el("span", {}, section.group));
    for (const item of section.items) {
      const a = el("a", { href: `#/${item.id}`, dataset: { view: item.id } },
        el("span", { class: "ico" }, item.icon),
        el("span", {}, item.label),
        item.dot ? el("span", { class: "badge-dot", id: "nav-training-dot",
                                style: "display:none" }) : null);
      g.append(a);
    }
    nav.append(g);
  }
}

/* Keyboard shortcuts: g+<key> jumps, ? shows the list. */
function installShortcuts() {
  const JUMP = { o: "overview", t: "training", p: "play", e: "evaluate",
                 c: "compare", x: "experiments", b: "benchmarks",
                 k: "checkpoints", l: "logs", s: "system" };
  let armed = false;
  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag) || e.target.isContentEditable) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "g") { armed = true; setTimeout(() => (armed = false), 1200); return; }
    if (armed && JUMP[e.key]) { App.go(JUMP[e.key]); armed = false; e.preventDefault(); return; }
    armed = false;
    if (e.key === "?") { showShortcuts(); e.preventDefault(); }
  });
}

function showShortcuts() {
  const rows = [
    ["g then o", "Overview"], ["g then t", "Training"], ["g then p", "Play"],
    ["g then e", "Evaluate"], ["g then c", "Compare"], ["g then k", "Checkpoints"],
    ["g then l", "Logs"], ["g then s", "System"],
    ["Arrows / WASD", "Move (human game)"], ["Space", "Pause or resume a watched game"],
    ["?", "This list"],
  ];
  const scrim = el("div", { class: "modal-scrim" });
  const close = el("button", { class: "btn btn-primary" }, "Close");
  scrim.append(el("div", { class: "modal" },
    el("h3", {}, "Keyboard shortcuts"),
    el("dl", { class: "kv" }, rows.flatMap(([k, v]) =>
      [el("dt", {}, el("kbd", {}, k)), el("dd", {}, v)])),
    el("div", { class: "btn-row", style: "margin-top:16px" }, close)));
  close.onclick = () => scrim.remove();
  scrim.onclick = (e) => { if (e.target === scrim) scrim.remove(); };
  document.body.append(scrim);
  close.focus();
}

document.addEventListener("DOMContentLoaded", boot);
