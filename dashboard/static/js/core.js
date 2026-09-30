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
  cap(s) {
    s = String(s || "");
    return s.charAt(0).toUpperCase() + s.slice(1);
  },
  tileClass(v) {
    return v >= 8192 ? "t8192" : `t${v}`;
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

/* The badge for the system state: filled while training, secondary while
 * other work runs, outline when idle. Shared by the header and Overview. */
function statusBadge(state) {
  const [variant, label, live] = {
    TRAINING: ["default", "Training", true],
    STARTING: ["secondary", "Starting"],
    STOPPING: ["outline", "Stopping"],
    EVALUATING: ["secondary", "Evaluating"],
    BENCHMARKING: ["secondary", "Benchmarking"],
    EXPERIMENTING: ["secondary", "Experimenting"],
    STOPPED: ["outline", "Idle"],
  }[state] || ["outline", "Idle"];
  return { variant, label, live: !!live };
}

/* ----------------------------------------------------------- notifications */
const Toast = {
  container: null,
  ICONS: { ok: "circle-check", warn: "triangle-alert", error: "circle-x", info: "info" },
  show(title, message, kind = "info", ms = 5200) {
    if (!this.container) this.container = $("#toasts");
    const close = el("button", { class: "t-close", title: "Dismiss", "aria-label": "Dismiss" },
      icon("x", 14));
    const node = el("div", { class: `toast ${kind}`, role: "status" },
      icon(this.ICONS[kind] || "info", 16),
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

/* A modal with a title, a description, optional body and a footer. Resolves
 * with whatever `done` is called with; Escape and the scrim resolve `false`. */
function modal(title, description, { body = null, buttons = [], initial = null } = {}) {
  return new Promise((resolve) => {
    const scrim = el("div", { class: "modal-scrim" });
    const titleId = `modal-${Date.now()}`;
    const box = el("div", { class: "modal", role: "dialog", "aria-modal": "true",
                            "aria-labelledby": titleId },
      el("h2", { class: "modal-title", id: titleId }, title),
      description ? el("p", { class: "modal-description" }, description) : null,
      body,
      el("div", { class: "modal-footer" }, ...buttons.map((b) => b.node)));
    scrim.append(box);
    const opener = document.activeElement;
    const done = (v) => {
      scrim.remove();
      document.removeEventListener("keydown", onKey, true);
      if (opener && opener.focus) opener.focus();
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.preventDefault(); done(false); }
      else if (e.key === "Enter" && e.target.tagName !== "BUTTON") {
        const primary = buttons.find((b) => b.primary);
        if (primary) { e.preventDefault(); done(primary.value()); }
      }
    };
    for (const b of buttons) b.node.onclick = () => done(b.value());
    scrim.onclick = (e) => { if (e.target === scrim) done(false); };
    document.addEventListener("keydown", onKey, true);
    document.body.append(scrim);
    (initial || buttons[buttons.length - 1]?.node)?.focus();
  });
}

function confirmDialog(title, message, { danger = false, confirmText = "Confirm" } = {}) {
  const cancel = el("button", { class: "btn btn-outline", type: "button" }, "Cancel");
  const go = el("button", { class: `btn ${danger ? "btn-destructive" : ""}`, type: "button" },
    confirmText);
  return modal(title, message, { buttons: [
    { node: cancel, value: () => false },
    { node: go, value: () => true, primary: true },
  ] });
}

/* A styled replacement for window.prompt(): resolves the text, or null. */
function promptDialog(title, message, value = "", { confirmText = "Save", placeholder = "",
                                                    maxlength = 80 } = {}) {
  const input = el("input", { type: "text", value, placeholder, maxlength,
                              "aria-label": title });
  const cancel = el("button", { class: "btn btn-outline", type: "button" }, "Cancel");
  const go = el("button", { class: "btn", type: "button" }, confirmText);
  const result = modal(title, message, { body: input, initial: input, buttons: [
    { node: cancel, value: () => null },
    { node: go, value: () => input.value, primary: true },
  ] });
  requestAnimationFrame(() => input.select());
  return result.then((v) => (v === false ? null : v));
}

/* ------------------------------------------------------------------- App */
const App = {
  views: {},
  current: null,
  currentName: null,
  run: "default",
  status: null,
  settings: {
    refresh_ms: 2000, playback_speed: 1, eval_games: 200,
    workers: 1, confirm_destructive: true, show_tooltips: true,
    compact_numbers: true,
  },
  listeners: new Set(),
  _es: null,
  _pollTimer: null,
  _polling: false,
  _pollGen: 0,
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
          Toast.ok(`${F.cap(job.type)} finished`, job.label);
          document.dispatchEvent(new CustomEvent("job:done", { detail: job }));
        } else if (job.state === "FAILED") {
          Toast.error(`${F.cap(job.type)} failed`, job.error || job.label);
        } else if (job.state === "CANCELLED") {
          Toast.show(`${F.cap(job.type)} stopped`, job.label, "warn");
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
    const { variant, label, live } = statusBadge(status.state);
    const pill = $("#global-status");
    if (pill) {
      pill.className = `badge${variant === "default" ? "" : ` badge-${variant}`} status-badge${live ? " live" : ""}`;
      pill.title = `Status: ${label}`;
      pill.replaceChildren(el("i"), el("span", { class: "status-label" }, label));
    }
    const dot = $("#nav-training-dot");
    if (dot) dot.hidden = !status.training?.running;
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
        // The stream is (back) up: the fallback poll has done its job.
        es.onopen = () => { if (this._es === es) this.stopPolling(); };
        es.addEventListener("status", (e) => {
          let status;
          try { status = JSON.parse(e.data); } catch (_) { return; }
          if (this._es === es && status && !status.error) this.stopPolling();
          this.setStatus(status);
        });
        es.addEventListener("events", (e) => {
          try {
            document.dispatchEvent(new CustomEvent("log:events", {
              detail: JSON.parse(e.data).events || [],
            }));
          } catch (_) { }
        });
        es.onerror = () => {
          // Either the browser is retrying on its own (and onopen will stop
          // the poll once it succeeds) or it has given up for good, say on a
          // 503; both ways, poll so the UI is never frozen. startPolling()
          // runs one loop at most however often this fires.
          if (this._es === es) this.startPolling();
        };
        this._es = es;
        return;
      } catch (_) { /* fall through to polling */ }
    }
    this.startPolling();
  },

  /* One polling loop at most. Each loop carries the generation it was started
   * in, and stopPolling() moves the generation on, so a tick that was already
   * waiting on the network when polling stopped cannot schedule another. */
  startPolling() {
    if (this._polling) return;
    this._polling = true;
    const gen = ++this._pollGen;
    const tick = async () => {
      let status = null;
      try { status = await API.get("/api/status", { run: this.run }); }
      catch (_) { }
      if (gen !== this._pollGen) return;
      if (status) this.setStatus(status);
      this._pollTimer = setTimeout(tick, this.settings.refresh_ms);
    };
    tick();
  },

  stopPolling() {
    this._pollGen++;
    this._polling = false;
    if (this._pollTimer) { clearTimeout(this._pollTimer); this._pollTimer = null; }
  },

  stopLive() {
    if (this._es) { this._es.close(); this._es = null; }
    this.stopPolling();
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

    $$(".nav-item").forEach((a) => {
      const on = a.dataset.view === this.currentName;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    $("#page-title").textContent = view.title || "";
    $("#page-sub").textContent = view.subtitle || "";
    const actions = $("#page-actions");
    actions.replaceChildren();

    const root = $("#view");
    root.replaceChildren();
    document.title = `${view.title || "2048 AI"} · 2048 AI Control Center`;
    try {
      view.mount(root, { args: rest, actions });
    } catch (e) {
      console.error(e);
      root.append(card(
        el("div", { class: "card-content" },
          emptyState("triangle-alert", "This page failed to load",
            el("span", { class: "mono" }, String(e && e.message || e))))));
    }
    if (this.status) {
      try { view.onStatus && view.onStatus(this.status); } catch (_) { }
    }
    closeMobileSidebar();
    window.scrollTo({ top: 0 });
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
};

/* ----------------------------------------------------------------- boot */
async function boot() {
  App.run = localStorage.getItem("run") || "default";

  try {
    const s = await API.get("/api/settings");
    App.settings = { ...App.settings, ...s.settings };
  } catch (_) { /* defaults are fine */ }

  buildNav();
  window.addEventListener("hashchange", () => App.route());
  App.route();
  App.startLive();
  await App.refreshNow();

  // On a fresh install, show both the optional pretrained demo and training.
  if (App.status && !App.status.has_any_run && !location.hash) {
    App.go("overview");
  }
  installShortcuts();
}

const NAV = [
  { group: "Monitor", items: [
    { id: "overview", label: "Overview", icon: "layout-dashboard" },
    { id: "training", label: "Training", icon: "activity", dot: true },
    { id: "play", label: "Play", icon: "gamepad-2" },
  ] },
  { group: "Measure", items: [
    { id: "evaluate", label: "Evaluate", icon: "clipboard-check" },
    { id: "compare", label: "Compare", icon: "git-compare" },
    { id: "experiments", label: "Experiments", icon: "flask-conical" },
    { id: "benchmarks", label: "Benchmarks", icon: "gauge" },
  ] },
  { group: "Manage", items: [
    { id: "checkpoints", label: "Checkpoints", icon: "database" },
    { id: "logs", label: "Logs", icon: "scroll-text" },
    { id: "system", label: "System", icon: "cpu" },
    { id: "settings", label: "Settings", icon: "settings" },
  ] },
];

function buildNav() {
  const nav = $(".nav");
  nav.replaceChildren();
  for (const section of NAV) {
    const menu = el("ul", { class: "nav-menu" });
    for (const item of section.items) {
      menu.append(el("li", {},
        el("a", { class: "nav-item", href: `#/${item.id}`,
                  dataset: { view: item.id, tooltip: item.label } },
          icon(item.icon),
          el("span", { class: "label" }, item.label)),
        item.dot ? el("span", { class: "nav-dot", id: "nav-training-dot",
                                title: "Training is running", hidden: true }) : null));
    }
    nav.append(el("div", { class: "nav-group", role: "group", "aria-label": section.group },
      el("div", { class: "nav-group-label" }, section.group), menu));
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
  const close = el("button", { class: "btn", type: "button" }, "Close");
  modal("Keyboard Shortcuts", null, {
    body: shortcutList(),
    buttons: [{ node: close, value: () => true, primary: true }],
  });
}

/* The shortcut reference, shared by the "?" dialog and the Settings page. */
function shortcutList() {
  const k = (t) => el("kbd", {}, t);
  const then = () => el("span", { class: "section-note" }, "then");
  return el("dl", { class: "keys" },
    el("dt", {}, k("g"), then(), k("o/t/p/e/c/x/b/k/l/s")), el("dd", {}, "Jump to a page"),
    el("dt", {}, el("span", { class: "kbd-group" }, k("↑"), k("↓"), k("←"), k("→")),
      el("span", { class: "section-note" }, "/"), k("WASD")),
    el("dd", {}, "Move, in a human game"),
    el("dt", {}, k("Space")), el("dd", {}, "Pause or resume a watched game"),
    el("dt", {}, k("Ctrl"), k("B")), el("dd", {}, "Collapse or expand the sidebar"),
    el("dt", {}, k("?")), el("dd", {}, "Show the shortcut list"));
}

document.addEventListener("DOMContentLoaded", boot);
