/* Logs, System and Settings — the three utility pages. */

/* ==================================================================== LOGS */
App.views.logs = {
  title: "Logs",
  subtitle: "What the control center and its jobs have been doing",

  mount(root, { actions }) {
    this.limit = 100;
    this.level = "";
    const limitSel = selectInput("l-limit", [
      { value: 100, label: "Last 100" },
      { value: 250, label: "Last 250" },
      { value: 500, label: "Last 500" },
      { value: 1000, label: "Last 1000" },
    ], this.limit);
    limitSel.onchange = () => { this.limit = Number(limitSel.value); this.load(); };
    const levelSel = selectInput("l-level", [
      { value: "", label: "All levels" },
      { value: "info", label: "Info and above" },
      { value: "warn", label: "Warnings and errors" },
      { value: "error", label: "Errors only" },
    ], "");
    levelSel.onchange = () => { this.level = levelSel.value; this.load(); };
    const clear = el("button", { class: "btn btn-sm btn-danger" }, "Clear");
    clear.onclick = async () => {
      const ok = !App.settings.confirm_destructive || await confirmDialog(
        "Clear the log?", "Recorded events are deleted. Training data is untouched.",
        { danger: true, confirmText: "Clear" });
      if (!ok) return;
      await API.post("/api/logs/clear", {}).catch((e) => Toast.error("Failed", e.message));
      this.load();
    };
    actions.append(limitSel, levelSel, clear);

    this.list = el("div", { class: "log-list" });
    root.append(el("section", { class: "panel" },
      el("div", { class: "panel-head" },
        el("h3", {}, "Recent events"),
        el("span", { class: "note" },
          "training milestones, evaluations, job state changes and errors")),
      this.list));

    this.load();
    // New events arrive on the live stream, so the page stays current.
    this._onEvents = (e) => {
      if (!this.list || !e.detail?.length) return;
      for (const ev of e.detail) this.list.prepend(this.line(ev));
      while (this.list.children.length > this.limit) this.list.lastChild.remove();
    };
    document.addEventListener("log:events", this._onEvents);
  },

  unmount() { document.removeEventListener("log:events", this._onEvents); },

  line(ev) {
    const extras = Object.entries(ev)
      .filter(([k]) => !["ts", "level", "message", "seq"].includes(k))
      .map(([k, v]) => `${k}=${v}`).join("  ");
    return el("div", { class: `log-line ${ev.level}` },
      el("span", { class: "ts" }, F.time(ev.ts)),
      el("span", { class: "lvl" }, ev.level),
      el("span", { class: "msg" }, ev.message,
        extras ? el("span", { class: "faint" }, "  " + extras) : null));
  },

  async load() {
    let data;
    try { data = await API.get("/api/logs", { limit: this.limit, level: this.level }); }
    catch (e) {
      this.list.innerHTML = "";
      this.list.append(el("div", { class: "faint" }, `Could not load: ${e.message}`));
      return;
    }
    const events = (data.events || []).slice().reverse();
    this.list.innerHTML = "";
    if (!events.length) {
      this.list.append(el("div", { class: "faint" }, "Nothing logged yet."));
      return;
    }
    for (const ev of events) this.list.append(this.line(ev));
  },
};

/* ================================================================== SYSTEM */
App.views.system = {
  title: "System",
  subtitle: "Platform, versions, storage and running processes",

  mount(root) {
    this.root = root;
    this.panel = el("section", { class: "panel" });
    root.append(this.panel);
    this.load();
    this._timer = setInterval(() => this.load(), 8000);
  },

  unmount() { clearInterval(this._timer); },

  async load() {
    let s;
    try { s = await API.get("/api/system"); }
    catch (e) {
      this.panel.innerHTML = "";
      this.panel.append(el("div", { class: "faint" }, `Could not load: ${e.message}`));
      return;
    }
    const na = (v) => (v === null || v === undefined ? "not available on this platform" : v);
    const mem = s.memory || {};
    const st = s.storage || {};
    const git = s.project.git;

    this.panel.innerHTML = "";
    this.panel.append(
      el("div", { class: "panel-head" },
        el("h3", {}, "Diagnostics"),
        el("span", { class: "note" }, `uptime ${F.dur(s.server.uptime_seconds)}`)),
      el("div", { class: "grid cols-2" },
        el("div", {},
          el("div", { class: "field-label", style: "margin-bottom:8px" }, "Project"),
          el("dl", { class: "kv" },
            el("dt", {}, "Version"), el("dd", {}, s.project.version),
            el("dt", {}, "Root"), el("dd", { class: "mono" }, s.project.root),
            el("dt", {}, "Git"), el("dd", {},
              git ? `${git.commit || "?"} (${git.branch || "?"})` : "not a git checkout"),
            el("dt", {}, "Python"), el("dd", {},
              `${s.python.version} ${s.python.implementation} · ${s.python.bits}-bit`),
            el("dt", {}, "Interpreter"), el("dd", { class: "mono" }, s.python.executable)),
          el("div", { class: "field-label", style: "margin:18px 0 8px" }, "Machine"),
          el("dl", { class: "kv" },
            el("dt", {}, "OS"), el("dd", {},
              `${s.os.system} ${s.os.release} (${s.os.machine})`),
            el("dt", {}, "CPU"), el("dd", {}, s.cpu.model || "—"),
            el("dt", {}, "Cores"), el("dd", {}, F.n(s.cpu.count)),
            el("dt", {}, "RAM total"), el("dd", {},
              mem.total_bytes ? F.bytes(mem.total_bytes) : na(null)),
            el("dt", {}, "RAM available"), el("dd", {},
              mem.available_bytes ? F.bytes(mem.available_bytes) : na(null)))),
        el("div", {},
          el("div", { class: "field-label", style: "margin-bottom:8px" }, "Server"),
          el("dl", { class: "kv" },
            el("dt", {}, "PID"), el("dd", {}, F.n(s.server.pid)),
            el("dt", {}, "Started"), el("dd", {}, F.date(s.server.started_at)),
            el("dt", {}, "Uptime"), el("dd", {}, F.dur(s.server.uptime_seconds)),
            el("dt", {}, "Workers use"), el("dd", {},
              s.server.worker_start_method,
              el("span", { class: "faint" },
                s.server.worker_start_method === "fork"
                  ? " — children inherit the parent's tables"
                  : " — each worker builds its tables once at startup"))),
          el("div", { class: "field-label", style: "margin:18px 0 8px" }, "Storage"),
          el("dl", { class: "kv" },
            el("dt", {}, "Checkpoints"), el("dd", {}, F.bytes(st.checkpoints_bytes)),
            el("dt", {}, "Training data"), el("dd", {}, F.bytes(st.data_bytes)),
            el("dt", {}, "Disk free"), el("dd", {},
              st.free_bytes ? F.bytes(st.free_bytes) : na(null))))),

      el("div", { class: "panel-head", style: "margin-top:22px" },
        el("h3", {}, "Running jobs"),
        el("span", { class: "note" }, `${(s.jobs || []).length} active`)),
      (s.jobs || []).length
        ? el("div", { class: "table-wrap" },
            el("table", { class: "data" },
              el("thead", {}, el("tr", {},
                el("th", {}, "Job"), el("th", {}, "Type"),
                el("th", { class: "num" }, "PID"), el("th", {}, "State"))),
              el("tbody", {}, ...(s.jobs || []).map((j) => el("tr", {},
                el("td", {}, j.label), el("td", {}, j.type),
                el("td", { class: "num mono" }, F.n(j.pid)),
                el("td", {}, j.state))))))
        : el("div", { class: "faint" }, "No jobs running."));
  },
};

/* ================================================================ SETTINGS */
App.views.settings = {
  title: "Settings",
  subtitle: "Preferences for this control center",

  mount(root) {
    const s = App.settings;
    const theme = selectInput("s-theme", [
      { value: "dark", label: "Dark" },
      { value: "light", label: "Light" },
      { value: "system", label: "Match the system" },
    ], s.theme);
    theme.onchange = () => { App.applyTheme(theme.value); };
    const refresh = selectInput("s-refresh", [
      { value: 1000, label: "1 second" },
      { value: 2000, label: "2 seconds" },
      { value: 5000, label: "5 seconds" },
      { value: 10000, label: "10 seconds" },
    ], s.refresh_ms);
    const speed = selectInput("s-speed", SPEEDS, s.playback_speed);
    const evalGames = numberInput("s-eval", s.eval_games, { min: 1, max: 100000, step: 50 });
    const workers = numberInput("s-workers", s.workers, { min: 1, max: 64 });
    const confirmBox = el("input", { type: "checkbox", checked: s.confirm_destructive });
    const tipsBox = el("input", { type: "checkbox", checked: s.show_tooltips });
    const compactBox = el("input", { type: "checkbox", checked: s.compact_numbers });

    const save = el("button", { class: "btn btn-primary" }, "Save settings");
    const status = el("span", { class: "faint", style: "font-size:12.5px" });
    save.onclick = async () => {
      save.disabled = true;
      try {
        const r = await API.post("/api/settings", {
          settings: {
            theme: theme.value,
            refresh_ms: Number(refresh.value),
            playback_speed: Number(speed.value),
            eval_games: Number(evalGames.value),
            workers: Number(workers.value),
            confirm_destructive: confirmBox.checked,
            show_tooltips: tipsBox.checked,
            compact_numbers: compactBox.checked,
          },
        });
        App.settings = r.settings;
        App.applyTheme(App.settings.theme);
        App.startLive();
        status.textContent = "Saved.";
        Toast.ok("Settings saved");
      } catch (e) {
        Toast.error("Could not save", e.message);
      } finally { save.disabled = false; }
    };

    const reset = el("button", { class: "btn" }, "Restore defaults");
    reset.onclick = async () => {
      try {
        const d = await API.get("/api/settings");
        const r = await API.post("/api/settings", { settings: d.defaults });
        App.settings = r.settings;
        App.applyTheme(App.settings.theme);
        Toast.ok("Defaults restored");
        App.route();
      } catch (e) { Toast.error("Could not reset", e.message); }
    };

    root.append(
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "Appearance")),
        el("div", { class: "form-grid" },
          field("Theme", theme),
          field("Dashboard refresh", refresh,
            "how often to poll when live updates are unavailable"),
          field("Numbers", el("label", { class: "check" }, compactBox,
            el("span", {}, "Abbreviate large numbers (12.3k)")))),
        el("div", { class: "panel-head", style: "margin-top:22px" },
          el("h3", {}, "Defaults")),
        el("div", { class: "form-grid" },
          field("Playback speed", speed, "for the live game viewer"),
          field("Evaluation games", evalGames, "pre-filled on the Evaluate page"),
          field("Workers", workers,
            `${navigator.hardwareConcurrency || "?"} logical cores detected`)),
        el("div", { class: "panel-head", style: "margin-top:22px" },
          el("h3", {}, "Behaviour")),
        el("div", { class: "form-grid" },
          field("Confirmations", el("label", { class: "check" }, confirmBox,
            el("span", {}, "Ask before deleting a checkpoint"))),
          field("Help", el("label", { class: "check" }, tipsBox,
            el("span", {}, "Show explanations of ML terms")))),
        el("div", { class: "btn-row", style: "margin-top:20px" }, save, reset, status),
        el("div", { class: "faint", style: "margin-top:14px;font-size:12px" },
          "Settings are stored in data/ui-settings.json alongside your training " +
          "data. Nothing is sent anywhere, and no credentials are stored.")),

      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "Keyboard shortcuts")),
        el("dl", { class: "kv" },
          el("dt", {}, el("kbd", {}, "g"), " then ", el("kbd", {}, "o/t/p/e/c/k/l/s")),
          el("dd", {}, "Jump to a page"),
          el("dt", {}, el("kbd", {}, "↑ ↓ ← →"), " / ", el("kbd", {}, "WASD")),
          el("dd", {}, "Move, in a human game"),
          el("dt", {}, el("kbd", {}, "Space")), el("dd", {}, "Pause or resume a watched game"),
          el("dt", {}, el("kbd", {}, "?")), el("dd", {}, "Show the shortcut list"))),

      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "The command line still works")),
        el("div", { class: "dim", style: "font-size:13.5px;max-width:75ch" },
          "This control center drives the same code the CLI does. Everything " +
          "here is also available from a terminal, which is what you want for " +
          "scripting, headless machines and automation:"),
        el("pre", { class: "mono", style:
          "margin-top:12px;background:var(--panel-2);padding:12px 14px;border-radius:9px;overflow-x:auto" },
          "python3 train.py --resume --games 20000 --workers 2\n" +
          "python3 evaluate.py --compare random heuristic learned --games 200\n" +
          "python3 experiment.py --run baseline --games 5000\n" +
          "python3 train.py --check")));
  },
};
