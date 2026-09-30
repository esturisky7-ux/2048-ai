/* Logs, System and Settings — the three utility pages. */

/* ==================================================================== LOGS */
App.views.logs = {
  title: "Logs",
  subtitle: "What the control center and its jobs have been doing",

  mount(root) {
    this.limit = 100;
    this.level = "";
    const limitSel = selectInput("l-limit", [
      { value: 100, label: "Last 100" },
      { value: 250, label: "Last 250" },
      { value: 500, label: "Last 500" },
      { value: 1000, label: "Last 1000" },
    ], this.limit, { class: "control-sm", "aria-label": "How many events",
                     style: "width:120px" });
    limitSel.onchange = () => { this.limit = Number(limitSel.value); this.load(); };
    const levelSel = selectInput("l-level", [
      { value: "", label: "All levels" },
      { value: "info", label: "Info and above" },
      { value: "warn", label: "Warnings and errors" },
      { value: "error", label: "Errors only" },
    ], "", { class: "control-sm", "aria-label": "Lowest level shown", style: "width:170px" });
    levelSel.onchange = () => { this.level = levelSel.value; this.load(); };
    const clear = button("Clear", { variant: "outline", size: "sm", icon: "trash-2" });
    clear.onclick = async () => {
      const ok = !App.settings.confirm_destructive || await confirmDialog(
        "Clear the log?", "Recorded events are deleted. Training data is untouched.",
        { danger: true, confirmText: "Clear" });
      if (!ok) return;
      await API.post("/api/logs/clear", {}).catch((e) => Toast.error("Failed", e.message));
      this.load();
    };

    this.list = el("div", { class: "log-list", role: "log", "aria-live": "off" });
    root.append(card(
      cardHeader("Recent Events",
        "Training milestones, evaluations, job state changes and errors.",
        [limitSel, levelSel, clear]),
      cardContent(this.list)));

    this.load();
    // New events arrive on the live stream, so the page stays current.
    this._onEvents = (e) => {
      if (!this.list || !e.detail?.length) return;
      this.list.querySelector(".log-empty")?.remove();
      for (const ev of e.detail) {
        if (this.shown(ev)) this.list.prepend(this.line(ev));
      }
      while (this.list.children.length > this.limit) this.list.lastChild.remove();
    };
    document.addEventListener("log:events", this._onEvents);
  },

  unmount() { document.removeEventListener("log:events", this._onEvents); },

  /* Streamed events obey the level filter the list was loaded with. */
  shown(ev) {
    const rank = { debug: 0, info: 1, warn: 2, error: 3 };
    return !this.level || (rank[ev.level] ?? 1) >= (rank[this.level] ?? 0);
  },

  line(ev) {
    const extras = Object.entries(ev)
      .filter(([k]) => !["ts", "level", "message", "seq"].includes(k))
      .map(([k, v]) => `${k}=${v}`).join("  ");
    const variant = { error: "destructive", warn: "outline", debug: "outline" }[ev.level] || "secondary";
    return el("div", { class: "log-row" },
      el("span", { class: "log-ts", title: F.date(ev.ts) }, F.time(ev.ts)),
      el("span", {}, badge(ev.level, variant)),
      el("span", { class: "log-msg" }, ev.message,
        extras ? el("span", { class: "log-extra" }, extras) : null));
  },

  async load() {
    let data;
    try { data = await API.get("/api/logs", { limit: this.limit, level: this.level }); }
    catch (e) {
      this.list.replaceChildren(el("div", { class: "note log-empty" }, `Could not load: ${e.message}`));
      return;
    }
    const events = (data.events || []).slice().reverse();
    if (!events.length) {
      this.list.replaceChildren(el("div", { class: "note log-empty" }, "Nothing logged yet."));
      return;
    }
    this.list.replaceChildren(...events.map((ev) => this.line(ev)));
  },
};

/* ================================================================== SYSTEM */
App.views.system = {
  title: "System",
  subtitle: "Platform, versions, storage and running processes",

  mount(root) {
    this.root = root;
    this.panel = card();
    this.jobs = card();
    root.append(this.panel, this.jobs);
    this.load();
    this._timer = setInterval(() => this.load(), 8000);
  },

  unmount() { clearInterval(this._timer); },

  async load() {
    let s;
    try { s = await API.get("/api/system"); }
    catch (e) {
      fillCard(this.panel, { title: "Diagnostics" },
        el("div", { class: "note" }, `Could not load: ${e.message}`));
      return;
    }
    const na = (v) => (v === null || v === undefined ? "not available on this platform" : v);
    const mem = s.memory || {};
    const st = s.storage || {};
    const git = s.project.git;
    const section = (title, pairs) => el("div", { class: "stack stack-10" },
      sectionHead(title), kv(pairs));

    fillCard(this.panel, {
      title: "Diagnostics", description: "Refreshes every 8 seconds.",
      action: badge(`uptime ${F.dur(s.server.uptime_seconds)}`, "outline"),
    },
    el("div", { class: "auto-grid", style: "--min:340px;gap:28px 40px" },
      section("Project", [
        ["Version", s.project.version],
        ["Root", el("span", { class: "mono", style: "font-size:13px" }, s.project.root)],
        ["Git", git ? `${git.commit || "?"} (${git.branch || "?"})` : "not a git checkout"],
        ["Python", `${s.python.version} ${s.python.implementation} · ${s.python.bits}-bit`],
        ["Interpreter", el("span", { class: "mono", style: "font-size:13px" }, s.python.executable)],
      ]),
      section("Server", [
        ["PID", el("span", { class: "mono" }, String(s.server.pid))],
        ["Started", F.date(s.server.started_at)],
        ["Uptime", F.dur(s.server.uptime_seconds)],
        ["Workers use", [s.server.worker_start_method,
          el("span", { class: "muted" },
            s.server.worker_start_method === "fork"
              ? " — children inherit the parent's tables"
              : " — each worker builds its tables once at startup")]],
      ]),
      section("Machine", [
        ["OS", `${s.os.system} ${s.os.release} (${s.os.machine})`],
        ["CPU", s.cpu.model || "—"],
        ["Cores", F.n(s.cpu.count)],
        ["RAM total", mem.total_bytes ? F.bytes(mem.total_bytes) : na(null)],
        ["RAM available", mem.available_bytes ? F.bytes(mem.available_bytes) : na(null)],
      ]),
      section("Storage", [
        ["Checkpoints", F.bytes(st.checkpoints_bytes)],
        ["Training data", F.bytes(st.data_bytes)],
        ["Disk free", st.free_bytes ? F.bytes(st.free_bytes) : na(null)],
      ])));

    const jobs = s.jobs || [];
    fillCard(this.jobs, {
      title: "Running Jobs",
      action: jobs.length ? badge(`${jobs.length} active`, "outline") : null,
    },
    jobs.length
      ? table(["Job", "Type", ["PID", { num: true }], "State"],
          jobs.map((j) => el("tr", {},
            el("td", { class: "wrap" }, j.label), el("td", {}, j.type),
            el("td", { class: "num mono", style: "font-size:13px" }, j.pid ? String(j.pid) : "—"),
            el("td", {}, jobStateBadge(j.state)))))
      : el("div", { class: "note", style: "font-size:14px" }, "No jobs running."));
  },
};

/* ================================================================ SETTINGS */
App.views.settings = {
  title: "Settings",
  subtitle: "Preferences for this control center",

  mount(root) {
    const s = App.settings;
    const refresh = selectInput("s-refresh", [
      { value: 1000, label: "1 second" },
      { value: 2000, label: "2 seconds" },
      { value: 5000, label: "5 seconds" },
      { value: 10000, label: "10 seconds" },
    ], s.refresh_ms);
    const speed = selectInput("s-speed", SPEEDS, s.playback_speed);
    const evalGames = numberInput("s-eval", s.eval_games, { min: 1, max: 100000, step: 50 });
    const workers = numberInput("s-workers", s.workers, { min: 1, max: 64 });
    const confirmBox = checkbox("Ask before deleting a checkpoint",
      { checked: s.confirm_destructive, switchStyle: true });
    const tipsBox = checkbox("Show explanations of ML terms",
      { checked: s.show_tooltips, switchStyle: true });
    const compactBox = checkbox("Abbreviate large numbers (12.3k)",
      { checked: s.compact_numbers, switchStyle: true });

    const status = el("span", { class: "note", role: "status" });
    const save = button("Save Settings");
    save.onclick = async () => {
      save.disabled = true;
      try {
        const r = await API.post("/api/settings", {
          settings: {
            refresh_ms: Number(refresh.value),
            playback_speed: Number(speed.value),
            eval_games: Number(evalGames.value),
            workers: Number(workers.value),
            confirm_destructive: confirmBox.input.checked,
            show_tooltips: tipsBox.input.checked,
            compact_numbers: compactBox.input.checked,
          },
        });
        App.settings = r.settings;
        App.startLive();
        status.textContent = "Saved.";
        Toast.ok("Settings saved");
      } catch (e) {
        Toast.error("Could not save", e.message);
      } finally { save.disabled = false; }
    };

    const reset = button("Restore Defaults", { variant: "outline" });
    reset.onclick = async () => {
      try {
        const d = await API.get("/api/settings");
        const r = await API.post("/api/settings", { settings: d.defaults });
        App.settings = r.settings;
        Toast.ok("Defaults restored");
        App.route();
      } catch (e) { Toast.error("Could not reset", e.message); }
    };

    const group = (title, fields) => el("div", { class: "stack stack-14" },
      sectionHead(title), formGrid(fields));

    root.append(
      card(
        cardHeader("Preferences",
          "Stored in data/ui-settings.json alongside your training data. Nothing is " +
          "sent anywhere, and no credentials are stored."),
        cardContent(el("div", { class: "stack stack-24" },
          group("Appearance", [
            field("Dashboard refresh", refresh,
              "how often to poll when live updates are unavailable"),
            field("Numbers", compactBox.node),
          ]),
          el("hr", { class: "separator" }),
          group("Defaults", [
            field("Playback speed", speed, "for the live game viewer"),
            field("Evaluation games", evalGames, "pre-filled on the Evaluate page"),
            field("Workers", workers,
              `${navigator.hardwareConcurrency || "?"} logical cores detected`),
          ]),
          el("hr", { class: "separator" }),
          group("Behaviour", [
            field("Confirmations", confirmBox.node),
            field("Help", tipsBox.node),
          ]),
          el("div", { class: "inline" }, save, reset, status)))),

      el("div", { class: "row" },
        el("div", { class: "col-half" }, card(
          cardHeader("Keyboard Shortcuts"),
          cardContent(shortcutList()))),
        el("div", { class: "col-wide" }, card(
          cardHeader("The Command Line Still Works",
            "Everything here is also available from a terminal — for scripting, " +
            "headless machines and automation."),
          cardContent(el("pre", { class: "code" },
            "python3 train.py --resume --games 20000 --workers 2\n" +
            "python3 evaluate.py --compare random heuristic learned --games 200\n" +
            "python3 experiment.py --run baseline --games 5000\n" +
            "python3 train.py --check"))))));
  },
};
