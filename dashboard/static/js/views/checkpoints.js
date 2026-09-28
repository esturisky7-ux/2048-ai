/* Checkpoints: what has been trained, and what to do with it. */

App.views.checkpoints = {
  title: "Checkpoints",
  subtitle: "Saved agents, their statistics and what you can do with them",

  mount(root) {
    this.root = root;
    this.panel = card();
    root.append(
      this.panel,
      card(cardHeader("About Checkpoints"),
        cardContent(prose(
          "A checkpoint is the agent itself: a flat array of float32 weights " +
          "plus the metadata describing the run that produced it. Snapshots " +
          "are frozen copies taken during training, which let you measure an " +
          "agent against its own younger self on identical games. Checkpoints " +
          "are never committed to Git — the default network is 268 MB — so to " +
          "share one, attach the file to a GitHub Release."))));
    this.load();
  },

  async load() {
    if (!this.panel.firstChild) {
      fillCard(this.panel, { title: "Saved Checkpoints", action: el("span",
        { class: "badge badge-outline skeleton", style: "width:56px" }, "loading") },
        el("div", { class: "skeleton", style: "height:120px" }));
    }
    let data;
    try { data = await API.get("/api/checkpoints"); }
    catch (e) {
      fillCard(this.panel, { title: "Saved Checkpoints" },
        el("div", { class: "note" }, `Could not load: ${e.message}`));
      return;
    }
    const items = data.checkpoints || [];
    if (!items.length) {
      fillCard(this.panel, {},
        emptyState("database", "No Checkpoints Yet", "Train an agent and it will appear here.",
          button("Start Training", { icon: "play", onClick: () => App.go("training") })));
      return;
    }

    // The strongest by evaluation score gets a marker, since that is usually
    // the one you actually want to use.
    let best = null;
    for (const c of items) {
      const m = c.eval?.mean_score;
      if (m && (!best || m > best.eval.mean_score)) best = c;
    }

    fillCard(this.panel, {
      title: "Saved Checkpoints",
      description: best ? "The starred checkpoint has the best evaluation score."
        : "Evaluate a checkpoint to see how strong it is.",
      action: badge(`${items.length} total`, "outline"),
    },
    table(["Checkpoint", ["Games", { num: true }], ["Evaluation", { num: true }],
           ["Best tile", { num: true }], ["Size", { num: true }], "Saved",
           ["Actions", { num: true }]],
      items.map((c) => this.row(c, best))));
  },

  row(c, best) {
    const isBest = best && best.id === c.id;
    const nameCell = el("td", {},
      el("div", { class: "cell-stack" },
        el("div", { class: "inline inline-6" },
          isBest ? el("span", { title: "Best evaluated checkpoint", style: "display:inline-flex" },
            icon("star", 14)) : null,
          el("span", { style: "font-weight:500" },
            c.label || (c.kind === "current" ? c.run : "snapshot")),
          badge(c.kind === "current" ? "current" : "snapshot", "outline"),
          c.tuple_set ? badge(c.tuple_set, "outline") : null),
        el("div", { class: "cell-sub mono" }, c.id)));

    const watch = button("Watch", { variant: "outline", size: "sm" });
    watch.onclick = () => {
      App.setRun(c.run);
      // A snapshot travels in the route, so the Play page can select it;
      // the current weights are that page's default already.
      App.go(c.kind === "snapshot"
        ? `play/watch/${encodeURIComponent(c.id)}` : "play/watch");
    };
    const evaluate = button("Evaluate", { variant: "outline", size: "sm" });
    evaluate.onclick = async () => {
      evaluate.disabled = true;
      try {
        const payload = { agent: "learned", games: App.settings.eval_games,
                          seed: 987654, run: c.run };
        if (c.kind === "snapshot") payload.checkpoint = c.id;
        await API.post("/api/evaluate", payload);
        Toast.ok("Evaluation started", `${c.id} · ${App.settings.eval_games} games`);
        App.go("evaluate");
      } catch (e) { Toast.error("Could not evaluate", e.message); }
      finally { evaluate.disabled = false; }
    };
    const resume = button("Resume", { variant: "outline", size: "sm",
      onClick: () => { App.setRun(c.run); App.go("training"); } });
    const rename = button("Label", { variant: "ghost", size: "sm", title: "Give it a label" });
    rename.onclick = async () => {
      const value = await promptDialog("Label this checkpoint",
        `A short name for ${c.id}, shown in place of its id. Leave empty to remove it.`,
        c.label || "", { placeholder: "e.g. before-alpha-change" });
      if (value === null) return;
      try {
        await API.post("/api/checkpoints/label", { id: c.id, label: value.trim() });
        this.load();
      } catch (e) { Toast.error("Could not save the label", e.message); }
    };
    const del = button("Delete", { variant: "ghost", size: "icon-sm", icon: "trash-2",
                                   title: `Delete ${c.id}` });
    del.onclick = async () => {
      const what = c.kind === "current"
        ? `the entire run “${c.run}” — its weights, history and every snapshot`
        : `the snapshot ${c.id}`;
      if (App.settings.confirm_destructive) {
        const ok = await confirmDialog("Delete this checkpoint?",
          `This permanently deletes ${what}. It cannot be undone.`,
          { danger: true, confirmText: "Delete" });
        if (!ok) return;
      }
      try {
        await API.post("/api/checkpoints/delete", { id: c.id, confirm: true });
        Toast.show("Deleted", c.id, "warn");
        this.load();
        App.refreshNow();
      } catch (e) { Toast.error("Could not delete", e.message); }
    };

    const evalCell = c.eval?.mean_score
      ? el("td", { class: "num" },
          el("div", {}, F.n(c.eval.mean_score, 0)),
          el("div", { class: "cell-sub" }, `n=${F.n(c.eval.games)}`))
      : el("td", { class: "num muted" }, "—");

    return el("tr", {},
      nameCell,
      el("td", { class: "num" }, F.compact(c.games)),
      evalCell,
      el("td", { class: "num" }, c.best_tile ? F.n(c.best_tile) : "—"),
      el("td", { class: "num muted" }, F.bytes(c.size_bytes)),
      el("td", { class: "muted" }, c.saved_at ? F.ago(c.saved_at) : "—"),
      el("td", {}, el("div", { class: "inline inline-6", style: "justify-content:flex-end;flex-wrap:nowrap" },
        watch, evaluate, c.kind === "current" ? resume : null, rename, del)));
  },
};
