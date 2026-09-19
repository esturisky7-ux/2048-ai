/* Checkpoints: what has been trained, and what to do with it. */

App.views.checkpoints = {
  title: "Checkpoints",
  subtitle: "Saved agents, their statistics and what you can do with them",

  mount(root) {
    this.root = root;
    this.panel = el("section", { class: "panel" });
    root.append(
      this.panel,
      el("section", { class: "panel" },
        el("div", { class: "panel-head" }, el("h3", {}, "About checkpoints")),
        el("div", { class: "dim", style: "font-size:13.5px;max-width:75ch" },
          "A checkpoint is the agent itself: a flat array of float32 weights " +
          "plus the metadata describing the run that produced it. Snapshots " +
          "are frozen copies taken during training, which let you measure an " +
          "agent against its own younger self on identical games. Checkpoints " +
          "are never committed to Git — the default network is 268 MB — so to " +
          "share one, attach the file to a GitHub Release.")));
    this.load();
  },

  async load() {
    this.panel.innerHTML = "";
    this.panel.append(el("div", { class: "panel-head" },
      el("h3", {}, "Saved checkpoints"),
      el("span", { class: "note skeleton" }, "loading")));
    let data;
    try { data = await API.get("/api/checkpoints"); }
    catch (e) {
      this.panel.innerHTML = "";
      this.panel.append(el("div", { class: "faint" }, `Could not load: ${e.message}`));
      return;
    }
    const items = data.checkpoints || [];
    this.panel.innerHTML = "";
    if (!items.length) {
      const go = el("button", { class: "btn btn-primary" }, "Start training");
      go.onclick = () => App.go("training");
      this.panel.append(el("div", { class: "empty" },
        el("div", { class: "big" }, "▤"),
        el("h3", {}, "No checkpoints yet"),
        el("div", {}, "Train an agent and it will appear here."),
        el("div", { class: "btn-row", style: "justify-content:center;margin-top:16px" }, go)));
      return;
    }

    // The strongest by evaluation score gets a marker, since that is usually
    // the one you actually want to use.
    let best = null;
    for (const c of items) {
      const m = c.eval?.mean_score;
      if (m && (!best || m > best.eval.mean_score)) best = c;
    }

    this.panel.append(el("div", { class: "panel-head" },
      el("h3", {}, "Saved checkpoints"),
      el("span", { class: "note" }, `${items.length} total`)));

    const rows = items.map((c) => this.row(c, best));
    this.panel.append(el("div", { class: "table-wrap" },
      el("table", { class: "data" },
        el("thead", {}, el("tr", {},
          el("th", {}, "Checkpoint"), el("th", { class: "num" }, "Games"),
          el("th", { class: "num" }, "Evaluation"), el("th", { class: "num" }, "Best tile"),
          el("th", { class: "num" }, "Size"), el("th", {}, "Saved"),
          el("th", {}, "Actions"))),
        el("tbody", {}, ...rows))));
  },

  row(c, best) {
    const isBest = best && best.id === c.id;
    const label = el("span", {}, c.label || (c.kind === "current" ? c.run : "snapshot"));
    const nameCell = el("td", {},
      el("div", { style: "display:flex;align-items:center;gap:7px;flex-wrap:wrap" },
        isBest ? el("span", { title: "Best evaluated checkpoint", style: "color:var(--accent)" }, "★") : null,
        label,
        el("span", { class: "tag" }, c.kind === "current" ? "current" : "snapshot"),
        c.tuple_set ? el("span", { class: "tag" }, c.tuple_set) : null),
      el("div", { class: "faint mono", style: "font-size:11px" }, c.id));

    const watch = el("button", { class: "btn btn-sm" }, "Watch");
    watch.onclick = () => { App.setRun(c.run); App.go("play/watch"); };
    const evaluate = el("button", { class: "btn btn-sm" }, "Evaluate");
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
    const resume = el("button", { class: "btn btn-sm" }, "Resume");
    resume.onclick = () => { App.setRun(c.run); App.go("training"); };
    const rename = el("button", { class: "btn btn-sm btn-ghost", title: "Label" }, "Label");
    rename.onclick = async () => {
      const value = prompt(`Label for ${c.id}`, c.label || "");
      if (value === null) return;
      try {
        await API.post("/api/checkpoints/label", { id: c.id, label: value.trim() });
        this.load();
      } catch (e) { Toast.error("Could not save the label", e.message); }
    };
    const del = el("button", { class: "btn btn-sm btn-danger" }, "Delete");
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
          el("div", { class: "faint", style: "font-size:11px" },
            `n=${F.n(c.eval.games)}`))
      : el("td", { class: "num faint" }, "—");

    return el("tr", {},
      nameCell,
      el("td", { class: "num" }, F.compact(c.games)),
      evalCell,
      el("td", { class: "num" }, c.best_tile ? F.n(c.best_tile) : "—"),
      el("td", { class: "num faint" }, F.bytes(c.size_bytes)),
      el("td", { class: "faint nowrap" }, c.saved_at ? F.ago(c.saved_at) : "—"),
      el("td", {}, el("div", { class: "btn-row" },
        watch, evaluate, c.kind === "current" ? resume : null, rename, del)));
  },
};
