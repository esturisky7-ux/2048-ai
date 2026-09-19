/* Shell wiring: the pieces of chrome that live outside any single view.
 *
 * Loaded last so every view has registered itself before boot() runs the
 * router (core.js waits for DOMContentLoaded, which fires after all of these).
 */

document.addEventListener("DOMContentLoaded", () => {
  const sidebar = $("#sidebar");

  $("#menu-btn").onclick = () => {
    const opening = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", opening);
    $(".scrim")?.remove();
    if (opening) {
      const scrim = el("div", { class: "scrim" });
      scrim.onclick = () => { sidebar.classList.remove("open"); scrim.remove(); };
      document.body.append(scrim);
    }
  };

  $("#theme-toggle").onclick = async () => {
    const order = ["dark", "light", "system"];
    const next = order[(order.indexOf(App.settings.theme) + 1) % order.length];
    App.settings.theme = next;
    App.applyTheme(next);
    // Redraw charts: their colours come from CSS custom properties.
    requestAnimationFrame(() => $$(".chart-box").forEach((b) => b._chart?.draw()));
    try { await API.post("/api/settings", { settings: { theme: next } }); }
    catch (_) { /* the visual change already happened */ }
    Toast.show("Theme", next, "info", 1600);
  };

  App.onStatus((s) => {
    const v = $("#foot-version");
    if (v && s.version) v.textContent = `v${s.version}`;
  });

  // A long-running server that goes away (Ctrl-C) should say so rather than
  // leaving a silently stale page.
  let failures = 0;
  setInterval(async () => {
    try {
      await fetch("/api/agents", { cache: "no-store" });
      failures = 0;
    } catch (_) {
      if (++failures === 3) {
        Toast.error("Lost contact with the server",
          "Is it still running? Restart with: python3 server.py");
      }
    }
  }, 15000);
});
