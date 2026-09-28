/* Runs in <head>, before the first paint: puts back the theme and the sidebar
 * state this browser last used, so a reload never flashes the wrong theme or
 * a sidebar that then snaps shut.
 *
 * The saved settings (data/ui-settings.json) stay the source of truth for the
 * theme; core.js applies them once they load and keeps this copy in step.
 */
(function () {
  var root = document.documentElement;
  try {
    var theme = localStorage.getItem("theme") || "dark";
    if (theme === "system") {
      theme = window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches
        ? "light" : "dark";
    }
    root.classList.toggle("dark", theme !== "light");
    if (localStorage.getItem("sidebarCollapsed") === "1") {
      root.classList.add("sidebar-collapsed");
    }
  } catch (_) { /* storage unavailable: the defaults are fine */ }
})();
