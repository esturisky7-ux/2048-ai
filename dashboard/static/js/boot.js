/* Runs in <head>, before the first paint: puts back the sidebar state this
 * browser last used, so a reload never shows a sidebar that then snaps shut.
 */
(function () {
  var root = document.documentElement;
  try {
    if (localStorage.getItem("sidebarCollapsed") === "1") {
      root.classList.add("sidebar-collapsed");
    }
  } catch (_) { /* storage unavailable: the defaults are fine */ }
})();
