/* Shell wiring: the pieces of chrome that live outside any single view —
 * the sidebar (collapsible to an icon rail, or a sheet on small screens), the
 * version footer and the lost-server warning.
 *
 * Loaded last so every view has registered itself before boot() runs the
 * router (core.js waits for DOMContentLoaded, which fires after all of these).
 */

const MOBILE = matchMedia("(max-width: 767.98px)");

function closeMobileSidebar() {
  const sidebar = $("#sidebar");
  if (!sidebar || !sidebar.classList.contains("open")) return;
  sidebar.classList.remove("open");
  $(".scrim")?.remove();
  $("#menu-btn")?.setAttribute("aria-expanded", "false");
}

function toggleSidebar() {
  const sidebar = $("#sidebar");
  const btn = $("#menu-btn");
  if (MOBILE.matches) {
    // A sheet over the page, dismissed by the scrim, Escape or navigating.
    const opening = !sidebar.classList.contains("open");
    if (!opening) { closeMobileSidebar(); return; }
    sidebar.classList.add("open");
    btn.setAttribute("aria-expanded", "true");
    const scrim = el("div", { class: "scrim" });
    scrim.onclick = closeMobileSidebar;
    document.body.append(scrim);
    sidebar.querySelector(".nav-item.active, .nav-item")?.focus();
    return;
  }
  const collapsed = document.documentElement.classList.toggle("sidebar-collapsed");
  btn.setAttribute("aria-expanded", String(!collapsed));
  try { localStorage.setItem("sidebarCollapsed", collapsed ? "1" : "0"); } catch (_) { }
  hideTooltip();
}

/* Collapsed to icons, the nav items name themselves in a tooltip. */
let _tooltip = null, _tooltipFor = null;
function showTooltip(target, text) {
  if (_tooltipFor === target) return;
  hideTooltip();
  const r = target.getBoundingClientRect();
  _tooltip = el("div", { class: "tooltip", role: "tooltip" }, text);
  _tooltipFor = target;
  document.body.append(_tooltip);
  _tooltip.style.left = `${r.right + 8}px`;
  _tooltip.style.top = `${r.top + r.height / 2 - _tooltip.offsetHeight / 2}px`;
}
function hideTooltip() { _tooltip?.remove(); _tooltip = _tooltipFor = null; }

document.addEventListener("DOMContentLoaded", () => {
  const sidebar = $("#sidebar");
  const menuBtn = $("#menu-btn");
  menuBtn.append(icon("panel-left"));
  menuBtn.setAttribute("aria-expanded",
    String(!document.documentElement.classList.contains("sidebar-collapsed")));
  menuBtn.onclick = toggleSidebar;
  $("#sidebar-foot").prepend(icon("git-branch"));

  const railTip = (e) => {
    const item = e.target.closest(".nav-item, .brand");
    if (!item || MOBILE.matches ||
        !document.documentElement.classList.contains("sidebar-collapsed")) {
      hideTooltip();
      return;
    }
    showTooltip(item, item.dataset.tooltip || "Overview");
  };
  sidebar.addEventListener("pointerover", railTip);
  sidebar.addEventListener("focusin", railTip);
  sidebar.addEventListener("pointerleave", hideTooltip);
  sidebar.addEventListener("focusout", hideTooltip);
  sidebar.addEventListener("click", hideTooltip);

  // The sheet belongs to small screens only: widening the window closes it.
  MOBILE.addEventListener("change", () => closeMobileSidebar());

  document.addEventListener("keydown", (e) => {
    // Ctrl+B / ⌘B toggles the sidebar, as in most editors.
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === "b") {
      e.preventDefault();
      toggleSidebar();
    } else if (e.key === "Escape" && sidebar.classList.contains("open")) {
      closeMobileSidebar();
      menuBtn.focus();
    }
  });

  // Version from every status frame.
  const paintFoot = (version) => {
    if (version) $("#foot-version").textContent = `v${version}`;
  };
  App.onStatus((s) => paintFoot(s.version));
  API.get("/api/system").then((s) => {
    paintFoot(s.project?.version || App.status?.version);
  }).catch(() => { });

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
