// =============================================================================
// PLANNER LAYOUT (plan.html): tabs, getting-started card, Share & export menu
// =============================================================================
// Tabs: Itinerary (edit days) · Schedule & map · AI review · Chat, plus Places on
// phones (on wide screens the places list is always on the right).
// - Opening "Schedule & map" runs the route optimizer when the trip changed since
//   the last run, so it always shows the current plan.
// - New, empty trips get a card: "Plan it for me" or "Start from scratch".
// =============================================================================

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const TAB_KEY = "plannerTab";
  let current = "itinerary";
  let scheduleStale = true;     // Trip changed since the last optimizer run
  let welcomeDismissed = false;

  const hasStops = () => window.PlanDom.collectDays().some((d) => d.stops.length);

  function show(name) {
    const wide = window.matchMedia("(min-width: 1024px)").matches;
    // "Places" is only a tab on phones; on wide screens the places list is always visible
    if (!document.querySelector(`[data-tab="${name}"]`) || (name === "places" && wide)) name = "itinerary";
    current = name;
    document.querySelectorAll("[data-tab]").forEach((b) => {
      const on = b.dataset.tab === name;
      b.setAttribute("aria-selected", String(on));
      b.className = `shrink-0 whitespace-nowrap border-b-2 px-3 py-2.5 text-sm font-semibold transition-colors ${on
        ? "border-blue-600 text-blue-700" : "border-transparent text-slate-500 hover:border-slate-300 hover:text-slate-800"}`
        + (b.dataset.tab === "places" ? " lg:hidden" : "");
    });
    document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== name));
    // Phones: the sidebar (places, dates) is its own tab
    $("#planner-side").classList.toggle("hidden", name !== "places");
    try { sessionStorage.setItem(TAB_KEY, name); } catch { /* ignore */ }

    if (name === "schedule") refreshSchedule();
    if (name === "chat") setTimeout(() => $("[data-chat-input]")?.focus(), 50);
  }

  function refreshSchedule() {
    const empty = $("#schedule-empty");
    if (!hasStops()) {
      empty.classList.remove("hidden");
      $("#route-panel").classList.add("hidden");
      return;
    }
    empty.classList.add("hidden");
    if (scheduleStale && window.RouteOptimizer) {
      scheduleStale = false;
      window.RouteOptimizer.optimize();
    }
  }

  function renderWelcome() {
    const card = $("#welcome-card");
    if (!card) return;
    card.classList.toggle("hidden", welcomeDismissed || hasStops() || window.Planner.getMode() === "cloud");
  }

  function initShareMenu() {
    const menu = $("#export-menu");
    if (!menu) return;
    // Close the menu after choosing something, or when clicking elsewhere
    menu.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => menu.removeAttribute("open")));
    document.addEventListener("click", (e) => { if (!menu.contains(e.target)) menu.removeAttribute("open"); });
  }

  function init() {
    if (!window.Planner || !$("#planner-tabs")) return;

    document.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

    // Arrow keys move between tabs (accessible tab list)
    $("#planner-tabs").addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
      const tabs = [...document.querySelectorAll("[data-tab]")].filter((b) => b.offsetParent !== null);
      const i = tabs.findIndex((b) => b.dataset.tab === current);
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
      next.focus();
      show(next.dataset.tab);
    });

    window.Planner.onChange(() => {
      scheduleStale = true;
      renderWelcome();
      if (current === "schedule" && !hasStops()) refreshSchedule();
    });

    $("#welcome-plan")?.addEventListener("click", () => {
      welcomeDismissed = true;
      renderWelcome();
      window.TripGenerator?.open();
    });
    $("#welcome-scratch")?.addEventListener("click", () => {
      welcomeDismissed = true;
      renderWelcome();
      document.querySelector('#trip-days [data-field="add"]')?.focus();
    });
    $("#schedule-go-itinerary")?.addEventListener("click", () => show("itinerary"));

    // The optimize button lives in the Schedule tab; clicking it means "run again now"
    $("#optimize-route-btn")?.addEventListener("click", () => { scheduleStale = false; });

    initShareMenu();
    renderWelcome();

    let saved = null;
    try { saved = sessionStorage.getItem(TAB_KEY); } catch { /* ignore */ }
    show(saved && saved !== "places" ? saved : "itinerary");

    // Wide screen <-> phone: the sidebar is always shown on wide screens
    window.matchMedia("(min-width: 1024px)").addEventListener("change", (e) => {
      if (e.matches && current === "places") show("itinerary");
    });
  }

  window.PlannerLayout = { show, get current() { return current; } };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
