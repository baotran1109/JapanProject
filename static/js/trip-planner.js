// =============================================================================
// PLAN MY TRIP FOR ME (plan.html: uses planner.js, then route-optimizer.js / ai-review.js)
// =============================================================================
// Adds a "✨ Plan my trip" button next to "Optimize Route". The traveler picks the
// route (cities in order, days in each), interests and pace; the server
// (POST /plan-trip) builds the days from the database. The plan replaces the
// planner's days, then the route optimizer runs on it so the schedule and map show.
// =============================================================================

(function () {
  "use strict";

  const PACES = { relaxed: "Relaxed", normal: "Normal", packed: "Packed" };
  const PACE_STORAGE_KEY = "aiReviewPace";       // shared with ai-review.js
  const FORM_STORAGE_KEY = "tripPlannerForm";
  // Shown if /plan-trip/options can't be loaded
  const DEFAULT_INTERESTS = {
    food: "Food", temples: "Temples & shrines", vintage: "Vintage shopping",
    fashion: "Fashion & designer shops", museums: "Museums & art", nature: "Parks & nature",
    views: "City views", anime: "Anime & games", history: "Castles & history",
    theme_parks: "Theme parks", daytrips: "Day trips",
  };
  const DEFAULT_PICKED = ["food", "temples"];

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
  const store = {
    get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
    // The pace is a plain string, as ai-review.js stores it
    getRaw(key) { try { return localStorage.getItem(key); } catch { return null; } },
    setRaw(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
  };

  // ---------------------------------------------------------------------------
  // The trip's cities (planner.js)
  // ---------------------------------------------------------------------------

  /**
   * Rows for the form, in travel order: the trip's cities first (with their day counts),
   * then the other cities with 0 days. [{city, name, icon, days}]
   */
  function cityRows() {
    const P = window.Planner;
    const trip = P.cityPlan();
    const hasStops = window.PlanDom.collectDays().some((d) => d.stops.length);
    const rows = trip.map((c, i) => ({ city: c.city, days: hasStops ? c.days : Math.max(c.days, i === 0 ? 3 : 2) }));
    for (const city of P.CITY_ORDER) {
      if (!rows.some((r) => r.city === city)) rows.push({ city, days: 0 });
    }
    return rows.map((r) => ({ ...r, name: P.CITIES[r.city].name, icon: P.CITIES[r.city].icon }));
  }

  function pageHasActivities() {
    return window.PlanDom.collectDays().some((d) => d.stops.length);
  }

  /** Put the generated plan into the planner */
  function fillPage(plan) {
    window.Planner.replaceDays(plan.days.map((d) => ({
      city: d.city_key, area: d.area, stops: d.stops.map((s) => ({ name: s.name, duration: s.duration })),
    })));
  }

  // ---------------------------------------------------------------------------
  // The form
  // ---------------------------------------------------------------------------

  function cityRowHtml(r) {
    return `
      <li class="flex items-center justify-between gap-2 rounded-lg border border-slate-200 px-2 py-1.5" data-plan-row="${esc(r.city)}">
        <span class="flex items-center gap-1">
          <button type="button" data-plan-move="-1" class="rounded px-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700" aria-label="Earlier">↑</button>
          <button type="button" data-plan-move="1" class="rounded px-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700" aria-label="Later">↓</button>
          <span class="text-sm font-medium text-slate-800">${r.icon} ${esc(r.name)}</span>
        </span>
        <span class="flex items-center gap-1 text-sm text-slate-600">
          <input type="number" min="0" max="14" value="${esc(r.days)}" data-plan-days="${esc(r.city)}"
            class="w-16 rounded-md border border-slate-300 px-2 py-1 text-right focus:outline-none focus:ring-2 focus:ring-violet-300" />
          days</span>
      </li>`;
  }

  function formHtml(rows, interests, saved) {
    const dayInputs = rows.map(cityRowHtml).join("");
    const picked = new Set(saved?.interests || DEFAULT_PICKED);
    // Keep a sensible order (the server's JSON comes back sorted alphabetically)
    const ids = [...Object.keys(DEFAULT_INTERESTS).filter((id) => id in interests),
                 ...Object.keys(interests).filter((id) => !(id in DEFAULT_INTERESTS))];
    const chips = ids.map((id) => [id, interests[id]]).map(([id, label]) => `
      <button type="button" data-plan-interest="${esc(id)}" aria-pressed="${picked.has(id)}"
        class="rounded-full border px-3 py-1 text-sm font-medium transition-colors">${esc(label)}</button>`).join("");
    const stored = store.getRaw(PACE_STORAGE_KEY);
    const pace = saved?.pace || (PACES[stored] ? stored : "normal");
    const paceButtons = Object.entries(PACES).map(([id, label]) => `
      <button type="button" data-plan-pace="${id}" aria-pressed="${id === pace}"
        class="px-3 py-1.5 text-sm font-semibold">${label}</button>`).join("");

    return `
      <div class="flex items-start justify-between gap-3">
        <div>
          <h3 class="text-lg font-semibold">✨ Plan my trip for me</h3>
          <p class="text-sm text-slate-500">Pick your days, interests and pace. We'll build every day from our place database,
            then schedule it with the route optimizer.</p>
        </div>
        <button type="button" data-plan-close class="shrink-0 rounded-md px-2 py-1 text-slate-500 hover:bg-slate-100" aria-label="Close">✕</button>
      </div>

      <div class="mt-4">
        <h4 class="text-sm font-semibold text-slate-700">Route and days <span class="font-normal text-slate-500">(top to bottom; 0 days skips a city)</span></h4>
        <ol class="mt-2 grid gap-1.5 sm:grid-cols-2" data-plan-rows>${dayInputs}</ol>
      </div>

      <div class="mt-4">
        <h4 class="text-sm font-semibold text-slate-700">Interests</h4>
        <div class="mt-2 flex flex-wrap gap-2">${chips}</div>
      </div>

      <div class="mt-4 flex flex-wrap items-center gap-3">
        <h4 class="text-sm font-semibold text-slate-700">Pace</h4>
        <div class="inline-flex overflow-hidden rounded-lg border border-slate-300">${paceButtons}</div>
      </div>

      <div class="mt-5 flex flex-wrap items-center gap-3">
        <button type="button" data-plan-go
          class="rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 px-5 py-2 font-semibold text-white hover:from-violet-700 hover:to-fuchsia-700 disabled:opacity-60">
          Build my plan</button>
        <span data-plan-status class="text-sm text-slate-500"></span>
      </div>
      <div data-plan-result></div>`;
  }

  function styleToggles(panel) {
    panel.querySelectorAll("[data-plan-interest]").forEach((b) => {
      const on = b.getAttribute("aria-pressed") === "true";
      b.className = `rounded-full border px-3 py-1 text-sm font-medium transition-colors ${on
        ? "border-violet-600 bg-violet-600 text-white" : "border-slate-300 bg-white text-slate-700 hover:bg-slate-50"}`;
    });
    panel.querySelectorAll("[data-plan-pace]").forEach((b) => {
      const on = b.getAttribute("aria-pressed") === "true";
      b.className = `px-3 py-1.5 text-sm font-semibold ${on ? "bg-violet-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`;
    });
  }

  function readForm(panel) {
    // Cities in the order shown (= travel order)
    const days = [...panel.querySelectorAll("[data-plan-days]")].map((el) => ({
      city: el.dataset.planDays, days: Math.max(0, Math.min(14, parseInt(el.value, 10) || 0)),
    }));
    const interests = [...panel.querySelectorAll("[data-plan-interest][aria-pressed='true']")].map((b) => b.dataset.planInterest);
    const pace = panel.querySelector("[data-plan-pace][aria-pressed='true']")?.dataset.planPace || "normal";
    return { days, interests, pace };
  }

  // ---------------------------------------------------------------------------
  // Result summary
  // ---------------------------------------------------------------------------

  const STOP_ICON = { lunch: "🍜", dinner: "🍽️", breakfast: "🥐", meal: "🍽️", snack: "☕", shopping: "🛍️", evening: "🌃" };

  function resultHtml(plan) {
    const fmt = window.PlanDom.formatMinutes;
    const score = plan.check?.score;
    const days = plan.days.map((d) => `
      <li class="rounded-lg border border-slate-200 p-3">
        <div class="flex flex-wrap items-baseline justify-between gap-x-3">
          <span class="font-semibold text-slate-900">${esc(d.label)}</span>
          <span class="text-xs text-slate-500">${esc(d.start_time)}–${esc(d.end_time)} · ${fmt(d.travel_minutes)} travel</span>
        </div>
        <div class="text-sm text-violet-700">${esc(d.theme)}</div>
        <div class="mt-1 text-xs text-slate-600">${d.stops.map((s) => `${STOP_ICON[s.type] || "📍"} ${esc(s.name)}`).join(" → ")}</div>
      </li>`).join("");
    const notes = (plan.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
    return `
      <div class="mt-5 border-t border-slate-200 pt-4">
        <div class="flex flex-wrap items-center gap-3">
          <span class="text-sm font-semibold text-slate-800">Plan added to your days below</span>
          ${score != null ? `<span class="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-800"
              title="Same checks as the AI review: timing, travel, meals, pace">Plan check ${esc(score)}/10</span>` : ""}
          <button type="button" data-plan-review
            class="rounded-lg border border-slate-300 bg-white px-3 py-1 text-sm font-semibold text-slate-800 hover:bg-slate-50">
            🤖 Ask AI to review it</button>
        </div>
        ${notes ? `<ul class="mt-2 list-disc pl-5 text-xs text-slate-500">${notes}</ul>` : ""}
        <ol class="mt-3 grid gap-2 md:grid-cols-2">${days}</ol>
        <p class="mt-2 text-xs text-slate-500">Edit anything you like: the optimizer and AI review work on your changes too.</p>
      </div>`;
  }

  // ---------------------------------------------------------------------------
  // Panel
  // ---------------------------------------------------------------------------

  let interestsPromise = null;
  function loadInterests() {
    interestsPromise = interestsPromise || fetch("/plan-trip/options")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => d.interests || DEFAULT_INTERESTS)
      .catch(() => { interestsPromise = null; return DEFAULT_INTERESTS; });
    return interestsPromise;
  }

  async function openPanel(panel) {
    if (!panel.dataset.ready) {
      const interests = await loadInterests();
      panel.innerHTML = formHtml(cityRows(), interests, store.get(FORM_STORAGE_KEY));
      styleToggles(panel);
      bindPanel(panel);
      panel.dataset.ready = "1";
    }
    else {
      const list = panel.querySelector("[data-plan-rows]");
      if (list) list.innerHTML = cityRows().map(cityRowHtml).join("");
    }
    panel.classList.remove("hidden");
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function bindPanel(panel) {
    panel.querySelector("[data-plan-close]").addEventListener("click", () => panel.classList.add("hidden"));
    panel.querySelectorAll("[data-plan-interest]").forEach((b) => b.addEventListener("click", () => {
      b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
      styleToggles(panel);
    }));
    panel.querySelectorAll("[data-plan-pace]").forEach((b) => b.addEventListener("click", () => {
      panel.querySelectorAll("[data-plan-pace]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      styleToggles(panel);
    }));
    panel.querySelector("[data-plan-go]").addEventListener("click", () => build(panel));
    panel.querySelector("[data-plan-rows]").addEventListener("click", (e) => {
      const b = e.target.closest("[data-plan-move]");
      if (!b) return;
      const row = b.closest("[data-plan-row]");
      const sibling = Number(b.dataset.planMove) < 0 ? row.previousElementSibling : row.nextElementSibling;
      if (!sibling) return;
      if (Number(b.dataset.planMove) < 0) sibling.before(row); else sibling.after(row);
    });
  }

  async function build(panel) {
    const form = readForm(panel);
    const status = panel.querySelector("[data-plan-status]");
    const go = panel.querySelector("[data-plan-go]");
    const cities = form.days.filter((c) => c.days > 0);
    if (!cities.length) {
      status.textContent = "Give at least one city a day or more.";
      return;
    }
    if (pageHasActivities() && !confirm("This replaces the activities already in your days. Continue?")) return;

    store.set(FORM_STORAGE_KEY, form);
    store.setRaw(PACE_STORAGE_KEY, form.pace);  // the AI review then checks the same pace
    go.disabled = true;
    status.textContent = "Building your plan…";
    try {
      const res = await fetch("/plan-trip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cities, interests: form.interests, pace: form.pace }),
      });
      const plan = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(plan.error || `Server error (${res.status})`);
      if (!plan.days?.length) throw new Error("We couldn't find enough places for that. Try other interests.");

      fillPage(plan);
      status.textContent = "";
      panel.querySelector("[data-plan-result]").innerHTML = resultHtml(plan);
      panel.querySelector("[data-plan-review]")?.addEventListener("click", () => {
        (document.getElementById("ask-ai-review") || document.getElementById("ask-ai"))?.click();
      });
      // Show the timed schedule, fares and map for the new plan
      setTimeout(() => window.RouteOptimizer?.optimize(), 50);
    } catch (e) {
      status.textContent = `❌ ${e.message}`;
    } finally {
      go.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Setup: button next to "Optimize Route" and an empty panel under the toolbar
  // ---------------------------------------------------------------------------

  function init() {
    const optimizeBtn = document.getElementById("optimize-route-btn");
    if (!optimizeBtn || !window.Planner || document.getElementById("plan-trip-btn")) return;

    const button = document.createElement("button");
    button.id = "plan-trip-btn";
    button.type = "button";
    button.className = "rounded-lg bg-gradient-to-r from-violet-600 to-fuchsia-600 px-4 py-2 font-semibold text-white hover:from-violet-700 hover:to-fuchsia-700";
    button.textContent = "✨ Plan my trip";
    optimizeBtn.insertAdjacentElement("afterend", button);

    const panel = document.createElement("section");
    panel.id = "plan-trip-panel";
    panel.className = "hidden rounded-2xl border border-violet-200 bg-white p-5 shadow-sm";
    optimizeBtn.parentElement.insertAdjacentElement("afterend", panel);

    button.addEventListener("click", () => {
      if (panel.classList.contains("hidden")) openPanel(panel);
      else panel.classList.add("hidden");
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
