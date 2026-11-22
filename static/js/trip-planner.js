// =============================================================================
// PLAN MY TRIP FOR ME (uses plan-dom.js, then route-optimizer.js / ai-review.js)
// =============================================================================
// Adds a "✨ Plan my trip" button next to "Optimize Route". The traveler picks how
// many days per city (the cities of this page), interests and pace; the server
// (POST /plan-trip) builds the days from the database. The plan is written into
// the day boxes, then the route optimizer runs on it so the schedule and map show.
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
  const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const store = {
    get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
    // The pace is a plain string, as ai-review.js stores it
    getRaw(key) { try { return localStorage.getItem(key); } catch { return null; } },
    setRaw(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
  };

  // ---------------------------------------------------------------------------
  // The page's cities and day boxes
  // ---------------------------------------------------------------------------

  /** [{city: "tokyo", container}] in page order ("#tokyo-days", or "#days" on the Tokyo-only page) */
  function pageCities() {
    const out = [];
    document.querySelectorAll("[id$='-days']").forEach((el) => {
      const city = el.id.replace(/-days$/, "");
      if (/^[a-z]+$/.test(city)) out.push({ city, container: el });
    });
    const single = document.getElementById("days");
    if (!out.length && single) out.push({ city: (window.ROUTE_DEFAULT_CITY || "tokyo").toLowerCase(), container: single, single: true });
    return out;
  }

  const boxes = (entry) => [...entry.container.querySelectorAll(".day-box")];

  function addDay(entry) {
    if (typeof window.addDay !== "function") return false;
    if (entry.single) window.addDay();
    else window.addDay(entry.city);
    return true;
  }

  /** Make a city section have exactly n day boxes (at least one stays, empty) */
  function setDayCount(entry, n) {
    let list = boxes(entry);
    while (list.length > Math.max(n, 1)) {
      const last = list[list.length - 1];
      const remove = last.querySelector("button[onclick*='emove']");
      if (remove) remove.click();
      else last.remove();
      const next = boxes(entry);
      if (next.length === list.length) break;  // couldn't remove: stop instead of looping
      list = next;
    }
    while (boxes(entry).length < n) {
      if (!addDay(entry)) break;
    }
  }

  function daysOfCity(city) {
    return window.PlanDom.collectDays().filter((d) => d.city.toLowerCase() === city);
  }

  function pageHasActivities() {
    return window.PlanDom.collectDays().some((d) => d.stops.length);
  }

  /** Write the generated plan into the day boxes */
  function fillPage(plan, counts) {
    for (const entry of pageCities()) {
      const planned = plan.days.filter((d) => d.city_key === entry.city);
      setDayCount(entry, counts[entry.city] ? planned.length || counts[entry.city] : 0);
      const pageDays = daysOfCity(entry.city);
      pageDays.forEach((day, i) => {
        const p = planned[i];
        window.PlanDom.writeDay(day.key, p ? p.stops.map((s) => ({ name: s.name, duration: s.duration })) : []);
        window.PlanDom.writeArea(day.key, p ? p.area : "");
      });
    }
  }

  // ---------------------------------------------------------------------------
  // The form
  // ---------------------------------------------------------------------------

  function formHtml(cities, interests, saved) {
    const dayInputs = cities.map(({ city }, i) => {
      const existing = daysOfCity(city).filter((d) => d.stops.length).length;
      const value = saved?.days?.[city] ?? (existing || (i === 0 ? 3 : 2));
      return `
        <label class="flex items-center justify-between gap-2 rounded-lg border border-slate-200 px-3 py-2">
          <span class="text-sm font-medium text-slate-800">${esc(capitalize(city))}</span>
          <span class="flex items-center gap-1 text-sm text-slate-600">
            <input type="number" min="0" max="14" value="${esc(value)}" data-plan-days="${esc(city)}"
              class="w-16 rounded-md border border-slate-300 px-2 py-1 text-right focus:outline-none focus:ring-2 focus:ring-violet-300" />
            days</span>
        </label>`;
    }).join("");
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
        <h4 class="text-sm font-semibold text-slate-700">Days per city <span class="font-normal text-slate-500">(0 to skip a city)</span></h4>
        <div class="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">${dayInputs}</div>
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
    const days = {};
    panel.querySelectorAll("[data-plan-days]").forEach((el) => {
      days[el.dataset.planDays] = Math.max(0, Math.min(14, parseInt(el.value, 10) || 0));
    });
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
      panel.innerHTML = formHtml(pageCities(), interests, store.get(FORM_STORAGE_KEY));
      styleToggles(panel);
      bindPanel(panel);
      panel.dataset.ready = "1";
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
  }

  async function build(panel) {
    const form = readForm(panel);
    const status = panel.querySelector("[data-plan-status]");
    const go = panel.querySelector("[data-plan-go]");
    const cities = pageCities()
      .map(({ city }) => ({ city, days: form.days[city] || 0 }))
      .filter((c) => c.days > 0);
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

      fillPage(plan, form.days);
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
    if (!optimizeBtn || !window.PlanDom || document.getElementById("plan-trip-btn")) return;

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
