// =============================================================================
// ROUTE OPTIMIZER (shared by all plan_*.html pages)
// =============================================================================
// - Reads every day box on the page (each day is optimized on its own)
// - Sends them to POST /optimize-route
// - Shows a timed schedule per day, travel between cities and JR Pass advice
// - "Apply" reorders the activity cards of a day to the optimized order
//
// Pages need plan-dom.js loaded first, plus:
//   <button id="optimize-route-btn">, <aside id="route-panel" class="hidden">
//   and optionally window.ROUTE_DEFAULT_CITY (used when day ids have no city prefix)
// =============================================================================

(function () {
  "use strict";

  let lastResult = null;   // Response from the server
  let lastDays = null;     // Days we sent (with DOM references for "Apply")

  const { esc, formatMinutes, yen } = window.PlanDom;

  // ---------------------------------------------------------------------------
  // Applying the optimized order back to the page
  // ---------------------------------------------------------------------------

  function applyDay(day, result) {
    if (!result || !result.changed) return false;
    return window.PlanDom.reorderDay(day.key, result.optimized_order);
  }

  function applyAll() {
    if (!lastResult || !lastDays) return;
    let changed = 0;
    lastResult.days.forEach((result, i) => { if (applyDay(lastDays[i], result)) changed++; });
    setStatus(changed ? `Applied the new order to ${changed} day${changed > 1 ? "s" : ""}. Don't forget to save your plan.`
                      : "Your days are already in the best order.");
    markApplied();
  }

  function markApplied(dayIndex) {
    document.querySelectorAll("#route-panel [data-apply-day]").forEach((btn) => {
      if (dayIndex === undefined || Number(btn.dataset.applyDay) === dayIndex) {
        btn.disabled = true;
        btn.textContent = "✓ Applied";
      }
    });
  }

  function setStatus(text) {
    const el = document.getElementById("route-status");
    if (el) el.textContent = text;
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  const MODE_ICON = { walk: "🚶", train: "🚃", "bus/car": "🚌", "long-distance": "🚄", shinkansen: "🚄", flight: "✈️" };
  const TYPE_BADGE = {
    breakfast: "🍳 Breakfast", lunch: "🍱 Lunch", dinner: "🍽️ Dinner", meal: "🍜 Meal", evening: "🌆 Evening",
    snack: "☕ Snack", shopping: "🛍️ Shopping",
  };

  function renderStop(stop, dayKey) {
    const travel = stop.travel_minutes != null
      ? `<div class="text-xs text-slate-500">${MODE_ICON[stop.mode] || "🚃"} ${formatMinutes(stop.travel_minutes)}
           · ${stop.distance_km} km${stop.fare_yen ? ` · ${yen(stop.fare_yen)}` : ""}</div>`
      : "";
    const free = stop.free_time_before
      ? `<div class="text-xs text-sky-700">☕ ${formatMinutes(stop.free_time_before)} free time before this</div>` : "";
    const matched = stop.matched_name && stop.matched_name !== stop.name
      ? `<div class="text-xs text-slate-400">📍 ${esc(stop.matched_name)}</div>` : "";
    const notOnMap = !stop.located
      ? `<span class="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-500">not on map</span>` : "";
    const badge = TYPE_BADGE[stop.type]
      ? `<span class="ml-1 rounded bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-700">${TYPE_BADGE[stop.type]}</span>` : "";

    return `
      <li class="flex gap-3">
        <div class="w-24 shrink-0 pt-0.5 text-right font-mono text-xs text-slate-500">${esc(stop.arrive)}–${esc(stop.depart)}</div>
        <div class="min-w-0 flex-1 border-l-2 border-green-200 pb-3 pl-3">
          ${travel}${free}
          <div class="font-medium text-slate-800">${esc(stop.name)}${badge}${notOnMap}</div>
          <div class="text-xs text-slate-400">⏱ ${formatMinutes(stop.duration)}${stop.duration_estimated ? " (estimated)" : ""}</div>
          ${matched}
          ${window.PlanDom.renderPicks(dayKey, stop.name, stop.type, stop.picks_near, stop.picks)}
        </div>
      </li>`;
  }

  function renderDay(result, index) {
    const saved = result.travel_saved_minutes;
    const timingFixed = result.timing_issues_before - result.timing_issues_after;
    const chips = [];
    if (!result.changed) chips.push(`<span class="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">Already in a good order</span>`);
    if (saved > 0) chips.push(`<span class="rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-800">−${formatMinutes(saved)} travel</span>`);
    if (result.changed && timingFixed > 0) chips.push(`<span class="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-800">Better meal / evening timing</span>`);

    const warnings = (result.warnings || []).map((w) =>
      `<li class="text-xs text-amber-800">⚠️ ${esc(w)}</li>`).join("");

    return `
      <section class="rounded-xl bg-white p-4 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h4 class="font-semibold text-slate-800">${esc(result.label)}</h4>
            <div class="text-xs text-slate-500">
              ${esc(result.start_time)}–${esc(result.end_time)} · travel ${formatMinutes(result.travel_minutes)}
              ${result.anchor ? ` · starts near ${esc(result.anchor)}` : ""}
            </div>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            ${chips.join("")}
            ${result.changed ? `<button type="button" data-apply-day="${index}"
                class="rounded-md border border-green-600 px-2.5 py-1 text-xs font-semibold text-green-700 hover:bg-green-50 disabled:opacity-60">
                Apply to this day</button>` : ""}
          </div>
        </div>
        ${warnings ? `<ul class="mt-2 space-y-1 rounded-lg bg-amber-50 p-2">${warnings}</ul>` : ""}
        <ol class="mt-3">${result.stops.map((stop) => renderStop(stop, result.key)).join("")}</ol>
      </section>`;
  }

  function renderIntercity(legs) {
    if (!legs.length) return "";
    const rows = legs.map((leg) => `
      <li class="flex flex-wrap justify-between gap-2 text-sm">
        <span>${MODE_ICON[leg.mode] || "🚄"} ${esc(leg.from_city)} → ${esc(leg.to_city)}
          <span class="text-xs text-slate-500">(${esc(leg.mode)}, ~${formatMinutes(leg.minutes)}${leg.jr ? ", JR Pass ok" : ""})</span></span>
        <span class="font-medium">${leg.estimated ? "~" : ""}${yen(leg.fare_yen)}</span>
      </li>`).join("");
    return `
      <div class="rounded-xl bg-white p-4 shadow-sm">
        <h4 class="font-semibold text-slate-700">🧳 Between cities</h4>
        <ul class="mt-2 space-y-1">${rows}</ul>
      </div>`;
  }

  function render(data) {
    const panel = document.getElementById("route-panel");
    if (!panel) return;
    const s = data.summary;
    const cost = data.cost_estimate;
    const pass = cost.jr_pass;
    const pct = s.original_travel_minutes ? Math.round((s.travel_saved_minutes / s.original_travel_minutes) * 100) : 0;
    const anyChanged = data.days.some((d) => d.changed);

    panel.innerHTML = `
      <div class="flex flex-wrap items-center justify-between gap-3">
        <h3 class="text-lg font-semibold text-green-900">🚀 Optimized Route</h3>
        <div class="text-sm text-green-800">
          ${s.travel_saved_minutes > 0
            ? `Saves <strong>${formatMinutes(s.travel_saved_minutes)}</strong> of travel (${pct}%)`
            : "Your travel time is already close to the best"}
          · ${s.days_changed}/${s.days} day${s.days > 1 ? "s" : ""} reordered
        </div>
      </div>

      <div class="mt-4 grid gap-4 md:grid-cols-2">
        <div class="rounded-xl bg-white p-4 shadow-sm">
          <h4 class="font-semibold text-slate-700">⏱️ Time</h4>
          <div class="mt-2 space-y-1 text-sm">
            <div class="flex justify-between"><span>Activities</span><span class="font-medium">${formatMinutes(s.activity_minutes)}</span></div>
            <div class="flex justify-between"><span>Travel (before → after)</span>
              <span class="font-medium">${formatMinutes(s.original_travel_minutes)} → ${formatMinutes(s.travel_minutes)}</span></div>
            ${s.unmatched ? `<div class="pt-1 text-xs text-slate-500">${s.unmatched} stop${s.unmatched > 1 ? "s" : ""} couldn't be found on the map.</div>` : ""}
          </div>
        </div>
        <div class="rounded-xl bg-white p-4 shadow-sm">
          <h4 class="font-semibold text-slate-700">💴 Transport cost (estimate)</h4>
          <div class="mt-2 space-y-1 text-sm">
            <div class="flex justify-between"><span>Local trains / buses</span><span class="font-medium">${yen(cost.local_fares_yen)}</span></div>
            <div class="flex justify-between"><span>Between cities</span><span class="font-medium">${yen(cost.intercity_fares_yen)}</span></div>
            <div class="flex justify-between border-t pt-1"><span class="font-semibold">Total</span><span class="font-bold">${yen(cost.total_without_pass)}</span></div>
            ${pass ? `<div class="mt-2 rounded p-2 text-xs ${pass.is_worth_it ? "bg-green-50 text-green-800" : "bg-yellow-50 text-yellow-800"}">
              <strong>${pass.is_worth_it ? "✅" : "💡"} ${esc(pass.recommendation)}</strong><br>
              ${esc(pass.pass_type)} pass ${yen(pass.pass_cost)} vs ${yen(pass.jr_fares_covered)} of JR fares
              (days ${pass.start_day}–${pass.end_day})
            </div>` : ""}
            <div class="text-[11px] text-slate-400">${esc(cost.note)}</div>
          </div>
        </div>
      </div>

      <div class="mt-4 space-y-4">
        ${renderIntercity(data.intercity)}
        ${data.days.map(renderDay).join("")}
      </div>

      <div class="mt-4 flex flex-wrap items-center gap-3">
        ${anyChanged ? `<button type="button" id="route-apply-all"
            class="rounded-lg bg-green-600 px-4 py-2 font-medium text-white hover:bg-green-700">✅ Apply new order to all days</button>` : ""}
        <span id="route-status" class="text-sm text-green-800"></span>
      </div>`;

    panel.classList.remove("hidden");
    panel.querySelector("#route-apply-all")?.addEventListener("click", applyAll);
    panel.querySelectorAll("[data-apply-day]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const i = Number(btn.dataset.applyDay);
        if (applyDay(lastDays[i], lastResult.days[i])) {
          markApplied(i);
          setStatus(`Applied to ${lastResult.days[i].label}. Don't forget to save your plan.`);
        }
      });
    });
    // "Use" a restaurant / shop: keep the optimized order pointing at the new name
    window.PlanDom.bindPicks(panel, (key, oldName, newName) => {
      const day = lastResult.days.find((d) => d.key === key);
      if (day) day.optimized_order = day.optimized_order.map((n) => (n === oldName ? newName : n));
      setStatus(`Added “${newName}” instead of “${oldName}”. Don't forget to save your plan.`);
    });
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------------------------------------------------------------------------
  // Button
  // ---------------------------------------------------------------------------

  async function optimize(button) {
    const days = window.PlanDom.collectDays();
    if (!days.some((d) => d.stops.length)) {
      alert("Add some activities to your days first.");
      return;
    }

    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = "⏳ Optimizing…";
    try {
      const authHeaders = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
      const res = await fetch("/optimize-route", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({
          start_time: "09:00",
          days: window.PlanDom.forServer(days),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      lastResult = data;
      lastDays = days;
      render(data);
    } catch (e) {
      console.error("Route optimization failed:", e);
      alert("Route optimization failed: " + (e?.message || e));
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  }

  function init() {
    const button = document.getElementById("optimize-route-btn");
    if (button) button.addEventListener("click", () => optimize(button));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  // Exposed for debugging in the browser console
  window.RouteOptimizer = { optimize: () => optimize(document.getElementById("optimize-route-btn")) };
})();
