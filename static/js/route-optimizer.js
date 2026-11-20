// =============================================================================
// ROUTE OPTIMIZER (shared by all plan_*.html pages)
// =============================================================================
// - Reads every day box on the page (each day is optimized on its own)
// - Sends them to POST /optimize-route
// - Shows a timed schedule per day, travel between cities and JR Pass advice
// - "Apply" reorders the activity cards of a day to the optimized order
//
// Pages only need:
//   <button id="optimize-route-btn">, <aside id="route-panel" class="hidden">
//   and optionally window.ROUTE_DEFAULT_CITY (used when day ids have no city prefix)
// =============================================================================

(function () {
  "use strict";

  let lastResult = null;   // Response from the server
  let lastDays = null;     // Days we sent (with DOM references for "Apply")

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
  const capitalize = (s) => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

  function formatMinutes(min) {
    min = Math.round(min || 0);
    if (min < 60) return `${min} min`;
    const h = Math.floor(min / 60), m = min % 60;
    return m ? `${h}h ${m}m` : `${h}h`;
  }
  const yen = (n) => `¥${Math.round(n || 0).toLocaleString()}`;

  // ---------------------------------------------------------------------------
  // Reading the plan from the page
  // ---------------------------------------------------------------------------

  // Matches "activities-3", "tokyo-activities-3", "kyoto-act-2", "osaka-notes-1", ...
  const ACTIVITY_ID = /^(?:([a-z]+)-)?(activities|act|notes)-(\d+)$/;

  function findActivityField(box) {
    for (const el of box.querySelectorAll("textarea[id]")) {
      const m = el.id.match(ACTIVITY_ID);
      if (m) return { el, prefix: m[1] || null, fieldType: m[2], n: Number(m[3]) };
    }
    return null;
  }

  function findInput(box, prefix, n, kinds) {
    for (const kind of kinds) {
      const id = prefix ? `${prefix}-${kind}-${n}` : `${kind}-${n}`;
      const el = box.querySelector(`#${CSS.escape(id)}`) || document.getElementById(id);
      if (el) return el;
    }
    return null;
  }

  function parseDurations(text, activityCount, hasLocation) {
    const values = (text || "").split(",").map((v) => parseInt(v.trim(), 10)).filter((v) => !Number.isNaN(v));
    // Older plans put the day's location first in the duration list
    const leadingLocation = hasLocation && values.length === activityCount + 1;
    const forActivities = leadingLocation ? values.slice(1) : values;
    return { values: forActivities, leadingLocation, leadingValue: leadingLocation ? values[0] : null };
  }

  function collectDays() {
    const defaultCity = window.ROUTE_DEFAULT_CITY || "Tokyo";
    const days = [];
    document.querySelectorAll(".day-box").forEach((box) => {
      const field = findActivityField(box);
      if (!field) return;

      const container = box.closest("[id$='-days']");
      const containerCity = container && container.id !== "days" ? container.id.replace(/-days$/, "") : null;
      const city = capitalize(field.prefix || containerCity || defaultCity);

      const locationEl = findInput(box, field.prefix, field.n, ["location", "loc", "place"]);
      const durationsEl = findInput(box, field.prefix, field.n, ["durations", "dur", "time"]);
      const area = (locationEl?.value || "").trim();

      const activities = (field.el.value || "").split("\n").map((a) => a.trim()).filter(Boolean);
      const durations = parseDurations(durationsEl?.value, activities.length, Boolean(area));

      days.push({
        key: field.el.id,
        label: `${city} · Day ${field.n}`,
        city,
        area,
        stops: activities.map((name, i) => ({ name, duration: durations.values[i] || 60 })),
        // Kept for "Apply" (not sent to the server)
        _field: field,
        _durationsEl: durationsEl,
        _durations: durations,
      });
    });
    return days;
  }

  // ---------------------------------------------------------------------------
  // Applying the optimized order back to the page
  // ---------------------------------------------------------------------------

  function applyDay(day, result) {
    if (!result || !result.changed) return false;
    const { el, prefix, fieldType, n } = day._field;
    const order = result.optimized_order;

    // Reorder durations the same way (only when the user filled them in)
    if (day._durationsEl && day._durations.values.length) {
      const byName = new Map();
      day.stops.forEach((s, i) => {
        const list = byName.get(s.name) || [];
        list.push(day._durations.values[i] ?? s.duration);
        byName.set(s.name, list);
      });
      const reordered = order.map((name) => (byName.get(name) || []).shift() ?? 60);
      if (day._durations.leadingLocation) reordered.unshift(day._durations.leadingValue);
      day._durationsEl.value = reordered.join(", ");
      day._durationsEl.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // Use the page's own helper so the activity cards and autosave update
    if (typeof window.setActivities === "function") {
      if (prefix) window.setActivities(prefix, n, order, fieldType);
      else window.setActivities(n, order);
    } else {
      el.value = order.join("\n");
      el.dispatchEvent(new Event("input", { bubbles: true }));
      if (typeof window.syncTextareaToCards === "function") {
        if (prefix) window.syncTextareaToCards(prefix, n, fieldType);
        else window.syncTextareaToCards(n);
      }
    }
    return true;
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
  };

  function renderStop(stop) {
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
          ${matched}
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
        <ol class="mt-3">${result.stops.map(renderStop).join("")}</ol>
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
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------------------------------------------------------------------------
  // Button
  // ---------------------------------------------------------------------------

  async function optimize(button) {
    const days = collectDays();
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
          days: days.map(({ key, label, city, area, stops }) => ({ key, label, city, area, stops })),
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
  window.RouteOptimizer = { collectDays, optimize: () => optimize(document.getElementById("optimize-route-btn")) };
})();
