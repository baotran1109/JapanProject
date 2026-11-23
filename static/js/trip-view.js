// =============================================================================
// TRIP VIEW (view.html): a read-only, printable trip
// =============================================================================
// view.html?share=<id>   a shared trip (GET /share/<id>)
// view.html?local=1      the trip just opened from the planner's "Print / PDF"
// Shows each day's timed schedule (in the traveler's order), the map, weather
// and closed days (with a start date), plus calendar download and printing.
// =============================================================================

(function () {
  "use strict";

  const { esc, formatMinutes, yen } = window.PlanDom;
  const { CITY_NAMES } = window.TripExport;
  const root = document.getElementById("trip-view");
  const MODE_ICON = { walk: "🚶", train: "🚃", "bus/car": "🚌", "long-distance": "🚄" };

  function fail(text) {
    root.innerHTML = `<div class="rounded-xl bg-white p-6 text-center shadow-sm">
      <p class="text-slate-700">${esc(text)}</p>
      <a href="/plan.html" class="mt-3 inline-block rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700">Plan a trip</a></div>`;
  }

  async function loadTrip() {
    const params = new URLSearchParams(location.search);
    if (params.get("share")) {
      const res = await fetch(`/share/${encodeURIComponent(params.get("share"))}`);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "This link doesn't work anymore.");
      return { trip: data.trip, shared: true };
    }
    if (params.get("local")) {
      const trip = JSON.parse(localStorage.getItem("tripViewLocal") || "null");
      if (!trip) throw new Error("Open this page from the planner's Print button.");
      return { trip, shared: false };
    }
    throw new Error("No trip to show.");
  }

  function dateOf(trip, i) {
    if (!trip.start_date) return null;
    const d = new Date(`${trip.start_date}T00:00:00`);
    d.setDate(d.getDate() + i);
    return d;
  }
  const longDate = (d) => d.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });
  const shortDate = (d) => d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

  async function checks(trip, labels) {
    if (!trip.start_date) return null;
    try {
      const res = await fetch("/trip-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_date: trip.start_date,
          days: trip.days.map((d, i) => ({ key: `d${i}`, label: labels[i], city: CITY_NAMES[d.city], area: d.area, stops: d.stops })),
        }),
      });
      return res.ok ? await res.json() : null;
    } catch {
      return null;
    }
  }

  function stopRow(stop, city) {
    const travel = stop.travel_minutes
      ? `<div class="ml-[4.5rem] border-l-2 border-dashed border-slate-200 py-1 pl-3 text-xs text-slate-400">
           ${MODE_ICON[stop.mode] || "➡️"} ${formatMinutes(stop.travel_minutes)}${stop.fare_yen ? ` · ${yen(stop.fare_yen)}` : ""}</div>` : "";
    return `${travel}
      <li class="flex gap-3">
        <div class="w-16 shrink-0 pt-0.5 text-right text-sm font-semibold tabular-nums text-slate-700">${esc(stop.arrive)}</div>
        <div class="min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2">
          <div class="font-medium text-slate-900">
            <a href="${esc(window.TripExport.mapsLink(stop, city))}" target="_blank" rel="noopener" class="hover:underline">${esc(stop.name)}</a>
          </div>
          <div class="text-xs text-slate-500">${esc(stop.arrive)}–${esc(stop.depart)} · ${formatMinutes(stop.duration)}</div>
        </div>
      </li>`;
  }

  function weatherLine(info) {
    const w = info?.weather;
    if (!w) return "";
    return `<span class="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-700">${w.icon} ${esc(w.summary)}
      ${w.temp_min != null ? ` · ${w.temp_min}–${w.temp_max}°C` : ""} · ${w.rain_chance}% rain${w.source === "typical" ? " (typical)" : ""}</span>`;
  }

  function render(trip, sched, check, shared) {
    const labels = window.TripExport.labels(trip);
    const route = [];
    trip.days.forEach((d) => { if (route[route.length - 1] !== d.city) route.push(d.city); });
    const first = dateOf(trip, 0), last = dateOf(trip, trip.days.length - 1);
    const stops = trip.days.reduce((n, d) => n + d.stops.length, 0);
    const cost = sched?.cost_estimate;

    const dayCards = trip.days.map((day, i) => {
      const s = sched?.days[i];
      const info = check?.days?.[i];
      const date = dateOf(trip, i);
      const leg = sched?.intercity?.find((l) => l.after_day_label === s?.label);
      const closures = (info?.closures || []).map((c) => `<li>⚠️ ${esc(c.stop)} is ${esc(c.why)}</li>`).join("");
      return `
        <section class="print-day rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5">
          <div class="flex flex-wrap items-baseline justify-between gap-2">
            <h2 class="text-lg font-semibold">${esc(labels[i])}${date ? ` <span class="font-normal text-slate-500">· ${longDate(date)}</span>` : ""}</h2>
            ${weatherLine(info)}
          </div>
          ${day.area ? `<p class="text-sm text-slate-500">Starting from ${esc(day.area)}</p>` : ""}
          ${closures ? `<ul class="mt-1 text-xs text-amber-800">${closures}</ul>` : ""}
          ${s && s.stops.length ? `<ol class="mt-3 space-y-1">${s.stops.map((st) => stopRow(st, s.city)).join("")}</ol>
             <p class="mt-2 text-xs text-slate-500">Ends around ${esc(s.end_time)} · ${formatMinutes(s.travel_minutes)} of travel</p>`
            : `<p class="mt-2 text-sm text-slate-400">Free day</p>`}
          ${leg ? `<div class="mt-3 rounded-lg bg-indigo-50 px-3 py-2 text-sm text-indigo-900">
              ${leg.mode === "flight" ? "✈️ Flight" : "🚄 Train"} to ${esc(leg.to_city)} · about ${formatMinutes(leg.minutes)} · ${yen(leg.fare_yen)}</div>` : ""}
        </section>`;
    }).join("");

    root.innerHTML = `
      <div class="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <h1 class="text-2xl font-bold tracking-tight">${esc(trip.title || "Japan trip")}</h1>
        <p class="mt-1 text-slate-600">${route.map((c) => esc(CITY_NAMES[c] || c)).join(" → ")}</p>
        <p class="text-sm text-slate-500">${trip.days.length} day${trip.days.length > 1 ? "s" : ""} · ${stops} stops
          ${first ? ` · ${shortDate(first)} – ${shortDate(last)}` : ""}
          ${cost ? ` · about ${yen(cost.total_without_pass)} in local trains and travel between cities` : ""}</p>
        <div class="mt-4 flex flex-wrap gap-2" data-no-print>
          ${trip.start_date ? `<button type="button" data-view-calendar class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700">📅 Add to calendar</button>` : ""}
          <button type="button" data-view-print class="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-50">🖨️ Print / Save as PDF</button>
          ${shared ? `<button type="button" data-view-copy class="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-50">✏️ Copy to my planner</button>` : ""}
        </div>
        ${cost?.jr_pass ? `<p class="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">🎫 ${esc(cost.jr_pass.recommendation)}</p>` : ""}
      </div>
      ${sched ? `<div class="mt-4" data-view-map></div>` : ""}
      <div class="mt-4 space-y-4">${dayCards}</div>
      ${check ? `<p class="mt-4 text-xs text-slate-400">${esc(check.note)} Weather marked “typical” is the average of the last 5 years for that date.</p>` : ""}`;

    root.querySelector("[data-view-print]")?.addEventListener("click", () => window.print());
    root.querySelector("[data-view-calendar]")?.addEventListener("click", async (e) => {
      try { await window.TripExport.downloadCalendar(trip); } catch (err) { alert(err.message); }
      e.target.textContent = "📅 Downloaded";
    });
    root.querySelector("[data-view-copy]")?.addEventListener("click", () => {
      let existing = null;
      try { existing = JSON.parse(localStorage.getItem("planDraftV2") || "null"); } catch { /* ignore */ }
      if (existing?.days?.some((d) => d.stops?.length) &&
          !confirm("This replaces the unsaved trip in your planner. Continue?")) return;
      try {
        localStorage.setItem("planDraftV2", JSON.stringify({
          title: trip.title, startDate: trip.start_date || "", days: trip.days, savedAt: Date.now(),
        }));
      } catch { /* ignore */ }
      location.href = "/plan.html";
    });
    const mapBox = root.querySelector("[data-view-map]");
    if (mapBox && window.RouteMap) window.RouteMap.render(mapBox, sched, { compare: false }).catch(() => mapBox.remove());
  }

  (async () => {
    let loaded;
    try {
      loaded = await loadTrip();
    } catch (e) {
      fail(e.message);
      return;
    }
    const { trip, shared } = loaded;
    document.title = `${trip.title || "Japan trip"} · Explore Japan`;
    const labels = window.TripExport.labels(trip);
    const [sched, check] = await Promise.all([
      window.TripExport.schedule(trip).catch((e) => { console.error("Schedule failed:", e); return null; }),
      checks(trip, labels),
    ]);
    render(trip, sched, check, shared);
  })();
})();
