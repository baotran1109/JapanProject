// =============================================================================
// TRIP PLANNER (plan.html): one page for any route, e.g. Tokyo → Kyoto → Osaka
// =============================================================================
// The trip lives here in JavaScript (not in the page's inputs):
//   state = { title, startDate: "2027-04-06" | "", days: [{id, city, area, stops: [{name, duration}]}] }
// Consecutive days in the same city are shown together as one "leg".
//
// It plugs into PlanDom (plan-dom.js), so the route optimizer, AI review,
// "Plan my trip" and chat edits all read and write this trip.
// Saving to your account is in planner-cloud.js; drafts are kept in localStorage.
// =============================================================================

(function () {
  "use strict";

  const CITIES = {
    tokyo: { name: "Tokyo", icon: "🗼" },
    kyoto: { name: "Kyoto", icon: "⛩️" },
    osaka: { name: "Osaka", icon: "🏯" },
    fukuoka: { name: "Fukuoka", icon: "🍜" },
    okinawa: { name: "Okinawa", icon: "🏝️" },
    aomori: { name: "Aomori", icon: "🍎" },
    hakodate: { name: "Hakodate", icon: "🌉" },
    sapporo: { name: "Sapporo", icon: "❄️" },
  };
  const CITY_ORDER = Object.keys(CITIES);
  const DRAFT_KEY = "planDraftV2";
  const MAX_DAYS = 60;
  const MAX_UNDO = 30;
  const KIND_LABEL = { attractions: "Sight", daytrips: "Day trip", shopping: "Shop", restaurants: "Food" };
  const KIND_ICON = { attractions: "📍", daytrips: "🚆", shopping: "🛍️", restaurants: "🍽️" };

  const { esc } = window.PlanDom;
  const $ = (sel, root = document) => root.querySelector(sel);

  let nextId = 1;
  const state = { title: "", startDate: "", days: [] };
  let mode = "draft";              // "draft" (kept in this browser) or "cloud" (a saved itinerary)
  let activeKey = null;            // Day that "Add" in the place browser adds to
  let checks = null;               // Last /trip-check result
  const undoStack = [];
  const listeners = [];

  const store = {
    get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
  };

  // ---------------------------------------------------------------------------
  // Trip helpers
  // ---------------------------------------------------------------------------

  function makeDay(city, area = "", stops = []) {
    return {
      id: nextId++,
      city: CITIES[city] ? city : "tokyo",
      area: area || "",
      stops: (stops || []).filter((s) => s && String(s.name || "").trim())
        .map((s) => ({ name: String(s.name).trim(), duration: Number.isFinite(+s.duration) && s.duration !== null && s.duration !== "" ? +s.duration : null })),
    };
  }

  const keyOf = (day) => `day-${day.id}`;
  const indexOfKey = (key) => state.days.findIndex((d) => keyOf(d) === key);

  function labelOf(index) {
    const day = state.days[index];
    const n = state.days.slice(0, index + 1).filter((d) => d.city === day.city).length;
    return `${CITIES[day.city].name} · Day ${n}`;
  }

  function dateOf(index) {
    if (!state.startDate) return null;
    const d = new Date(`${state.startDate}T00:00:00`);
    if (Number.isNaN(d.getTime())) return null;
    d.setDate(d.getDate() + index);
    return d;
  }

  const fmtDate = (d) => d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });

  /** Consecutive days in the same city: [{city, start, days: [indexes]}] */
  function legs() {
    const out = [];
    state.days.forEach((day, i) => {
      const last = out[out.length - 1];
      if (last && last.city === day.city) last.days.push(i);
      else out.push({ city: day.city, start: i, days: [i] });
    });
    return out;
  }

  function snapshot() {
    return JSON.stringify({ title: state.title, startDate: state.startDate, days: state.days });
  }

  function pushUndo() {
    undoStack.push(snapshot());
    if (undoStack.length > MAX_UNDO) undoStack.shift();
  }

  function undo() {
    const last = undoStack.pop();
    if (!last) return;
    const s = JSON.parse(last);
    state.title = s.title;
    state.startDate = s.startDate;
    state.days = s.days.map((d) => makeDay(d.city, d.area, d.stops));
    changed({ structure: true, skipUndo: true });
  }

  /** Call after any change. structure: re-draw the day list. */
  function changed({ structure = false } = {}) {
    if (structure) render();
    renderToolbarState();
    saveDraft();
    listeners.forEach((fn) => fn());
  }

  // ---------------------------------------------------------------------------
  // Loading trips (new format, older saved formats, drafts)
  // ---------------------------------------------------------------------------

  // The old pages named the day's area "location" (most) or "place" (Tokyo + Kansai + Fukuoka)
  const legacyArea = (entry) => String(entry.location ?? entry.place ?? entry.loc ?? "");

  function legacyDayStops(entry) {
    const text = String(entry.activities ?? entry.notes ?? entry.act ?? "");
    const names = text.split("\n").map((s) => s.trim()).filter(Boolean);
    const durations = String(entry.durations ?? "").split(",").map((v) => parseInt(v.trim(), 10));
    // Some old plans put the location first in the durations list
    const offset = legacyArea(entry) && durations.length === names.length + 1 ? 1 : 0;
    return names.map((name, i) => ({ name, duration: Number.isNaN(durations[i + offset]) ? null : durations[i + offset] }));
  }

  /**
   * Any saved plan -> {title, startDate, days}.
   *  - new:  {version: 2, startDate, days: [{city, area, stops}]}
   *  - old multi-city pages: {tokyo: [{day, location, activities}], kyoto: [...]}
   *  - old Tokyo page: [{day, location, activities}]
   */
  function fromSaved(data) {
    const plan = data?.plan ?? data;
    const title = data?.title || "";
    if (plan && plan.version === 2 && Array.isArray(plan.days)) {
      return { title, startDate: plan.startDate || "", days: plan.days.map((d) => makeDay(d.city, d.area, d.stops)) };
    }
    if (Array.isArray(plan)) {
      const city = String(data?.city || "tokyo").toLowerCase();
      return { title, startDate: "", days: plan.map((e) => makeDay(CITIES[city] ? city : "tokyo", legacyArea(e), legacyDayStops(e))) };
    }
    const days = [];
    if (plan && typeof plan === "object") {
      for (const city of CITY_ORDER) {
        for (const entry of Array.isArray(plan[city]) ? plan[city] : []) {
          days.push(makeDay(city, legacyArea(entry), legacyDayStops(entry)));
        }
      }
    }
    return { title, startDate: "", days };
  }

  /** Replace the whole trip */
  function load(trip, options = {}) {
    if (!options.skipUndo && state.days.some((d) => d.stops.length)) pushUndo();
    state.title = trip.title || "";
    state.startDate = trip.startDate || "";
    state.days = (trip.days || []).map((d) => makeDay(d.city, d.area, d.stops));
    if (!state.days.length) state.days = [makeDay("tokyo")];
    if (options.mode) mode = options.mode;
    checks = null;
    $("#trip-title").value = state.title;
    $("#trip-start").value = state.startDate;
    changed({ structure: true });
    loadPlaceLists();
  }

  function toSaved() {
    return {
      version: 2,
      startDate: state.startDate || null,
      days: state.days.map((d) => ({ city: d.city, area: d.area, stops: d.stops.map((s) => ({ name: s.name, duration: s.duration })) })),
    };
  }

  function saveDraft() {
    if (mode !== "draft") return;
    store.set(DRAFT_KEY, { title: state.title, startDate: state.startDate, days: toSaved().days, savedAt: Date.now() });
  }

  /** Draft from this page, or from one of the old planner pages (legacyKey) */
  function initialTrip() {
    const params = new URLSearchParams(location.search);
    const draft = store.get(DRAFT_KEY);
    const hasDraft = draft && (draft.days || []).some((d) => (d.stops || []).length);
    const presetCities = (params.get("cities") || "").split(",").map((c) => c.trim().toLowerCase()).filter((c) => CITIES[c]);

    const legacyKey = params.get("draft");
    if (!hasDraft && legacyKey && /^[A-Za-z]+$/.test(legacyKey)) {
      const old = store.get(legacyKey);
      if (old) {
        const trip = fromSaved({ plan: old });
        if (trip.days.some((d) => d.stops.length)) return trip;
      }
    }
    if (presetCities.length && (!hasDraft || confirm("Start a new trip for " +
        presetCities.map((c) => CITIES[c].name).join(" → ") + "? Your unsaved draft will be replaced."))) {
      return { title: "", startDate: "", days: presetCities.map((c) => makeDay(c)) };
    }
    if (draft && draft.days?.length) return { title: draft.title, startDate: draft.startDate, days: draft.days.map((d) => makeDay(d.city, d.area, d.stops)) };
    return { title: "", startDate: "", days: [makeDay("tokyo")] };
  }

  // ---------------------------------------------------------------------------
  // Edits
  // ---------------------------------------------------------------------------

  function addDayAfter(index, city) {
    if (state.days.length >= MAX_DAYS) return alert(`A trip can have up to ${MAX_DAYS} days.`);
    pushUndo();
    state.days.splice(index + 1, 0, makeDay(city || state.days[index]?.city || "tokyo"));
    activeKey = keyOf(state.days[index + 1]);
    changed({ structure: true });
  }

  function addCity(city) {
    if (state.days.length >= MAX_DAYS) return alert(`A trip can have up to ${MAX_DAYS} days.`);
    pushUndo();
    // An empty trip's only empty day is replaced instead of kept
    if (state.days.length === 1 && !state.days[0].stops.length && !state.days[0].area) state.days = [];
    state.days.push(makeDay(city));
    activeKey = keyOf(state.days[state.days.length - 1]);
    changed({ structure: true });
    loadPlaceLists();
  }

  function removeDay(index) {
    const day = state.days[index];
    if (day.stops.length && !confirm(`Delete ${labelOf(index)} and its ${day.stops.length} stops?`)) return;
    pushUndo();
    state.days.splice(index, 1);
    if (!state.days.length) state.days.push(makeDay(day.city));
    changed({ structure: true });
  }

  function moveDay(index, delta) {
    const j = index + delta;
    if (j < 0 || j >= state.days.length || state.days[j].city !== state.days[index].city) return;
    pushUndo();
    [state.days[index], state.days[j]] = [state.days[j], state.days[index]];
    changed({ structure: true });
  }

  function moveLeg(legIndex, delta) {
    const all = legs();
    const target = legIndex + delta;
    if (target < 0 || target >= all.length) return;
    pushUndo();
    const blocks = all.map((leg) => leg.days.map((i) => state.days[i]));
    [blocks[legIndex], blocks[target]] = [blocks[target], blocks[legIndex]];
    state.days = blocks.flat();
    changed({ structure: true });
  }

  function removeLeg(legIndex) {
    const leg = legs()[legIndex];
    const stops = leg.days.reduce((n, i) => n + state.days[i].stops.length, 0);
    if (stops && !confirm(`Remove ${CITIES[leg.city].name} (${leg.days.length} day${leg.days.length > 1 ? "s" : ""}, ${stops} stops)?`)) return;
    pushUndo();
    state.days = state.days.filter((_, i) => !leg.days.includes(i));
    if (!state.days.length) state.days.push(makeDay("tokyo"));
    changed({ structure: true });
  }

  function addStops(index, names) {
    const day = state.days[index];
    const clean = names.map((n) => n.trim()).filter(Boolean);
    if (!clean.length) return;
    pushUndo();
    clean.forEach((name) => day.stops.push({ name: name.slice(0, 120), duration: null }));
    activeKey = keyOf(day);
    changed({ structure: true });
  }

  function moveStop(index, s, delta) {
    const stops = state.days[index].stops;
    const j = s + delta;
    if (j < 0 || j >= stops.length) return;
    pushUndo();
    [stops[s], stops[j]] = [stops[j], stops[s]];
    changed({ structure: true });
  }

  function removeStop(index, s) {
    pushUndo();
    state.days[index].stops.splice(s, 1);
    changed({ structure: true });
  }

  /** Swap the plans of two days (stops + area); their dates stay where they are */
  function swapDays(keyA, keyB) {
    const a = state.days[indexOfKey(keyA)], b = state.days[indexOfKey(keyB)];
    if (!a || !b) return false;
    pushUndo();
    [a.stops, b.stops] = [b.stops, a.stops];
    [a.area, b.area] = [b.area, a.area];
    changed({ structure: true });
    return true;
  }

  function moveStopBetween(fromKey, toKey, name) {
    const from = state.days[indexOfKey(fromKey)], to = state.days[indexOfKey(toKey)];
    const i = from ? from.stops.findIndex((s) => s.name === name) : -1;
    if (!to || i < 0) return false;
    pushUndo();
    to.stops.push(...from.stops.splice(i, 1));
    changed({ structure: true });
    return true;
  }

  // ---------------------------------------------------------------------------
  // PlanDom store: the optimizer, AI review, trip generator and chat use these
  // ---------------------------------------------------------------------------

  function dayForPanels(day, index) {
    return {
      key: keyOf(day),
      label: labelOf(index),
      city: CITIES[day.city].name,
      area: day.area,
      date: dateOf(index)?.toISOString().slice(0, 10) || null,
      stops: day.stops.map((s) => ({ name: s.name, duration: s.duration })),
    };
  }

  window.PlanDom.setStore({
    collectDays: () => state.days.map(dayForPanels),
    getDay(key) {
      const i = indexOfKey(key);
      return i < 0 ? null : dayForPanels(state.days[i], i);
    },
    writeDay(key, stops) {
      const i = indexOfKey(key);
      if (i < 0) return false;
      pushUndo();
      state.days[i].stops = makeDay("tokyo", "", stops).stops;
      changed({ structure: true });
      return true;
    },
    writeArea(key, text) {
      const i = indexOfKey(key);
      if (i < 0) return false;
      state.days[i].area = String(text || "").slice(0, 80);
      changed({ structure: true });
      return true;
    },
  });

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  function weatherChip(info) {
    const w = info?.weather;
    if (!w) return "";
    const temps = w.temp_min != null ? ` ${w.temp_min}–${w.temp_max}°` : "";
    const rain = w.rain_chance != null ? ` · ${w.rain_chance}% rain` : "";
    const title = w.source === "forecast" ? "Forecast" : "Typical weather for this date (last 5 years)";
    return `<span class="rounded-full ${info.rain_risk ? "bg-sky-100 text-sky-900" : "bg-slate-100 text-slate-700"} px-2 py-0.5 text-xs" title="${title}">
      ${w.icon} ${esc(w.summary)}${temps}${rain}${w.source === "typical" ? " (typical)" : ""}</span>`;
  }

  function stopRow(day, index, stop, s) {
    const key = keyOf(day);
    return `
      <li class="group flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-2 py-1.5" data-stop="${s}">
        <span class="w-5 shrink-0 text-center text-xs font-semibold text-slate-400">${s + 1}</span>
        <input data-field="stop-name" data-key="${key}" data-stop="${s}" value="${esc(stop.name)}" aria-label="Stop name"
          class="min-w-0 flex-1 rounded border border-transparent bg-transparent px-1 py-0.5 text-sm hover:border-slate-200 focus:border-slate-300 focus:outline-none" />
        <input data-field="stop-duration" data-key="${key}" data-stop="${s}" value="${stop.duration ?? ""}" inputmode="numeric"
          placeholder="auto" title="Minutes (blank = we estimate it)" aria-label="Minutes"
          class="w-14 shrink-0 rounded border border-slate-200 px-1.5 py-0.5 text-right text-xs focus:outline-none focus:ring-2 focus:ring-blue-200" />
        <span class="hidden text-xs text-slate-400 sm:inline">min</span>
        <div class="flex shrink-0 items-center">
          <button type="button" data-action="stop-up" data-key="${key}" data-stop="${s}" class="rounded px-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700" aria-label="Move up">↑</button>
          <button type="button" data-action="stop-down" data-key="${key}" data-stop="${s}" class="rounded px-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700" aria-label="Move down">↓</button>
          <button type="button" data-action="stop-remove" data-key="${key}" data-stop="${s}" class="rounded px-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600" aria-label="Remove">✕</button>
        </div>
      </li>`;
  }

  function dayCard(day, index) {
    const key = keyOf(day);
    const date = dateOf(index);
    const info = checks?.days?.find((d) => d.key === key);
    const closures = (info?.closures || []).map((c) =>
      `<li>⚠️ <strong>${esc(c.stop)}</strong> is ${esc(c.why)}</li>`).join("");
    const sameCityPrev = index > 0 && state.days[index - 1].city === day.city;
    const sameCityNext = index + 1 < state.days.length && state.days[index + 1].city === day.city;
    return `
      <article class="day-box rounded-xl border ${activeKey === key ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"} bg-slate-50/60 p-3 sm:p-4" data-key="${key}">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex flex-wrap items-center gap-2">
            <h3 class="font-semibold text-slate-900">${esc(labelOf(index))}</h3>
            ${date ? `<span class="text-sm text-slate-500">${fmtDate(date)}</span>` : ""}
            ${weatherChip(info)}
          </div>
          <div class="flex items-center gap-1 text-sm">
            <button type="button" data-action="day-up" data-key="${key}" ${sameCityPrev ? "" : "disabled"} class="rounded px-1.5 text-slate-500 hover:bg-slate-200 disabled:opacity-30" aria-label="Move day earlier">↑</button>
            <button type="button" data-action="day-down" data-key="${key}" ${sameCityNext ? "" : "disabled"} class="rounded px-1.5 text-slate-500 hover:bg-slate-200 disabled:opacity-30" aria-label="Move day later">↓</button>
            <button type="button" data-action="day-remove" data-key="${key}" class="rounded px-1.5 text-slate-500 hover:bg-rose-50 hover:text-rose-600" aria-label="Delete day">🗑</button>
          </div>
        </div>
        <input data-field="area" data-key="${key}" value="${esc(day.area)}" placeholder="Where the day starts (hotel or area, e.g. Shinjuku)"
          class="mt-2 w-full rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-200" />
        ${closures ? `<ul class="mt-2 space-y-0.5 text-xs text-amber-800">${closures}</ul>` : ""}
        <ol class="mt-2 space-y-1.5">${day.stops.map((s, k) => stopRow(day, index, s, k)).join("") ||
          `<li class="rounded-lg border-2 border-dashed border-slate-200 px-3 py-3 text-center text-xs text-slate-400">No stops yet: add places below or from the place list</li>`}</ol>
        <div class="mt-2 flex gap-2">
          <input data-field="add" data-key="${key}" list="places-${day.city}" placeholder="Add a place, “Lunch”, “Shopping in Ginza”…"
            class="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-200" />
          <button type="button" data-action="add-stop" data-key="${key}" class="shrink-0 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700">Add</button>
        </div>
      </article>`;
  }

  function render() {
    const root = $("#trip-days");
    const all = legs();
    root.innerHTML = all.map((leg, li) => `
      <section class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5" data-leg="${li}">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-lg font-semibold">${CITIES[leg.city].icon} ${CITIES[leg.city].name}
            <span class="text-sm font-normal text-slate-500">· ${leg.days.length} day${leg.days.length > 1 ? "s" : ""}</span></h2>
          <div class="flex items-center gap-1 text-sm">
            <button type="button" data-action="leg-up" data-leg="${li}" ${li ? "" : "disabled"} class="rounded px-2 py-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30" title="Visit earlier">↑</button>
            <button type="button" data-action="leg-down" data-leg="${li}" ${li < all.length - 1 ? "" : "disabled"} class="rounded px-2 py-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30" title="Visit later">↓</button>
            <button type="button" data-action="leg-remove" data-leg="${li}" class="rounded px-2 py-1 text-slate-500 hover:bg-rose-50 hover:text-rose-600">Remove</button>
          </div>
        </div>
        <div class="mt-3 space-y-3">${leg.days.map((i) => dayCard(state.days[i], i)).join("")}</div>
        <button type="button" data-action="add-day" data-index="${leg.days[leg.days.length - 1]}"
          class="mt-3 w-full rounded-lg border border-dashed border-blue-300 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50">
          + Add a ${CITIES[leg.city].name} day</button>
      </section>
      ${li < all.length - 1 ? `<div class="text-center text-sm text-slate-400">↓ travel to ${CITIES[all[li + 1].city].name}</div>` : ""}`).join("");
    renderSummary();
    renderBrowserTarget();
  }

  function renderSummary() {
    const el = $("#trip-summary");
    if (!el) return;
    const cities = legs().map((l) => CITIES[l.city].name);
    const stops = state.days.reduce((n, d) => n + d.stops.length, 0);
    const end = dateOf(state.days.length - 1);
    el.textContent = `${state.days.length} day${state.days.length > 1 ? "s" : ""} · ${cities.join(" → ")} · ${stops} stops` +
      (state.startDate && end ? ` · ${fmtDate(dateOf(0))} – ${fmtDate(end)}` : "");
  }

  function renderToolbarState() {
    const undoBtn = $("#undo-btn");
    if (undoBtn) undoBtn.disabled = !undoStack.length;
    renderSummary();
  }

  // ---------------------------------------------------------------------------
  // Place browser (right column): search the database and add to a day
  // ---------------------------------------------------------------------------

  const placeLists = {};      // city -> [places]
  const browser = { city: null, kind: "all", query: "" };

  async function loadPlaceList(city) {
    if (placeLists[city]) return placeLists[city];
    placeLists[city] = [];
    try {
      const res = await fetch(`/places/${city}`);
      if (!res.ok) throw new Error(res.status);
      placeLists[city] = (await res.json()).places || [];
    } catch (e) {
      console.warn("Couldn't load places for", city, e);
      delete placeLists[city];
      return [];
    }
    // Suggestions for the "Add" boxes of this city's days
    let list = document.getElementById(`places-${city}`);
    if (!list) {
      list = document.createElement("datalist");
      list.id = `places-${city}`;
      document.body.appendChild(list);
    }
    list.innerHTML = placeLists[city].map((p) =>
      `<option value="${esc(p.name)}">${esc([KIND_LABEL[p.kind], p.area].filter(Boolean).join(" · "))}</option>`).join("");
    return placeLists[city];
  }

  function loadPlaceLists() {
    const cities = [...new Set(state.days.map((d) => d.city))];
    if (!browser.city || !cities.includes(browser.city)) browser.city = cities[0];
    cities.forEach((c) => loadPlaceList(c).then(() => { if (c === browser.city) renderBrowser(); }));
    renderBrowser();
  }

  function renderBrowserTarget() {
    const select = $("#browser-target");
    if (!select) return;
    const keys = state.days.map(keyOf);
    if (!keys.includes(activeKey)) activeKey = keys.find((k, i) => state.days[i].city === browser.city) || keys[0];
    select.innerHTML = state.days.map((d, i) =>
      `<option value="${keyOf(d)}" ${keyOf(d) === activeKey ? "selected" : ""}>${esc(labelOf(i))}</option>`).join("");
  }

  function renderBrowser() {
    const tabs = $("#browser-cities");
    const list = $("#browser-list");
    if (!tabs || !list) return;
    const cities = [...new Set(state.days.map((d) => d.city))];
    tabs.innerHTML = cities.map((c) => `
      <button type="button" data-browser-city="${c}" class="rounded-full px-3 py-1 text-xs font-semibold ${c === browser.city
        ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}">${CITIES[c].icon} ${CITIES[c].name}</button>`).join("");
    document.querySelectorAll("[data-browser-kind]").forEach((b) => {
      b.className = `rounded-md px-2 py-1 text-xs font-medium ${b.dataset.browserKind === browser.kind
        ? "bg-blue-600 text-white" : "bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50"}`;
    });
    const places = placeLists[browser.city];
    if (!places) {
      list.innerHTML = `<li class="py-6 text-center text-sm text-slate-400">Loading places…</li>`;
      return;
    }
    const q = browser.query.trim().toLowerCase();
    const shown = places.filter((p) => (browser.kind === "all" || p.kind === browser.kind) &&
      (!q || `${p.name} ${p.area || ""} ${p.cuisine || ""}`.toLowerCase().includes(q)));
    const inTrip = new Set(state.days.flatMap((d) => d.stops.map((s) => s.name.replace(/^(breakfast|brunch|lunch|dinner)\s*:\s*/i, "").toLowerCase())));
    list.innerHTML = shown.slice(0, 80).map((p) => {
      const meta = [KIND_LABEL[p.kind], p.area, p.cuisine, p.rating ? `★${p.rating}` : ""].filter(Boolean).join(" · ");
      const added = inTrip.has(p.name.toLowerCase());
      return `
        <li class="flex items-center justify-between gap-2 py-1.5">
          <div class="min-w-0">
            <div class="truncate text-sm font-medium text-slate-800">${KIND_ICON[p.kind] || "📍"} ${p.url
              ? `<a href="${esc(p.url)}" target="_blank" rel="noopener" class="hover:underline">${esc(p.name)}</a>` : esc(p.name)}</div>
            <div class="truncate text-xs text-slate-500">${esc(meta)}</div>
          </div>
          <button type="button" data-browser-add="${esc(p.name)}" data-kind="${p.kind}"
            class="shrink-0 rounded-md border px-2 py-0.5 text-xs font-semibold ${added
              ? "border-slate-200 text-slate-400" : "border-blue-300 text-blue-700 hover:bg-blue-50"}">${added ? "✓ In trip" : "+ Add"}</button>
        </li>`;
    }).join("") + (shown.length > 80 ? `<li class="py-2 text-center text-xs text-slate-400">${shown.length - 80} more: search to narrow down</li>` : "")
      || `<li class="py-6 text-center text-sm text-slate-400">No places match</li>`;
  }

  // ---------------------------------------------------------------------------
  // Dates, weather and closed days (/trip-check)
  // ---------------------------------------------------------------------------

  let checkTimer = null;
  let checkSeq = 0;

  function scheduleCheck(delay = 1500) {
    clearTimeout(checkTimer);
    if (!state.startDate || !state.days.some((d) => d.stops.length)) {
      checks = null;
      renderChecks();
      return;
    }
    checkTimer = setTimeout(runCheck, delay);
  }

  async function runCheck() {
    const seq = ++checkSeq;
    const status = $("#check-status");
    if (status) status.textContent = "Checking weather and opening days…";
    try {
      const authHeaders = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
      const res = await fetch("/trip-check", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({ start_date: state.startDate, days: window.PlanDom.forServer(window.PlanDom.collectDays()) }),
      });
      const data = await res.json().catch(() => ({}));
      if (seq !== checkSeq) return;  // An older check finished after a newer one started
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      checks = data;
      if (status) status.textContent = "";
    } catch (e) {
      if (seq !== checkSeq) return;
      checks = null;
      if (status) status.textContent = `Couldn't check the dates: ${e.message}`;
    }
    render();
    renderChecks();
  }

  function renderChecks() {
    const box = $("#check-panel");
    if (!box) return;
    if (!state.startDate) {
      box.innerHTML = `<p class="text-sm text-slate-500">Pick a start date to see each day's date, the weather and places that are closed that day.</p>`;
      return;
    }
    if (!checks) {
      box.innerHTML = `<p class="text-sm text-slate-500">${state.days.some((d) => d.stops.length) ? "Checking…" : "Add some stops to check them."}</p>`;
      return;
    }
    const rows = checks.suggestions.map((s, i) => `
      <li class="rounded-lg border border-slate-200 bg-white p-3">
        <div class="font-medium text-slate-900">${s.type === "tip" ? "☔" : s.type === "swap_days" ? "🔁" : "📅"} ${esc(s.title)}</div>
        <div class="mt-0.5 text-sm text-slate-600">${esc(s.reason)}</div>
        ${s.type !== "tip" ? `<button type="button" data-check-apply="${i}" class="mt-2 rounded-md bg-blue-600 px-3 py-1 text-xs font-semibold text-white hover:bg-blue-700">Apply</button>` : ""}
      </li>`).join("");
    const unavailable = checks.weather_unavailable?.length
      ? `<p class="mt-2 text-xs text-slate-500">Weather isn't available right now for ${esc(checks.weather_unavailable.join(", "))}.</p>` : "";
    box.innerHTML = (rows ? `<ul class="space-y-2">${rows}</ul>` : `<p class="text-sm text-emerald-700">✅ No rain problems or closed places found for these dates.</p>`)
      + unavailable + `<p class="mt-2 text-xs text-slate-400">${esc(checks.note)}</p>`;
  }

  function applyCheck(i) {
    const s = checks?.suggestions?.[i];
    if (!s) return;
    const ok = s.type === "swap_days" ? swapDays(s.day_key, s.with_day_key)
      : s.type === "move_stop" ? moveStopBetween(s.day_key, s.to_day_key, s.stop) : false;
    if (!ok) alert("That day changed since the check. It will be checked again.");
    scheduleCheck(300);
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------

  function bind() {
    const root = $("#trip-days");

    root.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-action]");
      if (!btn) {
        const box = e.target.closest(".day-box");
        if (box && box.dataset.key !== activeKey) { activeKey = box.dataset.key; renderBrowserTarget(); highlightActive(); }
        return;
      }
      const index = indexOfKey(btn.dataset.key);
      const s = Number(btn.dataset.stop);
      switch (btn.dataset.action) {
        case "add-stop": {
          const input = root.querySelector(`[data-field="add"][data-key="${btn.dataset.key}"]`);
          addStops(index, input.value.split("\n"));
          focusAdd(btn.dataset.key);
          break;
        }
        case "stop-up": moveStop(index, s, -1); break;
        case "stop-down": moveStop(index, s, 1); break;
        case "stop-remove": removeStop(index, s); break;
        case "day-up": moveDay(index, -1); break;
        case "day-down": moveDay(index, 1); break;
        case "day-remove": removeDay(index); break;
        case "add-day": addDayAfter(Number(btn.dataset.index)); break;
        case "leg-up": moveLeg(Number(btn.dataset.leg), -1); break;
        case "leg-down": moveLeg(Number(btn.dataset.leg), 1); break;
        case "leg-remove": removeLeg(Number(btn.dataset.leg)); break;
      }
      if (["add-stop", "stop-remove", "day-remove", "add-day", "leg-up", "leg-down", "leg-remove", "day-up", "day-down"].includes(btn.dataset.action)) scheduleCheck();
    });

    // Typing changes the trip without re-drawing (so the cursor stays put)
    let undoArmed = true;
    root.addEventListener("input", (e) => {
      const el = e.target;
      const index = indexOfKey(el.dataset.key);
      if (index < 0 || el.dataset.field === "add") return;
      if (undoArmed) { pushUndo(); undoArmed = false; }
      const day = state.days[index];
      if (el.dataset.field === "area") day.area = el.value.slice(0, 80);
      if (el.dataset.field === "stop-name") day.stops[Number(el.dataset.stop)].name = el.value.slice(0, 120);
      if (el.dataset.field === "stop-duration") {
        const n = parseInt(el.value, 10);
        day.stops[Number(el.dataset.stop)].duration = Number.isNaN(n) ? null : Math.max(5, Math.min(n, 600));
      }
      changed();
    });
    root.addEventListener("change", (e) => {
      undoArmed = true;
      const el = e.target;
      // An emptied stop name removes the stop
      if (el.dataset.field === "stop-name" && !el.value.trim()) {
        const index = indexOfKey(el.dataset.key);
        state.days[index].stops.splice(Number(el.dataset.stop), 1);
        changed({ structure: true });
      }
      if (["area", "stop-name", "stop-duration"].includes(el.dataset.field)) scheduleCheck();
    });

    root.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.dataset.field === "add") {
        e.preventDefault();
        const key = e.target.dataset.key;
        addStops(indexOfKey(key), e.target.value.split("\n"));
        focusAdd(key);
        scheduleCheck();
      }
    });
    // Pasting several lines adds several stops
    root.addEventListener("paste", (e) => {
      if (e.target.dataset.field !== "add") return;
      const text = e.clipboardData?.getData("text") || "";
      if (!text.includes("\n")) return;
      e.preventDefault();
      const key = e.target.dataset.key;
      addStops(indexOfKey(key), text.split(/\r?\n/));
      focusAdd(key);
      scheduleCheck();
    });

    $("#add-city-btn").addEventListener("click", () => {
      addCity($("#add-city-select").value);
      scheduleCheck();
    });
    $("#trip-title").addEventListener("input", (e) => { state.title = e.target.value.slice(0, 100); changed(); });
    $("#trip-start").addEventListener("change", (e) => {
      pushUndo();
      state.startDate = e.target.value || "";
      checks = null;
      changed({ structure: true });
      renderChecks();
      scheduleCheck(100);
    });
    $("#undo-btn").addEventListener("click", () => { undo(); scheduleCheck(); });

    // Place browser
    $("#browser-cities").addEventListener("click", (e) => {
      const b = e.target.closest("[data-browser-city]");
      if (!b) return;
      browser.city = b.dataset.browserCity;
      const firstDay = state.days.findIndex((d) => d.city === browser.city);
      const active = state.days[indexOfKey(activeKey)];
      if (!active || active.city !== browser.city) activeKey = keyOf(state.days[firstDay]);
      renderBrowserTarget();
      highlightActive();
      loadPlaceList(browser.city).then(renderBrowser);
      renderBrowser();
    });
    document.querySelectorAll("[data-browser-kind]").forEach((b) => b.addEventListener("click", () => {
      browser.kind = b.dataset.browserKind;
      renderBrowser();
    }));
    $("#browser-search").addEventListener("input", (e) => { browser.query = e.target.value; renderBrowser(); });
    $("#browser-target").addEventListener("change", (e) => { activeKey = e.target.value; highlightActive(); });
    $("#browser-list").addEventListener("click", (e) => {
      const b = e.target.closest("[data-browser-add]");
      if (!b) return;
      const index = indexOfKey(activeKey);
      if (index < 0) return;
      if (state.days[index].city !== browser.city &&
          !confirm(`${labelOf(index)} is in ${CITIES[state.days[index].city].name}. Add “${b.dataset.browserAdd}” there anyway?`)) return;
      // A restaurant added from the list becomes "Lunch: ..." / "Dinner: ..." when the day has no such meal yet
      let name = b.dataset.browserAdd;
      if (b.dataset.kind === "restaurants") {
        const stops = state.days[index].stops.map((s) => s.name.toLowerCase());
        const meal = !stops.some((s) => s.startsWith("lunch")) ? "Lunch" : !stops.some((s) => s.startsWith("dinner")) ? "Dinner" : null;
        if (meal) name = `${meal}: ${name}`;
      }
      addStops(index, [name]);
      renderBrowser();
      scheduleCheck();
    });

    $("#check-panel").addEventListener("click", (e) => {
      const b = e.target.closest("[data-check-apply]");
      if (b) applyCheck(Number(b.dataset.checkApply));
    });
  }

  function highlightActive() {
    document.querySelectorAll("#trip-days .day-box").forEach((box) => {
      const on = box.dataset.key === activeKey;
      box.classList.toggle("border-blue-300", on);
      box.classList.toggle("ring-2", on);
      box.classList.toggle("ring-blue-100", on);
      box.classList.toggle("border-slate-200", !on);
    });
  }

  function focusAdd(key) {
    const input = document.querySelector(`#trip-days [data-field="add"][data-key="${key}"]`);
    if (input) { input.value = ""; input.focus(); }
  }

  // ---------------------------------------------------------------------------
  // Public API (planner-cloud.js, trip-planner.js, chat-edit.js, trip-export.js)
  // ---------------------------------------------------------------------------

  window.Planner = {
    CITIES, CITY_ORDER,
    getTrip: () => ({ title: state.title, startDate: state.startDate, ...toSaved() }),
    /** For share / export: {title, start_date, days: [{city, area, stops}]} */
    exportTrip: () => ({ title: state.title, start_date: state.startDate || null, days: toSaved().days }),
    toSaved,
    fromSaved,
    load,
    setMode(m) { mode = m; },
    getMode: () => mode,
    clearDraft() { try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ } },
    /** Cities in trip order with their day counts: [{city, days}] */
    cityPlan: () => legs().reduce((acc, leg) => {
      const found = acc.find((c) => c.city === leg.city);
      if (found) found.days += leg.days.length; else acc.push({ city: leg.city, days: leg.days.length });
      return acc;
    }, []),
    /** Replace the days (e.g. from "Plan my trip"): [{city, area, stops}] */
    replaceDays(days) {
      pushUndo();
      state.days = days.map((d) => makeDay(d.city, d.area, d.stops));
      if (!state.days.length) state.days = [makeDay("tokyo")];
      checks = null;
      changed({ structure: true });
      loadPlaceLists();
      scheduleCheck(300);
    },
    swapDays, moveStopBetween,
    pushUndo, undo,
    onChange(fn) { listeners.push(fn); },
    recheck: () => scheduleCheck(200),
    get startDate() { return state.startDate; },
  };

  function init() {
    bind();
    const params = new URLSearchParams(location.search);
    if (params.get("itineraryId")) {
      // planner-cloud.js loads it once the user is signed in
      mode = "cloud";
      state.days = [makeDay("tokyo")];
      render();
      $("#trip-days").insertAdjacentHTML("afterbegin",
        `<p id="cloud-loading" class="rounded-lg bg-white p-4 text-sm text-slate-500 shadow-sm">Loading your saved trip…</p>`);
    } else {
      load(initialTrip(), { skipUndo: true, mode: "draft" });
      undoStack.length = 0;
      // Don't apply ?cities= / ?draft= again on reload
      if (params.has("cities") || params.has("draft")) history.replaceState(null, "", "/plan.html");
      renderToolbarState();
    }
    renderChecks();
    scheduleCheck(500);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
