// =============================================================================
// PLANNER: Share / Calendar / Print buttons, and "My places" (plan.html)
// =============================================================================
// Uses Planner (planner.js) and TripExport (trip-export.js).
// - Share: makes a read-only link anyone can open (view.html?share=...)
// - Calendar: downloads an .ics file (needs a start date)
// - Print / PDF: opens the trip as a clean day-by-day page (view.html?local=1)
// - My places: your own places (hotel...), matched first when planning
// =============================================================================

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const { esc } = window.PlanDom;
  const VIEW_KEY = "tripViewLocal";

  function tripOrWarn() {
    const trip = window.Planner.exportTrip();
    if (!trip.days.some((d) => d.stops.length)) {
      alert("Add some stops to your days first.");
      return null;
    }
    return trip;
  }

  async function withBusy(button, text, fn) {
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = text;
    try {
      await fn();
    } catch (e) {
      alert(e.message || e);
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  }

  function showShare(url) {
    const box = $("#share-result");
    box.classList.remove("hidden");
    box.innerHTML = `
      <div class="rounded-lg border border-blue-200 bg-blue-50 p-3">
        <div class="text-sm font-semibold text-blue-900">Your share link</div>
        <div class="mt-2 flex gap-2">
          <input readonly value="${esc(url)}" class="min-w-0 flex-1 rounded-md border border-blue-200 bg-white px-2 py-1.5 text-sm" data-share-url />
          <button type="button" data-share-copy class="shrink-0 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-blue-700">Copy</button>
          <a href="${esc(url)}" target="_blank" rel="noopener" class="shrink-0 rounded-md border border-blue-300 bg-white px-3 py-1.5 text-sm font-semibold text-blue-700 hover:bg-blue-100">Open</a>
        </div>
        <p class="mt-1 text-xs text-blue-800">Anyone with the link can view this version of the trip (no sign-in needed).
          After more changes, share again for a new link.</p>
      </div>`;
    box.querySelector("[data-share-copy]").addEventListener("click", async (e) => {
      const input = box.querySelector("[data-share-url]");
      try { await navigator.clipboard.writeText(input.value); } catch { input.select(); document.execCommand("copy"); }
      e.target.textContent = "Copied ✓";
    });
  }

  function init() {
    $("#share-btn")?.addEventListener("click", (e) => {
      const trip = tripOrWarn();
      if (trip) withBusy(e.currentTarget, "⏳ Sharing…", async () => showShare(await window.TripExport.share(trip)));
    });

    $("#calendar-btn")?.addEventListener("click", (e) => {
      const trip = tripOrWarn();
      if (!trip) return;
      if (!trip.start_date) {
        alert("Pick the trip's start date first, so each day gets a date in your calendar.");
        $("#trip-start").focus();
        return;
      }
      withBusy(e.currentTarget, "⏳ Preparing…", () => window.TripExport.downloadCalendar(trip));
    });

    $("#print-btn")?.addEventListener("click", () => {
      const trip = tripOrWarn();
      if (!trip) return;
      try { localStorage.setItem(VIEW_KEY, JSON.stringify(trip)); } catch { /* ignore */ }
      window.open("/view.html?local=1", "_blank", "noopener");
    });

    initMyPlaces();
  }

  // ---------------------------------------------------------------------------
  // My places (/user/places): private to the signed-in traveler
  // ---------------------------------------------------------------------------

  const CATEGORIES = {
    sight: "📍 Sight", food: "🍽️ Restaurant", cafe: "☕ Café / sweets",
    shop: "🛍️ Shop", stay: "🏨 Hotel / stay", other: "⭐ Other",
  };
  let mine = [];

  async function authed(url, options = {}) {
    const headers = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
    if (!headers.Authorization) throw new Error("Sign in to save your own places.");
    const res = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}), ...headers } });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  /** "35.6895, 139.6917" or a Google Maps link -> {lat, lng} */
  function parseCoords(text) {
    const t = String(text || "");
    const patterns = [/!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)/, /@(-?\d+\.\d+),(-?\d+\.\d+)/, /[?&](?:q|query|ll)=(-?\d+\.\d+),\s*(-?\d+\.\d+)/,
                      /(-?\d{1,3}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)/];
    for (const re of patterns) {
      const m = t.match(re);
      if (m) return { lat: parseFloat(m[1]), lng: parseFloat(m[2]) };
    }
    return null;
  }

  function renderMine() {
    const list = $("#my-places-list");
    if (!list) return;
    if (!mine.length) {
      list.innerHTML = `<li class="text-slate-500">No places yet.</li>`;
      return;
    }
    list.innerHTML = mine.map((p) => `
      <li class="rounded-md bg-white px-2 py-1.5 ring-1 ring-purple-100">
        <div class="flex items-center justify-between gap-2">
          <span class="min-w-0 truncate"><strong>${esc(p.name)}</strong> · ${esc(p.city)}</span>
          <button type="button" data-mine-delete="${esc(p.id)}" class="shrink-0 rounded px-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600" aria-label="Delete ${esc(p.name)}">✕</button>
        </div>
        <div class="mt-1 flex items-center gap-2">
          <select data-mine-category="${esc(p.id)}" aria-label="Category of ${esc(p.name)}" class="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]">
            ${Object.entries(CATEGORIES).map(([id, label]) => `<option value="${id}" ${p.category === id ? "selected" : ""}>${label}</option>`).join("")}
          </select>
          ${p.notes ? `<span class="truncate text-[11px] text-slate-500">${esc(p.notes)}</span>` : ""}
        </div>
      </li>`).join("");
  }

  async function loadMyPlaces() {
    try {
      mine = (await authed("/user/places")).places || [];
    } catch (e) {
      $("#my-places-list").innerHTML = `<li class="text-slate-500">${esc(e.message)}</li>`;
      mine = [];
    }
    renderMine();
    window.Planner.setMyPlaces(mine);
  }

  function initMyPlaces() {
    const form = $("#my-places-form");
    if (!form) return;
    $("#my-place-city").innerHTML = Object.entries(window.Planner.CITIES)
      .map(([key, c]) => `<option value="${key}">${c.icon} ${c.name}</option>`).join("");
    $("#my-places-toggle").addEventListener("click", () => form.classList.toggle("hidden"));

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const where = parseCoords($("#my-place-coords").value);
      if (!where) {
        alert("Paste the coordinates (e.g. 35.6895, 139.6917) or the place's Google Maps link.");
        return;
      }
      const button = form.querySelector("button[type=submit]");
      button.disabled = true;
      try {
        const { place } = await authed("/user/places", {
          method: "POST",
          body: JSON.stringify({
            name: $("#my-place-name").value, category: $("#my-place-category").value,
            city: $("#my-place-city").value, notes: $("#my-place-notes").value, ...where,
          }),
        });
        mine.push(place);
        form.reset();
        form.classList.add("hidden");
        renderMine();
        window.Planner.setMyPlaces(mine);
      } catch (err) {
        alert(err.message);
      } finally {
        button.disabled = false;
      }
    });

    $("#my-places-list").addEventListener("change", async (e) => {
      const id = e.target.dataset.mineCategory;
      if (!id) return;
      try {
        const { place } = await authed("/user/places", { method: "PATCH", body: JSON.stringify({ place_id: id, category: e.target.value }) });
        mine = mine.map((p) => (p.id === id ? place : p));
        window.Planner.setMyPlaces(mine);
      } catch (err) {
        alert(err.message);
        renderMine();
      }
    });
    $("#my-places-list").addEventListener("click", async (e) => {
      const id = e.target.closest("[data-mine-delete]")?.dataset.mineDelete;
      if (!id) return;
      const place = mine.find((p) => p.id === id);
      if (!confirm(`Delete “${place?.name}” from your places?`)) return;
      try {
        await authed("/user/places", { method: "DELETE", body: JSON.stringify({ place_id: id }) });
        mine = mine.filter((p) => p.id !== id);
        renderMine();
        window.Planner.setMyPlaces(mine);
      } catch (err) {
        alert(err.message);
      }
    });

    // Load once we know whether the traveler is signed in
    let tries = 0;
    const wait = setInterval(() => {
      tries++;
      if (window.currentUserId !== undefined || tries > 20) {
        clearInterval(wait);
        if (window.currentUserId) loadMyPlaces();
        else $("#my-places-list").innerHTML = `<li class="text-slate-500"><a href="/auth.html" class="font-semibold text-purple-800 hover:underline">Sign in</a> to save your own places.</li>`;
      }
    }, 250);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
