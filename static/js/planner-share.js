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
  // My places (/user/places, needs sign-in)
  // ---------------------------------------------------------------------------

  async function authed(url, options = {}) {
    const headers = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
    if (!headers.Authorization) throw new Error("Sign in to use your own places.");
    return fetch(url, { ...options, headers: { ...(options.headers || {}), ...headers } });
  }

  async function loadMyPlaces() {
    const list = $("#my-places-list");
    if (!list) return;
    try {
      const res = await authed("/user/places");
      const data = await res.json();
      list.innerHTML = (data.places || []).map((p) => `
        <li class="flex items-center justify-between gap-2 rounded-md bg-white px-2 py-1 ring-1 ring-purple-100">
          <span class="truncate"><strong>${esc(p.name)}</strong> · ${esc(p.city)}</span>
        </li>`).join("") || `<li class="text-slate-500">No places yet.</li>`;
    } catch (e) {
      list.innerHTML = `<li class="text-slate-500">${esc(e.message)}</li>`;
    }
  }

  function initMyPlaces() {
    const form = $("#my-places-form");
    if (!form) return;
    $("#my-place-city").innerHTML = Object.entries(window.Planner.CITIES)
      .map(([key, c]) => `<option value="${c.name}">${c.icon} ${c.name}</option>`).join("");
    $("#my-places-toggle").addEventListener("click", () => form.classList.toggle("hidden"));
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const place = {
        name: $("#my-place-name").value.trim(),
        lat: parseFloat($("#my-place-lat").value),
        lng: parseFloat($("#my-place-lng").value),
        city: $("#my-place-city").value,
        category: "custom",
      };
      if (!place.name || Number.isNaN(place.lat) || Number.isNaN(place.lng)) {
        alert("Fill in a name and valid coordinates.");
        return;
      }
      try {
        const res = await authed("/user/places", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(place),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || "Couldn't save the place");
        form.reset();
        form.classList.add("hidden");
        loadMyPlaces();
      } catch (err) {
        alert(err.message);
      }
    });
    // Load once we know whether the user is signed in
    let tries = 0;
    const wait = setInterval(() => {
      tries++;
      if (window.currentUserId !== undefined || tries > 20) {
        clearInterval(wait);
        if (window.currentUserId) loadMyPlaces();
        else $("#my-places-list").innerHTML = `<li class="text-slate-500"><a href="/auth.html" class="font-semibold text-purple-800 hover:underline">Sign in</a> to add your own places.</li>`;
      }
    }, 250);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
