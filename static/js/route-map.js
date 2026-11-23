// =============================================================================
// ROUTE MAP (used by route-optimizer.js)
// =============================================================================
// Draws each day's route on an OpenStreetMap map with Leaflet:
// - numbered pins in visiting order, a line between stops, one color per day
// - "Before / After" toggle to compare the planned and optimized order
// - day tabs ("All days" or one day at a time)
// Leaflet is loaded the first time a map is shown, so pages stay light.
// =============================================================================

(function () {
  "use strict";

  const LEAFLET_VERSION = "1.9.4";
  const LEAFLET_CSS = `https://cdnjs.cloudflare.com/ajax/libs/leaflet/${LEAFLET_VERSION}/leaflet.min.css`;
  const LEAFLET_JS = `https://cdnjs.cloudflare.com/ajax/libs/leaflet/${LEAFLET_VERSION}/leaflet.min.js`;
  const DAY_COLORS = ["#16a34a", "#2563eb", "#db2777", "#ea580c", "#7c3aed", "#0891b2", "#ca8a04", "#dc2626"];

  let leafletPromise = null;

  function loadLeaflet() {
    if (window.L) return Promise.resolve(window.L);
    if (leafletPromise) return leafletPromise;
    leafletPromise = new Promise((resolve, reject) => {
      const css = document.createElement("link");
      css.rel = "stylesheet";
      css.href = LEAFLET_CSS;
      document.head.appendChild(css);
      const script = document.createElement("script");
      script.src = LEAFLET_JS;
      script.onload = () => resolve(window.L);
      script.onerror = () => { leafletPromise = null; reject(new Error("Couldn't load the map library")); };
      document.head.appendChild(script);
    });
    return leafletPromise;
  }

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  // ---------------------------------------------------------------------------
  // Turning optimizer results into points
  // ---------------------------------------------------------------------------

  /** Located stops of a day, in optimized ("after") or planned ("before") order */
  function dayPoints(day, mode) {
    const located = day.stops.filter((s) => s.located && s.lat != null && s.lng != null);
    if (mode === "after") return located;
    // Planned order: follow original_order, taking each name once (names can repeat)
    const pool = [...located];
    const ordered = [];
    for (const name of day.original_order || []) {
      const i = pool.findIndex((s) => s.name === name);
      if (i >= 0) ordered.push(...pool.splice(i, 1));
    }
    return ordered;
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  function pinIcon(L, number, color, faded) {
    return L.divIcon({
      className: "",
      iconSize: [26, 26],
      iconAnchor: [13, 13],
      popupAnchor: [0, -12],
      html: `<div style="width:26px;height:26px;border-radius:9999px;background:${faded ? "#94a3b8" : color};
               color:#fff;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.35);
               display:flex;align-items:center;justify-content:center;font:700 12px/1 system-ui,sans-serif">${number}</div>`,
    });
  }

  function startIcon(L) {
    return L.divIcon({
      className: "",
      iconSize: [22, 22],
      iconAnchor: [11, 11],
      html: `<div style="width:22px;height:22px;border-radius:6px;background:#0f172a;color:#fff;border:2px solid #fff;
               box-shadow:0 1px 4px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;font-size:12px">⌂</div>`,
    });
  }

  function drawDay(L, layer, day, color, mode) {
    const points = dayPoints(day, mode);
    const latlngs = [];
    const anchor = day.anchor_point;
    if (anchor && anchor.lat != null) {
      latlngs.push([anchor.lat, anchor.lng]);
      L.marker([anchor.lat, anchor.lng], { icon: startIcon(L), zIndexOffset: -1000 })
        .bindPopup(`<strong>${esc(day.label)}</strong><br>Start: ${esc(anchor.name)}`)
        .addTo(layer);
    }
    points.forEach((stop, i) => {
      latlngs.push([stop.lat, stop.lng]);
      const when = mode === "after" ? `${esc(stop.arrive)}–${esc(stop.depart)}<br>` : "";
      L.marker([stop.lat, stop.lng], { icon: pinIcon(L, i + 1, color, mode === "before") })
        .bindPopup(`<strong>${i + 1}. ${esc(stop.name)}</strong><br>${when}<span style="color:#64748b">${esc(day.label)}</span>`)
        .addTo(layer);
    });
    if (latlngs.length > 1) {
      L.polyline(latlngs, mode === "before"
        ? { color: "#64748b", weight: 3, opacity: 0.8, dashArray: "6 8" }
        : { color, weight: 4, opacity: 0.85 }).addTo(layer);
    }
    return latlngs;
  }

  // ---------------------------------------------------------------------------
  // Public: render into a container
  // ---------------------------------------------------------------------------

  /**
   * container: an empty element; result: the /optimize-route response.
   * options.compare = false hides the Before / After toggle (a schedule that wasn't optimized).
   * Returns a promise (the map library loads on first use).
   */
  async function render(container, result, options = {}) {
    const compare = options.compare !== false;
    const days = result.days.filter((d) => d.stops.some((s) => s.located));
    if (!days.length) {
      container.innerHTML = "";
      return;
    }
    const state = { day: 0, mode: "after" };  // first day, optimized order

    container.innerHTML = `
      <div class="rounded-xl bg-white p-3 shadow-sm">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="flex flex-wrap gap-1" data-map-days>
            ${days.length > 1 ? `<button type="button" data-map-day="all" class="rounded-full px-3 py-1 text-xs font-semibold">All days</button>` : ""}
            ${days.map((d, i) => `<button type="button" data-map-day="${i}" class="rounded-full px-3 py-1 text-xs font-semibold">
                <span style="color:${DAY_COLORS[i % DAY_COLORS.length]}">●</span> ${esc(d.label)}</button>`).join("")}
          </div>
          <div class="${compare ? "inline-flex" : "hidden"} overflow-hidden rounded-lg border border-slate-300 text-xs font-semibold" data-map-modes>
            <button type="button" data-map-mode="before" class="px-3 py-1">Before</button>
            <button type="button" data-map-mode="after" class="px-3 py-1">After</button>
          </div>
        </div>
        <div data-map class="mt-2 h-80 w-full overflow-hidden rounded-lg bg-slate-100 md:h-96">
          <div class="flex h-full items-center justify-center text-sm text-slate-500">Loading map…</div>
        </div>
        <div data-map-caption class="mt-2 text-xs text-slate-500"></div>
      </div>`;

    let L;
    try {
      L = await loadLeaflet();
    } catch (e) {
      container.querySelector("[data-map]").innerHTML =
        `<div class="flex h-full items-center justify-center text-sm text-slate-500">${esc(e.message)}</div>`;
      return;
    }

    const mapEl = container.querySelector("[data-map]");
    mapEl.innerHTML = "";
    const map = L.map(mapEl, { scrollWheelZoom: false });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    const layer = L.layerGroup().addTo(map);

    function caption() {
      const shown = state.day === "all" ? days : [days[state.day]];
      const before = shown.reduce((sum, d) => sum + d.original_travel_minutes, 0);
      const after = shown.reduce((sum, d) => sum + d.travel_minutes, 0);
      const hidden = shown.reduce((sum, d) => sum + d.stops.filter((s) => !s.located).length, 0);
      const fmt = window.PlanDom ? window.PlanDom.formatMinutes : (m) => `${m} min`;
      const travel = compare ? `travel ${fmt(before)} → ${fmt(after)}` : `travel ${fmt(after)}`;
      return `${!compare ? "Route" : state.mode === "before" ? "Your order (dashed)" : "Optimized order"} · ${travel}`
        + (hidden ? ` · ${hidden} stop${hidden > 1 ? "s" : ""} not on the map (e.g. “Lunch”)` : "");
    }

    function draw() {
      layer.clearLayers();
      const shown = state.day === "all" ? days.map((d, i) => [d, i]) : [[days[state.day], state.day]];
      const all = [];
      shown.forEach(([d, i]) => all.push(...drawDay(L, layer, d, DAY_COLORS[i % DAY_COLORS.length], state.mode)));
      if (all.length === 1) map.setView(all[0], 15);
      else if (all.length) map.fitBounds(all, { padding: [30, 30], maxZoom: 16 });

      container.querySelectorAll("[data-map-day]").forEach((b) => {
        const active = String(state.day) === b.dataset.mapDay;
        b.className = `rounded-full px-3 py-1 text-xs font-semibold ${active ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`;
      });
      container.querySelectorAll("[data-map-mode]").forEach((b) => {
        const active = state.mode === b.dataset.mapMode;
        b.className = `px-3 py-1 ${active ? "bg-green-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`;
      });
      container.querySelector("[data-map-caption]").textContent = caption();
    }

    container.querySelectorAll("[data-map-day]").forEach((b) => b.addEventListener("click", () => {
      state.day = b.dataset.mapDay === "all" ? "all" : Number(b.dataset.mapDay);
      draw();
    }));
    container.querySelectorAll("[data-map-mode]").forEach((b) => b.addEventListener("click", () => {
      state.mode = b.dataset.mapMode;
      draw();
    }));

    draw();
    // The panel may have just become visible; let Leaflet measure it again
    setTimeout(() => { map.invalidateSize(); draw(); }, 150);
  }

  window.RouteMap = { render };
})();
