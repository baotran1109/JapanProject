// =============================================================================
// ROUTE MAP (used by route-optimizer.js and the trip view)
// =============================================================================
// Draws each day's route on a clean, English-labelled map:
// - numbered pins in visiting order, a line between stops, one color per day
// - "Before / After" toggle to compare the planned and optimized order
// - day tabs ("All days" or one day at a time)
//
// Map: MapLibre GL with OpenFreeMap's "Positron" style (free, no API key, data
// from OpenStreetMap). Labels use the English name, then the romanized name
// ("Shinagawa"), so the map isn't all in Japanese.
// The map library loads the first time a map is shown, so pages stay light.
// =============================================================================

(function () {
  "use strict";

  const MAPLIBRE_VERSION = "4.7.1";
  const MAPLIBRE_CSS = `https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/${MAPLIBRE_VERSION}/maplibre-gl.min.css`;
  const MAPLIBRE_JS = `https://cdnjs.cloudflare.com/ajax/libs/maplibre-gl/${MAPLIBRE_VERSION}/maplibre-gl.min.js`;
  const MAP_STYLE = "https://tiles.openfreemap.org/styles/positron";
  // English name first, then the Latin-alphabet name, then the local name
  const ENGLISH_LABEL = ["coalesce", ["get", "name:en"], ["get", "name_en"], ["get", "name:latin"], ["get", "name"]];
  const DAY_COLORS = ["#16a34a", "#2563eb", "#db2777", "#ea580c", "#7c3aed", "#0891b2", "#ca8a04", "#dc2626"];
  const BEFORE_COLOR = "#64748b";

  let libraryPromise = null;

  function loadLibrary() {
    if (window.maplibregl) return Promise.resolve(window.maplibregl);
    if (libraryPromise) return libraryPromise;
    libraryPromise = new Promise((resolve, reject) => {
      const css = document.createElement("link");
      css.rel = "stylesheet";
      css.href = MAPLIBRE_CSS;
      document.head.appendChild(css);
      const script = document.createElement("script");
      script.src = MAPLIBRE_JS;
      script.onload = () => resolve(window.maplibregl);
      script.onerror = () => { libraryPromise = null; reject(new Error("Couldn't load the map")); };
      document.head.appendChild(script);
    });
    return libraryPromise;
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
  // Markers (HTML elements, so they look the same as before)
  // ---------------------------------------------------------------------------

  function pinElement(number, color, faded) {
    const el = document.createElement("div");
    el.style.cssText = `width:26px;height:26px;border-radius:9999px;background:${faded ? "#94a3b8" : color};color:#fff;
      border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;
      font:700 12px/1 system-ui,sans-serif;cursor:pointer`;
    el.textContent = number;
    return el;
  }

  function startElement() {
    const el = document.createElement("div");
    el.style.cssText = `width:22px;height:22px;border-radius:6px;background:#0f172a;color:#fff;border:2px solid #fff;
      box-shadow:0 1px 4px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;font-size:12px;cursor:pointer`;
    el.textContent = "⌂";
    return el;
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
        <div data-map class="relative mt-2 h-80 w-full overflow-hidden rounded-lg bg-slate-100 md:h-96">
          <div data-map-loading class="absolute inset-0 flex items-center justify-center text-sm text-slate-500">Loading map…</div>
        </div>
        <div data-map-caption class="mt-2 text-xs text-slate-500"></div>
      </div>`;

    const mapEl = container.querySelector("[data-map]");
    let maplibregl, map;
    try {
      maplibregl = await loadLibrary();
      map = new maplibregl.Map({
        container: mapEl,
        style: MAP_STYLE,
        center: [139.76, 35.68],
        zoom: 11,
        cooperativeGestures: true,   // Scrolling the page doesn't zoom the map (Ctrl/⌘ + scroll does)
        attributionControl: { compact: true },
      });
    } catch (e) {
      mapEl.innerHTML = `<div class="flex h-full items-center justify-center p-4 text-center text-sm text-slate-500">
        ${esc(e.message || "This browser can't show the map")}</div>`;
      return;
    }
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    let markers = [];
    let ready = false;

    function lineFeatures() {
      const shown = state.day === "all" ? days.map((d, i) => [d, i]) : [[days[state.day], state.day]];
      const features = [];
      const points = [];
      markers.forEach((m) => m.remove());
      markers = [];
      for (const [day, i] of shown) {
        const color = DAY_COLORS[i % DAY_COLORS.length];
        const coords = [];
        const anchor = day.anchor_point;
        if (anchor && anchor.lat != null) {
          coords.push([anchor.lng, anchor.lat]);
          markers.push(new maplibregl.Marker({ element: startElement() })
            .setLngLat([anchor.lng, anchor.lat])
            .setPopup(new maplibregl.Popup({ offset: 14 }).setHTML(`<strong>${esc(day.label)}</strong><br>Start: ${esc(anchor.name)}`)));
        }
        dayPoints(day, state.mode).forEach((stop, k) => {
          coords.push([stop.lng, stop.lat]);
          const when = state.mode === "after" ? `${esc(stop.arrive)}–${esc(stop.depart)}<br>` : "";
          markers.push(new maplibregl.Marker({ element: pinElement(k + 1, color, state.mode === "before") })
            .setLngLat([stop.lng, stop.lat])
            .setPopup(new maplibregl.Popup({ offset: 16 }).setHTML(
              `<strong>${k + 1}. ${esc(stop.name)}</strong><br>${when}<span style="color:#64748b">${esc(day.label)}</span>`)));
        });
        if (coords.length > 1) {
          features.push({
            type: "Feature",
            properties: { color: state.mode === "before" ? BEFORE_COLOR : color },
            geometry: { type: "LineString", coordinates: coords },
          });
        }
        points.push(...coords);
      }
      // Pins on top of the start marker
      markers.forEach((m) => m.addTo(map));
      return { features, points };
    }

    function fit(points) {
      if (!points.length) return;
      if (points.length === 1) {
        map.jumpTo({ center: points[0], zoom: 15 });
        return;
      }
      const bounds = points.reduce((b, p) => b.extend(p), new maplibregl.LngLatBounds(points[0], points[0]));
      map.fitBounds(bounds, { padding: 40, maxZoom: 16, duration: 0 });
    }

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
      container.querySelectorAll("[data-map-day]").forEach((b) => {
        const active = String(state.day) === b.dataset.mapDay;
        b.className = `rounded-full px-3 py-1 text-xs font-semibold ${active ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`;
      });
      container.querySelectorAll("[data-map-mode]").forEach((b) => {
        const active = state.mode === b.dataset.mapMode;
        b.className = `px-3 py-1 ${active ? "bg-green-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50"}`;
      });
      container.querySelector("[data-map-caption]").textContent = caption();
      if (!ready) return;
      const { features, points } = lineFeatures();
      map.getSource("route").setData({ type: "FeatureCollection", features });
      map.setPaintProperty("route-line", "line-dasharray", state.mode === "before" ? [2, 2] : undefined);
      fit(points);
    }

    map.on("load", () => {
      // English (or romanized) labels instead of Japanese
      for (const layer of map.getStyle().layers) {
        if (layer.layout && layer.layout["text-field"]) {
          try { map.setLayoutProperty(layer.id, "text-field", ENGLISH_LABEL); } catch { /* layer without names */ }
        }
      }
      map.addSource("route", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "route-casing",
        type: "line",
        source: "route",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": "#ffffff", "line-width": 7, "line-opacity": 0.9 },
      });
      map.addLayer({
        id: "route-line",
        type: "line",
        source: "route",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": ["get", "color"], "line-width": 4, "line-opacity": 0.9 },
      });
      ready = true;
      container.querySelector("[data-map-loading]")?.remove();
      draw();
    });
    map.on("error", (e) => console.warn("Map:", e?.error?.message || e));

    container.querySelectorAll("[data-map-day]").forEach((b) => b.addEventListener("click", () => {
      state.day = b.dataset.mapDay === "all" ? "all" : Number(b.dataset.mapDay);
      draw();
    }));
    container.querySelectorAll("[data-map-mode]").forEach((b) => b.addEventListener("click", () => {
      state.mode = b.dataset.mapMode;
      draw();
    }));

    // The map may be created inside a hidden tab: re-measure (and re-fit) when it becomes visible
    let lastWidth = mapEl.clientWidth;
    new ResizeObserver(() => {
      map.resize();
      if (lastWidth === 0 && mapEl.clientWidth > 0) draw();
      lastWidth = mapEl.clientWidth;
    }).observe(mapEl);

    draw();
  }

  window.RouteMap = { render };
})();
