// =============================================================================
// PLAN DOM HELPERS (shared by route-optimizer.js and ai-review.js)
// =============================================================================
// Reads the day boxes of any plan_*.html page into a simple list of days,
// and writes a new list of activities back (cards, durations and autosave update).
//
// A day looks like:
//   { key: "tokyo-activities-1", label: "Tokyo · Day 1", city: "Tokyo", area: "Asakusa",
//     stops: [{ name: "Senso-ji", duration: 60 }, ...] }
// "key" is the id of the day's activities textarea, so it stays the same
// across reads and can be used to find the day again later.
// =============================================================================

(function () {
  "use strict";

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

  // "20, , 90" -> [20, null, 90]: a blank means "no duration given" (the server estimates it)
  function parseDurations(text, activityCount, hasLocation) {
    const raw = (text || "").trim();
    const values = raw ? raw.split(",").map((v) => {
      const n = parseInt(v.trim(), 10);
      return Number.isNaN(n) ? null : n;
    }) : [];
    // Older plans put the day's location first in the duration list
    const leadingLocation = hasLocation && values.length === activityCount + 1;
    const forActivities = leadingLocation ? values.slice(1) : values;
    return { values: forActivities, leadingLocation, leadingValue: leadingLocation ? values[0] : null };
  }

  function readDay(box) {
    const field = findActivityField(box);
    if (!field) return null;

    const container = box.closest("[id$='-days']");
    const containerCity = container && container.id !== "days" ? container.id.replace(/-days$/, "") : null;
    const city = capitalize(field.prefix || containerCity || window.ROUTE_DEFAULT_CITY || "Tokyo");

    const locationEl = findInput(box, field.prefix, field.n, ["location", "loc", "place"]);
    const durationsEl = findInput(box, field.prefix, field.n, ["durations", "dur", "time"]);
    const area = (locationEl?.value || "").trim();

    const activities = (field.el.value || "").split("\n").map((a) => a.trim()).filter(Boolean);
    const durations = parseDurations(durationsEl?.value, activities.length, Boolean(area));

    return {
      key: field.el.id,
      label: `${city} · Day ${field.n}`,
      city,
      area,
      stops: activities.map((name, i) => ({ name, duration: durations.values[i] ?? null })),
      // Kept for writing back (not sent to the server)
      _field: field,
      _durationsEl: durationsEl,
      _durations: durations,
    };
  }

  /** All days on the page, in page order */
  function collectDays() {
    return [...document.querySelectorAll(".day-box")].map(readDay).filter(Boolean);
  }

  /** One day by key, read fresh from the page (null if it no longer exists) */
  function getDay(key) {
    const el = document.getElementById(key);
    const box = el?.closest(".day-box");
    return box ? readDay(box) : null;
  }

  /** Strip fields that are only used in the browser */
  const forServer = (days) => days.map(({ key, label, city, area, stops }) => ({ key, label, city, area, stops }));

  /**
   * Replace a day's activities with 'stops' ([{name, duration}]).
   * Durations are written only if the user had filled some in; stops without one stay blank.
   */
  function writeDay(key, stops) {
    const day = getDay(key);
    if (!day) return false;
    const { el, prefix, fieldType, n } = day._field;
    const names = stops.map((s) => s.name);

    const hasAnyDuration = stops.some((s) => s.duration != null);
    if (day._durationsEl && (day._durations.values.length || hasAnyDuration)) {
      const values = stops.map((s) => s.duration ?? "");
      if (day._durations.leadingLocation) values.unshift(day._durations.leadingValue ?? "");
      while (values.length && values[values.length - 1] === "") values.pop();  // no trailing blanks
      day._durationsEl.value = values.join(", ");
      day._durationsEl.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // Use the page's own helper so the activity cards and autosave update
    if (typeof window.setActivities === "function") {
      if (prefix) window.setActivities(prefix, n, names, fieldType);
      else window.setActivities(n, names);
    } else {
      el.value = names.join("\n");
      el.dispatchEvent(new Event("input", { bubbles: true }));
      if (typeof window.syncTextareaToCards === "function") {
        if (prefix) window.syncTextareaToCards(prefix, n, fieldType);
        else window.syncTextareaToCards(n);
      }
    }
    return true;
  }

  /**
   * Put a day's current stops in the given order (by name). Stops not in 'order'
   * (e.g. added since) stay at the end; names no longer in the day are skipped.
   */
  function reorderDay(key, order) {
    const day = getDay(key);
    if (!day) return false;
    const remaining = [...day.stops];
    const ordered = [];
    for (const name of order) {
      const i = remaining.findIndex((s) => s.name === name);
      if (i >= 0) ordered.push(...remaining.splice(i, 1));
    }
    return writeDay(key, [...ordered, ...remaining]);
  }

  /** Set a day's area / location field ("Asakusa") */
  function writeArea(key, text) {
    const day = getDay(key);
    if (!day) return false;
    const { prefix, n } = day._field;
    const box = document.getElementById(key).closest(".day-box");
    const el = findInput(box, prefix, n, ["location", "loc", "place"]);
    if (!el) return false;
    el.value = text || "";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }

  /** Swap one stop for another (e.g. "Lunch" -> a restaurant), keeping its duration */
  function replaceStop(key, oldName, newName) {
    const day = getDay(key);
    if (!day) return false;
    const i = day.stops.findIndex((s) => s.name === oldName);
    if (i < 0) return false;
    const stops = [...day.stops];
    stops[i] = { ...stops[i], name: newName };
    return writeDay(key, stops);
  }

  // ---------------------------------------------------------------------------
  // Nearby picks ("Lunch" -> restaurants from the database), shared by both panels
  // ---------------------------------------------------------------------------

  const PICK_ICON = { shopping: "🛍️", snack: "☕" };
  const MODE_WORD = { walk: "walk", train: "by train", "bus/car": "by bus/car" };

  /** HTML for a list of picks with "Use" buttons; 'key' is the day, 'stop' the stop to replace */
  function renderPicks(key, stop, type, near, options) {
    if (!options || !options.length) return "";
    const rows = options.map((o) => {
      const facts = [
        o.cuisine ? esc(o.cuisine) : "",
        o.rating ? `★${esc(o.rating)}` : "",
        o.price_level ? "¥".repeat(o.price_level) : "",
        `${formatMinutes(o.minutes)} ${MODE_WORD[o.mode] || ""}`.trim(),
      ].filter(Boolean).join(" · ");
      const name = o.google_maps_url
        ? `<a href="${esc(o.google_maps_url)}" target="_blank" rel="noopener" class="font-medium text-slate-800 hover:underline">${esc(o.name)}</a>`
        : `<span class="font-medium text-slate-800">${esc(o.name)}</span>`;
      return `
        <li class="flex items-center justify-between gap-2 py-1">
          <div class="min-w-0 text-xs">${name}<div class="text-slate-500">${facts}</div></div>
          <button type="button" data-pick-key="${esc(key)}" data-pick-stop="${esc(stop)}" data-pick-name="${esc(o.name)}"
            class="shrink-0 rounded border border-orange-300 bg-white px-2 py-0.5 text-xs font-semibold text-orange-800 hover:bg-orange-50 disabled:opacity-60">
            Use</button>
        </li>`;
    }).join("");
    return `
      <div class="mt-1 rounded-md bg-orange-50 px-2 py-1.5" data-pick-group>
        <div class="text-xs font-medium text-orange-900">${PICK_ICON[type] || "🍽️"} Ideas for “${esc(stop)}” near ${esc(near)}</div>
        <ul class="divide-y divide-orange-100">${rows}</ul>
      </div>`;
  }

  /** Wire the "Use" buttons inside 'container'; onUsed(key, oldName, newName) after a swap */
  function bindPicks(container, onUsed) {
    container.querySelectorAll("[data-pick-name]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const { pickKey, pickStop } = btn.dataset;
        // "Lunch" -> "Lunch: <restaurant>", so it stays scheduled at lunchtime
        const pickName = /^(breakfast|brunch|lunch|dinner)$/i.test(pickStop.trim())
          ? `${pickStop.trim()}: ${btn.dataset.pickName}` : btn.dataset.pickName;
        if (!replaceStop(pickKey, pickStop, pickName)) {
          alert(`“${pickStop}” isn't on that day anymore.`);
          return;
        }
        btn.closest("[data-pick-group]").querySelectorAll("[data-pick-name]").forEach((b) => { b.disabled = true; });
        btn.textContent = "✓ Added";
        onUsed?.(pickKey, pickStop, pickName);
      });
    });
  }

  window.PlanDom = {
    collectDays, getDay, forServer, writeDay, writeArea, reorderDay, replaceStop, renderPicks, bindPicks,
    esc, formatMinutes, yen,
  };
})();
