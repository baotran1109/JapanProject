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

  function parseDurations(text, activityCount, hasLocation) {
    const values = (text || "").split(",").map((v) => parseInt(v.trim(), 10)).filter((v) => !Number.isNaN(v));
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
      stops: activities.map((name, i) => ({ name, duration: durations.values[i] || 60 })),
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
   * Durations are written only if the user had filled them in (or a stop has a custom one).
   */
  function writeDay(key, stops) {
    const day = getDay(key);
    if (!day) return false;
    const { el, prefix, fieldType, n } = day._field;
    const names = stops.map((s) => s.name);

    const hadDurations = day._durations.values.length > 0;
    const customDuration = stops.some((s) => s.duration && s.duration !== 60);
    if (day._durationsEl && (hadDurations || customDuration)) {
      const values = stops.map((s) => s.duration || 60);
      if (day._durations.leadingLocation) values.unshift(day._durations.leadingValue);
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

  window.PlanDom = { collectDays, getDay, forServer, writeDay, reorderDay, esc, formatMinutes, yen };
})();
