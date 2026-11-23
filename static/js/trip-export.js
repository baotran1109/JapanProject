// =============================================================================
// SHARE & EXPORT (used by plan.html and view.html)
// =============================================================================
// - TripExport.schedule(trip): timed schedule of each day in the traveler's own order
//   (POST /optimize-route with mode "as_planned")
// - TripExport.icsFile(trip, schedule): a calendar file (.ics) with one event per stop
// - TripExport.share(trip): a read-only link (POST /share)
// trip = {title, start_date, days: [{city, area, stops: [{name, duration}]}]}
// =============================================================================

(function () {
  "use strict";

  const CITY_NAMES = {
    tokyo: "Tokyo", kyoto: "Kyoto", osaka: "Osaka", fukuoka: "Fukuoka",
    okinawa: "Okinawa", aomori: "Aomori", hakodate: "Hakodate", sapporo: "Sapporo",
  };

  function labels(trip) {
    const seen = {};
    return trip.days.map((d) => {
      seen[d.city] = (seen[d.city] || 0) + 1;
      return `${CITY_NAMES[d.city] || d.city} · Day ${seen[d.city]}`;
    });
  }

  async function authHeaders() {
    return typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
  }

  /** /optimize-route in the traveler's order (null for an empty trip) */
  async function schedule(trip) {
    if (!trip.days.some((d) => d.stops.length)) return null;
    const names = labels(trip);
    const res = await fetch("/optimize-route", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeaders()) },
      body: JSON.stringify({
        mode: "as_planned",
        start_time: "09:00",
        days: trip.days.map((d, i) => ({
          key: `d${i}`, label: names[i], city: CITY_NAMES[d.city] || d.city, area: d.area, stops: d.stops,
        })),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  async function share(trip) {
    const res = await fetch("/share", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(await authHeaders()) },
      body: JSON.stringify({ trip }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return new URL(data.url, location.origin).href;
  }

  // ---------------------------------------------------------------------------
  // Calendar (.ics)
  // ---------------------------------------------------------------------------

  const pad = (n) => String(n).padStart(2, "0");

  /** "09:30" / "00:40+1" -> minutes after the day's midnight */
  function clockToMinutes(text) {
    const m = String(text || "").match(/^(\d{1,2}):(\d{2})(\+1)?$/);
    return m ? Number(m[1]) * 60 + Number(m[2]) + (m[3] ? 1440 : 0) : null;
  }

  function icsDate(start, dayIndex, minutes) {
    const d = new Date(`${start}T00:00:00Z`);
    d.setUTCDate(d.getUTCDate() + dayIndex);
    d.setUTCMinutes(minutes);
    return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}00`;
  }

  const icsText = (s) => String(s ?? "").replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/([,;])/g, "\\$1");

  /** Lines longer than 75 bytes are folded, as the calendar format requires */
  function fold(line) {
    const out = [];
    let current = "";
    let bytes = 0;
    for (const ch of line) {
      const size = new TextEncoder().encode(ch).length;
      if (bytes + size > 74) {
        out.push(current);
        current = " ";
        bytes = 1;
      }
      current += ch;
      bytes += size;
    }
    out.push(current);
    return out.join("\r\n");
  }

  function mapsLink(stop, city) {
    const q = stop.lat != null ? `${stop.lat},${stop.lng}` : `${stop.matched_name || stop.name} ${city}`;
    return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(q)}`;
  }

  /** The calendar file's text. Times are Japan time (Asia/Tokyo). */
  function icsFile(trip, sched) {
    if (!trip.start_date) throw new Error("Pick the trip's start date first.");
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
    const uidBase = `${trip.start_date}-${Math.random().toString(36).slice(2, 10)}`;
    const lines = [
      "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Explore Japan//Trip Planner//EN", "CALSCALE:GREGORIAN",
      "METHOD:PUBLISH", `X-WR-CALNAME:${icsText(trip.title || "Japan trip")}`,
      "BEGIN:VTIMEZONE", "TZID:Asia/Tokyo", "BEGIN:STANDARD", "DTSTART:19700101T000000",
      "TZOFFSETFROM:+0900", "TZOFFSETTO:+0900", "TZNAME:JST", "END:STANDARD", "END:VTIMEZONE",
    ];
    const event = (uid, dayIndex, startMin, endMin, summary, location, description) => {
      lines.push("BEGIN:VEVENT", `UID:${uid}@explore-japan`, `DTSTAMP:${stamp}`,
        `DTSTART;TZID=Asia/Tokyo:${icsDate(trip.start_date, dayIndex, startMin)}`,
        `DTEND;TZID=Asia/Tokyo:${icsDate(trip.start_date, dayIndex, Math.max(endMin, startMin + 5))}`,
        `SUMMARY:${icsText(summary)}`);
      if (location) lines.push(`LOCATION:${icsText(location)}`);
      if (description) lines.push(`DESCRIPTION:${icsText(description)}`);
      lines.push("END:VEVENT");
    };

    sched.days.forEach((day, i) => {
      const city = day.city;
      day.stops.forEach((stop, k) => {
        const start = clockToMinutes(stop.arrive);
        const end = clockToMinutes(stop.depart);
        if (start == null || end == null) return;
        const travel = stop.travel_minutes ? `Getting there: about ${stop.travel_minutes} min (${stop.mode}).\n` : "";
        event(`${uidBase}-${i}-${k}`, i, start, end, stop.name, `${stop.matched_name || stop.name}, ${city}`,
          `${day.label}\n${travel}${mapsLink(stop, city)}`);
      });
    });
    // Travel to the next city, after the last stop of the day
    (sched.intercity || []).forEach((leg, n) => {
      const i = sched.days.findIndex((d) => d.label === leg.after_day_label);
      const last = sched.days[i]?.stops.slice(-1)[0];
      const start = last ? clockToMinutes(last.depart) + 30 : 17 * 60;
      event(`${uidBase}-move-${n}`, i, start, start + leg.minutes,
        `${leg.mode === "flight" ? "✈️ Flight" : "🚄 Train"} to ${leg.to_city}`, leg.from_city,
        `About ${Math.round(leg.minutes / 60 * 10) / 10} h · around ¥${leg.fare_yen.toLocaleString()}`);
    });
    lines.push("END:VCALENDAR");
    return lines.map(fold).join("\r\n") + "\r\n";
  }

  function download(filename, text, type = "text/calendar;charset=utf-8") {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadCalendar(trip) {
    const sched = await schedule(trip);
    if (!sched) throw new Error("Add some stops first.");
    const name = (trip.title || "japan-trip").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "japan-trip";
    download(`${name}.ics`, icsFile(trip, sched));
  }

  window.TripExport = { schedule, share, icsFile, downloadCalendar, mapsLink, labels, clockToMinutes, CITY_NAMES };
})();
