// =============================================================================
// CHAT EDITS (plan.html): "make day 2 more relaxed", "add a ramen dinner on day 1"
// =============================================================================
// Sends the trip and the message to POST /ai/chat-edit. The answer lists the
// changes (checked by the server against real places) and the changed days;
// nothing changes until "Apply". "Undo" puts the days back.
// Needs plan-dom.js (and planner.js for undo), a button #chat-btn and #chat-panel.
// =============================================================================

(function () {
  "use strict";

  const { esc, formatMinutes } = window.PlanDom;
  const EXAMPLES = [
    "Make day 2 more relaxed",
    "Add a ramen dinner on day 1",
    "Add 3 vintage shops to day 1",
    "Swap day 1 and day 2",
  ];
  const history = [];      // [{role, text}] sent back so follow-ups make sense
  let busy = false;

  const panel = () => document.getElementById("chat-panel");

  function shell() {
    return `
      <div class="flex items-start justify-between gap-3">
        <div>
          <h3 class="text-lg font-semibold">💬 Edit your trip by chatting</h3>
          <p class="text-sm text-slate-500">Say what you'd like to change. You'll see the changes before anything is applied.</p>
        </div>
        <button type="button" data-chat-close class="shrink-0 rounded-md px-2 py-1 text-slate-500 hover:bg-slate-100" aria-label="Close">✕</button>
      </div>
      <div data-chat-log class="mt-3 max-h-[28rem] space-y-3 overflow-y-auto"></div>
      <div data-chat-examples class="mt-3 flex flex-wrap gap-2">
        ${EXAMPLES.map((e) => `<button type="button" data-chat-example="${esc(e)}"
          class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 hover:bg-slate-50">${esc(e)}</button>`).join("")}
      </div>
      <form data-chat-form class="mt-3 flex gap-2">
        <input data-chat-input maxlength="500" placeholder="e.g. Add a sushi lunch near Tsukiji on day 1"
          class="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-200" />
        <button type="submit" data-chat-send class="shrink-0 rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700 disabled:opacity-60">Send</button>
      </form>`;
  }

  function bubble(role, html) {
    const log = panel().querySelector("[data-chat-log]");
    const div = document.createElement("div");
    div.className = role === "user"
      ? "ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-violet-600 px-3 py-2 text-sm text-white"
      : "max-w-[95%] rounded-2xl rounded-bl-sm border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800";
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function dayChange(d) {
    const fmt = (x) => `${x.stops} stops · ${formatMinutes(x.active_minutes)} of sights & travel · ends ${esc(x.end_time)}`;
    return `
      <li class="rounded-md bg-white px-2 py-1.5 ring-1 ring-slate-200">
        <div class="font-medium">${esc(d.label)}</div>
        <div class="text-xs text-slate-500">Before: ${fmt(d.before)}</div>
        <div class="text-xs text-emerald-700">After: ${fmt(d.after)}</div>
        <div class="mt-0.5 text-xs text-slate-600">${d.stops.map((s) => esc(s.name)).join(" → ")}</div>
      </li>`;
  }

  function answerHtml(data) {
    const changes = data.changes.map((c) => `<li>✓ ${esc(c)}</li>`).join("");
    const rejected = data.rejected.map((c) => `<li>✗ ${esc(c)}</li>`).join("");
    return `
      <p>${esc(data.reply)}</p>
      ${changes ? `<ul class="mt-2 space-y-0.5 text-xs text-slate-700">${changes}</ul>` : ""}
      ${rejected ? `<ul class="mt-1 space-y-0.5 text-xs text-slate-400" title="Skipped: these didn't fit the plan">${rejected}</ul>` : ""}
      ${data.days.length ? `
        <ul class="mt-2 space-y-1.5">${data.days.map(dayChange).join("")}</ul>
        <div class="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" data-chat-apply class="rounded-md bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-700">Apply changes</button>
          <span data-chat-note class="text-xs text-slate-500">Changed days are put in their best order.</span>
        </div>` : ""}`;
  }

  function sameStops(a, b) {
    return JSON.stringify(a.map((s) => [s.name, s.duration ?? null])) === JSON.stringify(b.map((s) => [s.name, s.duration ?? null]));
  }

  function bindAnswer(div, data, sentDays) {
    const apply = div.querySelector("[data-chat-apply]");
    if (!apply) return;
    let before = null;
    apply.addEventListener("click", () => {
      const note = div.querySelector("[data-chat-note]");
      if (apply.dataset.applied) {  // Undo
        if (before && window.Planner) window.Planner.load(before);
        apply.dataset.applied = "";
        apply.textContent = "Apply changes";
        note.textContent = "Changes undone.";
        window.Planner?.recheck();
        return;
      }
      // Days edited since we asked: don't overwrite them silently
      const stale = data.days.filter((d) => {
        const now = window.PlanDom.getDay(d.key);
        const sent = sentDays.find((s) => s.key === d.key);
        return !now || !sent || !sameStops(now.stops, sent.stops);
      });
      if (stale.length && !confirm(`${stale.map((d) => d.label).join(", ")} changed since you asked. Apply anyway?`)) return;
      before = window.Planner ? { ...window.Planner.getTrip() } : null;
      data.days.forEach((d) => {
        window.PlanDom.writeDay(d.key, d.stops);
        if (typeof window.PlanDom.writeArea === "function") window.PlanDom.writeArea(d.key, d.area);
      });
      apply.dataset.applied = "1";
      apply.textContent = "Undo";
      apply.className = "rounded-md border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50";
      note.textContent = "Applied. Don't forget to save your trip.";
      window.Planner?.recheck();
    });
  }

  async function send(text) {
    text = text.trim();
    if (!text || busy) return;
    const days = window.PlanDom.collectDays();
    if (!days.some((d) => d.stops.length)) {
      bubble("assistant", "Add a few stops to your days first, then tell me what to change.");
      return;
    }
    busy = true;
    const p = panel();
    p.querySelector("[data-chat-send]").disabled = true;
    p.querySelector("[data-chat-examples]").classList.add("hidden");
    bubble("user", esc(text));
    const thinking = bubble("assistant", `<span class="text-slate-500">Thinking…</span>`);
    const sentDays = days.map((d) => ({ key: d.key, stops: d.stops }));
    let pace = "normal";
    try { pace = localStorage.getItem("aiReviewPace") || "normal"; } catch { /* ignore */ }
    try {
      const authHeaders = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
      const res = await fetch("/ai/chat-edit", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({ days: window.PlanDom.forServer(days), message: text, pace, history }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      thinking.innerHTML = answerHtml(data);
      bindAnswer(thinking, data, sentDays);
      history.push({ role: "user", text }, { role: "assistant", text: data.reply });
    } catch (e) {
      thinking.innerHTML = `<span class="text-rose-700">❌ ${esc(e.message)}</span>`;
    } finally {
      busy = false;
      p.querySelector("[data-chat-send]").disabled = false;
      p.querySelector("[data-chat-input]").focus();
    }
  }

  function init() {
    const button = document.getElementById("chat-btn");
    const p = panel();
    if (!button || !p) return;
    p.innerHTML = shell();
    button.addEventListener("click", () => {
      p.classList.toggle("hidden");
      if (!p.classList.contains("hidden")) {
        p.scrollIntoView({ behavior: "smooth", block: "nearest" });
        p.querySelector("[data-chat-input]").focus();
      }
    });
    p.querySelector("[data-chat-close]").addEventListener("click", () => p.classList.add("hidden"));
    p.querySelector("[data-chat-form]").addEventListener("submit", (e) => {
      e.preventDefault();
      const input = p.querySelector("[data-chat-input]");
      const text = input.value;
      input.value = "";
      send(text);
    });
    p.querySelectorAll("[data-chat-example]").forEach((b) => b.addEventListener("click", () => send(b.dataset.chatExample)));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
