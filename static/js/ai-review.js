// =============================================================================
// AI REVIEW (shared by all plan_*.html pages)
// =============================================================================
// - Sends every day of the plan to POST /ai/review with the chosen pace
// - Shows the score, summary, strengths, risks and per-day facts
// - Each suggestion has an "Apply" button that edits the plan:
//     reorder (best order for a day), swap, move (to another day), remove
//
// Pages need plan-dom.js loaded first, plus:
//   a button with id "ask-ai-review" or "ask-ai", and <aside id="ai-panel" class="hidden">
// =============================================================================

(function () {
  "use strict";

  const { esc, formatMinutes } = window.PlanDom;
  const PACES = { relaxed: "Relaxed", normal: "Normal", packed: "Packed" };
  const PACE_STORAGE_KEY = "aiReviewPace";

  let lastReview = null;

  function savedPace() {
    try {
      const pace = localStorage.getItem(PACE_STORAGE_KEY);
      return PACES[pace] ? pace : "normal";
    } catch {
      return "normal";
    }
  }

  function rememberPace(pace) {
    try { localStorage.setItem(PACE_STORAGE_KEY, pace); } catch { /* private mode etc. */ }
  }

  // ---------------------------------------------------------------------------
  // Applying suggestions
  // ---------------------------------------------------------------------------

  // Returns an error message, or null when the edit was applied
  function applySuggestion(s) {
    const day = window.PlanDom.getDay(s.dayKey);
    if (!day) return "That day isn't in your plan anymore.";
    const stops = [...day.stops];
    const indexOf = (name) => stops.findIndex((x) => x.name === name);
    const d = s.details || {};

    if (s.action === "reorder") {
      return window.PlanDom.reorderDay(s.dayKey, d.order) ? null : "Couldn't update that day.";
    }
    if (s.action === "swap") {
      const a = indexOf(d.a), b = indexOf(d.b);
      if (a < 0 || b < 0) return "Those stops have changed since the review.";
      [stops[a], stops[b]] = [stops[b], stops[a]];
      return window.PlanDom.writeDay(s.dayKey, stops) ? null : "Couldn't update that day.";
    }
    if (s.action === "remove") {
      const i = indexOf(d.item);
      if (i < 0) return `“${d.item}” isn't on that day anymore.`;
      stops.splice(i, 1);
      return window.PlanDom.writeDay(s.dayKey, stops) ? null : "Couldn't update that day.";
    }
    if (s.action === "move") {
      const target = window.PlanDom.getDay(d.toDayKey);
      const i = indexOf(d.item);
      if (!target) return "The target day isn't in your plan anymore.";
      if (i < 0) return `“${d.item}” isn't on that day anymore.`;
      const [moved] = stops.splice(i, 1);
      window.PlanDom.writeDay(s.dayKey, stops);
      window.PlanDom.writeDay(d.toDayKey, [...target.stops, moved]);
      return null;
    }
    return "Unknown suggestion type.";
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  const ACTION_LABEL = { reorder: "🔀 Reorder", swap: "⇄ Swap", move: "➡️ Move", remove: "✂️ Remove" };

  function scoreColor(score) {
    if (score >= 8) return "bg-green-100 text-green-800";
    if (score >= 6) return "bg-amber-100 text-amber-800";
    return "bg-red-100 text-red-800";
  }

  function describe(s) {
    const d = s.details || {};
    if (s.action === "reorder") return `New order: ${d.order.join(" → ")}`;
    if (s.action === "swap") return `Swap “${d.a}” and “${d.b}” on ${s.dayLabel}`;
    if (s.action === "move") return `Move “${d.item}” from ${s.dayLabel} to ${d.toDayLabel}`;
    if (s.action === "remove") return `Remove “${d.item}” from ${s.dayLabel}`;
    return "";
  }

  function renderSuggestion(s, i) {
    return `
      <div class="rounded-lg border border-slate-200 p-3">
        <div class="flex flex-wrap items-start justify-between gap-2">
          <div class="min-w-0">
            <div class="text-xs font-medium text-slate-500">${ACTION_LABEL[s.action] || ""} · ${esc(s.dayLabel)}</div>
            <div class="font-semibold text-slate-800">${esc(s.title)}</div>
          </div>
          <button type="button" data-apply-suggestion="${i}"
            class="shrink-0 rounded-md border border-slate-300 bg-white px-3 py-1 text-sm font-semibold hover:bg-slate-50 disabled:opacity-60">
            Apply</button>
        </div>
        ${s.reason ? `<p class="mt-1 text-sm text-slate-600">${esc(s.reason)}</p>` : ""}
        ${s.impact ? `<p class="mt-1 text-xs text-slate-500">Impact: ${esc(s.impact)}</p>` : ""}
        <p class="mt-1 break-words text-xs text-slate-400">${esc(describe(s))}</p>
      </div>`;
  }

  function renderDays(days) {
    const rows = days.filter((d) => !d.empty).map((d) => `
      <li class="py-2">
        <div class="flex flex-wrap justify-between gap-2 text-sm">
          <span class="font-medium text-slate-700">${esc(d.label)}</span>
          <span class="text-xs text-slate-500">${esc(d.start_time)}–${esc(d.end_time)}
            · travel ${formatMinutes(d.travel_minutes)}${d.travel_saved_minutes ? ` (could save ${formatMinutes(d.travel_saved_minutes)})` : ""}</span>
        </div>
        ${d.issues.length
          ? `<ul class="mt-1 space-y-0.5">${d.issues.map((t) => `<li class="text-xs text-amber-800">⚠️ ${esc(t)}</li>`).join("")}</ul>`
          : `<div class="mt-1 text-xs text-green-700">✓ Looks good</div>`}
        ${(d.picks || []).map((p) => window.PlanDom.renderPicks(d.key, p.stop, p.type, p.near, p.options)).join("")}
      </li>`).join("");
    return `
      <details class="mt-4 rounded-lg bg-slate-50 p-3">
        <summary class="cursor-pointer text-sm font-semibold text-slate-700">Day-by-day checks${
          days.some((d) => d.picks && d.picks.length) ? " · restaurant & shop ideas" : ""}</summary>
        <ul class="mt-2 divide-y divide-slate-200">${rows}</ul>
      </details>`;
  }

  function render(review, pace) {
    const panel = document.getElementById("ai-panel");
    if (!panel) return;
    const list = (items, empty) => items.length
      ? `<ul class="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">${items.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`
      : `<p class="mt-2 text-sm text-slate-500">${empty}</p>`;

    panel.innerHTML = `
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          <h3 class="text-lg font-semibold">🤖 AI Review</h3>
          <span class="rounded-full px-3 py-1 text-sm font-bold ${scoreColor(review.overallScore)}">${review.overallScore}/10</span>
        </div>
        <label class="flex items-center gap-2 text-sm text-slate-600">Pace
          <select id="ai-pace" class="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm">
            ${Object.entries(PACES).map(([value, label]) =>
              `<option value="${value}" ${value === pace ? "selected" : ""}>${label}</option>`).join("")}
          </select>
        </label>
      </div>
      ${review.summary ? `<p class="mt-2 text-slate-700">${esc(review.summary)}</p>` : ""}
      ${review.source === "rules" ? `<p class="mt-1 text-xs text-slate-500">Based on travel-time checks only (the AI part is unavailable right now).</p>` : ""}

      <div class="mt-4 grid gap-6 ${review.strengths.length ? "md:grid-cols-2" : ""}">
        ${review.strengths.length ? `<div><h4 class="font-semibold">✅ Strengths</h4>${list(review.strengths, "")}</div>` : ""}
        <div><h4 class="font-semibold">⚠️ Risks</h4>${list(review.risks, "Nothing major.")}</div>
      </div>

      <div class="mt-4">
        <h4 class="font-semibold">💡 Suggestions</h4>
        <div class="mt-2 grid gap-3">
          ${review.suggestions.length ? review.suggestions.map(renderSuggestion).join("")
            : `<p class="text-sm text-slate-500">No changes needed.</p>`}
        </div>
        <p id="ai-status" class="mt-2 text-sm text-green-800"></p>
      </div>
      ${renderDays(review.days || [])}`;

    panel.classList.remove("hidden");
    panel.querySelector("#ai-pace")?.addEventListener("change", (e) => {
      rememberPace(e.target.value);
      requestReview();
    });
    // "Use" a restaurant / shop: keep suggestions for that day pointing at the new name
    window.PlanDom.bindPicks(panel, (key, oldName, newName) => {
      const rename = (n) => (n === oldName ? newName : n);
      lastReview.suggestions.filter((s) => s.dayKey === key || s.details?.toDayKey === key).forEach((s) => {
        const d = s.details || {};
        if (d.order) d.order = d.order.map(rename);
        ["a", "b", "item"].forEach((f) => { if (d[f]) d[f] = rename(d[f]); });
      });
      const status = panel.querySelector("#ai-status");
      status.className = "mt-2 text-sm text-green-800";
      status.textContent = `Added “${newName}” instead of “${oldName}”. Don't forget to save your plan.`;
    });
    panel.querySelectorAll("[data-apply-suggestion]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s = lastReview.suggestions[Number(btn.dataset.applySuggestion)];
        const error = applySuggestion(s);
        const status = panel.querySelector("#ai-status");
        if (error) {
          status.className = "mt-2 text-sm text-amber-800";
          status.textContent = error + " Ask for a new review to get fresh suggestions.";
          return;
        }
        btn.disabled = true;
        btn.textContent = "✓ Applied";
        status.className = "mt-2 text-sm text-green-800";
        status.textContent = `Applied: ${s.title}. Don't forget to save your plan.`;
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Button
  // ---------------------------------------------------------------------------

  function reviewButton() {
    return document.getElementById("ask-ai-review") || document.getElementById("ask-ai");
  }

  async function requestReview() {
    const button = reviewButton();
    const days = window.PlanDom.collectDays();
    if (!days.some((d) => d.stops.length)) {
      alert("Add at least one activity before requesting an AI review.");
      return;
    }
    const pace = savedPace();
    const original = button?.innerHTML;
    if (button) { button.disabled = true; button.innerHTML = "⏳ Reviewing…"; }
    try {
      const authHeaders = typeof window.getAuthHeaders === "function" ? await window.getAuthHeaders() : {};
      const res = await fetch("/ai/review", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders },
        body: JSON.stringify({ days: window.PlanDom.forServer(days), pace }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      lastReview = data;
      render(data, pace);
      document.getElementById("ai-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      console.error("AI review failed:", e);
      alert("AI review failed: " + (e?.message || e));
    } finally {
      if (button) { button.disabled = false; button.innerHTML = original; }
    }
  }

  function init() {
    reviewButton()?.addEventListener("click", requestReview);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  window.AIReview = { request: requestReview };
})();
