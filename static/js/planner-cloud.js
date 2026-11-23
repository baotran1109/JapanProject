// =============================================================================
// PLANNER: account, saving and loading (plan.html)
// =============================================================================
// - Sign-in state in the nav, and window.getAuthHeaders() for the backend
// - ?itineraryId=...: load a saved trip (any of the old formats too) and save changes to it
// - Without it: "Save" creates a new saved trip in your account
// Trips are stored in Firestore: users/{uid}/itineraries/{id}
// =============================================================================

import { auth, db } from "/static/js/firebase-init.js?v=5";
import { onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-auth.js";
import {
  doc, getDoc, updateDoc, collection, addDoc, serverTimestamp,
} from "https://www.gstatic.com/firebasejs/10.8.1/firebase-firestore.js";

const params = new URLSearchParams(location.search);
let itineraryId = params.get("itineraryId");
let user = null;
let saving = false;

const $ = (sel) => document.querySelector(sel);
const msg = (text, tone = "ok") => {
  const el = $("#save-msg");
  if (!el) return;
  el.textContent = text;
  el.className = `text-sm font-medium ${tone === "error" ? "text-rose-600" : "text-emerald-700"}`;
};

// Sends the Firebase ID token so the backend can verify who the user is
window.getAuthHeaders = async function () {
  const u = auth.currentUser;
  return u ? { Authorization: `Bearer ${await u.getIdToken()}` } : {};
};

function cityNames(saved) {
  const names = [];
  for (const d of saved.days) {
    const name = window.Planner.CITIES[d.city]?.name;
    if (name && !d.stops.length && saved.days.filter((x) => x.city === d.city).every((x) => !x.stops.length)) continue;
    if (name && names[names.length - 1] !== name) names.push(name);
  }
  return names;
}

function record(title) {
  const saved = window.Planner.toSaved();
  const route = cityNames(saved);
  return {
    title,
    city: route.join(" → ") || "Japan",
    cities: [...new Set(route)],
    days: saved.days.length,
    plan: saved,
    updatedAt: serverTimestamp(),
  };
}

function defaultTitle() {
  const trip = window.Planner.getTrip();
  const route = cityNames(trip);
  return trip.title || `${route.join(" → ") || "Japan"} trip`;
}

async function save({ copy = false } = {}) {
  if (saving) return;
  if (!user) {
    if (confirm("Sign in to save trips to your account. Go to the sign-in page now? (Your draft stays in this browser.)")) {
      location.href = "/auth.html";
    }
    return;
  }
  const button = $("#save-btn");
  saving = true;
  if (button) button.disabled = true;
  try {
    if (itineraryId && !copy) {
      msg("Saving…");
      const title = window.Planner.getTrip().title || defaultTitle();
      await updateDoc(doc(db, `users/${user.uid}/itineraries/${itineraryId}`), record(title));
      msg("Saved ✓");
    } else {
      const suggested = copy ? `${defaultTitle()} (copy)` : defaultTitle();
      const title = prompt("Trip name?", suggested);
      if (title === null) return;
      msg("Saving…");
      const ref = await addDoc(collection(db, `users/${user.uid}/itineraries`), {
        ...record(title.trim() || suggested),
        createdAt: serverTimestamp(),
      });
      // From now on this page edits the saved trip
      itineraryId = ref.id;
      history.replaceState(null, "", `/plan.html?itineraryId=${ref.id}`);
      $("#trip-title").value = title.trim() || suggested;
      $("#trip-title").dispatchEvent(new Event("input"));
      window.Planner.setMode("cloud");
      window.Planner.clearDraft();
      showCopyButton();
      msg(copy ? "Saved as a new trip ✓" : "Saved to your account ✓");
    }
    setTimeout(() => { if ($("#save-msg")?.textContent.endsWith("✓")) msg(""); }, 2500);
  } catch (e) {
    console.error("Save failed:", e);
    msg(`Save failed: ${e.message || e}`, "error");
  } finally {
    saving = false;
    if (button) button.disabled = false;
  }
}

function showCopyButton() {
  if ($("#save-copy-btn") || !itineraryId) return;
  const btn = document.createElement("button");
  btn.id = "save-copy-btn";
  btn.type = "button";
  btn.className = "rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-50";
  btn.textContent = "Save as copy";
  btn.addEventListener("click", () => save({ copy: true }));
  $("#save-btn").after(btn);
}

async function loadSaved() {
  const loading = $("#cloud-loading");
  try {
    const snap = await getDoc(doc(db, `users/${user.uid}/itineraries/${itineraryId}`));
    if (!snap.exists()) throw new Error("This trip doesn't exist in your account.");
    window.Planner.load(window.Planner.fromSaved(snap.data()), { skipUndo: true, mode: "cloud" });
    showCopyButton();
  } catch (e) {
    console.error(e);
    if (loading) loading.textContent = `Couldn't open this trip: ${e.message || e}`;
  }
}

function renderAuth() {
  const nameEl = $("#auth-name");
  const linkEl = $("#auth-link");
  if (!nameEl || !linkEl) return;
  if (user) {
    nameEl.textContent = user.displayName || user.email;
    linkEl.textContent = "Sign out";
    linkEl.href = "#";
    linkEl.onclick = async (e) => { e.preventDefault(); await signOut(auth); };
  } else {
    nameEl.textContent = "Not signed in";
    linkEl.textContent = "Sign in";
    linkEl.href = "/auth.html";
    linkEl.onclick = null;
  }
}

let firstAuth = true;
onAuthStateChanged(auth, async (u) => {
  user = u || null;
  window.currentUserId = user ? user.uid : null;
  renderAuth();
  if (!firstAuth) return;
  firstAuth = false;
  if (itineraryId) {
    if (user) await loadSaved();
    else {
      const loading = $("#cloud-loading");
      if (loading) loading.innerHTML = `This is a saved trip. <a href="/auth.html" class="font-semibold text-blue-700 hover:underline">Sign in</a> to open it.`;
    }
  }
});

$("#save-btn")?.addEventListener("click", () => save());
