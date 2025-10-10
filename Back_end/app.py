# app.py
import os
import json
import logging
from math import radians, sin, cos, asin, sqrt
from typing import Dict, List, Any, Tuple, Set

from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import safe_join
from rapidfuzz import process, fuzz
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import safe_join        # add this import


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
APP_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
FRONTEND_ROOT = os.path.join(APP_DIR, "../Front_end")
STATIC_ROOT = os.path.join(APP_DIR, "../static")
DATA_ROOT = os.path.join(APP_DIR, "../data")


# Which city JSONs/folders to load as a mini knowledge-base (KB)
CITY_DIRS = {
    "tokyo": [
        "Tokyo/Akihabara_restaurants.json",
        "Tokyo/tokyo_other_area_restaurants.json",
        "Tokyo/Asakusa_restaurants.json",
        "Tokyo/Fujimount_daytrip.json",
        "Tokyo/Ginza_restaurants.json",
        "Tokyo/Hakone_daytrip.json",
        "Tokyo/Ikebukuro_restaurants.json",
        "Tokyo/Kamakura_daytrip.json",
        "Tokyo/Kawagoe_daytrip.json",
        "Tokyo/Meguro_restaurants.json",
        "Tokyo/Nikko_daytrip.json",
        "Tokyo/Roppongi_restaurants.json",
        "Tokyo/Shibuya_restaurants.json",
        "Tokyo/Shinjuku_restaurants.json",
        "Tokyo/Tokyo Attractions.json",
        "Tokyo/Tokyo Day Trip.json",
        "Tokyo/Tokyo shopping.json",
        "Tokyo/Yokohama_daytrip.json",
    ],
    "aomori": [
        "Aomori/Aomori_attractions.json",
        "Aomori/Aomori_restaurants.json",
        "Aomori/Aomori_shopping.json",
        "Aomori/Hachinohe_Day_Trip(Restaurants).json",
        "Aomori/Hachinohe_Day_(Attractions).json",
        "Aomori/Hirosaki Day Trip (Attractions + Shopping).json",
        "Aomori/Hirosaki_Day_Trip (Restaurants).json",
    ],
    "hakodate": [
        "Hakodate/Hakodate_Attractions.json",
        "Hakodate/Hakodate_Restaurants.json",
        "Hakodate/Onuma National Park Half Day Trip.json",
    ],
    "sapporo": [
        "Sapporo/Sapporo_Attractions.json",
        "Sapporo/Sapporo_Restaurants.json",
        "Sapporo/Sapporo_Shopping.json",
        "Sapporo/Otaru_Day_trip.json",
        "Sapporo/Noboribetsu_Day_Trip.json",
    ],
    "osaka": [
        "Osaka/Kobe_Attractions.json",
        "Osaka/Kobe_restaurants.json",
        "Osaka/Kobe_shopping.json",
        "Osaka/Nara_day_trip.json",
        "Osaka/Osaka_Attractions.json",
        "Osaka/Osaka_Shopping.json",
        "Osaka/Osaka_Restaurants.json",
    ],
    "kyoto": [
        "Kyoto/Kyoto_Attractions.json",
        "Kyoto/Kyoto_Restaurants.json",
        "Kyoto/Kyoto_Shopping.json",
    ],
    # These can be folders — we’ll expand to all *.json inside
    "fukuoka": ["Fukuoka/"],
    "okinawa": ["Okinawa/"],
}

load_dotenv()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__, static_folder=None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _expand_city_paths(rel_paths: List[str]) -> List[str]:
    """
    Expand any directory references to all *.json files inside those directories.
    Keep explicit .json files as-is if they exist.
    """
    expanded = []
    for rel in rel_paths or []:
        if not rel:
            continue
        full = os.path.join(DATA_ROOT, rel)
        if os.path.isdir(full):
            for name in sorted(os.listdir(full)):
                if name.lower().endswith(".json"):
                    expanded.append(os.path.join(rel, name))
        elif os.path.isfile(full) and rel.lower().endswith(".json"):
            expanded.append(rel)
        else:
            logger.debug("Path not found or unsupported (skipped): %s", rel)
    return expanded

def load_kb_for(cities: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load a compact KB only for the cities present in the plan to keep prompts small.
    Adds a __normname and normalizes lat/lng if present.
    Accepts both explicit file paths and directories in CITY_DIRS.
    """
    kb = {c: [] for c in cities}
    for c in cities:
        rels = _expand_city_paths(CITY_DIRS.get(c, []))
        for rel in rels:
            full = os.path.join(DATA_ROOT, rel)
            if not os.path.exists(full):
                continue
            try:
                items = load_json(full)
                if not isinstance(items, list):
                    if isinstance(items, dict):
                        for k in ("items", "results", "data", "places"):
                            if k in items and isinstance(items[k], list):
                                items = items[k]
                                break
                        else:
                            items = []
                    else:
                        items = []
            except Exception as e:
                logger.warning("Failed to parse JSON: %s (%s)", rel, e)
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue
                item["__normname"] = (item.get("name") or "").strip().lower()
                for k in list(item.keys()):
                    lk = k.lower()
                    if lk in ("lat", "latitude"):
                        item["lat"] = item[k]
                    if lk in ("lng", "lon", "longitude"):
                        item["lng"] = item[k]
                kb[c].append(item)
    return kb

def fuzzy_lookup(city_kb: List[Dict[str, Any]], name: str, limit=3, score_cut=70) -> List[Dict[str, Any]]:
    query = (name or "").strip()
    if not query or not city_kb:
        return []
    names = [i.get("name", "") for i in city_kb]
    matches = process.extract(query, names, scorer=fuzz.WRatio, limit=limit)
    out = []
    for _match_name, score, idx in matches:
        if score >= score_cut and 0 <= idx < len(city_kb):
            out.append(city_kb[idx])
    return out

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except Exception:
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def _tokenize_activities(text: str) -> List[str]:
    """
    Split activities by / | ; , newline. Dedupe while preserving order.
    """
    if not text:
        return []
    raw = []
    buff = []
    for ch in text:
        if ch in "/|;\n,":
            raw.append("".join(buff))
            buff = []
        else:
            buff.append(ch)
    if buff:
        raw.append("".join(buff))

    dedup: List[str] = []
    seen: Set[str] = set()
    for t in raw:
        s = t.strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            dedup.append(s)
    return dedup

def enrich_with_kb(plan: Dict[str, Any], kb: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    plan = { "tokyo":[{"day":1,"location":"...","activities":"..."},...], ... }
    For each location, attach best KB match (name/lat/lng/url/category if available).
    """
    enriched: Dict[str, List[Dict[str, Any]]] = {}
    for city, days in (plan or {}).items():
        if not isinstance(days, list):
            continue
        city_kb = kb.get(city, [])
        enriched[city] = []
        for d in days or []:
            if not isinstance(d, dict):
                continue
            loc = (d.get("location") or "").strip()
            best = fuzzy_lookup(city_kb, loc, limit=1, score_cut=70)
            match = best[0] if best else {}
            enriched[city].append({
                "day": d.get("day"),
                "location": loc,
                "activities": d.get("activities", ""),
                "kbMatch": {
                    "name": match.get("name"),
                    "lat": match.get("lat"),
                    "lng": match.get("lng"),
                    "url": match.get("google_maps_url") or match.get("google_map_url"),
                    "category": match.get("type") or match.get("category"),
                }
            })
    return enriched

def _build_allowed_map(enriched: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[int, List[str]]]:
    """
    AllowedPlaces[region][day] = list of place tokens user actually typed:
      - location field
      - tokenized 'activities'
    IMPORTANT: skip days that end up with zero tokens so the model can't
    move things into non-existent/empty days.
    """
    allowed: Dict[str, Dict[int, List[str]]] = {}
    for region, days in enriched.items():
        allowed[region] = {}
        for d in days:
            day_num = int(d.get("day") or 0) or 0
            if day_num <= 0:
                continue
            tokens: List[str] = []
            loc = (d.get("location") or "").strip()
            if loc:
                tokens.append(loc)
            for t in _tokenize_activities(d.get("activities") or ""):
                tokens.append(t)
            # de-dupe while preserving order
            uniq, seen = [], set()
            for t in tokens:
                key = t.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    uniq.append(t.strip())
            if not uniq:        # <-- key change
                continue
            allowed[region][day_num] = uniq
    return allowed

def _has_legal_option(allowed: Dict[str, Dict[int, List[str]]]) -> bool:
    for _region, days in allowed.items():
        if not days:
            continue
        day_numbers = sorted(days.keys())
        if any(len(days[d]) >= 2 for d in day_numbers):
            return True
        for d in day_numbers:
            if len(days[d]) >= 1 and ((d-1) in days or (d+1) in days):
                return True
    return False

# === Heuristic helpers to strengthen suggestions =============================
MEAL_WORDS    = {"breakfast", "brunch", "lunch", "dinner", "ramen", "sushi", "yakitori", "katsu", "tempura"}
SHOP_WORDS    = {"shopping", "mall", "department store", "donki", "don quijote", "vintage", "thrift"}
EVENING_WORDS = {"sky", "observatory", "tower", "night view", "sunset", "illumination"}

def _tag_token(token: str) -> Dict[str, bool]:
    t = (token or "").lower()
    return {
        "is_meal":     any(w in t for w in MEAL_WORDS),
        "is_shop":     any(w in t for w in SHOP_WORDS),
        "is_evening":  any(w in t for w in EVENING_WORDS),
    }

def _collect_all_items(allowed: Dict[str, Dict[int, List[str]]]) -> Dict[str, List[Tuple[str,int]]]:
    occ: Dict[str, List[Tuple[str,int]]] = {}
    for region, days in allowed.items():
        for d, items in days.items():
            for it in items:
                k = it.lower()
                occ.setdefault(k, []).append((region, d))
    return occ

def _heuristic_suggestions(enriched: Dict[str, List[Dict[str, Any]]],
                           allowed: Dict[str, Dict[int, List[str]]],
                           max_sugs: int = 3) -> List[Dict[str, Any]]:
    """
    Deterministic micro-edits to guarantee useful, concrete suggestions:
      1) Within a day: if 'shopping' appears before a meal, suggest SWAP so meal comes first.
      2) Within a day: if an 'evening' thing is first, swap it later.
      3) Across days: if the exact same item appears on multiple days, TRIM the later one.
    """
    suggestions: List[Dict[str, Any]] = []

    # (1) & (2) within-day ordering
    for region, days in allowed.items():
        for day in sorted(days.keys()):
            items = allowed[region][day]
            if len(items) < 2:
                continue
            tags = [_tag_token(x) for x in items]

            meal_idx = next((i for i,t in enumerate(tags) if t["is_meal"]), None)
            shop_idx = next((i for i,t in enumerate(tags) if t["is_shop"]), None)
            if meal_idx is not None and shop_idx is not None and shop_idx < meal_idx:
                a, b = items[shop_idx], items[meal_idx]
                suggestions.append({
                    "region": region, "day": day, "action": "swap",
                    "title": f"[{region}] Swap “{a}” ⇄ “{b}” (Day {day})",
                    "reason": "Eat before shopping to reduce fatigue and queue risk.",
                    "impact": "Smoother pacing; better energy for browsing.",
                    "confidence": 0.8,
                    "details": {"a": a, "b": b}
                })

            eve_idx = next((i for i,t in enumerate(tags) if t["is_evening"]), None)
            if eve_idx is not None and eve_idx == 0 and len(items) >= 2:
                a, b = items[0], items[1]
                suggestions.append({
                    "region": region, "day": day, "action": "swap",
                    "title": f"[{region}] Swap “{a}” ⇄ “{b}” (Day {day})",
                    "reason": "Move observatory/night view later for best timing.",
                    "impact": "Better views; natural day→night flow.",
                    "confidence": 0.75,
                    "details": {"a": a, "b": b}
                })

    # (3) duplicates across days → trim later occurrence
    occ = _collect_all_items(allowed)
    for item_lower, places in occ.items():
        if len(places) <= 1:
            continue
        latest_region, latest_day = places[-1]
        suggestions.append({
            "region": latest_region, "day": latest_day, "action": "trim",
            "title": f"[{latest_region}] Trim duplicate “{item_lower.title()}” (Day {latest_day})",
            "reason": "Remove repeated stop to create breathing room.",
            "impact": "Frees time; reduces redundancy.",
            "confidence": 0.7,
            "details": {"item": item_lower}
        })

    # Filter illegal and dedupe
    legal = [s for s in suggestions if _suggestion_legal(s, allowed)]
    seen, uniq = set(), []
    for s in legal:
        key = (s["region"], s["day"], s["action"], json.dumps(s.get("details", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq[:max_sugs]

# -----------------------------------------------------------------------------
# Prompt builder, validation & polishing
# -----------------------------------------------------------------------------
def build_reviewer_prompt(enriched: Dict[str, List[Dict[str, Any]]]):
    """
    Strict reviewer prompt with stronger structure + requirement for at least one
    concrete micro-edit when a legal option exists. Adds title/impact/confidence.
    """
    allowed = _build_allowed_map(enriched)
    has_legal = _has_legal_option(allowed)

    must_propose_text = (
        "If there is literally no legal edit (per constraints), suggestions may be empty."
        if not has_legal else
        "Unless the plan clearly deserves a score ≥ 9, provide at least one valid suggestion."
    )

    days_available = {region: sorted(days.keys()) for region, days in allowed.items()}

    system_text = f"""
You are an itinerary reviewer. Make SMALL, SURGICAL edits only.
Never add new places. Only use items from AllowedPlaces.
Only reference day numbers that exist in AllowedPlaces for that region.

Legal actions:
  - "swap": exchange two items within the SAME day
  - "move": move ONE item to an ADJACENT day (day±1)
  - "trim": remove a duplicate or low-value item
Forbidden:
  - New items not in AllowedPlaces
  - Moving to non-adjacent days
  - Mentioning days not present in AllowedPlaces
  - Vague advice with no concrete edit

Scoring (0–10):
  9–10 excellent; 7–8 good; 5–6 okay; 3–4 weak; 0–2 very weak.

Coverage rule:
  - {must_propose_text}

Prefer high-impact, realistic micro-edits:
  - Meals before big shopping blocks
  - Night views/observatories later in the day
  - Trim duplicates across days
  - Balance adjacent days (avoid one overloaded next to one empty)

Return JSON ONLY with this shape:
{{
  "overallScore": 0-10,
  "strengths": string[] (max 6),
  "risks": string[] (max 6),
  "suggestions": [ // max 10, highest impact first
    {{
      "region": "<one of: {', '.join(enriched.keys())}>",
      "day": <int>,
      "action": "swap" | "move" | "trim",
      "title": "<short headline for UI>",
      "reason": "≤200 chars, concrete",
      "impact": "<why this helps (≤120 chars)>",
      "confidence": <0..1>,
      "details": {{
        // swap: "a":"<item>","b":"<item>"
        // move: "item":"<item>","toDay":<int> (toDay = day±1)
        // trim: "item":"<item>"
      }}
    }}
  ]
}}

Hard constraints:
  - Every <item> MUST exist in AllowedPlaces[region][day] (source day)
  - For move, toDay must be day±1 and exist in data
Respond with JSON only.
"""
    user_payload = {
        "Cities": list(enriched.keys()),
        "Days": enriched,
        "AllowedPlaces": allowed,
        "DaysAvailable": days_available,
    }
    return system_text, user_payload, allowed

def _suggestion_legal(s: Dict[str, Any], allowed: Dict[str, Dict[int, List[str]]]) -> bool:
    try:
        region = s.get("region")
        day    = int(s.get("day"))
        action = (s.get("action") or "").lower()
        d      = s.get("details", {}) or {}
        if region not in allowed or day not in allowed[region]:
            return False
        items = allowed[region][day]
        canon = lambda x: (x or "").strip().lower()

        if action == "swap":
            a, b = d.get("a"), d.get("b")
            return bool(
                a and b and
                any(canon(a)==canon(x) for x in items) and
                any(canon(b)==canon(y) for y in items)
            )

        if action == "move":
            item, toDay = d.get("item"), d.get("toDay")
            if not (item and isinstance(toDay, int)):
                return False
            if toDay not in allowed[region]:
                return False
            return (toDay in (day-1, day+1)) and any(canon(item)==canon(x) for x in items)

        if action == "trim":
            item = (d.get("item") or "").strip()
            if not item or not any(canon(item)==canon(x) for x in items):
                return False
            # Only allow a trim if the same item exists on a different day in the same region
            exists_elsewhere = any(
                (other_day != day) and any(canon(item)==canon(x) for x in allowed[region][other_day])
                for other_day in allowed[region].keys()
            )
            return exists_elsewhere

        return False
    except Exception:
        return False

def _pick_fallback_suggestion(allowed: Dict[str, Dict[int, List[str]]]) -> Dict[str, Any]:
    for region, days in allowed.items():
        for d in sorted(days.keys()):
            items = days[d]
            if len(items) >= 3:
                return {
                    "region": region, "day": d, "action": "trim",
                    "title": f"[{region}] Trim “{items[-1]}” (Day {d})",
                    "reason": "Slightly reduce overload in this day.",
                    "impact": "Creates breathing room.",
                    "confidence": 0.6,
                    "details": {"item": items[-1]}
                }
    for region, days in allowed.items():
        for d in sorted(days.keys()):
            items = days[d]
            if len(items) >= 2:
                return {
                    "region": region, "day": d, "action": "swap",
                    "title": f"[{region}] Swap “{items[0]}” ⇄ “{items[1]}” (Day {d})",
                    "reason": "Small reordering for better flow.",
                    "impact": "Smoother sequence.",
                    "confidence": 0.6,
                    "details": {"a": items[0], "b": items[1]}
                }
    for region, days in allowed.items():
        day_nums = sorted(days.keys())
        for d in day_nums:
            items = days[d]
            if not items:
                continue
            for toDay in (d+1, d-1):
                if toDay in days:
                    return {
                        "region": region, "day": d, "action": "move",
                        "title": f"[{region}] Move “{items[-1]}” → Day {toDay} (Day {d})",
                        "reason": "Balance adjacent days.",
                        "impact": "Evens out pacing.",
                        "confidence": 0.6,
                        "details": {"item": items[-1], "toDay": toDay}
                    }
    return {}

def _validate_and_polish(model_json: Dict[str, Any], allowed: Dict[str, Dict[int, List[str]]]) -> Dict[str, Any]:
    out = {
        "overallScore": model_json.get("overallScore", 0),
        "strengths":    model_json.get("strengths") or [],
        "risks":        model_json.get("risks") or [],
        "suggestions":  model_json.get("suggestions") or [],
    }
    if not isinstance(out["strengths"], list): out["strengths"] = []
    if not isinstance(out["risks"], list):     out["risks"] = []
    if not isinstance(out["suggestions"], list): out["suggestions"] = []

    legal: List[Dict[str, Any]] = []
    for s in out["suggestions"]:
        if not isinstance(s, dict):
            continue
        if _suggestion_legal(s, allowed):
            s["action"]     = str(s.get("action", "")).lower()
            s["reason"]     = (s.get("reason") or "")[:200]
            s["impact"]     = (s.get("impact") or "")[:120]
            s["title"]      = (s.get("title")  or "").strip() or None
            try:
                conf = float(s.get("confidence", 0.6))
                s["confidence"] = max(0.0, min(1.0, conf))
            except Exception:
                s["confidence"] = 0.6
            legal.append(s)

    out["suggestions"] = legal[:10]

    try:
        out["overallScore"] = max(0, min(10, int(out["overallScore"])))
    except Exception:
        out["overallScore"] = 0

    out["strengths"] = out["strengths"][:6]
    out["risks"]     = out["risks"][:6]
    return out

def _prune_bad_risks(out: Dict[str, Any], allowed: Dict[str, Dict[int, List[str]]]) -> None:
    """Drop ‘duplicate across days’ claims when all regions have <2 days."""
    if not out.get("risks"):
        return
    few_days_everywhere = all(len(m.keys()) < 2 for m in allowed.values())
    if few_days_everywhere:
        out["risks"] = [x for x in out["risks"] if "duplicate" not in x.lower()]

# -----------------------------------------------------------------------------
# Routes: static + data
# -----------------------------------------------------------------------------
@app.route("/")
def serve_index():
    # Serve your index.html file
    return send_from_directory(FRONTEND_ROOT, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    # Serve other frontend HTML files (e.g. itineraries.html, cities.html)
    full = safe_join(FRONTEND_ROOT, filename)
    if full and os.path.isfile(full):
        return send_from_directory(FRONTEND_ROOT, filename)
    # fallback to 404.html if exists
    if os.path.isfile(os.path.join(FRONTEND_ROOT, "404.html")):
        return send_from_directory(FRONTEND_ROOT, "404.html"), 404
    abort(404)

@app.route("/static/<path:path>")
def serve_static(path):
    # Serve JS, CSS, and images
    full = safe_join(STATIC_ROOT, path)
    if not (full and os.path.isfile(full)):
        abort(404)
    return send_from_directory(STATIC_ROOT, path)



@app.route("/data/<city>/<filename>")
def load_city_data(city, filename):
    """Serve JSON from /data/<city>/<filename> so the front end can fetch lists."""
    try:
        safe_city = city.replace("..", "")
        safe_file = filename.replace("..", "")
        data_path = os.path.join(DATA_ROOT, safe_city, safe_file)
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()
        return app.response_class(response=data, status=200, mimetype="application/json")
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.exception("Failed to read file")
        return jsonify({"error": f"Failed to read file: {e}"}), 500


@app.route("/healthz")
def healthz():
    return "ok", 200

# -----------------------------------------------------------------------------
# AI Reviewer endpoint
# -----------------------------------------------------------------------------
@app.route("/ai/review", methods=["POST"])
def ai_review():
    """
    Body: {
      "plan": { "tokyo":[{"day":1,"location":"...","activities":"..."}], ... },
    }
    Returns: {
      overallScore, strengths[], risks[], suggestions[]
    }
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "Server missing OPENAI_API_KEY"}), 500

    body = request.get_json(silent=True) or {}
    plan = body.get("plan") or {}
    if not isinstance(plan, dict):
        return jsonify({"error": "Invalid plan payload. Expected object keyed by city."}), 400

    cities = [c for c in plan.keys() if c in CITY_DIRS]
    if not cities:
        return jsonify({"error": "No supported cities found in plan."}), 400

    try:
        kb = load_kb_for(cities)
        enriched = enrich_with_kb(plan, kb)
    except Exception as e:
        logger.exception("KB load/enrich failed")
        return jsonify({"error": f"Internal: failed to load KB: {e}"}), 500

    system_text, user_payload, allowed = build_reviewer_prompt(enriched)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.0,  # deterministic
            top_p=1.0,
            presence_penalty=0,
            frequency_penalty=0,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        model_json = json.loads(raw)
        polished = _validate_and_polish(model_json, allowed)
        _prune_bad_risks(polished, allowed)

        # If suggestions are empty or too few, augment with deterministic heuristics
        if len(polished.get("suggestions", [])) < 2 and _has_legal_option(allowed):
            extra = _heuristic_suggestions(enriched, allowed, max_sugs=3)
            existing_keys = {
                (s["region"], s["day"], s["action"], json.dumps(s.get("details", {}), sort_keys=True))
                for s in polished["suggestions"]
            }
            for s in extra:
                key = (s["region"], s["day"], s["action"], json.dumps(s.get("details", {}), sort_keys=True))
                if key not in existing_keys:
                    polished["suggestions"].append(s)
                    existing_keys.add(key)
            polished["suggestions"] = polished["suggestions"][:5]

        return jsonify(polished)
    except Exception as e:
        logger.exception("AI error")
        # Last-resort fallback so the UI shows something if a legal option exists
        try:
            if _has_legal_option(allowed):
                fb = _pick_fallback_suggestion(allowed)
                return jsonify({
                    "overallScore": 6,
                    "strengths": [],
                    "risks": ["AI parsing failed; using safe fallback."],
                    "suggestions": [fb] if fb else []
                }), 200
        except Exception:
            pass
        return jsonify({"error": f"AI error: {e}"}), 500
@app.route("/ai/debug_allowed", methods=["POST"])
def ai_debug_allowed():
    body = request.get_json(silent=True) or {}
    plan = body.get("plan") or {}
    cities = [c for c in plan.keys() if c in CITY_DIRS]
    if not cities:
        return jsonify({"error": "No supported cities in plan."}), 400
    kb = load_kb_for(cities)
    enriched = enrich_with_kb(plan, kb)
    allowed = _build_allowed_map(enriched)
    return jsonify({"allowed": allowed, "enriched": enriched})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
