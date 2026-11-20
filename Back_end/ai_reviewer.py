# =============================================================================
# AI ITINERARY REVIEWER
# =============================================================================
# Reviews a trip plan and suggests concrete edits the user can apply in one click.
#
# How it works:
# 1. Facts first: every day is scheduled with route_planner (real travel times,
#    meal / evening timing, how much the route optimizer would save).
# 2. Rule-based checks turn those facts into issues (overloaded day, no meal,
#    duplicate stop, lots of backtracking...) and safe suggestions.
# 3. The language model gets the facts + issues and writes the review: score,
#    summary, strengths, risks and suggested edits (strict JSON schema).
# 4. Every suggestion is checked against the real plan before it's returned,
#    and rule-based suggestions fill the gaps (or replace the AI if it fails).
#
# No Flask code here, so it can be tested on its own.
# =============================================================================

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import route_planner

# =============================================================================
# SETTINGS
# =============================================================================

# How much a day can hold, by the pace the traveler picked
PACE_LIMITS = {
    #           max active hours (sights + travel), latest finish
    "relaxed": {"active_minutes": 8 * 60,  "latest_end": 21 * 60},
    "normal":  {"active_minutes": 10 * 60, "latest_end": 22 * 60},
    "packed":  {"active_minutes": 12 * 60, "latest_end": 23 * 60},
}
DEFAULT_PACE = "normal"

LIGHT_DAY_MINUTES = 3 * 60         # Less than this of activities = a light day
REORDER_MIN_SAVING = 15            # Suggest reordering if it saves at least this many minutes
TRAVEL_HEAVY_SHARE = 0.4           # Travel taking >40% of the active day is a lot
MAX_SUGGESTIONS = 6

ACTIONS = ("reorder", "swap", "move", "remove")

# Short local knowledge per city, only sent for cities in the plan
CITY_TIPS = {
    "Tokyo": "Huge city: group stops by area (Shibuya/Harajuku/Shinjuku west; Asakusa/Ueno/Akihabara east; "
             "Ginza/Tsukiji/Tokyo Station central; Odaiba bay). Crossing the city costs 30-45 min. Rush hours "
             "7:30-9:30 and 17:30-19:30. Day trips (Hakone, Nikko, Kamakura, Fuji) take a full day.",
    "Kyoto": "Temples and shrines get crowded after 10am; Fushimi Inari and Kiyomizu-dera are best early. "
             "Arashiyama is far west, Fushimi south, Gion/Higashiyama east: don't zig-zag. Buses are slow and "
             "crowded; trains/subway are faster. Most temples close 16:30-17:30.",
    "Osaka": "Food city: Dotonbori/Namba (south) and Umeda (north) are the main hubs. Nara and Kobe are "
             "~45 min day trips. Universal Studios takes a whole day. Evenings are for street food in Namba.",
    "Sapporo": "Compact grid city; Otaru (40 min) and Noboribetsu (70 min) are half/full-day trips. "
               "Winter days are short and snowy: allow extra travel time.",
    "Hakodate": "Small city with streetcars. Morning market is best early; Mt. Hakodate night view is the "
                "highlight (go at sunset, ropeway queues). Goryokaku is 20 min from the bay area.",
    "Aomori": "Spread out: Hirosaki and Hachinohe are day trips by train with infrequent service, so plan "
              "around timetables. Nebuta Museum is next to Aomori station.",
    "Fukuoka": "Easy, compact city: Hakata (station) and Tenjin (shopping) are 5 min apart by subway. "
               "Dazaifu is a half-day trip. Yatai food stalls at Nakasu open in the evening.",
    "Okinawa": "No trains outside Naha's monorail: you need a car or long bus rides. Churaumi Aquarium is "
               "~2h north of Naha, so pair it with nearby northern sights instead of Naha sights.",
}


# =============================================================================
# 1-2. FACTS AND RULE-BASED ISSUES
# =============================================================================

def _canon(text: str) -> str:
    return " ".join((text or "").lower().split())


def analyze_trip(days: List[Dict[str, Any]], pace: str, start_minute: int = 9 * 60) -> Dict[str, Any]:
    """
    Schedule every day as planned and as optimized, and list issues per day.
    'days' come from prepare_trip_days() in app.py (stops already matched to places).
    """
    limits = PACE_LIMITS.get(pace, PACE_LIMITS[DEFAULT_PACE])
    out_days = []

    # Same place on several days (by matched place, else by name)
    seen: Dict[str, List[int]] = {}
    for i, day in enumerate(days):
        for stop in day["stops"]:
            if stop.get("generic"):
                continue
            key = _canon(stop.get("matched_name") or stop["name"])
            if i not in seen.setdefault(key, []):
                seen[key].append(i)
    duplicates = {key: idxs for key, idxs in seen.items() if len(idxs) > 1}

    for i, day in enumerate(days):
        planner = route_planner.DayPlanner(day["stops"], day["city"], start_minute, day.get("anchor"))
        current = planner.evaluate()
        best = planner.optimize()

        activity = current["activity_minutes"]
        travel = current["travel_minutes"]
        active = activity + travel
        end = current["end_minute"]
        meals = [s for s in day["stops"] if s.get("type") in ("breakfast", "lunch", "dinner", "meal")]

        issues = []
        if day["stops"] and active > limits["active_minutes"]:
            issues.append({"type": "overloaded", "excess": active - limits["active_minutes"],
                           "text": f"{_hours(active)} of sights and travel (more than {_hours(limits['active_minutes'])} for a {pace} pace)"})
        if day["stops"] and end > limits["latest_end"]:
            issues.append({"type": "late_finish", "excess": end - limits["latest_end"],
                           "text": f"Finishes around {current['end_time']}"})
        if len(day["stops"]) >= 4 and not meals:
            issues.append({"type": "no_meal", "text": "No meal break planned"})
        if travel >= 60 and active and travel / active > TRAVEL_HEAVY_SHARE:
            issues.append({"type": "travel_heavy",
                           "text": f"{_hours(travel)} of travel ({round(travel / active * 100)}% of the day)"})
        if best["changed"] and best["travel_saved_minutes"] >= REORDER_MIN_SAVING:
            issues.append({"type": "backtracking",
                           "text": f"A different order saves {best['travel_saved_minutes']} min of travel"})
        if best["changed"] and current["timing_issues_before"] - best["timing_issues_after"] >= 30:
            issues.append({"type": "timing", "text": "Some stops are at a bad time (meals / evening / closing time)"})
        for name in current["unmatched"]:
            issues.append({"type": "unknown_place", "text": f"“{name}” isn't in our map data"})
        for stop in day["stops"]:
            key = _canon(stop.get("matched_name") or stop["name"])
            if key in duplicates and duplicates[key][0] != i:
                first = days[duplicates[key][0]]["label"]
                issues.append({"type": "duplicate", "item": stop["name"],
                               "text": f"“{stop['name']}” is already on {first}"})
        for warning in current["warnings"]:
            if "km from the previous stop" in warning:
                issues.append({"type": "far_stop", "text": warning})

        out_days.append({
            "index": i + 1,
            "key": day.get("key"),
            "label": day["label"],
            "city": day["city"],
            "area": day.get("area") or "",
            "stops": [s["name"] for s in day["stops"]],
            "schedule": current["stops"],
            "start_time": current["start_time"],
            "end_time": current["end_time"],
            "activity_minutes": activity,
            "travel_minutes": travel,
            "light": bool(day["stops"]) and activity < LIGHT_DAY_MINUTES,
            "empty": not day["stops"],
            "issues": issues,
            "optimized_order": best["optimized_order"] if best["changed"] else None,
            "optimized_travel_minutes": best["travel_minutes"],
            "travel_saved_minutes": best["travel_saved_minutes"] if best["changed"] else 0,
            "timing_fixed": best["changed"] and current["timing_issues_before"] - best["timing_issues_after"] >= 30,
        })

    return {"pace": pace if pace in PACE_LIMITS else DEFAULT_PACE, "days": out_days,
            "cities": sorted({d["city"] for d in days})}


def _hours(minutes: float) -> str:
    minutes = int(round(minutes))
    h, m = divmod(minutes, 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


def rule_score(analysis: Dict[str, Any]) -> int:
    """
    A score from the facts alone (used if the AI is unavailable, and as a reference for it).
    Each day loses points per issue (more for bigger overloads / later finishes);
    the trip score is the average day, but never more than 2 above the worst day.
    """
    weights = {"no_meal": 0.7, "travel_heavy": 0.8, "backtracking": 0.7, "timing": 0.5,
               "unknown_place": 0.1, "duplicate": 0.5, "far_stop": 1.0}
    day_scores = []
    for day in analysis["days"]:
        if day["empty"]:
            continue
        penalty = 0.0
        for issue in day["issues"]:
            if issue["type"] == "overloaded":
                penalty += 1.5 + min(issue["excess"] / 120, 3.0)   # +1 per 2 extra hours
            elif issue["type"] == "late_finish":
                penalty += 0.7 + min(issue["excess"] / 120, 2.5)
            else:
                penalty += weights.get(issue["type"], 0.3)
        day_scores.append(max(1.0, 10.0 - penalty))
    if not day_scores:
        return 0
    average = sum(day_scores) / len(day_scores)
    return max(1, min(10, round(min(average, min(day_scores) + 2))))


def rule_suggestions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Safe, concrete edits worked out from the facts (no AI needed)"""
    suggestions = []
    days = analysis["days"]
    for day in days:
        types = {issue["type"] for issue in day["issues"]}

        if day["optimized_order"] and ("backtracking" in types or "timing" in types):
            saving = day["travel_saved_minutes"]
            suggestions.append({
                "action": "reorder", "day": day["index"],
                "title": f"Reorder {day['label']}",
                "reason": (f"Visiting in a different order saves about {saving} min of travel"
                           if saving >= REORDER_MIN_SAVING else
                           "Puts meals at meal times and evening spots in the evening"),
                "impact": f"Travel {_hours(day['travel_minutes'])} → {_hours(day['optimized_travel_minutes'])}",
            })

        for issue in day["issues"]:
            if issue["type"] == "duplicate":
                suggestions.append({
                    "action": "remove", "day": day["index"], "item": issue["item"],
                    "title": f"Remove the repeat of “{issue['item']}”",
                    "reason": issue["text"] + ", so this visit is probably a repeat.",
                    "impact": "Frees time for something new",
                })

        if "overloaded" in types and len(day["stops"]) >= 2:
            # Move the last non-meal stop to the lightest other day in the same city
            same_city = [d for d in days if d["city"] == day["city"] and d["index"] != day["index"]]
            target = min(same_city, key=lambda d: d["activity_minutes"], default=None)
            item = next((s["name"] for s in reversed(day["schedule"])
                         if s["type"] not in ("breakfast", "lunch", "dinner", "meal")), None)
            if item and target and target["activity_minutes"] + 120 < day["activity_minutes"]:
                suggestions.append({
                    "action": "move", "day": day["index"], "item": item, "to_day": target["index"],
                    "title": f"Move “{item}” to {target['label']}",
                    "reason": f"{day['label']} is overloaded while {target['label']} is lighter.",
                    "impact": "More balanced days",
                })
            elif item:
                suggestions.append({
                    "action": "remove", "day": day["index"], "item": item,
                    "title": f"Drop “{item}” from {day['label']}",
                    "reason": f"{day['label']} has more than fits comfortably in one day.",
                    "impact": "A more realistic day",
                })
    return suggestions


# =============================================================================
# 3. THE PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an experienced Japan travel planner reviewing a traveler's itinerary.

You get FACTS computed by a route planner: for each day, the stops in the traveler's order with
estimated arrival times, travel times between stops, detected issues, and how much a better order
would save. Trust these facts; don't invent travel times or opening hours that contradict them.

Write an honest, specific review:
- overallScore (1-10). Use the full range:
    9-10 realistic, well-paced, smart grouping, nothing important to fix
    7-8  good plan with a few fixable issues
    5-6  workable but tiring or inefficient (overloaded days, a lot of backtracking, missed meals)
    3-4  several days are unrealistic or badly ordered
    1-2  mostly not doable as written
  The reference score from the rule checks is given; stay within ±2 of it unless you explain why in the summary.
- summary: 1-2 sentences, the most important takeaway.
- strengths: up to 4 specific things that work (name places / days).
- risks: up to 5 specific problems, most important first (name places / days, use the facts).
- suggestions: up to 5 concrete edits, most valuable first. Allowed actions:
    "reorder": use the planner's better order for that day (only if the day has an optimizedOrder)
    "swap":    swap two stops within the same day   (item, otherItem)
    "move":    move one stop to another day in the SAME city   (item, toDay)
    "remove":  remove one stop from a day   (item)
  Use stop names EXACTLY as written in the plan. Day numbers are the "day" field of each day.
  Fill unused fields with "" or 0. Prefer fixing the detected issues; don't suggest edits for
  days that are fine. Respect the traveler's pace.
- Each suggestion: short title (max 60 chars), reason (why, max 200 chars), impact (benefit, max 120 chars).

In all text (summary, strengths, risks, titles, reasons), refer to days by their "label"
exactly as given (e.g. "Kyoto · Day 1"), never as "Day 3": the traveler only sees the labels.
The numeric "day" field is only for the suggestion's day / toDay values.

Write in plain, friendly English. No markdown."""

REVIEW_SCHEMA = {
    "name": "itinerary_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["overallScore", "summary", "strengths", "risks", "suggestions"],
        "properties": {
            "overallScore": {"type": "integer"},
            "summary": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action", "day", "item", "otherItem", "toDay", "title", "reason", "impact"],
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "day": {"type": "integer"},
                        "item": {"type": "string"},
                        "otherItem": {"type": "string"},
                        "toDay": {"type": "integer"},
                        "title": {"type": "string"},
                        "reason": {"type": "string"},
                        "impact": {"type": "string"},
                    },
                },
            },
        },
    },
}


def build_user_message(analysis: Dict[str, Any], reference_score: int) -> str:
    """Compact JSON with only what the model needs"""
    days = []
    for d in analysis["days"]:
        if d["empty"]:
            days.append({"day": d["index"], "label": d["label"], "city": d["city"], "stops": []})
            continue
        days.append({
            "day": d["index"],
            "label": d["label"],
            "city": d["city"],
            "startArea": d["area"] or None,
            "stops": [
                {
                    "name": s["name"],
                    "arrive": s["arrive"],
                    "minutes": s["duration"],
                    "kind": s["type"],
                    **({"travelFromPrevious": f"{s['travel_minutes']} min by {s['mode']}"}
                       if s.get("travel_minutes") is not None else {}),
                    **({"notOnMap": True} if not s["located"] else {}),
                }
                for s in d["schedule"]
            ],
            "ends": d["end_time"],
            "activityMinutes": d["activity_minutes"],
            "travelMinutes": d["travel_minutes"],
            "issues": [i["text"] for i in d["issues"]],
            **({"optimizedOrder": d["optimized_order"], "optimizedTravelMinutes": d["optimized_travel_minutes"]}
               if d["optimized_order"] else {}),
        })
    payload = {
        "pace": analysis["pace"],
        "referenceScore": reference_score,
        "cityTips": {c: CITY_TIPS[c] for c in analysis["cities"] if c in CITY_TIPS},
        "days": days,
    }
    return json.dumps(payload, ensure_ascii=False)


# =============================================================================
# 4. VALIDATION
# =============================================================================

def _find_item(name: str, stops: List[str]) -> Optional[str]:
    """Return the stop exactly as written in the plan, matching case-insensitively"""
    target = _canon(name)
    return next((s for s in stops if _canon(s) == target), None) if target else None


def validate_suggestion(s: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check a suggestion against the real plan; return a clean version or None"""
    days = {d["index"]: d for d in analysis["days"]}
    action = str(s.get("action") or "").lower()
    day = days.get(_to_int(s.get("day")))
    if action not in ACTIONS or not day or day["empty"]:
        return None

    out = {
        "action": action,
        "dayIndex": day["index"],
        "dayKey": day["key"],
        "dayLabel": day["label"],
        "title": str(s.get("title") or "").strip()[:60],
        "reason": str(s.get("reason") or "").strip()[:200],
        "impact": str(s.get("impact") or "").strip()[:120],
        "details": {},
    }

    if action == "reorder":
        if not day["optimized_order"]:
            return None
        out["details"] = {"order": day["optimized_order"]}
        out["title"] = out["title"] or f"Reorder {day['label']}"

    elif action == "swap":
        a = _find_item(s.get("item"), day["stops"])
        b = _find_item(s.get("otherItem") or s.get("other_item"), day["stops"])
        if not a or not b or a == b:
            return None
        out["details"] = {"a": a, "b": b}
        out["title"] = out["title"] or f"Swap “{a}” and “{b}”"

    elif action == "move":
        item = _find_item(s.get("item"), day["stops"])
        target = days.get(_to_int(s.get("toDay") or s.get("to_day")))
        if not item or not target or target["index"] == day["index"] or target["city"] != day["city"]:
            return None
        out["details"] = {"item": item, "toDayIndex": target["index"], "toDayKey": target["key"],
                          "toDayLabel": target["label"]}
        out["title"] = out["title"] or f"Move “{item}” to {target['label']}"

    elif action == "remove":
        item = _find_item(s.get("item"), day["stops"])
        if not item:
            return None
        out["details"] = {"item": item}
        out["title"] = out["title"] or f"Remove “{item}”"

    return out


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _suggestion_key(s: Dict[str, Any]) -> Tuple:
    return (s["action"], s["dayIndex"], json.dumps(s["details"], sort_keys=True, ensure_ascii=False))


def _clean_list(items: Any, limit: int) -> List[str]:
    if not isinstance(items, list):
        return []
    return [str(x).strip()[:300] for x in items if str(x or "").strip()][:limit]


# =============================================================================
# PUTTING IT TOGETHER
# =============================================================================

def review_trip(days: List[Dict[str, Any]], pace: str,
                ask_model: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Full review. 'ask_model(system, user, schema)' calls the language model and
    returns its parsed JSON; pass None (or let it raise) to use the rule-based review only.
    """
    analysis = analyze_trip(days, pace)
    reference = rule_score(analysis)

    model_out: Optional[Dict[str, Any]] = None
    if ask_model is not None:
        try:
            model_out = ask_model(SYSTEM_PROMPT, build_user_message(analysis, reference), REVIEW_SCHEMA)
        except Exception:
            model_out = None  # Caller logs the error; we fall back to rules

    suggestions: List[Dict[str, Any]] = []
    seen = set()

    def add(raw: Dict[str, Any]) -> None:
        clean = validate_suggestion(raw, analysis)
        if clean and _suggestion_key(clean) not in seen and len(suggestions) < MAX_SUGGESTIONS:
            seen.add(_suggestion_key(clean))
            suggestions.append(clean)

    if model_out:
        for raw in model_out.get("suggestions") or []:
            if isinstance(raw, dict):
                add(raw)
    for raw in rule_suggestions(analysis):
        add(raw)

    if model_out:
        try:
            score = int(model_out.get("overallScore"))
        except (TypeError, ValueError):
            score = reference
        score = max(1, min(10, score))
        result = {
            "overallScore": score,
            "summary": str(model_out.get("summary") or "").strip()[:400],
            "strengths": _clean_list(model_out.get("strengths"), 4),
            "risks": _clean_list(model_out.get("risks"), 5),
            "source": "ai",
        }
    else:
        issues = [f"{d['label']}: {i['text']}" for d in analysis["days"] for i in d["issues"]
                  if i["type"] != "unknown_place"]
        fine = [d["label"] for d in analysis["days"] if not d["empty"] and not d["issues"]]
        busiest = max((d for d in analysis["days"] if d["issues"]),
                      key=lambda d: len(d["issues"]), default=None)
        if busiest:
            summary = (f"{busiest['label']} needs the most attention: "
                       f"{busiest['issues'][0]['text'][0].lower()}{busiest['issues'][0]['text'][1:]}.")
        else:
            summary = "Your days look realistic and well paced."
        result = {
            "overallScore": reference,
            "summary": summary,
            "strengths": [f"{', '.join(fine[:4])} look{'s' if len(fine) == 1 else ''} well paced."] if fine else [],
            "risks": issues[:5],
            "source": "rules",
        }

    result["suggestions"] = suggestions
    result["referenceScore"] = reference
    result["pace"] = analysis["pace"]
    result["days"] = [
        {
            "index": d["index"], "key": d["key"], "label": d["label"], "city": d["city"],
            "start_time": d["start_time"], "end_time": d["end_time"],
            "activity_minutes": d["activity_minutes"], "travel_minutes": d["travel_minutes"],
            "travel_saved_minutes": d["travel_saved_minutes"],
            "issues": [i["text"] for i in d["issues"]],
            "empty": d["empty"],
        }
        for d in analysis["days"]
    ]
    return result
