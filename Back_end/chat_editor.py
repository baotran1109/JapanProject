# =============================================================================
# CHAT EDITS ("make day 2 more relaxed", "add a ramen dinner near Shibuya")
# =============================================================================
# The language model reads the trip (with schedule facts) and the traveler's
# message, and answers with a short reply plus a list of edit operations.
# It never writes place names on its own for food or shops: it asks for
# "add_meal" / "add_shops" and we pick real places from the database near
# that day's stops. Every operation is checked here; anything that doesn't fit
# the real plan is rejected with a reason instead of being applied.
#
# No Flask here: the app passes in the place lookups and the model call.
# =============================================================================

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

import place_picks
import route_planner
import trip_generator

MEALS = ("breakfast", "lunch", "dinner")
SHOP_INTERESTS = ("vintage", "fashion", "anime")
MAX_SHOPS = 6
MAX_OPERATIONS = 12
NAME_MATCH_MIN = 85          # How close a name the model writes must be to a stop in the day
CATALOG_PER_CITY = 120       # Sights per city listed for the model
HISTORY_TURNS = 6

OPERATIONS = ("add_place", "add_meal", "add_shops", "remove_stop", "move_stop", "reorder_day",
              "set_area", "swap_days")

SYSTEM_PROMPT = """You edit a traveler's Japan trip plan based on their message.

You get the trip (days with their stops, minutes per stop, the schedule end time, travel time
and issues found by our checks), a catalog of sights in each city, and the message.

Answer with a short friendly "reply" (1-3 sentences, say what you changed and why) and the
"operations" that make the change. Operations:
- add_place: add a sight to "day". "name" MUST be copied exactly from that city's catalog.
  "after": an existing stop it should follow (or null for the end).
- add_meal: add or replace a meal. "meal" = breakfast/lunch/dinner, optional "cuisine" (e.g. "ramen",
  "sushi", "italian") and "after" = the stop it should be near. We choose a real restaurant.
  It replaces the day's existing stop for that meal, if any.
- add_shops: add "count" (1-6) shops of an "interest" (vintage / fashion / anime) near that day's stops.
- remove_stop: remove "name" (an existing stop of that day).
- move_stop: move "name" from "day" to "to_day" (same city only).
- reorder_day: put "day" in the order given in "order" (existing stop names).
- set_area: set the area / neighborhood the day starts in ("area").
- swap_days: swap the plans of "day" and "to_day" (same city only).

Rules:
- Days are numbered as in the trip you were given ("day": 1 is the first day of the whole trip),
  also after a swap_days in the same answer: always use the numbers as they are now.
- Use existing stop names exactly as written in the trip.
- Never invent places: sights only from the catalog; food via add_meal, shops via add_shops.
- "More relaxed" = remove the least important stops so the day ends earlier or has less travel;
  keep meals. "More packed" = add catalog sights near the day's other stops.
- Keep changes minimal: only what the message asks for.
- If the request can't be done with these operations (e.g. adding a new day, booking tickets),
  return no operations and explain in the reply what the traveler can do instead.
- Unused fields are null."""

CHAT_SCHEMA = {
    "name": "trip_edit",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["reply", "operations"],
        "properties": {
            "reply": {"type": "string"},
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["op", "day", "to_day", "name", "after", "meal", "cuisine", "interest",
                                 "count", "order", "area"],
                    "properties": {
                        "op": {"type": "string", "enum": list(OPERATIONS)},
                        "day": {"type": "integer"},
                        "to_day": {"type": ["integer", "null"]},
                        "name": {"type": ["string", "null"]},
                        "after": {"type": ["string", "null"]},
                        "meal": {"type": ["string", "null"], "enum": ["breakfast", "lunch", "dinner", None]},
                        "cuisine": {"type": ["string", "null"]},
                        "interest": {"type": ["string", "null"], "enum": ["vintage", "fashion", "anime", None]},
                        "count": {"type": ["integer", "null"]},
                        "order": {"type": ["array", "null"], "items": {"type": "string"}},
                        "area": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


# -----------------------------------------------------------------------------
# What the model sees
# -----------------------------------------------------------------------------

def build_catalog(city_keys: List[str], kb: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    """Sights per city as 'Name (tags)', best known first"""
    out = {}
    for key in city_keys:
        rows = []
        seen = set()
        for place in kb.get(key, []):
            if place.get("_kind") not in ("attractions", "daytrips") or not place.get("name"):
                continue
            name = place["name"]
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            tags = sorted(trip_generator.place_interests(place))
            if place.get("_kind") == "daytrips":
                tags.append(f"day trip: {place.get('area') or ''}".strip(": "))
            rows.append(f"{name} ({', '.join(tags)})" if tags else name)
        out[key.capitalize()] = rows[:CATALOG_PER_CITY]
    return out


def build_user_message(days: List[Dict[str, Any]], facts: List[Dict[str, Any]], catalog: Dict[str, List[str]],
                       message: str, pace: str, history: List[Dict[str, str]]) -> str:
    trip = []
    for i, (day, fact) in enumerate(zip(days, facts), start=1):
        trip.append({
            "day": i,
            "label": day["label"],
            "area": day.get("area") or None,
            "stops": [{"name": s["name"], "minutes": s.get("duration"), "kind": s.get("type")}
                      for s in fact.get("stops", day["stops"])],
            "ends": fact.get("end_time"),
            "travelMinutes": fact.get("travel_minutes"),
            "issues": fact.get("issues", []),
        })
    return json.dumps({
        "pace": pace,
        "trip": trip,
        "catalog": catalog,
        "earlierMessages": history[-HISTORY_TURNS:],
        "message": message,
    }, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Applying operations
# -----------------------------------------------------------------------------

class Rejected(Exception):
    pass


def _find_stop(day: Dict[str, Any], name: Optional[str]) -> int:
    """Index of the stop the model means (exact, case-insensitive, then fuzzy)"""
    if not name:
        raise Rejected("no stop name given")
    names = [s["name"] for s in day["stops"]]
    lowered = [n.lower() for n in names]
    if name.lower() in lowered:
        return lowered.index(name.lower())
    scored = [(fuzz.WRatio(name.lower(), n), i) for i, n in enumerate(lowered)]
    best = max(scored, default=(0, -1))
    if best[0] >= NAME_MATCH_MIN:
        return best[1]
    raise Rejected(f"“{name}” isn't on {day['label']}")


def _insert(day: Dict[str, Any], stop: Dict[str, Any], after: Optional[str]) -> None:
    if after:
        try:
            i = _find_stop(day, after)
            day["stops"].insert(i + 1, stop)
            return
        except Rejected:
            pass
    day["stops"].append(stop)


def _trip_names(days: List[Dict[str, Any]]) -> Set[str]:
    names = set()
    for d in days:
        for s in d["stops"]:
            names.add(s["name"].lower())
            if ":" in s["name"]:
                names.add(s["name"].split(":", 1)[1].strip().lower())
    return names


class Editor:
    """
    days: page-format days [{key, label, city, city_key, area, stops: [{name, duration}]}]
    points: per day, {stop name: (lat, lng)} for located stops, plus "__anchor__"
    find_place(city_key, text, near) -> place or None; kb_for(city_key) -> places
    """

    def __init__(self, days, points, find_place: Callable, kb_for: Callable):
        self.days = copy.deepcopy(days)
        self.points = [dict(p) for p in points]
        self.find_place = find_place
        self.kb_for = kb_for
        self.changed: Set[int] = set()
        self.changes: List[str] = []
        self.rejected: List[str] = []

    def _day(self, n: Optional[int], field: str = "day") -> int:
        if not isinstance(n, int) or not 1 <= n <= len(self.days):
            raise Rejected(f"there's no day {n}")
        return n - 1

    def _center(self, i: int, near: Optional[str] = None) -> Optional[Tuple[float, float]]:
        pts = self.points[i]
        if near:
            for name, p in pts.items():
                if name.lower() == near.lower():
                    return p
        located = [p for name, p in pts.items() if name != "__anchor__"]
        if located:
            return (sum(p[0] for p in located) / len(located), sum(p[1] for p in located) / len(located))
        if "__anchor__" in pts:
            return pts["__anchor__"]
        hub = route_planner.CITY_HUBS.get(self.days[i]["city"])
        return hub

    def apply(self, ops: List[Dict[str, Any]]) -> None:
        # Day numbers refer to the trip as it was sent, so swaps happen after everything else
        ops = sorted(ops[:MAX_OPERATIONS], key=lambda op: op.get("op") == "swap_days")
        for op in ops:
            try:
                getattr(self, "_" + op.get("op", ""))(op)
            except Rejected as e:
                self.rejected.append(f"{op.get('op', '').replace('_', ' ')}: {e}")
            except AttributeError:
                self.rejected.append(f"unknown operation {op.get('op')}")

    # ---- operations ----------------------------------------------------------

    def _add_place(self, op):
        i = self._day(op.get("day"))
        day = self.days[i]
        name = (op.get("name") or "").split(" (")[0].strip()  # catalog lines carry "(tags)"
        place = self.find_place(day["city_key"], name, None)
        if not place:
            raise Rejected(f"couldn't find “{name}” in our {day['city']} places")
        real = place.get("name") or name
        if real.lower() in _trip_names(self.days):
            raise Rejected(f"“{real}” is already in the trip")
        _insert(day, {"name": real, "duration": None}, op.get("after"))
        self.points[i][real] = (float(place["lat"]), float(place["lng"]))
        self.changed.add(i)
        self.changes.append(f"Added “{real}” to {day['label']}")

    def _add_meal(self, op):
        i = self._day(op.get("day"))
        day = self.days[i]
        meal = (op.get("meal") or "").lower()
        if meal not in MEALS:
            raise Rejected("say which meal (breakfast, lunch or dinner)")
        label = meal.capitalize()
        # Letters only ("ramen", "sushi", "italian"): the model sometimes adds stray symbols
        cuisine = " ".join(re.findall(r"[a-zà-ÿ]+", (op.get("cuisine") or "").lower()))[:30]
        point = self._center(i, op.get("after"))
        if not point:
            raise Rejected(f"couldn't tell where {day['label']} is")
        kb = self.kb_for(day["city_key"])
        exclude = {n for n in (s["name"] for d in self.days for s in d["stops"])}
        exclude |= {n.split(":", 1)[1].strip() for n in exclude if ":" in n}
        options = place_picks.nearby({"lat": point[0], "lng": point[1]}, kb, "restaurants",
                                     stop_name=cuisine or meal, stop_type=meal, exclude=exclude)
        note = ""
        if cuisine and cuisine not in place_picks.CUISINE_HINTS:
            # place_picks only knows common cuisines; look for others in the cuisine / name text
            fitting = [o for o in _all_near(point, kb, exclude)
                       if cuisine in ((o.get("cuisine") or "") + " " + o["name"]).lower()][:3]
            if fitting:
                options = fitting
            else:
                note = f" (no {cuisine} place nearby in our data)"
        elif cuisine and not options:
            options = place_picks.nearby({"lat": point[0], "lng": point[1]}, kb, "restaurants",
                                         stop_name=meal, stop_type=meal, exclude=exclude)
            note = f" (no {cuisine} place nearby in our data)"
        if not options:
            raise Rejected(f"no restaurant near {day['label']}'s stops in our data")
        choice = options[0]
        stop = {"name": f"{label}: {choice['name']}", "duration": None}

        # Replace the day's existing stop for this meal ("Dinner", "Dinner: X")
        existing = next((k for k, s in enumerate(day["stops"])
                         if s["name"].lower() == meal or s["name"].lower().startswith(meal + ":")), None)
        if existing is not None:
            old = day["stops"][existing]["name"]
            day["stops"][existing] = stop
            self.changes.append(f"{day['label']}: {old} → {stop['name']}{note}")
        else:
            _insert(day, stop, op.get("after"))
            self.changes.append(f"Added {stop['name']} to {day['label']}{note}")
        self.changed.add(i)

    def _add_shops(self, op):
        i = self._day(op.get("day"))
        day = self.days[i]
        interest = op.get("interest")
        if interest not in SHOP_INTERESTS:
            raise Rejected("say which kind of shops (vintage, fashion or anime)")
        count = max(1, min(int(op.get("count") or 3), MAX_SHOPS))
        point = self._center(i)
        if not point:
            raise Rejected(f"couldn't tell where {day['label']} is")
        taken = _trip_names(self.days)
        shops = [c for c in trip_generator._candidates(self.kb_for(day["city_key"]), day["city"], {interest})
                 if c["shop"] and interest in c["matched"] and c["name"].lower() not in taken]
        ranked = sorted(shops, key=lambda c: -(c["score"] - 0.5 * trip_generator._km(c["point"], point)))
        near = [c for c in ranked if trip_generator._km(c["point"], point) <= 3.0]
        if len(near) < count:  # Not enough close by: look a bit further
            near = [c for c in ranked if trip_generator._km(c["point"], point) <= 6.0]
        if not near:
            raise Rejected(f"no {interest} shops near {day['label']}'s stops in our data")
        added = near[:count]
        for c in added:
            day["stops"].append({"name": c["name"], "duration": None})
            self.points[i][c["name"]] = c["point"]
        self.changed.add(i)
        self.changes.append(f"Added {len(added)} {interest} shop{'s' if len(added) > 1 else ''} to {day['label']}: "
                            + ", ".join(c["name"] for c in added))

    def _remove_stop(self, op):
        i = self._day(op.get("day"))
        day = self.days[i]
        k = _find_stop(day, op.get("name"))
        removed = day["stops"].pop(k)
        self.points[i].pop(removed["name"], None)
        self.changed.add(i)
        self.changes.append(f"Removed “{removed['name']}” from {day['label']}")

    def _move_stop(self, op):
        i = self._day(op.get("day"))
        j = self._day(op.get("to_day"))
        if i == j:
            raise Rejected("that's the same day")
        if self.days[i]["city_key"] != self.days[j]["city_key"]:
            raise Rejected(f"{self.days[i]['label']} and {self.days[j]['label']} are in different cities")
        k = _find_stop(self.days[i], op.get("name"))
        stop = self.days[i]["stops"].pop(k)
        self.days[j]["stops"].append(stop)
        if stop["name"] in self.points[i]:
            self.points[j][stop["name"]] = self.points[i].pop(stop["name"])
        self.changed.update((i, j))
        self.changes.append(f"Moved “{stop['name']}” from {self.days[i]['label']} to {self.days[j]['label']}")

    def _reorder_day(self, op):
        i = self._day(op.get("day"))
        day = self.days[i]
        order = op.get("order") or []
        remaining = list(day["stops"])
        ordered = []
        for name in order:
            try:
                k = _find_stop({"stops": remaining, "label": day["label"]}, name)
            except Rejected:
                continue
            ordered.append(remaining.pop(k))
        if not ordered:
            raise Rejected("none of those stops are on that day")
        day["stops"] = ordered + remaining
        self.changed.add(i)
        self.changes.append(f"Reordered {day['label']}")

    def _set_area(self, op):
        i = self._day(op.get("day"))
        area = (op.get("area") or "").strip()[:80]
        if not area:
            raise Rejected("no area given")
        self.days[i]["area"] = area
        self.changed.add(i)
        self.changes.append(f"{self.days[i]['label']} now starts in {area}")

    def _swap_days(self, op):
        i = self._day(op.get("day"))
        j = self._day(op.get("to_day"))
        if i == j:
            raise Rejected("that's the same day")
        a, b = self.days[i], self.days[j]
        if a["city_key"] != b["city_key"]:
            raise Rejected(f"{a['label']} and {b['label']} are in different cities")
        a["stops"], b["stops"] = b["stops"], a["stops"]
        a["area"], b["area"] = b["area"], a["area"]
        self.points[i], self.points[j] = self.points[j], self.points[i]
        self.changed.update((i, j))
        self.changes.append(f"Swapped {a['label']} and {b['label']}")


def _all_near(point, kb, exclude, radius_km: float = 4.0) -> List[Dict[str, Any]]:
    """Every restaurant near a point, closest first (for cuisines place_picks doesn't know)"""
    out = []
    for p in kb:
        if p.get("_kind") != "restaurants" or p.get("name") in exclude or p.get("lat") is None:
            continue
        d = route_planner.haversine_km(point[0], point[1], p["lat"], p["lng"])
        if d is not None and d <= radius_km:
            out.append((d, p))
    out.sort(key=lambda x: x[0])
    return [{"name": p["name"], "cuisine": p.get("cuisine"), "rating": p.get("rating")} for _, p in out]
