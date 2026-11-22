# =============================================================================
# TRIP GENERATOR ("Plan my trip for me")
# =============================================================================
# Builds a full itinerary from the places in the database:
#   1. Score every attraction / shop by the traveler's interests (temples, vintage...)
#   2. Group places into days by area (each day = one part of the city, or a day trip)
#   3. Add lunch and dinner at real restaurants close to that day's sights
#   4. Schedule each day with the route optimizer and drop the weakest sights until
#      the day fits the chosen pace (same limits the AI review checks)
#
# No Flask and no AI here: same input -> same plan, and it costs nothing to run.
# =============================================================================

import re
import statistics
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import place_picks
import route_planner
from ai_reviewer import PACE_LIMITS, DEFAULT_PACE

# -----------------------------------------------------------------------------
# Interests
# -----------------------------------------------------------------------------

INTERESTS = {
    "food":        "Food",
    "temples":     "Temples & shrines",
    "vintage":     "Vintage shopping",
    "fashion":     "Fashion & designer shops",
    "museums":     "Museums & art",
    "nature":      "Parks & nature",
    "views":       "City views",
    "anime":       "Anime & games",
    "history":     "Castles & history",
    "theme_parks": "Theme parks",
    "daytrips":    "Day trips",
}

TEMPLE_WORDS = ("shrine", "temple", "jinja", "jingu", "taisha", "torii", "hasedera", "kiyomizu")
VINTAGE_WORDS = ("vintage", "second hand", "secondhand", "used", "furugi", "古着", "kindal", "trefac",
                 "ragtag", "kinji", "jam ", "2nd street", "bookoff", "brand off", "komehyo", "flamingo",
                 "chicago", "desert snow", "ragla", "casanova", "amore", "flea market", "pawn", "daikokuya",
                 "mandarake", "y2k", "thrift")
MUSEUM_WORDS = ("museum", " art", "art ", "gallery", "teamlab", "design sight", "pavilion")
NATURE_WORDS = ("park", "garden", "gyoen", "zoo", "forest", "falls", "beach", "cave", "lake", "mount",
                "owakudani", "bamboo", "abyss", "hill", "botanic", "cruise", "aquarium", "gorge", "island")
VIEW_WORDS = ("tower", "sky", "observatory", "observation", "city view", "harukas", "sea candle")
ANIME_WORDS = ("anime", "manga", "ghibli", "gundam", "animate", "mandarake", "akihabara", "nintendo",
               "pokemon", "gamers", "denden", "nakano broadway", "medicom", "radio kaikan", "radio center",
               "fujio", "toy", "akiba")
HISTORY_WORDS = ("castle", "palace", "imperial", "folk", "historical", "village", "minka", "mound",
                 "shiryokan", "clock tower", "studio park", "old town")
# Restaurants that are really dessert / tea shops (not a lunch or dinner)
SWEET_NAME_WORDS = ("sweet", "parlor", "parlour", "matcha", "maccha", "tea house", "kissa", "arabica",
                    "blue bottle", "starbucks", "coffee", "cafe", "roaster")
FOOD_SIGHT_WORDS = ("market", "yokocho", "ameyoko", "dotonbori", "nishiki", "food hall", "depachika",
                    "beer museum")
THEME_PARK_WORDS = route_planner.THEME_PARK_WORDS + ("harry potter", "universal")

# -----------------------------------------------------------------------------
# Tuning
# -----------------------------------------------------------------------------

KIND_BASE_SCORE = {"attractions": 2.0, "daytrips": 1.5, "shopping": 0.5}
INTEREST_BONUS = 3.0           # Per interest a place matches
DAY_RADIUS_KM = 3.0            # A city day stays within walking / a few stops of its center...
WIDE_DAY_RADIUS_KM = 6.0       # ...unless there aren't enough places that close
DAYTRIP_RADIUS_KM = 15.0       # A day trip covers one destination (Hakone, Nara...)
DISTANCE_PENALTY = 0.5         # Score lost per km from the day's center
HUB_PENALTY = 0.05             # Score lost per km from the city center (Osaka before Kobe)
HUB_PENALTY_MAX_KM = 40
SEED_SPREAD_KM = 2.5
KNOWN_AREA_RADIUS_KM = 2.5     # A day is "in Ginza" if its center is this close to Ginza...
AREA_RADIUS_KM = 2.0           # ...or else this close to an area from the database           # Days start in different parts of the city
# Minutes of sights (not meals) to aim for before checking with the optimizer
SIGHT_MINUTES = {"relaxed": 270, "normal": 390, "packed": 510}
MAX_SHOPS = {"relaxed": 4, "normal": 6, "packed": 9}
FIT_MARGIN_MIN = 20            # Stay a bit under the pace limits so the plan isn't on the edge
STATION_BUFFER_MIN = 45        # Getting to the station before a train / flight to the next city
MAX_DAYS_PER_CITY = 14
MAX_TRIP_DAYS = 30


def _norm(text: str) -> str:
    """'Sensō-ji' -> 'senso-ji' (lowercase, no accents)"""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def place_interests(place: Dict[str, Any]) -> Set[str]:
    """Which interests a place matches (from its name, collection and cuisine)"""
    name = " " + _norm(place.get("name")) + " "
    kind = place.get("_kind")
    words = set(re.findall(r"[a-z]+", name))
    has = lambda vocab: any(w in name for w in vocab)
    found = set()

    if kind == "restaurants":
        return {"food"}
    if has(THEME_PARK_WORDS):
        return {"theme_parks"}
    if has(TEMPLE_WORDS) or re.search(r"\w-(ji|dera|in)\b", name) or words & route_planner.TEMPLE_NAMES:
        found.add("temples")
    if kind == "shopping":
        if has(VINTAGE_WORDS):
            found.add("vintage")
        elif has(ANIME_WORDS):
            found.add("anime")
        else:
            found.add("fashion")
        return found
    if has(MUSEUM_WORDS):
        found.add("museums")
    if has(NATURE_WORDS):
        found.add("nature")
    if has(VIEW_WORDS):
        found.add("views")
    if has(ANIME_WORDS):
        found.add("anime")
    if has(HISTORY_WORDS):
        found.add("history")
    if has(FOOD_SIGHT_WORDS):
        found.add("food")
    if kind == "daytrips":
        found.add("daytrips")
    return found


# -----------------------------------------------------------------------------
# Candidates
# -----------------------------------------------------------------------------

def _point(place: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        return float(place["lat"]), float(place["lng"])
    except (KeyError, TypeError, ValueError):
        return None


def _km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return route_planner.haversine_km(a[0], a[1], b[0], b[1]) or 0.0


def _candidates(city_kb: List[Dict[str, Any]], city: str, wanted: Set[str]) -> List[Dict[str, Any]]:
    """Attractions / shops worth visiting for these interests, best first"""
    hub = route_planner.CITY_HUBS.get(city)
    places = [p for p in city_kb if p.get("_kind") in KIND_BASE_SCORE and _point(p) and p.get("name")]

    # "Far away" = a day trip. Some cities are spread out (Okinawa), so it's relative.
    dists = [_km(_point(p), hub) for p in places if p.get("_kind") != "daytrips"] if hub else []
    far_km = max(30.0, 2 * statistics.median(dists)) if dists else 30.0

    out, seen = [], set()
    for place in places:
        key = _norm(place["name"]).strip()
        if key in seen:
            continue  # Same name twice (chains): keep one
        seen.add(key)
        interests = place_interests(place)
        far = place.get("_kind") == "daytrips" or (hub is not None and _km(_point(place), hub) > far_km)
        if far and "daytrips" not in wanted:
            continue
        if "theme_parks" in interests and "theme_parks" not in wanted:
            continue
        if place.get("_kind") == "shopping" and not interests & wanted:
            continue  # Shops only for people who asked for shopping
        matched = interests & wanted
        score = KIND_BASE_SCORE[place["_kind"]] + INTEREST_BONUS * len(matched)
        rating = place.get("rating")
        if isinstance(rating, (int, float)):
            score += (float(rating) - 4.0)
        if hub is not None and not far:
            score -= HUB_PENALTY * min(_km(_point(place), hub), HUB_PENALTY_MAX_KM)
        out.append({
            "place": place,
            "name": place["name"],
            "point": _point(place),
            "score": score,
            "interests": sorted(interests),
            "matched": sorted(matched),
            "far": far,
            "theme_park": "theme_parks" in interests,
            "shop": place["_kind"] == "shopping",
        })
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def _stop(cand: Dict[str, Any]) -> Dict[str, Any]:
    """A candidate as a stop in route_planner format"""
    place = cand["place"]
    stop_type = route_planner.classify(place["name"], place["_kind"], place.get("cuisine"))
    return {
        "name": place["name"],
        "matched_name": place["name"],
        "lat": cand["point"][0],
        "lng": cand["point"][1],
        "kind": place["_kind"],
        "type": stop_type,
        "generic": False,
        "duration": route_planner.default_duration(place["name"], stop_type, place["_kind"]),
        "_score": cand["score"],
        "_interests": cand["matched"],
    }


# -----------------------------------------------------------------------------
# Building one city's days
# -----------------------------------------------------------------------------

def _center(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _pick_seed(pool: List[Dict[str, Any]], centers: List[Tuple[float, float]],
               want_far: bool) -> Optional[Dict[str, Any]]:
    """Best remaining place to build a day around, away from the other days if possible"""
    options = [c for c in pool if c["far"] == want_far]
    for c in options:
        if all(_km(c["point"], other) >= SEED_SPREAD_KM for other in centers):
            return c
    return options[0] if options else None


def _grow_day(seed: Dict[str, Any], pool: List[Dict[str, Any]], pace: str,
              minutes_budget: int) -> List[Dict[str, Any]]:
    """Add the best nearby places to a day until it has enough sights"""
    members = [seed]
    if seed["theme_park"]:
        return members  # A theme park is the whole day
    radius = DAYTRIP_RADIUS_KM if seed["far"] else DAY_RADIUS_KM
    used_minutes = _stop(seed)["duration"]
    shops = 1 if seed["shop"] else 0

    def nearby(r):
        center = _center([m["point"] for m in members])
        taken = {id(m) for m in members}
        out = []
        for c in pool:
            if id(c) in taken or c["theme_park"] or c["far"] != seed["far"]:
                continue
            d = _km(c["point"], center)
            if d <= r:
                out.append((c["score"] - DISTANCE_PENALTY * d, c))
        return sorted(out, key=lambda x: x[0], reverse=True)

    while used_minutes < minutes_budget:
        options = nearby(radius) or (nearby(WIDE_DAY_RADIUS_KM) if not seed["far"] else [])
        options = [(v, c) for v, c in options if not (c["shop"] and shops >= MAX_SHOPS[pace])]
        if not options:
            break
        _, best = options[0]
        members.append(best)
        used_minutes += _stop(best)["duration"]
        shops += best["shop"]
    return members


def _meal(point: Tuple[float, float], city_kb: List[Dict[str, Any]], meal: str,
          used: Set[str], food_lover: bool) -> Optional[Dict[str, Any]]:
    """A real restaurant near the day's sights, as a 'Lunch: ...' / 'Dinner: ...' stop"""
    options = place_picks.nearby({"lat": point[0], "lng": point[1]}, city_kb, "restaurants",
                                 stop_name=meal, stop_type=meal.lower(), exclude=used)
    options = [o for o in options if not any(w in _norm(o["name"]) for w in SWEET_NAME_WORDS)]
    if not options:
        return None
    # Food lovers get the best rated of the close options, others simply the closest good one
    choice = (max(options, key=lambda o: o.get("rating") or 0) if food_lover
              else min(options, key=lambda o: o["distance_km"]))
    place = next(p for p in city_kb if p.get("_kind") == "restaurants" and p.get("name") == choice["name"])
    used.add(choice["name"])
    name = f"{meal}: {choice['name']}"
    stop_type = meal.lower()
    return {
        "name": name,
        "matched_name": choice["name"],
        "lat": float(place["lat"]),
        "lng": float(place["lng"]),
        "kind": "restaurants",
        "type": stop_type,
        "generic": False,
        "duration": route_planner.default_duration(name, stop_type, "restaurants"),
        "_restaurant": choice,
    }


def _area_label(center: Tuple[float, float], areas: Dict[str, Dict[str, Any]],
                seed: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Name of the neighborhood a day is in ('Asakusa'), and its point (where the day starts).
    Famous neighborhoods first ("Ueno" rather than "Yushima"), then areas from the database.
    """
    def nearest(candidates, limit):
        best, best_d = None, None
        for area in candidates:
            d = _km(center, (area["lat"], area["lng"]))
            if d <= limit and (best_d is None or d < best_d):
                best, best_d = area, d
        return best

    known = [a for a in areas.values() if a.get("known")]
    others = [a for a in areas.values() if not a.get("known")]
    if seed["far"]:
        best = nearest(areas.values(), DAYTRIP_RADIUS_KM)
    else:
        best = nearest(known, KNOWN_AREA_RADIUS_KM) or nearest(others, AREA_RADIUS_KM)
    if not best:
        return "", None
    return best["name"], {"name": best["name"], "matched_name": best.get("matched_name") or best["name"],
                          "lat": best["lat"], "lng": best["lng"]}


def _fits(stops: List[Dict[str, Any]], city: str, anchor, limits: Dict[str, int],
          reserve: int, start_minute: int) -> Tuple[bool, Dict[str, Any]]:
    result = route_planner.DayPlanner(stops, city, start_minute, anchor).optimize()
    active = result["activity_minutes"] + result["travel_minutes"]
    ok = (active <= limits["active_minutes"] - FIT_MARGIN_MIN - reserve
          and result["end_minute"] <= limits["latest_end"] - FIT_MARGIN_MIN - reserve)
    return ok, result


def _theme(stops: List[Dict[str, Any]], area: str, far: bool) -> str:
    counts: Dict[str, int] = {}
    for s in stops:
        for i in s.get("_interests", []):
            counts[i] = counts.get(i, 0) + 1
    top = [INTERESTS[i] for i, _ in sorted(counts.items(), key=lambda x: -x[1])[:2]]
    what = " & ".join(top) if top else "Highlights"
    if far and area:
        return f"Day trip to {area}"
    return f"{what} around {area}" if area else what


def build_city_days(city: str, n_days: int, city_kb: List[Dict[str, Any]], wanted: Set[str], pace: str,
                    areas: Dict[str, Dict[str, Any]], used_restaurants: Set[str], used_places: Set[str],
                    reserve_last_day: int = 0, start_minute: int = 9 * 60) -> List[Dict[str, Any]]:
    """
    Plan n_days in one city. Returns days in route_planner format (stops in optimized order).
    used_restaurants / used_places: names already in the trip (updated), so nothing repeats.
    """
    limits = PACE_LIMITS.get(pace, PACE_LIMITS[DEFAULT_PACE])
    pool = [c for c in _candidates(city_kb, city, wanted) if _norm(c["name"]).strip() not in used_places]
    food_lover = "food" in wanted

    # How many of the days are day trips (only if asked, and never all of them)
    n_far_places = sum(1 for c in pool if c["far"])
    n_daytrips = 0
    if "daytrips" in wanted and n_far_places and n_days >= 2:
        n_daytrips = 1 if n_days < 5 else 2

    days, centers = [], []
    for k in range(n_days):
        if not pool:
            break
        want_far = k >= 1 and n_daytrips > 0 and (k == 1 or (n_daytrips == 2 and k == n_days - 2))
        seed = _pick_seed(pool, centers, want_far) or _pick_seed(pool, centers, not want_far)
        if not seed:
            break
        if seed["far"]:
            n_daytrips -= 1
        members = _grow_day(seed, pool, pace, SIGHT_MINUTES.get(pace, SIGHT_MINUTES["normal"]))
        center = _center([m["point"] for m in members])
        area, anchor = _area_label(center, areas, seed)
        reserve = (reserve_last_day + STATION_BUFFER_MIN) if (reserve_last_day and k == n_days - 1) else 0

        sights = [_stop(m) for m in members]
        meals = []
        if not seed["theme_park"]:
            # Travel day: lunch here, dinner after arriving in the next city
            for meal in (("Lunch",) if reserve else ("Lunch", "Dinner")):
                # No restaurant in our data near here (some day trips): a plain "Lunch" stop,
                # which the optimizer still schedules at lunchtime
                meals.append(_meal(center, city_kb, meal, used_restaurants, food_lover) or {
                    "name": meal, "type": meal.lower(), "generic": True, "kind": None,
                    "duration": route_planner.default_duration(meal, meal.lower())})

        # Drop the weakest sights until the optimized day fits the pace
        ok, result = _fits(sights + meals, city, anchor, limits, reserve, start_minute)
        while not ok and len(sights) > 1:
            weakest = min(sights[1:], key=lambda s: s["_score"])  # never drop the day's main sight
            sights.remove(weakest)
            ok, result = _fits(sights + meals, city, anchor, limits, reserve, start_minute)
        if not ok and reserve and meals:
            meals = meals[:1]  # Travel day: skip dinner here, eat after arriving
            ok, result = _fits(sights + meals, city, anchor, limits, reserve, start_minute)

        kept = {s["name"] for s in sights}
        for m in members:
            if m["name"] in kept:
                pool.remove(m)
        used_places.update(_norm(name).strip() for name in kept)
        centers.append(center)

        by_name = {s["name"]: s for s in sights + meals}
        ordered = [by_name[name] for name in result["optimized_order"]]
        days.append({
            "city": city,
            "area": area,
            "anchor": anchor,
            "theme": _theme(sights, area, seed["far"]),
            "stops": ordered,
            "schedule": result,
        })
    return days


# -----------------------------------------------------------------------------
# Whole trip
# -----------------------------------------------------------------------------

class GeneratorInputError(ValueError):
    pass


def parse_request(body: Dict[str, Any], supported: Dict[str, str]) -> Dict[str, Any]:
    """Validate {cities: [{city, days}], interests: [...], pace}"""
    cities_in = body.get("cities")
    if not isinstance(cities_in, list) or not cities_in:
        raise GeneratorInputError("Choose at least one city and how many days to spend there.")
    cities = []
    for entry in cities_in:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("city") or "").strip().lower()
        if key not in supported:
            raise GeneratorInputError(f"Unsupported city '{entry.get('city')}'")
        try:
            n = int(entry.get("days"))
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            cities.append({"key": key, "city": supported[key], "days": min(n, MAX_DAYS_PER_CITY)})
    if not cities:
        raise GeneratorInputError("Give at least one city one day or more.")
    if sum(c["days"] for c in cities) > MAX_TRIP_DAYS:
        raise GeneratorInputError(f"That's more than {MAX_TRIP_DAYS} days. Try a shorter trip.")
    interests = {str(i) for i in (body.get("interests") or []) if str(i) in INTERESTS}
    pace = str(body.get("pace") or DEFAULT_PACE).lower()
    if pace not in PACE_LIMITS:
        pace = DEFAULT_PACE
    return {"cities": cities, "interests": interests, "pace": pace}


def generate_trip(request: Dict[str, Any], kb: Dict[str, List[Dict[str, Any]]],
                  areas_for: Callable[[str], Dict[str, Dict[str, Any]]],
                  start_minute: int = 9 * 60) -> Dict[str, Any]:
    """
    request: output of parse_request. kb: {city_key: places}. areas_for(city_key) -> neighborhoods.
    Returns {"days": [...], "notes": [...]} with days in route_planner format plus
    'city_key', 'label', 'theme' and the optimized schedule.
    """
    days_out: List[Dict[str, Any]] = []
    notes: List[str] = []
    used_restaurants: Set[str] = set()
    used_places: Set[str] = set()
    cities = request["cities"]

    for idx, entry in enumerate(cities):
        nxt = cities[idx + 1] if idx + 1 < len(cities) else None
        reserve = route_planner.intercity_leg(entry["city"], nxt["city"])["minutes"] if nxt else 0
        city_days = build_city_days(
            entry["city"], entry["days"], kb.get(entry["key"], []), request["interests"], request["pace"],
            areas_for(entry["key"]), used_restaurants, used_places, reserve, start_minute)
        if len(city_days) < entry["days"]:
            notes.append(f"{entry['city']}: only enough places in our data for {len(city_days)} "
                         f"of {entry['days']} days — the rest are left free.")
        for n, day in enumerate(city_days, start=1):
            day.update({"city_key": entry["key"], "day_in_city": n,
                        "label": f"{entry['city']} · Day {n}", "day": len(days_out) + 1})
            days_out.append(day)
        if nxt and city_days:
            notes.append(f"{city_days[-1]['label']} ends early to leave time to travel to {nxt['city']}.")
    return {"days": days_out, "notes": notes}
