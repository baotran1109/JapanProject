# =============================================================================
# NEARBY PICKS
# =============================================================================
# For general stops like "Lunch", "Ramen", "Coffee" or "Shopping", suggest real
# restaurants / shops from the database close to where the traveler will be at
# that point of the day (the stop before it, or the day's start area).
#
# Ranking: rating (when known) minus a penalty for distance, so a 4.5★ place
# 10 minutes away can beat a 4.0★ place next door. Places without a rating
# (most cities outside Tokyo) are ranked mostly by distance.
#
# No Flask code here, so it can be tested on its own.
# =============================================================================

from typing import Any, Dict, List, Optional

import route_planner

MAX_PICKS = 3
SEARCH_RADIUS_KM = 1.5          # First look this close...
WIDE_SEARCH_RADIUS_KM = 4.0     # ...and widen if there aren't enough places
DEFAULT_RATING = 3.9            # Used when a place has no rating
DISTANCE_PENALTY_PER_KM = 0.4   # 1 km further away "costs" 0.4 stars

MEAL_TYPES = ("breakfast", "lunch", "dinner", "meal")

# Words in the stop name -> cuisines to look for
CUISINE_HINTS = {
    "ramen": ("ramen", "tsukemen"),
    "tsukemen": ("tsukemen", "ramen"),
    "sushi": ("sushi",),
    "yakiniku": ("yakiniku", "bbq"),
    "bbq": ("yakiniku", "bbq"),
    "curry": ("curry",),
    "katsu": ("tonkatsu", "katsu"),
    "tonkatsu": ("tonkatsu", "katsu"),
    "tempura": ("tempura",),
    "izakaya": ("izakaya",),
    "yakitori": ("yakitori",),
    "soba": ("soba",),
    "udon": ("udon",),
    "okonomiyaki": ("okonomiyaki",),
    "kaiseki": ("kaiseki",),
    "french": ("french",),
    "italian": ("italian",),
    "coffee": ("cafe", "coffee", "bakery"),
    "cafe": ("cafe", "coffee", "bakery"),
    "café": ("cafe", "coffee", "bakery"),
    "breakfast": ("cafe", "coffee", "bakery", "pancake"),
    "bakery": ("bakery", "pastry", "patisserie"),
    "dessert": ("dessert", "patisserie", "pastry", "cake", "ice cream", "confectionery", "sweets", "chocolate"),
    "sweets": ("dessert", "patisserie", "pastry", "cake", "ice cream", "confectionery", "sweets", "chocolate"),
    "cake": ("cake", "patisserie", "pastry"),
}
# Places that are coffee/snacks/sweets, not a proper lunch or dinner (checked in the
# cuisine, and in the name for cities without cuisine data)
SNACK_PLACE_WORDS = ("confectionery", "pastry", "patisserie", "cake", "chocolate", "ice cream",
                     "dessert", "taiyaki", "sweets", "coffee", "bakery", "donut", "doughnut",
                     "gelato", "crepe", "tea salon", "roaster")


def wants_picks(stop: Dict[str, Any]) -> Optional[str]:
    """
    Which collection to suggest from for this stop ("restaurants" / "shopping"), or None.
    Only for general stops: unknown "Lunch", or an area like "Lunch in Ginza".
    """
    if stop.get("located") and stop.get("kind") != "area":
        return None  # Already a specific place
    if not stop.get("generic") and stop.get("kind") != "area":
        return None  # A specific name we couldn't find (e.g. a typo), not "Lunch"
    if stop.get("type") in MEAL_TYPES or stop.get("type") == "snack":
        return "restaurants"
    if stop.get("type") == "shopping":
        return "shopping"
    return None


def _cuisine_filter(stop_name: str, stop_type: str):
    """Returns (wanted cuisines or None, whether to skip sweets-only places)"""
    words = stop_name.lower().replace("-", " ").split()
    for word in words:
        if word in CUISINE_HINTS:
            return CUISINE_HINTS[word], False
    if stop_type == "snack":
        return CUISINE_HINTS["cafe"], False
    # Plain "Lunch" / "Dinner": a real meal, not a coffee stand or cake shop
    return None, stop_type in ("lunch", "dinner", "meal")


def nearby(point: Dict[str, Any], city_kb: List[Dict[str, Any]], collection: str,
           stop_name: str = "", stop_type: str = "", exclude: Optional[set] = None,
           filter_cuisine: bool = True) -> List[Dict[str, Any]]:
    """Best places of one collection near a point"""
    wanted, skip_sweets = (_cuisine_filter(stop_name, stop_type)
                           if collection == "restaurants" and filter_cuisine else (None, False))
    exclude = exclude or set()

    def matches(place):
        if place.get("_kind") != collection or place.get("name") in exclude:
            return False
        if place.get("lat") is None or place.get("lng") is None:
            return False
        cuisine = (place.get("cuisine") or "").lower()
        if wanted and not any(w in cuisine for w in wanted):
            return False
        name = (place.get("name") or "").lower()
        if skip_sweets and any(w in cuisine or w in name for w in SNACK_PLACE_WORDS):
            return False
        if skip_sweets and not cuisine and "cafe" in name:
            return False  # Unknown cuisine and called "cafe": probably not a full meal
        return True

    candidates = []
    for place in city_kb:
        if not matches(place):
            continue
        d = route_planner.haversine_km(point["lat"], point["lng"], place["lat"], place["lng"])
        if d is not None and d <= WIDE_SEARCH_RADIUS_KM:
            candidates.append((d, place))

    # This city has no cuisine data at all (only Tokyo does): suggest any restaurant instead.
    # Where cuisines are known, "Coffee" with no café nearby simply gets no suggestions.
    if not candidates and wanted and not any(p.get("cuisine") for p in city_kb if p.get("_kind") == collection):
        return nearby(point, city_kb, collection, stop_name, stop_type, exclude, filter_cuisine=False)

    close = [c for c in candidates if c[0] <= SEARCH_RADIUS_KM]
    pool = close if len(close) >= MAX_PICKS else candidates

    def score(item):
        d, place = item
        rating = place.get("rating")
        rating = float(rating) if isinstance(rating, (int, float)) else DEFAULT_RATING
        return rating - DISTANCE_PENALTY_PER_KM * d

    best = sorted(pool, key=score, reverse=True)[:MAX_PICKS]
    out = []
    for d, place in best:
        leg = route_planner.estimate_leg(d, "")
        out.append({
            "name": place["name"],
            "cuisine": place.get("cuisine"),
            "rating": place.get("rating"),
            "price_level": place.get("price_level"),
            "area": place.get("area") if place.get("area") not in (None, "General") else None,
            "distance_km": round(d, 2),
            "minutes": leg["minutes"],
            "mode": leg["mode"],
            "google_maps_url": place.get("google_maps_url") or place.get("google_map_url"),
        })
    return out


def picks_for_schedule(schedule: List[Dict[str, Any]], city_kb: List[Dict[str, Any]],
                       anchor: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    For each general meal / shopping stop in a day's schedule (route_planner output),
    suggest places near where the traveler is at that time.
    Returns [{"step", "stop", "type", "near", "options": [...]}].
    """
    results = []
    used = set()  # Don't suggest the same restaurant for lunch and dinner
    for i, stop in enumerate(schedule):
        collection = wants_picks(stop)
        if not collection:
            continue

        # Where will the traveler be? The stop itself if it's an area ("Lunch in Ginza"),
        # else the closest located stop before it, else after it, else the day's start.
        point = None
        if stop.get("located"):
            point = {"lat": stop["lat"], "lng": stop["lng"], "name": stop.get("matched_name") or stop["name"]}
        else:
            located = lambda s: s.get("located") and s.get("lat") is not None
            before = next((s for s in reversed(schedule[:i]) if located(s)), None)
            after = next((s for s in schedule[i + 1:] if located(s)), None)
            ref = before or after
            if ref:
                point = {"lat": ref["lat"], "lng": ref["lng"], "name": ref["name"]}
            elif anchor and anchor.get("lat") is not None:
                point = {"lat": anchor["lat"], "lng": anchor["lng"],
                         "name": anchor.get("matched_name") or anchor.get("name")}
        if not point:
            continue

        options = nearby(point, city_kb, collection, stop["name"], stop.get("type", ""), used)
        if not options:
            continue
        used.update(o["name"] for o in options)
        results.append({
            "step": stop.get("step", i + 1),
            "stop": stop["name"],
            "type": stop.get("type"),
            "near": point["name"],
            "options": options,
        })
    return results
