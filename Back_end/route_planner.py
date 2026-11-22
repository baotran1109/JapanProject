# =============================================================================
# ROUTE PLANNER
# =============================================================================
# Plans the best visiting order for each day of a trip.
#
# What it does:
# - Optimizes every day separately (days are never mixed together)
# - Uses Held-Karp dynamic programming for small days, nearest neighbor +
#   2-opt for big ones, then a local search that also respects meal times
#   and evening activities (night views, illuminations)
# - Builds a timed schedule for each day (arrive / leave times)
# - Estimates local fares, travel between cities, and the best JR Pass window
#
# This file has no Flask code so it can be tested on its own.
# Place matching (name -> GPS) happens in app.py and is passed in here.
# =============================================================================

import re
from math import radians, sin, cos, asin, sqrt
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# TRAVEL MODEL
# =============================================================================

WALK_MAX_KM = 1.2          # Up to this distance we assume walking
LOCAL_MAX_KM = 30.0        # Up to this distance: metro / local trains
REGIONAL_MAX_KM = 150.0    # Up to this distance: regional trains (day trips)
DETOUR_FACTOR = 1.3        # Real paths are longer than a straight line

# Share of local train rides that are on JR lines (rest is metro / private)
JR_SHARE_BY_CITY = {
    "Tokyo": 0.4,      # Lots of Tokyo Metro / Toei
    "Kyoto": 0.5,      # Buses, Keihan, Hankyu, subway
    "Osaka": 0.5,      # Osaka Metro is big
    "Sapporo": 0.3,    # Subway and streetcar
    "Hakodate": 0.3,   # Mostly streetcar
    "Aomori": 0.7,
    "Fukuoka": 0.3,    # Fukuoka subway / Nishitetsu
    "Okinawa": 0.0,    # No JR at all
}
DEFAULT_JR_SHARE = 0.5

# Cities where getting around is mostly by bus / car instead of trains
CAR_CITIES = {"Okinawa"}


def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    """Distance between two GPS points in kilometers"""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (TypeError, ValueError):
        return None
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def estimate_leg(distance_km: float, city: str) -> Dict[str, Any]:
    """
    Estimate how you'd travel a leg inside a city and what it costs.
    Returns: {mode, minutes, fare_yen, jr_share}
    """
    d = distance_km
    if d <= WALK_MAX_KM:
        # ~4.5 km/h walking on real streets
        return {"mode": "walk", "minutes": round(d * DETOUR_FACTOR / 4.5 * 60), "fare_yen": 0, "jr_share": 0.0}

    if city in CAR_CITIES:
        # Bus / rental car: ~30 km/h door to door, plus waiting/parking
        return {
            "mode": "bus/car",
            "minutes": round(10 + d * DETOUR_FACTOR / 30 * 60),
            "fare_yen": round(min(240 + 25 * d, 2500)),
            "jr_share": 0.0,
        }

    if d <= LOCAL_MAX_KM:
        # Metro / local train: walk to station + wait (~12 min) + ~28 km/h
        return {
            "mode": "train",
            "minutes": round(12 + d * DETOUR_FACTOR / 28 * 60),
            "fare_yen": round(min(180 + 15 * d, 650)),
            "jr_share": JR_SHARE_BY_CITY.get(city, DEFAULT_JR_SHARE),
        }

    if d <= REGIONAL_MAX_KM:
        # Regional / rapid trains for day trips (Hakone, Nikko, Nara, Otaru...)
        return {
            "mode": "train",
            "minutes": round(25 + d * 1.25 / 60 * 60),
            "fare_yen": round(400 + 22 * d),
            "jr_share": 0.5,
        }

    # Very long legs inside one "city" day are usually a mistake in the plan
    return {
        "mode": "long-distance",
        "minutes": round(30 + d / 150 * 60),
        "fare_yen": round(1500 + 30 * d),
        "jr_share": 0.9,
    }


# =============================================================================
# TRAVEL BETWEEN CITIES
# =============================================================================

# Main station / airport area for each city (used for city-to-city travel)
CITY_HUBS = {
    "Tokyo": (35.6812, 139.7671),      # Tokyo Station
    "Kyoto": (34.9858, 135.7588),      # Kyoto Station
    "Osaka": (34.7334, 135.5001),      # Shin-Osaka
    "Fukuoka": (33.5897, 130.4207),    # Hakata
    "Aomori": (40.8270, 140.6930),     # Shin-Aomori
    "Hakodate": (41.9048, 140.6486),   # Shin-Hakodate-Hokuto
    "Sapporo": (43.0687, 141.3508),    # Sapporo Station
    "Okinawa": (26.2124, 127.6792),    # Naha
}

# Approximate one-way fares (reserved seat) and travel times between cities.
# "jr" = covered by the JR Pass. Nozomi trains need an extra supplement.
INTERCITY_ROUTES = {
    ("Tokyo", "Kyoto"):      {"mode": "shinkansen", "minutes": 135, "fare_yen": 14170, "jr": True},
    ("Tokyo", "Osaka"):      {"mode": "shinkansen", "minutes": 150, "fare_yen": 14720, "jr": True},
    ("Kyoto", "Osaka"):      {"mode": "train",      "minutes": 30,  "fare_yen": 580,   "jr": True},
    ("Tokyo", "Fukuoka"):    {"mode": "shinkansen", "minutes": 300, "fare_yen": 23390, "jr": True},
    ("Kyoto", "Fukuoka"):    {"mode": "shinkansen", "minutes": 170, "fare_yen": 16360, "jr": True},
    ("Osaka", "Fukuoka"):    {"mode": "shinkansen", "minutes": 150, "fare_yen": 15600, "jr": True},
    ("Tokyo", "Aomori"):     {"mode": "shinkansen", "minutes": 190, "fare_yen": 17670, "jr": True},
    ("Tokyo", "Hakodate"):   {"mode": "shinkansen", "minutes": 240, "fare_yen": 23430, "jr": True},
    ("Aomori", "Hakodate"):  {"mode": "shinkansen", "minutes": 90,  "fare_yen": 7830,  "jr": True},
    ("Hakodate", "Sapporo"): {"mode": "train",      "minutes": 225, "fare_yen": 9440,  "jr": True},
    ("Aomori", "Sapporo"):   {"mode": "train",      "minutes": 330, "fare_yen": 17260, "jr": True},
    ("Tokyo", "Sapporo"):    {"mode": "flight",     "minutes": 180, "fare_yen": 15000, "jr": False},
    ("Osaka", "Sapporo"):    {"mode": "flight",     "minutes": 200, "fare_yen": 17000, "jr": False},
    ("Kyoto", "Sapporo"):    {"mode": "flight",     "minutes": 240, "fare_yen": 17000, "jr": False},
    ("Fukuoka", "Sapporo"):  {"mode": "flight",     "minutes": 220, "fare_yen": 20000, "jr": False},
    ("Tokyo", "Okinawa"):    {"mode": "flight",     "minutes": 210, "fare_yen": 15000, "jr": False},
    ("Osaka", "Okinawa"):    {"mode": "flight",     "minutes": 190, "fare_yen": 13000, "jr": False},
    ("Kyoto", "Okinawa"):    {"mode": "flight",     "minutes": 240, "fare_yen": 13000, "jr": False},
    ("Fukuoka", "Okinawa"):  {"mode": "flight",     "minutes": 160, "fare_yen": 12000, "jr": False},
}

# Real JR Pass prices (nationwide, ordinary car)
JR_PASS_COSTS = {7: 50000, 14: 80000, 21: 100000}


def intercity_leg(city_from: str, city_to: str) -> Dict[str, Any]:
    """How to get from one city to another (known route or a distance-based estimate)"""
    route = INTERCITY_ROUTES.get((city_from, city_to)) or INTERCITY_ROUTES.get((city_to, city_from))
    hub_a, hub_b = CITY_HUBS.get(city_from), CITY_HUBS.get(city_to)
    distance = haversine_km(*hub_a, *hub_b) if hub_a and hub_b else None
    if route:
        return {**route, "distance_km": round(distance, 1) if distance else None, "estimated": False}

    # Unknown pair: rough shinkansen estimate from straight-line distance
    d = distance or 300.0
    if "Okinawa" in (city_from, city_to) or d > 900:
        return {"mode": "flight", "minutes": 200, "fare_yen": 15000, "jr": False,
                "distance_km": round(d, 1), "estimated": True}
    fare = 1500 + 33 * d if d <= 500 else 18000 + 18 * (d - 500)
    return {"mode": "shinkansen", "minutes": round(30 + d / 180 * 60), "fare_yen": round(fare), "jr": True,
            "distance_km": round(d, 1), "estimated": True}


# =============================================================================
# ACTIVITY TYPES & TIME WINDOWS
# =============================================================================

MEAL_WORDS = {
    "breakfast", "brunch", "lunch", "dinner", "ramen", "sushi", "yakitori", "katsu",
    "tempura", "restaurant", "cafe", "café", "coffee", "curry", "tonkatsu", "udon", "soba",
    "izakaya", "yakiniku", "okonomiyaki", "takoyaki", "gyoza", "bakery", "food", "eat",
    "taco", "tacos", "burger", "pizza", "steak", "malatang", "noodle", "noodles", "dumpling", "bistro",
    "diner", "grill",
}
# Temples whose name is written without the hyphen ("Kinkakuji"). We can't treat every
# word ending in "-ji" as a temple: Kichijoji and Koenji are neighborhoods.
TEMPLE_NAMES = {
    "sensoji", "kinkakuji", "ginkakuji", "kiyomizudera", "todaiji", "ryoanji", "nanzenji", "kenninji",
    "tofukuji", "zojoji", "gotokuji", "sengakuji", "daitokuji", "toji", "chusonji", "kotokuin",
}
# Cuisines (from the database) that are a coffee / sweets stop rather than a meal
SNACK_CUISINES = ("cafe", "coffee", "bakery", "pastry", "patisserie", "cake", "dessert", "confectionery",
                  "ice cream", "chocolate", "taiyaki", "sweets")
EVENING_WORDS = {
    "night view", "night", "sunset", "illumination", "observatory", "observation deck",
    "sky tree", "skytree", "tokyo tower", "shibuya sky", "bar", "izakaya", "nightlife",
}
# Coffee / sweets breaks: food, but not a meal, and no particular time
SNACK_WORDS = {"coffee", "cafe", "café", "tea", "dessert", "sweets", "cake", "ice cream", "snack", "bakery"}
# Shops (most open ~10:00-11:00 and close ~20:00-21:00). Markets open early, so they aren't here.
SHOP_WORDS = {
    "shopping", "shop", "souvenir", "souvenirs", "mall", "department store", "don quijote", "donki",
    "uniqlo", "outlet", "boutique", "bookoff", "loft", "parco", "isetan", "takashimaya", "mitsukoshi",
    "store", "vintage", "second hand", "secondhand", "thrift", "used clothing", "select shop", "flagship",
    "古着", "ストア", "ショップ", "百貨店",
}
# An unknown stop on a day where at least this share of known stops are shops is probably a shop too
SHOPPING_DAY_SHARE = 0.5
# Places that usually close around 16:30-17:30 (shrine/temple halls, museums, gardens...)
DAYTIME_WORDS = {
    "shrine", "temple", "jingu", "jinja", "taisha", "-ji", "-dera", "museum", "garden", "gardens",
    "zoo", "aquarium", "castle", "palace", "gallery", "teamlab",
}

# Minutes after midnight
BREAKFAST_WINDOW = (7 * 60, 10 * 60)
LUNCH_WINDOW = (11 * 60 + 30, 14 * 60)
DINNER_WINDOW = (17 * 60 + 30, 20 * 60 + 30)
EVENING_WINDOW = (17 * 60, 22 * 60)
DAYTIME_WINDOW = (9 * 60, 16 * 60)  # Arrive by 4pm so there's time before closing
SHOPPING_WINDOW = (10 * 60 + 30, 20 * 60)  # Most shops are open by 10:30 and until ~20:00
DAY_END_WARNING = 22 * 60  # Warn when a day runs past 10pm

MEAL_PENALTY_PER_MIN = 1.5     # How much we care about eating at normal times
EVENING_PENALTY_PER_MIN = 0.6  # Night views are nicer at night, but fine in the day
DAYTIME_PENALTY_PER_MIN = 2.0  # Arriving after closing time is a real problem
SHOPPING_PENALTY_PER_MIN = 1.0 # Shops are closed early in the morning and late at night
WAIT_COST_PER_MIN = 0.25       # Waiting for lunch/evening is OK, but not free
FIRST_STOP_WAIT_COST = 1.0     # Waiting before the first stop just wastes the morning
MAX_WAIT_MIN = 240             # Longest free time we'll leave before a stop...
MAX_WAIT_REST_OF_DAY_MIN = 600 # ...unless only evening stops / meals are left (a free afternoon)
MAX_FIRST_STOP_WAIT_MIN = 90   # Never start the day with a long wait
LATE_TYPES = ("evening", "dinner", "meal")


def classify(name: str, kind: Optional[str], cuisine: Optional[str] = None) -> str:
    """
    Decide what type of stop this is: breakfast / lunch / dinner / meal / snack /
    evening / daytime / shopping / visit.
    'kind' is the database collection the place came from (restaurants, shopping, ...)
    and 'cuisine' the restaurant's cuisine when the database has it.
    """
    text = (name or "").lower()
    cuisine = (cuisine or "").lower()
    if "breakfast" in text:
        return "breakfast"
    if "lunch" in text or "brunch" in text:
        return "lunch"
    if "dinner" in text:
        return "dinner"
    if any(w in text for w in EVENING_WORDS):
        return "evening"
    # Whole words only, so "tea" doesn't match "teamLab" or "steak"
    words = set(re.findall(r"[a-zà-ÿ]+", text))
    has_word = lambda vocab: any((w in text) if " " in w else (w in words) for w in vocab)

    # Cafés / bakeries / sweets: by cuisine, or by name ("BAM BI COFFEE")
    if any(w in cuisine for w in SNACK_CUISINES) or has_word(SNACK_WORDS):
        return "snack"
    if kind == "restaurants" or has_word(MEAL_WORDS) or "ラーメン" in text:
        return "meal"
    if any(w in text for w in DAYTIME_WORDS) or words & TEMPLE_NAMES:
        return "daytime"
    if kind == "shopping" or any(w in text for w in SHOP_WORDS):
        return "shopping"
    return "visit"


def _window_penalty(minute: float, window: Tuple[int, int], weight: float) -> float:
    start, end = window
    if minute < start:
        return (start - minute) * weight
    if minute > end:
        return (minute - end) * weight
    return 0.0


# How long a stop usually takes when the traveler didn't say (minutes)
THEME_PARK_WORDS = ("disneyland", "disneysea", "disney sea", "universal studios", "usj", "fuji-q", "sanrio puroland")
HOTEL_WORDS = ("check in", "check-in", "checkin", "check out", "check-out", "checkout", "hotel")
DEPARTMENT_STORE_WORDS = ("department store", "isetan", "takashimaya", "mitsukoshi", "daimaru", "hankyu",
                          "matsuya", "sogo", "seibu", "tokyu", "lumine", "parco", "shibuya 109", "don quijote",
                          "donki", "mall", "outlet")
BIG_DAYTIME_WORDS = ("museum", "aquarium", "zoo", "teamlab", "gallery")
DEFAULT_DURATIONS = {
    "breakfast": 40, "lunch": 60, "dinner": 75, "meal": 60, "snack": 30,
    "evening": 75, "daytime": 60, "shopping": 20, "visit": 60,
}


def classify_unknown_by_context(stops: List[Dict[str, Any]]) -> None:
    """
    Stops we couldn't find on the map default to "visit". On a shopping-heavy day
    ("CC STORE", "THE ELEPHANT" between other boutiques) they're almost certainly shops.
    """
    known = [s for s in stops if s.get("lat") is not None and not s.get("generic")]
    if not known:
        return
    share = sum(1 for s in known if s.get("type") == "shopping") / len(known)
    if share < SHOPPING_DAY_SHARE:
        return
    for s in stops:
        if s.get("type") == "visit" and s.get("lat") is None and not s.get("generic"):
            s["type"] = "shopping"


def default_duration(name: str, stop_type: str, kind: Optional[str] = None, generic: bool = False) -> int:
    """
    A realistic time for a stop the traveler didn't give a duration for.
    Example: a boutique ~20 min, a department store ~60, a museum ~2h, "Shopping in Ginza" ~90.
    """
    text = (name or "").lower()
    if any(w in text for w in THEME_PARK_WORDS):
        return 480
    if any(w in text for w in HOTEL_WORDS):
        return 20
    if stop_type == "shopping":
        if generic or kind == "area":
            return 90   # A shopping session in an area, not one shop
        if any(w in text for w in DEPARTMENT_STORE_WORDS):
            return 60
        return 20       # One shop / boutique
    if stop_type == "daytime" and any(w in text for w in BIG_DAYTIME_WORDS):
        return 120
    if kind == "daytrips":
        return 120
    if kind == "area" and stop_type == "visit":
        return 90       # Walking around a neighborhood
    return DEFAULT_DURATIONS.get(stop_type, 60)


def window_start(stop_type: str, arrive_minute: float) -> Optional[int]:
    """Earliest good time for this stop, if arriving before it is worth waiting for"""
    if stop_type == "lunch":
        return LUNCH_WINDOW[0]
    if stop_type == "dinner":
        return DINNER_WINDOW[0]
    if stop_type == "meal":
        # Before the end of lunch time → wait for lunch, otherwise for dinner
        return LUNCH_WINDOW[0] if arrive_minute <= LUNCH_WINDOW[1] else DINNER_WINDOW[0]
    if stop_type == "evening":
        return EVENING_WINDOW[0]
    if stop_type == "shopping":
        return SHOPPING_WINDOW[0]
    return None


def time_penalty(stop_type: str, arrive_minute: float) -> float:
    """Penalty for arriving at a stop at a bad time (0 = perfect)"""
    if stop_type == "breakfast":
        return _window_penalty(arrive_minute, BREAKFAST_WINDOW, MEAL_PENALTY_PER_MIN)
    if stop_type == "lunch":
        return _window_penalty(arrive_minute, LUNCH_WINDOW, MEAL_PENALTY_PER_MIN)
    if stop_type == "dinner":
        return _window_penalty(arrive_minute, DINNER_WINDOW, MEAL_PENALTY_PER_MIN)
    if stop_type == "meal":
        # A generic meal can be lunch or dinner, whichever is closer
        return min(
            _window_penalty(arrive_minute, LUNCH_WINDOW, MEAL_PENALTY_PER_MIN),
            _window_penalty(arrive_minute, DINNER_WINDOW, MEAL_PENALTY_PER_MIN),
        )
    if stop_type == "evening":
        return _window_penalty(arrive_minute, EVENING_WINDOW, EVENING_PENALTY_PER_MIN)
    if stop_type == "daytime":
        return _window_penalty(arrive_minute, DAYTIME_WINDOW, DAYTIME_PENALTY_PER_MIN)
    if stop_type == "shopping":
        return _window_penalty(arrive_minute, SHOPPING_WINDOW, SHOPPING_PENALTY_PER_MIN)
    return 0.0


def fmt_clock(minute: float) -> str:
    """540 -> '09:00', past midnight -> '01:30+1'"""
    minute = int(round(minute))
    clock = f"{(minute // 60) % 24:02d}:{minute % 60:02d}"
    return clock + "+1" if minute >= 24 * 60 else clock


# =============================================================================
# DAY PLANNING
# =============================================================================

class DayPlanner:
    """
    Optimizes the order of one day's stops.

    Each stop: {name, duration, lat, lng, type}
    Stops without GPS (e.g. "Lunch") are "flexible": they happen wherever you are,
    so they add no travel but still get scheduled at a sensible time.
    """

    HELD_KARP_MAX = 12  # Exact DP up to this many located stops

    def __init__(self, stops: List[Dict[str, Any]], city: str, start_minute: int,
                 anchor: Optional[Dict[str, Any]] = None):
        self.stops = stops
        self.city = city
        self.start_minute = start_minute
        self.anchor = anchor  # Where the day starts (hotel / day's area), optional
        n = len(stops)

        # Travel legs between every pair of located stops
        self.legs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for i in range(n):
            for j in range(n):
                if i != j and self._located(stops[i]) and self._located(stops[j]):
                    self.legs[(i, j)] = self._leg(stops[i], stops[j])
        self.anchor_legs: Dict[int, Dict[str, Any]] = {}
        if anchor and self._located(anchor):
            for i in range(n):
                if self._located(stops[i]):
                    self.anchor_legs[i] = self._leg(anchor, stops[i])

    @staticmethod
    def _located(stop: Dict[str, Any]) -> bool:
        return stop.get("lat") is not None and stop.get("lng") is not None

    def _leg(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        d = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) or 0.0
        leg = estimate_leg(d, self.city)
        leg["distance_km"] = round(d, 2)
        return leg

    # ---- evaluating an order -------------------------------------------------

    def simulate(self, order: List[int]) -> Dict[str, Any]:
        """Walk through the day in this order and compute times, travel and penalties"""
        clock = float(self.start_minute)
        travel_total = 0
        wait_total = 0.0
        wait_cost = 0.0
        penalty = 0.0
        last_located: Optional[int] = None
        timeline = []

        for position, idx in enumerate(order):
            stop = self.stops[idx]
            leg = None
            if self._located(stop):
                if last_located is not None:
                    leg = self.legs[(last_located, idx)]
                elif self.anchor_legs:
                    leg = self.anchor_legs.get(idx)
                last_located = idx
            if leg:
                clock += leg["minutes"]
                travel_total += leg["minutes"]

            # Arrived too early for lunch / dinner / a night view? Wait (free time)
            wait = 0.0
            earliest = window_start(stop["type"], clock)
            if position == 0:
                max_wait = MAX_FIRST_STOP_WAIT_MIN
            elif all(self.stops[i]["type"] in LATE_TYPES for i in order[position:]):
                max_wait = MAX_WAIT_REST_OF_DAY_MIN
            else:
                max_wait = MAX_WAIT_MIN
            if earliest is not None and clock < earliest and earliest - clock <= max_wait:
                wait = earliest - clock
                clock = earliest
                wait_total += wait
                wait_cost += wait * (FIRST_STOP_WAIT_COST if position == 0 else WAIT_COST_PER_MIN)

            arrive = clock
            penalty += time_penalty(stop["type"], arrive)
            clock += stop["duration"]
            timeline.append({"index": idx, "arrive": arrive, "depart": clock, "leg": leg, "wait": wait})

        return {
            "timeline": timeline,
            "travel_minutes": travel_total,
            "wait_minutes": round(wait_total),
            "penalty": penalty,
            "end_minute": clock,
            "cost": travel_total + penalty + wait_cost,
        }

    def cost(self, order: List[int]) -> float:
        return self.simulate(order)["cost"]

    # ---- building a good starting order -------------------------------------

    def _travel(self, i: Optional[int], j: int) -> float:
        if i is None:
            return self.anchor_legs[j]["minutes"] if j in self.anchor_legs else 0.0
        return self.legs[(i, j)]["minutes"]

    def _held_karp(self, located: List[int]) -> List[int]:
        """Exact shortest open path through the located stops (bitmask DP)"""
        n = len(located)
        INF = float("inf")
        size = 1 << n
        dp = [[INF] * n for _ in range(size)]
        parent = [[-1] * n for _ in range(size)]

        # Start: travel from the anchor (or nothing) to the first stop
        for k in range(n):
            dp[1 << k][k] = self._travel(None, located[k])

        for mask in range(size):
            row = dp[mask]
            for last in range(n):
                cur = row[last]
                if cur == INF or not (mask >> last) & 1:
                    continue
                for nxt in range(n):
                    if (mask >> nxt) & 1:
                        continue
                    new_mask = mask | (1 << nxt)
                    cand = cur + self.legs[(located[last], located[nxt])]["minutes"]
                    if cand < dp[new_mask][nxt]:
                        dp[new_mask][nxt] = cand
                        parent[new_mask][nxt] = last

        full = size - 1
        last = min(range(n), key=lambda k: dp[full][k])
        path, mask = [], full
        while last != -1:
            path.append(located[last])
            prev = parent[mask][last]
            mask ^= 1 << last
            last = prev
        return list(reversed(path))

    def _nearest_neighbor_2opt(self, located: List[int]) -> List[int]:
        """Greedy path + 2-opt improvement for days with many stops"""
        remaining = set(located)
        current: Optional[int] = None
        path: List[int] = []
        while remaining:
            nxt = min(remaining, key=lambda j: self._travel(current, j))
            path.append(nxt)
            remaining.remove(nxt)
            current = nxt

        def length(p: List[int]) -> float:
            total = self._travel(None, p[0]) if p else 0.0
            return total + sum(self.legs[(p[k], p[k + 1])]["minutes"] for k in range(len(p) - 1))

        improved = True
        best = length(path)
        while improved:
            improved = False
            for i in range(len(path) - 1):
                for j in range(i + 1, len(path)):
                    cand = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                    cand_len = length(cand)
                    if cand_len + 1e-9 < best:
                        path, best, improved = cand, cand_len, True
        return path

    def _initial_order(self) -> Tuple[List[int], str]:
        located = [i for i, s in enumerate(self.stops) if self._located(s)]
        flexible = [i for i, s in enumerate(self.stops) if not self._located(s)]

        if len(located) <= 1:
            path, method = located, "single_location"
        elif len(located) <= self.HELD_KARP_MAX:
            path, method = self._held_karp(located), "held_karp"
        else:
            path, method = self._nearest_neighbor_2opt(located), "nearest_neighbor_2opt"

        # Flexible stops start at the end; the local search moves them to good times
        return path + flexible, method

    # ---- local search with time windows -------------------------------------

    def _local_search(self, order: List[int]) -> List[int]:
        """Try moving / swapping stops while it lowers travel + timing penalties"""
        best = list(order)
        best_cost = self.cost(best)
        n = len(best)
        improved = True
        rounds = 0
        while improved and rounds < 50:
            improved = False
            rounds += 1
            # Move one stop to another position
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cand = list(best)
                    item = cand.pop(i)
                    cand.insert(j, item)
                    c = self.cost(cand)
                    if c + 1e-9 < best_cost:
                        best, best_cost, improved = cand, c, True
            # Swap two stops
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = list(best)
                    cand[i], cand[j] = cand[j], cand[i]
                    c = self.cost(cand)
                    if c + 1e-9 < best_cost:
                        best, best_cost, improved = cand, c, True
        return best

    def evaluate(self) -> Dict[str, Any]:
        """Schedule and warnings for the day exactly as the user ordered it"""
        original = list(range(len(self.stops)))
        return self._result(original, original, "as_planned")

    def optimize(self) -> Dict[str, Any]:
        original = list(range(len(self.stops)))
        if len(self.stops) <= 1:
            return self._result(original, original, "single_location")

        initial, method = self._initial_order()
        optimized = self._local_search(initial)

        # Never return something worse than what the user already had
        if self.cost(original) <= self.cost(optimized) + 1e-9:
            optimized = self._local_search(original)
            if self.cost(original) <= self.cost(optimized) + 1e-9:
                optimized, method = original, "already_optimal"
        return self._result(original, optimized, method)

    # ---- output --------------------------------------------------------------

    def _result(self, original: List[int], optimized: List[int], method: str) -> Dict[str, Any]:
        before = self.simulate(original)
        after = self.simulate(optimized)
        stops_out = []
        local_fares = 0
        jr_fares = 0
        warnings = []

        for step, entry in enumerate(after["timeline"], start=1):
            stop = self.stops[entry["index"]]
            leg = entry["leg"]
            item = {
                "step": step,
                "name": stop["name"],
                "matched_name": stop.get("matched_name"),
                "type": stop["type"],
                "lat": stop.get("lat"),
                "lng": stop.get("lng"),
                "duration": stop["duration"],
                "duration_estimated": bool(stop.get("duration_estimated")),
                "arrive": fmt_clock(entry["arrive"]),
                "depart": fmt_clock(entry["depart"]),
                "located": self._located(stop),
                "kind": stop.get("kind"),
                "generic": bool(stop.get("generic")),
                "free_time_before": round(entry["wait"]),
            }
            if leg:
                item.update({
                    "travel_minutes": leg["minutes"],
                    "distance_km": leg["distance_km"],
                    "mode": leg["mode"],
                    "fare_yen": leg["fare_yen"],
                })
                local_fares += leg["fare_yen"]
                jr_fares += round(leg["fare_yen"] * leg["jr_share"])
                if leg["mode"] == "long-distance":
                    warnings.append(f"“{stop['name']}” is {leg['distance_km']:.0f} km from the previous stop — "
                                    f"is it on the right day?")

            if time_penalty(stop["type"], entry["arrive"]) > 30:
                label = {"evening": "best visited in the evening",
                         "daytime": "usually closed by early evening",
                         "shopping": "a shop (most are open about 10:30–20:00)"}.get(stop["type"], "a meal")
                warnings.append(f"“{stop['name']}” is {label} but is scheduled at {fmt_clock(entry['arrive'])}.")
            if not self._located(stop) and not stop.get("generic"):
                warnings.append(f"Couldn't find “{stop['name']}” on the map, so travel to it isn't counted. "
                                f"Try its full name, or add it as a custom place.")
            stops_out.append(item)

        if self.stops and not any(self._located(s) for s in self.stops):
            warnings.append("None of this day's stops could be found on the map, so the order wasn't changed.")
        if after["end_minute"] > DAY_END_WARNING:
            warnings.append(f"This day ends around {fmt_clock(after['end_minute'])} — consider moving something "
                            f"to another day.")

        return {
            "method": method,
            "original_order": [self.stops[i]["name"] for i in original],
            "optimized_order": [self.stops[i]["name"] for i in optimized],
            "changed": optimized != original,
            "stops": stops_out,
            "unmatched": [s["name"] for s in self.stops if not self._located(s) and not s.get("generic")],
            "start_time": fmt_clock(self.start_minute),
            "end_time": fmt_clock(after["end_minute"]),
            "end_minute": round(after["end_minute"]),
            "travel_minutes": after["travel_minutes"],
            "free_minutes": after["wait_minutes"],
            "original_travel_minutes": before["travel_minutes"],
            "travel_saved_minutes": max(0, before["travel_minutes"] - after["travel_minutes"]),
            "activity_minutes": sum(s["duration"] for s in self.stops),
            "timing_issues_before": round(before["penalty"]),
            "timing_issues_after": round(after["penalty"]),
            "local_fares_yen": local_fares,
            "jr_fares_yen": jr_fares,
            "warnings": warnings,
        }


# =============================================================================
# WHOLE TRIP
# =============================================================================

def best_jr_pass(day_jr_fares: List[int]) -> Dict[str, Any]:
    """
    Find the JR Pass (7/14/21 days) and start day that saves the most money.
    day_jr_fares[k] = JR fares spent on trip day k (including travel to the next city).
    """
    best = None
    n = len(day_jr_fares)
    for length, cost in JR_PASS_COSTS.items():
        span = min(length, n)
        for start in range(0, max(1, n - span + 1)):
            covered = sum(day_jr_fares[start:start + span])
            saving = covered - cost
            if best is None or saving > best["net_savings"]:
                best = {
                    "pass_type": f"{length}-day",
                    "pass_cost": cost,
                    "start_day": start + 1,
                    "end_day": start + span,
                    "jr_fares_covered": covered,
                    "net_savings": saving,
                }
    best["is_worth_it"] = best["net_savings"] > 0
    if best["is_worth_it"]:
        best["recommendation"] = (
            f"A {best['pass_type']} JR Pass from day {best['start_day']} saves about "
            f"¥{best['net_savings']:,}. Nozomi/Mizuho trains need an extra supplement."
        )
    else:
        best["recommendation"] = (
            f"Pay per ride — even the best JR Pass option ({best['pass_type']}) costs about "
            f"¥{-best['net_savings']:,} more than individual tickets."
        )
    return best


def plan_trip(days: List[Dict[str, Any]], start_minute: int = 9 * 60) -> Dict[str, Any]:
    """
    Optimize every day and add city-to-city travel + JR Pass analysis.

    days: [{"day": 1, "label": "Tokyo Day 1", "city": "Tokyo",
            "anchor": {name, lat, lng} | None,
            "stops": [{name, duration, lat, lng, type, matched_name}]}]
    """
    results = []
    day_jr_fares = []
    intercity = []

    for k, day in enumerate(days):
        planner = DayPlanner(day["stops"], day["city"], start_minute, day.get("anchor"))
        result = planner.optimize()
        result.update({"day": day.get("day"), "label": day.get("label"), "city": day["city"],
                       "key": day.get("key"),
                       "anchor": (day.get("anchor") or {}).get("matched_name")})
        results.append(result)
        day_jr_fares.append(result["jr_fares_yen"])

        # Travel to the next city happens at the end of this day
        if k + 1 < len(days) and days[k + 1]["city"] != day["city"]:
            leg = intercity_leg(day["city"], days[k + 1]["city"])
            leg.update({"from_city": day["city"], "to_city": days[k + 1]["city"],
                        "after_day_label": day.get("label")})
            intercity.append(leg)
            if leg["jr"]:
                day_jr_fares[-1] += leg["fare_yen"]

    local_total = sum(r["local_fares_yen"] for r in results)
    intercity_total = sum(leg["fare_yen"] for leg in intercity)
    total = local_total + intercity_total
    jr_pass = best_jr_pass(day_jr_fares) if days else None
    with_pass = total - jr_pass["jr_fares_covered"] + jr_pass["pass_cost"] if jr_pass else total

    return {
        "days": results,
        "intercity": intercity,
        "summary": {
            "days": len(results),
            "travel_minutes": sum(r["travel_minutes"] for r in results),
            "original_travel_minutes": sum(r["original_travel_minutes"] for r in results),
            "travel_saved_minutes": sum(r["travel_saved_minutes"] for r in results),
            "activity_minutes": sum(r["activity_minutes"] for r in results),
            "days_changed": sum(1 for r in results if r["changed"]),
            "unmatched": sum(len(r["unmatched"]) for r in results),
        },
        "cost_estimate": {
            "local_fares_yen": local_total,
            "intercity_fares_yen": intercity_total,
            "total_without_pass": total,
            "total_with_pass": with_pass,
            "jr_pass": jr_pass,
            "note": "Rough estimates. Local fares assume IC card (Suica/ICOCA) prices; flights vary a lot by date.",
        },
    }
