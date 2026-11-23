# =============================================================================
# TRIP CHECKS: weather and closed days
# =============================================================================
# Once a trip has a start date, every day has a real date. Then we can check:
#   - Weather: the forecast (next ~2 weeks) or, further ahead, what the weather is
#     usually like on those dates (the last few years from the Open-Meteo archive)
#   - Closed days: well-known places that close on a fixed weekday (most museums
#     close on Mondays, the Ghibli Museum on Tuesdays...)
# and suggest fixes: swap a rainy outdoor day with an indoor day, or move a stop
# to a day when it's open.
#
# Open-Meteo is free and needs no API key. Only a city's coordinates and the
# dates are sent. No Flask here: the weather fetcher is passed in.
# =============================================================================

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
import time
import unicodedata
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

import route_planner

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# -----------------------------------------------------------------------------
# Closed days
# -----------------------------------------------------------------------------
# Regular closing days of well-known places in our data. Names are matched as
# lowercase text without accents. Many museums open on a public holiday Monday and
# close the next day instead, so the warning always says to check.
#   closed: weekday numbers (0 = Monday); months: only in these months (optional)
CLOSED_DAYS = [
    {"match": ("ghibli museum",), "closed": (1,)},
    {"match": ("tokyo national museum",), "closed": (0,)},
    {"match": ("national museum of western art",), "closed": (0,)},
    {"match": ("ueno zoo",), "closed": (0,)},
    {"match": ("shinjuku gyoen",), "closed": (0,)},
    {"match": ("miraikan", "emerging science and innovation"), "closed": (1,)},
    {"match": ("nezu museum",), "closed": (0,)},
    {"match": ("teien art museum",), "closed": (0,)},
    {"match": ("museum of contemporary art tokyo",), "closed": (0,)},
    {"match": ("fujiko", "fujio museum"), "closed": (1,)},
    {"match": ("tokiwaso manga museum",), "closed": (0,)},
    {"match": ("nihon minka-en", "open-air folk museum"), "closed": (0,)},
    {"match": ("kyoto imperial palace",), "closed": (0,)},
    {"match": ("kyocera museum",), "closed": (0,)},
    {"match": ("nijo castle",), "closed": (1,), "months": (1, 7, 8, 12)},
    {"match": ("nintendo museum",), "closed": (1,)},
    {"match": ("hokkaido museum of modern art",), "closed": (0,)},
    {"match": ("okinawa prefectural museum",), "closed": (0,)},
]
# Open, but many shops are closed
PARTLY_CLOSED = [
    {"match": ("tsukiji outer market",), "closed": (2, 6), "text": "many shops are closed on Wednesdays and Sundays"},
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def closure_for(name: str, date: dt.date) -> Optional[str]:
    """Why this place may be closed on that date, or None"""
    text = _norm(name)
    for rule in CLOSED_DAYS:
        if any(m in text for m in rule["match"]) and date.weekday() in rule["closed"]:
            if rule.get("months") and date.month not in rule["months"]:
                continue
            return f"usually closed on {WEEKDAY_NAMES[date.weekday()]}s"
    for rule in PARTLY_CLOSED:
        if any(m in text for m in rule["match"]) and date.weekday() in rule["closed"]:
            return rule["text"]
    return None


def is_fully_closed(name: str, date: dt.date) -> bool:
    text = _norm(name)
    return any(any(m in text for m in r["match"]) and date.weekday() in r["closed"]
               and (not r.get("months") or date.month in r["months"]) for r in CLOSED_DAYS)


# -----------------------------------------------------------------------------
# Weather (Open-Meteo)
# -----------------------------------------------------------------------------

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_DAYS = 15          # Open-Meteo forecasts 16 days; keep a margin
TYPICAL_YEARS = 5           # "Usually" = these past years...
TYPICAL_WINDOW_DAYS = 3     # ...and this many days either side of each date
RAIN_MM = 1.0               # A day with at least this much rain counts as rainy
RAINY_PROBABILITY = 55      # Forecast / typical chance (%) from which we warn about rain
WEATHER_CACHE_SECONDS = 3 * 3600
_weather_cache: Dict[str, Tuple[float, Any]] = {}

# WMO weather codes -> (icon, words)
WEATHER_CODES = {
    0: ("☀️", "Sunny"), 1: ("🌤️", "Mostly sunny"), 2: ("⛅", "Partly cloudy"), 3: ("☁️", "Cloudy"),
    45: ("🌫️", "Fog"), 48: ("🌫️", "Fog"), 51: ("🌦️", "Drizzle"), 53: ("🌦️", "Drizzle"), 55: ("🌧️", "Drizzle"),
    61: ("🌧️", "Light rain"), 63: ("🌧️", "Rain"), 65: ("🌧️", "Heavy rain"), 66: ("🌧️", "Freezing rain"),
    67: ("🌧️", "Freezing rain"), 71: ("🌨️", "Light snow"), 73: ("🌨️", "Snow"), 75: ("❄️", "Heavy snow"),
    77: ("🌨️", "Snow"), 80: ("🌦️", "Showers"), 81: ("🌧️", "Showers"), 82: ("⛈️", "Heavy showers"),
    85: ("🌨️", "Snow showers"), 86: ("❄️", "Snow showers"), 95: ("⛈️", "Thunderstorm"),
    96: ("⛈️", "Thunderstorm"), 99: ("⛈️", "Thunderstorm"),
}


def http_get_json(url: str, params: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    cached = _weather_cache.get(full)
    if cached and time.time() - cached[0] < WEATHER_CACHE_SECONDS:
        return cached[1]
    req = urllib.request.Request(full, headers={"User-Agent": "trip-planner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    _weather_cache[full] = (time.time(), data)
    return data


def _forecast(lat: float, lng: float, dates: List[dt.date], get_json) -> Dict[dt.date, Dict[str, Any]]:
    data = get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lng, "timezone": "Asia/Tokyo",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
        "start_date": min(dates).isoformat(), "end_date": max(dates).isoformat(),
    })
    daily = data.get("daily") or {}
    out = {}
    for i, day in enumerate(daily.get("time") or []):
        date = dt.date.fromisoformat(day)
        code = (daily.get("weather_code") or [None])[i]
        prob = (daily.get("precipitation_probability_max") or [None])[i]
        rain = (daily.get("precipitation_sum") or [None])[i]
        icon, words = WEATHER_CODES.get(code, ("🌡️", "—"))
        out[date] = {
            "source": "forecast", "icon": icon, "summary": words,
            "temp_max": _round((daily.get("temperature_2m_max") or [None])[i]),
            "temp_min": _round((daily.get("temperature_2m_min") or [None])[i]),
            "rain_chance": prob if prob is not None else (80 if (rain or 0) >= 5 else 20 if (rain or 0) >= RAIN_MM else 5),
            "rain_mm": _round(rain, 1),
        }
    return out


def _typical(lat: float, lng: float, dates: List[dt.date], get_json) -> Dict[dt.date, Dict[str, Any]]:
    """Average of the same dates (± a few days) over the last few years"""
    samples: Dict[dt.date, List[Tuple[float, float, float]]] = {d: [] for d in dates}
    this_year = dt.date.today().year
    first = min(dates)

    def shift(d: dt.date, back: int) -> dt.date:
        """The same date 'back' years before this year (trips can cross New Year)"""
        years = first.year - (this_year - back)
        try:
            return d.replace(year=d.year - years)
        except ValueError:  # 29 February
            return d.replace(year=d.year - years, day=28)

    # One request covering all the past years we need
    window = dt.timedelta(days=TYPICAL_WINDOW_DAYS)
    data = get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lng, "timezone": "Asia/Tokyo",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "start_date": (shift(first, TYPICAL_YEARS) - window).isoformat(),
        "end_date": (shift(max(dates), 1) + window).isoformat(),
    })
    daily = data.get("daily") or {}
    by_date = {}
    for i, day in enumerate(daily.get("time") or []):
        tmax = (daily.get("temperature_2m_max") or [None])[i]
        tmin = (daily.get("temperature_2m_min") or [None])[i]
        rain = (daily.get("precipitation_sum") or [None])[i]
        if None not in (tmax, tmin, rain):
            by_date[dt.date.fromisoformat(day)] = (tmax, tmin, rain)
    for back in range(1, TYPICAL_YEARS + 1):
        for d in dates:
            center = shift(d, back)
            for k in range(-TYPICAL_WINDOW_DAYS, TYPICAL_WINDOW_DAYS + 1):
                v = by_date.get(center + dt.timedelta(days=k))
                if v:
                    samples[d].append(v)

    out = {}
    for d, values in samples.items():
        if not values:
            continue
        chance = round(100 * sum(1 for v in values if v[2] >= RAIN_MM) / len(values))
        tmax = sum(v[0] for v in values) / len(values)
        icon, words = (("🌧️", "Often rainy") if chance >= RAINY_PROBABILITY else
                       ("🌦️", "Sometimes rainy") if chance >= 35 else ("🌤️", "Usually dry"))
        out[d] = {
            "source": "typical", "icon": icon, "summary": words,
            "temp_max": _round(tmax), "temp_min": _round(sum(v[1] for v in values) / len(values)),
            "rain_chance": chance, "rain_mm": None,
        }
    return out


def _round(value, digits=0):
    if value is None:
        return None
    return round(value) if digits == 0 else round(value, digits)


def weather_for_city(city: str, dates: List[dt.date],
                     get_json: Callable = http_get_json) -> Dict[dt.date, Dict[str, Any]]:
    """Forecast for dates in the next ~2 weeks, typical weather for later ones"""
    hub = route_planner.CITY_HUBS.get(city)
    if not hub or not dates:
        return {}
    today = dt.date.today()
    soon = [d for d in dates if today <= d <= today + dt.timedelta(days=FORECAST_DAYS)]
    later = [d for d in dates if d not in soon]
    out: Dict[dt.date, Dict[str, Any]] = {}
    if soon:
        out.update(_forecast(hub[0], hub[1], soon, get_json))
    if later:
        out.update(_typical(hub[0], hub[1], later, get_json))
    return out


# -----------------------------------------------------------------------------
# Indoor / outdoor
# -----------------------------------------------------------------------------

OUTDOOR_WORDS = ("park", "garden", "gyoen", "zoo", "forest", "falls", "beach", "cave", "lake", "mount", "mt.",
                 "owakudani", "bamboo", "abyss", "hill", "cruise", "market", "crossing", "shrine", "jingu",
                 "taisha", "torii", "castle", "street", "yokocho", "ropeway", "cape", "island", "village",
                 "disney", "universal studios", "fuji", "walk", "hike", "onsen town", "-dera", "-ji", "temple")
INDOOR_WORDS = ("museum", "aquarium", "teamlab", "gallery", "department store", "mall", "isetan", "takashimaya",
                "mitsukoshi", "daimaru", "parco", "broadway", "arcade", "tower", "sky", "observatory",
                "observation", "studio tour", "warner", "art center", "design sight")


def stop_exposure(stop: Dict[str, Any]) -> Optional[str]:
    """'outdoor', 'indoor' or None (meals, unknown places)"""
    if stop.get("type") in ("breakfast", "lunch", "dinner", "meal", "snack"):
        return None
    text = " " + _norm(stop.get("matched_name") or stop.get("name")) + " "
    if any(w in text for w in INDOOR_WORDS):
        return "indoor"
    if stop.get("type") == "shopping":
        return "indoor"
    if stop.get("kind") == "daytrips" or any(w in text for w in OUTDOOR_WORDS):
        return "outdoor"
    return None


def outdoor_share(stops: List[Dict[str, Any]]) -> float:
    """Share of the day's sightseeing minutes spent outdoors"""
    total = outdoor = 0
    for s in stops:
        exposure = stop_exposure(s)
        if exposure is None:
            continue
        total += s.get("duration") or 60
        if exposure == "outdoor":
            outdoor += s.get("duration") or 60
    return outdoor / total if total else 0.0


# -----------------------------------------------------------------------------
# The check
# -----------------------------------------------------------------------------

RAINY_DAY_OUTDOOR_SHARE = 0.5    # Warn when at least half the day is outdoors on a rainy day
SWAP_MIN_GAIN = 0.25             # Only suggest a swap that moves a good chunk of the outdoors to a dry day


def parse_date(value: Any) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def check_trip(days: List[Dict[str, Any]], start: dt.date,
               weather_fn: Callable[[str, List[dt.date]], Dict[dt.date, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    days: prepare_trip_days() output (stops matched, typed, with durations), in trip order.
    Returns per-day date / weather / closures, and suggestions:
      {"type": "swap_days", "day": i, "with_day": j, ...} (indexes into days, 0-based)
      {"type": "move_stop", "day": i, "to_day": j, "stop": name, ...}
    """
    dates = [start + dt.timedelta(days=i) for i in range(len(days))]

    # Weather per city (one lookup per city)
    weather: Dict[int, Dict[str, Any]] = {}
    by_city: Dict[str, List[int]] = {}
    for i, day in enumerate(days):
        by_city.setdefault(day["city"], []).append(i)
    weather_errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {city: pool.submit(weather_fn, city, [dates[i] for i in idxs]) for city, idxs in by_city.items()}
    for city, idxs in by_city.items():
        try:
            result = futures[city].result()
        except Exception:  # The weather service being down shouldn't break the check
            weather_errors.append(city)
            result = {}
        for i in idxs:
            if dates[i] in result:
                weather[i] = result[dates[i]]

    out_days, suggestions = [], []
    shares = [outdoor_share(day["stops"]) for day in days]
    rainy = [bool(weather.get(i)) and weather[i]["rain_chance"] >= RAINY_PROBABILITY for i in range(len(days))]

    for i, day in enumerate(days):
        closures = []
        for stop in day["stops"]:
            why = closure_for(stop.get("matched_name") or stop["name"], dates[i])
            if why:
                closures.append({"stop": stop["name"], "why": why, "full": is_fully_closed(
                    stop.get("matched_name") or stop["name"], dates[i])})
        out_days.append({
            "index": i,
            "label": day["label"],
            "date": dates[i].isoformat(),
            "weekday": WEEKDAYS[dates[i].weekday()],
            "weather": weather.get(i),
            "outdoor_share": round(shares[i], 2),
            "rain_risk": rainy[i] and shares[i] >= RAINY_DAY_OUTDOOR_SHARE,
            "closures": closures,
        })

        # Closed: move it to another day in the same city when it's open there
        for c in closures:
            if not c["full"]:
                continue
            stop_name = c["stop"]
            place_name = _matched(day, stop_name) or stop_name
            options = [j for j in by_city[day["city"]] if j != i and not closure_for(place_name, dates[j])]
            if options:
                j = min(options, key=lambda j: abs(j - i))
                suggestions.append({
                    "type": "move_stop", "day": i, "to_day": j, "stop": stop_name,
                    "title": f"Move “{stop_name}” to {days[j]['label']}",
                    "reason": f"It's {c['why']} and {day['label']} is a {WEEKDAY_NAMES[dates[i].weekday()]}.",
                })

    # Rain: swap a rainy outdoor day with a drier, more indoor day in the same city
    used = set()
    for i, day in enumerate(days):
        if not out_days[i]["rain_risk"] or i in used:
            continue
        best, best_gain = None, 0.0
        for j in by_city[day["city"]]:
            if j == i or j in used or rainy[j]:
                continue
            gain = shares[i] - shares[j]
            if gain >= SWAP_MIN_GAIN and gain > best_gain:
                best, best_gain = j, gain
        w = weather[i]
        when = "Rain is forecast" if w["source"] == "forecast" else "It's often rainy around this date"
        if best is not None:
            used.update((i, best))
            suggestions.append({
                "type": "swap_days", "day": i, "with_day": best,
                "title": f"Swap {day['label']} and {days[best]['label']}",
                "reason": f"{when} ({w['rain_chance']}% chance) and {day['label']} is mostly outdoors; "
                          f"{days[best]['label']} is drier or more indoors.",
            })
        else:
            suggestions.append({
                "type": "tip", "day": i,
                "title": f"Pack an umbrella for {day['label']}",
                "reason": f"{when} ({w['rain_chance']}% chance) and most of the day is outdoors. "
                          f"Museums, shopping or teamLab make good rainy-day backups.",
            })

    return {
        "start_date": start.isoformat(),
        "days": out_days,
        "suggestions": suggestions,
        "weather_unavailable": weather_errors,
        "note": "Closed days are the usual ones; holidays can change them, so check before you go.",
    }


def _matched(day: Dict[str, Any], stop_name: str) -> Optional[str]:
    for s in day["stops"]:
        if s["name"] == stop_name:
            return s.get("matched_name")
    return None
