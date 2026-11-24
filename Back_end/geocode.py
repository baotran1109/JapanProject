# =============================================================================
# FINDING A HOTEL / ADDRESS ON THE MAP (OpenStreetMap Nominatim)
# =============================================================================
# Used for "where are you staying" when the place isn't one of the user's own
# places or in our database (most hotels aren't). Nominatim is free, but its
# usage policy asks for at most 1 request per second, an identifying
# User-Agent and caching, so every answer (also "not found") is cached and
# requests are spaced out.
# =============================================================================

import json
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

import route_planner

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "explore-japan-trip-planner/1.0"
MAX_DISTANCE_KM = 60          # Results further than this from the city are ignored
SEARCH_BOX_DEGREES = 0.6      # Search area around the city (about ±60 km)
CACHE_SECONDS = 7 * 24 * 3600
MIN_INTERVAL_SECONDS = 1.1
TIMEOUT_SECONDS = 6

_cache: Dict[Tuple[str, str], Tuple[float, Optional[Dict[str, Any]]]] = {}
_lock = threading.Lock()
_last_request = 0.0


def _get(params: Dict[str, Any]) -> Any:
    global _last_request
    with _lock:  # One request at a time, at most one per second
        wait = MIN_INTERVAL_SECONDS - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        req = urllib.request.Request(f"{SEARCH_URL}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))


def find(query: str, city: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Look up a hotel / address near a city (city=None: anywhere in Japan).
    Returns {name, lat, lng} or None. Never raises: if the service is down,
    the place just isn't found.
    """
    query = " ".join((query or "").split())[:120]
    hub = route_planner.CITY_HUBS.get(city) if city else None
    if not query or (city and not hub):
        return None
    key = (query.lower(), city or "japan")
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]

    params = {"q": query, "format": "jsonv2", "limit": 3, "countrycodes": "jp"}
    if hub:
        lat, lng = hub
        params.update({
            "viewbox": f"{lng - SEARCH_BOX_DEGREES},{lat + SEARCH_BOX_DEGREES},{lng + SEARCH_BOX_DEGREES},{lat - SEARCH_BOX_DEGREES}",
            "bounded": 1,
        })
    result = None
    try:
        rows = _get(params)
        for row in rows or []:
            plat, plng = float(row["lat"]), float(row["lon"])
            if not hub or (route_planner.haversine_km(hub[0], hub[1], plat, plng) or 0) <= MAX_DISTANCE_KM:
                label = row.get("name") or (row.get("display_name") or query).split(",")[0]
                result = {"name": label, "lat": plat, "lng": plng}
                break
    except Exception:
        return None  # Not cached: try again next time
    _cache[key] = (time.time(), result)
    return result
