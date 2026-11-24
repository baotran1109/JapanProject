# =============================================================================
# IMPORT PLACES FROM GOOGLE MAPS
# =============================================================================
# Two inputs, both turned into a preview list before anything is saved:
#   1. Google Maps links pasted by the traveler (one per line). Full links carry
#      the place's location; short share links (maps.app.goo.gl/...) are opened
#      to find the full link. Only Google Maps addresses are ever opened.
#   2. A Google My Maps export (.kml or .kmz): every pin with its location; the
#      map's layers ("Food", "Shops"...) help pick each place's category.
# Each place gets the nearest supported city (or is skipped when it's outside
# them) and a category guess the traveler can change.
#
# No Flask here; the app passes in the fetcher and the name lookup.
# =============================================================================

import io
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Tuple

import route_planner

MAX_LINKS = 50
MAX_FILE_BYTES = 5 * 1024 * 1024        # Uploaded file
MAX_KML_BYTES = 20 * 1024 * 1024        # KML inside a .kmz, after unzipping
MAX_PLACES = 300
MAX_REDIRECTS = 5

# A place belongs to the closest city within this distance of its main station
CITY_RADIUS_KM = {
    "Tokyo": 90, "Kyoto": 35, "Osaka": 45, "Fukuoka": 50, "Okinawa": 130,
    "Aomori": 70, "Hakodate": 60, "Sapporo": 80,
}

GOOGLE_HOSTS = ("google.com", "google.co.jp", "goo.gl", "app.goo.gl", "maps.app.goo.gl", "g.co")


# -----------------------------------------------------------------------------
# City and category
# -----------------------------------------------------------------------------

def nearest_city(lat: float, lng: float) -> Tuple[Optional[str], Optional[float]]:
    """('Tokyo', 3.2) for the closest supported city, or (None, km) when too far from all of them"""
    best, best_d = None, None
    for city, hub in route_planner.CITY_HUBS.items():
        d = route_planner.haversine_km(lat, lng, hub[0], hub[1])
        if d is not None and (best_d is None or d < best_d):
            best, best_d = city, d
    if best is None or best_d > CITY_RADIUS_KM.get(best, 50):
        return None, best_d
    return best, best_d


CATEGORY_WORDS = [
    ("stay", ("hotel", "hostel", "ryokan", "guesthouse", "guest house", "inn", "airbnb", "apartment", "capsule",
              "stay", "accommodation", "lodging", "minshuku", "ホテル", "旅館")),
    ("cafe", ("cafe", "café", "coffee", "kissa", "bakery", "dessert", "sweets", "matcha", "tea", "patisserie",
              "parfait", "pancake", "donut", "gelato", "カフェ", "珈琲", "喫茶")),
    ("food", ("ramen", "sushi", "restaurant", "izakaya", "yakitori", "yakiniku", "udon", "soba", "tempura",
              "tonkatsu", "okonomiyaki", "takoyaki", "gyoza", "curry", "eat", "food", "dinner", "lunch", "bistro",
              "grill", "diner", "kitchen", "bar ", "ラーメン", "寿司", "食堂", "居酒屋")),
    ("shop", ("shop", "store", "vintage", "thrift", "boutique", "mall", "outlet", "department", "market",
              "bookoff", "kindal", "ragtag", "2nd street", "shopping", "古着", "ショップ", "ストア")),
    ("sight", ("shrine", "temple", "jinja", "jingu", "taisha", "-ji", "-dera", "museum", "park", "garden",
               "tower", "castle", "view", "observatory", "palace", "beach", "falls", "onsen", "sight", "see",
               "attraction", "神社", "寺", "公園")),
]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return " " + "".join(c for c in text if not unicodedata.combining(c)).lower() + " "


def guess_category(name: str, layer: str = "") -> str:
    """Category from the My Maps layer name first ("Food"), then from the place name"""
    for text in (layer, name):
        t = _norm(text)
        if not t.strip():
            continue
        for category, words in CATEGORY_WORDS:
            if any(w in t for w in words):
                return category
    return "other"


# -----------------------------------------------------------------------------
# 1. Google Maps links
# -----------------------------------------------------------------------------

def is_google_maps_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if parts.scheme not in ("http", "https"):
        return False
    if not any(host == h or host.endswith("." + h) for h in GOOGLE_HOSTS):
        return False
    # Short links are only ever Maps links; on google.com only /maps paths count
    return host.endswith("goo.gl") or host == "g.co" or parts.path.startswith("/maps") or host.startswith("maps.")


def parse_maps_url(url: str) -> Dict[str, Any]:
    """Name and location from a full Google Maps link (either may be missing)"""
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote_plus(parts.path)
    query = urllib.parse.parse_qs(parts.query)
    name = None
    m = re.search(r"/maps/(?:place|search)/([^/@]+)", path)
    if m:
        name = m.group(1).strip()
    lat = lng = None
    # The place's own pin (!3d lat !4d lng) is more exact than the map's center (@lat,lng)
    pins = re.findall(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)", url)
    if pins:
        lat, lng = map(float, pins[-1])
    else:
        m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
        if m:
            lat, lng = float(m.group(1)), float(m.group(2))
    for key in ("q", "query", "ll", "center", "destination"):
        value = (query.get(key) or [""])[0].strip()
        if not value:
            continue
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", value)
        if m and lat is None:
            lat, lng = float(m.group(1)), float(m.group(2))
        elif not m and not name:
            name = value
    if name and re.fullmatch(r"-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?", name):
        name = None  # "35.68,139.76" is a location, not a name
    return {"name": name, "lat": lat, "lng": lng}


def expand_short_link(url: str, opener: Optional[Callable[[str], Optional[str]]] = None) -> Optional[str]:
    """
    Follow a short link (maps.app.goo.gl/...) to the full Google Maps link.
    Each hop must stay on a Google Maps address.
    """
    opener = opener or _redirect_target
    current = url
    for _ in range(MAX_REDIRECTS):
        if not is_google_maps_url(current):
            return None
        host = (urllib.parse.urlsplit(current).hostname or "").lower()
        if not (host.endswith("goo.gl") or host == "g.co"):
            return current   # Already a full link
        nxt = opener(current)
        if not nxt:
            return None
        current = urllib.parse.urljoin(current, nxt)
    return current if is_google_maps_url(current) else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _redirect_target(url: str) -> Optional[str]:
    """Where a short link points, without opening that next page"""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0 (trip planner import)"})
    try:
        opener.open(req, timeout=6)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            return e.headers.get("Location")
        return None
    except Exception:
        return None
    return None


def places_from_links(text: str, lookup: Callable[[str], Optional[Dict[str, Any]]],
                      opener: Optional[Callable[[str], Optional[str]]] = None) -> Dict[str, Any]:
    """
    Every Google Maps link in the text -> places.
    lookup(name) finds a location by name when a link has none ({name, lat, lng} or None).
    """
    urls = re.findall(r"https?://[^\s<>\"']+", text or "")
    seen, places, problems = set(), [], []
    for url in urls:
        url = url.rstrip(").,;")
        if url in seen:
            continue
        seen.add(url)
        if len(seen) > MAX_LINKS:
            problems.append(f"Only the first {MAX_LINKS} links are read at a time.")
            break
        if not is_google_maps_url(url):
            problems.append(f"Not a Google Maps link: {url[:80]}")
            continue
        full = expand_short_link(url, opener)
        if not full:
            problems.append(f"Couldn't open this link: {url[:80]}")
            continue
        info = parse_maps_url(full)
        source = "link"
        if info["lat"] is None and info["name"]:
            found = lookup(info["name"])
            if found:
                info.update(lat=found["lat"], lng=found["lng"])
                source = "looked up by name"
        if info["lat"] is None:
            what = f" for “{info['name']}”" if info["name"] else ""
            problems.append(f"Couldn't find a location{what} in this link: {url[:80]}")
            continue
        places.append({"name": info["name"] or "Dropped pin", "lat": info["lat"], "lng": info["lng"],
                       "layer": "", "notes": "", "source": source, "link": url})
    if not urls:
        problems.append("No links found. Paste Google Maps links, one per line.")
    return {"places": places, "problems": problems}


# -----------------------------------------------------------------------------
# 2. Google My Maps file (.kml / .kmz)
# -----------------------------------------------------------------------------

def _kml_bytes(data: bytes, filename: str) -> bytes:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("That file is too big (max 5 MB).")
    if data[:2] == b"PK" or filename.lower().endswith(".kmz"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not names:
                    raise ValueError("This .kmz file has no map inside.")
                name = "doc.kml" if "doc.kml" in names else names[0]
                if z.getinfo(name).file_size > MAX_KML_BYTES:
                    raise ValueError("That map is too big.")
                return z.read(name)
        except zipfile.BadZipFile:
            raise ValueError("That .kmz file is damaged.")
    return data


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_text(text: Optional[str], limit: int) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")        # descriptions can contain HTML
    return " ".join(text.split())[:limit]


def places_from_kml(data: bytes, filename: str = "") -> Dict[str, Any]:
    """Pins of a My Maps export -> places (lines and shapes are skipped)"""
    raw = _kml_bytes(data, filename)
    # Map files never need entity declarations; refusing them blocks XML tricks
    head = raw[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in raw.lower():
        raise ValueError("That file isn't a normal map export.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise ValueError("Couldn't read that file. Export it from Google My Maps as KML or KMZ.")

    places, problems = [], []
    skipped_shapes = 0

    def walk(node, layer):
        nonlocal skipped_shapes
        for child in node:
            tag = _local(child.tag)
            if tag in ("Folder", "Document"):
                name_el = next((c for c in child if _local(c.tag) == "name"), None)
                walk(child, (name_el.text or "").strip() if tag == "Folder" and name_el is not None else layer)
            elif tag == "Placemark":
                name = desc = coords = None
                for el in child.iter():
                    t = _local(el.tag)
                    if t == "name" and name is None:
                        name = el.text
                    elif t == "description" and desc is None:
                        desc = el.text
                    elif t == "Point":
                        c = next((x for x in el.iter() if _local(x.tag) == "coordinates"), None)
                        coords = c.text if c is not None else None
                if not coords:
                    skipped_shapes += 1
                    continue
                try:
                    lng, lat = [float(v) for v in coords.strip().split(",")[:2]]
                except ValueError:
                    continue
                places.append({"name": _clean_text(name, 120) or "Unnamed pin", "lat": lat, "lng": lng,
                               "layer": layer or "", "notes": _clean_text(desc, 300), "source": "map file"})
                if len(places) >= MAX_PLACES:
                    problems.append(f"Only the first {MAX_PLACES} pins are read.")
                    return

    walk(root, "")
    if skipped_shapes:
        problems.append(f"{skipped_shapes} line{'s' if skipped_shapes > 1 else ''} / shape{'s' if skipped_shapes > 1 else ''} "
                        f"skipped (only pins are places).")
    if not places:
        problems.append("No pins found in that map.")
    return {"places": places, "problems": problems}


# -----------------------------------------------------------------------------
# Preview
# -----------------------------------------------------------------------------

def preview(found: Dict[str, Any], existing: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add city, category and status to each found place.
    existing: the traveler's saved places (to spot duplicates).
    """
    have = {(str(p.get("name") or "").strip().lower(), str(p.get("city") or "").lower()) for p in existing}
    rows, seen = [], set()
    for p in found["places"]:
        city, km = nearest_city(p["lat"], p["lng"])
        row = {**p, "city": city, "category": guess_category(p["name"], p.get("layer", "")), "status": "new"}
        key = (p["name"].strip().lower(), (city or "").lower())
        if not city:
            row["status"] = "outside"
            row["reason"] = f"Not near one of our cities ({round(km)} km from the closest)" if km else "Not in Japan"
        elif key in have:
            row["status"] = "duplicate"
            row["reason"] = "Already in your places"
        elif key in seen:
            row["status"] = "duplicate"
            row["reason"] = "Listed twice"
        seen.add(key)
        rows.append(row)
    return {"places": rows, "problems": found.get("problems", []),
            "counts": {s: sum(1 for r in rows if r["status"] == s) for s in ("new", "duplicate", "outside")}}
