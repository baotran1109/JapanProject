# ==================================================
# Bao Tran
# app.py
# JAPAN TRAVEL APP - Main Server
# The main server file
# Serves HTML pages (tokyo.html, kyoto.html, etc.)
# Connects to MongoDB to get travel data
# Uses AI to review trip plans and give suggestions
# ===================================================

# Imports the neccessary libraries
import os
import json
import re
import time
import logging
import unicodedata
from typing import Dict, List, Any, Tuple, Optional

# Import Flask and related frameworks
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import safe_join
from rapidfuzz import process, fuzz
from openai import OpenAI
from dotenv import load_dotenv

# Route planning (per-day optimization, schedules, fares, JR Pass) and the AI review
import route_planner
import ai_reviewer
import place_picks
import trip_generator
import trip_checks
import geocode
import chat_editor
import secrets

# Trying to import MongoDB - falls back to JSON files if not available
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("pymongo not installed - MongoDB features disabled")

# Firebase Admin - verifies the ID tokens the frontend sends for logged-in users
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials as firebase_credentials
from google.auth.credentials import AnonymousCredentials


# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================
# Setting up all the folder paths

# Where the file lives
APP_DIR = os.path.abspath(os.path.dirname(__file__))

 # Main project folder
REPO_ROOT = os.path.dirname(APP_DIR)

# HTML pages
FRONTEND_ROOT = os.path.join(APP_DIR, "../Front_end")

# Images and assets
STATIC_ROOT = os.path.join(APP_DIR, "../static")

# JSON backup files
DATA_ROOT = os.path.join(APP_DIR, "../data")

# Mapping cities to their JSON files - helps the AI find places when users type names
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
        "Aomori/Hachinohe_Day_Trip (Restaurants).json",
        "Aomori/Hachinohe_Day_Trip_(Attractions).json",
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
        "Osaka/Kobe_Shopping.json",
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
    "fukuoka": [
        "Fukuoka/Fukuoka_Attractions.json",
        "Fukuoka/Fukuoka_Restaurants.json",
        "Fukuoka/Fukuoka_Shopping.json",
    ],
    "okinawa": [
        "Okinawa/Okinawa_Attractions.json",
        "Okinawa/Okinawa_Restaurants.json",
    ],
}

# Load environment variables from .env file
load_dotenv()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize OpenAI client for AI reviews. Few retries and a timeout so that,
# if OpenAI is down or out of credits, the rule-based review is shown quickly.
client = OpenAI(api_key=OPENAI_API_KEY, max_retries=1, timeout=30)

# Initialize Flask app
app = Flask(__name__, static_folder=None)
# Enable CORS so frontend can call API from any IP
CORS(app)

# Connect to MongoDB Atlas (where stores all travel data)
mongo_client = None
mongo_db = None
if MONGODB_AVAILABLE:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client["Japan"]  # Database name: "Japan"
        logging.info("Connected to MongoDB Atlas")
    except Exception as e:
        logging.warning(f"MongoDB connection failed: {e}")
        MONGODB_AVAILABLE = False

# Initialize Firebase Admin. Verifying ID tokens only needs the project ID and
# Google's public keys, so no service account is required (works locally too)
class _NoCredential(firebase_credentials.Base):
    def get_credential(self):
        return AnonymousCredentials()

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "japan-travel-project-f2be0")
firebase_admin.initialize_app(_NoCredential(), {"projectId": FIREBASE_PROJECT_ID})

# Debug-only endpoints (like /ai/debug_allowed) are off unless this is set
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "").lower() in ("1", "true", "yes")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# =============================================================================
# HELPER FUNCTIONS - Data Loading & Processing
# =============================================================================
# Functions to load and process travel data

def load_json(path: str) -> Any:
    """Loads JSON files"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Each city's places are cached for a few minutes so we don't reload
# hundreds of documents from MongoDB on every request
KB_CACHE_SECONDS = 600
_kb_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


def load_kb_for(cities: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Same as _load_kb_uncached, but reuses recently loaded cities"""
    now = time.time()
    kb: Dict[str, List[Dict[str, Any]]] = {}
    missing = []
    for city in cities:
        cached = _kb_cache.get(city)
        if cached and now - cached[0] < KB_CACHE_SECONDS:
            kb[city] = cached[1]
        else:
            missing.append(city)
    if missing:
        fresh = _load_kb_uncached(missing)
        for city in missing:
            items = fresh.get(city, [])
            if items:  # Don't cache an empty result (e.g. database hiccup)
                _kb_cache[city] = (now, items)
            kb[city] = items
    return kb


def _load_kb_uncached(cities: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads travel data for specific cities (Knowledge Base for AI).
    
    - Tries MongoDB first (faster)
    - Falls back to JSON files if MongoDB isn't available
    - Returns: {"tokyo": [{place1}, {place2}], "kyoto": [...]}
    
    Helps the AI match what users type to actual places.
    """
    kb = {c: [] for c in cities}
    
    # Try MongoDB first (faster and always in sync)
    if MONGODB_AVAILABLE and mongo_db is not None:
        try:
            for city in cities:
                # Capitalize city name to match MongoDB format
                city_name = city.capitalize()
                
                # Fetch all data types for this city from MongoDB
                # (_kind remembers which collection each place came from)
                all_items = []
                for kind in ("attractions", "restaurants", "shopping", "daytrips"):
                    for item in mongo_db[kind].find({"city": city_name}, {"_id": 0}):
                        item["_kind"] = kind
                        all_items.append(item)

                # Normalize items
                for item in all_items:
                    if not isinstance(item, dict):
                        continue
                    item["__normname"] = (item.get("name") or "").strip().lower()
                    # lat/lng should already be normalized from MongoDB
                    kb[city].append(item)
                
                logger.info(f"Loaded {len(all_items)} items for {city} from MongoDB")
            return kb
        except Exception as e:
            logger.warning(f"MongoDB KB load failed, falling back to JSON: {e}")
    
    # Fallback to JSON files (original method)
    for c in cities:
        rels = CITY_DIRS.get(c, [])
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

            rel_lower = rel.lower()
            if "restaurant" in rel_lower:
                kind = "restaurants"
            elif "shopping" in rel_lower:
                kind = "shopping"
            elif "day" in rel_lower and "trip" in rel_lower:
                kind = "daytrips"
            else:
                kind = "attractions"

            # Same area rule as import_to_mongodb.py: "Shibuya_restaurants.json" -> "Shibuya"
            base = os.path.basename(rel).replace(".json", "")
            file_area = base.split("_")[0] if "_" in base else "General"

            for item in items:
                if not isinstance(item, dict):
                    continue
                item["_kind"] = kind
                item.setdefault("area", file_area)
                item["__normname"] = (item.get("name") or "").strip().lower()
                for k in list(item.keys()):
                    lk = k.lower()
                    if lk in ("lat", "latitude"):
                        item["lat"] = item[k]
                    if lk in ("lng", "lon", "longitude"):
                        item["lng"] = item[k]
                kb[c].append(item)
    return kb


# =============================================================================
# FLASK ROUTES - Frontend (HTML Pages & Static Files)
# =============================================================================
# These routes serve your website pages and images

@app.route("/")
def serve_index():
    """Serve the home page (index.html)"""
    return send_from_directory(FRONTEND_ROOT, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    """Serve HTML pages (tokyo.html, kyoto.html, plan_tokyo.html, etc.)"""
    full = safe_join(FRONTEND_ROOT, filename)
    if full and os.path.isfile(full):
        return send_from_directory(FRONTEND_ROOT, filename)
    # fallback to 404.html if exists
    if os.path.isfile(os.path.join(FRONTEND_ROOT, "404.html")):
        return send_from_directory(FRONTEND_ROOT, "404.html"), 404
    abort(404)

@app.route("/static/<path:path>")
def serve_static(path):
    """Serve static files (images, CSS, JavaScript)"""
    full = safe_join(STATIC_ROOT, path)
    if not (full and os.path.isfile(full)):
        abort(404)
    return send_from_directory(STATIC_ROOT, path)



@app.route("/data/<city>/<filename>")
def load_city_data(city, filename):
    """
    Serve JSON files (backward compatibility for old pages).
    Example: /data/Tokyo/Tokyo_Attractions.json
    """
    try:
        data_path = safe_join(DATA_ROOT, city, filename)
        if not data_path:
            return jsonify({"error": "File not found"}), 404
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()
        return app.response_class(response=data, status=200, mimetype="application/json")
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.exception("Failed to read file")
        return jsonify({"error": f"Failed to read file: {e}"}), 500


def int_arg(name: str, default: int, min_value: int = 1, max_value: int = 1000) -> int:
    """Reads an integer query param, falling back to the default if it's missing or invalid"""
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(value, max_value))


def get_verified_user_id() -> Optional[str]:
    """
    Returns the Firebase UID of the logged-in user, or None.
    The frontend sends "Authorization: Bearer <Firebase ID token>",
    and the token is verified so nobody can pretend to be another user.
    """
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[len("Bearer "):].strip()
    if not token:
        return None
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception as e:
        logger.info(f"Rejected Firebase ID token: {e}")
        return None


@app.route("/healthz")
def healthz():
    """Health check endpoint"""
    if MONGODB_AVAILABLE and mongo_db is not None:
        try:
            mongo_client.admin.command('ping')
            return jsonify({"status": "healthy", "mongodb": "connected"}), 200
        except:
            return jsonify({"status": "healthy", "mongodb": "disconnected"}), 200
    return jsonify({"status": "healthy", "mongodb": "not_configured"}), 200

# =============================================================================
# MONGODB API ROUTES - Travel Data
# =============================================================================
# These endpoints fetch travel data from MongoDB database
# Used by: All HTML pages (tokyo.html, kyoto.html, etc.)

@app.route("/attractions")
def api_attractions():
    """
    Get tourist attractions from MongoDB.
    
    Query params:
    - city: Filter by city (e.g., "Tokyo", "Kyoto")
    - area: Filter by area (e.g., "Shibuya", "Ginza")
    - limit: Max results (default: 100)
    
    Example: /attractions?city=Tokyo&limit=50
    Returns: {"count": 50, "data": [{name, lat, lng, url}, ...]}
    """
    # Check if MongoDB is available
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    # Get query parameters from URL
    city = request.args.get("city")
    area = request.args.get("area")
    limit = int_arg("limit", 100)
    
    # Build MongoDB query
    search_query = {}
    
    if city:
        search_query["city"] = city
    
    if area:
        # Use regex for flexible area matching (case-insensitive)
        search_query["area"] = {"$regex": re.escape(area), "$options": "i"}
    
    # Query MongoDB and get results
    results = list(mongo_db["attractions"].find(search_query, {"_id": 0}).limit(limit))
    
    # Return results as JSON
    return jsonify({
        "count": len(results), 
        "data": results
    })

@app.route("/restaurants")
def api_restaurants():
    """
    Get restaurants from MongoDB.
    
    Query params: city, area, limit
    Example: /restaurants?city=Tokyo&area=Shinjuku&limit=100
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    city = request.args.get("city")
    area = request.args.get("area")
    limit = int_arg("limit", 100)
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": re.escape(area), "$options": "i"}
    
    results = list(mongo_db["restaurants"].find(query, {"_id": 0}).limit(limit))
    return jsonify({"count": len(results), "data": results})

@app.route("/shopping")
def api_shopping():
    """
    Get shopping locations from MongoDB.
    
    Query params: city, area, limit
    Example: /shopping?city=Kyoto&limit=50
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    city = request.args.get("city")
    area = request.args.get("area")
    limit = int_arg("limit", 100)
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": re.escape(area), "$options": "i"}
    
    results = list(mongo_db["shopping"].find(query, {"_id": 0}).limit(limit))
    return jsonify({"count": len(results), "data": results})

@app.route("/daytrips")
def api_daytrips():
    """
    Get day trip locations from MongoDB.
    
    Query params: city, area, limit
    Example: /daytrips?city=Tokyo&area=Hakone&limit=20
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    city = request.args.get("city")
    area = request.args.get("area")
    limit = int_arg("limit", 100)
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": re.escape(area), "$options": "i"}
    
    results = list(mongo_db["daytrips"].find(query, {"_id": 0}).limit(limit))
    return jsonify({"count": len(results), "data": results})

@app.route("/search")
def api_search():
    """
    Search for places across all categories using fuzzy matching.
    Handles variations like "Tokyo Skytree" vs "Tokyo Sky Tree", "Sensoji" vs "Sensō-ji".
    
    Query params:
    - q: Search text (e.g., "ramen", "temple", "Tokyo Skytree")
    - limit: Max results per category (default: 20)
    - min_score: Minimum fuzzy match score 0-100 (default: 60)
    
    Example: /search?q=tokyo+skytree&limit=10
    Returns: {attractions: [{name, lat, lng, score}, ...], restaurants: [], ...}
    """
    # Check if MongoDB is available
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    # Get search text from URL
    search_text = request.args.get("q", "")
    if not search_text:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    # Get parameters
    max_results_per_category = int_arg("limit", 20)
    min_fuzzy_score = int_arg("min_score", 60, 0, 100)
    
    # Normalize search text for better matching
    search_normalized = search_text.lower().strip()
    
    # Split search into keywords for broader initial search
    # e.g., "Tokyo Skytree" -> ["tokyo", "skytree"]
    keywords = search_normalized.split()
    
    # Create VERY broad regex pattern using just first 3-4 chars of each word
    # This casts a wide net, then fuzzy matching refines the results
    # e.g., "sensoji" -> "sen", "Tokyo Skytree" -> "tok|sky"
    def get_prefix(word, min_len=3):
        # Get first 3-4 chars to match prefix
        return word[:min(len(word), max(min_len, len(word) // 2))]
    
    # Create a broad regex pattern that matches any keyword prefix
    if len(keywords) > 1:
        # Match if name contains ANY keyword prefix
        keyword_patterns = "|".join([re.escape(get_prefix(k)) for k in keywords])
        search_pattern = {"$regex": keyword_patterns, "$options": "i"}
    else:
        # Single keyword - search for prefix
        prefix = re.escape(get_prefix(search_text))
        search_pattern = {"$regex": prefix, "$options": "i"}
    
    mongodb_query = {"name": search_pattern}
    
    # Helper function to fuzzy match and score results
    def fuzzy_search_collection(collection_name):
        # Get MORE candidates from MongoDB (very broad search)
        # Fuzzy matching will filter and rank them properly
        candidates = list(mongo_db[collection_name].find(mongodb_query, {"_id": 0}).limit(max_results_per_category * 5))
        
        # Apply fuzzy matching to rank results
        scored_results = []
        for item in candidates:
            item_name = item.get("name", "")
            item_city = item.get("city", "").lower()
            
            # Calculate fuzzy match score (0-100)
            # Use token_sort_ratio to ignore word order
            score = fuzz.token_sort_ratio(search_normalized, item_name.lower())
            
            # Also try partial match (for "Skytree" matching "Tokyo Sky Tree")
            partial_score = fuzz.partial_ratio(search_normalized, item_name.lower())
            
            # Use the better score
            final_score = max(score, partial_score)
            
            # Boost score if query mentions a city that matches the result's city
            # e.g., "Tokyo Skytree" gets boosted for Tokyo results
            city_names = ["tokyo", "kyoto", "osaka", "sapporo", "fukuoka", "okinawa", "aomori", "hakodate"]
            for city in city_names:
                if city in search_normalized and city == item_city:
                    # Boost by 15 points if city matches
                    final_score = min(100, final_score + 15)
                    break
            
            # Only include if above minimum score
            if final_score >= min_fuzzy_score:
                item["_match_score"] = final_score
                scored_results.append(item)
        
        # Sort by score (highest first), then by city relevance
        scored_results.sort(key=lambda x: x["_match_score"], reverse=True)
        
        # Return top N results
        return scored_results[:max_results_per_category]
    
    # Search across all 4 collections with fuzzy matching
    results = {
        "query": search_text,
        "attractions": fuzzy_search_collection("attractions"),
        "restaurants": fuzzy_search_collection("restaurants"),
        "shopping": fuzzy_search_collection("shopping"),
        "daytrips": fuzzy_search_collection("daytrips"),
    }
    
    # Calculate total results
    total_count = 0
    for category in results:
        if category != "query":  # Skip the "query" field
            total_count += len(results[category])
    
    results["total_results"] = total_count
    
    return jsonify(results)

# =============================================================================
# USER CUSTOM PLACES - For places not in global database
# =============================================================================

# The categories a traveler can file their own places under:
# id -> (label, which kind of place the planner treats it as)
USER_PLACE_CATEGORIES = {
    "stay":  ("Hotel / where I'm staying", "custom"),
    "sight": ("Sight / thing to see", "attractions"),
    "food":  ("Restaurant", "restaurants"),
    "cafe":  ("Café / sweets", "restaurants"),
    "shop":  ("Shop", "shopping"),
    "other": ("Other", "custom"),
}
MAX_USER_PLACES = 300


def user_place_category(place: Dict[str, Any]) -> str:
    """Stored category; older places (saved as "custom") count as other"""
    cat = str(place.get("category") or "").lower()
    return cat if cat in USER_PLACE_CATEGORIES else "other"


def _user_place_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    """A user's place as sent to the page (only the fields we keep)"""
    return {
        "id": str(doc.get("_id")),
        "name": doc.get("name"),
        "lat": doc.get("lat"),
        "lng": doc.get("lng"),
        "city": doc.get("city"),
        "category": user_place_category(doc),
        "notes": doc.get("notes") or "",
    }


def _clean_user_place(body: Dict[str, Any], partial: bool = False) -> Dict[str, Any]:
    """
    Validate a place sent by the page and keep only known fields.
    partial=True (editing): only the fields that were sent.
    """
    out: Dict[str, Any] = {}
    if not partial or "name" in body:
        name = " ".join(str(body.get("name") or "").split())[:120]
        if not name:
            raise PlanInputError("Give the place a name")
        out["name"] = name
    if not partial or "lat" in body or "lng" in body:
        try:
            lat, lng = float(body["lat"]), float(body["lng"])
        except (KeyError, TypeError, ValueError):
            raise PlanInputError("Invalid GPS coordinates")
        if not (20 <= lat <= 46 and 122 <= lng <= 154):
            raise PlanInputError("Those coordinates aren't in Japan")
        out["lat"], out["lng"] = lat, lng
    if not partial or "city" in body:
        city = str(body.get("city") or "").strip().lower()
        if city not in ROUTE_CITIES:
            raise PlanInputError(f"Unsupported city '{body.get('city')}'")
        out["city"] = ROUTE_CITIES[city]
    if not partial or "category" in body:
        cat = str(body.get("category") or "other").strip().lower()
        if cat not in USER_PLACE_CATEGORIES:
            raise PlanInputError(f"Unknown category '{body.get('category')}'")
        out["category"] = cat
    if not partial or "notes" in body:
        out["notes"] = str(body.get("notes") or "").strip()[:500]
    return out


@app.route("/user/places", methods=["GET", "POST", "PATCH", "DELETE"])
def user_places():
    """
    A traveler's own places (their hotel, a café a friend recommended...).

    They're private: stored in the "user_places" collection with the owner's
    Firebase UID, only ever read or changed for that verified user, and never
    added to the shared place lists everyone sees.

    GET:    {"places": [{id, name, lat, lng, city, category, notes}], "categories": {...}}
    POST:   {name, lat, lng, city, category, notes}  -> {"place": {...}}
    PATCH:  {place_id, ...fields to change}           -> {"place": {...}}
    DELETE: {place_id}

    Requires: Authorization: Bearer <Firebase ID token>
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "Saving your own places isn't available right now."}), 503

    # Get the Firebase UID from the verified ID token
    user_id = get_verified_user_id()
    if not user_id:
        return jsonify({"error": "Authentication required. Please sign in."}), 401
    col = mongo_db["user_places"]
    categories = {k: v[0] for k, v in USER_PLACE_CATEGORIES.items()}

    if request.method == "GET":
        docs = list(col.find({"user_id": user_id}))
        places = sorted((_user_place_out(d) for d in docs), key=lambda p: (p["city"] or "", (p["name"] or "").lower()))
        return jsonify({"places": places, "count": len(places), "categories": categories})

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    if request.method == "POST":
        try:
            place = _clean_user_place(body)
        except PlanInputError as e:
            return jsonify({"error": str(e)}), 400
        if col.count_documents({"user_id": user_id}) >= MAX_USER_PLACES:
            return jsonify({"error": f"You can save up to {MAX_USER_PLACES} places."}), 400
        same = col.find_one({"user_id": user_id, "city": place["city"],
                             "name": {"$regex": f"^{re.escape(place['name'])}$", "$options": "i"}})
        if same:
            return jsonify({"error": f"You already saved “{place['name']}” in {place['city']}."}), 409
        place.update({"user_id": user_id, "created_at": time.time(), "is_public": False})
        result = col.insert_one(place)
        place["_id"] = result.inserted_id
        logger.info("User %s added a place (%s)", user_id, place["category"])
        return jsonify({"success": True, "place": _user_place_out(place)})

    # PATCH / DELETE: only the owner's own place
    from bson import ObjectId
    from bson.errors import InvalidId
    try:
        object_id = ObjectId(str(body.get("place_id")))
    except (InvalidId, TypeError):
        return jsonify({"error": "Invalid place_id"}), 400
    owned = {"_id": object_id, "user_id": user_id}

    if request.method == "PATCH":
        try:
            changes = _clean_user_place({k: v for k, v in body.items() if k != "place_id"}, partial=True)
        except PlanInputError as e:
            return jsonify({"error": str(e)}), 400
        if not changes:
            return jsonify({"error": "Nothing to change"}), 400
        result = col.update_one(owned, {"$set": changes})
        if not result.matched_count:
            return jsonify({"error": "Place not found or not yours"}), 404
        return jsonify({"success": True, "place": _user_place_out(col.find_one(owned))})

    result = col.delete_one(owned)
    if result.deleted_count > 0:
        return jsonify({"success": True, "message": "Place deleted"})
    return jsonify({"error": "Place not found or not yours"}), 404


@app.route("/search/combined")
def search_combined():
    """
    Search BOTH global database AND user's custom places.
    Used by route optimizer and planning pages.
    
    Query params:
    - q: Search text
    - limit: Max results per category (default: 20)
    
    Headers:
    - Authorization: Bearer <Firebase ID token> (optional, for including custom places)
    
    Returns: Combined results from global + user's custom places
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    # Get search parameters
    search_text = request.args.get("q", "")
    if not search_text:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    max_results_per_category = int_arg("limit", 20)
    user_id = get_verified_user_id()  # Optional
    
    # Search global database using existing fuzzy search logic
    search_normalized = search_text.lower().strip()
    keywords = search_normalized.split()
    
    def get_prefix(word, min_len=3):
        return word[:min(len(word), max(min_len, len(word) // 2))]
    
    if len(keywords) > 1:
        keyword_patterns = "|".join([re.escape(get_prefix(k)) for k in keywords])
        search_pattern = {"$regex": keyword_patterns, "$options": "i"}
    else:
        prefix = re.escape(get_prefix(search_text))
        search_pattern = {"$regex": prefix, "$options": "i"}
    
    mongodb_query = {"name": search_pattern}
    
    # Helper function for fuzzy search (copied from /search endpoint)
    def fuzzy_search_collection(collection_name):
        candidates = list(mongo_db[collection_name].find(mongodb_query, {"_id": 0}).limit(max_results_per_category * 5))
        scored_results = []
        for item in candidates:
            item_name = item.get("name", "")
            score = fuzz.token_sort_ratio(search_normalized, item_name.lower())
            partial_score = fuzz.partial_ratio(search_normalized, item_name.lower())
            final_score = max(score, partial_score)
            if final_score >= 60:
                item["_match_score"] = final_score
                scored_results.append(item)
        scored_results.sort(key=lambda x: x["_match_score"], reverse=True)
        return scored_results[:max_results_per_category]
    
    # Search global collections
    global_results = {
        "attractions": fuzzy_search_collection("attractions"),
        "restaurants": fuzzy_search_collection("restaurants"),
        "shopping": fuzzy_search_collection("shopping"),
        "daytrips": fuzzy_search_collection("daytrips"),
    }
    
    # Search user's custom places (if logged in)
    user_custom_places = []
    if user_id:
        user_candidates = list(mongo_db["user_places"].find(
            {
                "user_id": user_id,
                "name": search_pattern
            },
            {"_id": 0}
        ))
        
        # Apply fuzzy matching to user's places
        for item in user_candidates:
            item_name = item.get("name", "")
            score = fuzz.token_sort_ratio(search_normalized, item_name.lower())
            partial_score = fuzz.partial_ratio(search_normalized, item_name.lower())
            final_score = max(score, partial_score)
            if final_score >= 60:
                item["_match_score"] = final_score
                item["is_custom"] = True  # Mark as custom
                user_custom_places.append(item)
        
        user_custom_places.sort(key=lambda x: x["_match_score"], reverse=True)
    
    # Combine results by category
    combined_results = {
        "query": search_text,
        "attractions": global_results["attractions"],
        "restaurants": global_results["restaurants"],
        "shopping": global_results["shopping"],
        "daytrips": global_results["daytrips"],
        "user_custom": user_custom_places,
        "has_custom_places": len(user_custom_places) > 0
    }
    
    # Add custom places to appropriate categories
    for place in user_custom_places:
        category = place.get("category", "custom")
        if category in combined_results:
            # Insert at top of category (user's places first)
            combined_results[category].insert(0, place)
    
    # Calculate total
    total_count = sum(len(combined_results[k]) for k in combined_results if k not in ["query", "has_custom_places"])
    combined_results["total_results"] = total_count
    
    return jsonify(combined_results)


@app.route("/cities")
def api_cities():
    """
    Get list of all cities with document counts.
    
    Example: /cities
    Returns: {"Tokyo": {attractions: 49, restaurants: 577, ...}, "Kyoto": {...}}
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    cities = {}
    for city in mongo_db["attractions"].distinct("city"):
        cities[city] = {
            "attractions": mongo_db["attractions"].count_documents({"city": city}),
            "restaurants": mongo_db["restaurants"].count_documents({"city": city}),
            "shopping": mongo_db["shopping"].count_documents({"city": city}),
            "daytrips": mongo_db["daytrips"].count_documents({"city": city})
        }
    
    return jsonify(cities)

@app.route("/stats")
def api_stats():
    """
    Get overall database statistics.
    
    Example: /stats
    Returns: {total_documents: {attractions: 184, restaurants: 1242, ...}, cities: [...]}
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    return jsonify({
        "total_documents": {
            "attractions": mongo_db["attractions"].count_documents({}),
            "restaurants": mongo_db["restaurants"].count_documents({}),
            "shopping": mongo_db["shopping"].count_documents({}),
            "daytrips": mongo_db["daytrips"].count_documents({})
        },
        "cities": list(mongo_db["attractions"].distinct("city"))
    })

@app.route("/areas/<city>")
def api_areas(city):
    """Get all areas in a specific city"""
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    return jsonify({
        "city": city,
        "areas": {
            "attractions": sorted(mongo_db["attractions"].distinct("area", {"city": city})),
            "restaurants": sorted(mongo_db["restaurants"].distinct("area", {"city": city})),
            "shopping": sorted(mongo_db["shopping"].distinct("area", {"city": city})),
            "daytrips": sorted(mongo_db["daytrips"].distinct("area", {"city": city}))
        }
    })

# =============================================================================
# ROUTE OPTIMIZATION ENDPOINT
# =============================================================================
# Plans the best order for each day, matching place names to GPS on the server

# Supported cities (lowercase key -> name used in MongoDB)
ROUTE_CITIES = {c: c.capitalize() for c in CITY_DIRS.keys()}

# Items like "Lunch" or "Shopping" aren't a specific place - they get scheduled
# at a good time but aren't pinned to a random spot on the map
GENERIC_WORDS = {
    "breakfast", "brunch", "lunch", "dinner", "snack", "coffee", "cafe", "café", "ramen", "sushi",
    "food", "eat", "meal", "shopping", "shop", "souvenirs", "hotel", "check", "in", "out", "check-in",
    "check-out", "rest", "free", "time", "relax", "explore", "walk", "around", "the", "a", "at", "and",
    "local", "nearby", "street", "konbini", "convenience", "store", "night", "bar", "drinks", "izakaya",
    "on", "to", "for", "of", "visit", "go",
}
ROUTE_CITY_NAMES = set(ROUTE_CITIES.keys())
KIND_PRIORITY = {"custom": 0, "attractions": 1, "daytrips": 2, "shopping": 3, "restaurants": 4}
MATCH_MIN_SCORE = 86     # How close a name must be to count as the same place
MATCH_TIE_SCORE = 6      # Candidates this close to the best are compared by distance
MAX_DAYS = 60
MAX_STOPS_PER_DAY = 40


def _is_generic(name: str) -> bool:
    words = re.findall(r"[a-zA-ZÀ-ɏ'-]+", (name or "").lower())
    return bool(words) and all(w in GENERIC_WORDS for w in words)


# Popular neighborhoods people type as a stop ("Gion", "Namba") that the
# database doesn't tag as an area. Approximate center points.
KNOWN_AREAS = {
    "tokyo": {
        "Harajuku": (35.6702, 139.7027), "Omotesando": (35.6652, 139.7123), "Odaiba": (35.6272, 139.7757),
        "Akihabara": (35.6984, 139.7731), "Ueno": (35.7138, 139.7773), "Shimokitazawa": (35.6616, 139.6680),
        "Nakameguro": (35.6440, 139.6990), "Ebisu": (35.6467, 139.7101), "Toyosu": (35.6550, 139.7964),
        "Kichijoji": (35.7033, 139.5797), "Yanaka": (35.7266, 139.7660), "Shinjuku": (35.6909, 139.7003),
        "Shibuya": (35.6595, 139.7005), "Ginza": (35.6717, 139.7650), "Asakusa": (35.7118, 139.7966),
        "Roppongi": (35.6628, 139.7314), "Ikebukuro": (35.7295, 139.7109),
    },
    "kyoto": {
        "Gion": (35.0037, 135.7788), "Arashiyama": (35.0094, 135.6668), "Higashiyama": (34.9965, 135.7810),
        "Pontocho": (35.0050, 135.7708), "Kawaramachi": (35.0037, 135.7690),
    },
    "osaka": {
        "Namba": (34.6659, 135.5013), "Umeda": (34.7025, 135.4959), "Shinsekai": (34.6525, 135.5063),
        "Shinsaibashi": (34.6743, 135.5010), "Tennoji": (34.6466, 135.5133), "Dotonbori": (34.6687, 135.5013),
    },
    "sapporo": {"Susukino": (43.0556, 141.3530)},
    "fukuoka": {"Tenjin": (33.5911, 130.3989), "Nakasu": (33.5938, 130.4067), "Hakata": (33.5902, 130.4207)},
    "okinawa": {"Naha": (26.2125, 127.6809), "Kokusai Street": (26.2155, 127.6866)},
    "hakodate": {"Motomachi": (41.7660, 140.7130)},
}

_areas_cache: Dict[int, Tuple[int, Dict[str, Dict[str, Any]]]] = {}


def _city_areas(city_kb: List[Dict[str, Any]], city_key: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Neighborhoods / day-trip areas in a city and their center point.
    Example: {"shibuya": {"name": "Shibuya", "lat": ..., "lng": ...}, "hakone": {...}}
    """
    cached = _areas_cache.get(id(city_kb))
    if cached and cached[0] == len(city_kb):
        return cached[1]
    points: Dict[str, List[Tuple[float, float]]] = {}
    labels: Dict[str, str] = {}
    for place in city_kb:
        area = (place.get("area") or "").strip()
        key = area.lower()
        # Skip "General", the whole city ("Tokyo"), and messy values like pasted URLs
        if (not key or key == "general" or key in ROUTE_CITY_NAMES or len(key) > 40 or "/" in key
                or place.get("lat") is None or place.get("lng") is None):
            continue
        try:
            points.setdefault(key, []).append((float(place["lat"]), float(place["lng"])))
        except (TypeError, ValueError):
            continue
        labels.setdefault(key, area)
    areas = {
        key: {
            "name": labels[key],
            "matched_name": f"{labels[key]} area",
            "lat": sum(p[0] for p in pts) / len(pts),
            "lng": sum(p[1] for p in pts) / len(pts),
            "_kind": "area",
        }
        for key, pts in points.items()
    }
    # Well-known neighborhoods use their real center instead of an average
    for label, (lat, lng) in KNOWN_AREAS.get(city_key, {}).items():
        areas[label.lower()] = {"name": label, "matched_name": f"{label} area", "lat": lat, "lng": lng,
                                "_kind": "area", "known": True}
    _areas_cache[id(city_kb)] = (len(city_kb), areas)
    return areas


def _area_center(city_kb: List[Dict[str, Any]], text: str, city_key: str = "") -> Optional[Dict[str, Any]]:
    """
    If 'text' names an area ("Shibuya", "Lunch in Ginza", "Shopping at Shinjuku"),
    return that area's center point.
    """
    areas = _city_areas(city_kb, city_key)
    words = re.findall(r"[a-z0-9À-ɏ'-]+", (text or "").lower())
    full = " ".join(words)
    if full in areas:  # e.g. "Kokusai Street" (where "street" is part of the name)
        return areas[full]
    remaining = " ".join(w for w in words if w not in GENERIC_WORDS)
    if not remaining:
        return None
    if remaining in areas:
        return areas[remaining]
    # "Lunch on Kokusai Street" -> "kokusai" should still find "Kokusai Street"
    for key, area in areas.items():
        if " ".join(w for w in key.split() if w not in GENERIC_WORDS) == remaining:
            return area
    return None


# Words that don't help tell places apart ("Meiji Shrine" = "Meiji Jingu")
PLACE_FILLER_WORDS = {"shrine", "temple", "jingu", "jinja", "taisha", "tera", "dera", "the"}
# Different spellings of the same word ("Mount Hakodate" = "Mt. Hakodate")
PLACE_WORD_ALIASES = {"mount": "mt", "st": "station", "stn": "station"}


def normalize_place_name(name: str) -> str:
    """'Sensō-ji Temple / 浅草寺' -> 'senso ji' so spelling variations still match"""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)  # drops Japanese text and punctuation
    words = [PLACE_WORD_ALIASES.get(w, w) for w in text.split()]
    return " ".join(w for w in words if w not in PLACE_FILLER_WORDS)


def _place_score(query: str, choice: str, **kwargs) -> float:
    """Best of a few fuzzy scores; 'skytree' vs 'sky tree' is handled by ignoring spaces"""
    if not query or not choice:
        return 0.0
    no_space = fuzz.ratio(query.replace(" ", ""), choice.replace(" ", ""))
    if len(choice) < 0.6 * len(query):
        # A much shorter name ("mon") shouldn't match just because it's inside the query
        return max(no_space, fuzz.token_sort_ratio(query, choice))
    score = max(fuzz.WRatio(query, choice), no_space)
    if len(query.split()) >= 2:
        # "shibuya crossing" vs "shibuya scramble crossing"
        score = max(score, fuzz.token_set_ratio(query, choice) - 2)
    return score


_normalized_names_cache: Dict[int, Tuple[int, List[str]]] = {}


def _normalized_names(candidates: List[Dict[str, Any]]) -> List[str]:
    cached = _normalized_names_cache.get(id(candidates))
    if cached and cached[0] == len(candidates):
        return cached[1]
    names = [normalize_place_name(c.get("name") or "") for c in candidates]
    _normalized_names_cache[id(candidates)] = (len(candidates), names)
    return names


def match_place(name: str, candidates: List[Dict[str, Any]],
                near: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Find the place a user means by 'name' (handles typos and partial names).
    When several places match about equally well, pick the one closest to 'near'.
    """
    if not name or not candidates or _is_generic(name):
        return None
    query = normalize_place_name(name)
    if not query:
        return None
    if query in ROUTE_CITY_NAMES:
        return None  # "Tokyo" alone isn't a specific place
    normalized = _normalized_names(candidates)
    matches = process.extract(query, normalized, scorer=_place_score, limit=8)
    good = [(score, idx) for _, score, idx in matches if score >= MATCH_MIN_SCORE]

    # One-word queries ("Ichiran", "Todaiji") should match restaurants/shops whose
    # name starts with that word, not ones that merely contain it
    # ("Honda Tokyo Noodle Works"). Sights can contain it anywhere ("Glico Sign Dotonbori").
    if len(query.split()) == 1:
        def one_word_ok(idx: int) -> bool:
            choice = normalized[idx]
            if candidates[idx].get("_kind") not in ("restaurants", "shopping"):
                return True
            first = (choice.split() or [""])[0]
            return fuzz.ratio(query, first) >= 90 or choice.replace(" ", "").startswith(query)
        good = [(score, idx) for score, idx in good if one_word_ok(idx)]
    if not good:
        return None

    best_score = max(score for score, _ in good)
    close = [candidates[idx] for score, idx in good if score >= best_score - MATCH_TIE_SCORE]
    if len(close) > 1:
        # Prefer your own places and sights over restaurants that share the name,
        # then whichever is closest to where the day is
        def rank(c):
            dist = (route_planner.haversine_km(near["lat"], near["lng"], c.get("lat"), c.get("lng"))
                    if near else None)
            return (KIND_PRIORITY.get(c.get("_kind"), 5), dist if dist is not None else float("inf"))
        close.sort(key=rank)
    for place in close:
        if place.get("lat") is not None and place.get("lng") is not None:
            return place
    return None


def _parse_clock(value: Any, default: int) -> int:
    """'09:30' -> 570"""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not m:
        return default
    hours, minutes = int(m.group(1)), int(m.group(2))
    if hours > 23 or minutes > 59:
        return default
    return hours * 60 + minutes


class PlanInputError(ValueError):
    """The trip sent by the page is missing something or invalid (-> HTTP 400)"""


def prepare_trip_days(days_in: Any) -> List[Dict[str, Any]]:
    """
    Normalize the days sent by a planner page and match every stop to a real place.
    Used by both the route optimizer and the AI review.

    Input:  [{"key", "label", "city", "area", "stops": [{"name", "duration"}]}]
    Output: same days with city_key, anchor (start point) and, per stop,
            lat/lng/matched_name/kind (when found), type (meal/evening/...) and generic.
    """
    if not isinstance(days_in, list) or not days_in:
        raise PlanInputError("Send a 'days' list: [{city, area, stops: [{name, duration}]}]")
    if len(days_in) > MAX_DAYS:
        raise PlanInputError(f"Too many days (max {MAX_DAYS})")

    # Normalize input and figure out which cities we need
    days: List[Dict[str, Any]] = []
    for n, day in enumerate(days_in, start=1):
        if not isinstance(day, dict):
            raise PlanInputError(f"Day {n} must be an object")
        city_key = str(day.get("city") or "").strip().lower()
        if city_key not in ROUTE_CITIES:
            raise PlanInputError(f"Day {n}: unsupported city '{day.get('city')}'")
        stops = []
        for stop in day.get("stops") or []:
            if isinstance(stop, str):
                stop = {"name": stop}
            if not isinstance(stop, dict):
                continue
            name = str(stop.get("name") or "").strip()
            if not name:
                continue
            # No duration given -> filled in later with a default for the stop's type
            try:
                duration = int(stop.get("duration"))
                duration = max(5, min(duration, 600))
            except (TypeError, ValueError):
                duration = None
            stops.append({"name": name[:120], "duration": duration})
        # Too many stops for one day: plan the first ones and say which were left out
        dropped = [s["name"] for s in stops[MAX_STOPS_PER_DAY:]]
        stops = stops[:MAX_STOPS_PER_DAY]
        days.append({
            "day": day.get("day") or n,
            "key": day.get("key"),
            "label": str(day.get("label") or f"Day {n}")[:60],
            "city_key": city_key,
            "city": ROUTE_CITIES[city_key],
            "area": str(day.get("area") or "").strip()[:80],
            "stay": str(day.get("stay") or "").strip()[:120],
            "stops": stops,
            "dropped_stops": dropped,
        })

    if not any(d["stops"] for d in days):
        raise PlanInputError("Add some activities to your days first.")

    kb = load_kb_for(sorted({d["city_key"] for d in days}))

    # The user's own places (hotels, friends' houses...) are matched first
    user_places_by_city = user_places_for_request()

    # Match every stop to a real place
    stay_cache: Dict[Tuple[str, str], Any] = {}
    for day in days:
        city_kb = kb.get(day["city_key"], [])
        custom = user_places_by_city.get(day["city_key"], [])

        def locate(text: str, near: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            # Your own places first, then neighborhoods ("Shibuya"), then the database
            return (match_place(text, custom, near)
                    or _area_center(city_kb, text, day["city_key"])
                    or match_place(text, city_kb, near))

        # Where the traveler is staying (the day starts and ends there)
        stay = None
        if day["stay"]:
            stay = stay_cache.get((day["city_key"], day["stay"].lower()))
            if stay is None:
                stay = locate_stay(day["stay"], day["city_key"], custom, city_kb) or False
                stay_cache[(day["city_key"], day["stay"].lower())] = stay
        day["stay_point"] = stay or None
        day["stay_found"] = bool(stay) if day["stay"] else None

        # Where the day starts: its own start area if given, else the stay
        anchor = None
        if day["area"]:
            place = locate(day["area"], None)
            if place:
                anchor = {"name": day["area"], "matched_name": place.get("matched_name") or place.get("name"),
                          "lat": place["lat"], "lng": place["lng"]}
        if anchor is None and stay:
            anchor = stay
        day["anchor"] = anchor
        day["end"] = stay or None

        for stop in day["stops"]:
            # "Lunch: Ichiran Shibuya" = a meal at that restaurant (found by its name)
            meal = MEAL_PREFIX.match(stop["name"])
            place = locate(meal.group(2) if meal else stop["name"], anchor)
            kind = place.get("_kind") if place else None
            cuisine = place.get("cuisine") if place else None
            if kind == "custom":
                # The traveler's own place: plan it as the category they chose
                category = user_place_category(place)
                kind = USER_PLACE_CATEGORIES[category][1]
                cuisine = "cafe" if category == "cafe" else cuisine
            if place:
                stop.update({"lat": place["lat"], "lng": place["lng"], "kind": kind,
                             "matched_name": place.get("matched_name") or place.get("name"),
                             "mine": place.get("_kind") == "custom"})
            stop["type"] = route_planner.classify(stop["name"], kind, cuisine)
            stop["generic"] = _is_generic(stop["name"])

        # Unknown places on a shopping day are probably shops, then fill in missing durations
        route_planner.classify_unknown_by_context(day["stops"])
        for stop in day["stops"]:
            if stop["duration"] is None:
                stop["duration"] = route_planner.default_duration(
                    stop["name"], stop["type"], stop.get("kind"), stop["generic"])
                stop["duration_estimated"] = True

    # Moving to another city after this day: it ends at the station, not back at the hotel
    for k, day in enumerate(days[:-1]):
        if days[k + 1]["city_key"] != day["city_key"]:
            day["end"] = None

    return days


def user_places_for_request() -> Dict[str, List[Dict[str, Any]]]:
    """The signed-in user's own places, by city key (empty when not signed in)"""
    out: Dict[str, List[Dict[str, Any]]] = {}
    user_id = get_verified_user_id()
    if user_id and MONGODB_AVAILABLE and mongo_db is not None:
        for place in mongo_db["user_places"].find({"user_id": user_id}, {"_id": 0}):
            city = str(place.get("city") or "").strip().lower()
            try:
                place["lat"], place["lng"] = float(place["lat"]), float(place["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            place["_kind"] = "custom"
            out.setdefault(city, []).append(place)
    return out


@app.route("/locate-stay")
def api_locate_stay():
    """
    Where is the place the traveler is staying? GET /locate-stay?city=tokyo&q=Hotel Gracery Shinjuku
    Returns {"found": true, "name", "source", "lat", "lng"} or {"found": false}.
    """
    key = str(request.args.get("city") or "").strip().lower()
    text = str(request.args.get("q") or "").strip()[:120]
    if key not in ROUTE_CITIES or not text:
        return jsonify({"error": "Send ?city= and ?q="}), 400
    kb = load_kb_for([key]).get(key, [])
    place = locate_stay(text, key, user_places_for_request().get(key, []), kb)
    if not place:
        return jsonify({"found": False})
    return jsonify({"found": True, "name": place["matched_name"], "source": place["source"],
                    "lat": place["lat"], "lng": place["lng"]})


def locate_stay(text: str, city_key: str, custom: List[Dict[str, Any]],
                city_kb: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find where the traveler is staying: their own places first, then our database,
    then OpenStreetMap (most hotels), then an area named in the text ("... Shinjuku").
    Returns {name, matched_name, lat, lng, source} or None.
    """
    def point(place, source, label=None):
        return {"name": text, "matched_name": label or place.get("matched_name") or place.get("name"),
                "lat": float(place["lat"]), "lng": float(place["lng"]), "source": source}

    place = match_place(text, custom, None)
    if place:
        return point(place, "my place")
    area = _area_center(city_kb, text, city_key)
    if area and len(text.split()) <= 2:  # Just an area ("Shinjuku", "Ueno"): its center is fine
        return point(area, "area")
    place = match_place(text, [p for p in city_kb if p.get("_kind") != "restaurants"], None)
    if place:
        return point(place, "database")
    found = geocode.find(text, ROUTE_CITIES[city_key])
    if found:
        return point(found, "map search", found["name"])
    # "Hotel Gracery Shinjuku" not found: at least the Shinjuku area
    words = re.findall(r"[a-z0-9À-ɏ'-]+", text.lower())
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            area = _area_center(city_kb, " ".join(words[i:i + n]), city_key)
            if area:
                return point(area, "area", f"{area['name']} area (approximate)")
    return None


# "Lunch: <restaurant>", "Dinner - <restaurant>" (written by "Plan my trip", or by hand)
MEAL_PREFIX = re.compile(r"^\s*(breakfast|brunch|lunch|dinner)(?:\s*[:：]|\s+[-–])\s*(.+)$", re.IGNORECASE)


def too_many_stops_text(day: Dict[str, Any]) -> str:
    dropped = day["dropped_stops"]
    names = ", ".join(f"“{n}”" for n in dropped[:3]) + (f" and {len(dropped) - 3} more" if len(dropped) > 3 else "")
    return (f"{day['label']} has more than {MAX_STOPS_PER_DAY} stops, so the last {len(dropped)} "
            f"({names}) weren't included. Try splitting this day.")


def day_picks(day: Dict[str, Any], schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nearby restaurant / shop suggestions for one day's general stops (see place_picks.py)"""
    city_kb = load_kb_for([day["city_key"]]).get(day["city_key"], [])
    return place_picks.picks_for_schedule(schedule, city_kb, day.get("anchor"))


@app.route("/optimize-route", methods=["POST"])
def optimize_route():
    """
    Optimize every day of a trip.

    POST body:
    {
      "days": [
        {"key": "tokyo-1", "label": "Tokyo · Day 1", "city": "Tokyo", "area": "Shibuya",
         "stops": [{"name": "Shibuya Sky", "duration": 90}, {"name": "Lunch", "duration": 60}]}
      ],
      "start_time": "09:00"
    }

    Optional Authorization header (Firebase ID token) lets your custom places be matched too.

    Returns per-day optimized order with a timed schedule, city-to-city travel,
    fare estimates and the best JR Pass option (see route_planner.plan_trip).
    """
    body = request.get_json(silent=True) or {}
    try:
        days = prepare_trip_days(body.get("days"))
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400
    start_minute = _parse_clock(body.get("start_time"), 9 * 60)
    # "as_planned": schedule the days in the traveler's order (calendar / shared view)
    optimize = str(body.get("mode") or "optimize") != "as_planned"

    try:
        result = route_planner.plan_trip(days, start_minute, optimize=optimize)
        # Restaurants / shops from the database for "Lunch", "Ramen", "Shopping"...
        for day, day_result in zip(days, result["days"]):
            if day["dropped_stops"]:
                day_result["warnings"].insert(0, too_many_stops_text(day))
            if day["stay"]:
                sp = day.get("stay_point")
                day_result["stay_point"] = ({"name": sp["matched_name"], "source": sp["source"]} if sp else None)
                if not sp:
                    day_result["warnings"].insert(0, f"Couldn't find where you're staying (“{day['stay']}”) on the map, "
                                                     f"so travel to and from it isn't counted. Try the hotel's full name, "
                                                     f"its area (e.g. “Shinjuku”), or add it under My places.")
            for pick in day_picks(day, day_result["stops"]):
                stop = day_result["stops"][pick["step"] - 1]
                stop["picks"] = pick["options"]
                stop["picks_near"] = pick["near"]
    except Exception:
        logger.exception("Route optimization failed")
        return jsonify({"error": "Route optimization failed. Please try again."}), 500
    return jsonify(result)

# =============================================================================
# PLAN MY TRIP (trip_generator.py)
# =============================================================================

@app.route("/plan-trip", methods=["POST"])
def plan_trip_for_me():
    """
    Build a whole itinerary from the database.

    POST body:
    {"cities": [{"city": "Tokyo", "days": 3}, {"city": "Kyoto", "days": 2}],
     "interests": ["food", "temples", "vintage"], "pace": "relaxed" | "normal" | "packed"}

    Returns:
    {"days": [{"label", "city", "city_key", "day_in_city", "area", "theme",
               "stops": [{"name", "duration", "type"}], "start_time", "end_time", "travel_minutes"}],
     "check": {"score", "issues"}, "notes": [...], "interests": {id: label}}
    """
    body = request.get_json(silent=True) or {}
    try:
        req = trip_generator.parse_request(body, ROUTE_CITIES)
    except trip_generator.GeneratorInputError as e:
        return jsonify({"error": str(e)}), 400

    try:
        kb = load_kb_for(sorted({c["key"] for c in req["cities"]}))
        my_places = user_places_for_request()

        def stay_for(entry):
            return locate_stay(entry["stay"], entry["key"], my_places.get(entry["key"], []), kb.get(entry["key"], []))

        plan = trip_generator.generate_trip(req, kb, lambda key: _city_areas(kb.get(key, []), key),
                                            stay_for=stay_for)
        days = plan["days"]
        # The same checks the AI review starts from, so the plan comes with its own score
        for day in days:
            day.setdefault("dropped_stops", [])
        analysis = ai_reviewer.analyze_trip(days, req["pace"]) if days else {"days": []}
        score = ai_reviewer.rule_score(analysis) if days else None
    except Exception:
        logger.exception("Trip generation failed")
        return jsonify({"error": "Couldn't build a plan. Please try again."}), 500

    out_days = []
    for day, checked in zip(days, analysis.get("days", [])):
        sched = day["schedule"]
        out_days.append({
            "label": day["label"],
            "city": day["city"],
            "city_key": day["city_key"],
            "day_in_city": day["day_in_city"],
            "area": day["area"],
            "stay": day.get("stay", ""),
            "theme": day["theme"],
            "stops": [{"name": s["name"], "duration": s["duration"], "type": s["type"]} for s in day["stops"]],
            "start_time": sched["start_time"],
            "end_time": sched["end_time"],
            "travel_minutes": sched["travel_minutes"],
            "issues": [i["text"] for i in checked.get("issues", [])],
        })
    return jsonify({
        "days": out_days,
        "pace": req["pace"],
        "check": {"score": score},
        "notes": plan["notes"],
        "interests": trip_generator.INTERESTS,
    })


@app.route("/plan-trip/options")
def plan_trip_options():
    """Interests and paces the 'Plan my trip' form can offer"""
    return jsonify({"interests": trip_generator.INTERESTS, "paces": list(ai_reviewer.PACE_LIMITS.keys())})


# =============================================================================
# PLACES OF A CITY (planner search / "add" list)
# =============================================================================

@app.route("/places/<city>")
def api_places(city):
    """
    Every place we know in a city, compact, for the planner's search box.
    Works with MongoDB or the JSON files. Example: /places/tokyo
    """
    key = (city or "").strip().lower()
    if key not in ROUTE_CITIES:
        return jsonify({"error": f"Unsupported city '{city}'"}), 404
    places = []
    seen = set()
    for p in load_kb_for([key]).get(key, []):
        name = (p.get("name") or "").strip()
        if not name or p.get("lat") is None or (name.lower(), p.get("_kind")) in seen:
            continue
        seen.add((name.lower(), p.get("_kind")))
        area = p.get("area")
        places.append({
            "name": name,
            "kind": p.get("_kind"),
            "area": area if area and str(area).lower() not in ("general", key) and len(str(area)) <= 40 else None,
            "cuisine": p.get("cuisine"),
            "rating": p.get("rating") if isinstance(p.get("rating"), (int, float)) else None,
            "price_level": p.get("price_level"),
            "lat": p.get("lat"),
            "lng": p.get("lng"),
            "url": p.get("google_maps_url") or p.get("google_map_url"),
        })
    order = {"attractions": 0, "daytrips": 1, "shopping": 2, "restaurants": 3}
    places.sort(key=lambda x: (order.get(x["kind"], 9), -(x["rating"] or 0), x["name"].lower()))
    resp = jsonify({"city": ROUTE_CITIES[key], "places": places})
    resp.headers["Cache-Control"] = "public, max-age=600"
    return resp


# =============================================================================
# SHARE LINKS
# =============================================================================
# A shared trip is a read-only snapshot anyone with the link can open (view.html).
# Stored in MongoDB ("shared_trips"); kept in memory when MongoDB isn't available
# (local development only: those links disappear when the server restarts).

SHARE_MAX_BYTES = 80_000
_shared_in_memory: Dict[str, Dict[str, Any]] = {}


def clean_trip(trip: Any) -> Dict[str, Any]:
    """Validate a trip sent by the planner: {title, start_date, days: [{city, area, stops}]}"""
    if not isinstance(trip, dict):
        raise PlanInputError("Send a 'trip' object")
    days_in = trip.get("days")
    if not isinstance(days_in, list) or not days_in:
        raise PlanInputError("The trip has no days")
    if len(days_in) > MAX_DAYS:
        raise PlanInputError(f"Too many days (max {MAX_DAYS})")
    days = []
    for n, day in enumerate(days_in, start=1):
        if not isinstance(day, dict):
            raise PlanInputError(f"Day {n} must be an object")
        city = str(day.get("city") or "").strip().lower()
        if city not in ROUTE_CITIES:
            raise PlanInputError(f"Day {n}: unsupported city '{day.get('city')}'")
        stops = []
        for stop in (day.get("stops") or [])[:MAX_STOPS_PER_DAY]:
            if isinstance(stop, str):
                stop = {"name": stop}
            if not isinstance(stop, dict) or not str(stop.get("name") or "").strip():
                continue
            try:
                duration = max(5, min(int(stop.get("duration")), 600))
            except (TypeError, ValueError):
                duration = None
            stops.append({"name": str(stop["name"]).strip()[:120], "duration": duration})
        days.append({"city": city, "area": str(day.get("area") or "").strip()[:80],
                     "stay": str(day.get("stay") or "").strip()[:120], "stops": stops})
    start = trip_checks.parse_date(trip.get("start_date"))
    return {
        "title": str(trip.get("title") or "").strip()[:100] or "My Japan trip",
        "start_date": start.isoformat() if start else None,
        "days": days,
    }


@app.route("/share", methods=["POST"])
def create_share():
    """Save a read-only copy of a trip and return its link: {"id", "url"}"""
    if request.content_length and request.content_length > SHARE_MAX_BYTES:
        return jsonify({"error": "That trip is too big to share"}), 413
    body = request.get_json(silent=True) or {}
    try:
        trip = clean_trip(body.get("trip"))
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400
    share_id = secrets.token_urlsafe(9)
    record = {"_id": share_id, "trip": trip, "owner": get_verified_user_id(), "created_at": time.time()}
    if MONGODB_AVAILABLE and mongo_db is not None:
        try:
            mongo_db["shared_trips"].insert_one(record)
        except Exception:
            logger.exception("Saving a shared trip failed")
            return jsonify({"error": "Couldn't create the link. Please try again."}), 500
    else:
        _shared_in_memory[share_id] = record
    return jsonify({"id": share_id, "url": f"/view.html?share={share_id}"})


@app.route("/share/<share_id>")
def get_share(share_id):
    """A shared trip by id"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,40}", share_id or ""):
        abort(404)
    if MONGODB_AVAILABLE and mongo_db is not None:
        record = mongo_db["shared_trips"].find_one({"_id": share_id})
    else:
        record = _shared_in_memory.get(share_id)
    if not record:
        return jsonify({"error": "This link doesn't exist or was removed."}), 404
    return jsonify({"trip": record["trip"], "created_at": record.get("created_at")})


# =============================================================================
# TRIP CHECKS: dates, weather, closed days (trip_checks.py)
# =============================================================================

@app.route("/trip-check", methods=["POST"])
def trip_check():
    """
    POST {"days": [...same as /optimize-route...], "start_date": "2026-04-02"}
    Returns each day's date, weather (forecast or typical), places that are usually
    closed that day, and suggestions (swap a rainy outdoor day, move a closed place).
    """
    body = request.get_json(silent=True) or {}
    start = trip_checks.parse_date(body.get("start_date"))
    if not start:
        return jsonify({"error": "Choose the trip's start date first."}), 400
    try:
        days = prepare_trip_days(body.get("days"))
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400
    try:
        result = trip_checks.check_trip(days, start, trip_checks.weather_for_city)
    except Exception:
        logger.exception("Trip check failed")
        return jsonify({"error": "The trip check failed. Please try again."}), 500
    for out, day in zip(result["days"], days):
        out["key"] = day.get("key")
    for s in result["suggestions"]:
        s["day_key"] = days[s["day"]].get("key")
        if "to_day" in s:
            s["to_day_key"] = days[s["to_day"]].get("key")
        if "with_day" in s:
            s["with_day_key"] = days[s["with_day"]].get("key")
    return jsonify(result)


# =============================================================================
# CHAT EDITS (chat_editor.py)
# =============================================================================

@app.route("/ai/chat-edit", methods=["POST"])
def ai_chat_edit():
    """
    POST {"days": [...same as /optimize-route...], "message": "make day 2 more relaxed",
          "pace": "normal", "history": [{"role": "user"|"assistant", "text": "..."}]}
    Returns {"reply", "changes": [...], "rejected": [...],
             "days": [{"key", "label", "area", "stops": [{name, duration}], "before", "after"}]}
    'days' are only the changed days, with stops in the optimized order. Nothing is
    changed on the page until the traveler presses Apply.
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "Chat editing needs the AI, which isn't set up on this server."}), 503
    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip()[:500]
    if not message:
        return jsonify({"error": "Type what you'd like to change."}), 400
    pace = str(body.get("pace") or "normal").lower()
    history = [{"role": str(h.get("role"))[:10], "text": str(h.get("text"))[:500]}
               for h in (body.get("history") or []) if isinstance(h, dict)][-chat_editor.HISTORY_TURNS:]
    try:
        days = prepare_trip_days(body.get("days"))
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400

    city_keys = sorted({d["city_key"] for d in days})
    kb = load_kb_for(city_keys)
    analysis = ai_reviewer.analyze_trip(days, pace)
    facts = [{"stops": day["stops"], "end_time": a["end_time"], "travel_minutes": a["travel_minutes"],
              "issues": [i["text"] for i in a["issues"]]} for day, a in zip(days, analysis["days"])]
    user_text = chat_editor.build_user_message(days, facts, chat_editor.build_catalog(city_keys, kb),
                                               message, pace, history)
    try:
        answer = _ask_openai(chat_editor.SYSTEM_PROMPT, user_text, chat_editor.CHAT_SCHEMA)
    except Exception:
        logger.exception("Chat edit AI call failed")
        return jsonify({"error": "The AI couldn't answer right now. Please try again."}), 502

    # The page's days, with durations as the traveler wrote them (blank = estimated)
    page_days, points = [], []
    for day in days:
        page_days.append({
            "key": day.get("key"), "label": day["label"], "city": day["city"], "city_key": day["city_key"],
            "area": day["area"], "stay": day["stay"],
            "stops": [{"name": s["name"], "duration": None if s.get("duration_estimated") else s["duration"]}
                      for s in day["stops"]],
        })
        pts = {s["name"]: (s["lat"], s["lng"]) for s in day["stops"] if s.get("lat") is not None}
        if day.get("anchor"):
            pts["__anchor__"] = (day["anchor"]["lat"], day["anchor"]["lng"])
        points.append(pts)

    editor = chat_editor.Editor(
        page_days, points,
        find_place=lambda key, text, near: match_place(text, kb.get(key, []), near),
        kb_for=lambda key: kb.get(key, []))
    editor.apply(answer.get("operations") or [])

    changed = sorted(editor.changed)
    out_days = []
    if changed:
        try:
            after = prepare_trip_days([{k: editor.days[i][k] for k in ("key", "label", "city", "area", "stay", "stops")}
                                       for i in changed])
        except PlanInputError:
            after = []
        def facts_of(day):
            # Both sides in their best order, so before / after compare fairly
            best = route_planner.DayPlanner(day["stops"], day["city"], 9 * 60, day.get("anchor"), day.get("end")).optimize()
            return best, {"end_time": best["end_time"], "travel_minutes": best["travel_minutes"],
                          "active_minutes": best["activity_minutes"] + best["travel_minutes"],
                          "stops": len(day["stops"])}

        for i, new_day in zip(changed, after):
            best, after_facts = facts_of(new_day)
            _, before_facts = facts_of(days[i])
            by_name = {s["name"]: s for s in editor.days[i]["stops"]}
            out_days.append({
                "key": editor.days[i]["key"],
                "label": editor.days[i]["label"],
                "area": editor.days[i]["area"],
                "stops": [by_name[n] for n in best["optimized_order"] if n in by_name],
                "before": before_facts,
                "after": after_facts,
            })
    return jsonify({
        "reply": str(answer.get("reply") or "")[:800],
        "changes": editor.changes,
        "rejected": editor.rejected,
        "days": out_days,
    })


# =============================================================================
# AI REVIEW ENDPOINT
# =============================================================================
# Reviews a trip plan: travel-time facts + rule checks (ai_reviewer.py), written up
# by the language model, with every suggested edit checked against the real plan

def _legacy_plan_to_days(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Old pages sent {"tokyo": [{"day", "location", "activities"}]}; turn it into 'days'"""
    days = []
    for city, entries in plan.items():
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            activities = re.split(r"[\n/|;,]", str(entry.get("activities") or ""))
            days.append({
                "label": f"{str(city).capitalize()} · Day {entry.get('day') or len(days) + 1}",
                "city": city,
                "area": entry.get("location") or "",
                "stops": [{"name": a.strip()} for a in activities if a.strip()],
            })
    return days


def _ask_openai(system_text: str, user_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Call the model with a strict JSON schema and return the parsed answer"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    message = resp.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused: {message.refusal}")
    return json.loads(message.content)


@app.route("/ai/review", methods=["POST"])
def ai_review():
    """
    Review a trip plan and suggest edits.

    POST body (same days format as /optimize-route):
    {
      "days": [{"key": "tokyo-activities-1", "label": "Tokyo · Day 1", "city": "Tokyo", "area": "Asakusa",
                "stops": [{"name": "Senso-ji", "duration": 60}, ...]}],
      "pace": "relaxed" | "normal" | "packed"
    }

    Returns:
    {
      "overallScore": 7, "summary": "...", "strengths": [...], "risks": [...],
      "suggestions": [{"action": "reorder" | "swap" | "move" | "remove", "dayIndex": 1, "dayKey": "...",
                       "title": "...", "reason": "...", "impact": "...", "details": {...}}],
      "days": [{"label", "end_time", "travel_minutes", "issues": [...]}],
      "source": "ai" | "rules"    // "rules" when the AI is unavailable
    }
    """
    body = request.get_json(silent=True) or {}
    days_in = body.get("days")
    if days_in is None and isinstance(body.get("plan"), dict):
        days_in = _legacy_plan_to_days(body["plan"])
    pace = str(body.get("pace") or body.get("pacePreference") or "normal").lower()

    try:
        days = prepare_trip_days(days_in)
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400

    def ask_model(system_text: str, user_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return _ask_openai(system_text, user_text, schema)
        except Exception:
            logger.exception("AI review call failed; using rule-based review")
            raise

    try:
        result = ai_reviewer.review_trip(days, pace, ask_model if OPENAI_API_KEY else None, picks=day_picks)
    except Exception:
        logger.exception("AI review failed")
        return jsonify({"error": "The review failed. Please try again."}), 500
    return jsonify(result)


@app.route("/ai/debug_review", methods=["POST"])
def ai_debug_review():
    """
    Debug endpoint: the facts and rule checks the AI review is built on (no AI call).
    Only enabled when ENABLE_DEBUG_ENDPOINTS is set.
    """
    if not ENABLE_DEBUG_ENDPOINTS:
        abort(404)
    body = request.get_json(silent=True) or {}
    try:
        days = prepare_trip_days(body.get("days"))
    except PlanInputError as e:
        return jsonify({"error": str(e)}), 400
    analysis = ai_reviewer.analyze_trip(days, str(body.get("pace") or "normal"))
    return jsonify({"analysis": analysis, "referenceScore": ai_reviewer.rule_score(analysis),
                    "ruleSuggestions": ai_reviewer.rule_suggestions(analysis)})


# =============================================================================
# MAIN - Start the Server
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print("\n" + "="*60)
    print("JAPAN TRAVEL APP - Starting Server")
    print("="*60)
    print(f"Server URL: http://localhost:{port}")
    print(f"Your website: http://localhost:{port}/tokyo.html")
    print(f"MongoDB: {'Connected' if (MONGODB_AVAILABLE and mongo_db is not None) else 'Not connected'}")
    print(f"OpenAI: {'Ready' if OPENAI_API_KEY is not None else 'Missing API key'}")
    print("="*60 + "\n")
    
    # Debug mode (auto-reload + interactive error pages) only when FLASK_DEBUG=1
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
