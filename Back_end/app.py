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
import logging
from math import radians, sin, cos, asin, sqrt
from typing import Dict, List, Any, Tuple, Set, Optional

# Import Flask and related frameworks
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import safe_join
from rapidfuzz import process, fuzz
from openai import OpenAI
from dotenv import load_dotenv

# Trying to import MongoDB - falls back to JSON files if not available
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logging.warning("pymongo not installed - MongoDB features disabled")


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

# Initialize OpenAI client for AI reviews
client = OpenAI(api_key=OPENAI_API_KEY)

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


def load_kb_for(cities: List[str]) -> Dict[str, List[Dict[str, Any]]]:
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
                attractions = list(mongo_db["attractions"].find({"city": city_name}, {"_id": 0}))
                restaurants = list(mongo_db["restaurants"].find({"city": city_name}, {"_id": 0}))
                shopping = list(mongo_db["shopping"].find({"city": city_name}, {"_id": 0}))
                daytrips = list(mongo_db["daytrips"].find({"city": city_name}, {"_id": 0}))
                
                # Combine all items
                all_items = attractions + restaurants + shopping + daytrips
                
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
    """
    Finds places even if users misspell them.
    Example: "Shibuya starbux" → Matches "Starbucks Shibuya"
    """
    # Clean up what the user typed
    query = (name or "").strip()
    
    # Return nothing if query is empty or no data available
    if not query or not city_kb:
        return []
    
    # Extract all place names from the knowledge base
    all_place_names = [place.get("name", "") for place in city_kb]
    
    # Use fuzzy matching to find similar names (handles typos)
    matches = process.extract(query, all_place_names, scorer=fuzz.WRatio, limit=limit)
    
    # Keep only the good matches (score 70+)
    results = []
    for matched_name, score, index in matches:
        # Check if match is good enough and index is valid
        if score >= score_cut and 0 <= index < len(city_kb):
            results.append(city_kb[index])
    
    return results

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculates distance between two GPS coordinates in kilometers"""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except Exception:
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def estimate_travel_time(distance_km, mode='train'):
    """
    Estimates travel time based on distance and transport mode.
    
    In Japan:
    - Train: ~30-40 km/h average (including wait times)
    - Walk: ~4 km/h
    - Taxi: ~25 km/h in cities
    """
    if distance_km is None:
        return None
    
    # Add base wait time for trains
    if mode == 'train':
        # Average speed including wait times
        travel_minutes = (distance_km / 35.0) * 60
        wait_time = 5  # Average 5 min wait for trains in Japan
        return round(travel_minutes + wait_time)
    elif mode == 'walk':
        # Walking speed
        return round((distance_km / 4.0) * 60)
    elif mode == 'taxi':
        return round((distance_km / 25.0) * 60)
    else:
        return round((distance_km / 30.0) * 60)

# =============================================================================
# JR PASS COST CALCULATION
# =============================================================================

# City-specific JR line coverage based on actual rail networks
JR_COVERAGE_BY_CITY = {
    'Tokyo': 0.55,      # 55% - Lots of Metro (Ginza, Hibiya, Tozai, etc.)
    'Kyoto': 0.85,      # 85% - Mostly JR (JR Nara, Sagano, etc.)
    'Osaka': 0.75,      # 75% - Mix (JR Loop, but also Osaka Metro)
    'Sapporo': 0.90,    # 90% - Mostly JR Hokkaido
    'Fukuoka': 0.80,    # 80% - JR Kyushu dominates
    'Okinawa': 0.10,    # 10% - No JR! (Okinawa monorail only)
    'Hakodate': 0.95,   # 95% - Almost all JR
    'Aomori': 0.95,     # 95% - JR East dominates
    'Default': 0.70     # Fallback for unlisted cities
}

# Real JR Pass costs (2025 prices)
JR_PASS_COSTS = {
    7: 50000,   # 7-day pass: ¥50,000
    14: 80000,  # 14-day pass: ¥80,000
    21: 100000  # 21-day pass: ¥100,000
}

def get_jr_coverage(city_name):
    """
    Get JR coverage percentage for a city.
    
    Args:
        city_name: Name of the city
    
    Returns:
        JR coverage rate (0.0 to 1.0)
    """
    return JR_COVERAGE_BY_CITY.get(city_name, JR_COVERAGE_BY_CITY['Default'])


def estimate_fare_by_distance(distance_km, city_from='Default', city_to='Default'):
    """
    Estimate train fare based on distance and cities.
    Different distances = different train types = different costs.
    
    Args:
        distance_km: Distance in kilometers
        city_from: Starting city
        city_to: Destination city
    
    Returns:
        Tuple of (base_fare, jr_coverage_rate)
    """
    
    # Within same city (local trains)
    if city_from == city_to:
        if distance_km < 5:
            return round(distance_km * 140), 0.60  # Local metro/JR mix
        elif distance_km < 15:
            return round(distance_km * 150), 0.65  # Local JR/private
        else:
            return round(distance_km * 160), 0.70  # Express local trains
    
    # Inter-city travel
    elif distance_km < 100:
        # Short inter-city (e.g., Tokyo → Yokohama, Kyoto → Osaka)
        return round(distance_km * 180), 0.80  # Likely JR express
    
    elif distance_km < 300:
        # Medium distance (e.g., Tokyo → Nagano, Osaka → Hiroshima)
        return round(distance_km * 200), 0.90  # Likely Shinkansen or Limited Express
    
    else:
        # Long distance (e.g., Tokyo → Kyoto, Tokyo → Sapporo)
        # Definitely Shinkansen
        return round(distance_km * 210), 0.95  # Almost certainly JR Shinkansen


def adjust_for_special_routes(city_from, city_to, distance_km, base_fare, jr_coverage):
    """
    Apply special rules for known routes and edge cases.
    
    Args:
        city_from: Starting city
        city_to: Destination city
        distance_km: Distance in kilometers
        base_fare: Calculated base fare
        jr_coverage: Calculated JR coverage
    
    Returns:
        Tuple of (adjusted_fare, adjusted_jr_coverage)
    """
    
    # Okinawa has NO JR lines - everything is private
    if city_from == 'Okinawa' or city_to == 'Okinawa':
        return base_fare, 0.0  # JR Pass doesn't work
    
    # Tokyo Metro area - very low JR coverage for short trips
    if city_from == 'Tokyo' and city_to == 'Tokyo' and distance_km < 10:
        return base_fare, 0.40  # Mostly Metro, not JR
    
    # Major Shinkansen routes (Tokyo-Kyoto, Tokyo-Osaka, etc.)
    major_routes = [
        ('Tokyo', 'Kyoto'),
        ('Tokyo', 'Osaka'),
        ('Tokyo', 'Hiroshima'),
        ('Tokyo', 'Fukuoka'),
        ('Tokyo', 'Sapporo'),
    ]
    
    if (city_from, city_to) in major_routes or (city_to, city_from) in major_routes:
        return base_fare, 0.95  # Almost certainly JR Shinkansen
    
    return base_fare, jr_coverage


def calculate_jr_pass_recommendation(jr_fares_only, trip_days=7):
    """
    Determine if JR Pass is worth it based on trip length and actual savings.
    
    Args:
        jr_fares_only: Total yen that would be spent on JR lines
        trip_days: Number of days in the trip
    
    Returns:
        Dict with recommendation details
    """
    
    # Choose appropriate pass based on trip length
    if trip_days <= 7:
        pass_cost = JR_PASS_COSTS[7]
        pass_type = "7-day"
    elif trip_days <= 14:
        pass_cost = JR_PASS_COSTS[14]
        pass_type = "14-day"
    else:
        pass_cost = JR_PASS_COSTS[21]
        pass_type = "21-day"
    
    # Calculate net savings (what you save minus pass cost)
    net_savings = jr_fares_only - pass_cost
    is_worth_it = net_savings > 0
    
    # Calculate break-even percentage
    value_percentage = round((jr_fares_only / pass_cost) * 100) if pass_cost > 0 else 0
    
    return {
        "pass_type": pass_type,
        "pass_cost": pass_cost,
        "total_jr_fares": jr_fares_only,  # What you'd pay for JR trains without pass
        "net_savings": net_savings,
        "is_worth_it": is_worth_it,
        "value_percentage": value_percentage,
        "break_even_point": pass_cost,
        "recommendation": (
            f"JR Pass saves you ¥{net_savings:,}!" if is_worth_it
            else f"JR Pass costs ¥{abs(net_savings):,} more than paying per ride"
        )
    }


def optimize_route_dp(locations, start_index=0):
    """
    Uses Dynamic Programming (Held-Karp algorithm) to find optimal route.
    
    This solves the Traveling Salesman Problem (TSP) efficiently:
    - Tries all possible routes using DP
    - Memoizes subproblems to avoid recalculation
    - Time complexity: O(n² × 2ⁿ) - much better than brute force O(n!)
    
    Args:
        locations: List of dicts with 'name', 'lat', 'lng', 'duration' (minutes at location)
        start_index: Index of starting location (usually hotel/first stop)
    
    Returns:
        {
            'optimal_order': [0, 2, 1, 3],  # Indices in best visit order
            'total_time': 180,               # Total minutes including travel
            'route_details': [...]            # Step-by-step breakdown
        }
    """
    n = len(locations)
    
    # Edge cases
    if n == 0:
        return {
            "optimal_order": [], 
            "total_time": 0, 
            "total_travel_time": 0,
            "total_activity_time": 0,
            "route_details": [],
            "optimization_method": "none"
        }
    if n == 1:
        duration = locations[0].get('duration', 60)
        return {
            "optimal_order": [0],
            "total_time": duration,
            "total_travel_time": 0,
            "total_activity_time": duration,
            "route_details": [{
                "step": 1,
                "location": locations[0]['name'], 
                "time_at_location": duration,
                "lat": locations[0].get('lat'),
                "lng": locations[0].get('lng')
            }],
            "optimization_method": "single_location"
        }
    
    # Build distance matrix (travel time between all pairs)
    INF = float('inf')  # Define INF here so both DP and greedy can use it
    travel_time_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                lat1, lng1 = locations[i].get('lat'), locations[i].get('lng')
                lat2, lng2 = locations[j].get('lat'), locations[j].get('lng')
                
                if lat1 and lng1 and lat2 and lng2:
                    dist_km = haversine_km(lat1, lng1, lat2, lng2)
                    
                    # Decide transport mode based on distance
                    if dist_km and dist_km < 1.5:
                        # Walking distance
                        time_estimate = estimate_travel_time(dist_km, 'walk')
                        travel_time_matrix[i][j] = time_estimate if time_estimate else 15
                    else:
                        # Train distance
                        time_estimate = estimate_travel_time(dist_km, 'train')
                        travel_time_matrix[i][j] = time_estimate if time_estimate else 30
                else:
                    # Unknown location - use default estimate
                    travel_time_matrix[i][j] = 30
    
    # For small number of locations (n <= 10), use exact DP solution
    if n <= 10:
        # DP with bitmask memoization (Held-Karp algorithm)
        # dp[mask][last] = minimum time to visit all locations in mask, ending at 'last'
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[(-1, -1)] * n for _ in range(1 << n)]
        
        # Base case: start at start_index
        dp[1 << start_index][start_index] = locations[start_index].get('duration', 60)
        
        # Fill DP table
        for mask in range(1 << n):
            for last in range(n):
                # Skip if this state is impossible
                if dp[mask][last] == INF:
                    continue
                
                # Skip if 'last' is not in the mask
                if not (mask & (1 << last)):
                    continue
                
                # Try adding each unvisited location
                for next_loc in range(n):
                    # Skip if already visited
                    if mask & (1 << next_loc):
                        continue
                    
                    # Calculate new state
                    new_mask = mask | (1 << next_loc)
                    time_to_next = travel_time_matrix[last][next_loc]
                    time_at_next = locations[next_loc].get('duration', 60)
                    new_time = dp[mask][last] + time_to_next + time_at_next
                    
                    # Update if better
                    if new_time < dp[new_mask][next_loc]:
                        dp[new_mask][next_loc] = new_time
                        parent[new_mask][next_loc] = (mask, last)
        
        # Find best final state (all locations visited)
        full_mask = (1 << n) - 1
        best_time = INF
        best_last = -1
        
        for last in range(n):
            if dp[full_mask][last] < best_time:
                best_time = dp[full_mask][last]
                best_last = last
        
        # Reconstruct path
        if best_last == -1:
            # Fallback to original order
            return {
                "optimal_order": list(range(n)),
                "total_time": sum(loc.get('duration', 60) for loc in locations),
                "route_details": []
            }
        
        # Backtrack to get the path
        path = []
        current_mask = full_mask
        current_loc = best_last
        
        while current_loc != -1:
            path.append(current_loc)
            prev_mask, prev_loc = parent[current_mask][current_loc]
            current_mask = prev_mask
            current_loc = prev_loc
        
        path.reverse()
        
        # Build detailed route
        route_details = []
        total_travel = 0
        total_activity = 0
        
        for i, loc_idx in enumerate(path):
            loc = locations[loc_idx]
            activity_time = loc.get('duration', 60)
            
            detail = {
                "step": i + 1,
                "location": loc['name'],
                "time_at_location": activity_time,
                "lat": loc.get('lat'),
                "lng": loc.get('lng')
            }
            
            if i > 0:
                prev_idx = path[i-1]
                travel = travel_time_matrix[prev_idx][loc_idx]
                distance = haversine_km(
                    locations[prev_idx].get('lat'), locations[prev_idx].get('lng'),
                    loc.get('lat'), loc.get('lng')
                )
                
                detail['travel_from_previous'] = travel
                detail['distance_km'] = round(distance, 2) if distance else None
                detail['transport_mode'] = 'walk' if (distance and distance < 1.5) else 'train'
                total_travel += travel
            
            total_activity += activity_time
            route_details.append(detail)
        
        return {
            "optimal_order": path,
            "total_time": round(best_time),
            "total_travel_time": round(total_travel),
            "total_activity_time": total_activity,
            "route_details": route_details,
            "optimization_method": "dynamic_programming"
        }
    
    else:
        # For larger sets (n > 10), use greedy nearest neighbor
        # (exact DP would be too slow)
        visited = [False] * n
        path = [start_index]
        visited[start_index] = True
        current_time = locations[start_index].get('duration', 60)
        
        current = start_index
        for _ in range(n - 1):
            best_next = -1
            best_time = INF
            
            # Find nearest unvisited location
            for next_loc in range(n):
                if not visited[next_loc]:
                    time = travel_time_matrix[current][next_loc]
                    if time < best_time:
                        best_time = time
                        best_next = next_loc
            
            if best_next != -1:
                path.append(best_next)
                visited[best_next] = True
                current_time += best_time + locations[best_next].get('duration', 60)
                current = best_next
        
        # Build route details
        route_details = []
        total_travel = 0
        
        for i, loc_idx in enumerate(path):
            loc = locations[loc_idx]
            detail = {
                "step": i + 1,
                "location": loc['name'],
                "time_at_location": loc.get('duration', 60),
                "lat": loc.get('lat'),
                "lng": loc.get('lng')
            }
            
            if i > 0:
                prev_idx = path[i-1]
                travel = travel_time[prev_idx][loc_idx]
                distance = haversine_km(
                    locations[prev_idx].get('lat'), locations[prev_idx].get('lng'),
                    loc.get('lat'), loc.get('lng')
                )
                detail['travel_from_previous'] = travel
                detail['distance_km'] = round(distance, 2) if distance else None
                detail['transport_mode'] = 'walk' if (distance and distance < 1.5) else 'train'
                total_travel += travel
            
            route_details.append(detail)
        
        return {
            "optimal_order": path,
            "total_time": round(current_time),
            "total_travel_time": round(total_travel),
            "total_activity_time": sum(loc.get('duration', 60) for loc in locations),
            "route_details": route_details,
            "optimization_method": "greedy_nearest_neighbor"
        }

def _tokenize_activities(text: str) -> List[str]:
    """
    Splits activity lists into separate items.
    Example: "Shibuya shopping / Ramen / Meiji Shrine" → ["Shibuya shopping", "Ramen", "Meiji Shrine"]
    Also removes duplicates while keeping the order.
    """
    if not text:
        return []
    
    # Split by common separators: / | ; , or newlines
    raw_items = []
    current_item = []
    
    for character in text:
        # Hit a separator - save current item
        if character in "/|;\n,":
            item = "".join(current_item)
            raw_items.append(item)
            current_item = []  # Start fresh for next item
        else:
            current_item.append(character)
    
    # Don't forget the last item
    if current_item:
        final_item = "".join(current_item)
        raw_items.append(final_item)

    # Remove duplicates but keep the order
    unique_items = []
    seen_items = set()
    
    for item in raw_items:
        cleaned_item = item.strip()
        lowercase_key = cleaned_item.lower()
        
        # Only add if not empty and not seen before
        if cleaned_item and lowercase_key not in seen_items:
            seen_items.add(lowercase_key)
            unique_items.append(cleaned_item)
    
    return unique_items

def enrich_with_kb(plan: Dict[str, Any], kb: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Adds details to the user's plan by matching location names to real places.
    
    Input: User types "Shibuya" 
    Output: Adds GPS coordinates, Google Maps link, category, etc.
    
    Helps the AI understand what the user actually wants to visit.
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
    Create a map of what places the user actually typed for each day.
    
    Example output:
    {
      "tokyo": {
        1: ["Shibuya", "Harajuku", "Ramen"],
        2: ["Asakusa", "Senso-ji", "Tokyo Skytree"]
      }
    }
    
    The AI can only suggest changes using these exact places.
    """
    allowed_places = {}
    
    # Go through each city/region in the plan
    for region, days in enriched.items():
        allowed_places[region] = {}
        
        # Go through each day in this region
        for day_info in days:
            # Get the day number (skip if invalid)
            day_number = int(day_info.get("day") or 0) or 0
            if day_number <= 0:
                continue
            
            # Collect all places mentioned for this day
            places_for_day = []
            
            # Add the main location
            location = (day_info.get("location") or "").strip()
            if location:
                places_for_day.append(location)
            
            # Add activities (split them first)
            activities_text = day_info.get("activities") or ""
            activity_list = _tokenize_activities(activities_text)
            for activity in activity_list:
                places_for_day.append(activity)
            
            # Remove duplicates while keeping order
            unique_places = []
            seen_places = set()
            
            for place in places_for_day:
                cleaned_place = place.strip()
                lowercase_key = cleaned_place.lower()
                
                if cleaned_place and lowercase_key not in seen_places:
                    seen_places.add(lowercase_key)
                    unique_places.append(cleaned_place)
            
            # Only save if there are actual places (skip empty days)
            if unique_places:
                allowed_places[region][day_number] = unique_places
    
    return allowed_places

def _has_legal_option(allowed: Dict[str, Dict[int, List[str]]]) -> bool:
    """
    Check if the AI can make any suggestions.
    Returns True if there are:
    - Days with 2+ items (can swap order)
    - Adjacent days (can move items between them)
    """
    for region, days in allowed.items():
        # Skip if no days in this region
        if not days:
            continue
        
        day_numbers = sorted(days.keys())
        
        # Check if any day has 2+ items (means we can swap)
        for day_num in day_numbers:
            if len(days[day_num]) >= 2:
                return True  # Can swap items within this day
        
        # Check if any day has adjacent days (means we can move items)
        for day_num in day_numbers:
            has_items = len(days[day_num]) >= 1
            has_previous_day = (day_num - 1) in days
            has_next_day = (day_num + 1) in days
            
            if has_items and (has_previous_day or has_next_day):
                return True  # Can move items to adjacent day
    
    # No legal options found
    return False

# =============================================================================
# AI REVIEW CONFIGURATION
# =============================================================================
# Keywords to identify different types of activities

# Words that indicate meal/food activities
MEAL_WORDS = {
    "breakfast", "brunch", "lunch", "dinner", "ramen", "sushi", "yakitori", 
    "katsu", "tempura", "restaurant", "cafe", "coffee", "curry", "tonkatsu", 
    "udon", "soba", "izakaya", "yakiniku", "tatsunoya", "ichiran", "afuri", 
    "ippudo", "maisen", "gyukatsu", "zurrirola"
}

# Words that indicate shopping activities
SHOP_WORDS = {
    "shopping", "mall", "department store", "donki", "don quijote", "vintage", 
    "thrift", "uniqlo", "gu", "beams", "isetan", "marui", "parco", "loft", 
    "tokyu hands"
}

# Words that indicate evening/night activities
EVENING_WORDS = {
    "sky", "observatory", "tower", "night view", "sunset", "illumination"
}

# Thresholds for day analysis
OVERLOAD_THRESHOLD = 8  # Days with 8+ activities might be too packed
LIGHT_THRESHOLD = 2      # Days with 2 or fewer activities might be too light
MEAL_GAP_THRESHOLD = 4   # Days with 4+ activities but no meals need a meal break

def _tag_token(token: str) -> Dict[str, bool]:
    """
    Identify what type of activity this is.
    Example: "Ramen Tatsunoya" → {is_meal: True, is_shop: False, is_evening: False}
    """
    t = (token or "").lower()
    return {
        "is_meal":     any(w in t for w in MEAL_WORDS),      # Is this food-related?
        "is_shop":     any(w in t for w in SHOP_WORDS),      # Is this shopping?
        "is_evening":  any(w in t for w in EVENING_WORDS),   # Is this a night activity?
    }

def _collect_all_items(allowed: Dict[str, Dict[int, List[str]]]) -> Dict[str, List[Tuple[str,int]]]:
    """
    Find all occurrences of each place across all days.
    Used to detect duplicates (visiting same place on multiple days).
    """
    place_occurrences = {}
    
    # Go through each region and day
    for region, days in allowed.items():
        for day_number, items in days.items():
            for item in items:
                # Use lowercase as key to catch duplicates (e.g., "Shibuya" = "shibuya")
                lowercase_item = item.lower()
                
                # Track where this place appears
                if lowercase_item not in place_occurrences:
                    place_occurrences[lowercase_item] = []
                
                place_occurrences[lowercase_item].append((region, day_number))
    
    return place_occurrences

def _summarize_region_day(day_entry: Dict[str, Any]) -> str:
    parts = [f"{day_entry['count']} items"]
    if day_entry.get("meals"):
        parts.append(f"{day_entry['meals']} meal")
    if day_entry.get("evening"):
        parts.append(f"{day_entry['evening']} night view")
    if day_entry.get("shops"):
        parts.append(f"{day_entry['shops']} shopping")
    return ", ".join(parts)

# =============================================================================
# AI ANALYSIS FUNCTIONS
# =============================================================================
# These functions analyze itineraries and detect issues

def _analyze_plan(enriched: Dict[str, List[Dict[str, Any]]],
                  allowed: Dict[str, Dict[int, List[str]]]) -> Dict[str, Any]:
    """
    Analyze the entire itinerary and find potential issues.
    
    Detects:
    - Overloaded days (too many activities)
    - Light days (too few activities)
    - Missing meals (lots of activities but no food)
    - Duplicate places (visiting same place twice)
    
    Returns statistics and issue list for the AI to consider.
    """
    analysis: Dict[str, Any] = {
        "regions": {},
        "issues": {
            "overloadedDays": [],
            "lightDays": [],
            "missingMeals": [],
            "duplicates": [],
        },
        "summary": "",
    }

    duplicates = _collect_all_items(allowed)
    duplicate_items = {
        name: places for name, places in duplicates.items() if len(places) > 1
    }
    analysis["issues"]["duplicates"] = [
        {
            "item": name,
            "occurrences": [(region, day) for region, day in places],
        }
        for name, places in duplicate_items.items()
    ]

    summary_lines: List[str] = []
    total_days = 0
    total_items = 0

    for region in sorted(enriched.keys()):
        days = allowed.get(region, {})
        region_entries: List[Dict[str, Any]] = []
        counts: List[int] = []
        for day in sorted(days.keys()):
            tokens = days[day]
            count = len(tokens)
            meals = sum(1 for t in tokens if _tag_token(t)["is_meal"])
            shops = sum(1 for t in tokens if _tag_token(t)["is_shop"])
            evening = sum(1 for t in tokens if _tag_token(t)["is_evening"])

            region_entries.append({
                "day": day,
                "count": count,
                "meals": meals,
                "shops": shops,
                "evening": evening,
            })

            counts.append(count)
            total_items += count
            total_days += 1

            if count >= OVERLOAD_THRESHOLD:
                analysis["issues"]["overloadedDays"].append({
                    "region": region,
                    "day": day,
                    "count": count,
                })
            if count <= LIGHT_THRESHOLD:
                analysis["issues"]["lightDays"].append({
                    "region": region,
                    "day": day,
                    "count": count,
                })
            if count >= MEAL_GAP_THRESHOLD and meals == 0:
                analysis["issues"]["missingMeals"].append({
                    "region": region,
                    "day": day,
                    "count": count,
                })

        if region_entries:
            analysis["regions"][region] = region_entries
            counts_str = "/".join(str(entry["count"]) for entry in region_entries)
            summary_lines.append(
                f"{region.title()}: {len(region_entries)} day(s), items per day {counts_str}"
            )

    if duplicate_items:
        summary_lines.append(
            f"Duplicates: {', '.join(sorted(name.title() for name in duplicate_items.keys()))}"
        )

    if total_days:
        avg = round(total_items / total_days, 2)
        summary_lines.append(f"Average items/day: {avg}")

    analysis["summary"] = " | ".join(summary_lines) if summary_lines else ""
    return analysis

def _pick_move_candidate(items: List[str]) -> Optional[str]:
    """Choose a token to move, preferring non-meal/non-primary entries."""
    for token in reversed(items):
        tags = _tag_token(token)
        if not tags["is_meal"]:
            return token
    return items[-1] if items else None

def _heuristic_suggestions(enriched: Dict[str, List[Dict[str, Any]]],
                           allowed: Dict[str, Dict[int, List[str]]],
                           analysis: Optional[Dict[str, Any]] = None,
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

    # (4) rebalance overloaded vs light days using analysis, if available
    if analysis:
        by_region = analysis.get("regions", {})
        light_lookup = {
            (issue["region"], issue["day"]): issue
            for issue in analysis.get("issues", {}).get("lightDays", [])
        }
        for issue in analysis.get("issues", {}).get("overloadedDays", []):
            region = issue["region"]
            day = issue["day"]
            items = allowed.get(region, {}).get(day, [])
            if not items:
                continue
            for target_day in (day - 1, day + 1):
                if (region, target_day) not in light_lookup:
                    continue
                candidate = _pick_move_candidate(items)
                if not candidate:
                    continue
                suggestions.append({
                    "region": region,
                    "day": day,
                    "action": "move",
                    "title": f"[{region}] Move “{candidate}” → Day {target_day}",
                    "reason": "Balance a heavy day with a light adjacent day.",
                    "impact": "Creates steadier pacing between days.",
                    "confidence": 0.7,
                    "details": {"item": candidate, "toDay": target_day},
                })
                break

    # Filter illegal and dedupe
    legal = [s for s in suggestions if _suggestion_legal(s, allowed)]
    seen, uniq = set(), []
    for s in legal:
        key = (s["region"], s["day"], s["action"], json.dumps(s.get("details", {}), sort_keys=True))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq[:max_sugs]

# =============================================================================
# AI PROMPT BUILDER
# =============================================================================
# Builds the instructions sent to ChatGPT

def build_reviewer_prompt(enriched: Dict[str, List[Dict[str, Any]]], analysis: Optional[Dict[str, Any]] = None):
    """
    Build the detailed instructions for ChatGPT to review an itinerary.
    
    Includes:
    - What to look for (pacing, meals, timing, flow)
    - Japan-specific travel expertise
    - What suggestions are allowed
    - Scoring criteria
    """
    allowed = _build_allowed_map(enriched)
    has_legal = _has_legal_option(allowed)

    must_propose_text = (
        "If there is literally no legal edit (per constraints), suggestions may be empty."
        if not has_legal else
        "Unless the plan clearly deserves a score ≥ 9, provide at least one valid suggestion."
    )

    days_available = {region: sorted(days.keys()) for region, days in allowed.items()}

    # Build context about the plan
    plan_summary = ""
    if analysis:
        plan_summary = analysis.get("summary", "")
    
    system_text = f"""
You are an expert Japan travel itinerary reviewer with deep knowledge of:
- Japanese culture, transportation, and local customs
- Optimal timing for attractions (temples open early, night views best at sunset)
- Meal timing and restaurant culture (lunch rushes, dinner reservations)
- Efficient routing and neighborhood proximity
- Realistic pacing for different traveler types

Your goal: Provide actionable, specific feedback that helps travelers have the best experience in Japan.

REVIEW GUIDELINES:

1. SCORING CRITERIA (0-10) - BE GENEROUS AND REALISTIC:
   - 9-10: Excellent - Well-balanced, realistic pacing, good meal timing, logical flow
   - 7-8: Good - Minor optimizations possible, generally well-structured. Most solid plans should score here.
   - 5-6: Okay - Some issues with pacing, timing, or flow that need attention
   - 3-4: Weak - Multiple serious problems: actually overloaded, poor timing, missing meals
   - 0-2: Very Weak - Unrealistic, too packed, or poorly organized
   
   DEFAULT TO 7-8 for reasonable plans. Only score 4 or below if there are MAJOR issues.

2. UNDERSTAND CONTEXT:
   - Day 1 arrival: Check-in, lighter activities, shopping-focused is NORMAL and GOOD
   - Shopping lists: "Uniqlo, GU, Beams, Isetan" counts as 1-2 hours of activity, not 4 separate stops
   - Restaurant names: "Ramen Tatsunoya", "Zurrirola", etc. ARE meal breaks
   - Geography: Shinjuku, Shibuya, Harajuku are walkable clusters (15-30 min)
   - Shopping days: 8-10 store names is realistic for a 4-5 hour shopping session

3. EVALUATE THESE ASPECTS:
   - Pacing: Days should have 3-8 actual activities (not counting individual shop names)
   - Timing: Meals at appropriate times? Evening activities scheduled for evening?
   - Flow: Logical sequence? Nearby places grouped? Minimize backtracking?
   - Meals: Look for restaurant names, "lunch", "dinner", "ramen", "sushi" etc.
   - Realism: Can this actually be done in one day? Travel time considered?
   - Variety: Shopping-focused, culture-focused, or food-focused days are ALL valid

4. STRENGTHS should highlight:
   - What's working well (e.g., "Good meal timing", "Well-balanced days", "Logical neighborhood grouping")
   - Smart choices (e.g., "Efficient route through Shibuya", "Good mix of modern and traditional")
   - Recognize when shopping lists are intentional and appropriate
   - Be specific and encouraging

5. RISKS should identify:
   - Actually overloaded days (not just long shopping lists)
   - Truly missing meal breaks (check for restaurant names first!)
   - Poor timing (e.g., observatory at 9am instead of evening)
   - Geographic backtracking (Shinjuku → Ginza → Shinjuku)
   - Unrealistic expectations
   - Be constructive, not just critical

6. SUGGESTIONS - Make SMALL, SURGICAL edits only:
   Legal actions:
     - "swap": Exchange two items within the SAME day (e.g., meal before shopping)
     - "move": Move ONE item to an ADJACENT day (day±1) to balance pacing
     - "trim": Remove a duplicate or low-value item to reduce overload
   
   Forbidden:
     - Adding new places not in AllowedPlaces
     - Moving to non-adjacent days
     - Vague advice without concrete edits
   
   If the plan is already good (score 7+), provide 1-2 minor optimizations or leave suggestions empty.

7. SUGGESTION QUALITY:
   - Title: Clear, actionable (e.g., "Move lunch before shopping" not "Better timing")
   - Reason: Specific why (e.g., "Eating before shopping prevents fatigue and long restaurant queues")
   - Impact: Concrete benefit (e.g., "More energy for browsing, avoids 1-2 hour wait times")
   - Confidence: 0.7-0.9 for high-confidence, 0.5-0.7 for moderate

JAPAN-SPECIFIC CONSIDERATIONS:
- Temples/shrines: Best early morning (6-9am) or late afternoon, avoid midday crowds
- Restaurants: Lunch 11:30-2pm, dinner 6-9pm. Popular spots have 1-2 hour waits
- Shopping clusters: Shinjuku (Isetan, Lumine, Marui), Shibuya (109, Parco), Harajuku (Laforet, Omotesando)
  → Multiple stores in same area = 1 activity session (2-4 hours total)
- Night views: Best 6-9pm (sunset varies by season)
- Transportation: Factor in 15-30 min between areas, rush hour 7-9am, 5-7pm
- Day 1 arrival: Check-in + shopping/light sightseeing is the STANDARD pattern

{plan_summary and f"PLAN SUMMARY: {plan_summary}" or ""}

Coverage rule: {must_propose_text}

Return JSON ONLY with this exact structure:
{{
  "overallScore": <0-10 integer>,
  "strengths": [
    "<specific strength 1>",
    "<specific strength 2>",
    ...
  ],
  "risks": [
    "<specific risk 1>",
    "<specific risk 2>",
    ...
  ],
  "suggestions": [
    {{
      "region": "<one of: {', '.join(enriched.keys())}>",
      "day": <integer>,
      "action": "swap" | "move" | "trim",
      "title": "<Clear, actionable title (max 60 chars)>",
      "reason": "<Specific why this helps (max 200 chars)>",
      "impact": "<Concrete benefit (max 120 chars)>",
      "confidence": <0.0-1.0 float>,
      "details": {{
        // For swap: {{"a": "<item>", "b": "<item>"}}
        // For move: {{"item": "<item>", "toDay": <integer>}}
        // For trim: {{"item": "<item>"}}
      }}
    }}
  ]
}}

CRITICAL CONSTRAINTS:
- Every item in suggestions MUST exist exactly in AllowedPlaces[region][day]
- For move: toDay must be day±1 AND exist in DaysAvailable
- Maximum 6 strengths, 6 risks, 10 suggestions
- Be specific and actionable, not generic

Respond with valid JSON only, no markdown, no explanations.
"""
    user_payload = {
        "Cities": list(enriched.keys()),
        "Days": enriched,
        "AllowedPlaces": allowed,
        "DaysAvailable": days_available,
    }
    
    # Add analysis if available
    if analysis:
        user_payload["Analysis"] = {
            "Summary": analysis.get("summary", ""),
            "Issues": {
                "OverloadedDays": analysis.get("issues", {}).get("overloadedDays", []),
                "LightDays": analysis.get("issues", {}).get("lightDays", []),
                "MissingMeals": analysis.get("issues", {}).get("missingMeals", []),
                "Duplicates": analysis.get("issues", {}).get("duplicates", []),
            }
        }
    
    return system_text, user_payload, allowed

def _suggestion_legal(s: Dict[str, Any], allowed: Dict[str, Dict[int, List[str]]]) -> bool:
    """
    Verify that a suggestion is valid and can actually be applied.
    
    Checks:
    - The places mentioned actually exist in the plan
    - Swap: Both items are on the same day
    - Move: Target day exists and is adjacent (day±1)
    - Trim: Item appears on multiple days
    """
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
    """
    Clean up ChatGPT's response and verify all suggestions are legal.
    
    - Validates suggestions against actual plan
    - Limits text lengths (titles, reasons, impacts)
    - Sorts by confidence
    - Ensures score is 0-10
    """
    out = {
        "overallScore": model_json.get("overallScore", 0),
        "strengths":    model_json.get("strengths") or [],
        "risks":        model_json.get("risks") or [],
        "suggestions":  model_json.get("suggestions") or [],
    }
    if not isinstance(out["strengths"], list): out["strengths"] = []
    if not isinstance(out["risks"], list):     out["risks"] = []
    if not isinstance(out["suggestions"], list): out["suggestions"] = []

    # Clean and validate strengths/risks
    out["strengths"] = [str(s).strip() for s in out["strengths"] if s and str(s).strip()][:6]
    out["risks"] = [str(r).strip() for r in out["risks"] if r and str(r).strip()][:6]
    
    # Remove empty strings
    out["strengths"] = [s for s in out["strengths"] if s]
    out["risks"] = [r for r in out["risks"] if r]

    legal: List[Dict[str, Any]] = []
    for s in out["suggestions"]:
        if not isinstance(s, dict):
            continue
        if _suggestion_legal(s, allowed):
            s["action"]     = str(s.get("action", "")).lower()
            s["reason"]     = (s.get("reason") or "").strip()[:200]
            s["impact"]     = (s.get("impact") or "").strip()[:120]
            s["title"]      = (s.get("title")  or "").strip()[:60] or None
            try:
                conf = float(s.get("confidence", 0.6))
                s["confidence"] = max(0.0, min(1.0, conf))
            except Exception:
                s["confidence"] = 0.6
            legal.append(s)

    # Sort suggestions by confidence (highest first), then limit
    legal.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    out["suggestions"] = legal[:10]

    try:
        score = model_json.get("overallScore", 0)
        if isinstance(score, (int, float)):
            out["overallScore"] = max(0, min(10, int(round(score))))
        else:
            out["overallScore"] = 0
    except Exception:
        out["overallScore"] = 0

    return out

def _prune_bad_risks(out: Dict[str, Any], allowed: Dict[str, Dict[int, List[str]]]) -> None:
    """Drop ‘duplicate across days’ claims when all regions have <2 days."""
    if not out.get("risks"):
        return
    few_days_everywhere = all(len(m.keys()) < 2 for m in allowed.values())
    if few_days_everywhere:
        out["risks"] = [x for x in out["risks"] if "duplicate" not in x.lower()]

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
    limit = int(request.args.get("limit", 100))
    
    # Build MongoDB query
    search_query = {}
    
    if city:
        search_query["city"] = city
    
    if area:
        # Use regex for flexible area matching (case-insensitive)
        search_query["area"] = {"$regex": area, "$options": "i"}
    
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
    limit = int(request.args.get("limit", 100))
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": area, "$options": "i"}
    
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
    limit = int(request.args.get("limit", 100))
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": area, "$options": "i"}
    
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
    limit = int(request.args.get("limit", 100))
    
    query = {}
    if city:
        query["city"] = city
    if area:
        query["area"] = {"$regex": area, "$options": "i"}
    
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
    max_results_per_category = int(request.args.get("limit", 20))
    min_fuzzy_score = int(request.args.get("min_score", 60))
    
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
        keyword_patterns = "|".join([get_prefix(k) for k in keywords])
        search_pattern = {"$regex": keyword_patterns, "$options": "i"}
    else:
        # Single keyword - search for prefix
        prefix = get_prefix(search_text)
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

@app.route("/user/places", methods=["GET", "POST", "DELETE"])
def user_places():
    """
    Manage user's custom places (hotels, friend's houses, local spots, etc.)
    
    GET: Retrieve all custom places for logged-in user
    POST: Add a new custom place
    DELETE: Remove a custom place (requires place_id in body)
    
    Requires: X-User-ID header (Firebase UID)
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    # Get Firebase UID from request header
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Authentication required. Please sign in."}), 401
    
    if request.method == "GET":
        # Get user's custom places
        user_places_list = list(mongo_db["user_places"].find(
            {"user_id": user_id},
            {"_id": 0}  # Exclude MongoDB _id from response
        ))
        return jsonify({"places": user_places_list, "count": len(user_places_list)})
    
    elif request.method == "POST":
        # Add new custom place
        place_data = request.json
        
        # Validate required fields
        required_fields = ["name", "lat", "lng", "city"]
        for field in required_fields:
            if field not in place_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Validate GPS coordinates
        try:
            lat = float(place_data["lat"])
            lng = float(place_data["lng"])
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                raise ValueError("Invalid coordinates")
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid GPS coordinates"}), 400
        
        # Add user metadata
        from datetime import datetime
        place_data["user_id"] = user_id
        place_data["created_at"] = datetime.utcnow().isoformat()
        place_data["category"] = place_data.get("category", "custom")
        place_data["is_public"] = False  # Always private
        
        # Insert into user_places collection
        result = mongo_db["user_places"].insert_one(place_data)
        
        logger.info(f"User {user_id} added custom place: {place_data.get('name')}")
        
        return jsonify({
            "success": True,
            "place_id": str(result.inserted_id),
            "message": f"'{place_data.get('name')}' added successfully!"
        })
    
    elif request.method == "DELETE":
        # Delete a custom place
        place_id = request.json.get("place_id")
        if not place_id:
            return jsonify({"error": "Missing place_id"}), 400
        
        # Delete only if owned by this user (security check)
        from bson import ObjectId
        result = mongo_db["user_places"].delete_one({
            "_id": ObjectId(place_id),
            "user_id": user_id  # Must match logged-in user
        })
        
        if result.deleted_count > 0:
            return jsonify({"success": True, "message": "Place deleted"})
        else:
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
    - X-User-ID: Firebase UID (optional, for including custom places)
    
    Returns: Combined results from global + user's custom places
    """
    if not MONGODB_AVAILABLE or mongo_db is None:
        return jsonify({"error": "MongoDB not available"}), 503
    
    # Get search parameters
    search_text = request.args.get("q", "")
    if not search_text:
        return jsonify({"error": "Missing query parameter 'q'"}), 400
    
    max_results_per_category = int(request.args.get("limit", 20))
    user_id = request.headers.get("X-User-ID")  # Optional
    
    # Search global database using existing fuzzy search logic
    search_normalized = search_text.lower().strip()
    keywords = search_normalized.split()
    
    def get_prefix(word, min_len=3):
        return word[:min(len(word), max(min_len, len(word) // 2))]
    
    if len(keywords) > 1:
        keyword_patterns = "|".join([get_prefix(k) for k in keywords])
        search_pattern = {"$regex": keyword_patterns, "$options": "i"}
    else:
        prefix = get_prefix(search_text)
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
# Smart commute optimizer using dynamic programming

@app.route("/optimize-route", methods=["POST"])
def optimize_route():
    """
    Smart Route Optimizer using Dynamic Programming (Held-Karp algorithm).
    
    Finds the optimal order to visit locations to minimize travel time.
    Uses TSP (Traveling Salesman Problem) solver with DP memoization.
    
    POST body:
    {
        "locations": [
            {"name": "Senso-ji", "lat": 35.7148, "lng": 139.7967, "duration": 90},
            {"name": "Tokyo Skytree", "lat": 35.7101, "lng": 139.8107, "duration": 120},
            {"name": "Shibuya Crossing", "lat": 35.6595, "lng": 139.7004, "duration": 60}
        ],
        "start_index": 0,  # Optional: which location to start from (default: 0)
        "has_jr_pass": false  # Optional: for cost calculation
    }
    
    Returns:
    {
        "original_order": ["Senso-ji", "Skytree", "Shibuya"],
        "original_time": 350,
        "optimal_order": [0, 1, 2],
        "optimal_route": ["Senso-ji", "Skytree", "Shibuya"],
        "total_time": 320,
        "time_saved": 30,
        "route_details": [
            {
                "step": 1,
                "location": "Senso-ji",
                "time_at_location": 90,
                "travel_from_previous": 0
            },
            {
                "step": 2,
                "location": "Tokyo Skytree",
                "time_at_location": 120,
                "travel_from_previous": 15,
                "distance_km": 2.1,
                "transport_mode": "train"
            }
        ],
        "cost_estimate": {"total_yen": 1200, "with_jr_pass": 360, "savings": 840}
    }
    """
    try:
        body = request.get_json(silent=True) or {}
        locations = body.get('locations', [])
        start_index = body.get('start_index', 0)
        has_jr_pass = body.get('has_jr_pass', False)
        
        # Validation
        if not locations:
            return jsonify({"error": "No locations provided"}), 400
        
        if not isinstance(locations, list):
            return jsonify({"error": "Locations must be a list"}), 400
        
        # Validate locations have required fields
        for i, loc in enumerate(locations):
            if not isinstance(loc, dict):
                return jsonify({"error": f"Location {i} must be an object"}), 400
            if 'name' not in loc:
                return jsonify({"error": f"Location {i} missing 'name'"}), 400
        
        # Check if at least some locations have coordinates
        coords_count = sum(1 for loc in locations if loc.get('lat') and loc.get('lng'))
        if coords_count == 0:
            return jsonify({
                "error": "No locations have GPS coordinates",
                "suggestion": "The optimizer needs at least some locations with lat/lng. Try typing exact place names."
            }), 400
        
        # Calculate original order stats (baseline comparison)
        original_time = 0
        original_travel = 0
        
        for i in range(len(locations)):
            original_time += locations[i].get('duration', 60)
            
            if i > 0:
                lat1, lng1 = locations[i-1].get('lat'), locations[i-1].get('lng')
                lat2, lng2 = locations[i].get('lat'), locations[i].get('lng')
                
                if lat1 and lng1 and lat2 and lng2:
                    dist = haversine_km(lat1, lng1, lat2, lng2)
                    travel = estimate_travel_time(dist, 'walk' if (dist and dist < 1.5) else 'train')
                    original_time += travel or 30
                    original_travel += travel or 30
        
        # Run the DP optimizer
        result = optimize_route_dp(locations, start_index)
        
        # Calculate ACCURATE cost estimate using city-specific JR coverage
        total_cost = 0
        jr_fares_only = 0  # Only fares on JR lines
        non_jr_fares = 0   # Fares on private lines
        
        # Build a city lookup for each location
        location_cities = {}
        for i, loc in enumerate(locations):
            # Try to get city from location data (may be passed from frontend)
            city = loc.get('city', 'Default')
            location_cities[i] = city
        
        for idx, detail in enumerate(result['route_details']):
            if 'travel_from_previous' not in detail:
                continue
            
            if detail.get('transport_mode') != 'train':
                continue
            
            distance_km = detail.get('distance_km', 0)
            if distance_km == 0:
                continue
            
            # Get cities for this segment
            loc_index = detail.get('location_index', 0)
            prev_index = loc_index - 1 if loc_index > 0 else 0
            
            city_from = location_cities.get(prev_index, 'Default')
            city_to = location_cities.get(loc_index, 'Default')
            
            # Calculate fare based on distance and cities
            base_fare, jr_coverage = estimate_fare_by_distance(distance_km, city_from, city_to)
            
            # Apply special route adjustments
            base_fare, jr_coverage = adjust_for_special_routes(
                city_from, city_to, distance_km, base_fare, jr_coverage
            )
            
            # Calculate JR vs non-JR split
            jr_portion = round(base_fare * jr_coverage)
            non_jr_portion = base_fare - jr_portion
            
            total_cost += base_fare
            jr_fares_only += jr_portion
            non_jr_fares += non_jr_portion
        
        # Calculate JR Pass recommendation (assume 7-day trip by default)
        trip_days = 7  # Could be passed from frontend in future
        jr_pass_analysis = calculate_jr_pass_recommendation(jr_fares_only, trip_days)
        
        # Cost WITH JR Pass = pass cost + non-JR fares
        cost_with_jr_pass = jr_pass_analysis['pass_cost'] + non_jr_fares
        actual_net_savings = total_cost - cost_with_jr_pass
        
        # Build response
        optimal_route_names = [locations[i]['name'] for i in result['optimal_order']]
        original_route_names = [loc['name'] for loc in locations]
        
        return jsonify({
            "original_order": original_route_names,
            "original_time": round(original_time),
            "original_travel_time": round(original_travel),
            "optimal_order": result['optimal_order'],
            "optimal_route": optimal_route_names,
            "total_time": result['total_time'],
            "time_saved": max(0, round(original_time - result['total_time'])),
            "time_saved_percentage": round((max(0, original_time - result['total_time']) / original_time * 100)) if original_time > 0 else 0,
            "total_travel_time": result['total_travel_time'],
            "total_activity_time": result['total_activity_time'],
            "route_details": result['route_details'],
            "optimization_method": result['optimization_method'],
            "algorithm_used": "Held-Karp DP" if len(locations) <= 10 else "Greedy Nearest Neighbor",
            "cost_estimate": {
                "total_without_pass": total_cost,
                "total_with_pass": cost_with_jr_pass,
                "actual_net_savings": actual_net_savings,
                "jr_fares_only": jr_fares_only,
                "private_line_fares": non_jr_fares,
                "pass_type": jr_pass_analysis['pass_type'],
                "pass_cost": jr_pass_analysis['pass_cost'],
                "is_worth_it": jr_pass_analysis['is_worth_it'],
                "recommendation": jr_pass_analysis['recommendation'],
                "value_percentage": jr_pass_analysis['value_percentage'],
                "breakdown_note": f"Total train fares: ¥{total_cost:,} | JR lines: ¥{jr_fares_only:,} | Private lines: ¥{non_jr_fares:,}"
            }
        })
    
    except Exception as e:
        logger.exception("Route optimization failed")
        # Return detailed error for debugging
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Full traceback: {error_details}")
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "suggestion": "Make sure locations have GPS coordinates (lat/lng). Check server console for details."
        }), 500

# =============================================================================
# AI REVIEW ENDPOINT
# =============================================================================
# This is the main AI feature - analyzes itineraries and suggests improvements

@app.route("/ai/review", methods=["POST"])
def ai_review():
    """
    AI-powered itinerary review using ChatGPT.
    
    POST body:
    {
      "plan": {
        "tokyo": [
          {"day": 1, "location": "Shibuya", "activities": "Shopping, Ramen"},
          {"day": 2, "location": "Asakusa", "activities": "Senso-ji, Tokyo Skytree"}
        ],
        "kyoto": [...]
      }
    }
    
    Returns:
    {
      "overallScore": 7,                              // 0-10 rating
      "strengths": ["Good meal timing", ...],         // What's working well
      "risks": ["Day 1 might be rushed", ...],        // Potential problems
      "suggestions": [                                 // Concrete edits
        {
          "region": "tokyo",
          "day": 1,
          "action": "swap",                            // swap, move, or trim
          "title": "Swap shopping and ramen",
          "reason": "Eat before shopping for better energy",
          "impact": "Avoid hunger and long waits",
          "confidence": 0.8,
          "details": {"a": "Shopping", "b": "Ramen"}
        }
      ]
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
        # Step 1: Load knowledge base (travel data for these cities)
        kb = load_kb_for(cities)
        
        # Step 2: Match user's location names to actual places
        enriched = enrich_with_kb(plan, kb)
        
        # Step 3: Build map of what user actually typed
        allowed = _build_allowed_map(enriched)
        
        # Step 4: Analyze plan for issues
        analysis = _analyze_plan(enriched, allowed)
    except Exception as e:
        logger.exception("KB load/enrich failed")
        return jsonify({"error": f"Internal: failed to load KB: {e}"}), 500

    # Step 5: Build the prompt for ChatGPT
    system_text, user_payload, allowed = build_reviewer_prompt(enriched, analysis)

    # Step 6: Call ChatGPT to analyze the itinerary
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,  # Lower temp = faster, cheaper, more consistent
            top_p=0.9,        # Tighter sampling = fewer tokens
            max_tokens=1500,  # Limit output to reduce costs
            presence_penalty=0.0,  # Simpler = cheaper
            frequency_penalty=0.0,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)},  # No indent = fewer tokens
            ],
            response_format={"type": "json_object"},
        )
        # Step 7: Parse ChatGPT's response
        raw = resp.choices[0].message.content
        model_json = json.loads(raw)
        
        # Step 8: Validate and clean up the response
        polished = _validate_and_polish(model_json, allowed)
        _prune_bad_risks(polished, allowed)

        # Step 9: If ChatGPT didn't give enough suggestions, add our own
        current_suggestions = polished.get("suggestions", [])
        suggestions_count = len(current_suggestions)
        
        if suggestions_count < 2 and _has_legal_option(allowed):
            # Generate backup suggestions using our own logic
            backup_suggestions = _heuristic_suggestions(enriched, allowed, analysis=analysis, max_sugs=3)
            
            # Track which suggestions we already have to avoid duplicates
            existing_keys = set()
            for suggestion in current_suggestions:
                key = (
                    suggestion["region"], 
                    suggestion["day"], 
                    suggestion["action"], 
                    json.dumps(suggestion.get("details", {}), sort_keys=True)
                )
                existing_keys.add(key)
            
            # Add backup suggestions if they're not duplicates
            for suggestion in backup_suggestions:
                key = (
                    suggestion["region"], 
                    suggestion["day"], 
                    suggestion["action"],
                    json.dumps(suggestion.get("details", {}), sort_keys=True)
                )
                
                if key not in existing_keys:
                    polished["suggestions"].append(suggestion)
                    existing_keys.add(key)
            
            # Limit to 5 suggestions total
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
    """
    Debug endpoint to see what the AI can work with.
    Shows the "allowed places" map and enriched data.
    Useful for troubleshooting when suggestions seem wrong.
    """
    body = request.get_json(silent=True) or {}
    plan = body.get("plan") or {}
    cities = [c for c in plan.keys() if c in CITY_DIRS]
    if not cities:
        return jsonify({"error": "No supported cities in plan."}), 400
    kb = load_kb_for(cities)
    enriched = enrich_with_kb(plan, kb)
    allowed = _build_allowed_map(enriched)
    return jsonify({"allowed": allowed, "enriched": enriched})


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
    
    app.run(host="0.0.0.0", port=port, debug=True)
