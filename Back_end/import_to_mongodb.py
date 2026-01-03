# =============================================================================
# MONGODB DATA IMPORT SCRIPT
# =============================================================================
# This script imports all JSON travel data into MongoDB Atlas database.
#
# What it does:
# - Reads JSON files from /data folder
# - Organizes into 4 collections: attractions, restaurants, shopping, daytrips
# - Adds city and area metadata to each place
# - Handles duplicate entries
# =============================================================================

import os
import json
import sys
from pymongo import MongoClient
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding for emoji output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# MONGODB CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Connect to MongoDB Atlas (cloud database)
MONGO_URI = os.getenv("MONGO_URI")
mongodb_client = MongoClient(MONGO_URI)
database = mongodb_client["Japan"]

# Set up collections (like tables in SQL)
attractions_collection = database["attractions"]
restaurants_collection = database["restaurants"]
shopping_collection = database["shopping"]
daytrips_collection = database["daytrips"]

def categorize_file(filename):
    """
    Figure out what type of data the file contains by looking at its name.
    
    Examples:
    - "Tokyo_Attractions.json" → "attractions"
    - "Shibuya_restaurants.json" → "restaurants"
    - "Hakone_daytrip.json" → "daytrips"
    """
    # Convert to lowercase for easier matching
    filename_lowercase = filename.lower()
    
    # Check for keywords in filename
    if 'attraction' in filename_lowercase:
        return 'attractions'
    elif 'restaurant' in filename_lowercase:
        return 'restaurants'
    elif 'shopping' in filename_lowercase:
        return 'shopping'
    elif 'day' in filename_lowercase and 'trip' in filename_lowercase:
        return 'daytrips'
    else:
        # Unknown category
        return None

def extract_area_from_filename(filename):
    """
    Extract the area/neighborhood name from the filename.
    
    Examples:
    - "Shibuya_restaurants.json" → "Shibuya"
    - "Akihabara_shopping.json" → "Akihabara"
    - "Tokyo_Attractions.json" → "Tokyo"
    - "general.json" → "General"
    """
    # Remove the .json extension
    name_without_extension = filename.replace('.json', '')
    
    # Check if filename has underscore (most common pattern)
    if '_' in name_without_extension:
        # Split by underscore
        parts = name_without_extension.split('_')
        
        # First part is usually the area name
        # Example: "Akihabara_restaurants" → ["Akihabara", "restaurants"]
        if len(parts) >= 2:
            return parts[0]
    
    # If no clear pattern, use "General" as default
    return "General"

def import_json_file(filepath, city, category, area):
    """
    Import a single JSON file into MongoDB.
    
    Steps:
    1. Read the JSON file
    2. Add metadata (city, area, source file)
    3. Insert into the correct collection
    
    Returns: Number of documents imported
    """
    print(f"  Importing {filepath.name} → {category} (city: {city}, area: {area})")
    
    try:
        # Step 1: Read the JSON file
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Validate: Data must be a list of places
        if not isinstance(data, list):
            print(f"    Skipping {filepath.name}: not a list format")
            return 0
        
        # Check if file has any data
        if len(data) == 0:
            print(f"    {filepath.name} is empty")
            return 0
        
        # Step 2: Add metadata to each place
        documents_to_insert = []
        
        for place in data:
            # Skip if not a dictionary
            if not isinstance(place, dict):
                continue
            
            # Add location metadata
            place['city'] = city
            place['area'] = area
            place['source_file'] = filepath.name
            
            # Fix: Standardize Google Maps URL field names
            # Some files use 'google_maps_url', others use 'google_map_url'
            if 'google_maps_url' in place and 'google_map_url' not in place:
                place['google_map_url'] = place['google_maps_url']
            if 'google_map_url' in place and 'google_maps_url' not in place:
                place['google_maps_url'] = place['google_map_url']
            
            documents_to_insert.append(place)
        
        # Step 3: Insert into the correct MongoDB collection
        if category == 'attractions':
            insert_result = attractions_collection.insert_many(documents_to_insert)
        elif category == 'restaurants':
            insert_result = restaurants_collection.insert_many(documents_to_insert)
        elif category == 'shopping':
            insert_result = shopping_collection.insert_many(documents_to_insert)
        elif category == 'daytrips':
            insert_result = daytrips_collection.insert_many(documents_to_insert)
        else:
            print(f"    Unknown category: {category}")
            return 0
        
        # Success!
        count_imported = len(insert_result.inserted_ids)
        print(f"    Imported {count_imported} documents")
        return count_imported
    
    except Exception as e:
        print(f"    Error importing {filepath.name}: {e}")
        return 0

def clear_collections():
    """
    Delete all existing data from MongoDB collections.
    This ensures a clean import without duplicates.
    """
    print("\nClearing existing collections...")
    
    # Delete all documents from each collection
    attractions_collection.delete_many({})
    restaurants_collection.delete_many({})
    shopping_collection.delete_many({})
    daytrips_collection.delete_many({})
    
    print("Collections cleared\n")

def import_all_data(data_directory, clear_first=True):
    """
    Import all JSON files from the data directory into MongoDB.
    
    Args:
        data_directory: Path to the data folder (e.g., "../data")
        clear_first: If True, deletes existing data before importing
    
    Returns:
        Total number of documents imported
    """
    data_path = Path(data_directory)
    
    if clear_first:
        clear_collections()
    
    total_imported = 0
    stats = {
        'attractions': 0,
        'restaurants': 0,
        'shopping': 0,
        'daytrips': 0
    }
    
    # Iterate through city directories
    for city_dir in sorted(data_path.iterdir()):
        if not city_dir.is_dir():
            continue
        
        city_name = city_dir.name
        
        # Skip kml data folder
        if city_name.lower() == 'kml data':
            continue
        
        print(f"\n📍 Processing {city_name}...")
        
        # Process all JSON files in this city
        json_files = list(city_dir.glob("*.json"))
        
        for json_file in sorted(json_files):
            category = categorize_file(json_file.name)
            
            if category is None:
                print(f"  Skipping {json_file.name}: couldn't categorize")
                continue
            
            area = extract_area_from_filename(json_file.name)
            count = import_json_file(json_file, city_name, category, area)
            
            total_imported += count
            stats[category] += count
    
    # Print summary
    print("\n" + "="*60)
    print("IMPORT SUMMARY")
    print("="*60)
    print(f"Total documents imported: {total_imported}")
    print(f"  • Attractions: {stats['attractions']}")
    print(f"  • Restaurants: {stats['restaurants']}")
    print(f"  • Shopping: {stats['shopping']}")
    print(f"  • Day Trips: {stats['daytrips']}")
    print("="*60)
    
    # Show collection stats
    print("\n📦 Collection Statistics:")
    print(f"  • attractions: {attractions_col.count_documents({})} documents")
    print(f"  • restaurants: {restaurants_col.count_documents({})} documents")
    print(f"  • shopping: {shopping_col.count_documents({})} documents")
    print(f"  • daytrips: {daytrips_col.count_documents({})} documents")
    
    # Show sample document from each collection
    print("\n📝 Sample Documents:")
    for name, collection in [
        ('attractions', attractions_col),
        ('restaurants', restaurants_col),
        ('shopping', shopping_col),
        ('daytrips', daytrips_col)
    ]:
        sample = collection.find_one()
        if sample:
            print(f"\n{name.upper()}:")
            print(f"  Name: {sample.get('name', 'N/A')}")
            print(f"  City: {sample.get('city', 'N/A')}")
            print(f"  Area: {sample.get('area', 'N/A')}")

if __name__ == "__main__":
    import sys
    
    # Get the data directory (relative to this script)
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    print("Starting MongoDB Import...")
    print(f"Data directory: {data_dir}")
    print(f"MongoDB URI: {MONGO_URI[:50]}...")
    
    # Check for --yes flag to skip confirmation
    skip_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    if not skip_confirm:
        # Ask for confirmation
        try:
            response = input("\nThis will CLEAR existing data. Continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Import cancelled")
                exit(0)
        except EOFError:
            print("\nNo input provided. Use --yes flag to skip confirmation.")
            exit(1)
    else:
        print("\nAuto-confirming (--yes flag provided)")
    
    import_all_data(data_dir, clear_first=True)
    
    print("\nImport completed successfully!")
    client.close()

