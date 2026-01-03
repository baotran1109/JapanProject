# =============================================================================
# KML TO JSON CONVERTER
# =============================================================================
# This script converts Google Maps KML files into JSON format.
#
# What it does:
# - Reads KML files (exported from Google Maps)
# - Extracts place names and GPS coordinates
# - Groups places by category (folders in Google Maps)
# - Saves as clean JSON file
# =============================================================================

import xml.etree.ElementTree as ET
import json

def parse_kml(kml_file_path, output_json_path):
    """
    Convert a KML file to JSON format.
    
    Args:
        kml_file_path: Path to the input KML file (e.g., "Okinawa Map.kml")
        output_json_path: Path for the output JSON file (e.g., "Okinawa_data.json")
    
    Output format:
    {
      "Attractions": [
        {"name": "Shuri Castle", "lat": 26.2173, "lng": 127.7189},
        ...
      ],
      "Restaurants": [...]
    }
    """
    # KML uses XML namespaces - we need to tell the parser about them
    kml_namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    # Parse the KML file
    tree = ET.parse(kml_file_path)
    root = tree.getroot()

    # Store results by category
    places_by_category = {}

    # Find all folders (categories) in the KML
    all_folders = root.findall(".//kml:Folder", kml_namespace)
    
    for folder in all_folders:
        # Get the folder name (e.g., "Attractions", "Restaurants")
        folder_name_element = folder.find("kml:name", kml_namespace)
        
        if folder_name_element is None:
            continue  # Skip folders without names
        
        category_name = folder_name_element.text.strip()
        places_by_category[category_name] = []

        # Find all places (placemarks) in this folder
        all_placemarks = folder.findall(".//kml:Placemark", kml_namespace)
        
        for placemark in all_placemarks:
            # Extract place name
            name_element = placemark.find("kml:name", kml_namespace)
            
            # Extract GPS coordinates
            coordinates_element = placemark.find(".//kml:Point/kml:coordinates", kml_namespace)
            
            # Only include if both name and coordinates exist
            if name_element is not None and coordinates_element is not None:
                place_name = name_element.text.strip()
                
                # Coordinates in KML format: "longitude,latitude,altitude"
                coordinates_text = coordinates_element.text.strip()
                coordinate_parts = coordinates_text.split(",")
                
                # Need at least longitude and latitude
                if len(coordinate_parts) >= 2:
                    longitude = float(coordinate_parts[0])
                    latitude = float(coordinate_parts[1])
                    
                    # Add this place to the category
                    places_by_category[category_name].append({
                        "name": place_name,
                        "lat": latitude,
                        "lng": longitude
                    })

    # Save as JSON file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            places_by_category, 
            json_file, 
            ensure_ascii=False,  # Keep Japanese characters readable
            indent=2  # Format nicely with indentation
        )
    
    print(f"Converted {kml_file_path} → {output_json_path}")
    print(f"   Found {sum(len(places) for places in places_by_category.values())} places in {len(places_by_category)} categories")

# =============================================================================
# RUN THE CONVERSION
# =============================================================================
if __name__ == "__main__":
    # Convert Okinawa map
    parse_kml("Okinawa Map.kml", "Okinawa_data.json")
