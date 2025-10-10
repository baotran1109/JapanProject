import xml.etree.ElementTree as ET
import json

def parse_kml(kml_file, json_file):
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    tree = ET.parse(kml_file)
    root = tree.getroot()

    result = {}

    for folder in root.findall(".//kml:Folder", ns):
        folder_name_elem = folder.find("kml:name", ns)
        if folder_name_elem is None:
            continue
        category = folder_name_elem.text.strip()
        result[category] = []

        for placemark in folder.findall(".//kml:Placemark", ns):
            name_elem = placemark.find("kml:name", ns)
            coords_elem = placemark.find(".//kml:Point/kml:coordinates", ns)
            
            if name_elem is not None and coords_elem is not None:
                name = name_elem.text.strip()
                coords = coords_elem.text.strip().split(",")
                if len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])
                    result[category].append({
                        "name": name,
                        "lat": lat,
                        "lng": lon
                    })

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# Run the conversion
parse_kml("Okinawa Map.kml", "Okinawa_data.json")
