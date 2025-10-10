from fastkml import kml
from shapely.geometry import Point

class Attraction:
    def __init__(self, name, area, type, rating=None, description=None, coords=None):
        self.name = name
        self.area = area
        self.type = type
        self.rating = rating
        self.description = description
        self.coords = coords

class Restaurant:
    def __init__(self, name, area, cuisine, rating=None, price_level=None, description=None, coords=None):
        self.name = name
        self.area = area
        self.cuisine = cuisine
        self.rating = rating
        self.price_level = price_level
        self.description = description
        self.coords = coords

class Trip:
    def __init__(self):
        self.areas = {}

    def add_area(self, area_name):
        if area_name not in self.areas:
            self.areas[area_name] = {"restaurants": [], "attractions": []}

    def add_restaurant(self, area_name, restaurant):
        self.add_area(area_name)
        self.areas[area_name]["restaurants"].append(restaurant)

    def add_attraction(self, area_name, attraction):
        self.add_area(area_name)
        self.areas[area_name]["attractions"].append(attraction)

def build_trip_from_kml(filepath):
    trip = Trip()

    with open(filepath, "rb") as f:
        kml_data = f.read()

    k = kml.KML()
    k.from_string(kml_data)

    for document in k.features():
        for folder in document.features():
            layer_name = folder.name.lower()
            for placemark in folder.features():
                coords = None
                if isinstance(placemark.geometry, Point):
                    lon, lat = placemark.geometry.coords[0]
                    coords = (lat, lon)

                if "restaurant" in layer_name:
                    r = Restaurant(
                        name=placemark.name,
                        area="Unknown",
                        cuisine="unknown",
                        coords=coords,
                        description=placemark.description,
                    )
                    trip.add_restaurant("Unknown", r)
                else:
                    a = Attraction(
                        name=placemark.name,
                        area="Unknown",
                        type="unknown",
                        coords=coords,
                        description=placemark.description,
                    )
                    trip.add_attraction("Unknown", a)

    return trip


if __name__ == "__main__":
    trip = build_trip_from_kml("Tokyo shopping.kml")

    for area, data in trip.areas.items():
        print(f"\n--- {area} ---")
        for r in data["restaurants"]:
            print(f"Restaurant: {r.name} - {r.coords}")
        for a in data["attractions"]:
            print(f"Attraction: {a.name} - {a.coords}")
