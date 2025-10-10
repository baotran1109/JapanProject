class Trip:
    def __init__(self, trip_name, cities):
        self.trip_name = trip_name
        self.cities = cities
        self.itineraries = {city : [] for city in cities}

    def add_itinerary(self, city, itinerary):
        if city in self.itineraries:
            self.itineraries[city].append(itinerary)
        else:
            print(f"{city} is not in your trip plan")
    