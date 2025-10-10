class Restaurant:
    def __init__(self, name, area, cuisine, rating=None, price_level=None, description=None, coords=None):
        self.name = name
        self.area = area
        self.cuisine = cuisine
        self.rating = rating
        self.price_level = price_level
        self.description = description
        self.coords = coords