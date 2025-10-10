class TokyoArea:
    def __init__(self, name, description=None, recommended_for=None, nearby_lines=None, hotel_options=None, landmarks=None):
        self.name = name
        self.city = "Tokyo"  
        self.description = description
        self.recommended_for = recommended_for or []
        self.nearby_lines = nearby_lines or []
        self.hotel_options = hotel_options or []
        self.landmarks = landmarks or []

        
