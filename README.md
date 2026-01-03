# Japan Travel Planning Website

A web app for planning trips to Japan. Includes city guides, itinerary planning, and AI-powered route optimization.

## What it does

- City guides for 8 major Japanese cities (Tokyo, Kyoto, Osaka, etc.)
- Itinerary planner with route optimization
- AI review system that analyzes your plans and suggests improvements
- Custom places feature - add your own hotels, friend's houses, etc.
- Search through 1,800+ locations (attractions, restaurants, shopping)

## Tech stack

- Flask (Python backend)
- MongoDB Atlas (database)
- Firebase (auth & hosting)
- OpenAI API (for AI reviews)
- Tailwind CSS (frontend)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
MONGO_URI=your-mongodb-connection-string
OPENAI_API_KEY=your-openai-api-key
```

3. Import data to MongoDB (one-time):
```bash
cd Back_end
python import_to_mongodb.py --yes
```

4. Run the server:
```bash
python Back_end/app.py
```

5. Open http://localhost:8080

## Project structure

```
Back_end/
  app.py              # Main Flask app
  import_to_mongodb.py # Data import script
  wsgi.py             # Gunicorn entry point

Front_end/            # HTML pages
  index.html
  tokyo.html
  plan_tokyo.html
  ... (20+ more pages)

data/                 # JSON source files
  Tokyo/
  Kyoto/
  ...

static/               # Images and JS files
```

## API endpoints

Main endpoints:
- `GET /attractions?city=Tokyo&area=Shibuya`
- `GET /restaurants?city=Kyoto`
- `GET /search?q=ramen`
- `POST /optimize-route` - Route optimization
- `POST /ai/review` - AI itinerary review
- `GET /user/places` - Custom places (requires auth)
- `GET /search/combined` - Search including custom places

## Database

MongoDB database `Japan` with collections:
- `attractions` (184 docs)
- `restaurants` (1,242 docs)
- `shopping` (330 docs)
- `daytrips` (83 docs)
- `user_places` (custom places per user)

## Deployment

Backend is deployed on Google Cloud Run. Frontend is on Firebase Hosting.

To deploy:
```bash
# Backend
gcloud run deploy japan-travel-backend --source . --region us-central1

# Frontend
firebase deploy --only hosting
```

## Notes

- Make sure `.env` is in `.gitignore` and not committed
- MongoDB connection requires IP whitelist setup in Atlas
- Custom places require user authentication (Firebase)

## Data

All location data is from personal travel experiences. Covers 8 cities:
- Tokyo (49 attractions, 577 restaurants)
- Kyoto (26 attractions, 116 restaurants)
- Osaka (22 attractions, 89 restaurants)
- Sapporo, Aomori, Hakodate, Fukuoka, Okinawa

Total: 184 attractions, 1,242 restaurants, 330 shopping spots, 83 day trips.
