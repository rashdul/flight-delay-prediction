## Supply Chain API (Flight Delay Prediction)

FastAPI backend that fetches flight + weather data, engineers features, and runs a trained classifier to predict delay risk. It can optionally generate a short natural-language summary using OpenAI.

### Features
- Fetches live/archived flight and weather data for a given flight/date
- Runs a pre-trained delay classifier (`models/delay_classifier_final.pkl`)
- Optional OpenAI summary + risk message
- CORS enabled for local dev and `https://rdulaijan.com`

### Project Structure (high level)
- `main.py` — FastAPI app and endpoints
- `models/` — trained model + model utilities
- `features/` — feature engineering pipeline
- `schemas/` — request/response schemas
- `openai_summarizer/` — optional OpenAI summary helpers
- `index.html` — static site entry (deployed via GitHub Pages workflow)

### Requirements
- Python 3.10+ recommended
- A trained model at `models/delay_classifier_final.pkl`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file (do not commit secrets). Common variables used:
- `API_KEY` / `API_URL`
- `AERODATABOX_API_KEY` / `AERODATABOX_BASE_URL`
- `ARCHIVE_URL`
- `FORECAST_URL`
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (optional, default `gpt-4o-mini`)
- `OPENAI_BASE_URL` (optional, for proxies/compatible providers)
- `OPENAI_TIMEOUT_SECONDS` (optional, default 20)

### Run Locally
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints
`GET /`
- Health check

`GET /flightsWeather`
- Query params:
  - `flight_prefix` (e.g., `UA`)
  - `flight_number` (e.g., `2012`)
  - `flight_date` (YYYY-MM-DD)

`GET /predictDelay`
- Query params:
  - `flight_prefix`
  - `flight_number`
  - `flight_date` (YYYY-MM-DD)
  - `summarize` (bool, default `true`)
  - `summarize_debug` (bool, default `false`)

Example:
```bash
curl "http://localhost:8000/predictDelay?flight_prefix=UA&flight_number=2012&flight_date=2025-01-12"
```

### Notes
- If `OPENAI_API_KEY` is not set, the summary and risk message fall back to deterministic text.
- If the model file is missing, the prediction endpoint returns a 503 with details.

### Deployment
- GitHub Pages workflow publishes `index.html` from this repo on pushes to `main`.
- API deployment can be handled via Docker or your preferred host (see `Dockerfile`/`fly.toml`).
