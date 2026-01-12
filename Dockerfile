FROM python:3.13.3-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.prod.txt ./
RUN pip install --no-cache-dir -r requirements.prod.txt

FROM python:3.13.3-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY main.py ./
COPY bootstrap_paths.py ./
COPY aeroapi-python/ ./aeroapi-python/
COPY aeromarket_api/ ./aeromarket_api/
COPY data/ ./data/
COPY features/ ./features/
COPY FlightWeather/ ./FlightWeather/
COPY models/ ./models/
COPY openai_summarizer/ ./openai_summarizer/
COPY openmeteo_api/ ./openmeteo_api/
COPY schemas/ ./schemas/

# Fly will route to internal_port (configured in fly.toml). Bind to 0.0.0.0 so it’s reachable.
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080", "--timeout", "120"]
