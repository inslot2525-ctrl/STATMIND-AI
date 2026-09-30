FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATMIND_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user; runtime data (DB, uploads, models) lives in /data.
# Mount a persistent volume at /data in production (e.g. a Railway volume).
RUN useradd --create-home --uid 1000 statmind \
    && mkdir -p /data \
    && chown -R statmind:statmind /data
USER statmind
VOLUME ["/data"]

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
