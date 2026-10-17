FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir kafka-python-ng opcua

COPY src/ingest/ /app/src/ingest/

ENV PYTHONUNBUFFERED=1
CMD ["python", "src/ingest/collector.py"]
