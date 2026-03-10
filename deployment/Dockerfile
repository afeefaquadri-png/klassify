# ──────────────────────────────────────────────────────────────────────────────
# Klassify – Production Dockerfile
# Multi-stage build: deps + app
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for scipy, lightgbm, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/models /app/data/experiments

# Non-root user
RUN addgroup --system klassify && adduser --system --group klassify
RUN chown -R klassify:klassify /app
USER klassify

EXPOSE 8000 8501

# Default: start API server
# Override CMD in docker-compose for Streamlit / Celery
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
