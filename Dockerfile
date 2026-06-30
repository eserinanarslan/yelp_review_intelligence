FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs ./configs
COPY src ./src
COPY models ./models

# Optional: copy outputs only if model-info or demo outputs need them
COPY outputs ./outputs

EXPOSE 8000

CMD ["uvicorn", "yelp_review_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]