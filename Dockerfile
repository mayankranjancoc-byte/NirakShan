# Use stable Python 3.11 for ML and Computer Vision compatibility
FROM python:3.11-slim

# Install system dependencies: Tesseract OCR, libgl for OpenCV, and build tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install clean Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy source code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Install the local mrz_scanner module
RUN pip install --no-deps -e /app/backend/vendor/mrz_scanner

# Set environment paths and flags
ENV PYTHONPATH="/app/backend:/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app/backend

EXPOSE 8000

# Run FastAPI app from backend directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
