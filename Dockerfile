# Use a slim Python 3.13 image
FROM python:3.13-slim

# Install system dependencies (including Tesseract OCR & OpenCV libs)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy all source files
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Install the local mrz_scanner module without fetching its dependencies again
RUN pip install --no-deps -e /app/backend/vendor/mrz_scanner

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI app (automatically mounts frontend from /app/frontend)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
