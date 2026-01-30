FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code with new modular structure
COPY backend/main.py .
COPY backend/app/ ./app/

# Copy frontend static files
COPY frontend/index.html ./static/

# Create necessary directories
RUN mkdir -p uploads chroma_db static

# Set environment variables
ENV PYTHONPATH=/app
ENV UPLOAD_DIR=/app/uploads
ENV CHROMA_DIR=/app/chroma_db
ENV STATIC_DIR=/app/static

EXPOSE 8000

# Use production-grade server without --reload
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
