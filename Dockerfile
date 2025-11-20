FROM python:3.11-slim

# Install FFmpeg and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp

# Expose the default port (Railway sets PORT; default to 8080)
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application with a single worker to keep memory low on small dynos.
# Access/error logs go to stdout/stderr for easier debugging.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 120 --access-logfile - --error-logfile - -b 0.0.0.0:${PORT:-8080} app:app"]
