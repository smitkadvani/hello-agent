# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
WORKDIR /backend
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY backend /backend

# Expose port
EXPOSE 9000

# Define entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
