FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 7860

# Run uvicorn server (Render uses $PORT)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
