FROM python:3.10-slim

# Install system dependencies, specifically FFmpeg which is required for pydub to function correctly
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt .

# Install python packages (CPU only versions matching constraints)
# Added --no-cache-dir to keep image lean
RUN pip install --no-cache-dir -r requirements.txt

# Disable HuggingFace symlinks and force CPU Usage via ENV
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_DISABLE_SYMLINKS=1
ENV NUMBA_CACHE_DIR=/tmp
ENV PORT=8000
ENV WEB_CONCURRENCY=1
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"
RUN pip install --no-cache-dir -r requirements.txt
# Copy all application files
COPY . .

# Expose port 8000
EXPOSE 8000

# Start Gunicorn with a configurable worker count.
# Keep the default at 1 because each worker loads the full ML model into memory.
CMD ["sh", "-c", "gunicorn backend.main:app --workers ${WEB_CONCURRENCY:-1} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
