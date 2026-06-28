# Argus Core Backend - Production Dockerfile
# Multi-stage build for optimized image size with ONNX Runtime and ML support

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    # ONNX Runtime dependencies
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .

# Install all dependencies (onnx/onnxruntime already in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libhdf5-hl-310 \
    libssl3t64 \
    libffi8 \
    # Video/Image processing
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    # Additional ML dependencies
    libgomp1 \
    # System utilities
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r argus && useradd -r -g argus argus

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/storage_fallback /models /app/.cache && \
    chown -R argus:argus /app /models

# Copy application code
COPY --chown=argus:argus . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Disable OpenCV GUI
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    # Model paths
    MODEL_PATH=/models \
    MODEL_CACHE_DIR=/models \
    # HuggingFace cache
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    # ONNX Runtime settings
    ORT_DISABLE_MEMORY_OPTIMIZATION=0 \
    # Download settings
    AUTO_DOWNLOAD_MODELS=true \
    DOWNLOAD_ON_STARTUP=true

# Switch to non-root user
USER argus

# Expose port
EXPOSE 8000

# Health check
ARG HEALTHCHECK_PORT=8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${HEALTHCHECK_PORT}/health || exit 1

# Startup script that downloads models before starting server
COPY --chown=argus:argus docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default command (can be overridden)
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
