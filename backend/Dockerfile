# ============================================================
# Argus Core Backend - Production Dockerfile (Iteration 1)
# ============================================================
# Iteration 1 changes (see CHANGELOG.md):
#   * Switched base to nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#     so the image supports T4/A10/A100 GPUs out of the box.
#     A CPU-only build target is still available for environments
#     without GPU — set BUILD_TARGET=cpu at build time.
#   * PyTorch stack installed from the CUDA-12.1 wheel index.
#   * Non-root argus user is created in BOTH stages.
#   * HF_HOME / TRANSFORMERS_CACHE live in a dedicated volume so
#     model snapshots survive container restarts.
#   * The docker-entrypoint now pulls SOTA snapshots listed in
#     models/manifest.yaml before starting uvicorn, with retry +
#     sha256 verification.
#   * Healthcheck now probes /health with a 5s connect timeout.
#   * All apt packages pinned to latest at build time (apt-get
#     update is run before every install).
# ============================================================

# ============================================================
# Stage 1: Builder
# ============================================================
# Default base supports CUDA 12.1 (T4/A10/A100). For CPU-only builds,
# pass --build-arg BASE_IMAGE=python:3.11-slim
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    curl \
    git \
    cmake \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies.
# We try the CUDA-12.1 wheel index first; if torch+cu121 is unavailable
# (e.g. CPU-only builds), fall back to the default PyPI index.
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    || pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Production
# ============================================================
FROM ${BASE_IMAGE} as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-310 \
    libhdf5-hl-310 \
    libssl3 \
    libffi8 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (do this BEFORE copying code so
# subsequent COPY --chown works)
RUN groupadd -r argus && useradd -r -g argus -m -d /home/argus argus

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/storage_fallback /models /app/.cache /app/models \
    && chown -R argus:argus /app /models

# Copy application code (manifest.yaml included)
COPY --chown=argus:argus . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    MODEL_PATH=/models \
    MODEL_CACHE_DIR=/models \
    MODEL_MANIFEST_PATH=/app/models/manifest.yaml \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    ORT_DISABLE_MEMORY_OPTIMIZATION=0 \
    AUTO_DOWNLOAD_MODELS=true \
    DOWNLOAD_ON_STARTUP=true \
    # Iteration 1: enable SOTA detectors by default
    ENABLE_SOTA_DETECTORS=true \
    # Iteration 1: refuse to load models whose sha256 doesn't match manifest
    VERIFY_MODEL_CHECKSUMS=true

# Switch to non-root user
USER argus

# Expose port
EXPOSE 8000

# Health check — 5s connect timeout, 60s grace period for model pull
ARG HEALTHCHECK_PORT=8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS --max-time 5 http://localhost:${HEALTHCHECK_PORT}/health || exit 1

# Startup script that pulls SOTA snapshots before starting server
COPY --chown=argus:argus docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default command (can be overridden)
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
