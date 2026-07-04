#!/usr/bin/env bash
# ============================================================
# Argus Core — Mac Apple Silicon Development Setup
# ============================================================
# This script sets up a native Python environment on macOS with
# MPS-enabled PyTorch so the backend can use the M1 Max GPU /
# Neural Engine. Docker on Mac CANNOT access MPS, so we run
# stateful services in Docker and the backend natively.
#
# Run from the project root:
#   chmod +x scripts/setup_mac_dev.sh
#   ./scripts/setup_mac_dev.sh
#
# Prerequisites:
#   - macOS 12.3+ on Apple Silicon (M1/M2/M3/M4)
#   - Xcode Command Line Tools: xcode-select --install
#   - Homebrew: https://brew.sh
#   - Docker Desktop for Mac (for stateful services)
# ============================================================
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect we're on macOS Apple Silicon
if [[ "$(uname)" != "Darwin" ]]; then
    error "This script is for macOS only."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "Not Apple Silicon (arm64). MPS will not be available."
    warn "Continue? (y/N)"
    read -r response
    [[ "$response" =~ ^[Yy]$ ]] || exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"

info "Project root: $PROJECT_ROOT"
info "Backend dir:  $BACKEND_DIR"
info "Venv dir:     $VENV_DIR"

# ============== Step 1: Check prerequisites ==============
info "Step 1: Checking prerequisites..."

command -v brew >/dev/null 2>&1 || error "Homebrew not installed. Install from https://brew.sh"
command -v docker >/dev/null 2>&1 || error "Docker not installed. Install Docker Desktop for Mac."

if ! docker info >/dev/null 2>&1; then
    error "Docker daemon not running. Start Docker Desktop."
fi
success "Docker is running"

if ! command -v python3.11 >/dev/null 2>&1; then
    info "Installing Python 3.11 via Homebrew..."
    brew install python@3.11
fi
success "Python 3.11 available: $(python3.11 --version)"

if ! xcode-select -p >/dev/null 2>&1; then
    info "Installing Xcode Command Line Tools..."
    xcode-select --install
    warn "After Xcode CLT install completes, re-run this script."
    exit 0
fi
success "Xcode Command Line Tools installed"

# ============== Step 2: Start stateful services in Docker ==============
info "Step 2: Starting stateful services (MongoDB, Redis, MinIO) in Docker..."

cd "$PROJECT_ROOT"
docker compose -f docker-compose.mac-dev.yml up -d

info "Waiting for services to become healthy..."
for service in mongodb redis minio; do
    for i in {1..30}; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "argus-$service" 2>/dev/null || echo "starting")
        if [[ "$status" == "healthy" ]]; then
            success "argus-$service is healthy"
            break
        fi
        if [[ $i -eq 30 ]]; then
            error "argus-$service did not become healthy in 30s."
        fi
        sleep 1
    done
done

# ============== Step 3: Create native Python venv ==============
info "Step 3: Creating native Python 3.11 virtual environment..."

if [[ -d "$VENV_DIR" ]]; then
    warn "Venv already exists at $VENV_DIR. Remove and recreate? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        info "Using existing venv"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    python3.11 -m venv "$VENV_DIR"
    success "Created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
success "Activated venv: $(which python)"

pip install --upgrade pip setuptools wheel

# ============== Step 4: Install MPS-enabled PyTorch ==============
info "Step 4: Installing PyTorch with MPS support..."

# Apple Silicon wheels from default PyPI include MPS support.
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
success "PyTorch installed"

info "Verifying MPS availability..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
if torch.backends.mps.is_available():
    x = torch.randn(1000, 1000).to('mps')
    y = x @ x.T
    print('MPS matmul test: OK (shape={})'.format(y.shape))
else:
    print('WARNING: MPS not available. PyTorch will use CPU.')
" || error "PyTorch MPS verification failed"

# ============== Step 5: Install backend dependencies ==============
info "Step 5: Installing backend dependencies..."

cd "$BACKEND_DIR"
pip install -r requirements.txt
success "Backend dependencies installed"

# ============== Step 6: Create .env file ==============
info "Step 6: Creating .env file for Mac development..."

ENV_FILE="$BACKEND_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    warn ".env already exists. Overwrite? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%s)"
    else
        info "Keeping existing .env"
        ENV_SKIP=1
    fi
fi

if [[ -z "${ENV_SKIP:-}" ]]; then
    cat > "$ENV_FILE" << EOF
# Argus Core — Mac Development Environment
# Generated by scripts/setup_mac_dev.sh
ENVIRONMENT=dev
LOG_LEVEL=INFO
LOG_FORMAT=console
JWT_SECRET=dev-only-mac-m1-max-do-not-use-in-prod-32chars
MONGO_URL=mongodb://argusadmin:arguspass123@localhost:27017/?authSource=admin
DB_NAME=argus_core
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
MINIO_ACCESS_KEY=argusadmin
MINIO_SECRET_KEY=arguspass123
MINIO_ENDPOINT=localhost:9000
MINIO_SECURE=false
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
GPU_PROFILE=mps
USE_GPU=true
FALLBACK_TO_CPU=true
EXECUTION_MODE=balanced
MODEL_CACHE_DIR=/Users/$(whoami)/.argus/models
HF_HOME=/Users/$(whoami)/.argus/hf_cache
ENABLE_TENSORRT=false
ENABLE_SOTA_DETECTORS=true
ENABLE_CALIBRATION=true
ENABLE_ATTN_LRP=true
ENABLE_EIGEN_CAM=true
ENABLE_DRIFT_DETECTION=true
EOF
    success ".env created at $ENV_FILE"
fi

# ============== Step 7: Create model directories ==============
info "Step 7: Creating model directories..."

MODEL_DIR="/Users/$(whoami)/.argus/models"
HF_DIR="/Users/$(whoami)/.argus/hf_cache"
mkdir -p "$MODEL_DIR/calibration"
mkdir -p "$MODEL_DIR/continuous_learning"
mkdir -p "$HF_DIR"
success "Model directories created at $MODEL_DIR"

# ============== Step 8: Run test suite ==============
info "Step 8: Running test suite (CPU-runnable tests)..."

cd "$BACKEND_DIR"
python -m pytest tests/ \
    --ignore=tests/test_lip_sync_module.py \
    --ignore=tests/test_training_pipeline.py \
    -q --tb=short 2>&1 | tail -20

success "Test suite complete"

# ============== Done ==============
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Argus Core Mac Development Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the venv:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Verify MPS works:"
echo "     python -c 'import torch; print(\"MPS:\", torch.backends.mps.is_available())'"
echo ""
echo "  3. Start the backend (native, with MPS):"
echo "     cd $BACKEND_DIR"
echo "     uvicorn server:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "  4. In another terminal, start the Celery worker (native):"
echo "     cd $BACKEND_DIR"
echo "     celery -A core.orchestrator.celery_app worker --loglevel=info --pool=solo"
echo ""
echo "  5. (Optional) Start monitoring stack:"
echo "     docker compose -f docker-compose.mac-dev.yml --profile monitoring up -d"
echo ""
echo "  6. (Optional) Start frontend (separate terminal):"
echo "     cd $PROJECT_ROOT/frontend"
echo "     yarn install && yarn dev"
echo ""
echo "Docker services running:"
echo "  - MongoDB:  localhost:27017 (argusadmin / arguspass123)"
echo "  - Redis:    localhost:6379"
echo "  - MinIO:    localhost:9000 (S3 API) / localhost:9001 (Web UI)"
echo ""
warn "Remember: backend + celery run NATIVELY (not in Docker) for MPS access."
