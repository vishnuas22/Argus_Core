#!/bin/bash
# Argus Core Backend - Docker Entrypoint Script
# Handles model downloads and application startup

set -e

echo "=========================================="
echo "Argus Core Backend - Starting..."
echo "=========================================="

# Print environment info
echo ""
echo "Environment:"
echo "  - Python: $(python --version)"
echo "  - Working Directory: $(pwd)"
echo "  - Model Directory: ${MODEL_PATH:-/models}"
echo "  - Auto Download: ${AUTO_DOWNLOAD_MODELS:-true}"
echo ""

# Create necessary directories
mkdir -p ${MODEL_PATH:-/models}
mkdir -p /app/logs
mkdir -p /app/.cache

# Check if ONNX Runtime is installed
echo "Checking ONNX Runtime..."
python -c "import onnxruntime; print(f'  - ONNX Runtime: {onnxruntime.__version__}'); print(f'  - Available Providers: {onnxruntime.get_available_providers()}')" || echo "  - ONNX Runtime not found"

# Check hardware capabilities
echo ""
echo "Detecting hardware..."
python -c "
from utils.hardware import detect_hardware, AcceleratorType
hw = detect_hardware()
print(f'  - Accelerator: {hw.accelerator.value}')
print(f'  - Device: {hw.device_name}')
print(f'  - Memory: {hw.memory_mb}MB')
print(f'  - Providers: {hw.available_providers}')
" || echo "  - Hardware detection failed"

# Remove placeholder models (files smaller than 1KB are placeholders)
echo ""
echo "Removing placeholder models..."
find ${MODEL_PATH:-/models} -name "*.onnx" -type f -size -1k -exec rm -f {} \; 2>/dev/null || true
echo "  - Placeholders removed"

# Download essential models if enabled
if [ "${AUTO_DOWNLOAD_MODELS}" = "true" ] && [ "${DOWNLOAD_ON_STARTUP}" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Downloading production models from HuggingFace..."
    echo "=========================================="
    
    # Download image models (essential for deepfake detection)
    echo ""
    echo "Downloading Image models..."
    python -m models.model_downloader --model efficientnet_b3_spatial --models-dir ${MODEL_PATH:-/models} --force || true
    python -m models.model_downloader --model siglip_deepfake --models-dir ${MODEL_PATH:-/models} --force || true
    
    # Download text models
    echo ""
    echo "Downloading Text models..."
    python -m models.model_downloader --model gpt2_perplexity --models-dir ${MODEL_PATH:-/models} || true
    python -m models.model_downloader --model modernbert_ai_detector --models-dir ${MODEL_PATH:-/models} || true
    
    # Download audio models
    echo ""
    echo "Downloading Audio models..."
    python -m models.model_downloader --model wav2vec2_base --models-dir ${MODEL_PATH:-/models} --force || true
    
    # Download video models
    echo ""
    echo "Downloading Video models..."
    python -m models.model_downloader --model xclip_temporal --models-dir ${MODEL_PATH:-/models} --force || true
    
    # Download feature extraction models
    echo ""
    echo "Downloading Feature Extraction models..."
    python -m models.model_downloader --model clip_vit_b16 --models-dir ${MODEL_PATH:-/models} || true
    
    echo ""
    echo "Model download step completed"
fi

# List available models with sizes
echo ""
echo "=========================================="
echo "Available Models Status:"
echo "=========================================="
if [ -d "${MODEL_PATH:-/models}" ]; then
    # Count real models (> 1MB) and placeholders
    real_models=$(find ${MODEL_PATH:-/models} -name "*.onnx" -type f -size +1M 2>/dev/null | wc -l)
    total_models=$(find ${MODEL_PATH:-/models} -name "*.onnx" -type f 2>/dev/null | wc -l)
    
    echo "  - Real models (>1MB): ${real_models}"
    echo "  - Total model files: ${total_models}"
    echo ""
    
    if [ "${total_models}" -gt 0 ]; then
        echo "Model files:"
        find ${MODEL_PATH:-/models} -name "*.onnx" -type f -exec ls -lh {} \; 2>/dev/null | while read line; do
            echo "  ${line}"
        done
    fi
else
    echo "  - Model directory not found"
fi

# Warm up models - preload into memory
echo ""
echo "=========================================="
echo "Warming up models..."
echo "=========================================="
python -c "
import sys
sys.path.insert(0, '/app')
from models.manager import get_model_manager
from models.registry import get_model_registry

try:
    manager = get_model_manager()
    registry = get_model_registry()
    
    # Get list of available models
    models = registry.list_models()
    print(f'Found {len(models)} registered models')
    
    # Preload essential models
    essential = ['efficientnet_b3_spatial', 'siglip_deepfake', 'clip_vit_b16']
    loaded = 0
    for model_name in essential:
        try:
            info = registry.get_model_info(model_name)
            if info and manager.is_model_available(model_name):
                print(f'  Preloading: {model_name}')
                manager.get_session(model_name)
                loaded += 1
        except Exception as e:
            print(f'  Skip {model_name}: {str(e)[:50]}')
    
    print(f'Preloaded {loaded} essential models')
except Exception as e:
    print(f'Model warmup warning: {e}')
" || echo "  - Model warmup skipped"

echo ""
echo "=========================================="
echo "Starting application..."
echo "=========================================="
echo ""

# Execute the main command
exec "$@"
