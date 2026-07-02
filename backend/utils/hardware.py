"""
Argus Core - Hardware Detection
================================
Detects available hardware accelerators (CUDA, MPS, CPU).

Supports:
- NVIDIA CUDA (CUDAExecutionProvider)
- Apple Metal Performance Shaders (CoreMLExecutionProvider)
- CPU fallback (CPUExecutionProvider)
"""

import os
import platform
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logging import get_logger

logger = get_logger(__name__)


class AcceleratorType(str, Enum):
    """Available accelerator types."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class HardwareInfo:
    """Hardware information container."""
    accelerator: AcceleratorType
    device_name: str
    memory_mb: int
    compute_capability: Optional[str] = None
    supports_fp16: bool = False
    supports_int8: bool = True
    recommended_batch_size: int = 8
    available_providers: List[str] = None
    
    def __post_init__(self):
        if self.available_providers is None:
            self.available_providers = ["CPUExecutionProvider"]


def detect_cuda() -> Optional[HardwareInfo]:
    """
    Detect NVIDIA CUDA availability.
    
    Returns:
        HardwareInfo if CUDA available, None otherwise
    """
    try:
        # Check for nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            logger.debug("nvidia-smi not available or no GPU found")
            return None
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return None
        
        parts = lines[0].split(',')
        device_name = parts[0].strip() if len(parts) > 0 else "Unknown NVIDIA GPU"
        memory_str = parts[1].strip() if len(parts) > 1 else "0 MB"
        compute_cap = parts[2].strip() if len(parts) > 2 else None
        
        # Parse memory (e.g., "4096 MiB" -> 4096)
        memory_mb = 0
        try:
            memory_parts = memory_str.split()
            if memory_parts:
                memory_mb = int(float(memory_parts[0]))
                if "GiB" in memory_str or "GB" in memory_str:
                    memory_mb = int(memory_mb * 1024)
        except (ValueError, IndexError) as e:
            # M6 fix: conservative fallback + loud warning.
            # Previously defaulted to 4GB silently, which over-allocates
            # on 2GB GPUs (OOM) and under-allocates on 16GB GPUs (starved).
            logger.warning(
                "Could not parse nvidia-smi memory output '%s': %s; "
                "assuming conservative 2GB VRAM limit", memory_str, e
            )
            memory_mb = 2048  # conservative fallback
        
        # Check ONNX Runtime CUDA provider
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            has_cuda = "CUDAExecutionProvider" in available_providers
        except ImportError:
            has_cuda = False
            available_providers = ["CPUExecutionProvider"]
        
        if not has_cuda:
            logger.warning("CUDA GPU detected but ONNX Runtime CUDAExecutionProvider not available")
            return None
        
        # Determine capabilities based on compute capability
        supports_fp16 = compute_cap and float(compute_cap) >= 5.3
        
        logger.info(f"CUDA detected: {device_name}, {memory_mb}MB, compute {compute_cap}")
        
        return HardwareInfo(
            accelerator=AcceleratorType.CUDA,
            device_name=device_name,
            memory_mb=memory_mb,
            compute_capability=compute_cap,
            supports_fp16=supports_fp16,
            supports_int8=True,
            recommended_batch_size=_get_recommended_batch_size(memory_mb),
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
    except FileNotFoundError:
        logger.debug("nvidia-smi not found - CUDA not available")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return None
    except Exception as e:
        logger.warning(f"Error detecting CUDA: {e}")
        return None


def detect_mps() -> Optional[HardwareInfo]:
    """
    Detect Apple Metal Performance Shaders (M1/M2/M3).
    
    Returns:
        HardwareInfo if MPS available, None otherwise
    """
    # Only available on macOS
    if platform.system() != "Darwin":
        return None
    
    try:
        # Check for Apple Silicon
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        cpu_brand = result.stdout.strip() if result.returncode == 0 else ""
        
        # Check if Apple Silicon
        is_apple_silicon = False
        device_name = "Apple CPU"
        
        # Check for M1/M2/M3
        if "Apple M1" in cpu_brand or "Apple M2" in cpu_brand or "Apple M3" in cpu_brand:
            is_apple_silicon = True
            device_name = cpu_brand
        else:
            # Alternative check via architecture
            arch = platform.machine()
            if arch == "arm64":
                is_apple_silicon = True
                device_name = "Apple Silicon (ARM64)"
        
        if not is_apple_silicon:
            logger.debug("Not Apple Silicon - MPS not available")
            return None
        
        # Check for CoreML provider in ONNX Runtime
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            has_coreml = "CoreMLExecutionProvider" in available_providers
        except ImportError:
            has_coreml = False
            available_providers = ["CPUExecutionProvider"]
        
        # MPS unified memory - estimate based on system memory
        try:
            import os
            # Get system memory in bytes, convert to MB
            system_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            memory_mb = (system_memory // (1024 * 1024)) // 2  # Assume half for GPU
        except Exception:
            logger.warning("Could not determine system memory for MPS, defaulting to 8GB")
            memory_mb = 8192  # Default 8GB assumption for Apple Silicon
        
        logger.info(f"MPS detected: {device_name}, unified memory ~{memory_mb}MB")
        
        return HardwareInfo(
            accelerator=AcceleratorType.MPS,
            device_name=device_name,
            memory_mb=memory_mb,
            supports_fp16=True,  # Apple Silicon has excellent FP16 support
            supports_int8=True,
            recommended_batch_size=_get_recommended_batch_size(memory_mb),
            available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"] if has_coreml else ["CPUExecutionProvider"]
        )
        
    except Exception as e:
        logger.warning(f"Error detecting MPS: {e}")
        return None


def detect_cpu() -> HardwareInfo:
    """
    Detect CPU capabilities.
    
    Returns:
        HardwareInfo for CPU
    """
    try:
        # Get CPU info
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_name = result.stdout.strip() if result.returncode == 0 else "Unknown CPU"
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
                    else:
                        cpu_name = "Unknown CPU"
            except Exception:
                logger.warning("Could not read /proc/cpuinfo, using fallback CPU name")
                cpu_name = "Unknown CPU"
        else:
            cpu_name = platform.processor() or "Unknown CPU"
        
        # Get available memory
        try:
            import os
            if hasattr(os, 'sysconf'):
                system_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                memory_mb = system_memory // (1024 * 1024)
            else:
                memory_mb = 8192  # Default
        except Exception:
            logger.warning("Could not determine system memory, defaulting to 8GB")
            memory_mb = 8192
        
        # Check available ONNX providers
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
        except ImportError:
            available_providers = ["CPUExecutionProvider"]
        
        logger.info(f"CPU mode: {cpu_name}, {memory_mb}MB RAM")
        
        return HardwareInfo(
            accelerator=AcceleratorType.CPU,
            device_name=cpu_name,
            memory_mb=memory_mb,
            supports_fp16=False,  # CPU FP16 often slower than FP32
            supports_int8=True,
            recommended_batch_size=4,  # Conservative for CPU
            available_providers=["CPUExecutionProvider"]
        )
        
    except Exception as e:
        logger.warning(f"Error detecting CPU: {e}")
        return HardwareInfo(
            accelerator=AcceleratorType.CPU,
            device_name="Unknown CPU",
            memory_mb=8192,
            recommended_batch_size=4,
            available_providers=["CPUExecutionProvider"]
        )


def _get_recommended_batch_size(memory_mb: int) -> int:
    """
    Get recommended batch size based on available memory.
    
    Args:
        memory_mb: Available memory in MB
        
    Returns:
        Recommended batch size
    """
    if memory_mb >= 16000:
        return 32
    elif memory_mb >= 8000:
        return 16
    elif memory_mb >= 4000:
        return 8
    elif memory_mb >= 2000:
        return 4
    else:
        return 2


def detect_hardware() -> HardwareInfo:
    """
    Detect best available hardware accelerator.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        HardwareInfo for best available accelerator
    """
    logger.info("Detecting hardware...")
    
    # Try CUDA first
    cuda_info = detect_cuda()
    if cuda_info:
        logger.info(f"Using CUDA acceleration: {cuda_info.device_name}")
        return cuda_info
    
    # Try MPS (Apple Silicon)
    mps_info = detect_mps()
    if mps_info:
        logger.info(f"Using MPS acceleration: {mps_info.device_name}")
        return mps_info
    
    # Fallback to CPU
    cpu_info = detect_cpu()
    logger.info(f"Using CPU mode: {cpu_info.device_name}")
    return cpu_info


def get_onnx_providers(hardware: Optional[HardwareInfo] = None) -> List[str]:
    """
    Get ONNX Runtime execution providers based on hardware.
    
    Args:
        hardware: Hardware info (will detect if not provided)
        
    Returns:
        List of provider names in priority order
    """
    if hardware is None:
        hardware = detect_hardware()
    
    return hardware.available_providers


def get_recommended_settings(hardware: Optional[HardwareInfo] = None) -> Dict:
    """
    Get recommended settings based on hardware.
    
    Args:
        hardware: Hardware info (will detect if not provided)
        
    Returns:
        Dict with recommended settings
    """
    if hardware is None:
        hardware = detect_hardware()
    
    return {
        "accelerator": hardware.accelerator.value,
        "device_name": hardware.device_name,
        "memory_mb": hardware.memory_mb,
        "batch_size": hardware.recommended_batch_size,
        "supports_fp16": hardware.supports_fp16,
        "supports_int8": hardware.supports_int8,
        "providers": hardware.available_providers,
        "use_gpu": hardware.accelerator != AcceleratorType.CPU,
        "vram_budget_mb": int(hardware.memory_mb * 0.8),  # 80% of available
    }


# Cached hardware detection
_hardware_cache: Optional[HardwareInfo] = None


def get_hardware_info() -> HardwareInfo:
    """
    Get cached hardware info, detecting if necessary.
    
    Returns:
        HardwareInfo for current system
    """
    global _hardware_cache
    if _hardware_cache is None:
        _hardware_cache = detect_hardware()
    return _hardware_cache


def clear_hardware_cache() -> None:
    """Clear cached hardware info."""
    global _hardware_cache
    _hardware_cache = None
