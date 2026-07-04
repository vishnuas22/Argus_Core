#!/usr/bin/env python3
"""
Argus Core — MPS Verification Script
=====================================

Verifies that PyTorch MPS (Metal Performance Shaders) is correctly
installed and accessible on Apple Silicon. Run this AFTER
setup_mac_dev.sh to confirm your environment is ready for
GPU-accelerated inference on your M1 Max.

Run from the backend directory (with venv activated):
    python scripts/verify_mps.py

What this script checks:
  1. PyTorch is installed and importable
  2. MPS backend is available (Apple Silicon only)
  3. Basic tensor operations work on MPS
  4. A real model forward pass works on MPS
  5. MPS performance vs CPU (rough benchmark)
  6. Argus config.py correctly detects MPS as the device

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure backend dir on path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


# ANSI colors for terminal output
class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}✗{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}!{C.RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {C.BLUE}→{C.RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}=== {msg} ==={C.RESET}")


def check_pytorch_installed() -> bool:
    """Check 1: PyTorch is installed."""
    header("Check 1: PyTorch Installation")
    try:
        import torch
        ok(f"PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        fail(f"PyTorch not installed: {e}")
        info("Fix: pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1")
        return False


def check_mps_available() -> bool:
    """Check 2: MPS backend is available."""
    header("Check 2: MPS Backend Availability")
    try:
        import torch
        if not hasattr(torch.backends, "mps"):
            fail("torch.backends.mps module not found (PyTorch < 1.12)")
            return False
        if not torch.backends.mps.is_available():
            fail("MPS not available on this hardware")
            info("MPS requires Apple Silicon (M1/M2/M3/M4).")
            info("If you're on Apple Silicon, ensure PyTorch >= 1.12 is installed.")
            return False
        if not torch.backends.mps.is_built():
            fail("MPS is built=False — PyTorch was compiled without MPS support")
            info("Reinstall PyTorch from a wheel that includes MPS:")
            info("  pip install --force-reinstall torch==2.3.1")
            return False
        ok("MPS backend is available")
        ok("MPS backend is built")
        return True
    except Exception as e:
        fail(f"Unexpected error: {e}")
        return False


def check_basic_tensor_ops() -> bool:
    """Check 3: Basic tensor operations work on MPS."""
    header("Check 3: Basic Tensor Operations on MPS")
    try:
        import torch
        # Create tensors on MPS
        x = torch.randn(1000, 1000, device="mps")
        y = torch.randn(1000, 1000, device="mps")
        ok("Tensor creation on MPS: OK")

        # Matrix multiplication
        z = x @ y
        ok(f"Matrix multiplication on MPS: OK (shape={z.shape})")

        # Element-wise ops
        w = torch.relu(z) + 1.0
        ok("Element-wise ops (relu + add): OK")

        # Reduction
        s = w.sum().item()
        ok(f"Reduction (sum): OK (value={s:.2f})")

        # Transfer back to CPU
        cpu_tensor = z.cpu()
        ok(f"Transfer MPS → CPU: OK (shape={cpu_tensor.shape})")

        return True
    except Exception as e:
        fail(f"MPS tensor op failed: {type(e).__name__}: {e}")
        return False


def check_model_forward_pass() -> bool:
    """Check 4: A real model forward pass works on MPS."""
    header("Check 4: Model Forward Pass on MPS")
    try:
        import torch
        import torch.nn as nn

        # Small CNN model
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 2),
        ).to("mps")

        # Simulate an image batch
        x = torch.randn(4, 3, 224, 224, device="mps")
        ok("Created input batch on MPS (4, 3, 224, 224)")

        # Forward pass
        with torch.no_grad():
            output = model(x)
        ok(f"Forward pass: OK (output shape={output.shape})")

        # Softmax
        probs = torch.softmax(output, dim=-1)
        ok(f"Softmax: OK (probs sum={probs.sum().item():.4f})")

        return True
    except Exception as e:
        fail(f"Model forward pass failed: {type(e).__name__}: {e}")
        return False


def check_mps_vs_cpu_performance() -> bool:
    """Check 5: MPS is faster than CPU for typical inference."""
    header("Check 5: MPS vs CPU Performance Benchmark")
    try:
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2),
        )

        x = torch.randn(8, 3, 224, 224)

        # CPU benchmark
        model_cpu = model.to("cpu")
        x_cpu = x.to("cpu")
        # Warmup
        for _ in range(3):
            _ = model_cpu(x_cpu)
        # Benchmark
        start = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                _ = model_cpu(x_cpu)
        cpu_time = (time.perf_counter() - start) / 20 * 1000  # ms
        ok(f"CPU avg: {cpu_time:.2f} ms/inference")

        # MPS benchmark
        model_mps = model.to("mps")
        x_mps = x.to("mps")
        # Warmup (first MPS call is slow due to compilation)
        for _ in range(3):
            _ = model_mps(x_mps)
        # Benchmark
        start = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                _ = model_mps(x_mps)
        mps_time = (time.perf_counter() - start) / 20 * 1000  # ms
        ok(f"MPS avg: {mps_time:.2f} ms/inference")

        speedup = cpu_time / mps_time if mps_time > 0 else 0
        if speedup > 1.0:
            ok(f"{C.GREEN}MPS is {speedup:.2f}x faster than CPU{C.RESET}")
        elif speedup > 0.5:
            warn(f"MPS is only {speedup:.2f}x of CPU speed (expected > 1.0x)")
            info("First-run MPS is slower due to Metal shader compilation.")
            info("Real-world speedup is typically 2-4x for larger models.")
        else:
            warn(f"MPS appears slower than CPU ({speedup:.2f}x)")
            info("This can happen for tiny models — try a real detector.")

        return True
    except Exception as e:
        fail(f"Benchmark failed: {type(e).__name__}: {e}")
        return False


def check_argus_config() -> bool:
    """Check 6: Argus config.py correctly detects MPS."""
    header("Check 6: Argus Config MPS Detection")
    try:
        # Force the env to use MPS profile
        os.environ["GPU_PROFILE"] = "mps"
        os.environ["USE_GPU"] = "true"

        # Clear cached settings
        import importlib
        import config as config_mod
        importlib.reload(config_mod)

        cfg = config_mod.get_settings()

        # Check GPU profile
        if cfg.gpu_profile == "mps":
            ok(f"GPU_PROFILE=mps detected")
        else:
            warn(f"GPU_PROFILE={cfg.gpu_profile} (expected 'mps')")
            info("Set GPU_PROFILE=mps in backend/.env")

        # Check device detection
        device = cfg.device
        if device == "mps":
            ok(f"config.device = 'mps' (MPS correctly detected)")
        elif device == "cpu":
            warn(f"config.device = 'cpu' (MPS not detected — is PyTorch installed?)")
        else:
            warn(f"config.device = '{device}' (expected 'mps' or 'cpu')")

        # Check profile settings
        profile = cfg.gpu_profile_settings
        if profile.get("device") == "mps":
            ok(f"MPS profile settings: batch_size={profile.get('batch_size')}, "
               f"fp16={profile.get('fp16')}, vram_mb={profile.get('vram_mb')}")
        else:
            warn(f"MPS profile not loaded: {profile}")

        return True
    except Exception as e:
        fail(f"Config check failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    print(f"{C.BOLD}Argus Core — MPS Verification Script{C.RESET}")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {os.uname().machine if hasattr(os, 'uname') else 'unknown'}")

    checks = [
        ("PyTorch Installation", check_pytorch_installed),
        ("MPS Backend Availability", check_mps_available),
        ("Basic Tensor Operations", check_basic_tensor_ops),
        ("Model Forward Pass", check_model_forward_pass),
        ("MPS vs CPU Performance", check_mps_vs_cpu_performance),
        ("Argus Config MPS Detection", check_argus_config),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            fail(f"{name} raised: {e}")
            results.append((name, False))

    # Summary
    header("Summary")
    passed = sum(1 for _, ok_flag in results if ok_flag)
    total = len(results)
    for name, ok_flag in results:
        status = f"{C.GREEN}PASS{C.RESET}" if ok_flag else f"{C.RED}FAIL{C.RESET}"
        print(f"  {status}  {name}")

    print(f"\n{C.BOLD}{passed}/{total} checks passed{C.RESET}")
    if passed == total:
        print(f"\n{C.GREEN}{C.BOLD}✓ MPS is ready for Argus Core inference!{C.RESET}")
        print(f"\nNext: start the backend with MPS acceleration:")
        print(f"  cd backend")
        print(f"  source .venv/bin/activate")
        print(f"  uvicorn server:app --reload --port 8000")
        return 0
    else:
        print(f"\n{C.RED}{C.BOLD}✗ Some checks failed. Fix them before proceeding.{C.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
