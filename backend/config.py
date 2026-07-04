"""
Argus Core - Configuration Loader
=================================
Centralized configuration management using pydantic-settings.

PORTABILITY (2026-07-03):
  * All file paths (model cache, calibration, drift reference, feedback
    buffer) are now auto-detected per-platform — no hardcoded /models.
  * GPU profile auto-detects: MPS on Apple Silicon, CUDA on NVIDIA,
    CPU otherwise. Override via GPU_PROFILE env var only if needed.
  * TensorRT defaults to False (NVIDIA-only; was True which broke MPS).
  * TimeSformer defaults to False (CC-BY-NC-4.0 non-commercial license).

Platform model cache paths (auto-created, no .env editing required):
  macOS:   ~/Library/Application Support/Argus/models
  Linux:   ~/.local/share/Argus/models (or $XDG_DATA_HOME)
  Windows: %LOCALAPPDATA%/Argus/models
  Fallback: ./backend/models/ (if home dir not writable)

Override any path with the corresponding env var (e.g., MODEL_CACHE_DIR)
only if you need a custom location.
"""

import hashlib
import os
import socket
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------
# Platform-aware path helpers
# ---------------------------------------------------------------------

def _default_model_cache_dir() -> str:
    """
    Return the platform-appropriate model cache directory.

    Follows OS conventions:
      macOS:   ~/Library/Application Support/Argus/models
      Linux:   ~/.local/share/Argus/models (XDG_DATA_HOME)
      Windows: %LOCALAPPDATA%/Argus/models

    Falls back to ./models relative to the backend directory if the home
    directory is not writable (e.g., running as a restricted service user).

    This makes the project portable across macOS / Linux / Windows without
    any .env editing.
    """
    app_name = "Argus"

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / app_name
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / app_name
    else:
        # Linux / Unix — follow XDG spec
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / app_name

    cache_dir = base / "models"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = cache_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return str(cache_dir)
    except (OSError, PermissionError):
        # Fallback: ./models relative to backend directory
        backend_dir = Path(__file__).resolve().parent
        fallback = backend_dir / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


def _default_data_dir(subdir: str) -> str:
    """
    Return a platform-appropriate subdirectory under the model cache dir.

    Used for calibration files, drift references, feedback buffers, etc.
    All derived paths stay inside the same platform-appropriate base as
    the models, so there's ONE location to back up / inspect.
    """
    base = Path(_default_model_cache_dir()) / subdir
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _default_manifest_path() -> str:
    """Return the default manifest path — next to the backend source."""
    backend_dir = Path(__file__).resolve().parent
    return str(backend_dir / "models" / "manifest.yaml")


# ---------------------------------------------------------------------
# Auto-detect GPU profile
# ---------------------------------------------------------------------

def _auto_detect_gpu_profile() -> str:
    """
    Auto-detect the best GPU profile for this hardware.

    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

    Returns the profile name to use as default. Operators can override
    via GPU_PROFILE env var.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "t4"  # Default CUDA profile (14GB VRAM)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------
# GPU profile presets
# ---------------------------------------------------------------------
# Each preset declares (vram_mb, batch_size_hint, use_fp16, use_tensorrt, device).
# Pick one via env: GPU_PROFILE=mps|t4|a10|a100|rtx3050|cpu
#
# Mac Apple Silicon notes:
#   - "mps" profile is for M1/M2/M3/M4 Macs running PyTorch natively (NOT in Docker).
#   - Docker on Mac cannot access MPS — use "cpu" profile inside containers.
#   - M1 Max has 24-32 GPU cores + 16-core Neural Engine; unified memory means
#     the "vram_mb" budget is whatever RAM you can spare (we set 16GB).
#   - TensorRT is NVIDIA-only; set to False for MPS.
#   - FP16 on MPS is supported in PyTorch 2.3+ but can be unstable for some ops;
#     we leave it enabled but the engine falls back to FP32 on unsupported ops.
GPU_PROFILES = {
    "cpu":     {"vram_mb": 0,     "batch_size": 1,  "fp16": False, "tensorrt": False, "device": "cpu"},
    "mps":     {"vram_mb": 16384, "batch_size": 4,  "fp16": True,  "tensorrt": False, "device": "mps"},  # M1/M2/M3/M4 Mac
    "rtx3050": {"vram_mb": 3500,  "batch_size": 2,  "fp16": True,  "tensorrt": True,  "device": "cuda"},
    "t4":      {"vram_mb": 14000, "batch_size": 8,  "fp16": True,  "tensorrt": True,  "device": "cuda"},
    "a10":     {"vram_mb": 22000, "batch_size": 16, "fp16": True,  "tensorrt": True,  "device": "cuda"},
    "a100":    {"vram_mb": 40000, "batch_size": 32, "fp16": True,  "tensorrt": True,  "device": "cuda"},
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables (12-factor)."""

    # ============== DATABASE ==============
    mongo_url: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name: str = os.environ.get("DB_NAME", "argus_core")

    # ============== STORAGE ==============
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = os.environ.get("MINIO_ACCESS_KEY", "")
    minio_secret_key: str = os.environ.get("MINIO_SECRET_KEY", "")
    minio_secure: bool = False
    minio_bucket_uploads: str = "argus-uploads"
    minio_bucket_preprocessed: str = "argus-preprocessed"
    minio_bucket_results: str = "argus-results"

    # ============== REDIS ==============
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ============== ML CONFIGURATION ==============
    # Platform-aware model cache — auto-detected, override via MODEL_CACHE_DIR env.
    #   macOS:   ~/Library/Application Support/Argus/models
    #   Linux:   ~/.local/share/Argus/models
    #   Windows: %LOCALAPPDATA%/Argus/models
    model_cache_dir: str = os.environ.get("MODEL_CACHE_DIR", _default_model_cache_dir())
    use_gpu: bool = True  # Will be overridden by hardware detection
    # GPU memory limit — auto-set from GPU_PROFILE if 0, otherwise use this value.
    # Default 0 means "use the profile's vram_mb".
    gpu_memory_limit_mb: int = 0
    # TensorRT is NVIDIA-only. Default False so MPS/CPU don't break.
    enable_tensorrt: bool = False
    fallback_to_cpu: bool = True

    @property
    def device(self) -> str:
        """Detect best available torch device: cuda, mps, or cpu."""
        if not self.use_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # GPU profile — auto-detects MPS/CUDA/CPU, override via GPU_PROFILE env.
    gpu_profile: str = os.environ.get("GPU_PROFILE", _auto_detect_gpu_profile())

    # ============== MODEL DOWNLOAD (DETERMINISTIC) ==============
    auto_download_models: bool = True
    # LAZY LOADING (2026-07-02 curation): when False (the new default),
    # the server does NOT download or warmup models at startup. Models
    # are downloaded and loaded on first inference call via
    # ModelManager.get_model(). This drops startup time from 30-60s
    # (download + load 2 models) to 2-3s (just import modules).
    download_on_startup: bool = False
    # When True, pre-loads the "likely needed" models (retinaface +
    # deepfake_detector_v3) into memory AFTER the server is accepting
    # requests, so the first image analysis is fast. Does NOT block
    # startup. Default True — this is pure upside with no downside.
    warmup_on_startup: bool = True
    huggingface_token: Optional[str] = None
    # Manifest path — defaults to backend/models/manifest.yaml (portable).
    model_manifest_path: str = os.environ.get(
        "MODEL_MANIFEST_PATH", _default_manifest_path()
    )
    verify_model_checksums: bool = True
    enable_sota_detectors: bool = True

    # ============== ITERATION 2: DEFENSES / CALIBRATION / XAI / DRIFT ==============
    enable_adversarial_defenses: bool = True
    enable_rps: bool = True
    enable_adversarial_gate: bool = False
    enable_rs_lite: bool = False

    # Calibration paths — derived from model_cache_dir for portability.
    enable_calibration: bool = True
    temperature_scaler_path: str = os.environ.get("TEMPERATURE_SCALER_PATH", "") or ""
    conformal_raps_path: str = os.environ.get("CONFORMAL_RAPS_PATH", "") or ""
    conformal_alpha: float = 0.10
    platt_params_path: str = os.environ.get("PLATT_PARAMS_PATH", "") or ""

    # XAI upgrades (Iteration 2)
    enable_attn_lrp: bool = True
    enable_eigen_cam: bool = True
    enable_audio_band_attribution: bool = True
    enable_temporal_attribution: bool = True

    # Drift detection (Iteration 2) — path derived from model_cache_dir.
    enable_drift_detection: bool = True
    drift_reference_path: str = os.environ.get("DRIFT_REFERENCE_PATH", "") or ""
    drift_check_interval: int = 100
    drift_psi_moderate: float = 0.10
    drift_psi_major: float = 0.25
    drift_mmd_threshold: float = 0.05

    # ============== ITERATION 4: DIVERSITY + CONTINUOUS LEARNING + XAI ==============
    # TimeSformer is CC-BY-NC-4.0 (non-commercial). Default OFF for
    # commercial safety. Set ENABLE_TIMESFORMER=true for research use.
    enable_timesformer: bool = os.environ.get("ENABLE_TIMESFORMER", "false").lower() in ("true", "1", "yes")
    enable_ecapa: bool = True

    # Continuous learning — path derived from model_cache_dir.
    enable_continuous_learning: bool = True
    feedback_buffer_path: str = os.environ.get("FEEDBACK_BUFFER_PATH", "") or ""
    retrain_schedule_hours: float = 24.0
    retrain_min_samples: int = 50
    retrain_max_samples: int = 1000
    retrain_ab_test_ratio: float = 0.1

    enable_xai_attribution_output: bool = True

    # ============== ITERATION 5: WATERMARKING + CERTIFIED ROBUSTNESS + BEAT ==============
    enable_model_watermarking: bool = True
    watermark_key_length: int = 256
    enable_certified_robustness: bool = False
    rs_certification_sigma: float = 0.25
    rs_certification_num_samples: int = 10000
    rs_certification_alpha: float = 0.001
    enable_celery_beat: bool = True

    # ============== ITERATION 6: OBSERVABILITY + MULTI-GPU + C2PA v2.3 ==============
    enable_prometheus_metrics: bool = True
    enable_multi_gpu: bool = True
    enable_c2pa_v2: bool = True
    c2pa_sign_cert: str = ""
    c2pa_private_key: str = ""
    c2pa_tsa_url: str = ""
    c2pa_signing_alg: str = "ES256"

    # ============== ITERATION 8: EXECUTION MODES + MEMORY GUARD ==============
    execution_mode: str = os.environ.get("EXECUTION_MODE", "")
    enable_memory_guard: bool = True
    memory_guard_limit_mb: int = 0

    # ============== PROCESSING ==============
    max_video_duration_seconds: int = 300
    max_file_size_mb: int = 500
    frame_sample_rate_short: int = 5
    frame_sample_rate_medium: int = 10
    frame_sample_rate_long: int = 15

    # ============== SCORING ==============
    score_weight_video_spatial: float = 0.30
    score_weight_video_temporal: float = 0.25
    score_weight_audio: float = 0.20
    score_weight_metadata: float = 0.15
    verdict_threshold_authentic: int = 80
    verdict_threshold_likely_authentic: int = 60
    verdict_threshold_uncertain: int = 40
    verdict_threshold_likely_fake: int = 20

    # ============== SECURITY ==============
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_rate_limit_per_minute: int = 100

    # ============== CORS ==============
    cors_origins: str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    environment: str = os.environ.get("ENVIRONMENT", "dev")

    # ============== LOGGING ==============
    log_level: str = "INFO"
    log_format: str = "json"

    # ============== API ==============
    api_version: str = "v1"
    api_title: str = "Argus Core API"
    api_description: str = "Multi-Modal Deepfake Detection & Forensic Analysis Platform"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list. Wildcard rejected in production."""
        if self.cors_origins == "*":
            if self.environment == "dev":
                return ["*"]
            import warnings
            warnings.warn(
                "CORS_ORIGINS='*' is forbidden in production (ENVIRONMENT="
                f"{self.environment}). Falling back to http://localhost:3000. "
                "Set CORS_ORIGINS to a comma-separated list of allowed origins.",
                RuntimeWarning,
            )
            return ["http://localhost:3000"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ------------------------------------------------------------------
    @property
    def gpu_profile_settings(self) -> dict:
        """Return the active GPU profile settings."""
        return GPU_PROFILES.get(self.gpu_profile, GPU_PROFILES["cpu"])

    # ------------------------------------------------------------------
    @property
    def effective_gpu_memory_limit_mb(self) -> int:
        """Return the effective VRAM limit — from gpu_memory_limit_mb or profile."""
        if self.gpu_memory_limit_mb > 0:
            return self.gpu_memory_limit_mb
        return self.gpu_profile_settings.get("vram_mb", 0)

    # ------------------------------------------------------------------
    @property
    def calibration_dir(self) -> str:
        """Calibration files directory — derived from model_cache_dir."""
        return _default_data_dir("calibration")

    @property
    def continuous_learning_dir(self) -> str:
        """Continuous learning directory — derived from model_cache_dir."""
        return _default_data_dir("continuous_learning")

    @property
    def drift_reference_dir(self) -> str:
        """Drift reference directory — derived from model_cache_dir."""
        return _default_data_dir("calibration/drift_reference")

    # ------------------------------------------------------------------
    def get_frame_sample_rate(self, duration_seconds: float) -> int:
        """Get appropriate frame sample rate based on video duration."""
        if duration_seconds <= 30:
            return self.frame_sample_rate_short
        elif duration_seconds <= 120:
            return self.frame_sample_rate_medium
        else:
            return self.frame_sample_rate_long

    def get_verdict(self, trust_score: float) -> str:
        """Determine verdict based on trust score (0-100)."""
        if trust_score >= self.verdict_threshold_authentic:
            return "authentic"
        elif trust_score >= self.verdict_threshold_likely_authentic:
            return "likely_authentic"
        elif trust_score >= self.verdict_threshold_uncertain:
            return "uncertain"
        elif trust_score >= self.verdict_threshold_likely_fake:
            return "likely_fake"
        else:
            return "fake"


def _default_jwt_secret() -> str:
    """
    Compute the default JWT secret when none is provided via env.

    Behavior:
      * If JWT_SECRET or SECRET_KEY is set, use it verbatim.
      * In production (ENVIRONMENT=production), refuse to boot with an
        empty secret — operators must set JWT_SECRET explicitly.
      * In dev, derive a stable per-host secret so tokens survive restarts.
    """
    secret = os.environ.get("JWT_SECRET") or os.environ.get("SECRET_KEY", "")
    if secret:
        return secret
    environment = os.environ.get("ENVIRONMENT", "dev")
    if environment == "production":
        raise RuntimeError(
            "JWT_SECRET must be set in production. "
            "Set the JWT_SECRET environment variable."
        )
    dev_seed = f"argus-dev-{socket.gethostname()}-{environment}"
    return hashlib.sha256(dev_seed.encode()).hexdigest()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Backfill the JWT secret if the field was left empty.
    if not settings.jwt_secret:
        settings.jwt_secret = _default_jwt_secret()

    # Backfill derived paths if not explicitly set. This keeps all
    # calibration / drift / feedback files inside the platform-aware
    # model_cache_dir so there's ONE location to back up.
    if not settings.temperature_scaler_path:
        settings.temperature_scaler_path = str(Path(settings.calibration_dir) / "temperature_scaler.json")
    if not settings.conformal_raps_path:
        settings.conformal_raps_path = str(Path(settings.calibration_dir) / "conformal_raps.json")
    if not settings.platt_params_path:
        settings.platt_params_path = str(Path(settings.calibration_dir) / "platt_params.json")
    if not settings.drift_reference_path:
        settings.drift_reference_path = settings.drift_reference_dir
    if not settings.feedback_buffer_path:
        settings.feedback_buffer_path = str(Path(settings.continuous_learning_dir) / "feedback_buffer.json")

    return settings


# Global settings instance for convenience.
config = get_settings()