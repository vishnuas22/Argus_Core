"""
Argus Core - Configuration Loader
=================================
Centralized configuration management using pydantic-settings.

Iteration 1 changes (see CHANGELOG.md):
  * JWT secret is read from BOTH `JWT_SECRET` (compose convention) and
    `SECRET_KEY` (legacy .env) — fixes the naming mismatch flagged in
    ENGINEERING_REVIEW.md.
  * CORS origins never default to "*" in production. The default is
    the Next.js dev origin; deployments must set CORS_ORIGINS explicitly.
  * Added GPU profile presets (rtx3050, t4, a10, a100) so the same
    image runs on a 4GB laptop or a 24GB cloud GPU without code changes.
  * Added `model_manifest_path` and `verify_model_checksums` flags —
    the model downloader now refuses to load a model whose sha256 does
    not match the manifest unless `verify_model_checksums=false`.
  * Added `enable_sota_detectors` flag — when true, the SOTA detector
    adapters added in Iteration 1 (CLIP+LoRA, DINOv2, AASIST3,
    Wav2Vec2-XLS-R, VideoMAE, AltFree) are loaded into the per-modality
    ensembles. Default true; can be disabled for backward-only mode.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------
# GPU profile presets
# ---------------------------------------------------------------------
# Each preset declares (vram_mb, batch_size_hint, use_fp16, use_tensorrt).
# Pick one via env: GPU_PROFILE=t4|a10|a100|rtx3050|cpu
GPU_PROFILES = {
    "cpu":     {"vram_mb": 0,     "batch_size": 1,  "fp16": False, "tensorrt": False},
    "rtx3050": {"vram_mb": 3500,  "batch_size": 2,  "fp16": True,  "tensorrt": True},
    "t4":      {"vram_mb": 14000, "batch_size": 8,  "fp16": True,  "tensorrt": True},
    "a10":     {"vram_mb": 22000, "batch_size": 16, "fp16": True,  "tensorrt": True},
    "a100":    {"vram_mb": 40000, "batch_size": 32, "fp16": True,  "tensorrt": True},
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables (12-factor)."""

    # ============== DATABASE ==============
    mongo_url: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name: str = os.environ.get("DB_NAME", "argus_core")

    # ============== STORAGE ==============
    minio_endpoint: str = "localhost:9000"
    # C2 fix: no insecure defaults — must be set via env vars
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
    model_cache_dir: str = "/models"
    use_gpu: bool = True  # Will be overridden by hardware detection
    gpu_memory_limit_mb: int = 3500  # RTX 3050 has 4GB, leave headroom
    enable_tensorrt: bool = True
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

    # New: GPU profile preset (cpu|rtx3050|t4|a10|a100)
    gpu_profile: str = os.environ.get("GPU_PROFILE", "rtx3050")

    # ============== MODEL DOWNLOAD (DETERMINISTIC) ==============
    auto_download_models: bool = True
    download_on_startup: bool = True
    huggingface_token: Optional[str] = None
    # New: path to the YAML manifest of pinned models with sha256
    model_manifest_path: str = os.environ.get(
        "MODEL_MANIFEST_PATH", "/app/models/manifest.yaml"
    )
    # New: refuse to load a model whose sha256 doesn't match the manifest
    verify_model_checksums: bool = True
    # New: enable Iteration-1 SOTA detector adapters in per-modality ensembles
    enable_sota_detectors: bool = True

    # ============== ITERATION 2: DEFENSES / CALIBRATION / XAI / DRIFT ==============
    # Adversarial defenses (training-free). All default ON except the
    # adversarial gate (which adds K+1 forward passes per input).
    enable_adversarial_defenses: bool = True
    enable_rps: bool = True                # Randomized Preprocessing Sanitizer
    enable_adversarial_gate: bool = False  # XAI adversarial gate (slow)
    enable_rs_lite: bool = False           # Randomized smoothing lite (slow)

    # Calibration
    enable_calibration: bool = True
    temperature_scaler_path: str = "/models/calibration/temperature_scaler.json"
    conformal_raps_path: str = "/models/calibration/conformal_raps.json"
    conformal_alpha: float = 0.10  # 1-alpha = target coverage
    platt_params_path: str = "/models/calibration/platt_params.json"

    # XAI upgrades (Iteration 2)
    enable_attn_lrp: bool = True     # AttnLRP for transformer backbones
    enable_eigen_cam: bool = True    # Eigen-CAM gradient-free cross-check
    enable_audio_band_attribution: bool = True
    enable_temporal_attribution: bool = True

    # Drift detection (Iteration 2)
    enable_drift_detection: bool = True
    drift_reference_path: str = "/models/calibration/drift_reference"
    drift_check_interval: int = 100  # check every N inferences
    drift_psi_moderate: float = 0.10
    drift_psi_major: float = 0.25
    drift_mmd_threshold: float = 0.05

    # ============== ITERATION 4: DIVERSITY + CONTINUOUS LEARNING + XAI ==============
    # New detectors (diversity expansion)
    enable_timesformer: bool = True   # TimeSformer video detector (cc-by-nc-4.0)
    enable_ecapa: bool = True         # ECAPA-TDNN audio detector (needs reference centroid)

    # Continuous learning (online LoRA retraining)
    enable_continuous_learning: bool = True
    feedback_buffer_path: str = "/models/continuous_learning/feedback_buffer.json"
    retrain_schedule_hours: float = 24.0  # retrain every N hours
    retrain_min_samples: int = 50  # minimum labeled samples to trigger retrain
    retrain_max_samples: int = 1000  # cap per retrain cycle
    retrain_ab_test_ratio: float = 0.1  # 10% of traffic uses new adapter

    # XAI wiring (display in analyzer output)
    enable_xai_attribution_output: bool = True  # include XAI in ModalityResult

    # ============== ITERATION 5: WATERMARKING + CERTIFIED ROBUSTNESS + BEAT ==============
    # Model watermarking (IP protection)
    enable_model_watermarking: bool = True
    watermark_key_length: int = 256
    # Certified robustness (BRONet + Randomized Smoothing)
    enable_certified_robustness: bool = False  # OFF by default — expensive
    rs_certification_sigma: float = 0.25
    rs_certification_num_samples: int = 10000  # full Cohen 2019 n
    rs_certification_alpha: float = 0.001      # 99.9% confidence
    # Celery Beat (automatic retraining + drift checks)
    enable_celery_beat: bool = True

    # ============== ITERATION 6: OBSERVABILITY + MULTI-GPU + C2PA v2.3 ==============
    # Prometheus metrics
    enable_prometheus_metrics: bool = True
    # Multi-GPU sharding
    enable_multi_gpu: bool = True
    # C2PA v2.3 full compliance
    enable_c2pa_v2: bool = True
    c2pa_sign_cert: str = ""       # path to X.509 signing cert (PEM)
    c2pa_private_key: str = ""     # path to EC private key (PEM)
    c2pa_tsa_url: str = ""         # RFC 3161 Time-Stamping Authority URL
    c2pa_signing_alg: str = "ES256"  # ES256 | ES384 | ES512 | PS256 | PS384 | PS512 | Ed25519

    # ============== ITERATION 8: EXECUTION MODES + MEMORY GUARD ==============
    # Execution mode: lite | balanced | research
    # If empty, auto-detects based on hardware.
    execution_mode: str = os.environ.get("EXECUTION_MODE", "")
    # Memory guard: trigger automatic fallback when memory is constrained
    enable_memory_guard: bool = True
    memory_guard_limit_mb: int = 0  # 0 = auto-detect from mode

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
    # C1 fix: refuse to boot with insecure default in production.
    # In dev, generate a stable per-environment secret so tokens survive restarts.
    _jwt_secret_raw = os.environ.get("JWT_SECRET") or os.environ.get("SECRET_KEY", "")
    if not _jwt_secret_raw:
        if os.environ.get("ENVIRONMENT", "dev") == "production":
            raise RuntimeError(
                "JWT_SECRET must be set in production. "
                "Set the JWT_SECRET environment variable."
            )
        # Dev-only: derive a stable secret from hostname+environment so all
        # workers in the same process tree share the same key.
        import hashlib
        import socket
        _dev_seed = f"argus-dev-{socket.gethostname()}-{os.environ.get('ENVIRONMENT', 'dev')}"
        _jwt_secret_raw = hashlib.sha256(_dev_seed.encode()).hexdigest()
    jwt_secret: str = _jwt_secret_raw
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_rate_limit_per_minute: int = 100

    # ============== CORS ==============
    # In production, MUST be set explicitly to a comma-separated list of
    # allowed origins. The wildcard is only honored when ENVIRONMENT=dev.
    cors_origins: str = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    environment: str = os.environ.get("ENVIRONMENT", "dev")

    # ============== LOGGING ==============
    log_level: str = "INFO"
    log_format: str = "json"  # "json" for production, "console" for development

    # ============== API ==============
    api_version: str = "v1"
    api_title: str = "Argus Core API"
    api_description: str = "Multi-Modal Deepfake Detection & Forensic Analysis Platform"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    # ------------------------------------------------------------------
    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parse CORS origins into a list.

        - If ENVIRONMENT == "dev", "*" is honored for local dev.
        - In production, "*" is rejected and replaced with the default
          Next.js origin, with a logged warning. This prevents the
          silent CORS-wildcard vulnerability flagged in
          ENGINEERING_REVIEW.md §5 (Risk #4).
        """
        if self.cors_origins == "*":
            if self.environment == "dev":
                return ["*"]
            # Production: refuse wildcard
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
        return GPU_PROFILES.get(self.gpu_profile, GPU_PROFILES["rtx3050"])

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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance for convenience
config = get_settings()
