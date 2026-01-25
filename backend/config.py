"""
Argus Core - Configuration Loader
=================================
Centralized configuration management using pydantic-settings.

Implements: PRIME_ARGUS_DOCUMENT.md - Appendix B: Configuration Reference
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, List
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Following 12-factor app principles, all configuration
    comes from environment variables with sensible defaults.
    """
    
    # ============== DATABASE ==============
    mongo_url: str = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name: str = os.environ.get("DB_NAME", "argus_core")
    
    # ============== STORAGE ==============
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
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
    use_gpu: bool = True
    gpu_memory_limit_mb: int = 3500  # RTX 3050 has 4GB, leave headroom
    enable_tensorrt: bool = True
    fallback_to_cpu: bool = True
    
    # ============== PROCESSING ==============
    max_video_duration_seconds: int = 300  # 5 minutes max
    max_file_size_mb: int = 500
    frame_sample_rate_short: int = 5   # Every 5th frame for <30s
    frame_sample_rate_medium: int = 10  # Every 10th frame for 30-120s
    frame_sample_rate_long: int = 15    # Every 15th frame for >120s
    
    # ============== SCORING ==============
    score_weight_video_spatial: float = 0.30
    score_weight_video_temporal: float = 0.25
    score_weight_audio: float = 0.20
    score_weight_metadata: float = 0.15
    score_weight_text: float = 0.10
    
    verdict_threshold_authentic: int = 80
    verdict_threshold_likely_authentic: int = 60
    verdict_threshold_uncertain: int = 40
    verdict_threshold_likely_fake: int = 20
    
    # ============== SECURITY ==============
    jwt_secret: str = "argus-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_rate_limit_per_minute: int = 100
    
    # ============== CORS ==============
    cors_origins: str = "*"
    
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
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def get_frame_sample_rate(self, duration_seconds: float) -> int:
        """
        Get appropriate frame sample rate based on video duration.
        
        Args:
            duration_seconds: Video duration
            
        Returns:
            Frame sample rate (every Nth frame)
        """
        if duration_seconds <= 30:
            return self.frame_sample_rate_short
        elif duration_seconds <= 120:
            return self.frame_sample_rate_medium
        else:
            return self.frame_sample_rate_long
    
    def get_verdict(self, trust_score: float) -> str:
        """
        Determine verdict based on trust score.
        
        Args:
            trust_score: Score from 0-100
            
        Returns:
            Verdict string
        """
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
    """
    Get cached settings instance.
    
    Uses LRU cache to avoid re-parsing environment on every access.
    
    Returns:
        Settings instance
    """
    return Settings()


# Global settings instance for convenience
config = get_settings()
