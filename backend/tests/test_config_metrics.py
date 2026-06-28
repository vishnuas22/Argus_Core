"""
Argus Core - Configuration & Metrics Tests
============================================
Tests for config.py (Settings) and utils/metrics.py (Prometheus metrics).

Config tests: Settings loading, defaults, verdict logic, frame sample rates.
Metrics tests: All metric recording functions execute without errors.

No mocks. Real Pydantic settings and real Prometheus metric objects.
"""

import os
import sys
from typing import Dict, Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from config import Settings, get_settings, config
from utils.metrics import (
    record_analysis_request, record_analysis_duration,
    record_inference_duration, record_http_request,
    update_vram_usage, set_active_analyses, set_queue_size,
    init_app_info, record_model_load, record_model_inference,
    record_model_unload,
    http_requests_total, analysis_requests_total,
)


# ============== SETTINGS TESTS ==============

class TestSettingsDefaults:
    """Test Settings default values."""

    def test_database_defaults(self) -> None:
        s = Settings()
        assert "mongodb" in s.mongo_url
        assert len(s.db_name) > 0

    def test_storage_defaults(self) -> None:
        s = Settings()
        assert s.minio_endpoint == "localhost:9000"
        assert s.minio_bucket_uploads == "argus-uploads"
        assert s.minio_bucket_preprocessed == "argus-preprocessed"
        assert s.minio_bucket_results == "argus-results"

    def test_redis_defaults(self) -> None:
        s = Settings()
        assert "redis" in s.redis_url

    def test_ml_defaults(self) -> None:
        s = Settings()
        assert s.model_cache_dir == "/models"
        assert s.gpu_memory_limit_mb == 3500

    def test_processing_defaults(self) -> None:
        s = Settings()
        assert s.max_video_duration_seconds == 300
        assert s.max_file_size_mb == 500
        assert s.frame_sample_rate_short == 5
        assert s.frame_sample_rate_medium == 10
        assert s.frame_sample_rate_long == 15

    def test_scoring_defaults(self) -> None:
        s = Settings()
        total_weight = (
            s.score_weight_video_spatial +
            s.score_weight_video_temporal +
            s.score_weight_audio +
            s.score_weight_metadata +
            s.score_weight_text
        )
        assert abs(total_weight - 1.0) < 0.01, "Score weights should sum to ~1.0"

    def test_verdict_thresholds_ascending(self) -> None:
        s = Settings()
        assert s.verdict_threshold_authentic > s.verdict_threshold_likely_authentic
        assert s.verdict_threshold_likely_authentic > s.verdict_threshold_uncertain
        assert s.verdict_threshold_uncertain > s.verdict_threshold_likely_fake
        assert s.verdict_threshold_likely_fake > 0

    def test_security_defaults(self) -> None:
        s = Settings()
        assert s.jwt_algorithm == "HS256"
        assert s.jwt_expire_minutes > 0
        assert s.api_rate_limit_per_minute > 0

    def test_api_defaults(self) -> None:
        s = Settings()
        assert s.api_version == "v1"
        assert len(s.api_title) > 0
        assert len(s.api_description) > 0


class TestSettingsMethods:
    """Test Settings computed properties and methods."""

    def test_cors_origins_wildcard(self) -> None:
        s = Settings(cors_origins="*")
        assert s.cors_origins_list == ["*"]

    def test_cors_origins_multiple(self) -> None:
        s = Settings(cors_origins="http://localhost:3000, https://example.com")
        origins = s.cors_origins_list
        assert len(origins) == 2
        assert "http://localhost:3000" in origins
        assert "https://example.com" in origins

    def test_frame_sample_rate_short(self) -> None:
        s = Settings()
        assert s.get_frame_sample_rate(10) == s.frame_sample_rate_short
        assert s.get_frame_sample_rate(30) == s.frame_sample_rate_short

    def test_frame_sample_rate_medium(self) -> None:
        s = Settings()
        assert s.get_frame_sample_rate(31) == s.frame_sample_rate_medium
        assert s.get_frame_sample_rate(120) == s.frame_sample_rate_medium

    def test_frame_sample_rate_long(self) -> None:
        s = Settings()
        assert s.get_frame_sample_rate(121) == s.frame_sample_rate_long
        assert s.get_frame_sample_rate(300) == s.frame_sample_rate_long

    def test_verdict_authentic(self) -> None:
        s = Settings()
        assert s.get_verdict(100) == "authentic"
        assert s.get_verdict(80) == "authentic"

    def test_verdict_likely_authentic(self) -> None:
        s = Settings()
        assert s.get_verdict(79) == "likely_authentic"
        assert s.get_verdict(60) == "likely_authentic"

    def test_verdict_uncertain(self) -> None:
        s = Settings()
        assert s.get_verdict(59) == "uncertain"
        assert s.get_verdict(40) == "uncertain"

    def test_verdict_likely_fake(self) -> None:
        s = Settings()
        assert s.get_verdict(39) == "likely_fake"
        assert s.get_verdict(20) == "likely_fake"

    def test_verdict_fake(self) -> None:
        s = Settings()
        assert s.get_verdict(19) == "fake"
        assert s.get_verdict(0) == "fake"

    def test_verdict_boundary_transitions(self) -> None:
        s = Settings()
        # Just above authentic threshold
        assert s.get_verdict(80.01) == "authentic"
        # Just below authentic threshold
        assert s.get_verdict(79.99) == "likely_authentic"

    def test_settings_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_global_config_instance(self) -> None:
        assert config is not None
        assert isinstance(config, Settings)


class TestSettingsFromEnv:
    """Test settings loaded from environment variables."""

    def test_custom_db_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_NAME", "custom_test_db")
        s = Settings()
        assert s.db_name == "custom_test_db"

    def test_custom_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"


# ============== METRICS TESTS ==============

class TestMetricsRecording:
    """Test all metric recording functions execute correctly."""

    def test_record_analysis_request(self) -> None:
        record_analysis_request("completed", "video")
        record_analysis_request("failed", "image")

    def test_record_analysis_duration(self) -> None:
        record_analysis_duration("video", 12.5)
        record_analysis_duration("audio", 3.2)

    def test_record_inference_duration(self) -> None:
        record_inference_duration("efficientnet", 0.15)
        record_inference_duration("clip", 0.08)

    def test_record_http_request(self) -> None:
        record_http_request("POST", "/api/v1/analyze", 202, 0.5)
        record_http_request("GET", "/api/v1/health", 200, 0.01)
        record_http_request("GET", "/api/v1/analyze/missing", 404, 0.02)

    def test_update_vram_usage(self) -> None:
        update_vram_usage("efficientnet", 1024000000)

    def test_set_active_analyses(self) -> None:
        set_active_analyses(0)
        set_active_analyses(5)
        set_active_analyses(0)

    def test_set_queue_size(self) -> None:
        set_queue_size("analysis", 10)
        set_queue_size("preprocessing", 3)

    def test_init_app_info(self) -> None:
        init_app_info("v1.0.0", "test")

    def test_record_model_load(self) -> None:
        record_model_load("test_model", True)
        record_model_load("test_model", False)

    def test_record_model_inference(self) -> None:
        record_model_inference("test_model", True, 0.5, confidence=0.85)
        record_model_inference("test_model", False, 0.1)

    def test_record_model_unload(self) -> None:
        record_model_unload("test_model")

    def test_metrics_not_none(self) -> None:
        """Verify all metric objects are properly initialized."""
        from utils.metrics import (
            model_load_total, model_inference_total,
            model_confidence_histogram, model_latency_seconds,
            analysis_duration_seconds, inference_duration_seconds,
            http_request_duration_seconds,
            model_vram_usage_bytes, active_analyses,
            model_cache_size, queue_size, model_loaded,
        )
        assert model_load_total is not None
        assert model_inference_total is not None
        assert active_analyses is not None
        assert queue_size is not None
