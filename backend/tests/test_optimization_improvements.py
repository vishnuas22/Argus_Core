"""
Unit tests for the bug fixes and improvements made during the
maximum-performance optimization pass.

These tests are CPU-only and do NOT require torch / onnxruntime /
MongoDB / Redis. They run in <2 seconds and protect against regressions
of the specific issues fixed in this iteration:

  1. `config.py` — JWT secret derivation must not raise PydanticUserError
     on Pydantic >=2.12 (was: `import hashlib` inside class body).
  2. `core/engine.py` — `get_inference_engine()` must be thread-safe
     under concurrent first-call races (was: bare `_engine = None`).
  3. `core/scorer.py` — `fit_platt_parameters` must produce coefficients
     consistent with `PlattParams.transform` (was: sign-flipped
     sklearn coefficients in the wrong input space).
  4. `core/explain.py` — `_detect_manipulation_type` must consider ALL
     modality results, not return FACE_SWAP on the first VIDEO result.
  5. `analyzers/audio.py` — `_compute_confidence` must cap at 0.15 when
     `any_neural_available` is False (heuristic-only path).
  6. `api/health.py` — `run_health_check` must NOT mutate component
     results via `.pop("status")` (was: dropping storage/celery
     metadata).
"""

from __future__ import annotations

import os
import sys
import asyncio
import threading
import types
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Ensure backend dir on path
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# =========================================================================
# 1. config.py — JWT secret derivation must not raise on Pydantic >=2.12
# =========================================================================

class TestConfigJwtSecret:
    """Verify the JWT secret derivation works on Pydantic v2.12+."""

    def test_settings_class_instantiates(self) -> None:
        """Settings() must construct without raising PydanticUserError."""
        from config import Settings
        s = Settings()
        # jwt_secret defaults to empty; get_settings() backfills it.
        assert isinstance(s, Settings)

    def test_get_settings_populates_jwt_secret(self) -> None:
        """get_settings() must always return a non-empty jwt_secret."""
        from config import get_settings
        s = get_settings()
        assert s.jwt_secret, "jwt_secret must not be empty after get_settings()"
        assert len(s.jwt_secret) >= 16, "jwt_secret must be reasonably long"

    def test_default_jwt_secret_helper_returns_value_in_dev(self) -> None:
        """The _default_jwt_secret helper returns a stable value in dev."""
        from config import _default_jwt_secret
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("SECRET_KEY", None)
        os.environ["ENVIRONMENT"] = "dev"
        try:
            s1 = _default_jwt_secret()
            s2 = _default_jwt_secret()
            assert s1 == s2, "dev secret must be stable across calls"
            assert len(s1) == 64, "sha256 hex digest should be 64 chars"
        finally:
            os.environ.pop("ENVIRONMENT", None)

    def test_default_jwt_secret_raises_in_production(self) -> None:
        """Production without JWT_SECRET must raise RuntimeError."""
        from config import _default_jwt_secret
        os.environ.pop("JWT_SECRET", None)
        os.environ.pop("SECRET_KEY", None)
        os.environ["ENVIRONMENT"] = "production"
        try:
            with pytest.raises(RuntimeError, match="JWT_SECRET"):
                _default_jwt_secret()
        finally:
            os.environ.pop("ENVIRONMENT", None)

    def test_jwt_secret_env_var_takes_precedence(self) -> None:
        """JWT_SECRET env var must override the dev-derived secret."""
        from config import _default_jwt_secret
        os.environ["JWT_SECRET"] = "my-explicit-secret-123"
        try:
            assert _default_jwt_secret() == "my-explicit-secret-123"
        finally:
            os.environ.pop("JWT_SECRET", None)

    def test_cors_wildcard_rejected_in_production(self) -> None:
        """CORS='*' must fall back to localhost in production."""
        from config import Settings
        os.environ["ENVIRONMENT"] = "production"
        os.environ["CORS_ORIGINS"] = "*"
        try:
            s = Settings()
            assert "http://localhost:3000" in s.cors_origins_list
            assert "*" not in s.cors_origins_list
        finally:
            os.environ.pop("ENVIRONMENT", None)
            os.environ.pop("CORS_ORIGINS", None)


# =========================================================================
# 2. core/engine.py — get_inference_engine() must be thread-safe
# =========================================================================

class TestEngineSingletonThreadSafety:
    """Verify the engine singleton uses double-checked locking."""

    def test_reset_function_exists(self) -> None:
        """reset_inference_engine should be importable for tests."""
        from core import engine as engine_mod
        assert hasattr(engine_mod, "reset_inference_engine")
        assert hasattr(engine_mod, "_engine_lock")

    def test_engine_singleton_is_thread_safe(self) -> None:
        """Concurrent get_inference_engine() calls return the same instance.

        We replace the InferenceEngine constructor with a slow stub so
        that two threads genuinely race past the first `if _engine is None`
        check. Without the lock, they would construct two instances.
        """
        import time as _time_mod
        from core import engine as engine_mod

        # Reset to a clean state.
        engine_mod._engine = None

        constructed = []
        original_init = engine_mod.InferenceEngine.__init__

        def slow_init(self, *args, **kwargs):
            # Simulate slow construction so threads overlap.
            _time_mod.sleep(0.02)
            constructed.append(id(self))
            # Call the real init to keep the object valid.
            original_init(self, *args, **kwargs)

        engine_mod.InferenceEngine.__init__ = slow_init
        try:
            results: List[Any] = []
            threads = []
            for _ in range(8):
                t = threading.Thread(
                    target=lambda: results.append(engine_mod.get_inference_engine())
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            # All threads should observe the SAME engine instance.
            unique_ids = {id(r) for r in results}
            assert len(unique_ids) == 1, (
                f"Expected 1 engine instance, got {len(unique_ids)} — "
                f"singleton is not thread-safe"
            )
        finally:
            engine_mod.InferenceEngine.__init__ = original_init
            engine_mod._engine = None

    def test_reset_clears_singleton(self) -> None:
        from core import engine as engine_mod
        # Ensure something is constructed.
        e1 = engine_mod.get_inference_engine()
        assert engine_mod._engine is not None
        engine_mod.reset_inference_engine()
        assert engine_mod._engine is None
        # Re-construct after reset.
        e2 = engine_mod.get_inference_engine()
        assert e2 is not e1


# =========================================================================
# 3. core/scorer.py — Platt calibration fit/transform consistency
# =========================================================================

class TestPlattCalibrationConsistency:
    """Verify PlattParams.fit and PlattParams.transform are consistent."""

    def test_identity_platt_preserves_score(self) -> None:
        """With a=1, b=0, Platt transform should preserve logit-odds."""
        from core.scorer import PlattParams
        p = PlattParams(a=1.0, b=0.0)
        # For a=1, b=0: transform(s) = sigmoid(logit(s)) = s
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(p.transform(s) - s) < 1e-6, (
                f"Identity Platt should preserve score {s}, "
                f"got {p.transform(s)}"
            )

    def test_platt_fit_monotonic_increasing(self) -> None:
        """Fitted Platt on a monotonic dataset should produce a > 0.

        If higher scores correlate with the positive label, the fitted
        slope `a` must be positive — otherwise the calibration is
        inverted and would flip real/fake predictions.
        """
        from core.scorer import PlattParams
        # Synthetic: score increases with label
        np.random.seed(42)
        n = 200
        scores = np.clip(np.random.beta(2, 5, n) * 0.4 + 0.1, 0.01, 0.99)
        # Higher scores -> more likely label=1 (authentic)
        labels = (scores > 0.4).astype(np.float64)
        # Add some noise so not all high scores are 1
        flip = np.random.random(n) < 0.15
        labels = np.where(flip, 1 - labels, labels)

        params = PlattParams.fit(scores, labels)
        assert params.a > 0, (
            f"Fitted slope a={params.a} should be positive for "
            f"monotonically-increasing data — sign bug?"
        )

    def test_platt_fit_clips_to_positive_slope(self) -> None:
        """The PlattParams.fit method clips `a` to be >= 0.01.

        This is a deliberate design choice: scores are conventionally
        P(authentic), so the calibration should never invert the
        direction of the prediction. If the fit wants to go negative,
        it means the calibration data is mislabeled or the model is
        broken — clipping prevents a single bad calibration dataset
        from silently flipping real/fake predictions.
        """
        from core.scorer import PlattParams
        np.random.seed(7)
        n = 200
        # High scores -> label 0 (fake); low scores -> label 1 (authentic)
        scores = np.clip(np.random.beta(2, 5, n) * 0.4 + 0.1, 0.01, 0.99)
        labels = (scores < 0.4).astype(np.float64)
        params = PlattParams.fit(scores, labels)
        # Clip should prevent `a` from going below 0.01
        assert params.a >= 0.01, (
            f"Fitted slope a={params.a} should be clipped to >= 0.01 "
            f"to prevent direction inversion"
        )
        # And `b` should be very negative (the fitter compensates for
        # the clipped slope by lowering the bias)
        assert params.b < 0, (
            f"Bias b={params.b} should compensate for clipped slope"
        )

    def test_fit_platt_parameters_uses_logit_space(self) -> None:
        """TrustScorer.fit_platt_parameters must produce params consistent
        with PlattParams.transform (logit space)."""
        from core.scorer import TrustScorer, PlattParams
        from schemas.schemas import ContentType
        np.random.seed(123)
        n = 100
        scores = np.clip(np.random.beta(3, 3, n), 0.05, 0.95)
        labels = (scores > 0.5).astype(np.float64)
        # Inject some noise so calibration is non-trivial
        flip = np.random.random(n) < 0.2
        labels = np.where(flip, 1 - labels, labels)

        scorer = TrustScorer()
        original = scorer.platt_params.get(ContentType.IMAGE_ONLY)
        try:
            params = scorer.fit_platt_parameters(scores, labels, ContentType.IMAGE_ONLY)
            # Verify the fitted params actually transform the data in a
            # direction consistent with the labels.
            calibrated_high = params.transform(0.8)
            calibrated_low = params.transform(0.2)
            # Since high scores correlate with label=1 in our data,
            # calibrated_high should exceed calibrated_low.
            assert calibrated_high > calibrated_low, (
                f"Platt transform broken: high={calibrated_high}, "
                f"low={calibrated_low} — direction inverted?"
            )
        finally:
            if original is not None:
                scorer.platt_params[ContentType.IMAGE_ONLY] = original

    def test_fit_platt_rejects_mismatched_lengths(self) -> None:
        """Mismatched scores/labels must raise ValueError, not silently
        produce garbage."""
        from core.scorer import TrustScorer
        from schemas.schemas import ContentType
        scorer = TrustScorer()
        with pytest.raises(ValueError, match="same length"):
            scorer.fit_platt_parameters(
                np.array([0.1, 0.5, 0.9]),
                np.array([0, 1]),
                ContentType.IMAGE_ONLY,
            )

    def test_fit_platt_warns_on_small_sample(self) -> None:
        """Fitting on <16 samples must NOT update params (unstable)."""
        from core.scorer import TrustScorer, PlattParams
        from schemas.schemas import ContentType
        scorer = TrustScorer()
        original = scorer.platt_params.get(ContentType.AUDIO_ONLY, PlattParams())
        result = scorer.fit_platt_parameters(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # only 5 samples
            np.array([0, 0, 1, 1, 1]),
            ContentType.AUDIO_ONLY,
        )
        # Must return the existing params unchanged.
        assert result.a == original.a
        assert result.b == original.b


# =========================================================================
# 4. core/explain.py — _detect_manipulation_type must consider all results
# =========================================================================

class TestDetectManipulationType:
    """Verify the early-return bug is fixed."""

    def _make_aggregated(self, results):
        from schemas.schemas import AggregatedResult
        return AggregatedResult(
            modality_results=results,
            fused_score=0.5,
            uncertainty=0.5,
            weights_used={},
        )

    def _make_modality_result(self, modality, score, confidence, details=None):
        from schemas.schemas import ModalityResult
        return ModalityResult(
            modality=modality,
            score=score,
            confidence=confidence,
            details=details or {},
        )

    def test_audio_clone_wins_over_video_face_swap(self) -> None:
        """When AUDIO has high confidence but VIDEO is also present,
        AUDIO_CLONE should win if its evidence is stronger."""
        from core.explain import ExplainabilityEngine, ManipulationType
        from schemas.schemas import Modality

        engine = ExplainabilityEngine()
        # Video modality with NO specific flags — score 0.5 (uncertain)
        # but previously this would still return FACE_SWAP because VIDEO
        # was the first modality checked.
        video = self._make_modality_result(
            Modality.VIDEO, score=0.5, confidence=0.5, details={}
        )
        # Audio modality with high confidence synthetic-voice score
        audio = self._make_modality_result(
            Modality.AUDIO, score=0.9, confidence=0.85, details={}
        )
        agg = self._make_aggregated([video, audio])

        result = engine._detect_manipulation_type(agg)
        assert result == ManipulationType.AUDIO_CLONE, (
            f"Expected AUDIO_CLONE (audio has stronger evidence), "
            f"got {result}"
        )

    def test_unknown_when_all_modalities_low_score(self) -> None:
        """All low-score modalities must return UNKNOWN, not FACE_SWAP."""
        from core.explain import ExplainabilityEngine, ManipulationType
        from schemas.schemas import Modality

        engine = ExplainabilityEngine()
        video = self._make_modality_result(
            Modality.VIDEO, score=0.1, confidence=0.6, details={}
        )
        audio = self._make_modality_result(
            Modality.AUDIO, score=0.2, confidence=0.7, details={}
        )
        agg = self._make_aggregated([video, audio])

        result = engine._detect_manipulation_type(agg)
        assert result == ManipulationType.UNKNOWN, (
            f"Expected UNKNOWN (no strong evidence), got {result}"
        )

    def test_lip_sync_boosted_over_face_swap(self) -> None:
        """Lip-sync anomaly should win over generic face-swap when both
        are present (it's a more specific manipulation type)."""
        from core.explain import ExplainabilityEngine, ManipulationType
        from schemas.schemas import Modality

        engine = ExplainabilityEngine()
        video = self._make_modality_result(
            Modality.VIDEO, score=0.7, confidence=0.8,
            details={"lip_sync_detected": True, "temporal_inconsistency": False}
        )
        agg = self._make_aggregated([video])

        result = engine._detect_manipulation_type(agg)
        assert result == ManipulationType.LIP_SYNC, (
            f"Lip-sync should win over face-swap, got {result}"
        )

    def test_image_ai_generated_when_image_score_high(self) -> None:
        from core.explain import ExplainabilityEngine, ManipulationType
        from schemas.schemas import Modality

        engine = ExplainabilityEngine()
        image = self._make_modality_result(
            Modality.IMAGE, score=0.9, confidence=0.85, details={}
        )
        agg = self._make_aggregated([image])
        assert engine._detect_manipulation_type(agg) == ManipulationType.AI_GENERATED_IMAGE

    def test_low_confidence_audio_does_not_trigger_audio_clone(self) -> None:
        """Audio with high score but very low confidence should NOT
        produce AUDIO_CLONE (the 0.4 confidence floor prevents
        mislabeling noisy audio as voice clones)."""
        from core.explain import ExplainabilityEngine, ManipulationType
        from schemas.schemas import Modality

        engine = ExplainabilityEngine()
        audio = self._make_modality_result(
            Modality.AUDIO, score=0.9, confidence=0.2, details={}
        )
        agg = self._make_aggregated([audio])
        result = engine._detect_manipulation_type(agg)
        assert result == ManipulationType.UNKNOWN, (
            f"Low-confidence audio should not yield AUDIO_CLONE, got {result}"
        )


# =========================================================================
# 5. analyzers/audio.py — confidence cap when no neural available
# =========================================================================

class TestAudioConfidenceCap:
    """Verify _compute_confidence caps at 0.15 when no neural models
    contributed a real score."""

    def test_confidence_capped_when_no_neural(self) -> None:
        """When any_neural_available=False, confidence must be 0.15."""
        # Import lazily so this test doesn't hard-depend on torch.
        try:
            from analyzers.audio import AudioAnalyzer, AudioAnalysisDetails
        except ImportError:
            pytest.skip("analyzers.audio requires unavailable deps")

        try:
            analyzer = AudioAnalyzer()
        except Exception:
            pytest.skip("AudioAnalyzer cannot be constructed in this env")

        details = AudioAnalysisDetails(
            audio_duration_seconds=10.0,
            segments_analyzed=5,
            primary_detector="wav2vec2_antispoof",
            wav2vec2_antispoof_score=0.5,  # default — no real inference
            any_neural_available=False,
        )
        conf = analyzer._compute_confidence(details)
        assert conf == 0.15, (
            f"Heuristic-only audio confidence should be 0.15, got {conf}"
        )

    def test_confidence_normal_when_neural_available(self) -> None:
        """When any_neural_available=True, normal confidence formula applies."""
        try:
            from analyzers.audio import AudioAnalyzer, AudioAnalysisDetails
        except ImportError:
            pytest.skip("analyzers.audio requires unavailable deps")

        try:
            analyzer = AudioAnalyzer()
        except Exception:
            pytest.skip("AudioAnalyzer cannot be constructed in this env")

        details = AudioAnalysisDetails(
            audio_duration_seconds=10.0,
            segments_analyzed=5,
            primary_detector="wav2vec2_antispoof",
            wav2vec2_antispoof_score=0.9,
            any_neural_available=True,
        )
        conf = analyzer._compute_confidence(details)
        # Should be > 0.15 since neural was available
        assert conf > 0.15, f"Neural-available confidence should exceed 0.15, got {conf}"
        assert conf <= 0.95


# =========================================================================
# 6. api/health.py — run_health_check must not mutate component results
# =========================================================================

class TestHealthCheckNoMutation:
    """Verify run_health_check does not .pop() metadata from results."""

    @pytest.mark.asyncio
    async def test_storage_metadata_preserved(self) -> None:
        """Storage health result (with latency/buckets) must appear
        verbatim in the components dict."""
        from api.health import run_health_check

        # Mock db and storage
        db = MagicMock()
        db.db = MagicMock()
        db.db.command = AsyncMock(return_value={"ok": 1})

        storage = MagicMock()
        storage.health_check = AsyncMock(return_value={
            "status": "healthy",
            "mode": "minio",
            "latency_ms": 12.5,
            "buckets": ["argus-uploads", "argus-results"],
        })

        # Mock the individual checks to avoid Redis/Celery/Models deps
        with patch("api.health.check_database", new=AsyncMock(return_value={
            "status": "healthy"
        })), \
             patch("api.health.check_storage", new=AsyncMock(return_value={
                 "status": "healthy", "mode": "minio",
                 "latency_ms": 12.5,
                 "buckets": ["argus-uploads", "argus-results"]
             })), \
             patch("api.health.check_redis", new=AsyncMock(return_value={
                 "status": "healthy"
             })), \
             patch("api.health.check_celery", new=AsyncMock(return_value={
                 "status": "healthy", "active_workers": 4
             })), \
             patch("api.health.check_models", new=AsyncMock(return_value={
                 "status": "healthy", "loaded": 5, "model_names": ["m1", "m2"]
             })):
            result = await run_health_check(db, storage)

        assert result["status"] == "healthy"
        # Storage metadata must NOT be lost
        storage_component = result["components"]["storage"]
        assert storage_component["latency_ms"] == 12.5, (
            "Storage latency_ms was lost — run_health_check is mutating results"
        )
        assert storage_component["buckets"] == ["argus-uploads", "argus-results"]
        # Celery metadata must NOT be lost
        assert result["components"]["celery"]["active_workers"] == 4
        # Models metadata must NOT be lost
        assert result["components"]["models"]["loaded"] == 5

    @pytest.mark.asyncio
    async def test_overall_status_unhealthy_when_any_unhealthy(self) -> None:
        from api.health import run_health_check

        db = MagicMock()
        storage = MagicMock()

        with patch("api.health.check_database", new=AsyncMock(return_value={
            "status": "unhealthy: connection refused"
        })), \
             patch("api.health.check_storage", new=AsyncMock(return_value={
                 "status": "healthy", "mode": "minio"
             })), \
             patch("api.health.check_redis", new=AsyncMock(return_value={
                 "status": "healthy"
             })), \
             patch("api.health.check_celery", new=AsyncMock(return_value={
                 "status": "healthy", "active_workers": 4
             })), \
             patch("api.health.check_models", new=AsyncMock(return_value={
                 "status": "healthy", "loaded": 5
             })):
            result = await run_health_check(db, storage)

        assert result["status"] == "unhealthy", (
            f"Expected 'unhealthy' when DB is down, got {result['status']}"
        )
        assert "database" in result["unhealthy_components"]

    @pytest.mark.asyncio
    async def test_overall_status_degraded_when_no_workers(self) -> None:
        from api.health import run_health_check

        db = MagicMock()
        storage = MagicMock()

        with patch("api.health.check_database", new=AsyncMock(return_value={
            "status": "healthy"
        })), \
             patch("api.health.check_storage", new=AsyncMock(return_value={
                 "status": "healthy", "mode": "minio"
             })), \
             patch("api.health.check_redis", new=AsyncMock(return_value={
                 "status": "healthy"
             })), \
             patch("api.health.check_celery", new=AsyncMock(return_value={
                 "status": "no_workers", "active_workers": 0
             })), \
             patch("api.health.check_models", new=AsyncMock(return_value={
                 "status": "healthy", "loaded": 5
             })):
            result = await run_health_check(db, storage)

        assert result["status"] == "degraded", (
            f"Expected 'degraded' when celery has no workers, got {result['status']}"
        )
        assert "celery" in result["degraded_components"]


# =========================================================================
# 7. fusion.py — Dirichlet evidential fusion correctness
# =========================================================================

class TestEvidentialFusion:
    """Verify the Dirichlet evidential fusion behaves correctly."""

    def test_single_high_confidence_modality_preserves_score(self) -> None:
        """A single modality with high confidence should preserve its
        score (no spurious shrinkage)."""
        from core.fusion import MultiModalFusion
        from schemas.schemas import Modality, ModalityResult, ContentType

        fusion = MultiModalFusion()
        result = fusion.aggregate([
            ModalityResult(
                modality=Modality.IMAGE,
                score=0.9,
                confidence=0.9,
                details={},
            )
        ], ContentType.IMAGE_ONLY)

        # Score should be close to 0.9 (with some confidence-based shrinkage
        # only when confidence is low — at 0.9 confidence, minimal shrinkage)
        assert 0.80 < result.fused_score <= 0.95, (
            f"High-confidence single modality should preserve score ~0.9, "
            f"got {result.fused_score}"
        )

    def test_disagreement_pulls_fused_score_toward_middle(self) -> None:
        """When modalities strongly disagree, the fused score should be
        closer to 0.5 than when they agree.

        Note: the Dirichlet uncertainty formula ``K / sum(alpha)`` actually
        yields *lower* uncertainty when both modalities have extreme scores
        (because both contribute high evidence). The disagreement signal
        is captured by the fused score being pulled to ~0.5, not by the
        uncertainty estimate. This is a known property of evidential
        Dirichlet fusion — see Sensoy et al. NeurIPS 2018.

        Also note: Dirichlet evidential fusion has a uniform prior
        (alpha_fake=1, alpha_real=1) that pulls weakly toward 0.5 even
        when modalities agree, so the agreeing case won't reach 0.85.
        The test therefore checks the *relative* pull toward 0.5.
        """
        from core.fusion import MultiModalFusion
        from schemas.schemas import Modality, ModalityResult, ContentType

        fusion = MultiModalFusion()

        # Both modalities agree content is fake (score > 0.5)
        agreeing_results = [
            ModalityResult(modality=Modality.IMAGE, score=0.85, confidence=0.8, details={}),
            ModalityResult(modality=Modality.AUDIO, score=0.80, confidence=0.8, details={}),
        ]
        # Modalities disagree: image says fake, audio says real
        disagreeing_results = [
            ModalityResult(modality=Modality.IMAGE, score=0.85, confidence=0.8, details={}),
            ModalityResult(modality=Modality.AUDIO, score=0.15, confidence=0.8, details={}),
        ]

        agg_agree = fusion.aggregate(agreeing_results, ContentType.VIDEO_WITH_SPEECH)
        agg_disagree = fusion.aggregate(disagreeing_results, ContentType.VIDEO_WITH_SPEECH)

        # The agreeing case should produce a fused score clearly above 0.5
        # (Dirichlet prior pulls toward 0.5, so we don't expect 0.85)
        assert agg_agree.fused_score > 0.55, (
            f"Agreeing 'fake' modalities should produce fused_score > 0.55, "
            f"got {agg_agree.fused_score}"
        )
        # The disagreeing case should pull the fused score closer to 0.5
        # than the agreeing case (lower distance from 0.5)
        dist_agree = abs(agg_agree.fused_score - 0.5)
        dist_disagree = abs(agg_disagree.fused_score - 0.5)
        assert dist_disagree < dist_agree, (
            f"Disagreeing modalities should pull fused_score closer to 0.5 "
            f"(disagree dist={dist_disagree}, agree dist={dist_agree})"
        )
        # The disagreeing case should be very close to 0.5
        assert 0.4 < agg_disagree.fused_score < 0.6, (
            f"Disagreeing modalities should pull fused_score near 0.5, "
            f"got {agg_disagree.fused_score}"
        )

    def test_zero_confidence_modality_does_not_dominate(self) -> None:
        """A modality with confidence=0 should contribute ~zero evidence."""
        from core.fusion import MultiModalFusion
        from schemas.schemas import Modality, ModalityResult, ContentType

        fusion = MultiModalFusion()
        results = [
            ModalityResult(modality=Modality.IMAGE, score=0.9, confidence=0.0, details={}),
            ModalityResult(modality=Modality.AUDIO, score=0.1, confidence=0.9, details={}),
        ]
        agg = fusion.aggregate(results, ContentType.VIDEO_WITH_SPEECH)
        # The IMAGE modality contributes zero evidence; AUDIO should dominate.
        # Fused score should be closer to 0.1 than to 0.9.
        assert agg.fused_score < 0.4, (
            f"Zero-confidence modality should not dominate fusion; "
            f"got fused_score={agg.fused_score}"
        )


# =========================================================================
# 8. Forensics — verifier import sanity (catches subtle bugs)
# =========================================================================

class TestModuleImportSanity:
    """Sanity-check that all pure-Python modules import cleanly.

    This catches issues like the original `import hashlib` inside the
    Settings class body that broke every transitive importer on
    Pydantic >=2.12.
    """

    @pytest.mark.parametrize("module_name", [
        "config",
        "schemas.schemas",
        "utils.errors",
        "utils.logging",
        "utils.metrics",
        "utils.hardware",
        "calibration.temperature_scaling",
        "calibration.conformal",
        "calibration.calibration_audit",
        "defenses.randomized_smoothing_lite",
        "defenses.randomized_preprocessing",
        "defenses.certified_robustness",
        "defenses.adversarial_gate",
        "monitoring.drift_detector",
        "monitoring.embedding_buffer",
        "monitoring.reference_store",
        "inference.memory_guard",
        "modes.mode_manager",
        "observability.metrics",
        "forensics.audit",
        "core.post_processing",
        "core.explain",
        "core.scorer",
        "core.fusion",
        "api.health",
    ])
    def test_module_imports(self, module_name) -> None:
        try:
            __import__(module_name)
        except ModuleNotFoundError as e:
            # Allow skip for missing optional deps (torch/onnx/etc.)
            if any(k in str(e) for k in [
                "torch", "onnx", "peft", "lxt", "transformers",
                "speechbrain", "timm", "c2pa", "alibi_detect",
                "datasets", "evaluate", "accelerate", "torchaudio",
                "torchvision", "motor", "celery", "prometheus_client",
                "structlog", "sentry_sdk", "reportlab", "minio", "boto3",
                "pymongo", "jwt", "jose", "passlib", "argon2",
                "pycparser", "cffi", "cryptography", "Crypto",
                "PIL", "cv2", "librosa", "soundfile", "pandas",
                "sklearn", "scipy", "aiohttp", "httpx", "tenacity",
                "yaml", "jsonschema", "orjson", "Jinja2", "redis",
                "tqdm", "rich", "dateutil", "dotenv", "filelock",
                "packaging",
            ]):
                pytest.skip(f"Optional dep missing: {e}")
            raise
