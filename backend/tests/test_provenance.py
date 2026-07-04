"""
Unit tests for :mod:`core.provenance` — inference provenance tracking.

These tests verify that:
  1. The ProvenanceRecorder correctly classifies prediction origins.
  2. Successful inference produces a "model_inference" record.
  3. Failed inference (exception) produces a "placeholder" record.
  4. Heuristic-only paths can be explicitly marked.
  5. Records are JSON-serializable for MongoDB / PDF embedding.
  6. The `is_real_inference()` helper correctly distinguishes real
     inference from heuristic-only / placeholder outputs.

These tests are pure-Python and run on CPU-only hosts without torch,
onnxruntime, MongoDB, or Redis.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.provenance import (
    ProvenanceRecord,
    ProvenanceRecorder,
    classify_prediction_origin,
    ORIGIN_MODEL_INFERENCE,
    ORIGIN_HEURISTIC_ONLY,
    ORIGIN_PLACEHOLDER,
    STATUS_OK,
    STATUS_ERROR,
    STATUS_SKIPPED,
)


# =========================================================================
# ProvenanceRecorder behavior
# =========================================================================

class TestProvenanceRecorderSuccess:
    """Verify the happy path: real model inference → real record."""

    @pytest.mark.asyncio
    async def test_successful_inference_classified_as_model_inference(self) -> None:
        """A successful inference call with a recorded score must be
        classified as ORIGIN_MODEL_INFERENCE."""
        async with ProvenanceRecorder(
            modality="audio",
            model_name="wav2vec2_antispoof",
            registry=None,
            device="cpu",
        ) as rec:
            # Simulate recording input + output
            rec.record_input(np.zeros((1, 48000), dtype=np.float32))
            rec.record_output(score=0.85, confidence=0.92)

        assert rec.record.status == STATUS_OK
        assert rec.record.origin == ORIGIN_MODEL_INFERENCE
        assert rec.record.output_score == 0.85
        assert rec.record.output_confidence == 0.92
        assert rec.record.modality == "audio"
        assert rec.record.model_name == "wav2vec2_antispoof"
        assert rec.record.latency_ms >= 0.0
        # is_real_inference should be True
        assert rec.record.is_real_inference() is True

    @pytest.mark.asyncio
    async def test_input_hash_is_stable_for_same_input(self) -> None:
        """Same input → same hash, different input → different hash."""
        inp1 = np.zeros((1, 100), dtype=np.float32)
        inp2 = np.zeros((1, 100), dtype=np.float32)
        inp3 = np.ones((1, 100), dtype=np.float32)

        async with ProvenanceRecorder("image", "clip_image", None) as rec1:
            rec1.record_input(inp1)
            rec1.record_output(0.5)

        async with ProvenanceRecorder("image", "clip_image", None) as rec2:
            rec2.record_input(inp2)
            rec2.record_output(0.5)

        async with ProvenanceRecorder("image", "clip_image", None) as rec3:
            rec3.record_input(inp3)
            rec3.record_output(0.5)

        assert rec1.record.input_hash == rec2.record.input_hash, (
            "Same input must produce same hash"
        )
        assert rec1.record.input_hash != rec3.record.input_hash, (
            "Different input must produce different hash"
        )

    @pytest.mark.asyncio
    async def test_dict_input_handled(self) -> None:
        """Multi-input models (dict of named tensors) must be hashed correctly."""
        inp = {
            "input_values": np.zeros((1, 48000), dtype=np.float32),
            "attention_mask": np.ones((1, 48000), dtype=np.int32),
        }
        async with ProvenanceRecorder("audio", "wav2vec2", None) as rec:
            rec.record_input(inp)
            rec.record_output(0.7)

        assert rec.record.input_hash, "Dict input must produce a hash"
        assert "input_values" in rec.record.input_shape
        assert "attention_mask" in rec.record.input_shape


class TestProvenanceRecorderFailure:
    """Verify error paths are classified correctly."""

    @pytest.mark.asyncio
    async def test_exception_inside_block_records_error(self) -> None:
        """An exception raised inside the `async with` block must:
          - record status=error
          - record the exception type and message
          - classify origin as placeholder (no output recorded)
          - propagate the exception (return False from __aexit__)
        """
        with pytest.raises(ValueError, match="model failed"):
            async with ProvenanceRecorder(
                "image", "dinov2_image", None
            ) as rec:
                rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
                raise ValueError("model failed to load")

        assert rec.record.status == STATUS_ERROR
        assert rec.record.error_type == "ValueError"
        assert "model failed" in rec.record.error_message
        assert rec.record.origin == ORIGIN_PLACEHOLDER
        assert rec.record.is_real_inference() is False

    @pytest.mark.asyncio
    async def test_exception_after_output_records_error_but_keeps_score(self) -> None:
        """If output was recorded before the exception, origin stays
        as model_inference (the model DID produce a real output; the
        exception happened in post-processing)."""
        with pytest.raises(RuntimeError):
            async with ProvenanceRecorder("audio", "wav2vec2", None) as rec:
                rec.record_input(np.zeros(100, dtype=np.float32))
                rec.record_output(score=0.7, confidence=0.8)
                raise RuntimeError("post-processing failed")

        assert rec.record.status == STATUS_ERROR
        # Output was recorded before the exception, so origin is still
        # model_inference — the model's prediction is real.
        assert rec.record.origin == ORIGIN_MODEL_INFERENCE
        assert rec.record.output_score == 0.7

    @pytest.mark.asyncio
    async def test_skipped_when_output_not_recorded(self) -> None:
        """If the caller forgets to call record_output, status is SKIPPED."""
        async with ProvenanceRecorder("image", "clip", None) as rec:
            rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
            # No record_output call!

        assert rec.record.status == STATUS_SKIPPED
        assert rec.record.origin == ORIGIN_PLACEHOLDER
        assert rec.record.is_real_inference() is False


class TestProvenanceRecorderHeuristic:
    """Verify heuristic-only paths can be explicitly marked."""

    @pytest.mark.asyncio
    async def test_heuristic_only_origin_set_explicitly(self) -> None:
        """When a modality falls back to heuristics, the caller should
        pass origin=ORIGIN_HEURISTIC_ONLY to record_output."""
        async with ProvenanceRecorder("audio", "vocoder_artifacts", None) as rec:
            rec.record_input(np.zeros(100, dtype=np.float32))
            rec.record_output(
                score=0.5,
                confidence=0.15,
                origin=ORIGIN_HEURISTIC_ONLY,
            )

        assert rec.record.origin == ORIGIN_HEURISTIC_ONLY
        assert rec.record.is_real_inference() is False, (
            "Heuristic-only records must not be classified as real inference"
        )


class TestProvenanceRecordSerialization:
    """Verify records are JSON-serializable for MongoDB / PDF embedding."""

    @pytest.mark.asyncio
    async def test_record_to_dict_json_serializable(self) -> None:
        """to_dict() must produce a JSON-serializable dict."""
        async with ProvenanceRecorder("image", "clip", None) as rec:
            rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
            rec.record_output(score=0.9, confidence=0.95)

        d = rec.record.to_dict()
        # Must be JSON-serializable
        s = json.dumps(d)
        d2 = json.loads(s)
        assert d2["modality"] == "image"
        assert d2["model_name"] == "clip"
        assert d2["output_score"] == 0.9
        assert d2["origin"] == ORIGIN_MODEL_INFERENCE

    def test_empty_record_serializable(self) -> None:
        """A bare ProvenanceRecord() with defaults must serialize."""
        rec = ProvenanceRecord()
        d = rec.to_dict()
        s = json.dumps(d)
        assert "record_id" in d
        assert "timestamp" in d


class TestProvenanceRegistryLookup:
    """Verify the recorder populates model_version from the registry."""

    @pytest.mark.asyncio
    async def test_registry_lookup_populates_version(self) -> None:
        """When a registry is provided, model_version is populated."""
        mock_registry = MagicMock()
        mock_md = MagicMock()
        mock_md.version = "1.2.3"
        # sha256 is truncated to 16 chars by the recorder
        mock_md.sha256 = "abc123def4560000" + "0" * 48
        mock_registry.get_model_metadata.return_value = mock_md

        async with ProvenanceRecorder(
            "image", "clip", registry=mock_registry
        ) as rec:
            rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
            rec.record_output(0.5)

        assert rec.record.model_version == "1.2.3"
        assert rec.record.model_sha256 == "abc123def4560000", (
            f"Expected first 16 chars of sha256, got {rec.record.model_sha256}"
        )

    @pytest.mark.asyncio
    async def test_registry_lookup_failure_is_silent(self) -> None:
        """If the registry raises, the recorder must not crash."""
        mock_registry = MagicMock()
        mock_registry.get_model_metadata.side_effect = RuntimeError("db down")

        async with ProvenanceRecorder(
            "image", "clip", registry=mock_registry
        ) as rec:
            rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
            rec.record_output(0.5)

        # Registry failure is best-effort; version stays "unknown".
        assert rec.record.model_version == "unknown"
        assert rec.record.status == STATUS_OK


# =========================================================================
# classify_prediction_origin helper
# =========================================================================

class TestClassifyPredictionOrigin:
    """Verify the pure-function origin classifier."""

    def test_real_inference_classification(self) -> None:
        """A confident, non-0.5 score from a real model is model_inference."""
        assert classify_prediction_origin(
            score=0.85, confidence=0.9, any_neural_available=True,
            model_name="wav2vec2", status=STATUS_OK,
        ) == ORIGIN_MODEL_INFERENCE

    def test_placeholder_when_status_error(self) -> None:
        """Error status → placeholder."""
        assert classify_prediction_origin(
            score=0.5, confidence=0.5, any_neural_available=True,
            model_name="wav2vec2", status=STATUS_ERROR,
        ) == ORIGIN_PLACEHOLDER

    def test_placeholder_when_no_model_name(self) -> None:
        """Empty model name → placeholder."""
        assert classify_prediction_origin(
            score=0.85, confidence=0.9, any_neural_available=True,
            model_name="", status=STATUS_OK,
        ) == ORIGIN_PLACEHOLDER

    def test_heuristic_only_when_no_neural(self) -> None:
        """any_neural_available=False → heuristic_only."""
        assert classify_prediction_origin(
            score=0.5, confidence=0.15, any_neural_available=False,
            model_name="vocoder", status=STATUS_OK,
        ) == ORIGIN_HEURISTIC_ONLY

    def test_placeholder_for_default_score_with_low_confidence(self) -> None:
        """Score=0.5 with low confidence looks like a default placeholder."""
        assert classify_prediction_origin(
            score=0.5, confidence=0.2, any_neural_available=True,
            model_name="wav2vec2", status=STATUS_OK,
        ) == ORIGIN_PLACEHOLDER

    def test_real_inference_for_default_score_with_high_confidence(self) -> None:
        """Score=0.5 with high confidence is a real (if uncertain) prediction."""
        assert classify_prediction_origin(
            score=0.5, confidence=0.8, any_neural_available=True,
            model_name="wav2vec2", status=STATUS_OK,
        ) == ORIGIN_MODEL_INFERENCE


# =========================================================================
# Integration: latency recording
# =========================================================================

class TestProvenanceLatency:
    """Verify latency is recorded accurately."""

    @pytest.mark.asyncio
    async def test_latency_recorded_on_success(self) -> None:
        """Latency must be > 0 for a successful inference."""
        async with ProvenanceRecorder("image", "clip", None) as rec:
            rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
            await asyncio.sleep(0.05)  # 50ms
            rec.record_output(0.5)

        # Latency should be at least 50ms (with some tolerance for CI).
        assert rec.record.latency_ms >= 40.0, (
            f"Latency {rec.record.latency_ms}ms should be >= 40ms"
        )

    @pytest.mark.asyncio
    async def test_latency_recorded_on_error(self) -> None:
        """Latency must be recorded even when an exception occurs."""
        with pytest.raises(RuntimeError):
            async with ProvenanceRecorder("image", "clip", None) as rec:
                rec.record_input(np.zeros((1, 3, 224, 224), dtype=np.float32))
                await asyncio.sleep(0.02)
                raise RuntimeError("inference failed")

        assert rec.record.latency_ms >= 15.0, (
            f"Latency {rec.record.latency_ms}ms should be recorded on error"
        )
