"""
Argus Core - Inference Provenance Tracking
==========================================

Records a structured audit trail for every prediction so that downstream
consumers (forensic reports, court evidence, regression tests) can verify
that the prediction originated from real model inference rather than
heuristics or placeholder logic.

This module enforces the user's protocol requirement:
    "Predictions must originate from verified model inference rather
    than heuristics or placeholder logic."

Usage
-----
The :class:`ProvenanceRecorder` is a context manager that wraps a single
modality inference call. On success it records the model name, model
version (from the registry), input hash, output score, and latency. On
failure it records the exception class and message. The resulting
:class:`ProvenanceRecord` can be embedded in :class:`ModalityResult.details`
under the ``provenance`` key.

Example
-------
    >>> async with ProvenanceRecorder(
    ...     modality="audio",
    ...     model_name="wav2vec2_antispoof",
    ...     registry=get_model_registry(),
    ... ) as rec:
    ...     result = await engine.infer("wav2vec2_antispoof", audio_batch)
    ...     rec.record_output(score=float(result.confidence))
    >>> rec.record  # ProvenanceRecord instance
    ProvenanceRecord(modality='audio', model_name='wav2vec2_antispoof',
                     model_version='1.0.0', status='ok', ...)

Scientific Integrity
--------------------
The recorder distinguishes three prediction origins:

  * ``"model_inference"``  — real ML model produced the score
  * ``"heuristic_only"``   — no neural model contributed (e.g., audio
                             analyzer's dampened fallback path)
  * ``"placeholder"``      — score is a constant default (0.5) because
                             the model failed to load or raised

This classification flows into the final forensic report so reviewers
can weight evidence appropriately. A ``placeholder`` prediction must
never be the sole basis for a "fake" or "authentic" verdict.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Dict, Optional


# =========================================================================
# Constants
# =========================================================================

# Prediction origin classification — see module docstring.
ORIGIN_MODEL_INFERENCE = "model_inference"
ORIGIN_HEURISTIC_ONLY = "heuristic_only"
ORIGIN_PLACEHOLDER = "placeholder"

# Status codes for the recorder.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class ProvenanceRecord:
    """
    Structured audit record for a single inference call.

    Fields are intentionally JSON-serializable so the record can be
    embedded in MongoDB documents and forensic PDF reports without
    custom encoders.
    """
    # Identity
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # ISO-8601 timestamp with microseconds; rendered as UTC.
    timestamp: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()
    ))

    # What was run
    modality: str = ""               # "image" | "audio" | "video"
    model_name: str = ""             # registry key, e.g. "wav2vec2_antispoof"
    model_version: str = "unknown"   # from registry metadata, if available
    model_sha256: str = ""           # sha256 of model file, if verify_model_checksums

    # What was fed in
    input_hash: str = ""             # sha256 of the input tensor bytes (first 16 hex chars)
    input_shape: str = ""            # human-readable shape, e.g. "(1, 3, 224, 224)"

    # What came out
    output_score: float = 0.5        # raw model score in [0, 1]
    output_confidence: float = 0.0   # model confidence in [0, 1]
    status: str = STATUS_OK          # "ok" | "error" | "skipped"

    # Provenance classification — see module docstring
    origin: str = ORIGIN_MODEL_INFERENCE

    # Diagnostics
    latency_ms: float = 0.0
    error_type: str = ""             # exception class name, if status == "error"
    error_message: str = ""          # truncated to 500 chars

    # Optional: hardware context (useful for CPU-vs-GPU audits)
    device: str = "cpu"              # "cpu" | "cuda" | "mps"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict view (nested dataclasses are recursively converted)."""
        return asdict(self)

    def is_real_inference(self) -> bool:
        """True iff this record represents real ML model inference."""
        return (
            self.status == STATUS_OK
            and self.origin == ORIGIN_MODEL_INFERENCE
            and self.model_name != ""
        )


# =========================================================================
# Helpers
# =========================================================================

def _sha256_of_bytes(data: bytes) -> str:
    """Compute the first 16 hex chars of sha256(data).

    We truncate to 16 chars (64 bits) to keep MongoDB documents small —
    this is a provenance fingerprint, not a content-addressable hash.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def _shape_to_str(shape: Any) -> str:
    """Render a tensor shape (tuple or list) as a human-readable string."""
    if shape is None:
        return ""
    try:
        return str(tuple(shape))
    except TypeError:
        return str(shape)


def _safe_error_message(msg: Any, max_len: int = 500) -> str:
    """Truncate error message to max_len chars, replacing newlines."""
    s = str(msg).replace("\n", " ").replace("\r", " ")
    return s[:max_len]


# =========================================================================
# Recorder
# =========================================================================

class ProvenanceRecorder:
    """
    Async context manager that records a single inference call's provenance.

    The recorder is intentionally decoupled from the inference engine —
    it does not run the inference itself, it just times and classifies
    whatever inference the caller performs inside the ``async with``
    block. This keeps the recorder testable in isolation.

    The caller MUST call :meth:`record_output` inside the block to
    populate the score. If an exception is raised inside the block,
    :meth:`__aexit__` records it as an error and returns ``False`` (so
    the exception propagates), but the partial record is preserved on
    ``self.record``.
    """

    def __init__(
        self,
        modality: str,
        model_name: str,
        registry: Any = None,
        device: str = "cpu",
    ):
        self.modality = modality
        self.model_name = model_name
        self.registry = registry
        self.device = device

        self.record = ProvenanceRecord(
            modality=modality,
            model_name=model_name,
            device=device,
        )
        self._start_time: Optional[float] = None
        self._output_recorded = False

        # Populate model_version and model_sha256 from registry, if available.
        if registry is not None:
            self._populate_model_metadata()

    def _populate_model_metadata(self) -> None:
        """Look up model version and sha256 from the registry."""
        try:
            md = self.registry.get_model_metadata(self.model_name)
            if md is not None:
                # ModelMetadata has `version` and (optional) `sha256` fields.
                self.record.model_version = getattr(md, "version", "unknown") or "unknown"
                sha = getattr(md, "sha256", None)
                if sha:
                    self.record.model_sha256 = sha[:16]
        except Exception:
            # Registry lookup is best-effort — never block inference on it.
            pass

    def record_input(self, input_data: Any) -> None:
        """Record the input tensor's hash and shape.

        For numpy arrays, uses ``tobytes()``. For other types, falls back
        to ``str(input_data).encode()``.
        """
        try:
            if hasattr(input_data, "tobytes"):
                # numpy ndarray
                self.record.input_hash = _sha256_of_bytes(input_data.tobytes())
                self.record.input_shape = _shape_to_str(getattr(input_data, "shape", ""))
            elif isinstance(input_data, dict):
                # Multi-input model — hash a stable concatenation of values
                import io
                buf = io.BytesIO()
                for k in sorted(input_data.keys()):
                    v = input_data[k]
                    if hasattr(v, "tobytes"):
                        buf.write(k.encode("utf-8"))
                        buf.write(v.tobytes())
                self.record.input_hash = _sha256_of_bytes(buf.getvalue())
                self.record.input_shape = ", ".join(
                    f"{k}={_shape_to_str(getattr(v, 'shape', ''))}"
                    for k, v in input_data.items()
                )
            else:
                self.record.input_hash = _sha256_of_bytes(
                    str(input_data).encode("utf-8", errors="ignore")
                )
        except Exception:
            # Input hashing is best-effort.
            pass

    def record_output(
        self,
        score: float,
        confidence: Optional[float] = None,
        origin: str = ORIGIN_MODEL_INFERENCE,
    ) -> None:
        """Record the model's output score and (optional) confidence."""
        self.record.output_score = float(score)
        self.record.output_confidence = (
            float(confidence) if confidence is not None else float(score)
        )
        self.record.origin = origin
        self._output_recorded = True

    async def __aenter__(self) -> "ProvenanceRecorder":
        self._start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Always compute latency, even on error.
        if self._start_time is not None:
            self.record.latency_ms = (time.perf_counter() - self._start_time) * 1000.0

        if exc_val is not None:
            self.record.status = STATUS_ERROR
            self.record.error_type = exc_type.__name__ if exc_type else "UnknownError"
            self.record.error_message = _safe_error_message(exc_val)
            # If no output was recorded before the exception, the score
            # is a placeholder.
            if not self._output_recorded:
                self.record.origin = ORIGIN_PLACEHOLDER
            # Return False so the exception propagates.
            return False

        if not self._output_recorded:
            # Caller forgot to record output — treat as skipped.
            self.record.status = STATUS_SKIPPED
            self.record.origin = ORIGIN_PLACEHOLDER

        return False  # do not suppress exceptions


# =========================================================================
# Convenience: classify a modality result's prediction origin
# =========================================================================

def classify_prediction_origin(
    score: float,
    confidence: float,
    any_neural_available: bool = True,
    model_name: str = "",
    status: str = STATUS_OK,
) -> str:
    """
    Classify a prediction's origin based on its provenance signals.

    This is a pure function — given the signals, returns one of the
    ``ORIGIN_*`` constants. Used by analyzers that don't use the
    :class:`ProvenanceRecorder` directly but still want to classify
    their outputs.
    """
    if status == STATUS_ERROR or model_name == "":
        return ORIGIN_PLACEHOLDER
    if not any_neural_available:
        return ORIGIN_HEURISTIC_ONLY
    # A real model produced a non-default score.
    # If the score is exactly 0.5 AND confidence is low, that's suspicious —
    # it might be a placeholder that wasn't flagged as such.
    if abs(score - 0.5) < 1e-6 and confidence < 0.3:
        return ORIGIN_PLACEHOLDER
    return ORIGIN_MODEL_INFERENCE


__all__ = [
    "ProvenanceRecord",
    "ProvenanceRecorder",
    "classify_prediction_origin",
    "ORIGIN_MODEL_INFERENCE",
    "ORIGIN_HEURISTIC_ONLY",
    "ORIGIN_PLACEHOLDER",
    "STATUS_OK",
    "STATUS_ERROR",
    "STATUS_SKIPPED",
]
