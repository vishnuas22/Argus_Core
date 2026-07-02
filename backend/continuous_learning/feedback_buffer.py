"""
Argus Core - Feedback Buffer (Iteration 4)
============================================
Appends labeled samples to a JSON Lines file for online retraining.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackSample:
    """A single labeled feedback sample."""
    timestamp: float
    modality: str          # image | audio | video
    input_hash: str        # sha256 of input bytes (for dedup)
    label: int             # 0 = real, 1 = fake
    predicted_score: float # what the detector predicted
    confidence: float      # detector confidence
    model_version: str     # which adapter version produced the prediction
    source: str = "user"   # user | expert | adversarial_gate
    notes: str = ""


class FeedbackBuffer:
    """
    Thread-safe append-only feedback buffer.

    Stores samples as JSON Lines (one JSON object per line) for easy
    incremental writes. The file is the source of truth for the
    retrain scheduler.
    """

    def __init__(self, path: str, max_samples: int = 10000):
        self._path = path
        self._max_samples = max_samples
        self._lock = threading.Lock()
        self._count = 0
        self._hashes: set = set()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Count existing samples and load hashes for dedup
        if os.path.exists(path):
            with open(path, "r") as fh:
                for line in fh:
                    self._count += 1
                    try:
                        obj = json.loads(line)
                        h = obj.get("input_hash")
                        if h:
                            self._hashes.add(h)
                    except json.JSONDecodeError:
                        pass
        logger.info(
            "FeedbackBuffer initialized: %s (%d existing samples, %d unique hashes)",
            path, self._count, len(self._hashes),
        )

    # ------------------------------------------------------------------
    def append(self, sample: FeedbackSample) -> bool:
        """
        Append a sample to the buffer.

        Args:
            sample: FeedbackSample to append.

        Returns:
            True if appended, False if buffer is full or sample is a
            duplicate (by input_hash).
        """
        if self._count >= self._max_samples:
            logger.warning("FeedbackBuffer full (%d samples); dropping", self._count)
            return False

        with self._lock:
            # Dedup by input_hash using in-memory set (O(1) lookup)
            if sample.input_hash in self._hashes:
                logger.debug("Duplicate sample %s; skipping", sample.input_hash[:12])
                return False

            with open(self._path, "a") as fh:
                fh.write(json.dumps(asdict(sample)) + "\n")
            self._count += 1
            if sample.input_hash:
                self._hashes.add(sample.input_hash)
            # Iteration 7: record feedback buffer size
            try:
                from observability import get_default_metrics
                get_default_metrics().record_feedback_buffer(sample.modality, self._count)
            except Exception:
                pass
            return True

    # ------------------------------------------------------------------
    def get_samples(
        self,
        modality: Optional[str] = None,
        since_timestamp: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[FeedbackSample]:
        """
        Read samples from the buffer.

        Args:
            modality: Filter by modality (None = all).
            since_timestamp: Only samples after this timestamp.
            limit: Maximum number of samples to return.

        Returns:
            List of FeedbackSample.
        """
        samples: List[FeedbackSample] = []
        if not os.path.exists(self._path):
            return samples

        with self._lock:
            with open(self._path, "r") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if modality and obj.get("modality") != modality:
                            continue
                        if since_timestamp and obj.get("timestamp", 0) < since_timestamp:
                            continue
                        samples.append(FeedbackSample(**obj))
                        if limit and len(samples) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        return samples

    # ------------------------------------------------------------------
    def count(self, modality: Optional[str] = None) -> int:
        """Count samples in the buffer."""
        if not modality:
            return self._count
        return len(self.get_samples(modality=modality))

    def clear(self) -> None:
        """Clear the buffer (after a successful retrain)."""
        with self._lock:
            if os.path.exists(self._path):
                os.remove(self._path)
            self._count = 0
            self._hashes.clear()
        # Also flush the Redis embedding buffer
        try:
            from monitoring.embedding_buffer import get_default_embedding_buffer
            buf = get_default_embedding_buffer()
            if buf:
                buf.clear()
        except Exception:
            pass
        logger.info("FeedbackBuffer cleared")

    def archive(self, archive_path: str) -> None:
        """Archive the buffer (rename to archive_path)."""
        with self._lock:
            if os.path.exists(self._path):
                os.rename(self._path, archive_path)
                self._count = 0
                self._hashes.clear()
        # Also flush the Redis embedding buffer
        try:
            from monitoring.embedding_buffer import get_default_embedding_buffer
            buf = get_default_embedding_buffer()
            if buf:
                buf.clear()
        except Exception:
            pass
        logger.info("FeedbackBuffer archived to %s", archive_path)


# ---------------------------------------------------------------------
_default_buffer: Optional[FeedbackBuffer] = None


def get_default_feedback_buffer() -> FeedbackBuffer:
    global _default_buffer
    if _default_buffer is None:
        from config import config
        _default_buffer = FeedbackBuffer(
            path=getattr(config, "feedback_buffer_path",
                         "/models/continuous_learning/feedback_buffer.json"),
        )
    return _default_buffer
