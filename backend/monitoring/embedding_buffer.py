"""
Argus Core - Embedding Buffer
==============================
Redis-backed ring buffer for recent inference embeddings.

Feeds the drift detection pipeline by accumulating embeddings from every
analysis and exposing them for batch drift checks.

Uses Redis LPUSH + LTRIM for O(1) append with automatic cap enforcement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

# Default cap per modality
DEFAULT_MAX_EMBEDDINGS = 500


@dataclass
class EmbeddingEntry:
    """A single buffered embedding."""
    embedding: List[float]
    modality: str
    analysis_id: str
    score: float
    timestamp: float


class EmbeddingBuffer:
    """
    Redis-backed ring buffer storing recent inference embeddings.

    Key format: "argus:embeddings:{modality}"
    Each entry is a JSON-serialized EmbeddingEntry.
    LPUSH + LTRIM keeps only the most recent N entries per modality.
    """

    def __init__(self, redis_client, max_per_modality: int = DEFAULT_MAX_EMBEDDINGS):
        self._redis = redis_client
        self._max = max_per_modality
        logger.info(
            "EmbeddingBuffer initialized: max_per_modality=%d",
            max_per_modality,
        )

    def _key(self, modality: str) -> str:
        return f"argus:embeddings:{modality}"

    def append(
        self,
        embedding: np.ndarray,
        modality: str,
        analysis_id: str,
        score: float = 0.5,
    ) -> bool:
        """
        Append an embedding to the buffer for the given modality.

        Args:
            embedding: (D,) numpy array of embedding values.
            modality: "image" | "audio" | "video".
            analysis_id: The analysis ID that produced this embedding.
            score: The trust score from the analysis.

        Returns:
            True if appended, False on error.
        """
        if embedding is None or len(embedding) == 0:
            return False
        try:
            entry = EmbeddingEntry(
                embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
                modality=modality,
                analysis_id=analysis_id,
                score=score,
                timestamp=time.time(),
            )
            key = self._key(modality)
            pipe = self._redis.pipeline()
            pipe.lpush(key, json.dumps(asdict(entry)))
            pipe.ltrim(key, 0, self._max - 1)
            pipe.execute()
            return True
        except Exception as e:
            logger.warning("EmbeddingBuffer append failed: %s", e)
            return False

    def get_embeddings(
        self,
        modality: str,
        limit: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve buffered embeddings for a modality as a numpy array.

        Args:
            modality: "image" | "audio" | "video".
            limit: Max entries to retrieve (None = all).

        Returns:
            (N, D) numpy array of embeddings, or None if empty.
        """
        try:
            key = self._key(modality)
            raw = self._redis.lrange(key, 0, (limit or self._max) - 1)
            if not raw:
                return None
            entries = []
            for item in raw:
                data = json.loads(item)
                entries.append(data["embedding"])
            if not entries:
                return None
            return np.array(entries, dtype=np.float64)
        except Exception as e:
            logger.warning("EmbeddingBuffer get_embeddings failed: %s", e)
            return None

    def count(self, modality: str) -> int:
        """Return the number of buffered embeddings for a modality."""
        try:
            return self._redis.llen(self._key(modality))
        except Exception:
            return 0

    def counts_all(self) -> Dict[str, int]:
        """Return embedding counts for all modalities."""
        result = {}
        for modality in ("image", "audio", "video"):
            result[modality] = self.count(modality)
        return result

    def clear(self, modality: Optional[str] = None) -> None:
        """Clear the buffer for one or all modalities."""
        try:
            modalities = [modality] if modality else ["image", "audio", "video"]
            for m in self._redis.delete(*[self._key(m) for m in modalities]):
                pass
        except Exception as e:
            logger.warning("EmbeddingBuffer clear failed: %s", e)


# ---------------------------------------------------------------------
_default_buffer: Optional[EmbeddingBuffer] = None


def get_default_embedding_buffer() -> Optional[EmbeddingBuffer]:
    """
    Get or create the default EmbeddingBuffer backed by the configured Redis.

    Returns None if Redis is unavailable.
    """
    global _default_buffer
    if _default_buffer is not None:
        return _default_buffer
    try:
        import redis as redis_lib
        from config import config
        redis_url = getattr(config, "redis_url", "redis://localhost:6379/0")
        client = redis_lib.from_url(redis_url, decode_responses=True, socket_timeout=5)
        client.ping()
        _default_buffer = EmbeddingBuffer(client)
        return _default_buffer
    except Exception as e:
        logger.warning("EmbeddingBuffer unavailable (Redis not connected): %s", e)
        return None
