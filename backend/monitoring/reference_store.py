"""
Argus Core - Reference Store (Iteration 2)
============================================
Stores the reference distribution for drift detection.

The reference is a compact summary of "normal" embeddings collected
during a calibration phase. We store:
- A subsample of the raw embeddings (for MMD).
- Bin edges + counts (for PSI).

The store is persisted to disk as a JSON + npz file pair.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReferenceStore:
    """
    Persisted reference distribution for drift detection.

    Attributes:
        embeddings: Subsample of raw embeddings (for MMD).
        bin_edges: Per-dimension bin edges (for PSI).
        bin_counts: Per-dimension bin counts.
        num_samples: Number of samples used to build the reference.
        modality: Modality tag (image/audio/video).
        created_at: ISO timestamp.
    """
    embeddings: Optional[np.ndarray] = None  # (N_ref, D)
    bin_edges: Optional[List[List[float]]] = None
    bin_counts: Optional[List[List[int]]] = None
    num_samples: int = 0
    modality: str = "image"
    created_at: str = ""
    max_reference_size: int = 1000  # cap for memory

    # ------------------------------------------------------------------
    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        modality: str = "image",
        num_bins: int = 20,
    ) -> None:
        """
        Build the reference from a set of embeddings.

        Args:
            embeddings: (N, D) array of embeddings.
            modality: Modality tag.
            num_bins: Number of bins per dimension for PSI.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        N, D = embeddings.shape

        # Subsample if too large
        if N > self.max_reference_size:
            idx = np.random.choice(N, self.max_reference_size, replace=False)
            embeddings = embeddings[idx]
            N = self.max_reference_size

        self.embeddings = embeddings
        self.num_samples = N
        self.modality = modality
        import datetime
        self.created_at = datetime.datetime.utcnow().isoformat()

        # Build per-dimension bins
        self.bin_edges = []
        self.bin_counts = []
        for d in range(D):
            col = embeddings[:, d]
            percentiles = np.linspace(0, 100, num_bins + 1)
            edges = np.percentile(col, percentiles)
            edges = np.unique(edges)
            if len(edges) < 2:
                continue
            edges[0] = -np.inf
            edges[-1] = np.inf
            counts = np.histogram(col, bins=edges)[0]
            self.bin_edges.append([float(e) if np.isfinite(e) else float("inf") for e in edges])
            self.bin_counts.append(counts.tolist())

        logger.info(
            "ReferenceStore built: %d samples, %d dims, modality=%s",
            N, D, modality,
        )

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the reference to disk.

        Args:
            path: Base path (without extension). Two files will be
                created: <path>.json (metadata) and <path>.npz (embeddings).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Save embeddings as npz
        if self.embeddings is not None:
            np.savez(f"{path}.npz", embeddings=self.embeddings)
        # Save metadata as json
        metadata = {
            "num_samples": self.num_samples,
            "modality": self.modality,
            "created_at": self.created_at,
            "max_reference_size": self.max_reference_size,
            "bin_edges": self.bin_edges,
            "bin_counts": self.bin_counts,
            "embedding_shape": list(self.embeddings.shape) if self.embeddings is not None else None,
        }
        with open(f"{path}.json", "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("ReferenceStore saved to %s.{json,npz}", path)

    @classmethod
    def load(cls, path: str) -> "ReferenceStore":
        """Load a reference from disk."""
        with open(f"{path}.json", "r") as fh:
            metadata = json.load(fh)
        store = cls(
            num_samples=metadata["num_samples"],
            modality=metadata["modality"],
            created_at=metadata["created_at"],
            max_reference_size=metadata.get("max_reference_size", 1000),
            bin_edges=metadata.get("bin_edges"),
            bin_counts=metadata.get("bin_counts"),
        )
        npz_path = f"{path}.npz"
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            store.embeddings = data["embeddings"]
        return store


# ---------------------------------------------------------------------
_default_store: Optional[ReferenceStore] = None


def get_default_reference_store() -> ReferenceStore:
    global _default_store
    if _default_store is None:
        _default_store = ReferenceStore()
    return _default_store
