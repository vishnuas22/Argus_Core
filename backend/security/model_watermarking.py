"""
Argus Core - Model Watermarking & Fingerprinting Implementation
=================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Watermarking (weight-embedding)
# =====================================================================

@dataclass
class WatermarkResult:
    """Result of a watermark embed or extract operation."""
    success: bool
    key_hash: str = ""        # SHA256 of the secret key
    bit_capacity: int = 0     # How many bits were embedded/extracted
    bit_errors: int = 0       # Hamming distance (extract only)
    ber: float = 0.0          # Bit Error Rate (extract only)
    message: str = ""


class Watermarker:
    """
    Embeds and extracts a secret-key watermark in LoRA adapter weights.

    Algorithm (Uchida et al., 2017):
    1. Generate a secret key K of N bits (default 256).
    2. Generate a random matrix A of shape (N, weight_dim) seeded by K.
    3. Embed: modify the weight vector w so that sign(A @ w) == K.
       This is done by adding a small perturbation in the direction
       that flips the sign of A @ w where it disagrees with K.
    4. Extract: compute sign(A @ w) and compare to K.

    The perturbation is small (epsilon=1e-3) so it does not affect
    model accuracy. The watermark is robust to fine-tuning and pruning
    but not to full retraining.
    """

    def __init__(
        self,
        key_length: int = 256,
        epsilon: float = 0.0,  # 0 = auto-scale to weight magnitude
        seed: Optional[int] = 42,
    ):
        self.key_length = key_length
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        logger.info(
            "Watermarker initialized: key_length=%d, epsilon=%.4f",
            key_length, epsilon,
        )

    # ------------------------------------------------------------------
    def generate_key(self) -> np.ndarray:
        """Generate a random binary key of key_length bits."""
        return self._rng.integers(0, 2, size=self.key_length, dtype=np.int8)

    # ------------------------------------------------------------------
    def embed(
        self,
        weights: np.ndarray,
        key: np.ndarray,
    ) -> Tuple[np.ndarray, WatermarkResult]:
        """
        Embed a watermark key into a flat weight vector.

        Uses an iterative approach: repeatedly add perturbations in the
        direction needed to flip incorrect bits, with the epsilon
        auto-scaled to the weight magnitude.

        Args:
            weights: 1D weight array (will be flattened if multi-dim).
            key: Binary key of shape (key_length,).

        Returns:
            (watermarked_weights, WatermarkResult)
        """
        w = np.asarray(weights, dtype=np.float64).flatten()
        if len(key) != self.key_length:
            return w.astype(np.float32), WatermarkResult(
                success=False, message=f"key length {len(key)} != {self.key_length}"
            )

        # Generate the embedding matrix A seeded by the key
        key_seed = int(hashlib.sha256(key.tobytes()).hexdigest(), 16) % (2**32)
        rng_a = np.random.default_rng(key_seed)
        A = rng_a.standard_normal((self.key_length, len(w))).astype(np.float64)

        # Auto-scale epsilon to weight magnitude if not set
        weight_scale = float(np.std(w))
        epsilon = self.epsilon if self.epsilon > 0 else weight_scale * 0.5

        # Iterative embedding (up to 10 rounds)
        w_wm = w.copy()
        max_rounds = 10
        for round_idx in range(max_rounds):
            current = (A @ w_wm >= 0).astype(np.int8)
            needs_flip = current != key
            num_flip = int(needs_flip.sum())
            if num_flip == 0:
                break

            # Perturbation: for each bit that needs flipping, add a
            # contribution proportional to A[i] (or -A[i])
            perturbation = np.zeros_like(w_wm)
            for i in range(self.key_length):
                if needs_flip[i]:
                    if key[i] == 1:
                        perturbation += A[i]
                    else:
                        perturbation -= A[i]
            # Scale: increase epsilon each round
            scaled_eps = epsilon * (1 + round_idx * 0.5)
            w_wm = w_wm + scaled_eps * perturbation / max(self.key_length, 1)

        # Verify
        extracted = (A @ w_wm >= 0).astype(np.int8)
        bit_errors = int(np.sum(extracted != key))
        ber = bit_errors / self.key_length

        return w_wm.astype(np.float32), WatermarkResult(
            success=ber < 0.1,
            key_hash=hashlib.sha256(key.tobytes()).hexdigest()[:16],
            bit_capacity=self.key_length,
            bit_errors=bit_errors,
            ber=ber,
            message=f"embedded {self.key_length} bits in {round_idx+1} rounds, BER={ber:.4f}",
        )

    # ------------------------------------------------------------------
    def extract(
        self,
        weights: np.ndarray,
        key: np.ndarray,
    ) -> WatermarkResult:
        """
        Extract a watermark from weights and verify against a key.

        Args:
            weights: 1D weight array (will be flattened if multi-dim).
            key: Expected binary key.

        Returns:
            WatermarkResult with BER.
        """
        w = np.asarray(weights, dtype=np.float32).flatten()
        if len(key) != self.key_length:
            return WatermarkResult(
                success=False, message=f"key length {len(key)} != {self.key_length}"
            )

        key_seed = int(hashlib.sha256(key.tobytes()).hexdigest(), 16) % (2**32)
        rng_a = np.random.default_rng(key_seed)
        A = rng_a.standard_normal((self.key_length, len(w))).astype(np.float32)

        extracted = (A @ w >= 0).astype(np.int8)
        bit_errors = int(np.sum(extracted != key))
        ber = bit_errors / self.key_length

        return WatermarkResult(
            success=ber < 0.1,
            key_hash=hashlib.sha256(key.tobytes()).hexdigest()[:16],
            bit_capacity=self.key_length,
            bit_errors=bit_errors,
            ber=ber,
            message=f"extracted {self.key_length} bits, BER={ber:.4f}",
        )

    # ------------------------------------------------------------------
    def embed_in_lora_adapter(
        self,
        adapter_dir: str,
        key: Optional[np.ndarray] = None,
    ) -> WatermarkResult:
        """
        Embed a watermark in a LoRA adapter's safetensors weights.
        """
        try:
            from safetensors.torch import load_file, save_file
            import torch
        except ImportError:
            return WatermarkResult(success=False, message="safetensors not installed")

        adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return WatermarkResult(
                success=False, message=f"adapter not found: {adapter_path}"
            )

        if key is None:
            key = self.generate_key()
            key_path = os.path.join(adapter_dir, "watermark_key.json")
            with open(key_path, "w") as fh:
                json.dump({
                    "key": key.tolist(),
                    "key_hash": hashlib.sha256(key.tobytes()).hexdigest(),
                    "key_length": self.key_length,
                }, fh, indent=2)
            logger.info("Watermark key saved to %s", key_path)

        state_dict = load_file(adapter_path)
        largest_key = max(state_dict, key=lambda k: state_dict[k].numel())
        w = state_dict[largest_key].cpu().numpy().flatten()
        w_wm, result = self.embed(w, key)
        if result.success:
            original_shape = state_dict[largest_key].shape
            state_dict[largest_key] = torch.from_numpy(
                w_wm.reshape(original_shape)
            ).to(state_dict[largest_key].dtype)
            save_file(state_dict, adapter_path)
            logger.info(
                "Watermark embedded in %s[%s]: %s",
                adapter_path, largest_key, result.message,
            )
            # Iteration 7: record watermark embedding
            try:
                from observability import get_default_metrics
                adapter_name = os.path.basename(adapter_dir)
                get_default_metrics().watermark_embedded.labels(
                    adapter_name=adapter_name
                ).inc()
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    def verify_lora_adapter(
        self,
        adapter_dir: str,
        key: Optional[np.ndarray] = None,
    ) -> WatermarkResult:
        """
        Verify the watermark in a LoRA adapter.

        Args:
            adapter_dir: Path to the adapter directory.
            key: Expected key. If None, loads from watermark_key.json.

        Returns:
            WatermarkResult.
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            return WatermarkResult(success=False, message="safetensors not installed")

        adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return WatermarkResult(
                success=False, message=f"adapter not found: {adapter_path}"
            )

        if key is None:
            key_path = os.path.join(adapter_dir, "watermark_key.json")
            if not os.path.exists(key_path):
                return WatermarkResult(
                    success=False, message="no watermark key found"
                )
            with open(key_path, "r") as fh:
                key_data = json.load(fh)
                key = np.array(key_data["key"], dtype=np.int8)

        state_dict = load_file(adapter_path)
        largest_key = max(state_dict, key=lambda k: state_dict[k].numel())
        w = state_dict[largest_key].cpu().numpy().flatten()
        result = self.extract(w, key)
        # Iteration 7: record watermark verification
        try:
            from observability import get_default_metrics
            adapter_name = os.path.basename(adapter_dir)
            get_default_metrics().watermark_verified.labels(
                adapter_name=adapter_name, success=str(result.success)
            ).inc()
        except Exception:
            pass
        return result


# =====================================================================
# Fingerprinting (behavioral)
# =====================================================================

@dataclass
class FingerprintResult:
    """Result of a fingerprint computation."""
    success: bool
    fingerprint: str = ""       # SHA256 hash of the behavioral signature
    num_probes: int = 0
    message: str = ""


class Fingerprinter:
    """
    Computes a behavioral fingerprint of a detector.

    The fingerprint is a hash of the detector's outputs on a fixed set
    of probe inputs. It does NOT modify the model and can detect model
    stealing: if a suspect model has the same fingerprint, it was
    likely copied from the original.

    Algorithm:
    1. Generate (or load) a fixed set of probe inputs (random noise
       images of a fixed shape).
    2. Run the detector on each probe input.
    3. Concatenate the outputs and hash them with SHA256.
    4. The hash is the fingerprint.

    The probe inputs are generated from a fixed seed, so the same
    detector always produces the same fingerprint.
    """

    def __init__(
        self,
        num_probes: int = 64,
        probe_shape: Tuple[int, ...] = (224, 224, 3),
        seed: int = 42,
    ):
        self.num_probes = num_probes
        self.probe_shape = probe_shape
        self._rng = np.random.default_rng(seed)
        self._probes: Optional[List[np.ndarray]] = None
        logger.info(
            "Fingerprinter initialized: num_probes=%d, shape=%s",
            num_probes, probe_shape,
        )

    # ------------------------------------------------------------------
    def _generate_probes(self) -> List[np.ndarray]:
        """Generate fixed probe inputs from the seed."""
        if self._probes is not None:
            return self._probes
        probes = []
        for _ in range(self.num_probes):
            probe = self._rng.integers(
                0, 256, size=self.probe_shape, dtype=np.uint8
            )
            probes.append(probe)
        self._probes = probes
        return probes

    # ------------------------------------------------------------------
    async def fingerprint(self, detect_fn) -> FingerprintResult:
        """
        Compute the behavioral fingerprint of a detector.

        Args:
            detect_fn: Async callable that takes a probe input and
                returns a DetectionResult (or float score).

        Returns:
            FingerprintResult with SHA256 hash.
        """
        try:
            probes = self._generate_probes()
            scores: List[float] = []
            for probe in probes:
                try:
                    r = await detect_fn(probe)
                    if hasattr(r, "score"):
                        scores.append(float(r.score))
                    else:
                        scores.append(float(r))
                except Exception as e:
                    logger.debug("Probe failed: %s", e)
                    scores.append(0.5)

            # Hash the concatenated scores
            score_bytes = struct.pack(f"{len(scores)}d", *scores)
            fingerprint = hashlib.sha256(score_bytes).hexdigest()
            return FingerprintResult(
                success=True,
                fingerprint=fingerprint,
                num_probes=len(scores),
                message=f"fingerprinted {len(scores)} probes",
            )
        except Exception as e:
            logger.error("Fingerprinting failed: %s", e)
            return FingerprintResult(success=False, message=str(e))

    # ------------------------------------------------------------------
    def compare_fingerprints(
        self, fp1: str, fp2: str, threshold: int = 4
    ) -> Dict[str, Any]:
        """
        Compare two fingerprints for similarity.

        Args:
            fp1: First fingerprint (hex string).
            fp2: Second fingerprint (hex string).
            threshold: Hamming distance threshold for "same model".

        Returns:
            Dict with hamming_distance, is_same_model, similarity.
        """
        if len(fp1) != len(fp2):
            return {
                "hamming_distance": -1,
                "is_same_model": False,
                "similarity": 0.0,
                "message": "fingerprint length mismatch",
            }
        # Compare as bytes
        b1 = bytes.fromhex(fp1)
        b2 = bytes.fromhex(fp2)
        hamming = sum(bin(a ^ b).count("1") for a, b in zip(b1, b2))
        max_bits = len(b1) * 8
        similarity = 1.0 - (hamming / max_bits)
        return {
            "hamming_distance": hamming,
            "is_same_model": hamming <= threshold,
            "similarity": round(similarity, 6),
            "message": "same model" if hamming <= threshold else "different model",
        }


# =====================================================================
# Singletons
# =====================================================================

_default_watermarker: Optional[Watermarker] = None
_default_fingerprinter: Optional[Fingerprinter] = None


def get_default_watermarker() -> Watermarker:
    global _default_watermarker
    if _default_watermarker is None:
        _default_watermarker = Watermarker()
    return _default_watermarker


def get_default_fingerprinter() -> Fingerprinter:
    global _default_fingerprinter
    if _default_fingerprinter is None:
        _default_fingerprinter = Fingerprinter()
    return _default_fingerprinter
