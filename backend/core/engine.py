"""
Argus Core - Inference Engine
=============================
Model inference engine with VRAM management and dynamic batching.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - core/engine.py

SOTA Algorithms:
- Model Loading: LRU cache with VRAM pressure monitoring (via ModelManager)
- Inference: ONNX Runtime with TensorRT/OpenVINO execution providers
- Batching: Dynamic batching based on available memory

Target Hardware: RTX 3050 (4GB VRAM) with 3.5GB budget

Integration:
- Imports: models/manager.py, config.py, schemas/internal.py
- Inputs: preprocessed_data: PreprocessedData, model_name: str
- Outputs: InferenceResult with confidence scores
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from config import config
from models.manager import ModelManager, get_model_manager
from models.registry import ModelRegistry, get_model_registry, ModelMetadata
from schemas.schemas import Modality, PreprocessedData
from utils.logging import get_logger
from utils.errors import InferenceError, ModelLoadError

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """
    Result container for model inference.
    
    Attributes:
        predictions: Raw model output predictions
        confidence: Model confidence in predictions (0-1)
        class_probabilities: Per-class probability distribution
        inference_time_ms: Time taken for inference
        model_name: Name of model used
        batch_size: Batch size used for inference
    """
    predictions: np.ndarray
    confidence: float
    class_probabilities: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    batch_size: int = 1
    
    def get_prediction_label(self, labels: List[str] = None) -> str:
        """Get human-readable prediction label."""
        labels = labels or ["real", "fake"]
        if self.predictions.ndim > 1:
            pred_idx = int(np.argmax(self.predictions[0]))
        else:
            pred_idx = int(self.predictions[0] > 0.5)
        return labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    
    def is_fake_prediction(self) -> bool:
        """Check if prediction indicates fake content."""
        if self.class_probabilities is not None:
            # Assuming class 1 is "fake"
            return float(self.class_probabilities[..., 1].mean()) > 0.5
        return float(self.predictions.mean()) > 0.5


@dataclass
class BatchInferenceResult:
    """
    Aggregated results for batch inference.
    
    Contains individual results plus batch-level statistics.
    """
    results: List[InferenceResult] = field(default_factory=list)
    total_inference_time_ms: float = 0.0
    mean_confidence: float = 0.0
    model_name: str = ""
    
    def aggregate_predictions(self) -> np.ndarray:
        """Aggregate all predictions into single array."""
        if not self.results:
            return np.array([])
        return np.concatenate([r.predictions for r in self.results], axis=0)
    
    def aggregate_confidences(self) -> np.ndarray:
        """Get all confidence values."""
        return np.array([r.confidence for r in self.results])


class InferenceEngine:
    """
    Centralized model inference engine with hardware optimization.
    
    Manages model inference across all analyzers with:
    - Automatic VRAM management via ModelManager
    - Dynamic batching based on available memory
    - Thread-safe concurrent inference
    - Comprehensive error handling and logging
    
    Usage:
        engine = InferenceEngine()
        result = await engine.infer("efficientnet_b3_spatial", image_batch)
    """
    
    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        max_concurrent_models: int = 3,
        default_batch_size: int = 8
    ):
        """
        Initialize inference engine.
        
        Args:
            model_manager: ModelManager instance (creates default if None)
            model_registry: ModelRegistry instance (creates default if None)
            max_concurrent_models: Maximum models to keep loaded
            default_batch_size: Default batch size for inference
        """
        self.model_manager = model_manager or get_model_manager()
        self.registry = model_registry or get_model_registry()
        self.max_concurrent_models = max_concurrent_models
        self.default_batch_size = default_batch_size
        
        # Thread pool for CPU-bound preprocessing
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Inference statistics
        self._stats = {
            "total_inferences": 0,
            "total_batches": 0,
            "total_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(
            f"InferenceEngine initialized: max_concurrent={max_concurrent_models}, "
            f"default_batch={default_batch_size}"
        )
    
    async def infer(
        self,
        model_name: str,
        inputs: Union[np.ndarray, List[np.ndarray]],
        batch_size: Optional[int] = None,
        return_probabilities: bool = True
    ) -> InferenceResult:
        """
        Run inference with automatic batching.
        
        Handles single inputs, batches, and lists of inputs.
        Automatically determines optimal batch size if not specified.
        
        Args:
            model_name: Model name from registry
            inputs: Input tensor(s) - single array or list of arrays
            batch_size: Override auto-batch sizing (None = automatic)
            return_probabilities: Whether to compute class probabilities
            
        Returns:
            InferenceResult with predictions and confidence
            
        Raises:
            InferenceError: If inference fails
            ModelLoadError: If model cannot be loaded
        """
        start_time = time.time()
        
        # Handle dict inputs (multi-input models like ModernBERT)
        if isinstance(inputs, dict):
            return await self._infer_multi_input(model_name, inputs, return_probabilities)
        
        # Normalize inputs to list
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                # Single image: add batch dimension
                inputs = np.expand_dims(inputs, 0)
            input_batch = inputs
        else:
            # List of arrays: stack them
            input_batch = np.stack(inputs, axis=0)
        
        actual_batch_size = input_batch.shape[0]
        
        # Get optimal batch size
        if batch_size is None:
            try:
                metadata = self.registry.get_model_metadata(model_name)
                batch_size = self.get_optimal_batch_size(
                    model_name,
                    input_batch.shape[1:]
                )
            except Exception:
                batch_size = self.default_batch_size
        
        try:
            # Load model (uses LRU cache in ModelManager)
            session = await self.model_manager.get_model(model_name)
            
            # Run inference in batches
            all_outputs = []
            
            for i in range(0, actual_batch_size, batch_size):
                batch = input_batch[i:i + batch_size]
                
                # Ensure correct dtype
                if batch.dtype != np.float32:
                    batch = batch.astype(np.float32)
                
                # Get input name from session
                input_name = session.get_inputs()[0].name
                
                # Run inference
                outputs = session.run(None, {input_name: batch})
                all_outputs.append(outputs[0])
            
            # Concatenate all outputs
            predictions = np.concatenate(all_outputs, axis=0)
            
            # Compute confidence and probabilities
            if return_probabilities and predictions.shape[-1] >= 2:
                # Apply softmax if not already probabilities
                if predictions.max() > 1.0 or predictions.min() < 0.0:
                    class_probabilities = self._softmax(predictions)
                else:
                    class_probabilities = predictions
                
                # Confidence is max probability
                confidence = float(np.max(class_probabilities))
            else:
                class_probabilities = None
                confidence = float(np.mean(np.abs(predictions)))
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self._stats["total_inferences"] += actual_batch_size
            self._stats["total_batches"] += 1
            self._stats["total_time_ms"] += inference_time_ms
            
            logger.debug(
                f"Inference complete: model={model_name}, batch={actual_batch_size}, "
                f"time={inference_time_ms:.2f}ms"
            )
            
            return InferenceResult(
                predictions=predictions,
                confidence=confidence,
                class_probabilities=class_probabilities,
                inference_time_ms=inference_time_ms,
                model_name=model_name,
                batch_size=actual_batch_size
            )
            
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            raise InferenceError(model_name, str(e))
    
    async def infer_batch(
        self,
        model_name: str,
        inputs: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> BatchInferenceResult:
        """
        Run inference on a large batch with automatic chunking.
        
        Useful for processing many inputs efficiently.
        
        Args:
            model_name: Model name from registry
            inputs: List of input tensors
            batch_size: Chunk size for processing
            
        Returns:
            BatchInferenceResult with all individual results
        """
        start_time = time.time()
        
        if batch_size is None:
            metadata = self.registry.get_model_metadata(model_name)
            batch_size = self.get_optimal_batch_size(
                model_name,
                inputs[0].shape
            )
        
        results = []
        
        # Process in chunks
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i:i + batch_size]
            result = await self.infer(model_name, chunk, batch_size=batch_size)
            
            # Split batch result into individual results
            for j in range(result.predictions.shape[0]):
                individual = InferenceResult(
                    predictions=result.predictions[j:j+1],
                    confidence=result.confidence,
                    class_probabilities=(
                        result.class_probabilities[j:j+1]
                        if result.class_probabilities is not None else None
                    ),
                    inference_time_ms=result.inference_time_ms / result.batch_size,
                    model_name=model_name,
                    batch_size=1
                )
                results.append(individual)
        
        total_time = (time.time() - start_time) * 1000
        mean_conf = np.mean([r.confidence for r in results]) if results else 0.0
        
        return BatchInferenceResult(
            results=results,
            total_inference_time_ms=total_time,
            mean_confidence=mean_conf,
            model_name=model_name
        )
    
    async def infer_multi_model(
        self,
        model_inputs: Dict[str, np.ndarray]
    ) -> Dict[str, InferenceResult]:
        """
        Run inference on multiple models concurrently.
        
        Useful for parallel analysis across different analyzers.
        
        Args:
            model_inputs: Dict mapping model names to their inputs
            
        Returns:
            Dict mapping model names to their InferenceResults
        """
        tasks = {}
        for model_name, inputs in model_inputs.items():
            tasks[model_name] = self.infer(model_name, inputs)
        
        # Run all inferences concurrently
        results = {}
        for model_name, task in tasks.items():
            try:
                results[model_name] = await task
            except Exception as e:
                logger.error(f"Multi-model inference failed for {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def get_optimal_batch_size(
        self,
        model_name: str,
        input_shape: Tuple[int, ...]
    ) -> int:
        """
        Calculate optimal batch size given VRAM constraints.
        
        Uses model metadata and current VRAM availability to determine
        the largest batch that can fit in memory.
        
        Args:
            model_name: Model name from registry
            input_shape: Shape of single input (excluding batch dim)
            
        Returns:
            Optimal batch size
        """
        try:
            metadata = self.registry.get_model_metadata(model_name)
            
            # Get available VRAM
            available_mb = self.model_manager.get_available_vram()
            model_vram = metadata.vram_mb
            
            # Estimate activation memory per sample
            # Heuristic: activation memory ~ 2x input size for transformers
            input_size_bytes = np.prod(input_shape) * 4  # float32
            activation_per_sample_mb = (input_size_bytes * 2) / (1024 * 1024)
            
            # Calculate max batch based on available VRAM
            headroom_mb = 500  # Leave 500MB headroom
            usable_mb = max(0, available_mb - model_vram - headroom_mb)
            
            if activation_per_sample_mb > 0:
                max_batch = int(usable_mb / activation_per_sample_mb)
            else:
                max_batch = metadata.optimal_batch_size
            
            # Clamp to model limits
            optimal = min(
                max(1, max_batch),
                metadata.max_batch_size,
                metadata.optimal_batch_size * 2  # Don't go too far above optimal
            )
            
            # Round down to power of 2 for efficiency
            optimal = 2 ** int(np.log2(optimal)) if optimal > 1 else 1
            
            logger.debug(
                f"Optimal batch for {model_name}: {optimal} "
                f"(available={available_mb}MB, per_sample={activation_per_sample_mb:.1f}MB)"
            )
            
            return optimal
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}, using default")
            return self.default_batch_size
    
    async def _infer_multi_input(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        return_probabilities: bool = True
    ) -> InferenceResult:
        """
        Run inference on models with multiple inputs (e.g., ModernBERT).
        
        Args:
            model_name: Model name from registry
            inputs: Dict mapping input names to arrays
            return_probabilities: Whether to compute class probabilities
            
        Returns:
            InferenceResult with predictions and confidence
        """
        start_time = time.time()
        
        try:
            session = await self.model_manager.get_model(model_name)
            
            # Prepare inputs with correct dtypes
            feed_dict = {}
            for inp in session.get_inputs():
                name = inp.name
                if name in inputs:
                    arr = inputs[name]
                    if arr.dtype != np.int64 and arr.dtype != np.int32:
                        arr = arr.astype(np.int64)
                    feed_dict[name] = arr
            
            # Run inference
            outputs = session.run(None, feed_dict)
            predictions = outputs[0]
            
            # Compute confidence and probabilities
            if return_probabilities and predictions.shape[-1] >= 2:
                if predictions.max() > 1.0 or predictions.min() < 0.0:
                    class_probabilities = self._softmax(predictions)
                else:
                    class_probabilities = predictions
                confidence = float(np.max(class_probabilities))
            else:
                class_probabilities = None
                confidence = float(np.mean(np.abs(predictions)))
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            return InferenceResult(
                predictions=predictions,
                confidence=confidence,
                class_probabilities=class_probabilities,
                inference_time_ms=inference_time_ms,
                model_name=model_name,
                batch_size=1
            )
            
        except Exception as e:
            logger.error(f"Multi-input inference failed for {model_name}: {e}")
            raise InferenceError(model_name, str(e))
    
    async def warmup_model(
        self,
        model_name: str,
        dummy_input_shape: Optional[Tuple[int, ...]] = None
    ) -> bool:
        """
        Warmup a single model by loading and running dummy inference.
        
        JIT compilation and memory allocation happen on first inference,
        so warming up reduces latency for first real request.
        
        Args:
            model_name: Model to warmup
            dummy_input_shape: Custom input shape (None = use model default)
            
        Returns:
            True if warmup succeeded, False otherwise
        """
        try:
            # Get model metadata for input shape
            metadata = self.registry.get_model_metadata(model_name)
            
            if dummy_input_shape:
                shape = dummy_input_shape
            else:
                shape = tuple(metadata.input_shape)
            
            # Create dummy input
            dummy = np.random.randn(*shape).astype(np.float32)
            
            # Run inference
            await self.infer(model_name, dummy, batch_size=1)
            
            return True
            
        except Exception as e:
            logger.warning(f"Warmup failed for {model_name}: {e}")
            return False
    
    async def warmup(
        self,
        model_names: List[str],
        dummy_input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    ) -> Dict[str, bool]:
        """
        Warmup multiple models by running dummy inference.
        
        JIT compilation and memory allocation happen on first inference,
        so warming up reduces latency for first real request.
        
        Args:
            model_names: Models to warmup
            dummy_input_shapes: Custom input shapes per model
            
        Returns:
            Dict mapping model names to warmup success status
        """
        results = {}
        
        for model_name in model_names:
            shape = dummy_input_shapes.get(model_name) if dummy_input_shapes else None
            results[model_name] = await self.warmup_model(model_name, shape)
            if results[model_name]:
                logger.info(f"Warmed up model: {model_name}")
        
        return results
    
    def preprocess_input(
        self,
        input_data: np.ndarray,
        model_name: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess input for specific model.
        
        Applies model-specific normalization and formatting.
        
        Args:
            input_data: Raw input data
            model_name: Target model
            normalize: Whether to apply normalization
            
        Returns:
            Preprocessed input ready for inference
        """
        try:
            metadata = self.registry.get_model_metadata(model_name)
        except Exception:
            # Use defaults if model not in registry
            return input_data.astype(np.float32)
        
        processed = input_data.copy()
        
        # Ensure correct shape
        expected_shape = metadata.input_shape[1:]  # Exclude batch dim
        if processed.shape[-len(expected_shape):] != tuple(expected_shape):
            logger.warning(
                f"Input shape mismatch for {model_name}: "
                f"got {processed.shape}, expected [..., {expected_shape}]"
            )
        
        # Normalize if required
        if normalize and metadata.normalize_input:
            low, high = metadata.input_range
            
            # ImageNet normalization (common for vision models)
            if metadata.category.value in ["spatial", "temporal", "image", "face_detection"]:
                # Standard ImageNet mean/std
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                
                # Assume input is in [0, 255]
                if processed.max() > 1.0:
                    processed = processed / 255.0
                
                processed = (processed - mean) / std
            else:
                # Generic normalization to input_range
                if processed.max() > high or processed.min() < low:
                    processed = (processed - processed.min()) / (processed.max() - processed.min())
                    processed = processed * (high - low) + low
        
        return processed.astype(np.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get inference statistics.
        
        Returns:
            Dict with inference counts, timing, cache stats
        """
        avg_time = (
            self._stats["total_time_ms"] / self._stats["total_batches"]
            if self._stats["total_batches"] > 0 else 0
        )
        
        return {
            **self._stats,
            "avg_batch_time_ms": round(avg_time, 2),
            "loaded_models": self.model_manager.get_loaded_models(),
            "vram_usage_mb": self.model_manager.get_vram_usage(),
            "available_vram_mb": self.model_manager.get_available_vram()
        }
    
    def reset_stats(self) -> None:
        """Reset inference statistics."""
        self._stats = {
            "total_inferences": 0,
            "total_batches": 0,
            "total_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
        await self.model_manager.clear_cache()
        logger.info("InferenceEngine cleanup complete")


# Singleton instance
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """
    Get singleton inference engine instance.
    
    Thread-safe lazy initialization.
    """
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


async def initialize_inference_engine(
    warmup_models: Optional[List[str]] = None
) -> InferenceEngine:
    """
    Initialize inference engine with optional warmup.
    
    Args:
        warmup_models: Models to preload and warmup
        
    Returns:
        Initialized InferenceEngine
    """
    engine = get_inference_engine()
    
    if warmup_models:
        await engine.warmup(warmup_models)
    
    return engine
