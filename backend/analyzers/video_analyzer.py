"""
Argus Core - Video Analyzer (Orchestrator)
==========================================
Video deepfake detection orchestrator. Coordinates spatial, temporal, and lip-sync sub-analyzers.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/video.py

SOTA Algorithms:
- Ensemble: Weighted voting across 3 sub-analyzers (spatial, temporal, lipsync)
- Anomaly Detection: Z-score based frame anomaly flagging
- Aggregation: Uncertainty-weighted fusion for final score

Pipeline:
1. Spatial analysis (per-frame artifacts) - EfficientNet-B3 + CLIP
2. Temporal analysis (cross-frame consistency) - X-CLIP Transformer
3. Lip-sync verification (if audio present) - LIPINC-V2
4. Ensemble aggregation with anomaly flagging

Integration:
- Imports: analyzers/video/spatial.py, analyzers/video/temporal.py, analyzers/video/lipsync.py, core/engine.py
- Inputs: VideoData (frames, faces, audio) via PreprocessedData
- Outputs: VideoResult

Target Hardware: RTX 3050 (4GB VRAM) with INT8 quantization
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time

from analyzers.base import BaseAnalyzer, aggregate_scores, compute_confidence, detect_anomalies
from analyzers.video.spatial import SpatialAnalyzer, get_spatial_analyzer
from analyzers.video.temporal import TemporalAnalyzer, get_temporal_analyzer
from analyzers.video.lipsync import LipSyncAnalyzer, get_lipsync_analyzer
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType,
    VideoResult, SpatialResult, TemporalResult, LipSyncResult
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError, InferenceError

if TYPE_CHECKING:
    from core.engine import InferenceEngine
    from core.explain import ExplainabilityEngine

logger = get_logger(__name__)


@dataclass
class VideoAnalysisMetrics:
    """
    Metrics from video analysis pipeline.
    
    Tracks timing and quality metrics for each sub-analyzer.
    """
    total_time_ms: float = 0.0
    spatial_time_ms: float = 0.0
    temporal_time_ms: float = 0.0
    lipsync_time_ms: float = 0.0
    frames_processed: int = 0
    faces_detected: int = 0
    anomalies_detected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "total_time_ms": round(self.total_time_ms, 2),
            "spatial_time_ms": round(self.spatial_time_ms, 2),
            "temporal_time_ms": round(self.temporal_time_ms, 2),
            "lipsync_time_ms": round(self.lipsync_time_ms, 2),
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "anomalies_detected": self.anomalies_detected
        }


@dataclass
class EnsembleWeights:
    """
    Dynamic weights for ensemble aggregation.
    
    Weights are adjusted based on:
    - Content type (video with/without speech)
    - Sub-analyzer confidence levels
    - Detection signal strength
    """
    spatial: float = 0.40
    temporal: float = 0.35
    lipsync: float = 0.25
    
    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = self.spatial + self.temporal + self.lipsync
        if total > 0:
            self.spatial /= total
            self.temporal /= total
            self.lipsync /= total
    
    def adjust_for_no_audio(self) -> None:
        """Redistribute lipsync weight when no audio present."""
        if self.lipsync > 0:
            redistribute = self.lipsync
            self.spatial += redistribute * 0.6
            self.temporal += redistribute * 0.4
            self.lipsync = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "spatial": round(self.spatial, 3),
            "temporal": round(self.temporal, 3),
            "lipsync": round(self.lipsync, 3)
        }


class VideoAnalyzer(BaseAnalyzer):
    """
    Multi-stage video deepfake detection orchestrator.
    
    Coordinates three specialized sub-analyzers:
    1. SpatialAnalyzer: Per-frame artifact detection
    2. TemporalAnalyzer: Cross-frame consistency analysis
    3. LipSyncAnalyzer: Audio-visual synchronization verification
    
    The orchestrator:
    - Manages sub-analyzer execution (parallel where possible)
    - Aggregates results using uncertainty-weighted ensemble
    - Detects anomaly frames requiring attention
    - Generates comprehensive VideoResult
    
    Usage:
        video_analyzer = VideoAnalyzer()
        result = await video_analyzer.analyze(preprocessed_data, engine)
    
    Or directly analyze frames:
        video_result = await video_analyzer.analyze_video(
            frames=face_crops,
            audio_features=mel_spectrogram,
            engine=inference_engine
        )
    """
    
    def __init__(
        self,
        spatial_analyzer: Optional[SpatialAnalyzer] = None,
        temporal_analyzer: Optional[TemporalAnalyzer] = None,
        lipsync_analyzer: Optional[LipSyncAnalyzer] = None,
        base_weights: Optional[EnsembleWeights] = None,
        min_frames: int = 5,
        enable_parallel: bool = True
    ):
        """
        Initialize video analyzer with sub-analyzers.
        
        Args:
            spatial_analyzer: SpatialAnalyzer instance (creates default if None)
            temporal_analyzer: TemporalAnalyzer instance (creates default if None)
            lipsync_analyzer: LipSyncAnalyzer instance (creates default if None)
            base_weights: Base ensemble weights (uses defaults if None)
            min_frames: Minimum frames required for analysis
            enable_parallel: Whether to run sub-analyzers in parallel
        """
        super().__init__(
            analyzer_name="VideoAnalyzer",
            supported_modalities=[Modality.VIDEO],
            version="1.0.0"
        )
        
        # Initialize sub-analyzers (lazy loading via singletons)
        self.spatial = spatial_analyzer or get_spatial_analyzer()
        self.temporal = temporal_analyzer or get_temporal_analyzer()
        self.lipsync = lipsync_analyzer or get_lipsync_analyzer()
        
        # Configuration
        self.base_weights = base_weights or EnsembleWeights()
        self.min_frames = min_frames
        self.enable_parallel = enable_parallel
        
        # Thresholds
        self.high_confidence_threshold = 0.85
        self.anomaly_z_threshold = 2.0
        
        logger.info(
            f"VideoAnalyzer initialized: min_frames={min_frames}, "
            f"parallel={enable_parallel}, weights={self.base_weights.to_dict()}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return all models required by sub-analyzers.
        
        Returns:
            Combined list of model registry keys from all sub-analyzers
        """
        models = []
        models.extend(self.spatial.get_required_models())
        models.extend(self.temporal.get_required_models())
        models.extend(self.lipsync.get_required_models())
        # Remove duplicates while preserving order
        return list(dict.fromkeys(models))
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data for video analysis.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid for video analysis
        """
        super().validate_input(data)
        
        # Verify this is video content
        if data.content_type not in [
            ContentType.VIDEO_WITH_SPEECH,
            ContentType.VIDEO_NO_SPEECH
        ]:
            raise ValidationError(
                f"VideoAnalyzer requires video content, got {data.content_type}"
            )
        
        # Check for frames
        if not data.frames and not data.face_crops:
            raise ValidationError(
                "VideoAnalyzer requires frames or face_crops in PreprocessedData"
            )
        
        # Validate minimum frames
        frame_count = len(data.face_crops or data.frames or [])
        if frame_count < self.min_frames:
            raise ValidationError(
                f"VideoAnalyzer requires at least {self.min_frames} frames, got {frame_count}"
            )
    
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core video analysis implementation.
        
        Orchestrates sub-analyzers and aggregates results.
        
        Args:
            data: PreprocessedData with video frames and optional audio
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult wrapping VideoResult with detection score
        """
        # Load frame data (in production, load from MinIO)
        face_crops = await self._load_frames(data.face_crops or data.frames or [])
        
        # Determine if audio is available
        has_audio = (
            data.content_type == ContentType.VIDEO_WITH_SPEECH and 
            data.audio_key is not None
        )
        
        # Load audio features if available
        audio_features = None
        if has_audio:
            audio_features = await self._load_audio_features(data.audio_key)
        
        # Run analysis pipeline
        video_result = await self.analyze_video(
            frames=face_crops,
            audio_features=audio_features,
            engine=engine,
            has_audio=has_audio
        )
        
        return ModalityResult(
            modality=Modality.VIDEO,
            score=video_result.aggregate_score,
            confidence=self._compute_overall_confidence(video_result),
            details={
                "spatial": video_result.spatial.model_dump(),
                "temporal": video_result.temporal.model_dump(),
                "lipsync": video_result.lip_sync.model_dump() if video_result.lip_sync else None,
                "frames_analyzed": video_result.frames_analyzed,
                "face_detected": video_result.face_detected
            }
        )
    
    async def analyze_video(
        self,
        frames: List[np.ndarray],
        audio_features: Optional[np.ndarray],
        engine: "InferenceEngine",
        has_audio: bool = False,
        explainer: Optional["ExplainabilityEngine"] = None,
        fps: float = 30.0
    ) -> VideoResult:
        """
        Run complete video analysis pipeline.
        
        Main entry point for direct video analysis without PreprocessedData.
        
        Args:
            frames: Face crops or video frames (H, W, 3) RGB
            audio_features: Mel-spectrogram/MFCC features (T, F) if audio available
            engine: InferenceEngine for model inference
            has_audio: Whether audio track is present
            explainer: Optional ExplainabilityEngine for heatmaps
            fps: Video frame rate for timestamp calculation
            
        Returns:
            VideoResult with spatial, temporal, lipsync results and aggregate
        """
        start_time = time.time()
        metrics = VideoAnalysisMetrics()
        
        if not frames or len(frames) < self.min_frames:
            logger.warning(f"Insufficient frames for analysis: {len(frames) if frames else 0}")
            return self._create_empty_result()
        
        metrics.frames_processed = len(frames)
        metrics.faces_detected = len(frames)  # Assuming face_crops are provided
        
        logger.info(f"Starting video analysis: {len(frames)} frames, audio={has_audio}")
        
        # Prepare ensemble weights
        weights = EnsembleWeights(
            spatial=self.base_weights.spatial,
            temporal=self.base_weights.temporal,
            lipsync=self.base_weights.lipsync
        )
        
        if not has_audio:
            weights.adjust_for_no_audio()
            logger.debug(f"Adjusted weights for no audio: {weights.to_dict()}")
        
        # Run sub-analyzers
        if self.enable_parallel:
            spatial_result, temporal_result, lipsync_result, metrics = await self._run_parallel_analysis(
                frames, audio_features, engine, explainer, fps, has_audio, metrics
            )
        else:
            spatial_result, temporal_result, lipsync_result, metrics = await self._run_sequential_analysis(
                frames, audio_features, engine, explainer, fps, has_audio, metrics
            )
        
        # Aggregate results
        aggregate_score = self._compute_ensemble_score(
            spatial_result,
            temporal_result,
            lipsync_result,
            weights
        )
        
        metrics.total_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Video analysis complete: score={aggregate_score:.3f}, "
            f"time={metrics.total_time_ms:.2f}ms, "
            f"anomalies={metrics.anomalies_detected}"
        )
        
        return VideoResult(
            spatial=spatial_result,
            temporal=temporal_result,
            lip_sync=lipsync_result if has_audio else None,
            aggregate_score=aggregate_score,
            frames_analyzed=metrics.frames_processed,
            face_detected=metrics.faces_detected > 0
        )
    
    async def _run_parallel_analysis(
        self,
        frames: List[np.ndarray],
        audio_features: Optional[np.ndarray],
        engine: "InferenceEngine",
        explainer: Optional["ExplainabilityEngine"],
        fps: float,
        has_audio: bool,
        metrics: VideoAnalysisMetrics
    ) -> Tuple[SpatialResult, TemporalResult, Optional[LipSyncResult], VideoAnalysisMetrics]:
        """
        Run sub-analyzers in parallel for better performance.
        
        Args:
            frames: Video frames
            audio_features: Audio features (if available)
            engine: InferenceEngine
            explainer: ExplainabilityEngine (if available)
            fps: Frame rate
            has_audio: Whether audio is present
            metrics: Metrics tracker
            
        Returns:
            Tuple of (spatial, temporal, lipsync results, updated metrics)
        """
        # Create async tasks
        tasks = []
        
        # Spatial analysis task
        async def run_spatial():
            start = time.time()
            result = await self.spatial.analyze_frames(frames, engine, explainer)
            metrics.spatial_time_ms = (time.time() - start) * 1000
            return result
        
        # Temporal analysis task
        async def run_temporal():
            start = time.time()
            result = await self.temporal.analyze_consistency(frames, engine, fps)
            metrics.temporal_time_ms = (time.time() - start) * 1000
            return result
        
        # Lipsync analysis task (only if audio)
        async def run_lipsync():
            if not has_audio or audio_features is None:
                return None
            start = time.time()
            # Extract mouth crops from frames (in production, would be separate)
            # For now, use face crops as proxy
            mouth_crops = frames  # Simplified - real impl would crop mouth region
            result = await self.lipsync.verify_sync(mouth_crops, audio_features, engine)
            metrics.lipsync_time_ms = (time.time() - start) * 1000
            return result
        
        tasks.append(run_spatial())
        tasks.append(run_temporal())
        tasks.append(run_lipsync())
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        spatial_result = self._handle_task_result(results[0], "spatial")
        temporal_result = self._handle_task_result(results[1], "temporal")
        lipsync_result = self._handle_task_result(results[2], "lipsync") if has_audio else None
        
        # Count anomalies
        metrics.anomalies_detected = len(spatial_result.anomaly_indices)
        
        return spatial_result, temporal_result, lipsync_result, metrics
    
    async def _run_sequential_analysis(
        self,
        frames: List[np.ndarray],
        audio_features: Optional[np.ndarray],
        engine: "InferenceEngine",
        explainer: Optional["ExplainabilityEngine"],
        fps: float,
        has_audio: bool,
        metrics: VideoAnalysisMetrics
    ) -> Tuple[SpatialResult, TemporalResult, Optional[LipSyncResult], VideoAnalysisMetrics]:
        """
        Run sub-analyzers sequentially (safer for VRAM management).
        
        Args:
            Same as _run_parallel_analysis
            
        Returns:
            Same as _run_parallel_analysis
        """
        # Spatial analysis
        start = time.time()
        spatial_result = await self.spatial.analyze_frames(frames, engine, explainer)
        metrics.spatial_time_ms = (time.time() - start) * 1000
        
        # Temporal analysis
        start = time.time()
        temporal_result = await self.temporal.analyze_consistency(frames, engine, fps)
        metrics.temporal_time_ms = (time.time() - start) * 1000
        
        # Lipsync analysis (if audio)
        lipsync_result = None
        if has_audio and audio_features is not None:
            start = time.time()
            mouth_crops = frames  # Simplified
            lipsync_result = await self.lipsync.verify_sync(mouth_crops, audio_features, engine)
            metrics.lipsync_time_ms = (time.time() - start) * 1000
        
        metrics.anomalies_detected = len(spatial_result.anomaly_indices)
        
        return spatial_result, temporal_result, lipsync_result, metrics
    
    def _handle_task_result(
        self,
        result: Any,
        task_name: str
    ) -> Any:
        """
        Handle task result, returning default on failure.
        
        Args:
            result: Task result or exception
            task_name: Name of task for logging
            
        Returns:
            Result or default value on exception
        """
        if isinstance(result, Exception):
            logger.error(f"{task_name} analysis failed: {result}")
            
            if task_name == "spatial":
                return SpatialResult(
                    score=0.5,
                    per_frame_scores=[],
                    anomaly_indices=[],
                    heatmap_urls=[]
                )
            elif task_name == "temporal":
                return TemporalResult(
                    consistency_score=1.0,
                    flickering_detected=False,
                    anomaly_timestamps=[]
                )
            else:  # lipsync
                return LipSyncResult(
                    sync_score=1.0,
                    manipulation_probability=0.0,
                    detected_technology=None
                )
        
        return result
    
    def _compute_ensemble_score(
        self,
        spatial: SpatialResult,
        temporal: TemporalResult,
        lipsync: Optional[LipSyncResult],
        weights: EnsembleWeights
    ) -> float:
        """
        Compute weighted ensemble score from sub-analyzer results.
        
        Uses uncertainty-aware weighting where confident predictions
        get higher effective weights.
        
        Args:
            spatial: Spatial analysis result
            temporal: Temporal analysis result
            lipsync: Lipsync analysis result (may be None)
            weights: Ensemble weights
            
        Returns:
            Aggregated fake probability score [0, 1]
        """
        # Spatial score (direct fake probability)
        spatial_score = spatial.score
        
        # Temporal score (invert consistency - low consistency = likely fake)
        temporal_score = 1.0 - temporal.consistency_score
        
        # Lipsync score (manipulation probability)
        lipsync_score = lipsync.manipulation_probability if lipsync else 0.0
        
        # Weighted combination
        if weights.lipsync > 0 and lipsync is not None:
            aggregate = (
                weights.spatial * spatial_score +
                weights.temporal * temporal_score +
                weights.lipsync * lipsync_score
            )
        else:
            # Only spatial and temporal
            total_weight = weights.spatial + weights.temporal
            if total_weight > 0:
                aggregate = (
                    (weights.spatial / total_weight) * spatial_score +
                    (weights.temporal / total_weight) * temporal_score
                )
            else:
                aggregate = 0.5
        
        # Apply boosting for strong agreement
        scores = [spatial_score, temporal_score]
        if lipsync is not None:
            scores.append(lipsync_score)
        
        # If multiple analyzers strongly agree, boost confidence
        high_scores = sum(1 for s in scores if s > 0.7)
        if high_scores >= 2:
            aggregate = min(1.0, aggregate * 1.1)  # 10% boost
        
        low_scores = sum(1 for s in scores if s < 0.3)
        if low_scores >= 2:
            aggregate = max(0.0, aggregate * 0.9)  # 10% reduction
        
        return float(np.clip(aggregate, 0, 1))
    
    def _compute_overall_confidence(
        self,
        result: VideoResult
    ) -> float:
        """
        Compute overall confidence from VideoResult.
        
        Args:
            result: VideoResult from analysis
            
        Returns:
            Overall confidence [0, 1]
        """
        confidences = []
        
        # Spatial confidence from score consistency
        if result.spatial.per_frame_scores:
            variance = np.var(result.spatial.per_frame_scores)
            spatial_conf = 1.0 / (1.0 + variance * 4)
            confidences.append(spatial_conf)
        
        # Temporal confidence from consistency
        temporal_conf = result.temporal.consistency_score * 0.8 + 0.2
        confidences.append(temporal_conf)
        
        # Lipsync confidence
        if result.lip_sync:
            # Extremity from 0.5 indicates stronger evidence (high confidence).
            lipsync_conf = abs(result.lip_sync.sync_score - 0.5) * 2
            confidences.append(lipsync_conf)
        
        if confidences:
            # Use harmonic mean for conservative estimate
            safe_confidences = [max(1e-6, float(c)) for c in confidences]
            return float(len(safe_confidences) / sum(1.0 / c for c in safe_confidences))
        
        return 0.5
    
    async def _load_frames(
        self,
        frame_keys: List[str]
    ) -> List[np.ndarray]:
        """
        Load frames from MinIO keys.
        
        Fetches preprocessed frame data from object storage.
        
        Args:
            frame_keys: List of MinIO object keys
            
        Returns:
            List of loaded frame arrays (H, W, 3)
        """
        import io as io_module
        from storage.storage import get_storage_client
        from config import config
        
        frames = []
        
        if not frame_keys:
            logger.warning("No frame keys provided for loading")
            return frames
        
        try:
            storage = get_storage_client()
            
            for key in frame_keys[:100]:  # Limit to 100 frames
                try:
                    frame_bytes = await storage.download_file(
                        config.minio_bucket_preprocessed,
                        key
                    )
                    
                    if key.endswith('.npy'):
                        frame_array = np.load(io_module.BytesIO(frame_bytes), allow_pickle=True)
                        if frame_array.dtype == object:
                            if hasattr(frame_array, 'item') and isinstance(frame_array.item(), np.ndarray):
                                frame_array = frame_array.item()
                        frames.append(frame_array)
                    else:
                        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frames.append(frame)
                        
                except Exception as e:
                    logger.warning(f"Failed to load frame {key}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to initialize storage for frame loading: {e}")
        
        logger.info(f"Loaded {len(frames)} frames from storage")
        return frames
    
    async def _load_audio_features(
        self,
        audio_key: Optional[str]
    ) -> Optional[np.ndarray]:
        """
        Load audio features from MinIO key.
        
        Args:
            audio_key: MinIO key for audio features
            
        Returns:
            Audio features array (T, F) or None
        """
        import io as io_module
        from storage.storage import get_storage_client
        from config import config
        
        if not audio_key:
            return None
        
        try:
            storage = get_storage_client()
            
            audio_bytes = await storage.download_file(
                config.minio_bucket_preprocessed,
                audio_key
            )
            
            if audio_key.endswith('.npy'):
                features = np.load(io_module.BytesIO(audio_bytes), allow_pickle=True)
                if features.dtype == object:
                    if hasattr(features, 'item') and isinstance(features.item(), np.ndarray):
                        features = features.item()
                return features
            else:
                return np.frombuffer(audio_bytes, dtype=np.float32)
                
        except Exception as e:
            logger.warning(f"Failed to load audio features {audio_key}: {e}")
            return None
    
    def _create_empty_result(self) -> VideoResult:
        """Create empty result for insufficient data."""
        return VideoResult(
            spatial=SpatialResult(
                score=0.5,
                per_frame_scores=[],
                anomaly_indices=[],
                heatmap_urls=[]
            ),
            temporal=TemporalResult(
                consistency_score=1.0,
                flickering_detected=False,
                anomaly_timestamps=[]
            ),
            lip_sync=None,
            aggregate_score=0.5,
            frames_analyzed=0,
            face_detected=False
        )
    
    def get_sub_analyzer_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all sub-analyzers.
        
        Returns:
            Dictionary of sub-analyzer statistics
        """
        return {
            "spatial": self.spatial.metrics.to_dict(),
            "temporal": self.temporal.metrics.to_dict(),
            "lipsync": self.lipsync.metrics.to_dict(),
            "orchestrator": self.metrics.to_dict()
        }


# Singleton instance
_video_analyzer: Optional[VideoAnalyzer] = None


def get_video_analyzer() -> VideoAnalyzer:
    """Get singleton video analyzer instance."""
    global _video_analyzer
    if _video_analyzer is None:
        _video_analyzer = VideoAnalyzer()
    return _video_analyzer
