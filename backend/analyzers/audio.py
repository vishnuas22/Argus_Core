"""
Argus Core - Audio Analyzer
============================
Audio deepfake detection using Purdue-M2 architecture.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/audio.py

SOTA Algorithms:
- Model: Purdue-M2 AI-Synthesized Voice Generalization (AAAI 2025)
- Features: Mel-spectrogram (80 mel bands), MFCC (13 coefficients + deltas), raw waveform
- Detection: Vocoder artifacts, spectral inconsistencies, voice consistency

Artifacts Detected:
- Vocoder artifacts (phase discontinuities)
- Unnatural harmonics
- Bandwidth limitations
- Background noise inconsistencies
- Spectral envelope anomalies

Integration:
- Imports: core/engine.py, processing/transform.py
- Inputs: audio_data: np.ndarray (waveform)
- Outputs: AudioResult

Target Hardware: RTX 3050 (4GB VRAM) with optimized inference
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
import time

from analyzers.base import BaseAnalyzer
from detectors.wav2vec2_detector import Wav2Vec2AudioDetector
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType, AudioResult
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError, InferenceError
from models.model_init import ensure_models_for_analyzer, is_model_ready

logger = get_logger(__name__)

# Disable librosa caching (set once at module level)
os.environ['LIBROSA_CACHE_LEVEL'] = '0'

# Iteration 1: SOTA audio detector ensemble (lazy import)
# Iteration 4: added ECAPATDNNAudioDetector for further diversity
# Iteration 6: added CDPMambaDetector for state-space audio analysis
try:
    from detectors import (
        AASIST3AudioDetector,
        Wav2Vec2XLSRMoELoRADetector,
        ECAPATDNNAudioDetector,
        CDPMambaDetector,
        combine_detector_results,
    )
    _SOTA_AUDIO_AVAILABLE = True
except ImportError as _e:
    _SOTA_AUDIO_AVAILABLE = False
    logger.warning("SOTA audio detectors unavailable: %s", _e)

if TYPE_CHECKING:
    from core.engine import InferenceEngine


# Default audio sample rate for analysis
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class SpectralFeatures:
    """
    Spectral analysis features for synthetic voice detection.
    
    Captures frequency-domain characteristics that differ
    between natural and synthesized speech.
    """
    spectral_centroid: float = 0.0  # Center of mass of spectrum
    spectral_bandwidth: float = 0.0  # Spread of spectrum
    spectral_rolloff: float = 0.0  # Frequency below which 85% energy
    spectral_flatness: float = 0.0  # How tone-like vs noise-like
    zero_crossing_rate: float = 0.0  # Rate of sign changes
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "spectral_centroid": round(self.spectral_centroid, 4),
            "spectral_bandwidth": round(self.spectral_bandwidth, 4),
            "spectral_rolloff": round(self.spectral_rolloff, 4),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4)
        }


@dataclass
class VocoderArtifactFeatures:
    """
    Features for detecting vocoder artifacts.
    
    Neural vocoders (WaveNet, HiFi-GAN, etc.) leave characteristic
    artifacts in the spectral domain.
    """
    artifact_score: float = 0.0  # Overall vocoder artifact score
    phase_discontinuity: float = 0.0  # Phase coherence measure
    harmonic_distortion: float = 0.0  # Harmonic structure anomalies
    bandwidth_limitation: float = 0.0  # Unnatural frequency cutoffs
    periodic_artifacts: bool = False  # Periodic patterns detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_score": round(self.artifact_score, 4),
            "phase_discontinuity": round(self.phase_discontinuity, 4),
            "harmonic_distortion": round(self.harmonic_distortion, 4),
            "bandwidth_limitation": round(self.bandwidth_limitation, 4),
            "periodic_artifacts": self.periodic_artifacts
        }


@dataclass
class VoiceConsistencyFeatures:
    """
    Voice consistency analysis features.
    
    Analyzes consistency of voice characteristics across
    the audio, which may vary unnaturally in synthesized speech.
    """
    pitch_variance: float = 0.0  # F0 variance
    pitch_consistency: float = 0.0  # How consistent pitch is
    energy_variance: float = 0.0  # Energy level variance
    formant_consistency: float = 0.0  # Formant stability
    speaking_rate_variance: float = 0.0  # Rhythm consistency
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "pitch_variance": round(self.pitch_variance, 4),
            "pitch_consistency": round(self.pitch_consistency, 4),
            "energy_variance": round(self.energy_variance, 4),
            "formant_consistency": round(self.formant_consistency, 4),
            "speaking_rate_variance": round(self.speaking_rate_variance, 4)
        }


@dataclass
class AudioAnalysisDetails:
    """
    Detailed audio analysis results.
    
    Contains all intermediate analysis features for transparency.
    """
    # Neural detector scores
    wav2vec2_antispoof_score: float = 0.0  # Primary: Wav2Vec2 XLSR ONNX (4.01% EER)
    wav2vec2_score: float = 0.0  # Wav2Vec2 anti-spoofing (DeepFense-style)
    
    # Legacy fields kept for backward compatibility
    aasist_score: float = 0.5
    purdue_m2_score: float = 0.0
    rawnet_score: float = 0.0
    
    # Classification (real/VC/TTS)
    classification: str = "unknown"
    classification_scores: Dict[str, float] = field(default_factory=dict)
    
    # Feature-based scores
    spectral_features: Optional[SpectralFeatures] = None
    frequency_anomaly_score: float = 0.0
    vocoder_artifacts: Optional[VocoderArtifactFeatures] = None
    voice_consistency: Optional[VoiceConsistencyFeatures] = None
    
    # Metadata
    audio_duration_seconds: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    segments_analyzed: int = 0
    primary_detector: str = "wav2vec2_antispoof"  # Which detector was used as primary
    # True iff at least one neural detector produced a non-default (non-0.5)
    # score. When False, downstream fusion should treat the audio modality
    # as low-confidence because no real ML inference contributed to the
    # result. Set by AudioAnalyzer.analyze after all detector calls return.
    any_neural_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ModalityResult details."""
        return {
            "wav2vec2_antispoof_score": round(self.wav2vec2_antispoof_score, 4),
            "aasist_score": round(self.aasist_score, 4),
            "purdue_m2_score": round(self.purdue_m2_score, 4),
            "rawnet_score": round(self.rawnet_score, 4),
            "wav2vec2_score": round(self.wav2vec2_score, 4),
            "classification": self.classification,
            "classification_scores": {k: round(v, 4) for k, v in self.classification_scores.items()},
            "spectral_features": self.spectral_features.to_dict() if self.spectral_features else None,
            "frequency_anomaly_score": round(self.frequency_anomaly_score, 4),
            "vocoder_artifacts": self.vocoder_artifacts.to_dict() if self.vocoder_artifacts else None,
            "voice_consistency": self.voice_consistency.to_dict() if self.voice_consistency else None,
            "audio_duration_seconds": round(self.audio_duration_seconds, 2),
            "sample_rate": self.sample_rate,
            "segments_analyzed": self.segments_analyzed,
            "primary_detector": self.primary_detector,
            "any_neural_available": self.any_neural_available,
        }


class AudioAnalyzer(BaseAnalyzer):
    """
    Synthetic voice and audio deepfake detection.
    
    Multi-stage detection pipeline:
    1. Audio preprocessing (resampling, normalization)
    2. Feature extraction (mel-spectrogram, MFCC, spectral features)
    3. Neural detection (Purdue-M2, optional RawNet)
    4. Vocoder artifact analysis
    5. Voice consistency analysis
    6. Aggregated scoring
    
    Supported Detection:
    - TTS-generated speech (Tacotron, FastSpeech, VITS)
    - Voice conversion (So-VITS, RVC, etc.)
    - Voice cloning (Vall-E, etc.)
    - Audio splicing and editing
    
    Usage:
        analyzer = AudioAnalyzer()
        result = await analyzer.analyze(preprocessed_data, engine)
    
    Or direct analysis:
        audio_result = await analyzer.analyze_audio(waveform, sample_rate, engine)
    """
    
    def __init__(
        self,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
        segment_length_seconds: float = 3.0,
        segment_overlap: float = 0.5,
        n_mels: int = 80,
        n_mfcc: int = 13
    ):
        """
        Initialize audio analyzer.
        
        Args:
            target_sample_rate: Target sample rate for analysis (16kHz standard)
            segment_length_seconds: Length of analysis segments
            segment_overlap: Overlap ratio between segments
            n_mels: Number of mel frequency bands
            n_mfcc: Number of MFCC coefficients
        """
        super().__init__(
            analyzer_name="AudioAnalyzer",
            supported_modalities=[Modality.AUDIO],
            version="1.0.0"
        )
        
        self.target_sample_rate = target_sample_rate
        self.segment_length_seconds = segment_length_seconds
        self.segment_overlap = segment_overlap
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Computed segment parameters
        self.segment_samples = int(segment_length_seconds * target_sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - segment_overlap))
        
        # Wav2Vec2 detector (DeepFense-style)
        self.wav2vec2_detector: Optional[Wav2Vec2AudioDetector] = None

        # ===========================================================
        # Iteration 1: SOTA audio detector ensemble
        # Iteration 4: added ECAPA-TDNN as 3rd detector (embedding-distance)
        # ===========================================================
        # AASIST3 (ASVspoof 2024) + Wav2Vec2-XLS-R-300M + MoE-LoRA
        # (arxiv 2025 SOTA: 0.28% EER on ASVspoof 2019 LA) + ECAPA-TDNN
        # (INTERSPEECH 2020, embedding-distance-based, MIT license).
        # Lazy-initialized when first needed; weights auto-downweighted
        # by DiversityEnsemble if a detector's adapter is missing or
        # ECAPA's reference centroid is not built.
        self._sota_audio_detectors = None
        self._sota_audio_prior_weights = [0.94, 0.97, 0.85, 0.88]  # AASIST3, XLS-R, ECAPA, CDP-Mamba

        # Weight configuration for aggregation
        # Primary: Wav2Vec2 Large XLSR fine-tuned on ASVspoof2019 (4.01% EER, INT8 ONNX).
        # Support: Wav2Vec2 PyTorch, vocoder artifacts, voice consistency.
        # NOTE: purdue_m2 and aasist_antispoof removed — model files unavailable.
        self.weights = {
            "wav2vec2_antispoof": 0.55,  # Primary: Wav2Vec2 XLSR antispoofing ONNX
            "wav2vec2": 0.20,            # Wav2Vec2 anti-spoofing (DeepFense-style)
            "vocoder_artifacts": 0.15,   # Vocoder artifact analysis
            "voice_consistency": 0.10    # Voice consistency
        }
        
        # Thresholds
        self.min_audio_duration = 1.0  # Minimum seconds for analysis
        
        logger.info(
            f"AudioAnalyzer initialized: sr={target_sample_rate}, "
            f"segment={segment_length_seconds}s, n_mels={n_mels}"
        )
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for audio analysis.
        
        Returns:
            List of model registry keys
        """
        return [
            "wav2vec2_antispoof",  # Primary: Wav2Vec2 XLSR ONNX (4.01% EER)
            "wav2vec2_base"        # Feature extraction for voice consistency
        ]
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data for audio analysis.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid for audio analysis
        """
        super().validate_input(data)
        
        # Verify this has audio content
        if data.content_type not in [
            ContentType.VIDEO_WITH_SPEECH,
            ContentType.AUDIO_ONLY
        ]:
            raise ValidationError(
                f"AudioAnalyzer requires audio content, got {data.content_type}"
            )
        
        # Check for audio key
        if not data.audio_key:
            raise ValidationError(
                "AudioAnalyzer requires audio_key in PreprocessedData"
            )
    
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core audio analysis implementation.
        
        Args:
            data: PreprocessedData with audio key
            engine: InferenceEngine for model inference
            
        Returns:
            ModalityResult with detection score and details
        """
        # Load audio waveform (in production, load from MinIO)
        waveform, sample_rate = await self._load_audio(data.audio_key)
        
        if waveform is None or len(waveform) == 0:
            logger.warning("No audio data available")
            return ModalityResult(
                modality=Modality.AUDIO,
                score=0.5,
                confidence=0.3,
                details={"error": "No audio data available"}
            )
        
        # Run analysis
        _audio_start = time.time()
        audio_result, details = await self.analyze_audio(waveform, sample_rate, engine, data.analysis_id)

        # ===========================================================
        # Iteration 7: Prometheus metrics recording
        # ===========================================================
        try:
            from observability import get_default_metrics
            _latency_s = time.time() - _audio_start
            _verdict = "fake" if audio_result.synthetic_probability >= 0.5 else "real"
            _metrics = get_default_metrics()
            _metrics.record_inference("audio", _verdict, _latency_s)
            if getattr(config, "enable_adversarial_defenses", False) and \
               getattr(config, "enable_rps", False):
                _metrics.record_adversarial_flag("audio", "rps")
        except Exception as _e:
            logger.debug("Audio metrics recording failed: %s", _e)

        return ModalityResult(
            modality=Modality.AUDIO,
            score=audio_result.synthetic_probability,
            confidence=self._compute_confidence(details),
            details=details.to_dict()
        )
    
    async def analyze_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        engine: "InferenceEngine",
        analysis_id: str = "",
    ) -> Tuple[AudioResult, AudioAnalysisDetails]:
        """
        Analyze audio waveform for synthetic generation.
        
        Main entry point for direct audio analysis.
        
        Args:
            audio_data: Raw waveform (1D numpy array)
            sample_rate: Audio sample rate
            engine: InferenceEngine
            analysis_id: Optional analysis ID for embedding buffer provenance
            
        Returns:
            Tuple of (AudioResult, AudioAnalysisDetails)
        """
        start_time = time.time()
        details = AudioAnalysisDetails()
        
        # Validate duration
        duration = len(audio_data) / sample_rate
        details.audio_duration_seconds = duration
        details.sample_rate = sample_rate
        
        if duration < self.min_audio_duration:
            logger.warning(f"Audio too short: {duration:.2f}s < {self.min_audio_duration}s")
            return self._create_default_result(), details
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        
        # Normalize audio
        audio_data = self._normalize_audio(audio_data)
        
        # Extract features
        mel_spectrogram = self.extract_mel_spectrogram(audio_data, sample_rate)
        mfcc_features = self.extract_mfcc(audio_data, sample_rate)
        
        # Segment audio for analysis
        segments = self._segment_audio(audio_data)
        details.segments_analyzed = len(segments)
        
        logger.debug(f"Analyzing {len(segments)} audio segments ({duration:.2f}s)")
        
        # Ensure audio models are available
        model_status = ensure_models_for_analyzer("audio", self.get_required_models())
        wav2vec2_antispoof_available = model_status.get("wav2vec2_antispoof", False)
        
        if not wav2vec2_antispoof_available:
            logger.warning(
                "No audio neural models available — analysis will use heuristic "
                "features only (vocoder artifacts, voice consistency). Results "
                "have limited reliability and will be dampened toward neutral."
            )
        
        # 1. Primary: Wav2Vec2 Large XLSR antispoofing (ASVspoof2019, 4.01% EER)
        wav2vec2_antispoof_score = 0.5
        use_antispoof = False
        if wav2vec2_antispoof_available and is_model_ready("wav2vec2_antispoof"):
            try:
                wav2vec2_antispoof_score = await self._run_wav2vec2_antispoof(audio_data, engine)
                details.wav2vec2_antispoof_score = wav2vec2_antispoof_score
                details.primary_detector = "wav2vec2_antispoof"
                use_antispoof = True
                logger.info(f"Wav2Vec2 XLSR antispoofing: spoof_prob={wav2vec2_antispoof_score:.4f}")
            except Exception as e:
                logger.warning("RIVP: wav2vec2_antispoof FAILED: %s — score remains 0.5", e)
        
        # purdue_m2 and aasist_antispoof removed — model files unavailable.
        aasist_score = 0.5
        
        # 2. Wav2Vec2 anti-spoofing detector (DeepFense-style PyTorch)
        wav2vec2_score = 0.5
        try:
            if self.wav2vec2_detector is None:
                self.wav2vec2_detector = Wav2Vec2AudioDetector(device="cpu")
            wav2vec2_result = await self.wav2vec2_detector.detect(audio_data, sample_rate)
            if wav2vec2_result.error is None:
                wav2vec2_score = wav2vec2_result.score
                details.wav2vec2_score = wav2vec2_score
                logger.info(f"Wav2Vec2 PyTorch: spoof_prob={wav2vec2_score:.4f}")
        except Exception as e:
            logger.warning("RIVP: wav2vec2 PyTorch FAILED (non-critical): %s — score remains 0.5", e)

        # 5. Vocoder artifact analysis
        vocoder_artifacts = self.detect_vocoder_artifacts(mel_spectrogram, audio_data)
        details.vocoder_artifacts = vocoder_artifacts
        
        # 6. Voice consistency analysis
        voice_consistency = self.analyze_voice_consistency(segments, sample_rate)
        details.voice_consistency = voice_consistency
        
        # 7. Spectral feature analysis
        spectral_features = self.extract_spectral_features(audio_data, sample_rate)
        details.spectral_features = spectral_features
        
        # 8. Aggregate scores
        any_neural_available = any(
            s != 0.5 for s in [
                wav2vec2_antispoof_score,
                wav2vec2_score
            ] if s is not None
        )
        # Persist the neural-availability flag on `details` so downstream
        # confidence computation and the orchestrator's fusion step can
        # detect when no real ML inference contributed to the score.
        # The user's protocol explicitly requires that heuristic-only
        # results be marked as low-confidence so they cannot dominate
        # the final verdict.
        details.any_neural_available = any_neural_available
        synthetic_probability = self._compute_aggregate_score(
            wav2vec2_antispoof_score=wav2vec2_antispoof_score,
            vocoder_artifacts=vocoder_artifacts,
            voice_consistency=voice_consistency,
            use_wav2vec2_antispoof=use_antispoof,
            wav2vec2_score=wav2vec2_score,
            any_neural_available=any_neural_available,
        )

        # ===========================================================
        # Iteration 1 — SOTA audio detector ensemble integration
        # (additive, strict-compat)
        # ===========================================================
        # When config.enable_sota_detectors is True, run AASIST3 and
        # Wav2Vec2-XLS-R + MoE-LoRA on the same waveform and blend
        # their outputs with the existing fusion.
        if (
            getattr(config, "enable_sota_detectors", False)
            and _SOTA_AUDIO_AVAILABLE
            and duration >= self.min_audio_duration
        ):
            try:
                sota_audio_score = await self._run_sota_audio_ensemble(
                    audio_data, sample_rate
                )
                if sota_audio_score is not None:
                    # Blend: 65% SOTA ensemble, 35% legacy fusion.
                    # The SOTA detectors have higher benchmark EER, so they
                    # get the majority weight; the legacy fusion still
                    # contributes vocoder + voice-consistency signals.
                    synthetic_probability = float(
                        0.65 * sota_audio_score + 0.35 * synthetic_probability
                    )
                    logger.info(
                        "SOTA audio ensemble integrated: sota=%.4f, blended=%.4f",
                        sota_audio_score, synthetic_probability,
                    )
            except Exception as e:
                logger.warning("SOTA audio ensemble failed (non-fatal): %s", e)

        # ===========================================================
        # Iteration 2 — Calibration + Conformal + Adversarial flag
        # (additive, strict-compat)
        # ===========================================================
        try:
            from core.post_processing import apply_post_processing
            _synth = np.array([
                synthetic_probability,
                details.wav2vec2_antispoof_score,
                details.wav2vec2_score,
                details.frequency_anomaly_score,
                details.vocoder_artifacts.artifact_score if details.vocoder_artifacts else 0.5,
                details.voice_consistency.pitch_consistency if details.voice_consistency else 0.5,
                details.voice_consistency.formant_consistency if details.voice_consistency else 0.5,
            ], dtype=np.float64)
            pp = apply_post_processing(
                score=synthetic_probability,
                confidence=self._compute_confidence(details),
                embedding=_synth,
                modality="audio",
                analysis_id=analysis_id,
            )
            if pp.calibrated_score != pp.original_score:
                logger.info(
                    "Audio temperature scaling: %.4f -> %.4f (T=%.4f)",
                    pp.original_score, pp.calibrated_score, pp.temperature,
                )
                synthetic_probability = pp.calibrated_score
        except Exception as e:
            logger.debug("Audio post-processing failed (non-fatal): %s", e)

        # 6. Determine if vocoder artifacts detected
        vocoder_detected = vocoder_artifacts.artifact_score > 0.5
        
        # 7. Compute voice consistency score (invert - low consistency = suspicious)
        voice_consistency_score = voice_consistency.pitch_consistency * 0.5 + \
                                  voice_consistency.formant_consistency * 0.5
        
        # 8. Derive frequency anomaly score from spectral features
        frequency_anomaly_score = 0.0
        if spectral_features is not None:
            flatness = spectral_features.spectral_flatness
            zcr = spectral_features.zero_crossing_rate
            frequency_anomaly_score = float(np.clip(
                (1.0 - flatness) * 0.5 + zcr * 0.5, 0, 1
            ))
        details.frequency_anomaly_score = frequency_anomaly_score
        
        inference_time = (time.time() - start_time) * 1000
        confidence = self._compute_confidence(details)
        self._metrics.record_analysis(True, inference_time, confidence)
        
        logger.info(
            f"Audio analysis complete: synthetic_prob={synthetic_probability:.3f}, "
            f"vocoder={vocoder_detected}, time={inference_time:.2f}ms"
        )
        
        audio_result = AudioResult(
            synthetic_probability=synthetic_probability,
            vocoder_artifacts_detected=vocoder_detected,
            voice_consistency_score=voice_consistency_score,
            spectrogram_url=None,  # Would be set after uploading to MinIO
            frequency_anomaly_score=frequency_anomaly_score,
            aasist_score=aasist_score if aasist_score != 0.5 else None,
        )

        return audio_result, details

    # ------------------------------------------------------------------
    # Iteration 1: SOTA audio detector ensemble helper
    # ------------------------------------------------------------------
    async def _run_sota_audio_ensemble(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> Optional[float]:
        """
        Run AASIST3 and Wav2Vec2-XLS-R + MoE-LoRA on the audio waveform
        and fuse their outputs via DiversityEnsemble.

        Args:
            audio_data: 1D float32 waveform at the analyzer's target sr.
            sample_rate: Sample rate of audio_data.

        Returns:
            Fused spoof probability in [0, 1], or None on failure.
        """
        if self._sota_audio_detectors is None:
            detectors = [
                AASIST3AudioDetector(),
                Wav2Vec2XLSRMoELoRADetector(),
            ]
            # Iteration 4: add ECAPA-TDNN if enabled (gated because it
            # needs a reference centroid that operators must build).
            if getattr(config, "enable_ecapa", True):
                try:
                    detectors.append(ECAPATDNNAudioDetector())
                except Exception as e:
                    logger.warning("ECAPA-TDNN init failed (non-fatal): %s", e)
            # Iteration 6: add CDP-Mamba for state-space analysis
            try:
                detectors.append(CDPMambaDetector())
            except Exception as e:
                logger.warning("CDP-Mamba init failed (non-fatal): %s", e)
            self._sota_audio_detectors = detectors

        detector_results = []
        for det in self._sota_audio_detectors:
            try:
                r = await det.detect(audio_data, sample_rate=sample_rate)
                detector_results.append(r)
            except Exception as e:
                logger.warning("SOTA audio detector %s failed: %s", det.name, e)

        if not detector_results:
            return None

        fused = combine_detector_results(
            detector_results,
            prior_weights=self._sota_audio_prior_weights[:len(detector_results)],
        )
        if fused.error and "all_members_failed" in fused.error:
            return None
        return float(fused.score)

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int = DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features.
        
        Args:
            audio: Raw waveform (1D array)
            sr: Sample rate
            
        Returns:
            Mel-spectrogram (T, n_mels)
        """
        try:
            import librosa
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512,
                power=2.0
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (T, n_mels)
            return mel_spec_db.T.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"librosa not available or failed ({e}), using simplified mel extraction")
            return self._simple_mel_spectrogram(audio, sr)
    
    def _simple_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Simplified mel-spectrogram without librosa.
        
        Uses basic FFT-based approach.
        """
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        num_frames = (len(audio) - n_fft) // hop_length + 1
        spectrogram = np.zeros((num_frames, n_fft // 2 + 1))
        
        window = np.hanning(n_fft)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            windowed = frame * window
            fft = np.fft.rfft(windowed)
            spectrogram[i] = np.abs(fft) ** 2
        
        # Simple mel approximation (linear binning)
        mel_bins = np.linspace(0, spectrogram.shape[1] - 1, self.n_mels + 2, dtype=int)
        mel_spec = np.zeros((num_frames, self.n_mels))
        
        for m in range(self.n_mels):
            mel_spec[:, m] = np.mean(spectrogram[:, mel_bins[m]:mel_bins[m+2]], axis=1)
        
        # Log scale
        mel_spec = np.log10(mel_spec + 1e-10)
        
        return mel_spec.astype(np.float32)
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: int = DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        """
        Extract MFCC features with deltas.
        
        Args:
            audio: Raw waveform
            sr: Sample rate
            
        Returns:
            MFCC features (T, n_mfcc * 3) including deltas
        """
        try:
            import librosa
            
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            # Compute deltas
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Stack features
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            return features.T.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"librosa MFCC extraction failed ({e}), using simplified MFCC fallback")
            return self._simple_mfcc(audio, sr)

    def _simple_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simplified MFCC-like feature extraction without librosa.

        Uses log mel features followed by cosine projection and temporal deltas.
        """
        mel = self._simple_mel_spectrogram(audio, sr)
        if mel.ndim != 2 or mel.shape[0] == 0:
            return np.zeros((1, self.n_mfcc * 3), dtype=np.float32)

        num_bins = mel.shape[1]
        coeff_indices = np.arange(self.n_mfcc, dtype=np.float32)
        bin_indices = np.arange(num_bins, dtype=np.float32)
        # DCT-II style cosine basis.
        basis = np.cos(
            (np.pi / num_bins) * (bin_indices[:, None] + 0.5) * coeff_indices[None, :]
        ).astype(np.float32)

        mfcc = np.dot(mel, basis)  # (T, n_mfcc)
        if mfcc.shape[0] > 1:
            mfcc_delta = np.gradient(mfcc, axis=0)
            mfcc_delta2 = np.gradient(mfcc_delta, axis=0)
        else:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)

        return np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=1).astype(np.float32)
    
    def extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> SpectralFeatures:
        """
        Extract spectral features for analysis.
        
        Args:
            audio: Raw waveform
            sr: Sample rate
            
        Returns:
            SpectralFeatures
        """
        try:
            import librosa
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
            
            return SpectralFeatures(
                spectral_centroid=float(np.mean(centroid)),
                spectral_bandwidth=float(np.mean(bandwidth)),
                spectral_rolloff=float(np.mean(rolloff)),
                spectral_flatness=float(np.mean(flatness)),
                zero_crossing_rate=float(np.mean(zcr))
            )
            
        except Exception as e:
            logger.warning(f"librosa spectral features failed ({e}), using defaults")
            return SpectralFeatures(
                spectral_centroid=0.5,
                spectral_bandwidth=0.5,
                spectral_rolloff=0.5,
                spectral_flatness=0.5,
                zero_crossing_rate=0.1
            )
    
    def detect_vocoder_artifacts(
        self,
        spectrogram: np.ndarray,
        audio: np.ndarray
    ) -> VocoderArtifactFeatures:
        """
        Analyze spectral patterns for vocoder signatures.
        
        Neural vocoders leave characteristic artifacts:
        - Phase discontinuities at frame boundaries
        - Unnatural harmonic structure
        - Bandwidth limitations
        - Periodic patterns from autoregressive generation
        
        Args:
            spectrogram: Mel-spectrogram (T, n_mels)
            audio: Raw waveform
            
        Returns:
            VocoderArtifactFeatures
        """
        # 1. Phase discontinuity analysis
        phase_disc = self._analyze_phase_discontinuity(audio)
        
        # 2. Harmonic distortion analysis
        harmonic_dist = self._analyze_harmonic_structure(spectrogram)
        
        # 3. Bandwidth limitation check
        bandwidth_limit = self._check_bandwidth_limitation(spectrogram)
        
        # 4. Periodic artifact detection
        periodic = self._detect_periodic_artifacts(spectrogram)
        
        # Compute overall artifact score
        artifact_score = (
            0.3 * phase_disc +
            0.3 * harmonic_dist +
            0.25 * bandwidth_limit +
            0.15 * (1.0 if periodic else 0.0)
        )
        
        return VocoderArtifactFeatures(
            artifact_score=float(np.clip(artifact_score, 0, 1)),
            phase_discontinuity=float(phase_disc),
            harmonic_distortion=float(harmonic_dist),
            bandwidth_limitation=float(bandwidth_limit),
            periodic_artifacts=periodic
        )
    
    def _analyze_phase_discontinuity(self, audio: np.ndarray) -> float:
        """
        Analyze phase coherence across frames.
        
        Vocoders often have phase discontinuities at frame boundaries.
        
        Returns:
            Discontinuity score [0, 1]
        """
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        num_frames = (len(audio) - n_fft) // hop_length + 1
        if num_frames < 2:
            return 0.5
        
        window = np.hanning(n_fft)
        phases = []
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            
            fft = np.fft.rfft(frame * window)
            phases.append(np.angle(fft))
        
        # Compute phase differences
        phase_diffs = []
        for i in range(1, len(phases)):
            diff = np.abs(phases[i] - phases[i-1])
            # Wrap to [0, pi]
            diff = np.minimum(diff, 2 * np.pi - diff)
            phase_diffs.append(np.mean(diff))
        
        if not phase_diffs:
            return 0.5
        
        # High variance in phase differences indicates discontinuity
        variance = np.var(phase_diffs)
        
        # Normalize to [0, 1]
        return float(np.clip(variance / 0.5, 0, 1))
    
    def _analyze_harmonic_structure(self, spectrogram: np.ndarray) -> float:
        """
        Analyze harmonic structure for anomalies.
        
        Natural speech has predictable harmonic patterns.
        Synthesized speech may have distorted harmonics.
        
        Returns:
            Distortion score [0, 1]
        """
        # Analyze vertical patterns in spectrogram
        # Harmonics appear as parallel horizontal lines
        
        # Compute column correlation
        if spectrogram.shape[0] < 10:
            return 0.5
        
        correlations = []
        for i in range(1, spectrogram.shape[0]):
            corr = np.corrcoef(spectrogram[i], spectrogram[i-1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        if not correlations:
            return 0.5
        
        # High correlation variance indicates inconsistent harmonics
        variance = np.var(correlations)
        
        # Very high or very low correlation is suspicious
        mean_corr = np.mean(correlations)
        if mean_corr > 0.99 or mean_corr < 0.5:
            return 0.6
        
        return float(np.clip(variance * 2, 0, 1))
    
    def _check_bandwidth_limitation(self, spectrogram: np.ndarray) -> float:
        """
        Check for unnatural bandwidth limitations.
        
        Some vocoders have characteristic frequency cutoffs.
        
        Returns:
            Bandwidth limitation score [0, 1]
        """
        # Check energy distribution across frequency bands
        energy_per_band = np.mean(np.abs(spectrogram), axis=0)
        
        # Normalize
        if energy_per_band.max() > 0:
            energy_per_band = energy_per_band / energy_per_band.max()
        
        # Check if high frequencies are unnaturally low
        high_freq_energy = np.mean(energy_per_band[int(len(energy_per_band) * 0.75):])
        low_freq_energy = np.mean(energy_per_band[:int(len(energy_per_band) * 0.25)])
        
        if low_freq_energy > 0:
            ratio = high_freq_energy / low_freq_energy
        else:
            ratio = 0.5
        
        # Very low ratio indicates bandwidth limitation
        if ratio < 0.1:
            return 0.8
        elif ratio < 0.2:
            return 0.5
        
        return float(np.clip(0.3 - ratio * 0.3, 0, 1))
    
    def _detect_periodic_artifacts(self, spectrogram: np.ndarray) -> bool:
        """
        Detect periodic patterns from autoregressive vocoders.
        
        Returns:
            True if periodic artifacts detected
        """
        if spectrogram.shape[0] < 20:
            return False
        
        # Compute autocorrelation of energy envelope
        energy = np.mean(np.abs(spectrogram), axis=1)
        
        # Remove mean
        energy = energy - np.mean(energy)
        
        # Autocorrelation
        autocorr = np.correlate(energy, energy, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        
        # Look for periodic peaks (excluding lag 0)
        if len(autocorr) > 10:
            peaks = autocorr[5:50]  # Check lags 5-50
            if len(peaks) > 0 and np.max(peaks) > 0.5:
                return True
        
        return False
    
    def analyze_voice_consistency(
        self,
        segments: List[np.ndarray],
        sr: int
    ) -> VoiceConsistencyFeatures:
        """
        Analyze voice consistency across segments.
        
        Synthesized speech may have inconsistent voice characteristics.
        
        Args:
            segments: List of audio segments
            sr: Sample rate
            
        Returns:
            VoiceConsistencyFeatures
        """
        if len(segments) < 2:
            return VoiceConsistencyFeatures(
                pitch_consistency=0.8,
                formant_consistency=0.8
            )
        
        # Extract features per segment
        segment_features = []
        
        for segment in segments[:10]:  # Limit to 10 segments
            features = {
                'energy': np.mean(np.abs(segment)),
                'zcr': np.sum(np.diff(np.sign(segment)) != 0) / len(segment),
                'spectral_centroid': self._quick_spectral_centroid(segment, sr)
            }
            segment_features.append(features)
        
        # Compute consistency metrics
        energies = [f['energy'] for f in segment_features]
        zcrs = [f['zcr'] for f in segment_features]
        centroids = [f['spectral_centroid'] for f in segment_features]
        
        # Energy variance (normalized)
        energy_variance = np.var(energies) / (np.mean(energies) + 1e-8)
        
        # Pitch consistency (approximated by ZCR consistency)
        if np.mean(zcrs) > 0:
            pitch_consistency = 1.0 - np.std(zcrs) / np.mean(zcrs)
        else:
            pitch_consistency = 0.5
        
        # Formant consistency (approximated by spectral centroid consistency)
        if np.mean(centroids) > 0:
            formant_consistency = 1.0 - np.std(centroids) / np.mean(centroids)
        else:
            formant_consistency = 0.5
        
        return VoiceConsistencyFeatures(
            pitch_variance=float(np.var(zcrs)),
            pitch_consistency=float(np.clip(pitch_consistency, 0, 1)),
            energy_variance=float(energy_variance),
            formant_consistency=float(np.clip(formant_consistency, 0, 1)),
            speaking_rate_variance=0.0  # Would require speech recognition
        )
    
    def _quick_spectral_centroid(self, audio: np.ndarray, sr: int) -> float:
        """Quick spectral centroid calculation."""
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        if np.sum(fft) > 0:
            centroid = np.sum(freqs * fft) / np.sum(fft)
        else:
            centroid = sr / 4
        
        return float(centroid)
    
    async def _run_wav2vec2_antispoof(
        self,
        audio: np.ndarray,
        engine: "InferenceEngine"
    ) -> float:
        """
        Run Wav2Vec2 Large XLSR antispoofing ONNX model.
        
        Fine-tuned on ASVspoof2019 with 4.01% EER (INT8 ONNX variant).
        Uses the HuggingFace optimum-exported model with Wav2Vec2 XLSR backbone.
        
        Args:
            audio: Raw waveform (1D array) at 16kHz
            engine: InferenceEngine
            
        Returns:
            Spoof probability [0, 1]
        """
        try:
            # Wav2Vec2 XLSR expects ~4 seconds at 16kHz for optimal performance
            target_length = 64600
            
            # Pad or truncate to target length
            if len(audio) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                padded[:len(audio)] = audio
                audio = padded
            elif len(audio) > target_length:
                audio = audio[:target_length]
            
            # Normalize to zero mean, unit variance (Wav2Vec2 expects this)
            audio = audio.astype(np.float32)
            mean = audio.mean()
            std = audio.std()
            if std > 1e-6:
                audio = (audio - mean) / std
            else:
                audio = audio - mean
            
            # Add batch dimension
            batch = np.expand_dims(audio, 0)  # (1, 64600)
            
            # Build attention mask (int32 — ONNX model expects int32, not int64)
            attention_mask = np.ones((1, target_length), dtype=np.int32)
            
            # Run inference via InferenceEngine
            # Pass dict with both required inputs (input_values + attention_mask)
            result = await engine.infer(
                "wav2vec2_antispoof",
                {"input_values": batch, "attention_mask": attention_mask},
                return_probabilities=True
            )
            
            if result.class_probabilities is not None:
                probs = result.class_probabilities[0]
                spoof_prob = float(probs[1]) if len(probs) >= 2 else float(probs[0])
            else:
                spoof_prob = float(result.predictions[0])
            
            logger.debug(f"Wav2Vec2 XLSR antispoof result: spoof_prob={spoof_prob:.4f}")
            return spoof_prob
            
        except ImportError as e:
            logger.error(f"Wav2Vec2 antispoof dependencies missing: {e}")
            raise InferenceError("wav2vec2_antispoof", f"Missing dependencies: {e}")
        except RuntimeError as e:
            logger.error(f"Wav2Vec2 antispoof ONNX runtime error: {e}")
            raise InferenceError("wav2vec2_antispoof", f"Runtime error: {e}")
        except ValueError as e:
            logger.error(f"Wav2Vec2 antispoof input validation error: {e}")
            raise InferenceError("wav2vec2_antispoof", f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"Wav2Vec2 antispoof inference failed: {type(e).__name__}: {e}")
            raise InferenceError("wav2vec2_antispoof", f"Unexpected error: {type(e).__name__}: {e}")

    def _compute_aggregate_score(
        self,
        wav2vec2_antispoof_score: float = 0.5,
        vocoder_artifacts: Optional[VocoderArtifactFeatures] = None,
        voice_consistency: Optional[VoiceConsistencyFeatures] = None,
        use_wav2vec2_antispoof: bool = True,
        wav2vec2_score: float = 0.5,
        any_neural_available: bool = True,
    ) -> float:
        """
        Compute aggregate synthetic probability.

        Args:
            wav2vec2_antispoof_score: Wav2Vec2 XLSR ONNX score (primary)
            vocoder_artifacts: Vocoder artifact analysis
            voice_consistency: Voice consistency analysis
            use_wav2vec2_antispoof: Whether Wav2Vec2 antispoofing was used
            wav2vec2_score: Wav2Vec2 anti-spoofing score (DeepFense-style PyTorch)
            any_neural_available: Whether any neural model produced a non-default score
            
        Returns:
            Aggregate synthetic probability [0, 1]
        """
        # Voice inconsistency score (low consistency = suspicious)
        if voice_consistency is not None:
            inconsistency_score = 1.0 - (
                voice_consistency.pitch_consistency * 0.5 +
                voice_consistency.formant_consistency * 0.5
            )
        else:
            inconsistency_score = 0.5
        
        vocoder_score = vocoder_artifacts.artifact_score if vocoder_artifacts is not None else 0.5
        
        if use_wav2vec2_antispoof:
            aggregate = (
                self.weights["wav2vec2_antispoof"] * wav2vec2_antispoof_score +
                self.weights["wav2vec2"] * wav2vec2_score +
                self.weights["vocoder_artifacts"] * vocoder_score +
                self.weights["voice_consistency"] * inconsistency_score
            )
        else:
            # Fallback: redistribute antispoof weight across remaining detectors
            total_fallback = (
                self.weights["wav2vec2"] +
                self.weights["vocoder_artifacts"] + self.weights["voice_consistency"]
            )
            aggregate = (
                (self.weights["wav2vec2"] / total_fallback) * wav2vec2_score +
                (self.weights["vocoder_artifacts"] / total_fallback) * vocoder_score +
                (self.weights["voice_consistency"] / total_fallback) * inconsistency_score
            )
        
        # When no neural models produced a meaningful score, dampen the aggregate
        # toward neutral (0.5). Heuristic-only features (vocoder artifacts, voice
        # consistency) are unreliable on their own and can produce false positives
        # for simple signals like pure sine waves or silence.
        if not any_neural_available:
            logger.warning(
                "RIVP: HEURISTIC-ONLY AUDIO: No neural models produced a non-neutral "
                f"score. Dampening aggregate {aggregate:.4f} toward 0.5 (80%% reduction). "
                "Audio modality should be treated as low-confidence. "
                "No real ML inference contributed to this result."
            )
            aggregate = 0.5 + (aggregate - 0.5) * 0.2
        
        return float(np.clip(aggregate, 0, 1))
    
    def _compute_confidence(self, details: AudioAnalysisDetails) -> float:
        """
        Compute confidence based on analysis details.

        Confidence reflects how much weight the fusion layer should give
        this modality. It is bounded to [0.15, 0.95] — never 1.0 because
        every detector has a non-zero false-positive rate, and never 0
        because even a heuristic-only result still carries weak signal.

        IMPORTANT: when ``details.any_neural_available`` is False (i.e.
        every neural detector returned its default 0.5 score — typically
        because the model files failed to load or all inferences raised),
        we cap confidence at 0.15 so the orchestrator's evidential fusion
        contributes near-zero evidence from this modality. This honors
        the user's protocol: "Never replace inference with heuristic
        estimates because GPU resources are unavailable" — we still
        produce a score for telemetry, but we will not let it dominate
        the final verdict.

        Args:
            details: Analysis details

        Returns:
            Confidence score [0.15, 0.95]
        """
        # If no neural detector produced a real score, return a low
        # confidence immediately. The aggregate score is dampened to
        # ~0.5 by `_compute_aggregate_score`, and a low confidence here
        # ensures the evidential fusion in core/fusion.py contributes
        # negligible evidence for this modality.
        if not getattr(details, "any_neural_available", True):
            logger.warning(
                "Audio confidence capped at 0.15 — no neural detector "
                "produced a real score (heuristic-only path)."
            )
            return 0.15

        # Base confidence from duration
        duration_factor = min(1.0, details.audio_duration_seconds / 10.0)

        # Segments factor
        segments_factor = min(1.0, details.segments_analyzed / 5.0)

        # Score extremity factor - use primary detector score
        if details.primary_detector == "wav2vec2_antispoof":
            score = details.wav2vec2_antispoof_score
        else:
            score = 0.5
        extremity_factor = abs(score - 0.5) * 2

        # Primary detector bonus (Wav2Vec2 XLSR antispoofing is most reliable)
        detector_bonus = 0.1 if details.primary_detector == "wav2vec2_antispoof" else 0.0

        confidence = (
            0.4 * duration_factor +
            0.3 * segments_factor +
            0.2 * extremity_factor +
            detector_bonus
        )

        return float(np.clip(confidence, 0.3, 0.95))

    def _segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Segment audio for analysis.
        
        Args:
            audio: Full audio waveform
            
        Returns:
            List of audio segments
        """
        segments: List[np.ndarray] = []

        if len(audio) == 0:
            return segments

        if len(audio) <= self.segment_samples:
            padded = np.pad(audio, (0, self.segment_samples - len(audio)))
            return [padded.astype(np.float32)]

        max_start = len(audio) - self.segment_samples
        for start in range(0, max_start + 1, self.hop_samples):
            segment = audio[start:start + self.segment_samples]
            segments.append(segment.astype(np.float32))

        if not segments:
            # Safety fallback for edge cases.
            segments.append(audio[:self.segment_samples].astype(np.float32))

        return segments
    
    def _resample_audio(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Input waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input waveform
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    async def _load_audio(
        self,
        audio_key: Optional[str]
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Load audio from MinIO key.
        
        Args:
            audio_key: MinIO object key
            
        Returns:
            Tuple of (waveform, sample_rate) or (None, 0)
        """
        import io as io_module
        from storage.storage import get_storage_client
        from config import config
        
        if not audio_key:
            return None, 0
        
        try:
            storage = get_storage_client()
            
            audio_bytes = await storage.download_file(
                config.minio_bucket_preprocessed,
                audio_key
            )

            waveform: Optional[np.ndarray] = None

            # Preferred format: NumPy file (.npy)
            if audio_key.endswith(".npy"):
                try:
                    loaded = np.load(io_module.BytesIO(audio_bytes), allow_pickle=True)
                    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
                        if hasattr(loaded, "item") and isinstance(loaded.item(), np.ndarray):
                            loaded = loaded.item()
                    waveform = np.asarray(loaded, dtype=np.float32).reshape(-1)
                except Exception as npy_error:
                    logger.warning(
                        f"Failed to parse {audio_key} as .npy ({npy_error}), "
                        "falling back to raw float32 decode"
                    )

            # Backward-compatible fallback: raw float32 bytes
            if waveform is None:
                waveform = np.frombuffer(audio_bytes, dtype=np.float32).copy()
            
            # Validate waveform
            if len(waveform) == 0:
                logger.warning(f"Empty audio waveform loaded from {audio_key}")
                return None, 0
                
            logger.debug(f"Loaded audio waveform: {len(waveform)} samples from {audio_key}")
            return waveform, self.target_sample_rate
                
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_key}: {e}")
            return None, 0
    
    def _create_default_result(self) -> AudioResult:
        """Create default result for insufficient data."""
        return AudioResult(
            synthetic_probability=0.5,
            vocoder_artifacts_detected=False,
            voice_consistency_score=1.0,
            spectrogram_url=None
        )


# Singleton instance
_audio_analyzer: Optional[AudioAnalyzer] = None


def get_audio_analyzer() -> AudioAnalyzer:
    """Get singleton audio analyzer instance."""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer
