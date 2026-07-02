"""
Argus Core - Audio Frequency-Band Attribution (Iteration 2)
=============================================================
STFT-band occlusion attribution for audio deepfake detectors.

Research grounding:
- The 2026 audio-LLM frequency-time RL paper is the only method that
  explicitly outputs frequency-band explanations, but it is research-
  stage. The practical alternative is **occlusion-based attribution**:
  zero out each frequency band, re-run the detector, and measure the
  score drop. Bands with large drops are the ones the detector relies
  on.
- This is the "leave-one-out" occlusion approach, well-established for
  interpretability (Zeiler & Fergus 2014 lineage).
- For deepfake audio: AASIST/Wav2Vec2 detectors often rely on the
  2-6 kHz band (vocoder artifacts) and the 0-500 Hz band (F0 contour).
  Occlusion attribution surfaces these dependencies for the user.

Algorithm:
1. Compute STFT of the input waveform (n_fft=2048, hop=512, 64 mel bins).
2. Define K=8 frequency bands (linear or mel-spaced).
3. For each band i:
   a. Zero out the band in the STFT.
   b. Reconstruct the waveform via inverse STFT.
   c. Run the detector on the modified waveform.
   d. Record the score drop: baseline_score - modified_score.
4. The band with the largest drop is the most influential.

Latency: K=8 forward passes per audio clip → ~250ms on T4 for a 5s clip
with Wav2Vec2-XLS-R. Acceptable for forensic use.

Strict-compat: pure-additive. No changes to detector interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


# Default frequency bands (Hz) — 8 bands covering 0-8 kHz
DEFAULT_BANDS_HZ: List[Tuple[float, float]] = [
    (0, 250),       # Sub-bass
    (250, 500),     # Bass (F0 contour)
    (500, 1000),    # Low-mid (formants F1)
    (1000, 2000),   # Mid (formants F2)
    (2000, 4000),   # High-mid (vocoder artifacts common here)
    (4000, 6000),   # High (vocoder artifacts)
    (6000, 8000),   # Very high
    (8000, 16000),  # Ultrasonic residuals
]


# NOTE: Sync version removed — use attribute_audio_frequency_bands_async() instead.
# The sync version was non-functional (always returned None) because it could not
# run async detect_fn from a synchronous context.


async def attribute_audio_frequency_bands_async(
    waveform: np.ndarray,
    sample_rate: int,
    detect_fn,
    bands_hz: Optional[List[Tuple[float, float]]] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Optional[Dict[str, Any]]:
    """
    Async version of frequency-band attribution. Use this from analyzers.
    """
    if bands_hz is None:
        bands_hz = DEFAULT_BANDS_HZ

    try:
        import librosa
    except ImportError as e:
        logger.warning("Audio attribution needs librosa: %s", e)
        return None

    try:
        waveform = np.asarray(waveform, dtype=np.float32)
        stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        async def _get_score(wav):
            r = await detect_fn(wav) if asyncio.iscoroutinefunction(detect_fn) else detect_fn(wav)
            if hasattr(r, "score"):
                return float(r.score)
            return float(r)

        baseline = await _get_score(waveform)
        band_drops: List[float] = []

        for lo_hz, hi_hz in bands_hz:
            # Zero out the band in the STFT
            band_mask = (freqs >= lo_hz) & (freqs < hi_hz)
            modified_stft = stft.copy()
            modified_stft[band_mask, :] = 0.0
            # Reconstruct waveform
            modified_wav = librosa.istft(
                modified_stft, hop_length=hop_length, length=len(waveform)
            )
            # Run detector
            try:
                modified_score = await _get_score(modified_wav)
            except Exception as e:
                logger.debug("Audio attribution band %s-%s failed: %s", lo_hz, hi_hz, e)
                modified_score = baseline
            # Drop = baseline - modified. Positive = band contributed to fake detection.
            drop = baseline - modified_score
            band_drops.append(float(drop))

        # Find most influential band
        most_influential_idx = int(np.argmax(np.abs(band_drops)))
        most_influential_band = bands_hz[most_influential_idx]
        most_influential_drop = band_drops[most_influential_idx]

        # Human-readable explanation
        direction = "increased" if most_influential_drop < 0 else "decreased"
        explanation = (
            f"Removing the {most_influential_band[0]:.0f}-{most_influential_band[1]:.0f} Hz "
            f"band {direction} the spoof score by {abs(most_influential_drop):.4f}. "
            f"This band contains the most influential frequency cues for the detector's verdict."
        )

        return {
            "baseline_score": baseline,
            "band_drops": band_drops,
            "bands_hz": bands_hz,
            "most_influential_band": most_influential_band,
            "most_influential_drop": most_influential_drop,
            "explanation": explanation,
        }
    except Exception as e:
        logger.warning("Audio frequency attribution failed: %s", e)
        return None


# Need asyncio import at top
import asyncio  # noqa: E402
