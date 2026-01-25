"""
Argus Core - Data Transforms
============================
Transform raw data for model inference.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - processing/transform.py

Transforms:
- Image normalization (ImageNet mean/std)
- Mel-spectrogram extraction
- MFCC feature extraction
- Tensor format conversion
"""

from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DataTransformer:
    """
    Transform raw data for model inference.
    
    Provides standardized preprocessing for images and audio
    across all analyzers.
    """
    
    def __init__(
        self,
        target_image_size: Tuple[int, int] = (224, 224),
        audio_sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 13
    ):
        """
        Initialize transformer.
        
        Args:
            target_image_size: Target size for images
            audio_sample_rate: Target audio sample rate
            n_mels: Number of mel bands for spectrogram
            n_mfcc: Number of MFCC coefficients
        """
        self.target_image_size = target_image_size
        self.audio_sample_rate = audio_sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def transform_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        to_chw: bool = True
    ) -> np.ndarray:
        """
        Prepare image for model input.
        
        Applies:
        - Resize to target_size
        - RGB normalization (ImageNet mean/std)
        - Channel ordering (CHW for PyTorch models)
        
        Args:
            image: Input image (H, W, C) RGB
            target_size: Optional override for target size
            normalize: Apply ImageNet normalization
            to_chw: Convert to CHW format (PyTorch)
            
        Returns:
            Transformed image tensor
        """
        size = target_size or self.target_image_size
        
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        
        # Resize
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(size, Image.LANCZOS)
        image = np.array(pil_image)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        if normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        
        # Convert to CHW format for PyTorch models
        if to_chw:
            image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def transform_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """
        Transform batch of images.
        
        Args:
            images: List of images
            **kwargs: Arguments passed to transform_image
            
        Returns:
            Batched tensor (N, C, H, W)
        """
        transformed = [self.transform_image(img, **kwargs) for img in images]
        return np.stack(transformed, axis=0)
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        n_mels: Optional[int] = None,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract mel-spectrogram features from audio.
        
        Args:
            audio: Raw waveform (1D array)
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length between frames
            
        Returns:
            Mel-spectrogram (n_mels, time_frames)
        """
        try:
            import librosa
            
            sr = sample_rate or self.audio_sample_rate
            mel_bands = n_mels or self.n_mels
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=mel_bands,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except ImportError:
            logger.warning("librosa not available, using basic FFT")
            return self._basic_spectrogram(audio, n_fft, hop_length)
    
    def _basic_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int,
        hop_length: int
    ) -> np.ndarray:
        """Basic spectrogram without librosa."""
        from scipy import signal
        
        f, t, Sxx = signal.spectrogram(
            audio,
            fs=self.audio_sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return Sxx_db
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
        include_deltas: bool = True
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Raw waveform
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            include_deltas: Include delta and delta-delta features
            
        Returns:
            MFCC features (n_mfcc * 3 if deltas, else n_mfcc, time_frames)
        """
        try:
            import librosa
            
            sr = sample_rate or self.audio_sample_rate
            n_coef = n_mfcc or self.n_mfcc
            
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_coef
            )
            
            if include_deltas:
                delta = librosa.feature.delta(mfccs)
                delta2 = librosa.feature.delta(mfccs, order=2)
                mfccs = np.vstack([mfccs, delta, delta2])
            
            return mfccs
            
        except ImportError:
            logger.warning("librosa not available for MFCC")
            return np.zeros((self.n_mfcc, 100))
    
    def normalize_audio(
        self,
        audio: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio to target loudness.
        
        Args:
            audio: Raw waveform
            target_db: Target loudness in dB
            
        Returns:
            Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-10:
            return audio
        
        # Calculate current dB
        current_db = 20 * np.log10(rms)
        
        # Calculate gain
        gain = 10 ** ((target_db - current_db) / 20)
        
        # Apply gain with clipping
        normalized = np.clip(audio * gain, -1.0, 1.0)
        
        return normalized
    
    def resample_audio(
        self,
        audio: np.ndarray,
        original_sr: int,
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Raw waveform
            original_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        target = target_sr or self.audio_sample_rate
        
        if original_sr == target:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=original_sr, target_sr=target)
        except ImportError:
            from scipy import signal
            
            num_samples = int(len(audio) * target / original_sr)
            return signal.resample(audio, num_samples)
    
    def denormalize_image(
        self,
        image: np.ndarray,
        from_chw: bool = True
    ) -> np.ndarray:
        """
        Convert normalized tensor back to displayable image.
        
        Args:
            image: Normalized image tensor
            from_chw: Convert from CHW to HWC
            
        Returns:
            RGB image (H, W, C) in [0, 255]
        """
        if from_chw:
            image = np.transpose(image, (1, 2, 0))
        
        # Reverse ImageNet normalization
        image = (image * IMAGENET_STD) + IMAGENET_MEAN
        
        # Clip and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
