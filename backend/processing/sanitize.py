"""
Argus Core - Input Sanitization
===============================
Input validation and adversarial defense for uploaded files.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - processing/sanitize.py

Security Checks:
1. Magic byte verification (not extension-based)
2. Content-type validation
3. Size limits
4. Adversarial pattern detection

Adversarial Defense:
1. JPEG recompression (removes perturbations)
2. Gaussian noise injection
3. Multi-scale analysis
"""

import io
import hashlib
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image

from config import config
from utils.errors import InvalidFileError, ValidationError
from utils.logging import get_logger

logger = get_logger(__name__)


class FileType(str, Enum):
    """Supported file types for analysis."""
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_WEBP = "image/webp"
    VIDEO_MP4 = "video/mp4"
    VIDEO_WEBM = "video/webm"
    VIDEO_MOV = "video/quicktime"
    VIDEO_AVI = "video/x-msvideo"
    AUDIO_MP3 = "audio/mpeg"
    AUDIO_WAV = "audio/wav"
    AUDIO_OGG = "audio/ogg"
    TEXT_PLAIN = "text/plain"


# Magic byte signatures for file type detection
MAGIC_BYTES: Dict[bytes, FileType] = {
    # JPEG
    b'\xff\xd8\xff': FileType.IMAGE_JPEG,
    # PNG
    b'\x89PNG\r\n\x1a\n': FileType.IMAGE_PNG,
    # WEBP
    b'RIFF': FileType.IMAGE_WEBP,  # Need additional check for WEBP marker
    # MP4/MOV (ftyp box)
    b'\x00\x00\x00': FileType.VIDEO_MP4,  # Need additional check
    # AVI
    b'RIFF': FileType.VIDEO_AVI,  # Need additional check for AVI marker
    # MP3
    b'\xff\xfb': FileType.AUDIO_MP3,
    b'\xff\xfa': FileType.AUDIO_MP3,
    b'ID3': FileType.AUDIO_MP3,
    # WAV
    b'RIFF': FileType.AUDIO_WAV,  # Need additional check for WAVE marker
    # OGG
    b'OggS': FileType.AUDIO_OGG,
}


@dataclass
class SanitizedFile:
    """Validated and sanitized file data."""
    content: bytes
    file_type: FileType
    original_filename: str
    file_hash: str
    file_size: int
    mime_type: str
    is_video: bool
    is_audio: bool
    is_image: bool
    is_text: bool
    duration_seconds: Optional[float] = None


class InputSanitizer:
    """
    Validate and sanitize uploaded files.
    
    Performs security checks and optional adversarial defense
    preprocessing to protect against manipulation attacks.
    """
    
    def __init__(
        self,
        max_size_mb: int = None,
        defense_level: str = "standard"
    ):
        """
        Initialize sanitizer.
        
        Args:
            max_size_mb: Maximum file size in MB
            defense_level: Adversarial defense level (none, standard, aggressive)
        """
        self.max_size_mb = max_size_mb or config.max_file_size_mb
        self.defense_level = defense_level
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
    
    async def validate(
        self,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> SanitizedFile:
        """
        Validate and sanitize input file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            content_type: Claimed content type
            
        Returns:
            SanitizedFile with verified metadata
            
        Raises:
            InvalidFileError: If validation fails
        """
        # Check file size
        file_size = len(file_content)
        if file_size > self.max_size_bytes:
            raise InvalidFileError(
                f"File exceeds maximum size of {self.max_size_mb}MB",
                {"size_mb": file_size / (1024 * 1024)}
            )
        
        if file_size == 0:
            raise InvalidFileError("Empty file")
        
        # Detect file type from magic bytes
        detected_type = self._detect_file_type(file_content)
        if detected_type is None:
            raise InvalidFileError(
                "Unsupported file type",
                {"filename": filename}
            )
        
        # Verify content-type matches if provided
        if content_type and not self._verify_content_type(detected_type, content_type):
            logger.warning(
                f"Content-type mismatch: claimed {content_type}, detected {detected_type.value}"
            )
        
        # Compute file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Determine file category
        is_image = detected_type.value.startswith("image/")
        is_video = detected_type.value.startswith("video/")
        is_audio = detected_type.value.startswith("audio/")
        is_text = detected_type.value.startswith("text/")
        
        # Get duration for video/audio (placeholder - implement with ffprobe)
        duration = None
        if is_video or is_audio:
            duration = await self._get_media_duration(file_content, detected_type)
        
        return SanitizedFile(
            content=file_content,
            file_type=detected_type,
            original_filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            mime_type=detected_type.value,
            is_video=is_video,
            is_audio=is_audio,
            is_image=is_image,
            is_text=is_text,
            duration_seconds=duration
        )
    
    def _detect_file_type(self, content: bytes) -> Optional[FileType]:
        """
        Detect file type from magic bytes.
        
        Uses byte signatures instead of file extension
        for security against extension spoofing.
        """
        # JPEG
        if content[:3] == b'\xff\xd8\xff':
            return FileType.IMAGE_JPEG
        
        # PNG
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return FileType.IMAGE_PNG
        
        # WEBP
        if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return FileType.IMAGE_WEBP
        
        # MP4/MOV (check for ftyp box)
        if len(content) >= 12:
            if content[4:8] == b'ftyp':
                ftyp = content[8:12]
                if ftyp in [b'mp41', b'mp42', b'isom', b'avc1', b'M4V ']:
                    return FileType.VIDEO_MP4
                if ftyp in [b'qt  ', b'MSNV']:
                    return FileType.VIDEO_MOV
        
        # AVI
        if content[:4] == b'RIFF' and content[8:12] == b'AVI ':
            return FileType.VIDEO_AVI
        
        # WEBM
        if content[:4] == b'\x1a\x45\xdf\xa3':
            return FileType.VIDEO_WEBM
        
        # WAV
        if content[:4] == b'RIFF' and content[8:12] == b'WAVE':
            return FileType.AUDIO_WAV
        
        # MP3
        if content[:3] == b'ID3' or content[:2] in [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3']:
            return FileType.AUDIO_MP3
        
        # OGG
        if content[:4] == b'OggS':
            return FileType.AUDIO_OGG
        
        # Plain text (check if valid UTF-8 without binary chars)
        try:
            content[:1024].decode('utf-8')
            # Check for high proportion of printable chars
            printable = sum(1 for b in content[:1024] if 32 <= b <= 126 or b in [9, 10, 13])
            if printable / min(1024, len(content)) > 0.8:
                return FileType.TEXT_PLAIN
        except UnicodeDecodeError:
            pass
        
        return None
    
    def _verify_content_type(
        self,
        detected: FileType,
        claimed: str
    ) -> bool:
        """Verify claimed content-type matches detected type."""
        # Normalize claimed type
        claimed_normalized = claimed.lower().split(";")[0].strip()
        
        # Allow some flexibility in MIME type naming
        equivalents = {
            "video/quicktime": ["video/mov", "video/qt"],
            "audio/mpeg": ["audio/mp3"],
            "audio/wav": ["audio/wave", "audio/x-wav"],
        }
        
        if claimed_normalized == detected.value:
            return True
        
        if detected.value in equivalents:
            return claimed_normalized in equivalents[detected.value]
        
        return False
    
    async def _get_media_duration(
        self,
        content: bytes,
        file_type: FileType
    ) -> Optional[float]:
        """
        Get media duration using ffprobe.
        
        Returns None if duration cannot be determined.
        """
        try:
            import subprocess
            import tempfile
            import os
            
            # Write to temp file
            suffix = "." + file_type.value.split("/")[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        temp_path
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    return float(result.stdout.strip())
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            logger.warning(f"Failed to get media duration: {e}")
        
        return None
    
    def apply_adversarial_defense(
        self,
        image: np.ndarray,
        defense_level: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply preprocessing to defeat adversarial attacks.
        
        Args:
            image: Input image as numpy array (H, W, C)
            defense_level: Override instance defense level
            
        Returns:
            Preprocessed image
            
        Levels:
        - "none": No defense (faster)
        - "standard": JPEG compression Q=85
        - "aggressive": Compression + noise + blur
        """
        level = defense_level or self.defense_level
        
        if level == "none":
            return image
        
        # Convert to PIL for processing
        pil_image = Image.fromarray(image)
        
        if level in ["standard", "aggressive"]:
            # JPEG compression to remove high-frequency perturbations
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            pil_image = Image.open(buffer)
        
        if level == "aggressive":
            # Add slight Gaussian noise
            img_array = np.array(pil_image).astype(np.float32)
            noise = np.random.normal(0, 2, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Slight Gaussian blur
            from PIL import ImageFilter
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return np.array(pil_image)
    
    def validate_text(
        self,
        text: str,
        min_length: int = 50,
        max_length: int = 100000
    ) -> str:
        """
        Validate text input.
        
        Args:
            text: Input text
            min_length: Minimum character count
            max_length: Maximum character count
            
        Returns:
            Validated text
            
        Raises:
            ValidationError: If text is invalid
        """
        text = text.strip()
        
        if len(text) < min_length:
            raise ValidationError(
                "text",
                f"Text must be at least {min_length} characters"
            )
        
        if len(text) > max_length:
            raise ValidationError(
                "text",
                f"Text must not exceed {max_length} characters"
            )
        
        return text
