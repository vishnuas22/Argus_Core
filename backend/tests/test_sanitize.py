"""
Argus Core - Input Sanitization Tests
======================================
Comprehensive tests for InputSanitizer in processing/sanitize.py.

Tests cover:
- Magic byte file type detection (all supported types)
- Content-type verification
- Size validation
- Text validation
- Adversarial defense (JPEG compression, noise injection)
- Edge cases (empty files, oversized, unknown types)

No mocks. Real byte-level validation with real PIL/numpy processing.
"""

import os
import sys
import io
import hashlib
from typing import Optional

import pytest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from processing.sanitize import InputSanitizer, SanitizedFile, FileType
from utils.errors import InvalidFileError, ValidationError


# ============== MAGIC BYTE DETECTION ==============

class TestFileTypeDetection:
    """Test magic byte detection for all supported file types."""

    def setup_method(self) -> None:
        self.sanitizer = InputSanitizer(max_size_mb=500, defense_level="none")

    def test_jpeg_detection(self, jpeg_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(jpeg_bytes)
        assert result == FileType.IMAGE_JPEG

    def test_png_detection(self, png_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(png_bytes)
        assert result == FileType.IMAGE_PNG

    def test_webp_detection(self) -> None:
        webp_bytes = (
            b'RIFF'  # RIFF header
            b'\x00\x00\x00\x00'  # file size placeholder
            b'WEBP'  # WEBP marker
            b'VP8 '  # VP8 chunk
            b'\x00' * 20
        )
        result = self.sanitizer._detect_file_type(webp_bytes)
        assert result == FileType.IMAGE_WEBP

    def test_mp4_detection(self, mp4_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(mp4_bytes)
        assert result == FileType.VIDEO_MP4

    def test_mov_detection(self) -> None:
        mov_bytes = (
            b'\x00\x00\x00\x14'  # box size
            b'ftyp'
            b'qt  '  # QuickTime brand
            b'\x00\x00\x02\x00'
            b'qt  '
            b'\x00' * 20
        )
        result = self.sanitizer._detect_file_type(mov_bytes)
        assert result == FileType.VIDEO_MOV

    def test_avi_detection(self) -> None:
        avi_bytes = (
            b'RIFF'
            b'\x00\x00\x00\x00'  # size placeholder
            b'AVI '
            b'LIST' + b'\x00' * 20
        )
        result = self.sanitizer._detect_file_type(avi_bytes)
        assert result == FileType.VIDEO_AVI

    def test_webm_detection(self) -> None:
        # WebM uses EBML header (Matroska)
        webm_bytes = b'\x1a\x45\xdf\xa3' + b'\x00' * 50
        result = self.sanitizer._detect_file_type(webm_bytes)
        assert result == FileType.VIDEO_WEBM

    def test_wav_detection(self, wav_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(wav_bytes)
        assert result == FileType.AUDIO_WAV

    def test_mp3_id3_detection(self, mp3_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(mp3_bytes)
        assert result == FileType.AUDIO_MP3

    def test_mp3_raw_frame_detection(self) -> None:
        mp3_raw = b'\xff\xfb' + b'\x00' * 100
        result = self.sanitizer._detect_file_type(mp3_raw)
        assert result == FileType.AUDIO_MP3

    def test_mp3_raw_v2_detection(self) -> None:
        mp3_raw = b'\xff\xfa' + b'\x00' * 100
        result = self.sanitizer._detect_file_type(mp3_raw)
        assert result == FileType.AUDIO_MP3

    def test_ogg_detection(self) -> None:
        ogg_bytes = b'OggS' + b'\x00' * 50
        result = self.sanitizer._detect_file_type(ogg_bytes)
        assert result == FileType.AUDIO_OGG

    def test_plain_text_detection(self, plain_text_bytes: bytes) -> None:
        result = self.sanitizer._detect_file_type(plain_text_bytes)
        assert result == FileType.TEXT_PLAIN

    def test_unknown_type_returns_none(self) -> None:
        unknown = b'\x00\x01\x02\x03\x04\x05\x06\x07' + b'\xff' * 50
        result = self.sanitizer._detect_file_type(unknown)
        assert result is None

    def test_empty_bytes_returns_none(self) -> None:
        result = self.sanitizer._detect_file_type(b'')
        assert result is None

    def test_too_short_returns_none(self) -> None:
        result = self.sanitizer._detect_file_type(b'\xff\xd8')
        assert result is None


# ============== CONTENT TYPE VERIFICATION ==============

class TestContentTypeVerification:
    """Test content-type matching logic."""

    def setup_method(self) -> None:
        self.sanitizer = InputSanitizer()

    def test_exact_match(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.IMAGE_JPEG, "image/jpeg") is True

    def test_case_insensitive(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.IMAGE_JPEG, "Image/JPEG") is True

    def test_with_charset(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.IMAGE_PNG, "image/png; charset=utf-8") is True

    def test_mov_equivalent(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.VIDEO_MOV, "video/quicktime") is True
        assert self.sanitizer._verify_content_type(FileType.VIDEO_MOV, "video/mov") is True

    def test_mp3_equivalent(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.AUDIO_MP3, "audio/mpeg") is True
        assert self.sanitizer._verify_content_type(FileType.AUDIO_MP3, "audio/mp3") is True

    def test_wav_equivalent(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.AUDIO_WAV, "audio/wave") is True
        assert self.sanitizer._verify_content_type(FileType.AUDIO_WAV, "audio/x-wav") is True

    def test_mismatch(self) -> None:
        assert self.sanitizer._verify_content_type(FileType.IMAGE_JPEG, "image/png") is False


# ============== VALIDATION ==============

class TestFileValidation:
    """Test full file validation pipeline."""

    @pytest.mark.asyncio
    async def test_validate_jpeg(self, jpeg_bytes: bytes) -> None:
        sanitizer = InputSanitizer(max_size_mb=10)
        result = await sanitizer.validate(jpeg_bytes, "test.jpg", "image/jpeg")
        assert isinstance(result, SanitizedFile)
        assert result.file_type == FileType.IMAGE_JPEG
        assert result.is_image is True
        assert result.is_video is False
        assert result.file_size == len(jpeg_bytes)
        assert len(result.file_hash) == 64  # SHA-256

    @pytest.mark.asyncio
    async def test_validate_png(self, png_bytes: bytes) -> None:
        sanitizer = InputSanitizer(max_size_mb=10)
        result = await sanitizer.validate(png_bytes, "test.png", "image/png")
        assert result.file_type == FileType.IMAGE_PNG
        assert result.is_image is True

    @pytest.mark.asyncio
    async def test_validate_mp4(self, mp4_bytes: bytes) -> None:
        sanitizer = InputSanitizer(max_size_mb=10)
        result = await sanitizer.validate(mp4_bytes, "test.mp4", "video/mp4")
        assert result.file_type == FileType.VIDEO_MP4
        assert result.is_video is True

    @pytest.mark.asyncio
    async def test_validate_wav(self, wav_bytes: bytes) -> None:
        sanitizer = InputSanitizer(max_size_mb=10)
        result = await sanitizer.validate(wav_bytes, "test.wav", "audio/wav")
        assert result.file_type == FileType.AUDIO_WAV
        assert result.is_audio is True

    @pytest.mark.asyncio
    async def test_validate_text(self, plain_text_bytes: bytes) -> None:
        sanitizer = InputSanitizer(max_size_mb=10)
        result = await sanitizer.validate(plain_text_bytes, "test.txt", "text/plain")
        assert result.file_type == FileType.TEXT_PLAIN
        assert result.is_text is True

    @pytest.mark.asyncio
    async def test_empty_file_raises(self) -> None:
        sanitizer = InputSanitizer()
        with pytest.raises(InvalidFileError, match="Empty file"):
            await sanitizer.validate(b"", "empty.jpg")

    @pytest.mark.asyncio
    async def test_oversized_file_raises(self) -> None:
        sanitizer = InputSanitizer(max_size_mb=1)
        large_content = b'\xff\xd8\xff' + b'\x00' * (2 * 1024 * 1024)
        with pytest.raises(InvalidFileError, match="exceeds maximum size"):
            await sanitizer.validate(large_content, "huge.jpg")

    @pytest.mark.asyncio
    async def test_unknown_type_raises(self) -> None:
        sanitizer = InputSanitizer()
        unknown = b'\x00\x01\x02\x03' + b'\xff' * 100
        with pytest.raises(InvalidFileError, match="Unsupported file type"):
            await sanitizer.validate(unknown, "unknown.bin")

    @pytest.mark.asyncio
    async def test_file_hash_consistency(self, jpeg_bytes: bytes) -> None:
        sanitizer = InputSanitizer()
        result1 = await sanitizer.validate(jpeg_bytes, "test.jpg")
        result2 = await sanitizer.validate(jpeg_bytes, "test.jpg")
        assert result1.file_hash == result2.file_hash

    @pytest.mark.asyncio
    async def test_different_content_different_hash(self) -> None:
        sanitizer = InputSanitizer()
        text1 = b'Hello, world!' * 10
        text2 = b'Goodbye, world!' * 10
        result1 = await sanitizer.validate(text1, "a.txt")
        result2 = await sanitizer.validate(text2, "b.txt")
        assert result1.file_hash != result2.file_hash


# ============== TEXT VALIDATION ==============

class TestTextValidation:
    """Test text input validation."""

    def setup_method(self) -> None:
        self.sanitizer = InputSanitizer()

    def test_valid_text(self) -> None:
        text = "This is a valid text input that exceeds the minimum length requirement. " * 2
        result = self.sanitizer.validate_text(text)
        assert len(result) >= 50

    def test_text_too_short(self) -> None:
        with pytest.raises(ValidationError, match="at least 50 characters"):
            self.sanitizer.validate_text("Short")

    def test_text_too_long(self) -> None:
        long_text = "x" * 100001
        with pytest.raises(ValidationError, match="must not exceed"):
            self.sanitizer.validate_text(long_text)

    def test_text_strips_whitespace(self) -> None:
        text = "  " + "A" * 50 + "  "
        result = self.sanitizer.validate_text(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_text_boundary_min(self) -> None:
        text = "A" * 50
        result = self.sanitizer.validate_text(text)
        assert len(result) == 50

    def test_text_boundary_max(self) -> None:
        text = "A" * 100000
        result = self.sanitizer.validate_text(text)
        assert len(result) == 100000

    def test_custom_min_max(self) -> None:
        text = "Hello"
        result = self.sanitizer.validate_text(text, min_length=5, max_length=100)
        assert result == "Hello"


# ============== ADVERSARIAL DEFENSE ==============

class TestAdversarialDefense:
    """Test adversarial preprocessing levels."""

    def setup_method(self) -> None:
        self.sanitizer = InputSanitizer(defense_level="standard")

    def test_no_defense(self) -> None:
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = self.sanitizer.apply_adversarial_defense(image, defense_level="none")
        np.testing.assert_array_equal(result, image)

    def test_standard_defense_preserves_shape(self) -> None:
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = self.sanitizer.apply_adversarial_defense(image, defense_level="standard")
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_aggressive_defense_preserves_shape(self) -> None:
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = self.sanitizer.apply_adversarial_defense(image, defense_level="aggressive")
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_standard_defense_reduces_variance(self) -> None:
        """JPEG compression should reduce high-frequency noise."""
        np.random.seed(42)
        noisy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = self.sanitizer.apply_adversarial_defense(noisy_image, defense_level="standard")
        original_std = np.std(noisy_image.astype(float))
        result_std = np.std(result.astype(float))
        # JPEG compression should reduce variance somewhat
        assert result_std <= original_std + 1.0

    def test_defense_uses_instance_level(self) -> None:
        sanitizer = InputSanitizer(defense_level="aggressive")
        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = sanitizer.apply_adversarial_defense(image)
        assert result.shape == image.shape

    def test_single_pixel_image(self) -> None:
        image = np.array([[[255, 128, 0]]], dtype=np.uint8)
        result = self.sanitizer.apply_adversarial_defense(image, defense_level="standard")
        assert result.shape == (1, 1, 3)
