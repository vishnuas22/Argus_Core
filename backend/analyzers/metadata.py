"""
Argus Core - Metadata Analyzer
===============================
Media metadata analysis including C2PA Content Credentials and EXIF data.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - analyzers/metadata.py

SOTA Algorithms:
- C2PA: Content Credentials verification per C2PA v2.3 specification
- EXIF: Anomaly detection via metadata consistency analysis
- Hash verification for file integrity

Analysis Capabilities:
- C2PA Content Credentials extraction and validation
- EXIF data consistency analysis
- File structure anomaly detection
- Hash verification (SHA256, perceptual hashing)
- Provenance chain reconstruction

Integration:
- Imports: forensics/forensics.py (future), schemas/schemas.py
- Inputs: file_bytes: bytes, original_filename: str
- Outputs: MetadataResult

Target Hardware: CPU-only (no GPU required for metadata analysis)
"""

import asyncio
import hashlib
import struct
import re
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

from analyzers.base import BaseAnalyzer
from schemas.schemas import (
    Modality, PreprocessedData, ModalityResult, ContentType,
    MetadataResult, C2PAManifest
)
from config import config
from utils.logging import get_logger
from utils.errors import ValidationError

if TYPE_CHECKING:
    from core.engine import InferenceEngine

logger = get_logger(__name__)


# Magic bytes for common file formats
MAGIC_BYTES = {
    # Images
    b'\xff\xd8\xff': 'image/jpeg',
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'GIF87a': 'image/gif',
    b'GIF89a': 'image/gif',
    b'RIFF': 'image/webp',  # RIFF....WEBP
    
    # Video
    b'\x00\x00\x00\x1cftyp': 'video/mp4',
    b'\x00\x00\x00 ftyp': 'video/mp4',
    b'\x00\x00\x00\x18ftyp': 'video/mp4',
    b'\x00\x00\x00\x14ftyp': 'video/quicktime',
    b'\x1aE\xdf\xa3': 'video/webm',
    
    # Audio
    b'ID3': 'audio/mpeg',
    b'\xff\xfb': 'audio/mpeg',
    b'\xff\xfa': 'audio/mpeg',
    b'OggS': 'audio/ogg',
    b'fLaC': 'audio/flac',
}


@dataclass
class ExifData:
    """
    Extracted EXIF metadata from media files.
    
    Contains camera/device info, timestamps, GPS data,
    and software editing markers.
    """
    # Device information
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    
    # Timestamps
    datetime_original: Optional[datetime] = None
    datetime_digitized: Optional[datetime] = None
    datetime_modified: Optional[datetime] = None
    
    # Image parameters
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    orientation: Optional[int] = None
    color_space: Optional[str] = None
    
    # GPS data
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    
    # Technical metadata
    exposure_time: Optional[str] = None
    f_number: Optional[float] = None
    iso_speed: Optional[int] = None
    focal_length: Optional[float] = None
    
    # Raw EXIF dictionary
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "make": self.make,
            "model": self.model,
            "software": self.software,
            "datetime_original": self.datetime_original.isoformat() if self.datetime_original else None,
            "datetime_digitized": self.datetime_digitized.isoformat() if self.datetime_digitized else None,
            "datetime_modified": self.datetime_modified.isoformat() if self.datetime_modified else None,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "orientation": self.orientation,
            "color_space": self.color_space,
            "gps_latitude": self.gps_latitude,
            "gps_longitude": self.gps_longitude,
            "exposure_time": self.exposure_time,
            "f_number": self.f_number,
            "iso_speed": self.iso_speed,
            "focal_length": self.focal_length
        }


@dataclass
class FileStructureInfo:
    """
    File structure analysis results.
    
    Detects anomalies in file format structure that may
    indicate manipulation or corruption.
    """
    format_detected: str = ""
    format_valid: bool = True
    size_bytes: int = 0
    
    # Structure integrity
    header_valid: bool = True
    footer_valid: bool = True
    chunk_structure_valid: bool = True
    
    # Anomalies
    anomalies: List[str] = field(default_factory=list)
    
    # Additional metadata
    embedded_thumbnails: int = 0
    embedded_data_blocks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format_detected": self.format_detected,
            "format_valid": self.format_valid,
            "size_bytes": self.size_bytes,
            "header_valid": self.header_valid,
            "footer_valid": self.footer_valid,
            "chunk_structure_valid": self.chunk_structure_valid,
            "anomalies": self.anomalies,
            "embedded_thumbnails": self.embedded_thumbnails,
            "embedded_data_blocks": self.embedded_data_blocks
        }


@dataclass
class HashInfo:
    """
    File hash information for integrity verification.
    """
    sha256: str = ""
    md5: str = ""
    perceptual_hash: Optional[str] = None  # For images
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "sha256": self.sha256,
            "md5": self.md5,
            "perceptual_hash": self.perceptual_hash
        }


@dataclass
class C2PAAnalysisResult:
    """
    C2PA Content Credentials analysis results.
    
    Implements C2PA v2.3 specification for content authenticity.
    """
    present: bool = False
    valid: Optional[bool] = None
    
    # Certificate info
    issuer: Optional[str] = None
    subject: Optional[str] = None
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Trust chain
    trust_chain_valid: bool = False
    trust_list_member: bool = False
    
    # Assertions
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Provenance
    provenance_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation details
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "present": self.present,
            "valid": self.valid,
            "issuer": self.issuer,
            "subject": self.subject,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "trust_chain_valid": self.trust_chain_valid,
            "trust_list_member": self.trust_list_member,
            "assertions_count": len(self.assertions),
            "provenance_chain_length": len(self.provenance_chain),
            "validation_errors": self.validation_errors
        }


@dataclass
class MetadataAnalysisDetails:
    """
    Complete metadata analysis results.
    
    Contains all extracted metadata and analysis findings.
    """
    # Core analysis results
    exif_data: Optional[ExifData] = None
    file_structure: Optional[FileStructureInfo] = None
    hash_info: Optional[HashInfo] = None
    c2pa_result: Optional[C2PAAnalysisResult] = None
    
    # Anomaly detection
    exif_anomalies: List[str] = field(default_factory=list)
    tampering_indicators: List[str] = field(default_factory=list)
    
    # Authenticity signals
    authenticity_score: float = 0.5  # 0=likely fake, 1=likely authentic
    
    # Metadata
    original_filename: str = ""
    detected_mime_type: str = ""
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ModalityResult details."""
        return {
            "exif_data": self.exif_data.to_dict() if self.exif_data else None,
            "file_structure": self.file_structure.to_dict() if self.file_structure else None,
            "hash_info": self.hash_info.to_dict() if self.hash_info else None,
            "c2pa_result": self.c2pa_result.to_dict() if self.c2pa_result else None,
            "exif_anomalies": self.exif_anomalies,
            "tampering_indicators": self.tampering_indicators,
            "authenticity_score": round(self.authenticity_score, 4),
            "original_filename": self.original_filename,
            "detected_mime_type": self.detected_mime_type,
            "analysis_time_ms": round(self.analysis_time_ms, 2)
        }


class MetadataAnalyzer(BaseAnalyzer):
    """
    Media metadata and provenance analysis.
    
    Multi-stage analysis pipeline:
    1. File format detection and validation
    2. EXIF metadata extraction and consistency analysis
    3. C2PA Content Credentials extraction and validation
    4. Hash computation for integrity verification
    5. Anomaly detection and tampering indicators
    
    Analysis Targets:
    - C2PA Content Credentials (if present)
    - EXIF data consistency
    - File structure anomalies
    - Hash verification
    - Timestamp consistency
    - Software editing markers
    
    Usage:
        analyzer = MetadataAnalyzer()
        result = await analyzer.analyze(preprocessed_data, engine)
    
    Or direct analysis:
        metadata_result = await analyzer.analyze_file(file_bytes, filename)
    """
    
    def __init__(self):
        """Initialize metadata analyzer."""
        super().__init__(
            analyzer_name="MetadataAnalyzer",
            supported_modalities=[Modality.VIDEO, Modality.AUDIO, Modality.IMAGE],
            version="1.0.0"
        )
        
        # Known editing software patterns (may indicate manipulation)
        self.editing_software_patterns = [
            r'adobe\s*photoshop',
            r'adobe\s*premiere',
            r'adobe\s*after\s*effects',
            r'gimp',
            r'davinci\s*resolve',
            r'final\s*cut',
            r'lightroom',
            r'capture\s*one',
            r'affinity\s*photo',
            r'pixelmator',
            r'canva',
            r'midjourney',
            r'dall[\-\s]*e',
            r'stable\s*diffusion',
            r'firefly',
            r'deepfacelab',
            r'faceswap',
            r'reface',
            r'faceapp'
        ]
        
        # Suspicious EXIF patterns
        self.suspicious_patterns = [
            r'synthetic',
            r'generated',
            r'ai[\-\s]*generated',
            r'created\s*with\s*ai'
        ]
        
        # Weight configuration for authenticity scoring
        self.weights = {
            "c2pa_valid": 0.35,  # Strong positive signal if valid C2PA
            "exif_consistency": 0.25,  # EXIF timestamp/data consistency
            "file_structure": 0.20,  # File format integrity
            "editing_markers": 0.20  # Editing software detected
        }
        
        logger.info("MetadataAnalyzer initialized")
    
    def get_required_models(self) -> List[str]:
        """
        Return models required for metadata analysis.
        
        Metadata analysis is primarily rule-based and doesn't
        require ML models, except for optional perceptual hashing.
        
        Returns:
            List of model registry keys (empty for metadata)
        """
        return []  # No ML models required
    
    def validate_input(self, data: PreprocessedData) -> None:
        """
        Validate input data for metadata analysis.
        
        Metadata analyzer can work with any content type that
        has associated file bytes or metadata.
        
        Args:
            data: PreprocessedData to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        # Call parent validation
        if not data.analysis_id:
            raise ValidationError("analysis_id is required")
        
        # Metadata analyzer needs metadata dict with file info
        if not data.metadata:
            logger.warning("No metadata available for analysis")
    
    async def _analyze_impl(
        self,
        data: PreprocessedData,
        engine: "InferenceEngine"
    ) -> ModalityResult:
        """
        Core metadata analysis implementation.
        
        Args:
            data: PreprocessedData with metadata
            engine: InferenceEngine (not used for metadata analysis)
            
        Returns:
            ModalityResult with metadata analysis results
        """
        # Extract file info from preprocessed data metadata
        file_bytes = data.metadata.get("file_bytes")
        original_filename = data.metadata.get("original_filename", "unknown")
        
        if file_bytes is None:
            # Try to load from storage (in production)
            file_key = data.metadata.get("file_key")
            if file_key:
                file_bytes = await self._load_file_bytes(file_key)
        
        if file_bytes is None:
            logger.warning("No file bytes available for metadata analysis")
            return ModalityResult(
                modality=Modality.IMAGE,  # Default modality
                score=0.5,  # Neutral score
                confidence=0.3,
                details={"error": "No file bytes available"}
            )
        
        # Run analysis
        metadata_result, details = await self.analyze_file(
            file_bytes,
            original_filename
        )
        
        # Compute overall score (0=fake signals, 1=authentic signals)
        # Invert for consistency with other analyzers (higher = more suspicious)
        fake_score = 1.0 - details.authenticity_score
        
        return ModalityResult(
            modality=Modality.IMAGE,  # Metadata applies to all
            score=fake_score,
            confidence=self._compute_confidence(details),
            details=details.to_dict()
        )
    
    async def analyze_file(
        self,
        file_bytes: bytes,
        original_filename: str
    ) -> Tuple[MetadataResult, MetadataAnalysisDetails]:
        """
        Analyze file metadata for authenticity signals.
        
        Main entry point for direct file analysis.
        
        Args:
            file_bytes: Raw file content
            original_filename: Original filename for extension check
            
        Returns:
            Tuple of (MetadataResult, MetadataAnalysisDetails)
        """
        start_time = time.time()
        details = MetadataAnalysisDetails()
        details.original_filename = original_filename
        
        # 1. Detect file format
        mime_type = self._detect_mime_type(file_bytes)
        details.detected_mime_type = mime_type
        
        logger.debug(f"Analyzing metadata for {original_filename} ({mime_type})")
        
        # 2. Analyze file structure
        file_structure = self._analyze_file_structure(file_bytes, mime_type)
        details.file_structure = file_structure
        
        # 3. Compute file hashes
        hash_info = self._compute_hashes(file_bytes)
        details.hash_info = hash_info
        
        # 4. Extract EXIF metadata (for images/videos)
        if mime_type.startswith('image/') or mime_type.startswith('video/'):
            exif_data = self._extract_exif(file_bytes)
            details.exif_data = exif_data
            
            # Analyze EXIF for anomalies
            exif_anomalies = self._analyze_exif_consistency(exif_data)
            details.exif_anomalies = exif_anomalies
        
        # 5. Extract and validate C2PA Content Credentials
        c2pa_result = self._extract_c2pa_manifest(file_bytes)
        details.c2pa_result = c2pa_result
        
        # 6. Detect tampering indicators
        tampering_indicators = self._detect_tampering_indicators(
            details.exif_data,
            file_structure,
            original_filename
        )
        details.tampering_indicators = tampering_indicators
        
        # 7. Compute authenticity score
        details.authenticity_score = self._compute_authenticity_score(details)
        
        # Record timing
        details.analysis_time_ms = (time.time() - start_time) * 1000
        
        # Build standard MetadataResult
        c2pa_manifest = C2PAManifest(
            present=c2pa_result.present if c2pa_result else False,
            valid=c2pa_result.valid if c2pa_result else None,
            issuer=c2pa_result.issuer if c2pa_result else None,
            issued_at=c2pa_result.issued_at if c2pa_result else None,
            assertions=c2pa_result.assertions if c2pa_result else []
        )
        
        metadata_result = MetadataResult(
            c2pa=c2pa_manifest,
            exif_anomalies=details.exif_anomalies,
            file_structure_valid=file_structure.format_valid if file_structure else True
        )
        
        logger.info(
            f"Metadata analysis complete: authenticity={details.authenticity_score:.3f}, "
            f"c2pa_present={c2pa_result.present if c2pa_result else False}, "
            f"anomalies={len(details.exif_anomalies)}"
        )
        
        return metadata_result, details
    
    def _detect_mime_type(self, file_bytes: bytes) -> str:
        """
        Detect MIME type from magic bytes.
        
        Args:
            file_bytes: File content
            
        Returns:
            Detected MIME type string
        """
        if not file_bytes:
            return "application/octet-stream"
        
        # Check magic bytes
        for magic, mime_type in MAGIC_BYTES.items():
            if file_bytes.startswith(magic):
                # Special handling for WebP (RIFF container)
                if magic == b'RIFF' and len(file_bytes) >= 12:
                    if file_bytes[8:12] == b'WEBP':
                        return 'image/webp'
                    elif file_bytes[8:12] == b'AVI ':
                        return 'video/avi'
                    continue
                return mime_type
        
        # Check for MP4 (ftyp box can be at different offsets)
        if b'ftyp' in file_bytes[:32]:
            return 'video/mp4'
        
        return "application/octet-stream"
    
    def _analyze_file_structure(
        self,
        file_bytes: bytes,
        mime_type: str
    ) -> FileStructureInfo:
        """
        Analyze file structure for anomalies.
        
        Args:
            file_bytes: File content
            mime_type: Detected MIME type
            
        Returns:
            FileStructureInfo with structure analysis
        """
        info = FileStructureInfo()
        info.format_detected = mime_type
        info.size_bytes = len(file_bytes)
        
        anomalies = []
        
        if mime_type == 'image/jpeg':
            info = self._analyze_jpeg_structure(file_bytes, info)
        elif mime_type == 'image/png':
            info = self._analyze_png_structure(file_bytes, info)
        elif mime_type.startswith('video/'):
            info = self._analyze_video_structure(file_bytes, info)
        
        info.anomalies = anomalies
        return info
    
    def _analyze_jpeg_structure(
        self,
        file_bytes: bytes,
        info: FileStructureInfo
    ) -> FileStructureInfo:
        """
        Analyze JPEG file structure.
        
        Args:
            file_bytes: JPEG file bytes
            info: FileStructureInfo to populate
            
        Returns:
            Updated FileStructureInfo
        """
        # JPEG should start with SOI marker
        if not file_bytes.startswith(b'\xff\xd8'):
            info.header_valid = False
            info.anomalies.append("Invalid JPEG SOI marker")
            info.format_valid = False
            return info
        
        # JPEG should end with EOI marker
        if not file_bytes.endswith(b'\xff\xd9'):
            info.footer_valid = False
            info.anomalies.append("Invalid JPEG EOI marker")
        
        # Count APP markers (EXIF, etc.)
        app_count = 0
        i = 2
        while i < len(file_bytes) - 1:
            if file_bytes[i] == 0xff:
                marker = file_bytes[i + 1]
                if 0xe0 <= marker <= 0xef:  # APP0-APP15
                    app_count += 1
                    # Get segment length
                    if i + 3 < len(file_bytes):
                        length = struct.unpack('>H', file_bytes[i+2:i+4])[0]
                        i += 2 + length
                        continue
            i += 1
        
        info.embedded_data_blocks = app_count
        
        # Multiple APP1 segments might indicate editing
        if app_count > 5:
            info.anomalies.append(f"Unusual number of APP markers: {app_count}")
        
        return info
    
    def _analyze_png_structure(
        self,
        file_bytes: bytes,
        info: FileStructureInfo
    ) -> FileStructureInfo:
        """
        Analyze PNG file structure.
        
        Args:
            file_bytes: PNG file bytes
            info: FileStructureInfo to populate
            
        Returns:
            Updated FileStructureInfo
        """
        # PNG signature
        if not file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            info.header_valid = False
            info.anomalies.append("Invalid PNG signature")
            info.format_valid = False
            return info
        
        # Parse chunks
        chunks = []
        i = 8
        while i + 8 <= len(file_bytes):
            length = struct.unpack('>I', file_bytes[i:i+4])[0]
            chunk_type = file_bytes[i+4:i+8].decode('ascii', errors='ignore')
            chunks.append(chunk_type)
            
            # Move to next chunk
            i += 12 + length  # length + type + data + CRC
            
            if chunk_type == 'IEND':
                break
        
        # Check for required chunks
        if 'IHDR' not in chunks:
            info.anomalies.append("Missing IHDR chunk")
            info.format_valid = False
        
        if 'IEND' not in chunks:
            info.footer_valid = False
            info.anomalies.append("Missing IEND chunk")
        
        # Check for text chunks (may contain metadata)
        text_chunks = sum(1 for c in chunks if c in ['tEXt', 'iTXt', 'zTXt'])
        info.embedded_data_blocks = text_chunks
        
        return info
    
    def _analyze_video_structure(
        self,
        file_bytes: bytes,
        info: FileStructureInfo
    ) -> FileStructureInfo:
        """
        Analyze video file structure (MP4/MOV).
        
        Args:
            file_bytes: Video file bytes
            info: FileStructureInfo to populate
            
        Returns:
            Updated FileStructureInfo
        """
        # Look for ftyp box
        ftyp_pos = file_bytes.find(b'ftyp')
        if ftyp_pos == -1:
            info.anomalies.append("Missing ftyp box")
            info.format_valid = False
            return info
        
        # Check for moov box (movie metadata)
        if b'moov' not in file_bytes:
            info.anomalies.append("Missing moov box")
        
        # Check for mdat box (media data)
        if b'mdat' not in file_bytes:
            info.anomalies.append("Missing mdat box")
        
        return info
    
    def _compute_hashes(self, file_bytes: bytes) -> HashInfo:
        """
        Compute file hashes for integrity verification.
        
        Args:
            file_bytes: File content
            
        Returns:
            HashInfo with computed hashes
        """
        hash_info = HashInfo()
        
        # SHA256
        hash_info.sha256 = hashlib.sha256(file_bytes).hexdigest()
        
        # MD5 (for legacy compatibility)
        hash_info.md5 = hashlib.md5(file_bytes).hexdigest()
        
        # Perceptual hash would require image decoding
        # Placeholder for now
        hash_info.perceptual_hash = None
        
        return hash_info
    
    def _extract_exif(self, file_bytes: bytes) -> ExifData:
        """
        Extract EXIF metadata from image/video.
        
        Args:
            file_bytes: File content
            
        Returns:
            ExifData with extracted metadata
        """
        exif_data = ExifData()
        
        try:
            # Try using PIL/Pillow for image EXIF
            from PIL import Image
            from PIL.ExifTags import TAGS
            import io
            
            img = Image.open(io.BytesIO(file_bytes))
            exif_raw = img._getexif()
            
            if exif_raw:
                # Convert tag IDs to names
                exif_dict = {}
                for tag_id, value in exif_raw.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    exif_dict[tag_name] = value
                
                exif_data.raw_data = exif_dict
                
                # Extract common fields
                exif_data.make = exif_dict.get('Make')
                exif_data.model = exif_dict.get('Model')
                exif_data.software = exif_dict.get('Software')
                
                # Image dimensions
                exif_data.image_width = exif_dict.get('ImageWidth') or exif_dict.get('ExifImageWidth')
                exif_data.image_height = exif_dict.get('ImageHeight') or exif_dict.get('ExifImageHeight')
                exif_data.orientation = exif_dict.get('Orientation')
                
                # Timestamps
                datetime_original = exif_dict.get('DateTimeOriginal')
                if datetime_original:
                    try:
                        exif_data.datetime_original = datetime.strptime(
                            datetime_original, '%Y:%m:%d %H:%M:%S'
                        )
                    except (ValueError, TypeError):
                        pass
                
                datetime_digitized = exif_dict.get('DateTimeDigitized')
                if datetime_digitized:
                    try:
                        exif_data.datetime_digitized = datetime.strptime(
                            datetime_digitized, '%Y:%m:%d %H:%M:%S'
                        )
                    except (ValueError, TypeError):
                        pass
                
                # Camera parameters
                exif_data.iso_speed = exif_dict.get('ISOSpeedRatings')
                exif_data.exposure_time = str(exif_dict.get('ExposureTime', ''))
                exif_data.f_number = exif_dict.get('FNumber')
                exif_data.focal_length = exif_dict.get('FocalLength')
                
        except ImportError:
            logger.debug("PIL not available, using basic EXIF extraction")
            exif_data = self._extract_exif_basic(file_bytes)
        except Exception as e:
            logger.debug(f"EXIF extraction failed: {e}")
        
        return exif_data
    
    def _extract_exif_basic(self, file_bytes: bytes) -> ExifData:
        """
        Basic EXIF extraction without PIL.
        
        Parses JPEG APP1 segment for EXIF data.
        
        Args:
            file_bytes: File content
            
        Returns:
            ExifData with basic extracted data
        """
        exif_data = ExifData()
        
        # Find APP1 marker with EXIF
        exif_marker = b'\xff\xe1'
        pos = file_bytes.find(exif_marker)
        
        if pos == -1:
            return exif_data
        
        # Check for EXIF header
        exif_header = b'Exif\x00\x00'
        if exif_header not in file_bytes[pos:pos+100]:
            return exif_data
        
        # Look for common strings
        software_match = re.search(
            rb'Software[^\x00]*\x00([^\x00]+)',
            file_bytes[pos:pos+10000]
        )
        if software_match:
            try:
                exif_data.software = software_match.group(1).decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
        
        make_match = re.search(
            rb'Make[^\x00]*\x00([^\x00]+)',
            file_bytes[pos:pos+10000]
        )
        if make_match:
            try:
                exif_data.make = make_match.group(1).decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
        
        model_match = re.search(
            rb'Model[^\x00]*\x00([^\x00]+)',
            file_bytes[pos:pos+10000]
        )
        if model_match:
            try:
                exif_data.model = model_match.group(1).decode('utf-8', errors='ignore').strip()
            except Exception:
                pass
        
        return exif_data
    
    def _analyze_exif_consistency(self, exif_data: Optional[ExifData]) -> List[str]:
        """
        Analyze EXIF metadata for consistency anomalies.
        
        Checks for:
        - Timestamp inconsistencies
        - Missing expected fields
        - Suspicious software markers
        - Conflicting device information
        
        Args:
            exif_data: Extracted EXIF data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not exif_data:
            return anomalies
        
        # Check for editing software
        if exif_data.software:
            software_lower = exif_data.software.lower()
            
            for pattern in self.editing_software_patterns:
                if re.search(pattern, software_lower, re.IGNORECASE):
                    anomalies.append(f"Editing software detected: {exif_data.software}")
                    break
            
            # Check for AI generation markers
            for pattern in self.suspicious_patterns:
                if re.search(pattern, software_lower, re.IGNORECASE):
                    anomalies.append(f"AI generation marker: {exif_data.software}")
                    break
        
        # Check timestamp consistency
        if exif_data.datetime_original and exif_data.datetime_digitized:
            if exif_data.datetime_original != exif_data.datetime_digitized:
                diff = abs(
                    (exif_data.datetime_original - exif_data.datetime_digitized).total_seconds()
                )
                if diff > 60:  # More than 1 minute difference
                    anomalies.append(
                        f"Timestamp inconsistency: original vs digitized differ by {diff:.0f}s"
                    )
        
        # Check for future dates
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if exif_data.datetime_original and exif_data.datetime_original > now:
            anomalies.append("DateTime original is in the future")
        
        # Check for very old dates that might be default values
        if exif_data.datetime_original:
            if exif_data.datetime_original.year < 2000:
                anomalies.append(f"Suspicious date: {exif_data.datetime_original.year}")
        
        # Check for missing camera info on photo
        if not exif_data.make and not exif_data.model:
            if exif_data.software:
                # Has software but no camera = likely edited
                anomalies.append("No camera info but software marker present")
        
        # Check for impossible camera settings
        if exif_data.f_number and exif_data.f_number < 0.5:
            anomalies.append(f"Invalid f-number: {exif_data.f_number}")
        
        if exif_data.iso_speed and (exif_data.iso_speed < 1 or exif_data.iso_speed > 1000000):
            anomalies.append(f"Suspicious ISO speed: {exif_data.iso_speed}")
        
        return anomalies
    
    def _extract_c2pa_manifest(self, file_bytes: bytes) -> C2PAAnalysisResult:
        """
        Extract and validate C2PA Content Credentials.
        
        Implements C2PA v2.3 specification.
        
        Args:
            file_bytes: File content
            
        Returns:
            C2PAAnalysisResult with extraction results
        """
        result = C2PAAnalysisResult()
        
        # C2PA manifest is stored as:
        # - XMP metadata with 'stds:C2PA' namespace
        # - JUMBF box in JPEG/PNG
        # - 'c2pa' box in MP4/MOV
        
        # Check for C2PA markers
        c2pa_markers = [
            b'c2pa',
            b'C2PA',
            b'stds:C2PA',
            b'jumb',  # JUMBF container
            b'c2ma',  # C2PA manifest
        ]
        
        manifest_found = False
        for marker in c2pa_markers:
            if marker in file_bytes:
                manifest_found = True
                break
        
        if not manifest_found:
            # No C2PA manifest present
            result.present = False
            return result
        
        result.present = True
        
        # Try to parse C2PA using c2pa-python library (if available)
        try:
            import c2pa
            
            # Create temporary file for c2pa library
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                f.write(file_bytes)
                temp_path = f.name
            
            try:
                reader = c2pa.Reader.from_file(temp_path)
                
                if reader.active_manifest:
                    manifest = reader.active_manifest
                    
                    result.valid = True
                    result.issuer = manifest.claim_generator_info.get('name') if manifest.claim_generator_info else None
                    
                    # Extract assertions
                    result.assertions = [
                        {"label": a.label, "data": str(a.data)[:200]}
                        for a in manifest.assertions
                    ] if hasattr(manifest, 'assertions') else []
                    
                    # Build provenance chain
                    if hasattr(reader, 'manifest_store'):
                        result.provenance_chain = [
                            {"title": m.title} for m in reader.manifest_store.manifests
                        ]
                    
            except Exception as e:
                result.valid = False
                result.validation_errors.append(f"C2PA validation failed: {str(e)}")
            finally:
                os.unlink(temp_path)
                
        except ImportError:
            logger.debug("c2pa library not available, using basic detection")
            # Basic validation - just confirm presence
            result.valid = None  # Unknown
            result.validation_errors.append("Full C2PA validation requires c2pa library")
        
        return result
    
    def _detect_tampering_indicators(
        self,
        exif_data: Optional[ExifData],
        file_structure: Optional[FileStructureInfo],
        original_filename: str
    ) -> List[str]:
        """
        Detect potential tampering indicators.
        
        Args:
            exif_data: Extracted EXIF data
            file_structure: File structure analysis
            original_filename: Original filename
            
        Returns:
            List of tampering indicators
        """
        indicators = []
        
        # Check filename for suspicious patterns
        filename_lower = original_filename.lower()
        suspicious_names = ['fake', 'deepfake', 'edited', 'modified', 'generated', 'ai_']
        for pattern in suspicious_names:
            if pattern in filename_lower:
                indicators.append(f"Suspicious filename pattern: {pattern}")
        
        # Check file structure
        if file_structure:
            if not file_structure.format_valid:
                indicators.append("Invalid file format structure")
            
            if file_structure.anomalies:
                indicators.extend([
                    f"Structure anomaly: {a}" for a in file_structure.anomalies[:3]
                ])
        
        # Check EXIF for known AI generator signatures
        if exif_data and exif_data.software:
            ai_generators = ['midjourney', 'dall-e', 'stable diffusion', 'firefly', 'runway']
            software_lower = exif_data.software.lower()
            
            for gen in ai_generators:
                if gen in software_lower:
                    indicators.append(f"AI generator detected: {gen}")
                    break
        
        # Missing EXIF in photo that should have it
        if exif_data and not exif_data.raw_data:
            # No EXIF at all can be suspicious
            indicators.append("No EXIF metadata present")
        
        return indicators
    
    def _compute_authenticity_score(
        self,
        details: MetadataAnalysisDetails
    ) -> float:
        """
        Compute overall authenticity score from metadata analysis.
        
        Higher score = more authentic signals
        Lower score = more manipulation signals
        
        Args:
            details: Complete analysis details
            
        Returns:
            Authenticity score [0, 1]
        """
        score = 0.5  # Start neutral
        
        # C2PA presence and validity (strong signal)
        if details.c2pa_result:
            if details.c2pa_result.present:
                if details.c2pa_result.valid:
                    score += 0.3  # Valid C2PA = strong authentic signal
                elif details.c2pa_result.valid is False:
                    score -= 0.2  # Invalid C2PA = suspicious
                else:
                    score += 0.1  # Present but unverified = minor positive
        
        # EXIF consistency
        anomaly_count = len(details.exif_anomalies)
        if anomaly_count == 0:
            score += 0.15  # Clean EXIF = positive
        elif anomaly_count <= 2:
            score -= 0.1  # Minor anomalies
        else:
            score -= 0.25  # Many anomalies = suspicious
        
        # File structure
        if details.file_structure:
            if details.file_structure.format_valid:
                score += 0.05
            else:
                score -= 0.15
        
        # Tampering indicators
        indicator_count = len(details.tampering_indicators)
        if indicator_count > 0:
            score -= min(0.3, indicator_count * 0.1)
        
        # Check for known editing software
        if details.exif_data and details.exif_data.software:
            software_lower = details.exif_data.software.lower()
            
            # Known AI generators (strong negative)
            ai_generators = ['midjourney', 'dall-e', 'dalle', 'stable diffusion', 
                           'firefly', 'runway', 'deepfake', 'faceswap']
            for gen in ai_generators:
                if gen in software_lower:
                    score -= 0.3
                    break
            
            # Photo editing (minor negative)
            else:
                for pattern in self.editing_software_patterns[:5]:  # Common editors
                    if re.search(pattern, software_lower, re.IGNORECASE):
                        score -= 0.1
                        break
        
        # Ensure score is in valid range
        return float(np.clip(score, 0, 1))
    
    def _compute_confidence(self, details: MetadataAnalysisDetails) -> float:
        """
        Compute confidence in the metadata analysis.
        
        Args:
            details: Analysis details
            
        Returns:
            Confidence score [0, 1]
        """
        confidence = 0.5  # Base confidence
        
        # C2PA provides high confidence
        if details.c2pa_result and details.c2pa_result.present:
            if details.c2pa_result.valid:
                confidence += 0.3
            elif details.c2pa_result.valid is not None:
                confidence += 0.15
        
        # Rich EXIF data provides more confidence
        if details.exif_data:
            exif_fields = [
                details.exif_data.make,
                details.exif_data.model,
                details.exif_data.software,
                details.exif_data.datetime_original
            ]
            filled_fields = sum(1 for f in exif_fields if f is not None)
            confidence += filled_fields * 0.05
        
        # File structure analysis
        if details.file_structure and details.file_structure.format_valid:
            confidence += 0.1
        
        return float(np.clip(confidence, 0.3, 0.95))
    
    async def _load_file_bytes(self, file_key: str) -> Optional[bytes]:
        """
        Load file bytes from MinIO storage.
        
        Args:
            file_key: MinIO object key
            
        Returns:
            File bytes or None
        """
        # TODO: Integrate with StorageClient
        # In production:
        # bytes_data = await storage.download_file("argus-uploads", file_key)
        # return bytes_data
        
        logger.debug(f"Would load file from key: {file_key}")
        return None


# Singleton instance
_metadata_analyzer: Optional[MetadataAnalyzer] = None


def get_metadata_analyzer() -> MetadataAnalyzer:
    """Get singleton metadata analyzer instance."""
    global _metadata_analyzer
    if _metadata_analyzer is None:
        _metadata_analyzer = MetadataAnalyzer()
    return _metadata_analyzer
