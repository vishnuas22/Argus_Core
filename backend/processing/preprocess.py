"""
Argus Core - Preprocessing Orchestrator
=======================================
Orchestrates media preprocessing pipeline.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - processing/preprocess.py

Pipeline:
1. Download file from MinIO
2. Detect media type
3. Route to appropriate extractor
4. Upload preprocessed data
5. Update job status
"""

from typing import Optional, Dict, Any
import asyncio
import io as io_module
import numpy as np

from config import config
from schemas import (
    ContentType,
    PreprocessedData,
    AnalysisStatus,
    Modality
)
from storage.storage import StorageClient
from storage.db import DatabaseClient
from processing.sanitize import InputSanitizer, SanitizedFile, FileType
from processing.extract import MediaExtractor, VideoData, AudioData
from utils.errors import PreprocessingError
from utils.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Orchestrates media preprocessing.
    
    Handles routing of different media types to appropriate
    extractors and manages the preprocessing pipeline.
    """
    
    def __init__(
        self,
        storage: Optional[StorageClient] = None,
        db: Optional[DatabaseClient] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            storage: StorageClient instance
            db: DatabaseClient instance
        """
        self.storage = storage
        self.db = db
        self.sanitizer = InputSanitizer()
        self.extractor = MediaExtractor()
    
    async def process(
        self,
        analysis_id: str,
        file_key: str,
        file_type: FileType,
        options: Optional[Dict[str, Any]] = None
    ) -> PreprocessedData:
        """
        Preprocess media file for analysis.
        
        Args:
            analysis_id: Analysis ID
            file_key: MinIO object key for uploaded file
            file_type: Detected file type
            options: Processing options
            
        Returns:
            PreprocessedData with extracted features
        """
        options = options or {}
        
        try:
            # Update status
            if self.db:
                await self.db.update_analysis_status(
                    analysis_id,
                    AnalysisStatus.PREPROCESSING
                )
            
            # Download file from storage
            logger.info(f"Downloading file for analysis {analysis_id}")
            
            if self.storage:
                file_bytes = await self.storage.download_file(
                    config.minio_bucket_uploads,
                    file_key
                )
            else:
                raise PreprocessingError("storage", "Storage client not configured")
            
            # Route based on file type
            content_type = self._determine_content_type(file_type)
            
            if file_type.value.startswith("video/"):
                return await self._process_video(
                    analysis_id,
                    file_bytes,
                    content_type,
                    options
                )
            
            elif file_type.value.startswith("audio/"):
                return await self._process_audio(
                    analysis_id,
                    file_bytes,
                    options
                )
            
            elif file_type.value.startswith("image/"):
                return await self._process_image(
                    analysis_id,
                    file_bytes,
                    options
                )
            
            elif file_type.value.startswith("text/"):
                return await self._process_text(
                    analysis_id,
                    file_bytes,
                    options
                )
            
            else:
                raise PreprocessingError(
                    "routing",
                    f"Unsupported file type: {file_type.value}"
                )
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            
            if self.db:
                await self.db.update_analysis_status(
                    analysis_id,
                    AnalysisStatus.FAILED,
                    error_message=str(e)
                )
            
            raise PreprocessingError("process", str(e))
    
    def _determine_content_type(self, file_type: FileType) -> ContentType:
        """Determine content type from file type."""
        if file_type.value.startswith("video/"):
            # Will be refined after checking for audio track
            return ContentType.VIDEO_NO_SPEECH
        elif file_type.value.startswith("audio/"):
            return ContentType.AUDIO_ONLY
        elif file_type.value.startswith("image/"):
            return ContentType.IMAGE_ONLY
        else:
            raise ValueError(f"Unsupported content type: {file_type}")
    
    async def _process_video(
        self,
        analysis_id: str,
        file_bytes: bytes,
        content_type: ContentType,
        options: Dict[str, Any]
    ) -> PreprocessedData:
        """
        Process video file.
        
        Extracts frames, faces, and audio track.
        """
        logger.info(f"Processing video for analysis {analysis_id}")
        
        video_data: VideoData = await self.extractor.extract_video_data(
            file_bytes,
            frame_sample_rate=options.get("frame_sample_rate"),
            max_frames=options.get("max_frames", 100),
            extract_audio=True
        )
        
        # Refine content type based on audio presence
        if video_data.has_audio:
            content_type = ContentType.VIDEO_WITH_SPEECH
        
        # Upload extracted data to storage
        frame_keys = []
        face_crop_keys = []
        mouth_crop_keys = []
        audio_key = None
        
        if self.storage:
            # Upload frames as proper .npy files
            for i, frame in enumerate(video_data.frames):
                key = f"{analysis_id}/frames/frame_{i:06d}.npy"
                buffer = io_module.BytesIO()
                np.save(buffer, frame)
                buffer.seek(0)
                await self.storage.upload_file(
                    buffer.read(),
                    config.minio_bucket_preprocessed,
                    key,
                    "application/octet-stream"
                )
                frame_keys.append(key)
            
            # Upload face crops as proper .npy files
            for i, crop in enumerate(video_data.face_crops):
                key = f"{analysis_id}/faces/face_{i:06d}.npy"
                buffer = io_module.BytesIO()
                np.save(buffer, crop)
                buffer.seek(0)
                await self.storage.upload_file(
                    buffer.read(),
                    config.minio_bucket_preprocessed,
                    key,
                    "application/octet-stream"
                )
                face_crop_keys.append(key)
            
            # Upload mouth crops (landmark-based, for precise lip-sync analysis)
            for i, crop in enumerate(video_data.mouth_crops):
                key = f"{analysis_id}/mouths/mouth_{i:06d}.npy"
                buffer = io_module.BytesIO()
                np.save(buffer, crop)
                buffer.seek(0)
                await self.storage.upload_file(
                    buffer.read(),
                    config.minio_bucket_preprocessed,
                    key,
                    "application/octet-stream"
                )
                mouth_crop_keys.append(key)
            
            # Upload audio if present as proper .npy file
            if video_data.audio is not None:
                audio_key = f"{analysis_id}/audio/track.npy"
                buffer = io_module.BytesIO()
                np.save(buffer, video_data.audio)
                buffer.seek(0)
                await self.storage.upload_file(
                    buffer.read(),
                    config.minio_bucket_preprocessed,
                    audio_key,
                    "application/octet-stream"
                )
        
        return PreprocessedData(
            analysis_id=analysis_id,
            content_type=content_type,
            frames=frame_keys,
            face_crops=face_crop_keys,
            audio_key=audio_key,
            metadata={
                "fps": video_data.fps,
                "duration_seconds": video_data.duration_seconds,
                "frame_indices": video_data.frame_indices,
                "width": video_data.width,
                "height": video_data.height,
                "num_faces": len(video_data.face_detections),
                "has_audio": video_data.has_audio,
                "mouth_crop_keys": mouth_crop_keys
            }
        )
    
    async def _process_audio(
        self,
        analysis_id: str,
        file_bytes: bytes,
        options: Dict[str, Any]
    ) -> PreprocessedData:
        """
        Process audio file.
        
        Extracts waveform and metadata.
        """
        logger.info(f"Processing audio for analysis {analysis_id}")
        
        audio_data: AudioData = await self.extractor.extract_audio_file(file_bytes)
        
        audio_key = None
        
        if self.storage:
            audio_key = f"{analysis_id}/audio/track.npy"
            buffer = io_module.BytesIO()
            np.save(buffer, audio_data.waveform.astype(np.float32))
            buffer.seek(0)
            await self.storage.upload_file(
                buffer.read(),
                config.minio_bucket_preprocessed,
                audio_key,
                "application/octet-stream"
            )
        
        return PreprocessedData(
            analysis_id=analysis_id,
            content_type=ContentType.AUDIO_ONLY,
            audio_key=audio_key,
            metadata={
                "sample_rate": audio_data.sample_rate,
                "duration_seconds": audio_data.duration_seconds,
                "channels": audio_data.channels
            }
        )
    
    async def _process_image(
        self,
        analysis_id: str,
        file_bytes: bytes,
        options: Dict[str, Any]
    ) -> PreprocessedData:
        """
        Process image file.
        
        Detects faces and prepares for analysis.
        """
        logger.info(f"Processing image for analysis {analysis_id}")
        
        import numpy as np
        from PIL import Image
        import io
        
        # Load image
        img = Image.open(io.BytesIO(file_bytes))
        img_array = np.array(img.convert("RGB"))
        
        # Detect faces
        face_detections = await self.extractor._detect_faces(img_array)
        
        # Crop faces
        face_crops = []
        for det in face_detections:
            crop = self.extractor._crop_face(img_array, det)
            if crop is not None:
                face_crops.append(crop)
        
        # Upload data
        frame_keys = []
        face_crop_keys = []
        
        if self.storage:
            # Upload original image as proper .npy format
            import io
            key = f"{analysis_id}/frames/image.npy"
            buffer = io.BytesIO()
            np.save(buffer, img_array)
            buffer.seek(0)
            await self.storage.upload_file(
                buffer.read(),
                config.minio_bucket_preprocessed,
                key,
                "application/octet-stream"
            )
            frame_keys.append(key)
            
            # Upload face crops
            for i, crop in enumerate(face_crops):
                key = f"{analysis_id}/faces/face_{i:06d}.npy"
                buffer = io.BytesIO()
                np.save(buffer, crop)
                buffer.seek(0)
                await self.storage.upload_file(
                    buffer.read(),
                    config.minio_bucket_preprocessed,
                    key,
                    "application/octet-stream"
                )
                face_crop_keys.append(key)
        
        return PreprocessedData(
            analysis_id=analysis_id,
            content_type=ContentType.IMAGE_ONLY,
            frames=frame_keys,
            face_crops=face_crop_keys,
            metadata={
                "width": img_array.shape[1],
                "height": img_array.shape[0],
                "num_faces": len(face_detections),
                "image_shape": list(img_array.shape)
            }
        )
    

