"""
Argus Core - Media Extraction
=============================
Frame extraction, audio separation, and face detection.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - processing/extract.py

Capabilities:
- Video frame extraction via FFmpeg
- Audio track separation
- Face detection via RetinaFace
- Smart keyframe sampling
"""

import io
import subprocess
import tempfile
import os
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
import asyncio

from config import config
from utils.errors import PreprocessingError
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FaceDetection:
    """Detected face with bounding box and landmarks."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None  # eye_left, eye_right, nose, mouth_left, mouth_right
    frame_index: int = 0


@dataclass
class VideoData:
    """Extracted video data for analysis."""
    frames: List[np.ndarray] = field(default_factory=list)
    face_crops: List[np.ndarray] = field(default_factory=list)
    face_detections: List[FaceDetection] = field(default_factory=list)
    audio: Optional[np.ndarray] = None
    audio_sample_rate: int = 16000
    fps: float = 0.0
    duration_seconds: float = 0.0
    frame_indices: List[int] = field(default_factory=list)
    has_audio: bool = False
    width: int = 0
    height: int = 0


@dataclass
class AudioData:
    """Extracted audio data for analysis."""
    waveform: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    channels: int = 1


class MediaExtractor:
    """
    Extract analyzable data from media files.
    
    Uses FFmpeg for video/audio processing and
    RetinaFace for face detection.
    """
    
    def __init__(
        self,
        target_frame_size: Tuple[int, int] = (640, 480),
        face_crop_size: Tuple[int, int] = (224, 224),
        audio_sample_rate: int = 16000
    ):
        """
        Initialize extractor.
        
        Args:
            target_frame_size: Target frame dimensions
            face_crop_size: Face crop dimensions for models
            audio_sample_rate: Target audio sample rate
        """
        self.target_frame_size = target_frame_size
        self.face_crop_size = face_crop_size
        self.audio_sample_rate = audio_sample_rate
        
        self._face_detector = None
    
    async def extract_video_data(
        self,
        video_bytes: bytes,
        frame_sample_rate: Optional[int] = None,
        max_frames: int = 100,
        extract_audio: bool = True
    ) -> VideoData:
        """
        Extract frames, faces, and audio from video.
        
        Args:
            video_bytes: Raw video file bytes
            frame_sample_rate: Sample every Nth frame (None = auto)
            max_frames: Maximum frames to extract
            extract_audio: Whether to extract audio track
            
        Returns:
            VideoData with extracted content
        """
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            # Get video metadata
            metadata = await self._get_video_metadata(temp_path)
            
            duration = metadata.get("duration", 0)
            fps = metadata.get("fps", 30)
            width = metadata.get("width", 0)
            height = metadata.get("height", 0)
            
            # Auto-determine sample rate if not specified
            if frame_sample_rate is None:
                frame_sample_rate = config.get_frame_sample_rate(duration)
            
            # Extract frames
            frames, frame_indices = await self._extract_frames(
                temp_path,
                frame_sample_rate=frame_sample_rate,
                max_frames=max_frames
            )
            
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Detect faces and crop
            face_crops = []
            face_detections = []
            
            for i, frame in enumerate(frames):
                detections = await self._detect_faces(frame)
                
                for det in detections:
                    det.frame_index = frame_indices[i]
                    face_detections.append(det)
                    
                    # Crop face region
                    crop = self._crop_face(frame, det)
                    if crop is not None:
                        face_crops.append(crop)
            
            logger.info(f"Detected {len(face_detections)} faces")
            
            # Extract audio if requested
            audio = None
            has_audio = False
            
            if extract_audio:
                audio_data = await self._extract_audio_track(temp_path)
                if audio_data is not None:
                    audio = audio_data.waveform
                    has_audio = True
            
            return VideoData(
                frames=frames,
                face_crops=face_crops,
                face_detections=face_detections,
                audio=audio,
                audio_sample_rate=self.audio_sample_rate,
                fps=fps,
                duration_seconds=duration,
                frame_indices=frame_indices,
                has_audio=has_audio,
                width=width,
                height=height
            )
            
        finally:
            os.unlink(temp_path)
    
    async def _get_video_metadata(self, path: str) -> Dict[str, Any]:
        """Get video metadata using ffprobe."""
        try:
            result = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await result.communicate()
            
            import json
            data = json.loads(stdout.decode())
            
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})
            
            # Parse frame rate
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": fps,
                "duration": float(fmt.get("duration", stream.get("duration", 0)))
            }
            
        except Exception as e:
            logger.warning(f"Failed to get video metadata: {e}")
            return {}
    
    async def _extract_frames(
        self,
        path: str,
        frame_sample_rate: int = 5,
        max_frames: int = 100
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from video using FFmpeg.
        
        Uses scene detection and keyframe sampling for efficiency.
        """
        frames = []
        frame_indices = []
        
        try:
            # Use ffmpeg to extract frames at sample rate
            with tempfile.TemporaryDirectory() as temp_dir:
                output_pattern = os.path.join(temp_dir, "frame_%06d.jpg")
                
                # Extract every Nth frame
                result = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i", path,
                    "-vf", f"select='not(mod(n\\,{frame_sample_rate}))',setpts=N/FRAME_RATE/TB",
                    "-frames:v", str(max_frames),
                    "-q:v", "2",
                    output_pattern,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await result.communicate()
                
                # Load extracted frames
                frame_files = sorted([
                    f for f in os.listdir(temp_dir)
                    if f.startswith("frame_")
                ])
                
                for i, fname in enumerate(frame_files):
                    fpath = os.path.join(temp_dir, fname)
                    img = Image.open(fpath)
                    frame = np.array(img)
                    frames.append(frame)
                    frame_indices.append(i * frame_sample_rate)
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise PreprocessingError("frame_extraction", str(e))
        
        return frames, frame_indices
    
    async def _detect_faces(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.8
    ) -> List[FaceDetection]:
        """
        Detect faces in frame using RetinaFace.
        
        Falls back to OpenCV Haar cascade if RetinaFace unavailable.
        """
        detections = []
        
        try:
            # Try RetinaFace first
            if self._face_detector is None:
                try:
                    from retinaface import RetinaFace
                    self._face_detector = RetinaFace
                except ImportError:
                    logger.warning("RetinaFace not available, using OpenCV")
                    self._face_detector = "opencv"
            
            if self._face_detector != "opencv":
                faces = self._face_detector.detect_faces(frame)
                
                for face_id, face_data in faces.items():
                    if face_data["score"] >= confidence_threshold:
                        bbox = face_data["facial_area"]
                        landmarks = face_data.get("landmarks", {})
                        
                        detections.append(FaceDetection(
                            bbox=tuple(bbox),
                            confidence=face_data["score"],
                            landmarks=landmarks
                        ))
            else:
                # OpenCV fallback
                import cv2
                
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    detections.append(FaceDetection(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.9,  # OpenCV doesn't provide confidence
                        landmarks=None
                    ))
                    
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
        
        return detections
    
    def _crop_face(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        padding: float = 0.2
    ) -> Optional[np.ndarray]:
        """
        Crop and resize face region from frame.
        
        Args:
            frame: Source frame
            detection: Face detection with bbox
            padding: Padding ratio around face
            
        Returns:
            Cropped face image or None
        """
        try:
            x1, y1, x2, y2 = detection.bbox
            h, w = frame.shape[:2]
            
            # Add padding
            face_w = x2 - x1
            face_h = y2 - y1
            
            pad_x = int(face_w * padding)
            pad_y = int(face_h * padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            # Crop
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
            
            # Resize to standard size
            pil_crop = Image.fromarray(crop)
            pil_crop = pil_crop.resize(self.face_crop_size, Image.LANCZOS)
            
            return np.array(pil_crop)
            
        except Exception as e:
            logger.warning(f"Face crop failed: {e}")
            return None
    
    async def _extract_audio_track(
        self,
        path: str
    ) -> Optional[AudioData]:
        """
        Extract audio track from video as 16kHz mono WAV.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_audio = f.name
            
            try:
                # Extract audio with ffmpeg
                result = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-i", path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(self.audio_sample_rate),
                    "-ac", "1",
                    temp_audio,
                    "-y",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                await result.communicate()
                
                if result.returncode != 0:
                    return None
                
                # Load audio
                import wave
                
                with wave.open(temp_audio, 'rb') as wav:
                    sample_rate = wav.getframerate()
                    channels = wav.getnchannels()
                    n_frames = wav.getnframes()
                    
                    audio_bytes = wav.readframes(n_frames)
                    waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    waveform = waveform / 32768.0  # Normalize to [-1, 1]
                    
                    duration = n_frames / sample_rate
                
                return AudioData(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    duration_seconds=duration,
                    channels=channels
                )
                
            finally:
                if os.path.exists(temp_audio):
                    os.unlink(temp_audio)
                    
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            return None
    
    async def extract_audio_file(
        self,
        audio_bytes: bytes
    ) -> AudioData:
        """
        Extract audio data from audio file.
        
        Args:
            audio_bytes: Raw audio file bytes
            
        Returns:
            AudioData with waveform
        """
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            result = await self._extract_audio_track(temp_path)
            
            if result is None:
                raise PreprocessingError("audio_extraction", "Failed to extract audio")
            
            return result
            
        finally:
            os.unlink(temp_path)
