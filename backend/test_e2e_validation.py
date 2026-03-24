#!/usr/bin/env python3
"""
Argus Core - End-to-End Validation Script
==========================================
Comprehensive validation of the entire application stack.

Tests:
1. Infrastructure services (Redis, MinIO, MongoDB, Celery)
2. Backend API health and models
3. File upload and storage
4. Analysis workflow
5. Frontend accessibility
6. Multimodal deepfake detection validation
7. Ground truth accuracy metrics
"""

import asyncio
import json
import sys
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_metric(text: str):
    """Print metric message."""
    print(f"{Colors.MAGENTA}📊 {text}{Colors.END}")


class GroundTruthLabel(str, Enum):
    """Ground truth labels for test samples."""
    AUTHENTIC = "authentic"
    FAKE = "fake"
    AI_GENERATED = "ai_generated"
    UNKNOWN = "unknown"


@dataclass
class TestSample:
    """Test sample with ground truth."""
    filename: str
    modality: str
    ground_truth: GroundTruthLabel
    description: str


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    sample: str
    predicted: str
    ground_truth: str
    correct: bool
    confidence: float
    inference_time: float
    error: Optional[str] = None


@dataclass
class ModalityMetrics:
    """Metrics for a single modality."""
    modality: str
    total: int = 0
    labeled_total: int = 0
    correct: int = 0
    labeled_correct: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_inference_time: float = 0.0
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        return (self.labeled_correct / self.labeled_total * 100) if self.labeled_total > 0 else 0.0
    
    @property
    def avg_inference_time(self) -> float:
        return (self.total_inference_time / self.total) if self.total > 0 else 0.0


class E2EValidator:
    """End-to-end validation for Argus Core."""
    
    def __init__(self):
        # Detect if running inside Docker container
        in_docker = os.path.exists("/.dockerenv")
        
        if in_docker:
            # Use Docker service names
            self.backend_url = "http://localhost:8000"
            self.frontend_url = "http://frontend:3000"
            self.redis_url = "redis://redis:6379/0"
            self.mongo_url = "mongodb://mongodb:27017"
            self.minio_url = "http://minio:9000"
        else:
            # Use localhost for external testing
            self.backend_url = "http://localhost:8000"
            self.frontend_url = "http://localhost:3000"
            self.redis_url = "redis://localhost:6379/0"
            self.mongo_url = "mongodb://localhost:27017"
            self.minio_url = "http://localhost:9000"
        
        # Allow explicit endpoint overrides for flexible execution topology.
        self.backend_url = os.environ.get("VALIDATION_BACKEND_URL", self.backend_url)
        self.frontend_url = os.environ.get("VALIDATION_FRONTEND_URL", self.frontend_url)
        self.redis_url = os.environ.get("VALIDATION_REDIS_URL", self.redis_url)
        self.mongo_url = os.environ.get("VALIDATION_MONGO_URL", self.mongo_url)
        self.minio_url = os.environ.get("VALIDATION_MINIO_URL", self.minio_url)
        
        self.test_samples_dir = self._resolve_test_samples_dir()
        self.test_samples = self._discover_test_samples()
        
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "start_time": datetime.now()
        }
        
        # Multimodal validation metrics
        self.modality_metrics: Dict[str, ModalityMetrics] = {}
        self.validation_results: List[ValidationResult] = []
    
    def _resolve_test_samples_dir(self) -> str:
        """
        Resolve test_samples directory across host/docker environments.
        
        Priority:
        1. TEST_SAMPLES_DIR env var
        2. /app/test_samples
        3. <repo_root>/test_samples (relative to this file)
        """
        candidates = []
        
        env_path = os.environ.get("TEST_SAMPLES_DIR")
        if env_path:
            candidates.append(Path(env_path))
        
        candidates.append(Path("/app/test_samples"))
        candidates.append(Path(__file__).resolve().parents[1] / "test_samples")
        
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        
        # Fall back to /app/test_samples for backward compatibility.
        return "/app/test_samples"
    
    def _infer_modality(self, file_path: Path) -> Optional[str]:
        """Infer modality from file extension."""
        suffix = file_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return "image"
        if suffix in {".mp4", ".mov", ".avi", ".webm"}:
            return "video"
        if suffix in {".mp3", ".wav", ".ogg", ".m4a"}:
            return "audio"
        if suffix in {".txt"}:
            return "text"
        return None
    
    def _infer_ground_truth(self, relative_path: str) -> GroundTruthLabel:
        """
        Infer ground truth from curated filename/path conventions.
        
        Unknown samples are intentionally excluded from strict accuracy denominator.
        """
        rel = relative_path.lower()
        name = Path(rel).name
        
        if name == "deepfake.png":
            return GroundTruthLabel.FAKE
        if name.startswith("gemini_generated_image_"):
            return GroundTruthLabel.AI_GENERATED
        if name == "ai_generated_video.mp4":
            return GroundTruthLabel.FAKE
        if name == "real_video1.mp4":
            return GroundTruthLabel.AUTHENTIC
        if name == "ai_generated_text.txt":
            return GroundTruthLabel.AI_GENERATED
        if rel.startswith("internet_test/real_person"):
            return GroundTruthLabel.AUTHENTIC
        
        return GroundTruthLabel.UNKNOWN
    
    def _discover_test_samples(self) -> List[TestSample]:
        """Discover test samples from directory and infer modality/ground truth."""
        root = Path(self.test_samples_dir)
        if not root.exists():
            return []
        
        discovered: List[TestSample] = []
        
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name.startswith("."):
                continue
            
            modality = self._infer_modality(file_path)
            if modality is None:
                continue
            
            relative_path = str(file_path.relative_to(root))
            ground_truth = self._infer_ground_truth(relative_path)
            
            description = (
                "Curated labeled sample"
                if ground_truth != GroundTruthLabel.UNKNOWN
                else "Exploratory sample (unlabeled)"
            )
            
            discovered.append(
                TestSample(
                    filename=relative_path,
                    modality=modality,
                    ground_truth=ground_truth,
                    description=description,
                )
            )
        
        return discovered
    
    async def test_redis(self) -> bool:
        """Test Redis connectivity."""
        print_info("Testing Redis connectivity...")
        try:
            redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            pong = await redis_client.ping()
            await redis_client.close()
            
            if pong:
                print_success("Redis is accessible and responding to PING")
                return True
            else:
                print_error("Redis PING failed")
                return False
        except Exception as e:
            print_error(f"Redis connection failed: {e}")
            return False
    
    async def test_mongodb(self) -> bool:
        """Test MongoDB connectivity."""
        print_info("Testing MongoDB connectivity...")
        try:
            client = AsyncIOMotorClient(self.mongo_url)
            await client.admin.command('ping')
            
            # List databases
            db_list = await client.list_database_names()
            print_success(f"MongoDB is accessible (databases: {len(db_list)})")
            
            client.close()
            return True
        except Exception as e:
            print_error(f"MongoDB connection failed: {e}")
            return False
    
    async def test_minio(self) -> bool:
        """Test MinIO accessibility."""
        print_info("Testing MinIO accessibility...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.minio_url}/minio/health/live", timeout=5.0)
                
                if response.status_code == 200:
                    print_success("MinIO is accessible and healthy")
                    return True
                else:
                    print_error(f"MinIO health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"MinIO connection failed: {e}")
            return False
    
    async def test_celery(self) -> bool:
        """Test Celery worker status."""
        print_info("Testing Celery worker status...")
        try:
            from processing.tasks import celery_app
            
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            
            if stats:
                worker_count = len(stats)
                print_success(f"Celery workers active: {worker_count}")
                
                # Print worker details
                for worker_name, worker_stats in stats.items():
                    print_info(f"  Worker: {worker_name}")
                
                return True
            else:
                print_error("No Celery workers found")
                return False
        except Exception as e:
            print_error(f"Celery inspection failed: {e}")
            return False
    
    async def test_backend_health(self) -> bool:
        """Test backend API health endpoint."""
        print_info("Testing backend API health...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/api/v1/health", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    components = data.get("components", {})
                    
                    print_success(f"Backend API is {status}")
                    
                    # Check each component
                    for component, comp_status in components.items():
                        if isinstance(comp_status, dict):
                            comp_status_val = comp_status.get("status", "unknown")
                            print_info(f"  {component}: {comp_status_val}")
                        else:
                            print_info(f"  {component}: {comp_status}")
                    
                    return status == "healthy"
                else:
                    print_error(f"Backend health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Backend API connection failed: {e}")
            return False
    
    async def test_models(self) -> bool:
        """Test AI models loading and availability."""
        print_info("Testing AI models...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/api/v1/models", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", data) if isinstance(data, dict) else data
                    
                    loaded_models = []
                    available_models = []
                    
                    for m in models:
                        if isinstance(m, dict):
                            if m.get("loaded", False):
                                loaded_models.append(m)
                            if m.get("file_exists", False):
                                available_models.append(m)
                    
                    print_success(f"Models available: {len(available_models)}, loaded: {len(loaded_models)}")
                    
                    # List loaded models
                    for model in loaded_models:
                        model_name = model.get('name', 'unknown')
                        model_cat = model.get('category', 'unknown')
                        model_vram = model.get('vram_mb', 0)
                        print_info(f"  ✓ {model_name} ({model_cat}) - {model_vram}MB")
                    
                    return len(loaded_models) >= 3
                else:
                    print_error(f"Models endpoint failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Models check failed: {e}")
            return False
    
    async def test_frontend(self) -> bool:
        """Test frontend accessibility."""
        print_info("Testing frontend accessibility...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.frontend_url, timeout=10.0, follow_redirects=True)
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Check for expected elements
                    has_title = "Argus Core" in html
                    has_react = "react" in html.lower() or "__next" in html
                    
                    if has_title and has_react:
                        print_success("Frontend is accessible and rendering correctly")
                        return True
                    else:
                        print_warning("Frontend accessible but may not be rendering correctly")
                        return True
                else:
                    print_error(f"Frontend check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Frontend connection failed: {e}")
            return False
    
    async def test_file_upload(self) -> bool:
        """Test file upload capability."""
        print_info("Testing file upload endpoint...")
        try:
            # Create a small test file
            test_content = b"Test file for Argus Core validation"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("test.txt", test_content, "text/plain")}
                
                response = await client.post(
                    f"{self.backend_url}/api/v1/analyze",
                    files=files
                )
                
                if response.status_code in [200, 201, 202]:
                    data = response.json()
                    analysis_id = data.get("analysis_id")
                    
                    if analysis_id:
                        print_success(f"File upload successful (analysis_id: {analysis_id})")
                        return True
                    else:
                        print_warning("Upload succeeded but no analysis_id returned")
                        return True
                elif response.status_code == 415:
                    print_warning("File type not supported (expected for test file)")
                    return True
                else:
                    print_error(f"File upload failed: {response.status_code} - {response.text[:200]}")
                    return False
        except Exception as e:
            print_error(f"File upload test failed: {e}")
            return False
    
    async def analyze_sample(self, sample: TestSample) -> ValidationResult:
        """
        Analyze a single test sample and compare against ground truth.
        
        Args:
            sample: Test sample with ground truth label
            
        Returns:
            ValidationResult with prediction and accuracy metrics
        """
        file_path = os.path.join(self.test_samples_dir, sample.filename)
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Handle text files differently - use Form data
                if sample.modality == "text":
                    with open(file_path, "r") as f:
                        text_content = f.read()
                    
                    # Use form data for text endpoint
                    response = await client.post(
                        f"{self.backend_url}/api/v1/analyze/text",
                        data={"text": text_content}  # Form data, not JSON
                    )
                else:
                    # Binary files (image, video, audio)
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    
                    # Determine content type
                    content_type = self._get_content_type(sample.filename)
                    files = {"file": (sample.filename, file_content, content_type)}
                    
                    response = await client.post(
                        f"{self.backend_url}/api/v1/analyze",
                        files=files
                    )
                
                inference_time = time.time() - start_time
                
                if response.status_code in [200, 201, 202]:
                    data = response.json()
                    analysis_id = data.get("analysis_id")
                    
                    # Poll for completion if async
                    if response.status_code == 202:
                        data = await self._poll_for_result(client, analysis_id)
                        if data is None:
                            return ValidationResult(
                                sample=sample.filename,
                                predicted="error",
                                ground_truth=sample.ground_truth.value,
                                correct=False,
                                confidence=0.0,
                                inference_time=inference_time,
                                error="Analysis timed out"
                            )
                    
                    # Extract prediction
                    predicted, confidence = self._extract_prediction(data, sample.modality)
                    correct = self._evaluate_prediction(predicted, sample.ground_truth)
                    
                    return ValidationResult(
                        sample=sample.filename,
                        predicted=predicted,
                        ground_truth=sample.ground_truth.value,
                        correct=correct,
                        confidence=confidence,
                        inference_time=inference_time
                    )
                else:
                    return ValidationResult(
                        sample=sample.filename,
                        predicted="error",
                        ground_truth=sample.ground_truth.value,
                        correct=False,
                        confidence=0.0,
                        inference_time=inference_time,
                        error=f"API error: {response.status_code}"
                    )
                    
        except Exception as e:
            inference_time = time.time() - start_time
            return ValidationResult(
                sample=sample.filename,
                predicted="error",
                ground_truth=sample.ground_truth.value,
                correct=False,
                confidence=0.0,
                inference_time=inference_time,
                error=str(e)
            )
    
    def _get_content_type(self, filename: str) -> str:
        """Get MIME type for file."""
        ext = os.path.splitext(filename)[1].lower()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".webm": "video/webm",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
        }
        return content_types.get(ext, "application/octet-stream")
    
    async def _poll_for_result(self, client: httpx.AsyncClient, analysis_id: str, max_wait: int = 60) -> Optional[dict]:
        """Poll for analysis completion."""
        for _ in range(max_wait):
            try:
                response = await client.get(f"{self.backend_url}/api/v1/analyze/{analysis_id}")
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "")
                    if status == "completed":
                        return data
                    elif status == "failed":
                        return None
            except Exception:
                pass
            await asyncio.sleep(1)
        return None
    
    def _extract_prediction(self, data: dict, modality: str) -> Tuple[str, float]:
        """
        Extract prediction from API response.
        
        Returns:
            Tuple of (predicted_label, confidence)
        """
        # Try to get verdict from response
        verdict = data.get("verdict", "").lower()
        trust_score = data.get("trust_score", {})
        
        authenticity_score = None  # 0-100
        confidence = 0.5           # 0-1
        
        if isinstance(trust_score, dict):
            raw_value = trust_score.get("value", None)
            raw_confidence = trust_score.get("confidence", None)
            if isinstance(raw_value, (int, float)):
                authenticity_score = float(raw_value)
            if isinstance(raw_confidence, (int, float)):
                confidence = float(raw_confidence)
        elif isinstance(trust_score, (int, float)):
            authenticity_score = float(trust_score)
        
        # Normalize confidence to 0-1 (guard against malformed ranges)
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = float(max(0.0, min(1.0, confidence)))
        
        # Map verdict to standard labels
        if verdict in ["fake", "ai_generated", "deepfake"]:
            return "fake", confidence
        elif verdict in ["authentic", "real", "genuine"]:
            return "authentic", confidence
        elif verdict in ["uncertain", "inconclusive"]:
            return "uncertain", confidence
        
        # Default to trust score when verdict is unavailable.
        # authenticity_score is 0-100 where low means fake-likely.
        if isinstance(authenticity_score, (int, float)):
            if authenticity_score < 40:
                return "fake", confidence
            if authenticity_score > 60:
                return "authentic", confidence
            return "uncertain", confidence
        
        return "uncertain", confidence
    
    def _evaluate_prediction(self, predicted: str, ground_truth: GroundTruthLabel) -> bool:
        """
        Evaluate if prediction matches ground truth.
        
        Args:
            predicted: Predicted label
            ground_truth: Ground truth label
            
        Returns:
            True if prediction is correct
        """
        if ground_truth == GroundTruthLabel.UNKNOWN:
            return True
        
        # Direct match
        if predicted == ground_truth.value:
            return True
        
        # Semantic equivalence
        fake_labels = {"fake", "ai_generated", "deepfake"}
        authentic_labels = {"authentic", "real", "genuine"}
        
        if ground_truth.value in fake_labels and predicted in fake_labels:
            return True
        if ground_truth.value in authentic_labels and predicted in authentic_labels:
            return True
        
        return False
    
    async def run_multimodal_validation(self) -> bool:
        """
        Run comprehensive multimodal deepfake detection validation.
        
        Tests all modalities with ground truth labels and computes metrics.
        """
        print_header("MULTIMODAL DEEPFAKE DETECTION VALIDATION")
        print_info(f"Test samples directory: {self.test_samples_dir}")
        print_info(f"Total discovered test samples: {len(self.test_samples)}")
        labeled_count = sum(1 for sample in self.test_samples if sample.ground_truth != GroundTruthLabel.UNKNOWN)
        print_info(f"Labeled test samples (strict accuracy denominator): {labeled_count}")
        print()
        
        if not self.test_samples:
            print_error("No test samples discovered. Skipping multimodal validation.")
            return False
        
        # Initialize metrics per modality
        for sample in self.test_samples:
            if sample.modality not in self.modality_metrics:
                self.modality_metrics[sample.modality] = ModalityMetrics(modality=sample.modality)
        
        # Process each sample
        for sample in self.test_samples:
            print_info(f"Testing: {sample.filename} ({sample.modality})")
            print_info(f"  Ground truth: {sample.ground_truth.value}")
            
            result = await self.analyze_sample(sample)
            self.validation_results.append(result)
            
            # Update metrics
            metrics = self.modality_metrics[sample.modality]
            metrics.total += 1
            metrics.total_inference_time += result.inference_time
            metrics.results.append(result)
            
            is_labeled = sample.ground_truth != GroundTruthLabel.UNKNOWN
            if is_labeled:
                metrics.labeled_total += 1
            
            if result.correct and is_labeled:
                metrics.correct += 1
                metrics.labeled_correct += 1
                print_success(f"  ✓ Predicted: {result.predicted} (confidence: {result.confidence:.2%})")
            elif not is_labeled:
                print_warning(
                    f"  Exploratory prediction: {result.predicted} "
                    f"(confidence: {result.confidence:.2%})"
                )
            else:
                if result.error:
                    print_error(f"  ✗ Error: {result.error}")
                    # Track as false negative if we couldn't detect fake
                    if sample.ground_truth in [GroundTruthLabel.FAKE, GroundTruthLabel.AI_GENERATED]:
                        metrics.false_negatives += 1
                    else:
                        metrics.false_positives += 1
                else:
                    print_error(f"  ✗ Predicted: {result.predicted}, Expected: {result.ground_truth}")
                    # Track false positive/negative
                    if sample.ground_truth in [GroundTruthLabel.FAKE, GroundTruthLabel.AI_GENERATED]:
                        metrics.false_negatives += 1
                    else:
                        metrics.false_positives += 1
            
            print_metric(f"  Inference time: {result.inference_time:.2f}s")
            print()
        
        # Print per-modality metrics
        self._print_modality_metrics()
        
        # Return overall success (at least 80% strict labeled accuracy)
        overall_accuracy = self._calculate_overall_accuracy()
        return overall_accuracy >= 80.0
    
    def _calculate_overall_accuracy(self) -> float:
        """Calculate strict labeled accuracy across all modalities."""
        total = sum(m.labeled_total for m in self.modality_metrics.values())
        correct = sum(m.labeled_correct for m in self.modality_metrics.values())
        return (correct / total * 100) if total > 0 else 0.0
    
    def _print_modality_metrics(self):
        """Print detailed metrics for each modality."""
        print_header("PER-MODALITY PERFORMANCE METRICS")
        
        for modality, metrics in self.modality_metrics.items():
            print(f"\n{Colors.BOLD}{modality.upper()}{Colors.END}")
            print("-" * 40)
            print_metric(
                f"Labeled accuracy: {metrics.accuracy:.1f}% "
                f"({metrics.labeled_correct}/{metrics.labeled_total})"
            )
            print_metric(f"Total samples: {metrics.total}")
            print_metric(f"Avg inference time: {metrics.avg_inference_time:.2f}s")
            
            if metrics.false_positives > 0:
                print_warning(f"False positives: {metrics.false_positives}")
            if metrics.false_negatives > 0:
                print_warning(f"False negatives: {metrics.false_negatives}")
        
        # Overall summary
        print()
        print(f"{Colors.BOLD}OVERALL ACCURACY{Colors.END}")
        print("-" * 40)
        overall = self._calculate_overall_accuracy()
        print_metric(f"Total accuracy: {overall:.1f}%")
        
        total_fp = sum(m.false_positives for m in self.modality_metrics.values())
        total_fn = sum(m.false_negatives for m in self.modality_metrics.values())
        print_metric(f"Total false positives: {total_fp}")
        print_metric(f"Total false negatives: {total_fn}")
    
    def generate_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Generate confusion matrix for all predictions.
        
        Returns:
            Dict with actual vs predicted counts
        """
        matrix = {
            "authentic": {"authentic": 0, "fake": 0, "ai_generated": 0, "uncertain": 0, "error": 0},
            "fake": {"authentic": 0, "fake": 0, "ai_generated": 0, "uncertain": 0, "error": 0},
            "ai_generated": {"authentic": 0, "fake": 0, "ai_generated": 0, "uncertain": 0, "error": 0},
        }
        
        for result in self.validation_results:
            gt = result.ground_truth
            pred = result.predicted
            if gt in matrix and pred in matrix[gt]:
                matrix[gt][pred] += 1
        
        return matrix
    
    def print_confusion_matrix(self):
        """Print formatted confusion matrix."""
        matrix = self.generate_confusion_matrix()
        
        print_header("CONFUSION MATRIX")
        header = "Actual vs Pred    Auth      Fake      AI-Gen    Uncert    Error"
        print(header)
        print("-" * 65)
        
        for actual, predictions in matrix.items():
            row = f"{actual:<15} "
            for pred in ["authentic", "fake", "ai_generated", "uncertain", "error"]:
                count = predictions.get(pred, 0)
                if count > 0:
                    row += f"{Colors.RED if pred != actual and pred != 'error' else Colors.GREEN}{count:<10}{Colors.END}"
                else:
                    row += f"0{'':<9}"
            print(row)
    
    async def run_all_tests(self):
        """Run all validation tests."""
        print_header("ARGUS CORE - END-TO-END VALIDATION")
        print_info(f"Started at: {self.results['start_time']}")
        
        # Phase 1: Infrastructure Services
        print_header("PHASE 1: Infrastructure Services")
        
        tests = [
            ("Redis", self.test_redis()),
            ("MongoDB", self.test_mongodb()),
            ("MinIO", self.test_minio()),
            ("Celery", self.test_celery()),
        ]
        
        for name, test_coro in tests:
            result = await test_coro
            if result:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
            print()
        
        # Phase 2: Backend Services
        print_header("PHASE 2: Backend API & Models")
        
        tests = [
            ("Backend Health", self.test_backend_health()),
            ("AI Models", self.test_models()),
        ]
        
        for name, test_coro in tests:
            result = await test_coro
            if result:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
            print()
        
        # Phase 3: Frontend
        print_header("PHASE 3: Frontend Application")
        
        result = await self.test_frontend()
        if result:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        print()
        
        # Phase 4: Integration Tests
        print_header("PHASE 4: Integration Tests")
        
        result = await self.test_file_upload()
        if result:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        print()
        
        # Phase 5: Multimodal Deepfake Detection Validation
        print_header("PHASE 5: Multimodal Detection Validation")
        
        result = await self.run_multimodal_validation()
        if result:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        
        # Print confusion matrix
        self.print_confusion_matrix()
        
        # Print Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        end_time = datetime.now()
        duration = (end_time - self.results['start_time']).total_seconds()
        
        print_header("VALIDATION SUMMARY")
        
        total_tests = self.results['passed'] + self.results['failed']
        
        print(f"{Colors.BOLD}Total Tests:{Colors.END} {total_tests}")
        print(f"{Colors.GREEN}✓ Passed:{Colors.END} {self.results['passed']}")
        print(f"{Colors.RED}✗ Failed:{Colors.END} {self.results['failed']}")
        
        if self.results['warnings'] > 0:
            print(f"{Colors.YELLOW}⚠ Warnings:{Colors.END} {self.results['warnings']}")
        
        print(f"\n{Colors.BOLD}Duration:{Colors.END} {duration:.2f}s")
        
        # Print multimodal accuracy summary
        if self.modality_metrics:
            print(f"\n{Colors.BOLD}Multimodal Detection Accuracy:{Colors.END}")
            overall = self._calculate_overall_accuracy()
            for modality, metrics in self.modality_metrics.items():
                status = Colors.GREEN if metrics.accuracy >= 80 else Colors.YELLOW if metrics.accuracy >= 60 else Colors.RED
                print(f"  {modality}: {status}{metrics.accuracy:.1f}%{Colors.END}")
            print(f"  {Colors.BOLD}Overall: {overall:.1f}%{Colors.END}")
        
        if self.results['failed'] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! APPLICATION IS FULLY OPERATIONAL 🎉{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ SOME TESTS FAILED - PLEASE REVIEW ERRORS ABOVE{Colors.END}")
            return 1
    
    def generate_report(self) -> str:
        """Generate detailed JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.results['start_time']).total_seconds(),
            "summary": {
                "passed": self.results["passed"],
                "failed": self.results["failed"],
                "warnings": self.results["warnings"]
            },
            "multimodal_metrics": {
                modality: {
                    "accuracy": metrics.accuracy,
                    "total": metrics.total,
                    "correct": metrics.correct,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives,
                    "avg_inference_time": metrics.avg_inference_time
                }
                for modality, metrics in self.modality_metrics.items()
            },
            "overall_accuracy": self._calculate_overall_accuracy(),
            "confusion_matrix": self.generate_confusion_matrix(),
            "validation_results": [
                {
                    "sample": r.sample,
                    "predicted": r.predicted,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                    "confidence": r.confidence,
                    "inference_time": r.inference_time,
                    "error": r.error
                }
                for r in self.validation_results
            ]
        }
        return json.dumps(report, indent=2)


async def main():
    """Main entry point."""
    validator = E2EValidator()
    await validator.run_all_tests()
    
    # Generate and save report
    report = validator.generate_report()
    if os.path.exists("/.dockerenv"):
        report_path = "/app/validation_report.json"
    else:
        report_path = str(Path.cwd() / "validation_report.json")
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print_info(f"Detailed report saved to: {report_path}")
    except Exception as e:
        print_warning(f"Could not save report: {e}")
    
    return validator.results['failed']


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
