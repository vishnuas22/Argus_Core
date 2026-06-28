"""
Argus Core - Production Model Downloader
========================================
Download and export pretrained models from HuggingFace to ONNX format.

Implements: plans/MODEL_ARCHITECTURE_REALIGNMENT.md - Phase 2.2

Features:
- Downloads models from HuggingFace Hub
- Exports PyTorch models to ONNX format
- Supports Audio (AASIST, Wav2Vec2), Video (X-CLIP)
- Validates model integrity
- GPU/CPU fallback

Usage:
    python -m models.model_downloader --all
    python -m models.model_downloader --modality audio
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

# Set environment variables before importing torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be limited.")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX/ONNX Runtime not available.")

try:
    from transformers import (
        AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM,
        AutoTokenizer, AutoFeatureExtractor, AutoProcessor,
        Wav2Vec2Model, Wav2Vec2ForCTC, RobertaForSequenceClassification,
        GPT2LMHeadModel, AutoConfig
    )
    from huggingface_hub import hf_hub_download, login, HfApi
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Model download will be limited.")

from utils.logging import get_logger
from config import config

logger = get_logger(__name__)


# ============== MODEL CONFIGURATIONS ==============

@dataclass
class ModelDownloadConfig:
    """Configuration for downloading and exporting a model."""
    name: str
    hf_model_id: str
    model_type: str  # "audio", "video", "text", "image"
    task: str  # "classification", "feature_extraction", "detection"
    output_path: str
    input_shapes: Dict[str, Tuple[int, ...]]
    output_names: List[str]
    num_classes: int = 2
    class_labels: List[str] = None
    max_seq_length: int = 512
    audio_sample_rate: int = 16000
    onnx_opset: int = 14
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    requires_tokenizer: bool = False
    requires_processor: bool = False
    hf_onnx_filename: Optional[str] = None  # Exact ONNX filename in HuggingFace repo (direct download)
    description: str = ""
    license: str = "Apache-2.0"
    academic_reference: str = ""
    
    def __post_init__(self):
        if self.class_labels is None:
            self.class_labels = ["real", "fake"]


# Production model configurations with verified HuggingFace sources
MODEL_CONFIGS: Dict[str, ModelDownloadConfig] = {
    # ============== AUDIO MODELS ==============
    "aasist_antispoof": ModelDownloadConfig(
        name="aasist_antispoof",
        hf_model_id="clovaai/aasist-l",
        model_type="audio",
        task="detection",
        output_path="/models/aasist_antispoof.onnx",
        input_shapes={"audio": (1, 64600)},  # ~4 seconds at 16kHz
        output_names=["logits"],
        num_classes=2,
        class_labels=["bonafide", "spoof"],
        audio_sample_rate=16000,
        description="AASIST - ASVspoof 2021 winner for synthetic voice detection",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2110.01216"
    ),
    
    "wav2vec2_base": ModelDownloadConfig(
        name="wav2vec2_base",
        hf_model_id="facebook/wav2vec2-base-960h",
        model_type="audio",
        task="feature_extraction",
        output_path="/models/wav2vec2_base.onnx",
        input_shapes={"input_values": (1, 16000)},
        output_names=["last_hidden_state", "extract_features"],
        num_classes=0,
        audio_sample_rate=16000,
        description="Wav2Vec2 base model for audio feature extraction",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2006.11477"
    ),
    
    "wav2vec2_large": ModelDownloadConfig(
        name="wav2vec2_large",
        hf_model_id="facebook/wav2vec2-large-960h",
        model_type="audio",
        task="feature_extraction",
        output_path="/models/wav2vec2_large.onnx",
        input_shapes={"input_values": (1, 16000)},
        output_names=["last_hidden_state"],
        num_classes=0,
        audio_sample_rate=16000,
        description="Wav2Vec2 large model for audio feature extraction",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2006.11477"
    ),

    "wav2vec2_antispoof": ModelDownloadConfig(
        name="wav2vec2_antispoof",
        hf_model_id="pranjal-pravesh/wav2vec2-large-xlsr-deepfake-audio-classification",
        hf_onnx_filename="model_int8.onnx",
        model_type="audio",
        task="classification",
        output_path="/models/wav2vec2_antispoof.onnx",
        input_shapes={"input_values": (1, 64600)},
        output_names=["logits"],
        num_classes=2,
        class_labels=["bonafide", "spoof"],
        audio_sample_rate=16000,
        description="Wav2Vec2 Large XLSR fine-tuned for audio deepfake detection (ASVspoof2019, 4.01% EER). INT8 quantized ONNX.",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2006.11477"
    ),

    # ============== VIDEO MODELS ==============
    "xclip_temporal": ModelDownloadConfig(
        name="xclip_temporal",
        hf_model_id="microsoft/xclip-base-patch16",
        model_type="video",
        task="feature_extraction",
        output_path="/models/xclip_temporal.onnx",
        input_shapes={
            "pixel_values": (1, 16, 3, 224, 224),  # 16 frames
            "input_ids": (1, 77)  # Text tokens
        },
        output_names=["logits_per_video", "logits_per_text"],
        num_classes=0,
        requires_processor=True,
        description="X-CLIP for video-text understanding and temporal analysis",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2207.07285"
    ),
    
    "videomae_feature": ModelDownloadConfig(
        name="videomae_feature",
        hf_model_id="MCG-NJU/videomae-base",
        model_type="video",
        task="feature_extraction",
        output_path="/models/videomae_feature.onnx",
        input_shapes={
            "pixel_values": (1, 16, 3, 224, 224)
        },
        output_names=["last_hidden_state"],
        num_classes=0,
        description="VideoMAE for video feature extraction",
        license="Apache-2.0",
        academic_reference="https://arxiv.org/abs/2203.12602"
    ),
    
    # ============== FEATURE EXTRACTION MODELS ==============
    "clip_vit_b16": ModelDownloadConfig(
        name="clip_vit_b16",
        hf_model_id="openai/clip-vit-base-patch16",
        model_type="image",
        task="feature_extraction",
        output_path="/models/clip_vit_b16.onnx",
        input_shapes={
            "pixel_values": (1, 3, 224, 224),
            "input_ids": (1, 77)
        },
        output_names=["logits_per_image", "logits_per_text"],
        num_classes=0,
        description="CLIP ViT-Base for image-text feature extraction",
        license="MIT",
        academic_reference="https://arxiv.org/abs/2103.00020"
    ),
}


class ProductionModelDownloader:
    """
    Download and export pretrained models from HuggingFace.
    
    Supports:
    - Direct ONNX download from HuggingFace
    - PyTorch model export to ONNX
    - Model validation and verification
    - GPU/CPU fallback
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_gpu: bool = True,
        verify_integrity: bool = True
    ):
        """
        Initialize the model downloader.
        
        Args:
            model_dir: Directory to save models
            use_gpu: Whether to use GPU for export
            verify_integrity: Whether to verify model integrity after download
        """
        self.model_dir = Path(model_dir or config.model_cache_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and torch.cuda.is_available() if TORCH_AVAILABLE else False
        self.device = torch.device("cuda" if self.use_gpu else "cpu") if TORCH_AVAILABLE else "cpu"
        self.verify_integrity = verify_integrity
        
        logger.info(f"ProductionModelDownloader initialized: model_dir={self.model_dir}, device={self.device}")
    
    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def download_onnx_direct(
        self,
        model_config: ModelDownloadConfig
    ) -> Tuple[bool, str]:
        """
        Download ONNX model directly from HuggingFace.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Tuple of (success, message)
        """
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers library not available"
        
        try:
            # Try to download ONNX file from HuggingFace
            onnx_filename = f"{model_config.name}.onnx"
            
            # Common ONNX file names in HuggingFace repos
            possible_names = []
            if model_config.hf_onnx_filename:
                possible_names.append(model_config.hf_onnx_filename)
            possible_names += [
                onnx_filename,
                "model.onnx",
                "onnx/model.onnx",
                f"{model_config.name.split('_')[0]}.onnx"
            ]
            
            for filename in possible_names:
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_config.hf_model_id,
                        filename=filename,
                        local_dir=self.model_dir,
                        local_dir_use_symlinks=False
                    )
                    
                    # Move to target path
                    target_path = self.model_dir / f"{model_config.name}.onnx"
                    if Path(downloaded_path) != target_path:
                        import shutil
                        shutil.move(downloaded_path, target_path)
                    
                    file_size_mb = target_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Downloaded {model_config.name}: {file_size_mb:.1f}MB")
                    
                    return True, f"Downloaded {file_size_mb:.1f}MB from HuggingFace"
                    
                except Exception as e:
                    logger.debug(f"File {filename} not found in {model_config.hf_model_id}: {e}")
                    continue
            
            return False, "ONNX file not found in HuggingFace repo"
            
        except Exception as e:
            return False, f"Download failed: {str(e)}"
    
    def export_pytorch_to_onnx(
        self,
        model_config: ModelDownloadConfig
    ) -> Tuple[bool, str]:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Tuple of (success, message)
        """
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            return False, "PyTorch/Transformers not available"
        
        try:
            logger.info(f"Loading {model_config.name} from {model_config.hf_model_id}")
            
            # Load appropriate model class based on task
            model = None
            tokenizer = None
            
            if model_config.task == "classification":
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_config.hf_model_id,
                        torch_dtype=torch.float32
                    )
                except Exception:
                    # Try as causal LM for perplexity models
                    model = GPT2LMHeadModel.from_pretrained(
                        model_config.hf_model_id,
                        torch_dtype=torch.float32
                    )
            elif model_config.task == "feature_extraction":
                if model_config.model_type == "audio":
                    model = Wav2Vec2Model.from_pretrained(
                        model_config.hf_model_id,
                        torch_dtype=torch.float32
                    )
                elif model_config.model_type == "video":
                    try:
                        from transformers import AutoModelForVideoClassification
                        model = AutoModelForVideoClassification.from_pretrained(
                            model_config.hf_model_id,
                            torch_dtype=torch.float32
                        )
                    except Exception:
                        model = AutoModel.from_pretrained(
                            model_config.hf_model_id,
                            torch_dtype=torch.float32
                        )
                else:
                    model = AutoModel.from_pretrained(
                        model_config.hf_model_id,
                        torch_dtype=torch.float32
                    )
            elif model_config.task == "detection":
                # For audio deepfake detection models
                model = AutoModel.from_pretrained(
                    model_config.hf_model_id,
                    torch_dtype=torch.float32
                )
            else:
                model = AutoModel.from_pretrained(
                    model_config.hf_model_id,
                    torch_dtype=torch.float32
                )
            
            model = model.to(self.device)
            model.eval()
            
            # Prepare dummy inputs
            dummy_inputs = self._create_dummy_inputs(model_config)
            
            # Export to ONNX
            output_path = self.model_dir / f"{model_config.name}.onnx"
            
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()) if isinstance(dummy_inputs, dict) else dummy_inputs,
                    str(output_path),
                    input_names=list(dummy_inputs.keys()) if isinstance(dummy_inputs, dict) else None,
                    output_names=model_config.output_names,
                    dynamic_axes=model_config.dynamic_axes,
                    do_constant_folding=True,
                    opset_version=model_config.onnx_opset,
                    verbose=False
                )
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Exported {model_config.name} to ONNX: {file_size_mb:.1f}MB")
            
            return True, f"Exported to ONNX ({file_size_mb:.1f}MB)"
            
        except Exception as e:
            logger.error(f"Failed to export {model_config.name}: {e}")
            return False, f"Export failed: {str(e)}"
    
    def _create_dummy_inputs(
        self,
        model_config: ModelDownloadConfig
    ) -> Dict[str, torch.Tensor]:
        """Create dummy inputs for ONNX export."""
        dummy_inputs = {}
        
        for input_name, shape in model_config.input_shapes.items():
            if "input_ids" in input_name or "attention_mask" in input_name:
                # Integer inputs for text
                dummy_inputs[input_name] = torch.randint(0, 50000, shape, dtype=torch.long, device=self.device)
            elif "audio" in input_name or "input_values" in input_name:
                # Float inputs for audio
                dummy_inputs[input_name] = torch.randn(shape, dtype=torch.float32, device=self.device)
            elif "pixel_values" in input_name:
                # Float inputs for images/video
                dummy_inputs[input_name] = torch.randn(shape, dtype=torch.float32, device=self.device)
            else:
                # Default float
                dummy_inputs[input_name] = torch.randn(shape, dtype=torch.float32, device=self.device)
        
        return dummy_inputs
    
    def validate_onnx_model(
        self,
        model_config: ModelDownloadConfig
    ) -> Tuple[bool, str]:
        """
        Validate ONNX model integrity.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Tuple of (valid, message)
        """
        if not ONNX_AVAILABLE:
            return True, "ONNX validation skipped (not available)"
        
        model_path = self.model_dir / f"{model_config.name}.onnx"
        
        if not model_path.exists():
            return False, "Model file not found"
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            
            # Try inference with ONNX Runtime
            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            
            # Get input info
            input_info = {inp.name: inp.shape for inp in session.get_inputs()}
            logger.debug(f"Model inputs: {input_info}")
            
            # Run test inference
            dummy_inputs = self._create_dummy_inputs_for_inference(model_config, session)
            outputs = session.run(None, dummy_inputs)
            
            # Verify output shapes
            output_info = [(out.name, out.shape) for out in session.get_outputs()]
            logger.debug(f"Model outputs: {output_info}")
            
            return True, f"Valid ONNX model with {len(output_info)} outputs"
            
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    def _create_dummy_inputs_for_inference(
        self,
        model_config: ModelDownloadConfig,
        session: ort.InferenceSession
    ) -> Dict[str, Any]:
        """Create dummy inputs for ONNX Runtime inference."""
        import numpy as np
        dummy_inputs = {}
        
        for inp in session.get_inputs():
            shape = inp.shape
            # Replace dynamic dimensions with 1
            shape = [1 if isinstance(d, str) else d for d in shape]
            
            if inp.type == 'tensor(int64)' or 'int64' in inp.type:
                dummy_inputs[inp.name] = np.random.randint(0, 50000, shape).astype(np.int64)
            else:
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        return dummy_inputs
    
    def download_model(
        self,
        model_name: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Download and prepare a model.
        
        Args:
            model_name: Name of the model to download
            force: Force re-download even if exists
            
        Returns:
            Dict with download results
        """
        if model_name not in MODEL_CONFIGS:
            return {
                "model": model_name,
                "success": False,
                "message": f"Unknown model: {model_name}",
                "available_models": list(MODEL_CONFIGS.keys())
            }
        
        model_config = MODEL_CONFIGS[model_name]
        output_path = self.model_dir / f"{model_name}.onnx"
        
        start_time = time.time()
        
        # Check if already exists
        if output_path.exists() and not force:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1:  # More than 1MB - likely a real model
                # Validate existing model
                valid, msg = self.validate_onnx_model(model_config)
                if valid:
                    return {
                        "model": model_name,
                        "success": True,
                        "message": f"Model already exists ({file_size_mb:.1f}MB)",
                        "file_size_mb": file_size_mb,
                        "download_time": 0,
                        "source": "cache"
                    }
        
        # Try direct ONNX download first
        success, message = self.download_onnx_direct(model_config)
        
        # If direct download failed, try PyTorch export
        if not success:
            logger.info(f"Direct download failed, trying PyTorch export for {model_name}")
            success, message = self.export_pytorch_to_onnx(model_config)
        
        # Validate the model
        if success and self.verify_integrity:
            valid, val_msg = self.validate_onnx_model(model_config)
            if not valid:
                success = False
                message = f"Validation failed: {val_msg}"
        
        download_time = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        
        return {
            "model": model_name,
            "success": success,
            "message": message,
            "file_size_mb": file_size_mb,
            "download_time": download_time,
            "source": "download" if success else "failed"
        }
    
    def download_modality(
        self,
        modality: str,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Download all models for a specific modality.
        
        Args:
            modality: Modality type (audio, video, text, image)
            force: Force re-download
            
        Returns:
            List of download results
        """
        results = []
        
        for name, config in MODEL_CONFIGS.items():
            if config.model_type == modality:
                result = self.download_model(name, force=force)
                results.append(result)
        
        return results
    
    def download_all(
        self,
        force: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Download all configured models.
        
        Args:
            force: Force re-download
            
        Returns:
            List of download results
        """
        results = []
        
        for name in MODEL_CONFIGS.keys():
            result = self.download_model(name, force=force)
            results.append(result)
        
        return results
    
    def print_report(self, results: List[Dict[str, Any]]) -> None:
        """Print a formatted download report."""
        print("\n" + "=" * 70)
        print("MODEL DOWNLOAD REPORT")
        print("=" * 70)
        
        success_count = sum(1 for r in results if r.get("success"))
        total_count = len(results)
        
        for result in results:
            status = "✓" if result.get("success") else "✗"
            size_str = f"{result.get('file_size_mb', 0):.1f}MB" if result.get("file_size_mb") else "N/A"
            time_str = f"{result.get('download_time', 0):.1f}s" if result.get("download_time") else "N/A"
            
            print(f"\n{status} {result.get('model', 'unknown')}")
            print(f"  Status: {result.get('message', 'No message')}")
            print(f"  Size: {size_str}")
            print(f"  Time: {time_str}")
            
            if result.get("error"):
                print(f"  Error: {result.get('error')}")
        
        print("\n" + "-" * 70)
        print(f"SUMMARY: {success_count}/{total_count} models downloaded successfully")
        print("=" * 70 + "\n")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Download and export ML models for Argus"
    )
    parser.add_argument(
        "--model", "-m",
        help="Download a specific model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--modality",
        help="Download all models for a modality (audio, video, text, image)",
        type=str,
        default=None
    )
    parser.add_argument(
        "--all", "-a",
        help="Download all configured models",
        action="store_true"
    )
    parser.add_argument(
        "--force", "-f",
        help="Force re-download even if model exists",
        action="store_true"
    )
    parser.add_argument(
        "--list", "-l",
        help="List available models",
        action="store_true"
    )
    parser.add_argument(
        "--models-dir",
        help="Directory to save models",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    downloader = ProductionModelDownloader(model_dir=args.models_dir)
    
    if args.list:
        print("\nAvailable Models:")
        print("-" * 70)
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name}")
            print(f"    Type: {config.model_type}")
            print(f"    Task: {config.task}")
            print(f"    Source: {config.hf_model_id}")
            print(f"    Description: {config.description[:50]}...")
            print()
        return
    
    results = []
    
    if args.model:
        result = downloader.download_model(args.model, force=args.force)
        results = [result]
    elif args.modality:
        results = downloader.download_modality(args.modality, force=args.force)
    elif args.all:
        results = downloader.download_all(force=args.force)
    else:
        parser.print_help()
        return
    
    downloader.print_report(results)
    
    # Exit with error code if any downloads failed
    if not all(r.get("success") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
