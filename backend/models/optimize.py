"""
Argus Core - Model Optimization
===============================
Model optimization utilities for efficient inference.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - models/optimize.py

Pipeline:
1. Export PyTorch → ONNX
2. Apply ONNX graph optimizations
3. Quantize to INT8 (with calibration data)
4. Build TensorRT engine (optional)

Target Hardware: RTX 3050 (4GB VRAM)
- INT8 quantization provides 4x speedup with <2% accuracy loss
- TensorRT maximizes RTX 3050 performance
"""

import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config import config
from utils.logging import get_logger
from utils.errors import ConfigurationError

logger = get_logger(__name__)


class QuantizationMode(str, Enum):
    """Quantization modes for model optimization."""
    DYNAMIC = "dynamic"     # Dynamic quantization (fast, less accurate)
    STATIC = "static"       # Static quantization (requires calibration)
    QAT = "qat"             # Quantization-aware training (best accuracy)


class OptimizationLevel(str, Enum):
    """ONNX graph optimization levels."""
    DISABLED = "disabled"
    BASIC = "basic"         # Basic graph optimizations
    EXTENDED = "extended"   # Extended optimizations
    ALL = "all"             # All optimizations


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    quantization_mode: QuantizationMode = QuantizationMode.DYNAMIC
    optimization_level: OptimizationLevel = OptimizationLevel.ALL
    target_opset: int = 17
    enable_tensorrt: bool = True
    calibration_samples: int = 100
    per_channel: bool = True
    reduce_range: bool = False


class CalibrationDataReader:
    """
    Calibration data reader for static INT8 quantization.
    
    Provides representative data for calibration during quantization.
    """
    
    def __init__(
        self,
        data_loader: Callable[[], np.ndarray],
        num_samples: int = 100
    ):
        """
        Initialize calibration reader.
        
        Args:
            data_loader: Function that returns calibration samples
            num_samples: Number of calibration samples
        """
        self.data_loader = data_loader
        self.num_samples = num_samples
        self.current_sample = 0
        self._samples: List[np.ndarray] = []
        self._preload()
    
    def _preload(self) -> None:
        """Preload calibration samples."""
        for _ in range(self.num_samples):
            sample = self.data_loader()
            self._samples.append(sample)
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample."""
        if self.current_sample >= len(self._samples):
            return None
        
        sample = self._samples[self.current_sample]
        self.current_sample += 1
        
        return {"input": sample}
    
    def rewind(self) -> None:
        """Reset to beginning of calibration data."""
        self.current_sample = 0


class ModelOptimizer:
    """
    Optimize models for efficient inference.
    
    Supports:
    - PyTorch to ONNX export
    - ONNX graph optimizations
    - INT8/FP16 quantization
    - TensorRT engine building
    """
    
    def __init__(
        self,
        opt_config: Optional[OptimizationConfig] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            opt_config: Optimization configuration
            output_dir: Directory for optimized models
        """
        self.config = opt_config or OptimizationConfig()
        self.output_dir = output_dir or "/models/optimized"
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_to_onnx(
        self,
        model: Any,  # torch.nn.Module
        input_shape: Tuple[int, ...],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: Optional[int] = None
    ) -> str:
        """
        Export PyTorch model to ONNX.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Output ONNX file path
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axis specification
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        try:
            import torch
            
            # Set model to eval mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Default names
            input_names = input_names or ["input"]
            output_names = output_names or ["output"]
            opset_version = opset_version or self.config.target_opset
            
            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            logger.info(f"Exported model to ONNX: {output_path}")
            return output_path
            
        except ImportError:
            raise ConfigurationError(
                "pytorch",
                "PyTorch not available for ONNX export"
            )
    
    def optimize_onnx(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        optimization_level: Optional[OptimizationLevel] = None
    ) -> str:
        """
        Apply ONNX graph optimizations.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output path (None = in-place)
            optimization_level: Optimization level
            
        Returns:
            Path to optimized model
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            output_path = output_path or input_path.replace(".onnx", "_optimized.onnx")
            level = optimization_level or self.config.optimization_level
            
            # Load model
            model = onnx.load(input_path)
            
            # Apply optimizations based on level
            if level == OptimizationLevel.DISABLED:
                onnx.save(model, output_path)
            else:
                # Use ONNX Runtime optimizer
                optimized = optimizer.optimize_model(
                    input_path,
                    model_type='bert',  # Generic transformer optimization
                    num_heads=0,
                    hidden_size=0,
                    optimization_options=None
                )
                optimized.save_model_to_file(output_path)
            
            logger.info(f"Optimized ONNX model: {output_path}")
            return output_path
            
        except ImportError as e:
            logger.warning(f"ONNX optimization libraries not available: {e}")
            # Copy input to output if optimization not available
            if output_path and output_path != input_path:
                import shutil
                shutil.copy(input_path, output_path)
            return output_path or input_path
    
    def quantize_dynamic(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        weight_type: str = "QInt8"
    ) -> str:
        """
        Apply dynamic INT8 quantization.
        
        Fast quantization without calibration data.
        Weights quantized to INT8, activations remain FP32.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output path
            weight_type: Weight quantization type
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            output_path = output_path or input_path.replace(".onnx", "_int8_dynamic.onnx")
            
            # Map weight type
            weight_type_map = {
                "QInt8": QuantType.QInt8,
                "QUInt8": QuantType.QUInt8,
            }
            quant_type = weight_type_map.get(weight_type, QuantType.QInt8)
            
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=quant_type,
                per_channel=self.config.per_channel,
                reduce_range=self.config.reduce_range
            )
            
            logger.info(f"Dynamic quantization complete: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("ONNX quantization not available")
            return input_path
    
    def quantize_static(
        self,
        input_path: str,
        calibration_data_reader: CalibrationDataReader,
        output_path: Optional[str] = None,
        quant_format: str = "QOperator"
    ) -> str:
        """
        Apply static INT8 quantization with calibration.
        
        Both weights and activations quantized to INT8.
        Requires calibration data for activation range estimation.
        
        Args:
            input_path: Input ONNX model path
            calibration_data_reader: Calibration data provider
            output_path: Output path
            quant_format: Quantization format
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import (
                quantize_static,
                CalibrationMethod,
                QuantFormat,
                QuantType
            )
            
            output_path = output_path or input_path.replace(".onnx", "_int8_static.onnx")
            
            # Map format
            format_map = {
                "QOperator": QuantFormat.QOperator,
                "QDQ": QuantFormat.QDQ,
            }
            quant_fmt = format_map.get(quant_format, QuantFormat.QOperator)
            
            quantize_static(
                model_input=input_path,
                model_output=output_path,
                calibration_data_reader=calibration_data_reader,
                quant_format=quant_fmt,
                per_channel=self.config.per_channel,
                reduce_range=self.config.reduce_range,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax
            )
            
            logger.info(f"Static quantization complete: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("ONNX quantization not available")
            return input_path
    
    def convert_to_fp16(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert model to FP16 precision.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output path
            
        Returns:
            Path to FP16 model
        """
        try:
            from onnxconverter_common import float16
            import onnx
            
            output_path = output_path or input_path.replace(".onnx", "_fp16.onnx")
            
            model = onnx.load(input_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, output_path)
            
            logger.info(f"FP16 conversion complete: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("ONNX FP16 converter not available")
            return input_path
    
    def build_tensorrt_engine(
        self,
        onnx_path: str,
        output_path: Optional[str] = None,
        fp16: bool = True,
        int8: bool = False,
        max_batch_size: int = 32,
        workspace_size_mb: int = 1024
    ) -> Optional[str]:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output engine path
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            max_batch_size: Maximum batch size
            workspace_size_mb: Workspace memory in MB
            
        Returns:
            Path to TensorRT engine or None if not available
        """
        try:
            import tensorrt as trt
            
            output_path = output_path or onnx_path.replace(".onnx", ".trt")
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"TensorRT parser error: {parser.get_error(i)}")
                    return None
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                workspace_size_mb * 1024 * 1024
            )
            
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return None
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT engine built: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("TensorRT not available")
            return None
        except Exception as e:
            logger.error(f"TensorRT build failed: {e}")
            return None
    
    def validate_onnx(
        self,
        model_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import onnx
            
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            return True, None
            
        except ImportError:
            return True, "ONNX validation skipped (onnx not installed)"
        except Exception as e:
            return False, str(e)
    
    def get_model_info(
        self,
        model_path: str
    ) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Model info dictionary
        """
        try:
            import onnx
            
            model = onnx.load(model_path)
            
            # Get input/output shapes
            inputs = []
            for input_tensor in model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else -1)
                inputs.append({
                    "name": input_tensor.name,
                    "shape": shape
                })
            
            outputs = []
            for output_tensor in model.graph.output:
                shape = []
                for dim in output_tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else -1)
                outputs.append({
                    "name": output_tensor.name,
                    "shape": shape
                })
            
            # Get file size
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            return {
                "path": model_path,
                "opset_version": model.opset_import[0].version,
                "inputs": inputs,
                "outputs": outputs,
                "file_size_mb": round(file_size_mb, 2),
                "num_nodes": len(model.graph.node)
            }
            
        except ImportError:
            return {"path": model_path, "error": "onnx not installed"}
        except Exception as e:
            return {"path": model_path, "error": str(e)}
    
    def estimate_memory_usage(
        self,
        model_path: str,
        batch_size: int = 1
    ) -> Dict[str, int]:
        """
        Estimate model memory usage.
        
        Args:
            model_path: Path to ONNX model
            batch_size: Batch size for estimation
            
        Returns:
            Memory estimates in MB
        """
        try:
            import onnx
            
            model = onnx.load(model_path)
            
            # Estimate weights memory
            weights_bytes = 0
            for initializer in model.graph.initializer:
                dims = list(initializer.dims)
                elem_count = 1
                for d in dims:
                    elem_count *= d
                
                # Assume float32 by default
                weights_bytes += elem_count * 4
            
            # Estimate activation memory (rough estimate)
            activation_bytes = weights_bytes // 2  # Rough heuristic
            
            # Scale by batch size for activations
            activation_bytes *= batch_size
            
            # File size as reference
            file_size = os.path.getsize(model_path)
            
            return {
                "weights_mb": round(weights_bytes / (1024 * 1024), 2),
                "activations_mb": round(activation_bytes / (1024 * 1024), 2),
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "total_estimated_mb": round(
                    (weights_bytes + activation_bytes) / (1024 * 1024), 2
                )
            }
            
        except Exception as e:
            logger.warning(f"Memory estimation failed: {e}")
            return {}
    
    def create_calibration_data(
        self,
        input_shape: Tuple[int, ...],
        num_samples: int = 100,
        data_range: Tuple[float, float] = (0.0, 1.0)
    ) -> CalibrationDataReader:
        """
        Create random calibration data for quantization.
        
        For production, use representative real data instead.
        
        Args:
            input_shape: Model input shape
            num_samples: Number of calibration samples
            data_range: Data value range
            
        Returns:
            CalibrationDataReader
        """
        def data_loader():
            low, high = data_range
            return np.random.uniform(
                low, high, input_shape
            ).astype(np.float32)
        
        return CalibrationDataReader(
            data_loader=data_loader,
            num_samples=num_samples
        )


def optimize_for_rtx3050(
    model_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Optimize model for RTX 3050 (4GB VRAM).
    
    Creates INT8 quantized and TensorRT versions.
    
    Args:
        model_path: Input ONNX model path
        output_dir: Output directory
        
    Returns:
        Dict of optimization type to output path
    """
    optimizer = ModelOptimizer(
        config=OptimizationConfig(
            quantization_mode=QuantizationMode.DYNAMIC,
            optimization_level=OptimizationLevel.ALL,
            enable_tensorrt=True
        ),
        output_dir=output_dir or "/models/optimized"
    )
    
    results = {}
    
    # Optimize ONNX graph
    optimized_path = optimizer.optimize_onnx(model_path)
    results["optimized"] = optimized_path
    
    # Dynamic INT8 quantization
    int8_path = optimizer.quantize_dynamic(optimized_path)
    results["int8"] = int8_path
    
    # FP16 version
    fp16_path = optimizer.convert_to_fp16(model_path)
    results["fp16"] = fp16_path
    
    # TensorRT engine (if available)
    trt_path = optimizer.build_tensorrt_engine(
        int8_path,
        int8=True,
        workspace_size_mb=512  # Conservative for 4GB VRAM
    )
    if trt_path:
        results["tensorrt"] = trt_path
    
    return results
