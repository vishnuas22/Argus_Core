"""
Argus Core - Eigen-CAM Gradient-Free Attribution (Iteration 2)
================================================================
Cheap gradient-free attribution via SVD on the feature maps.

Research grounding:
- Muhammad & Yeasin, "Eigen-CAM: Class Activation Map using
  Eigenvalues", IJCNN 2020. Computes the principal components of the
  feature map and uses the first component as the activation map.
- Why use it: 10-30ms per image on T4, works on ONNX (no gradients
  needed), good for INT8-quantized models where GradCAM gradients are
  noisy.
- Use case in Argus: cheap cross-check against AttnLRP. If Eigen-CAM
  and AttnLRP agree on the salient region, the explanation is robust.

Algorithm:
1. Extract feature map F from the last convolutional layer (or the
   pre-classifier token grid for ViTs).
2. Flatten F to (C, H*W).
3. Compute the first principal component via SVD.
4. Reshape to (H, W) and normalize to [0, 1].

Strict-compat: pure-additive. No changes to detector interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


def eigen_cam_from_features(features: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute Eigen-CAM from a feature map.

    Args:
        features: (C, H, W) or (1, C, H, W) feature map.

    Returns:
        (H, W) heatmap in [0, 1], or None on failure.
    """
    try:
        f = np.asarray(features, dtype=np.float32)
        if f.ndim == 4:
            f = f[0]
        if f.ndim != 3:
            logger.warning("Eigen-CAM expects 3D features, got shape %s", f.shape)
            return None
        C, H, W = f.shape
        # Flatten to (C, H*W)
        flat = f.reshape(C, -1)
        # SVD: U S V^T = flat
        # First principal component is the first column of V (transposed)
        # Use the first right singular vector
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        # First PC: Vt[0] is (H*W,)
        pc1 = Vt[0].reshape(H, W)
        # Take absolute value (Eigen-CAM is sign-agnostic)
        pc1 = np.abs(pc1)
        # Normalize
        return _normalize(pc1)
    except Exception as e:
        logger.warning("Eigen-CAM failed: %s", e)
        return None


def eigen_cam_from_tokens(
    tokens: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> Optional[np.ndarray]:
    """
    Compute Eigen-CAM from ViT patch tokens.

    Args:
        tokens: (N_patches, hidden) array of patch token activations.
        grid_h: Patch grid height.
        grid_w: Patch grid width.

    Returns:
        (grid_h, grid_w) heatmap in [0, 1], or None on failure.
    """
    try:
        t = np.asarray(tokens, dtype=np.float32)
        if t.ndim != 2:
            logger.warning("Eigen-CAM tokens expects 2D, got %s", t.shape)
            return None
        # Transpose: (hidden, N_patches) → SVD
        flat = t.T  # (hidden, N_patches)
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        pc1 = Vt[0]  # (N_patches,)
        pc1 = np.abs(pc1)
        # Reshape to grid
        if len(pc1) != grid_h * grid_w:
            target = grid_h * grid_w
            if len(pc1) < target:
                pc1 = np.pad(pc1, (0, target - len(pc1)))
            else:
                pc1 = pc1[:target]
        heatmap = pc1.reshape(grid_h, grid_w)
        return _normalize(heatmap)
    except Exception as e:
        logger.warning("Eigen-CAM tokens failed: %s", e)
        return None


def _normalize(h: np.ndarray) -> np.ndarray:
    h_min = float(h.min())
    h_max = float(h.max())
    if h_max - h_min < 1e-8:
        return np.zeros_like(h)
    return (h - h_min) / (h_max - h_min)
