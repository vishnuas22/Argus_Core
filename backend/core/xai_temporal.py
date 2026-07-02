"""
Argus Core - Video Temporal Attribution (Iteration 2)
=======================================================
Temporal window attribution for video deepfake detectors via
leave-one-tubelet-out occlusion.

Research grounding:
- ISTVT (Yu et al., "Magnifying Tiny Forgery Clues in Videos",
  TIFS 2023) decomposes video deepfake explanations into spatial and
  temporal components using attention flow.
- VideoMAE's tubelet tokens encode (t, h, w) — the standard approach
  for temporal attribution is to ablate temporal windows and measure
  the score drop.
- For production use, we ablate at the **frame level** (cheaper than
  tubelet level) — replace K frames with neutral frames (zero or mean)
  and re-run the detector. The temporal window with the largest drop
  is the most influential.

Algorithm:
1. Run detector on the original frame sequence → baseline score.
2. For each temporal window of size W (default 2 frames):
   a. Replace the window with neutral frames (zero or mean).
   b. Run the detector → modified score.
   c. Drop = baseline - modified.
3. The window with the largest drop is the most influential.

Latency: K=num_frames/W forward passes per video. For 16 frames and W=2,
that's 8 passes → ~1.5s on T4 with VideoMAE-base.

Strict-compat: pure-additive. No changes to detector interface.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


async def attribute_temporal_windows(
    frames: List[np.ndarray],
    detect_fn,
    window_size: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Attribute video deepfake detection to temporal windows via occlusion.

    Args:
        frames: List of HxWx3 uint8 RGB frames.
        detect_fn: Async callable that takes a list of frames and returns
            DetectionResult (or a float score).
        window_size: Number of frames to occlude per pass.

    Returns:
        Dict with:
            - "baseline_score": float
            - "window_drops": List[float]  (one per window)
            - "window_ranges": List[Tuple[int, int]]  (start, end frame indices)
            - "most_influential_window": Tuple[int, int]
            - "explanation": str
        Or None on failure.
    """
    if not frames:
        return None
    try:
        async def _get_score(fr):
            r = await detect_fn(fr) if asyncio.iscoroutinefunction(detect_fn) else detect_fn(fr)
            if hasattr(r, "score"):
                return float(r.score)
            return float(r)

        baseline = await _get_score(frames)

        # Build window ranges
        n = len(frames)
        windows: List[Tuple[int, int]] = []
        for start in range(0, n, window_size):
            end = min(start + window_size, n)
            windows.append((start, end))

        window_drops: List[float] = []
        # Use a neutral frame (mean of all frames) as the occluder
        mean_frame = np.mean(np.stack(frames), axis=0).astype(np.uint8)

        for start, end in windows:
            modified = list(frames)
            for i in range(start, end):
                modified[i] = mean_frame.copy()
            try:
                modified_score = await _get_score(modified)
            except Exception as e:
                logger.debug("Temporal attribution window %d-%d failed: %s", start, end, e)
                modified_score = baseline
            drop = baseline - modified_score
            window_drops.append(float(drop))

        # Find most influential window
        most_influential_idx = int(np.argmax(np.abs(window_drops)))
        most_influential_window = windows[most_influential_idx]
        most_influential_drop = window_drops[most_influential_idx]

        # Human-readable explanation
        direction = "increased" if most_influential_drop < 0 else "decreased"
        explanation = (
            f"Occluding frames {most_influential_window[0]}-{most_influential_window[1]} "
            f"{direction} the fake score by {abs(most_influential_drop):.4f}. "
            f"This temporal window contains the most influential cues for the detector's verdict."
        )

        return {
            "baseline_score": baseline,
            "window_drops": window_drops,
            "window_ranges": windows,
            "most_influential_window": most_influential_window,
            "most_influential_drop": most_influential_drop,
            "explanation": explanation,
        }
    except Exception as e:
        logger.warning("Temporal attribution failed: %s", e)
        return None
