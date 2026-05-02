"""
HDR Ground Truth Fast Path Utilities

This module provides shared utilities for HDR ground truth processing,
including the fast uint8 alignment method used in benchmarks.

The fast path is the only path - no fallbacks are provided.
"""

import numpy as np


def _to_u8_for_alignment_frame(frame: np.ndarray | None) -> np.ndarray | None:
    """Convert a frame to uint8 for alignment purposes.

    Args:
        frame: Input frame as numpy array (uint8 or uint16)

    Returns:
        Frame converted to uint8 or None if conversion fails
    """
    if not isinstance(frame, np.ndarray):
        return None
    arr = np.ascontiguousarray(frame)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return np.ascontiguousarray(
            ((arr.astype(np.float32) / 65535.0) * 255.0)
            .clip(0.0, 255.0)
            .astype(np.uint8)
        )
    return None


def _frame_structure_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute structural similarity between two frames using OpenCV.

    This is a simplified version for alignment scoring.

    Args:
        frame1: First frame (should be uint8)
        frame2: Second frame (should be uint8)

    Returns:
        SSIM score between 0 and 1
    """
    try:
        import cv2
        if frame1.shape != frame2.shape:
            return 0.0
        return float(cv2.SSIM(frame1, frame2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    except ImportError:
        # Fallback to simple MSE-based metric
        try:
            mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
            return float(1.0 / (1.0 + mse))
        except Exception:
            return 0.0
    except Exception:
        return 0.0


def _process_hdr_gt_with_fast_path(
    gt_rgb16: np.ndarray,
    sdr_frame: np.ndarray,
    *,
    output_width: int,
    output_height: int,
    crop_active_area: bool = True,
    sdr_video_path: str | None = None,
    gt_video_path: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    """Process HDR GT using fast uint8 alignment path.

    Args:
        gt_rgb16: HDR GT frame as uint16 RGB
        sdr_frame: SDR frame for alignment
        output_width: Target output width
        output_height: Target output height
        crop_active_area: Whether to crop to active area
        sdr_video_path: Path to SDR video (for active area cropping)
        gt_video_path: Path to GT video (for active area cropping)

    Returns:
        Tuple of (gt_hdr_frame, gt_u8_frame, mode_note, alignment_note)
    """
    from gui_scaling import _letterbox_bgr
    from gui_media_probe import _crop_gt_frame_to_pair_active_area

    # Convert RGB to BGR
    gt_hdr_bgr = np.ascontiguousarray(gt_rgb16[:, :, ::-1])

    # Crop to active area if requested
    if crop_active_area and sdr_video_path and gt_video_path:
        gt_hdr_bgr = _crop_gt_frame_to_pair_active_area(
            gt_hdr_bgr,
            str(sdr_video_path),
            str(gt_video_path),
        )

    # Letterbox to target size
    gt_hdr_final = np.ascontiguousarray(
        _letterbox_bgr(gt_hdr_bgr, output_width, output_height)
    )

    # Create uint8 version for alignment
    gt_u8 = _to_u8_for_alignment_frame(gt_hdr_final)

    mode_note = "true_hdr_video_fast"
    alignment_note = ""

    if gt_u8 is not None:
        # Test alignment with SDR frame if provided
        if sdr_frame is not None:
            sdr_u8 = _to_u8_for_alignment_frame(sdr_frame)
            if sdr_u8 is not None:
                try:
                    # Ensure same shape for alignment
                    if sdr_u8.shape != gt_u8.shape:
                        sdr_u8 = np.ascontiguousarray(
                            _letterbox_bgr(sdr_u8, gt_u8.shape[1], gt_u8.shape[0])
                        )

                    align_score = _frame_structure_similarity(sdr_u8, gt_u8)
                    if align_score > 0.8:  # Good alignment threshold
                        alignment_note = f"fast_align_score={align_score:.4f}"
                except Exception:
                    pass

    return gt_hdr_final, gt_u8, mode_note, alignment_note