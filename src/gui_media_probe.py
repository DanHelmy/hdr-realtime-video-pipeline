"""Media probing and SDR/HDR compatibility helpers for the GUI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import cv2
import numpy as np

_IMAGE_FILE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".exr",
    ".hdr",
    ".dpx",
    ".ppm",
    ".pgm",
    ".pfm",
}
_VIDEO_FILE_EXTS = {
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".mts",
    ".ogv",
    ".3gp",
}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
        if value > 0.0:
            return value
    except Exception:
        pass
    return float(default)


_GT_SYNC_TOLERANCE_S = max(
    0.25,
    _env_float("HDRTVNET_GT_SYNC_TOLERANCE_S", 2.0),
)
_GT_EXACT_FRAME_TOLERANCE = 2


def _looks_like_video_path(path: str | None) -> bool:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _IMAGE_FILE_EXTS:
        return False
    return ext in _VIDEO_FILE_EXTS


def _cancel_check_requested(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _metadata_duration_delta_s(
    src_meta: dict,
    gt_meta: dict,
    src_fps: float,
    gt_fps: float,
) -> float:
    src_d = float(src_meta.get("duration_s", 0.0) or 0.0)
    gt_d = float(gt_meta.get("duration_s", 0.0) or 0.0)
    if src_d > 0.0 and gt_d > 0.0:
        return abs(src_d - gt_d)

    src_n = int(src_meta.get("frame_count", 0) or 0)
    gt_n = int(gt_meta.get("frame_count", 0) or 0)
    if src_n > 0 and gt_n > 0:
        if src_fps > 0.0 and gt_fps > 0.0:
            return abs((float(src_n) / src_fps) - (float(gt_n) / gt_fps))
        fps = src_fps if src_fps > 0.0 else gt_fps
        if fps > 0.0:
            return abs(float(src_n - gt_n)) / fps
    return 0.0


def _validate_video_timing_compatibility(
    src_meta: dict | None,
    gt_meta: dict | None,
    *,
    source_label: str = "source",
    gt_label: str = "GT",
    metadata_error_message: str = "Could not read video metadata.",
    enforce_sync_tolerance: bool = True,
) -> tuple[bool, str | None, list[str]]:
    if not isinstance(src_meta, dict) or not isinstance(gt_meta, dict):
        return False, str(metadata_error_message), []

    src_fps = float(src_meta.get("fps", 0.0) or 0.0)
    gt_fps = float(gt_meta.get("fps", 0.0) or 0.0)
    if src_fps > 0.0 and gt_fps > 0.0 and abs(src_fps - gt_fps) > 0.25:
        return (
            False,
            f"FPS mismatch: {source_label} {src_fps:.3f} vs {gt_label} {gt_fps:.3f}.",
            [],
        )

    notes: list[str] = []
    duration_delta_s = _metadata_duration_delta_s(
        src_meta,
        gt_meta,
        src_fps,
        gt_fps,
    )

    src_n = int(src_meta.get("frame_count", 0) or 0)
    gt_n = int(gt_meta.get("frame_count", 0) or 0)
    if (
        src_n > 0
        and gt_n > 0
        and abs(src_n - gt_n) > _GT_EXACT_FRAME_TOLERANCE
    ):
        if duration_delta_s <= 0.0 or duration_delta_s > _GT_SYNC_TOLERANCE_S:
            if not enforce_sync_tolerance and duration_delta_s > 0.0:
                notes.append(
                    f"length differs by {duration_delta_s:.2f}s; using content sync"
                )
            elif not enforce_sync_tolerance:
                notes.append(
                    f"frame count differs ({source_label} {src_n} vs {gt_label} {gt_n}); using content sync"
                )
            else:
                return (
                    False,
                    f"Frame-count mismatch: {source_label} {src_n} vs {gt_label} {gt_n}.",
                    [],
                )
        else:
            notes.append(
                f"length differs by {duration_delta_s:.2f}s; using overlap sync"
            )

    src_d = float(src_meta.get("duration_s", 0.0) or 0.0)
    gt_d = float(gt_meta.get("duration_s", 0.0) or 0.0)
    if src_d > 0.0 and gt_d > 0.0 and abs(src_d - gt_d) > 0.25:
        if duration_delta_s > _GT_SYNC_TOLERANCE_S:
            if enforce_sync_tolerance:
                return (
                    False,
                    f"Duration mismatch: {source_label} {src_d:.2f}s vs {gt_label} {gt_d:.2f}s.",
                    [],
                )
            note = f"length differs by {duration_delta_s:.2f}s; using content sync"
        else:
            note = f"length differs by {duration_delta_s:.2f}s; using overlap sync"
        if note not in notes:
            notes.append(note)

    return True, None, notes


def _probe_hdr_input(video_path: str) -> dict:
    """Detect whether the input stream appears to be HDR."""
    info = {
        "is_hdr": False,
        "transfer": "unknown",
        "primaries": "unknown",
        "pix_fmt": "unknown",
        "bits": 0,
        "reason": "",
    }
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        info["reason"] = "ffprobe not found; assuming SDR input"
        return info

    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=color_transfer,color_primaries,pix_fmt,bits_per_raw_sample",
        "-of", "json",
        video_path,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            info["reason"] = "no video stream metadata found; assuming SDR input"
            return info

        stream = streams[0]
        trc = str(stream.get("color_transfer") or "unknown").lower()
        prim = str(stream.get("color_primaries") or "unknown").lower()
        pix_fmt = str(stream.get("pix_fmt") or "unknown").lower()

        bits = 0
        raw_bits = stream.get("bits_per_raw_sample")
        if raw_bits is not None and str(raw_bits).isdigit():
            bits = int(raw_bits)
        if bits == 0:
            m = re.search(r"p(10|12|16)", pix_fmt)
            if m:
                bits = int(m.group(1))

        info.update({
            "transfer": trc,
            "primaries": prim,
            "pix_fmt": pix_fmt,
            "bits": bits,
        })

        hdr_trc = trc in {"smpte2084", "arib-std-b67"}
        hdr_bt2020_inferred = (
            prim.startswith("bt2020")
            and bits >= 10
            and trc not in {"bt709", "unknown", "iec61966-2-1"}
        )
        info["is_hdr"] = bool(hdr_trc or hdr_bt2020_inferred)
        if info["is_hdr"]:
            if hdr_trc:
                info["reason"] = f"HDR transfer detected ({trc})"
            else:
                info["reason"] = f"BT.2020 {bits}-bit stream detected"
        else:
            info["reason"] = "no HDR transfer metadata detected"
    except Exception as exc:
        info["reason"] = f"ffprobe failed ({exc}); assuming SDR input"
    return info


def _probe_audio_streams(video_path: str) -> list[dict]:
    """Return detected audio streams with basic metadata via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,channels:stream_tags=language,title:stream_disposition=default",
        "-of", "json",
        video_path,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
    except Exception:
        return []

    out = []
    for i, s in enumerate(streams):
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        lang = str(tags.get("language") or "und").strip().lower()
        title = str(tags.get("title") or "").strip()
        codec = str(s.get("codec_name") or "audio").strip().lower()
        ch = s.get("channels")
        try:
            ch = int(ch)
        except Exception:
            ch = None
        out.append({
            "ordinal": i,
            "index": s.get("index", i),
            "language": lang,
            "title": title,
            "codec": codec,
            "channels": ch,
            "default": bool(disp.get("default", 0)),
        })
    return out


def _norm_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:
        return str(path)


def _probe_video_timing_info(video_path: str) -> dict | None:
    """Read basic video timing/geometry info via OpenCV."""
    if not video_path or not os.path.isfile(video_path):
        return None
    if not _looks_like_video_path(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = 0.0
        if fps > 0.0 and frame_count > 0:
            duration = float(frame_count) / float(fps)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": duration,
        }
    except Exception:
        return None
    finally:
        cap.release()


def _frame_structure_similarity(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float | None:
    """Estimate structural similarity while being tolerant to color differences."""
    if a_bgr is None or b_bgr is None:
        return None
    try:
        a_bgr = _crop_frame_to_active_area(a_bgr)
        b_bgr = _crop_frame_to_active_area(b_bgr)
        a_small = cv2.resize(a_bgr, (256, 144), interpolation=cv2.INTER_AREA)
        b_small = cv2.resize(b_bgr, (256, 144), interpolation=cv2.INTER_AREA)
        a_gray = cv2.cvtColor(a_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b_gray = cv2.cvtColor(b_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        a_gray = cv2.normalize(a_gray, None, 0.0, 1.0, cv2.NORM_MINMAX)
        b_gray = cv2.normalize(b_gray, None, 0.0, 1.0, cv2.NORM_MINMAX)

        def _corr(x: np.ndarray, y: np.ndarray) -> float:
            xv = x.reshape(-1).astype(np.float32)
            yv = y.reshape(-1).astype(np.float32)
            xv = xv - float(np.mean(xv))
            yv = yv - float(np.mean(yv))
            den = float(np.linalg.norm(xv) * np.linalg.norm(yv))
            if den <= 1e-8:
                return 0.0
            return float(np.dot(xv, yv) / den)

        lum_corr = _corr(a_gray, b_gray)

        a_edge = cv2.Canny((a_gray * 255.0).astype(np.uint8), 60, 160).astype(np.float32) / 255.0
        b_edge = cv2.Canny((b_gray * 255.0).astype(np.uint8), 60, 160).astype(np.float32) / 255.0
        edge_corr = _corr(a_edge, b_edge)

        ax = cv2.Sobel(a_gray, cv2.CV_32F, 1, 0, ksize=3)
        ay = cv2.Sobel(a_gray, cv2.CV_32F, 0, 1, ksize=3)
        bx = cv2.Sobel(b_gray, cv2.CV_32F, 1, 0, ksize=3)
        by = cv2.Sobel(b_gray, cv2.CV_32F, 0, 1, ksize=3)
        a_mag = np.sqrt(ax * ax + ay * ay)
        b_mag = np.sqrt(bx * bx + by * by)
        grad_corr = _corr(a_mag, b_mag)

        score = (0.30 * lum_corr) + (0.45 * edge_corr) + (0.25 * grad_corr)
        return max(-1.0, min(1.0, float(score)))
    except Exception:
        return None


def _frame_active_content_bounds(
    frame_bgr: np.ndarray,
    *,
    min_border_ratio: float = 0.015,
    black_level_ratio: float = 0.006,
) -> tuple[int, int, int, int]:
    """Detect encoded black bars and return (top, bottom, left, right)."""
    if frame_bgr is None or frame_bgr.ndim < 2:
        return 0, 0, 0, 0
    h, w = frame_bgr.shape[:2]
    if h < 4 or w < 4:
        return 0, h, 0, w
    try:
        arr = frame_bgr.astype(np.float32, copy=False)
        if arr.ndim == 3:
            signal = np.max(arr, axis=2)
        else:
            signal = arr
        peak = float(np.percentile(signal, 99.9))
        if not np.isfinite(peak) or peak <= 0.0:
            return 0, h, 0, w
        low_floor = 2.0 if peak <= 255.0 else 64.0
        threshold = max(low_floor, peak * float(black_level_ratio))

        row_level = np.percentile(signal, 95, axis=1)
        col_level = np.percentile(signal, 95, axis=0)

        def _adaptive_level_threshold(levels: np.ndarray) -> float:
            low = float(np.percentile(levels, 5))
            high = float(np.percentile(levels, 95))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                return threshold
            # Encoded video often decodes black bars slightly above zero
            # because of limited-range conversion and compression residue.
            return max(threshold, low + ((high - low) * 0.08))

        row_threshold = _adaptive_level_threshold(row_level)
        col_threshold = _adaptive_level_threshold(col_level)
        row_has_signal = row_level > row_threshold
        col_has_signal = col_level > col_threshold
        if not np.any(row_has_signal) or not np.any(col_has_signal):
            return 0, h, 0, w

        top = int(np.argmax(row_has_signal))
        bottom = h - int(np.argmax(row_has_signal[::-1]))
        left = int(np.argmax(col_has_signal))
        right = w - int(np.argmax(col_has_signal[::-1]))
        if bottom - top < max(4, h // 10) or right - left < max(4, w // 10):
            return 0, h, 0, w

        min_border_px = max(2, int(round(min(h, w) * float(min_border_ratio))))
        if max(top, h - bottom, left, w - right) < min_border_px:
            return 0, h, 0, w
        return top, bottom, left, right
    except Exception:
        return 0, h, 0, w


def _crop_frame_to_active_area(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop encoded letterbox/pillarbox bars when they are clearly present."""
    if frame_bgr is None or frame_bgr.ndim < 2:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    top, bottom, left, right = _frame_active_content_bounds(frame_bgr)
    if top <= 0 and left <= 0 and bottom >= h and right >= w:
        return frame_bgr
    if bottom <= top or right <= left:
        return frame_bgr
    return np.ascontiguousarray(frame_bgr[top:bottom, left:right])


_PAIR_GT_ACTIVE_CROP_CACHE: dict[tuple, dict | None] = {}


def _center_crop_bounds_for_aspect(
    width: int,
    height: int,
    target_aspect: float,
) -> tuple[int, int, int, int] | None:
    """Return a stable centered crop matching target_aspect."""
    w = int(width)
    h = int(height)
    ar = float(target_aspect or 0.0)
    if w <= 1 or h <= 1 or ar <= 0.0:
        return None
    current_ar = float(w) / float(h)
    if abs(current_ar - ar) <= 0.01:
        return None

    if ar > current_ar:
        new_h = int(round(float(w) / ar))
        if new_h <= 0 or new_h >= h:
            return None
        if new_h < int(round(h * 0.45)):
            return None
        top = max(0, (h - new_h) // 2)
        bottom = min(h, top + new_h)
        return top, bottom, 0, w

    new_w = int(round(float(h) * ar))
    if new_w <= 0 or new_w >= w:
        return None
    if new_w < int(round(w * 0.45)):
        return None
    left = max(0, (w - new_w) // 2)
    right = min(w, left + new_w)
    return 0, h, left, right


def _stable_active_aspect_from_info(
    info: dict | None,
    *,
    width: int,
    height: int,
) -> float:
    if not isinstance(info, dict):
        return 0.0
    if not bool(info.get("cropped_bars", False)):
        return 0.0
    try:
        active_w = int(info.get("active_width") or 0)
        active_h = int(info.get("active_height") or 0)
        active_ar = float(info.get("active_aspect") or 0.0)
    except Exception:
        return 0.0
    if active_w <= 0 or active_h <= 0 or active_ar <= 0.0:
        return 0.0
    # Accept real letterbox/pillarbox estimates, but reject tiny object crops
    # from dark or low-detail samples. The actual crop rectangle stays geometric.
    if active_w < int(round(float(width) * 0.72)):
        return 0.0
    if active_h < int(round(float(height) * 0.42)):
        return 0.0
    return float(active_ar)


def _scale_crop_info_to_frame(
    crop_info: dict,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    try:
        src_w = int(crop_info.get("width") or 0)
        src_h = int(crop_info.get("height") or 0)
        top, bottom, left, right = crop_info.get("bounds") or (0, 0, 0, 0)
        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)
    except Exception:
        return None
    if src_w <= 0 or src_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    if src_w != int(frame_w):
        left = int(round(float(left) * float(frame_w) / float(src_w)))
        right = int(round(float(right) * float(frame_w) / float(src_w)))
    if src_h != int(frame_h):
        top = int(round(float(top) * float(frame_h) / float(src_h)))
        bottom = int(round(float(bottom) * float(frame_h) / float(src_h)))
    top = max(0, min(int(frame_h) - 1, top))
    bottom = max(top + 1, min(int(frame_h), bottom))
    left = max(0, min(int(frame_w) - 1, left))
    right = max(left + 1, min(int(frame_w), right))
    if top <= 0 and left <= 0 and bottom >= frame_h and right >= frame_w:
        return None
    return top, bottom, left, right


def _video_pair_gt_active_crop_info(
    source_path: str,
    candidate_path: str,
) -> dict | None:
    """Compute one stable GT crop rectangle for the whole video pair."""
    if not (
        _looks_like_video_path(source_path)
        and _looks_like_video_path(candidate_path)
    ):
        return None
    cache_key = _video_sync_cache_key(source_path, candidate_path, 0)
    if cache_key in _PAIR_GT_ACTIVE_CROP_CACHE:
        cached = _PAIR_GT_ACTIVE_CROP_CACHE.get(cache_key)
        return dict(cached) if isinstance(cached, dict) else None

    crop_info = None
    try:
        src_meta = _probe_video_timing_info(source_path)
        gt_meta = _probe_video_timing_info(candidate_path)
        if not isinstance(src_meta, dict) or not isinstance(gt_meta, dict):
            raise ValueError("missing media metadata")
        src_w = int(src_meta.get("width") or 0)
        src_h = int(src_meta.get("height") or 0)
        gt_w = int(gt_meta.get("width") or 0)
        gt_h = int(gt_meta.get("height") or 0)
        if src_w <= 0 or src_h <= 0 or gt_w <= 0 or gt_h <= 0:
            raise ValueError("invalid media dimensions")

        src_ar = float(src_w) / float(src_h)
        gt_ar = float(gt_w) / float(gt_h)
        if abs(src_ar - gt_ar) > 0.01:
            src_active = _probe_video_active_area_info(source_path, sample_count=5)
            src_active_ar = _stable_active_aspect_from_info(
                src_active,
                width=src_w,
                height=src_h,
            )
            target_ar = src_active_ar if src_active_ar > 0.0 else src_ar
            bounds = _center_crop_bounds_for_aspect(gt_w, gt_h, target_ar)
            if bounds is not None:
                top, bottom, left, right = bounds
                crop_info = {
                    "width": int(gt_w),
                    "height": int(gt_h),
                    "bounds": (
                        int(top),
                        int(bottom),
                        int(left),
                        int(right),
                    ),
                    "target_aspect": float(target_ar),
                    "source_aspect": float(src_ar),
                    "gt_aspect": float(gt_ar),
                }
    except Exception:
        crop_info = None

    _PAIR_GT_ACTIVE_CROP_CACHE[cache_key] = dict(crop_info) if crop_info else None
    return dict(crop_info) if crop_info else None


def _video_pair_should_crop_gt_active_area(
    source_path: str,
    candidate_path: str,
) -> bool:
    """Only crop GT bars when the encoded pair genuinely has different canvas AR."""
    return _video_pair_gt_active_crop_info(source_path, candidate_path) is not None


def _crop_gt_frame_to_pair_active_area(
    frame_bgr: np.ndarray,
    source_path: str,
    candidate_path: str,
) -> np.ndarray:
    if frame_bgr is None or frame_bgr.ndim < 2:
        return frame_bgr
    crop_info = _video_pair_gt_active_crop_info(source_path, candidate_path)
    if not isinstance(crop_info, dict):
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    bounds = _scale_crop_info_to_frame(crop_info, w, h)
    if bounds is None:
        return frame_bgr
    top, bottom, left, right = bounds
    if bottom <= top or right <= left:
        return frame_bgr
    return np.ascontiguousarray(frame_bgr[top:bottom, left:right])


def _probe_video_active_area_info(
    video_path: str,
    sample_count: int = 7,
    cancel_check=None,
) -> dict | None:
    """Estimate the active picture area after encoded black-bar cropping."""
    if _cancel_check_requested(cancel_check):
        return None
    meta = _probe_video_timing_info(video_path)
    if meta is None:
        return None
    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        fps = float(meta.get("fps", 0.0) or 0.0)
        frame_count = int(meta.get("frame_count", 0) or 0)
        duration_s = float(meta.get("duration_s", 0.0) or 0.0)
        # Adaptive probe count based on video length
        if duration_s < 5.0:
            probe_count = max(2, min(5, int(sample_count)))
        elif duration_s < 10.0:
            probe_count = max(3, min(8, int(sample_count)))
        else:
            probe_count = max(3, int(sample_count))
        if duration_s > 2.0 and fps > 0.0:
            # Adaptive margins based on video length
            if duration_s < 10.0:
                # For very short videos, sample all frames
                start_time = 0.0
                end_time = duration_s
            elif duration_s < 30.0:
                # For short videos, use smaller margins (5%)
                start_time = duration_s * 0.05
                end_time = duration_s * 0.95
            else:
                # For longer videos, use original margins (12%)
                start_time = duration_s * 0.12
                end_time = duration_s * 0.88

            # Adjust probe count for very short videos
            if duration_s < 5.0:
                probe_count = max(2, min(10, int(frame_count)))

            if start_time < end_time:
                times_s = np.linspace(
                    start_time,
                    end_time,
                    num=probe_count,
                    dtype=np.float64,
                )
            else:
                # Fallback if margins would overlap
                times_s = np.array([0.0], dtype=np.float64)
            idxs = np.rint(times_s * fps).astype(np.int64)
        elif frame_count > 1:
            # Adaptive margins based on frame count
            if frame_count < 10:
                # For very few frames, sample all frames
                start_idx = 0
                end_idx = frame_count - 1
            elif frame_count < 50:
                # For few frames, use smaller margins (5%)
                start_idx = max(0, int(round(frame_count * 0.05)))
                end_idx = max(1, int(round((frame_count - 1) * 0.95)))
            else:
                # For more frames, use original margins (12%)
                start_idx = max(0, int(round(frame_count * 0.12)))
                end_idx = max(1, int(round((frame_count - 1) * 0.88)))

            # Ensure end_idx > start_idx
            if start_idx >= end_idx:
                start_idx = 0
                end_idx = max(1, frame_count - 1)

            idxs = np.linspace(
                start_idx,
                end_idx,
                num=probe_count,
                dtype=np.int64,
            )
        else:
            idxs = np.array([0], dtype=np.int64)

        bounds = []
        for idx in idxs:
            if _cancel_check_requested(cancel_check):
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if _frame_texture_score(frame) < 4.0:
                continue
            top, bottom, left, right = _frame_active_content_bounds(frame)
            bounds.append((top, bottom, left, right))

        if not bounds:
            active = (0, height, 0, width)
        else:
            arr = np.asarray(bounds, dtype=np.float64)
            top = int(round(float(np.median(arr[:, 0]))))
            bottom = int(round(float(np.median(arr[:, 1]))))
            left = int(round(float(np.median(arr[:, 2]))))
            right = int(round(float(np.median(arr[:, 3]))))
            if bottom <= top or right <= left:
                active = (0, height, 0, width)
            else:
                active = (
                    max(0, min(height - 1, top)),
                    max(1, min(height, bottom)),
                    max(0, min(width - 1, left)),
                    max(1, min(width, right)),
                )

        top, bottom, left, right = active
        active_w = max(1, int(right - left))
        active_h = max(1, int(bottom - top))
        return {
            **meta,
            "active_top": int(top),
            "active_bottom": int(bottom),
            "active_left": int(left),
            "active_right": int(right),
            "active_width": int(active_w),
            "active_height": int(active_h),
            "active_aspect": float(active_w) / float(active_h),
            "cropped_bars": bool(
                top > 0 or left > 0 or bottom < height or right < width
            ),
        }
    finally:
        cap.release()


def _frame_texture_score(frame_bgr: np.ndarray) -> float:
    """Return a small grayscale texture/detail score for validation sampling."""
    if frame_bgr is None:
        return 0.0
    try:
        small = cv2.resize(frame_bgr, (256, 144), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        if contrast <= 1e-6:
            return 0.0
        edges = cv2.Canny(gray, 40, 140)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        return float(contrast + (edge_density * 255.0))
    except Exception:
        return 0.0


def _content_similarity_summary(scores: list[float]) -> float | None:
    """Score content match while tolerating occasional fades or weak seeks."""
    if not scores:
        return None
    ordered = sorted((float(s) for s in scores if np.isfinite(s)), reverse=True)
    if not ordered:
        return None
    # HDR/SDR pairs can differ heavily in grade, and OpenCV random access on
    # long HEVC files can land on weak/fade frames. Use the strongest stable
    # interior matches rather than letting one textureless sample dominate.
    top_n = min(3, len(ordered))
    return float(np.mean(ordered[:top_n]))


_VIDEO_SYNC_INFO_CACHE: dict[tuple, dict] = {}


def _video_sync_search_seconds() -> float:
    try:
        value = float(os.environ.get("HDRTVNET_GT_SYNC_OFFSET_SEARCH_S", "2.0"))
        if value > 0.0:
            return min(10.0, value)
    except Exception:
        pass
    return 2.0


def _video_sync_offset_min_gain() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("HDRTVNET_GT_SYNC_OFFSET_MIN_GAIN", "0.06")),
        )
    except Exception:
        return 0.06


def _video_sync_cache_key(
    source_path: str,
    candidate_path: str,
    sample_count: int = 5,
) -> tuple:
    def _sig(path: str) -> tuple:
        try:
            st = os.stat(path)
            return (
                os.path.normcase(os.path.abspath(path)),
                int(st.st_mtime_ns),
                int(st.st_size),
            )
        except Exception:
            return (os.path.normcase(os.path.abspath(path)), 0, 0)

    return (
        _sig(source_path),
        _sig(candidate_path),
        max(1, int(sample_count)),
        round(_video_sync_search_seconds(), 3),
    )


def _video_sync_offset_candidates(
    gt_fps: float,
    duration_delta_s: float = 0.0,
) -> list[int]:
    fps = float(gt_fps or 0.0)
    if fps <= 0.0:
        fps = 24.0
    max_frames = max(6, int(round(fps * _video_sync_search_seconds())))
    small_near = min(max_frames, 6)
    offsets = set(range(-small_near, small_near + 1))
    coarse_step = max(3, int(round(fps * 0.25)))
    for off in range(-max_frames, max_frames + 1, coarse_step):
        offsets.add(off)
    return sorted(offsets)


def _probe_video_sync_info(
    source_path: str,
    candidate_path: str,
    sample_count: int = 5,
    cancel_check=None,
) -> dict:
    """Estimate same-content score and a constant GT frame offset."""
    def _empty_info(*, canceled: bool = False) -> dict:
        return {
            "score": None,
            "offset_score": None,
            "sampled": 0,
            "offset_frames": 0,
            "offset_s": 0.0,
            "canceled": bool(canceled),
        }

    if _cancel_check_requested(cancel_check):
        return _empty_info(canceled=True)
    if not (
        _looks_like_video_path(source_path)
        and _looks_like_video_path(candidate_path)
    ):
        return _empty_info()
    if sample_count < 1:
        sample_count = 1
    cache_key = _video_sync_cache_key(source_path, candidate_path, sample_count)
    cached = _VIDEO_SYNC_INFO_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)

    cap_src = cv2.VideoCapture(source_path)
    cap_gt = cv2.VideoCapture(candidate_path)
    if not cap_src.isOpened() or not cap_gt.isOpened():
        cap_src.release()
        cap_gt.release()
        return _empty_info()
    try:
        if _cancel_check_requested(cancel_check):
            return _empty_info(canceled=True)
        src_n = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        gt_n = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if src_n <= 1 or gt_n <= 1:
            return _empty_info()
        src_fps = float(cap_src.get(cv2.CAP_PROP_FPS) or 0.0)
        gt_fps = float(cap_gt.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps > 0.0 and gt_fps > 0.0:
            src_duration_s = float(src_n) / src_fps
            gt_duration_s = float(gt_n) / gt_fps
            max_duration_s = min(src_duration_s, gt_duration_s)
            duration_delta_s = abs(src_duration_s - gt_duration_s)
        else:
            max_duration_s = 0.0
            duration_delta_s = 0.0
        probe_count = max(int(sample_count), 3)
        if max_duration_s > 2.0 and src_fps > 0.0 and gt_fps > 0.0:
            # Avoid leader/trailer/fade samples. They are common in movie
            # encodes and are poor evidence for "same content".
            times_s = np.linspace(
                max_duration_s * 0.08,
                max_duration_s * 0.92,
                num=probe_count,
                dtype=np.float64,
            )
            src_idxs = np.rint(times_s * src_fps).astype(np.int64)
            gt_base_idxs = np.rint(times_s * gt_fps).astype(np.int64)
        else:
            max_n = max(2, min(src_n, gt_n))
            src_idxs = np.linspace(
                max(0, int(round(max_n * 0.08))),
                max(1, int(round((max_n - 1) * 0.92))),
                num=probe_count,
                dtype=np.int64,
            )
            gt_base_idxs = src_idxs

        def _read_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            return frame

        src_samples = []
        for src_idx, gt_base_idx in zip(src_idxs, gt_base_idxs):
            if _cancel_check_requested(cancel_check):
                return _empty_info(canceled=True)
            src_frame = _read_at(cap_src, int(src_idx))
            if src_frame is None:
                continue
            src_texture = _frame_texture_score(src_frame)
            if src_texture < 4.0:
                continue
            src_samples.append(
                (int(src_idx), int(gt_base_idx), src_frame, float(src_texture))
            )

        offset_scores: dict[int, list[float]] = {}
        per_sample_best: list[float] = []
        tried_offsets: set[int] = set()

        def _score_offsets(offsets: list[int]) -> None:
            new_offsets: list[int] = []
            for offset in offsets:
                if int(offset) in tried_offsets:
                    continue
                tried_offsets.add(int(offset))
                new_offsets.append(int(offset))
                offset_scores.setdefault(int(offset), [])
            if not new_offsets:
                return
            for _src_idx, gt_base_idx, src_frame, src_texture in src_samples:
                if _cancel_check_requested(cancel_check):
                    return
                best_for_sample = None
                idx_to_offsets: dict[int, list[int]] = {}
                for offset in new_offsets:
                    gt_idx = max(0, min(gt_n - 1, gt_base_idx + int(offset)))
                    idx_to_offsets.setdefault(int(gt_idx), []).append(int(offset))
                if not idx_to_offsets:
                    continue
                target_idxs = sorted(idx_to_offsets)

                def _score_gt_frame(gt_idx: int, gt_frame: np.ndarray) -> None:
                    nonlocal best_for_sample
                    offsets_for_idx = idx_to_offsets.get(int(gt_idx))
                    if not offsets_for_idx:
                        return
                    gt_texture = _frame_texture_score(gt_frame)
                    if max(src_texture, gt_texture) < 4.0:
                        return
                    s = _frame_structure_similarity(src_frame, gt_frame)
                    if s is None:
                        return
                    for offset in offsets_for_idx:
                        offset_scores.setdefault(offset, []).append(float(s))
                    if best_for_sample is None or float(s) > best_for_sample:
                        best_for_sample = float(s)

                start_idx = target_idxs[0]
                end_idx = target_idxs[-1]
                span = max(1, int(end_idx - start_idx + 1))
                if len(target_idxs) * 2 < span:
                    for gt_idx in target_idxs:
                        if _cancel_check_requested(cancel_check):
                            return
                        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, int(gt_idx))
                        ok, gt_frame = cap_gt.read()
                        if not ok or gt_frame is None:
                            continue
                        _score_gt_frame(int(gt_idx), gt_frame)
                else:
                    cap_gt.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                    for gt_idx in range(start_idx, end_idx + 1):
                        if _cancel_check_requested(cancel_check):
                            return
                        ok, gt_frame = cap_gt.read()
                        if not ok or gt_frame is None:
                            break
                        _score_gt_frame(int(gt_idx), gt_frame)
                if best_for_sample is not None:
                    per_sample_best.append(float(best_for_sample))

        initial_offsets = _video_sync_offset_candidates(gt_fps, duration_delta_s)
        _score_offsets(initial_offsets)
        if _cancel_check_requested(cancel_check):
            return _empty_info(canceled=True)

        summaries: dict[int, tuple[float, int]] = {}
        for offset, scores in offset_scores.items():
            summary = _content_similarity_summary(scores)
            if summary is None:
                continue
            summaries[int(offset)] = (float(summary), int(len(scores)))

        max_offset = max(abs(v) for v in initial_offsets) if initial_offsets else 0
        dense_offsets = len(initial_offsets) >= (2 * int(max_offset) + 1)
        if not dense_offsets:
            refine_offsets: set[int] = set()
            for offset, (_score, _count) in sorted(
                summaries.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )[:4]:
                for delta in range(-3, 4):
                    refine_offsets.add(int(offset) + int(delta))
            refine_offsets = {
                int(v) for v in refine_offsets if abs(int(v)) <= int(max_offset)
            }
            _score_offsets(sorted(refine_offsets))
            if _cancel_check_requested(cancel_check):
                return _empty_info(canceled=True)

        summaries = {}
        for offset, scores in offset_scores.items():
            summary = _content_similarity_summary(scores)
            if summary is None:
                continue
            summaries[int(offset)] = (float(summary), int(len(scores)))

        if not summaries:
            info = _empty_info()
        else:
            best_offset, (best_score, best_count) = max(
                summaries.items(),
                key=lambda item: (item[1][0], item[1][1], -abs(item[0])),
            )
            zero_summary = summaries.get(0)
            zero_score_out = None
            offset_gain = None
            if best_offset != 0 and zero_summary is not None:
                zero_score, zero_count = zero_summary
                zero_score_out = float(zero_score)
                offset_gain = float(best_score) - float(zero_score)
                required_gain = _video_sync_offset_min_gain()
                if abs(int(best_offset)) <= 5:
                    required_gain = max(required_gain, 0.08)
                if float(best_score) < (float(zero_score) + float(required_gain)):
                    best_offset = 0
                    best_score = float(zero_score)
                    best_count = int(zero_count)
            elif zero_summary is not None:
                zero_score_out = float(zero_summary[0])
                offset_gain = float(best_score) - float(zero_summary[0])
            flexible_score = _content_similarity_summary(per_sample_best)
            final_score = best_score
            if flexible_score is not None:
                final_score = max(float(final_score), float(flexible_score))
            offset_s = (
                float(best_offset) / float(gt_fps)
                if gt_fps > 0.0
                else 0.0
            )
            info = {
                "score": float(final_score),
                "offset_score": float(best_score),
                "sampled": int(best_count),
                "offset_frames": int(best_offset),
                "offset_s": float(offset_s),
                "source_fps": float(src_fps),
                "gt_fps": float(gt_fps),
                "zero_offset_score": zero_score_out,
                "offset_gain": offset_gain,
            }
        if bool(info.get("canceled", False)):
            return info
        _VIDEO_SYNC_INFO_CACHE[cache_key] = dict(info)
        return info
    finally:
        cap_src.release()
        cap_gt.release()


def _content_similarity_score(
    source_path: str,
    candidate_path: str,
    sample_count: int = 5,
) -> tuple[float | None, int]:
    """Compare sampled frames to verify both videos show the same content."""
    info = _probe_video_sync_info(source_path, candidate_path, sample_count)
    try:
        score = info.get("score")
        sampled = int(info.get("sampled", 0) or 0)
    except Exception:
        return None, 0
    return (None if score is None else float(score)), sampled


def _map_video_pair_frame_index(
    source_path: str,
    candidate_path: str,
    source_frame_idx: int,
    *,
    sync_info: dict | None = None,
) -> int:
    """Map a source frame index onto the synced GT timeline."""
    idx = max(0, int(source_frame_idx))
    if not (
        _looks_like_video_path(source_path)
        and _looks_like_video_path(candidate_path)
    ):
        return idx
    src_meta = _probe_video_timing_info(source_path)
    gt_meta = _probe_video_timing_info(candidate_path)
    if gt_meta is None:
        return idx
    src_fps = float(src_meta.get("fps", 0.0) or 0.0) if src_meta else 0.0
    gt_fps = float(gt_meta.get("fps", 0.0) or 0.0)
    if src_fps > 0.0 and gt_fps > 0.0 and abs(src_fps - gt_fps) > 1e-3:
        idx = int(round((float(idx) / src_fps) * gt_fps))
    if sync_info is None:
        sync_info = _probe_video_sync_info(source_path, candidate_path, sample_count=3)
    try:
        idx += int(sync_info.get("offset_frames", 0) or 0)
    except Exception:
        pass
    gt_n = int(gt_meta.get("frame_count", 0) or 0)
    if gt_n > 0:
        idx = min(idx, gt_n - 1)
    return max(0, int(idx))

