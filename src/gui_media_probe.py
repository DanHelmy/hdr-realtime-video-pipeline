"""Media probing and SDR/HDR compatibility helpers for the GUI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import cv2
import numpy as np


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


def _probe_video_active_area_info(
    video_path: str,
    sample_count: int = 7,
) -> dict | None:
    """Estimate the active picture area after encoded black-bar cropping."""
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
        probe_count = max(3, int(sample_count))
        if duration_s > 2.0 and fps > 0.0:
            times_s = np.linspace(
                duration_s * 0.12,
                duration_s * 0.88,
                num=probe_count,
                dtype=np.float64,
            )
            idxs = np.rint(times_s * fps).astype(np.int64)
        elif frame_count > 1:
            idxs = np.linspace(
                max(0, int(round(frame_count * 0.12))),
                max(1, int(round((frame_count - 1) * 0.88))),
                num=probe_count,
                dtype=np.int64,
            )
        else:
            idxs = np.array([0], dtype=np.int64)

        bounds = []
        for idx in idxs:
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


def _video_sync_offset_candidates(gt_fps: float) -> list[int]:
    fps = float(gt_fps or 0.0)
    if fps <= 0.0:
        fps = 24.0
    max_frames = max(2, int(round(fps * _video_sync_search_seconds())))
    offsets = {0, -1, 1, -2, 2}
    near = min(max_frames, int(round(fps)))
    for off in range(-near, near + 1):
        offsets.add(off)
    coarse_step = max(3, int(round(fps * 0.25)))
    for off in range(-max_frames, max_frames + 1, coarse_step):
        offsets.add(off)
    return sorted(offsets)


def _probe_video_sync_info(
    source_path: str,
    candidate_path: str,
    sample_count: int = 5,
) -> dict:
    """Estimate same-content score and a constant GT frame offset."""
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
        return {
            "score": None,
            "offset_score": None,
            "sampled": 0,
            "offset_frames": 0,
            "offset_s": 0.0,
        }
    try:
        src_n = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        gt_n = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if src_n <= 1 or gt_n <= 1:
            return {
                "score": None,
                "offset_score": None,
                "sampled": 0,
                "offset_frames": 0,
                "offset_s": 0.0,
            }
        src_fps = float(cap_src.get(cv2.CAP_PROP_FPS) or 0.0)
        gt_fps = float(cap_gt.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps > 0.0 and gt_fps > 0.0:
            max_duration_s = min(float(src_n) / src_fps, float(gt_n) / gt_fps)
        else:
            max_duration_s = 0.0
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
                best_for_sample = None
                idx_to_offsets: dict[int, list[int]] = {}
                for offset in new_offsets:
                    gt_idx = max(0, min(gt_n - 1, gt_base_idx + int(offset)))
                    idx_to_offsets.setdefault(int(gt_idx), []).append(int(offset))
                if not idx_to_offsets:
                    continue
                start_idx = min(idx_to_offsets)
                end_idx = max(idx_to_offsets)
                cap_gt.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                for gt_idx in range(start_idx, end_idx + 1):
                    ok, gt_frame = cap_gt.read()
                    if not ok or gt_frame is None:
                        break
                    offsets_for_idx = idx_to_offsets.get(int(gt_idx))
                    if not offsets_for_idx:
                        continue
                    gt_texture = _frame_texture_score(gt_frame)
                    if max(src_texture, gt_texture) < 4.0:
                        continue
                    s = _frame_structure_similarity(src_frame, gt_frame)
                    if s is None:
                        continue
                    for offset in offsets_for_idx:
                        offset_scores.setdefault(offset, []).append(float(s))
                    if best_for_sample is None or float(s) > best_for_sample:
                        best_for_sample = float(s)
                if best_for_sample is not None:
                    per_sample_best.append(float(best_for_sample))

        initial_offsets = _video_sync_offset_candidates(gt_fps)
        _score_offsets(initial_offsets)

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

        summaries = {}
        for offset, scores in offset_scores.items():
            summary = _content_similarity_summary(scores)
            if summary is None:
                continue
            summaries[int(offset)] = (float(summary), int(len(scores)))

        if not summaries:
            info = {
                "score": None,
                "offset_score": None,
                "sampled": 0,
                "offset_frames": 0,
                "offset_s": 0.0,
            }
        else:
            best_offset, (best_score, best_count) = max(
                summaries.items(),
                key=lambda item: (item[1][0], item[1][1], -abs(item[0])),
            )
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
            }
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

