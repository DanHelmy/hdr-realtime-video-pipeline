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


def _content_similarity_score(
    source_path: str,
    candidate_path: str,
    sample_count: int = 5,
) -> tuple[float | None, int]:
    """Compare sampled frames to verify both videos show the same content."""
    if sample_count < 1:
        sample_count = 1
    cap_src = cv2.VideoCapture(source_path)
    cap_gt = cv2.VideoCapture(candidate_path)
    if not cap_src.isOpened() or not cap_gt.isOpened():
        cap_src.release()
        cap_gt.release()
        return None, 0
    try:
        src_n = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        gt_n = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if src_n <= 1 or gt_n <= 1:
            return None, 0
        max_n = max(2, min(src_n, gt_n))
        idxs = np.linspace(0, max_n - 1, num=sample_count, dtype=np.int64)
        scores = []

        def _read_at(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            return frame

        for idx in idxs:
            src_frame = _read_at(cap_src, int(idx))
            if src_frame is None:
                continue
            best = None
            for shift in (-1, 0, 1):
                gt_idx = max(0, min(max_n - 1, int(idx) + shift))
                gt_frame = _read_at(cap_gt, gt_idx)
                if gt_frame is None:
                    continue
                s = _frame_structure_similarity(src_frame, gt_frame)
                if s is None:
                    continue
                if best is None or s > best:
                    best = s
            if best is not None:
                scores.append(float(best))

        if not scores:
            return None, 0
        return float(np.mean(scores)), int(len(scores))
    finally:
        cap_src.release()
        cap_gt.release()

