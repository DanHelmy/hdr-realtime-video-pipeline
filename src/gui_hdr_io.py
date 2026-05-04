from __future__ import annotations

from collections import OrderedDict
import json
import os
import shutil
import subprocess
import math
import re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

FFMPEG_WINDOWS_DOWNLOAD_URL = "https://ffmpeg.org/download.html#build-windows"
_FFPROBE_STREAM_CACHE: OrderedDict[str, dict | None] = OrderedDict()
_HDR_FRAME_CACHE: OrderedDict[tuple[str, int, int, int], np.ndarray | None] = OrderedDict()
_HDR_FRAME_CACHE_MAX = 8
_STREAM_CACHE_MAX = 16
_HDR_FAST_SEEK_ENABLED = str(
    os.environ.get("HDRTVNET_HDR_FAST_SEEK", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_HDR_FAST_SEEK_PTS_GUARD_ENABLED = str(
    os.environ.get("HDRTVNET_HDR_FAST_SEEK_PTS_GUARD", "1")
).strip().lower() not in {"0", "false", "no", "off"}
try:
    _HDR_FAST_SEEK_PTS_TOL_FRAMES = max(
        0.10,
        float(os.environ.get("HDRTVNET_HDR_FAST_SEEK_PTS_TOL_FRAMES", "0.60")),
    )
except Exception:
    _HDR_FAST_SEEK_PTS_TOL_FRAMES = 0.60
try:
    _HDR_EXACT_BATCH_MAX_FRAMES = max(
        1,
        min(16, int(os.environ.get("HDRTVNET_HDR_EXACT_BATCH_MAX_FRAMES", "10"))),
    )
except Exception:
    _HDR_EXACT_BATCH_MAX_FRAMES = 10


def hdr_ffmpeg_ready() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _ensure_openexr_enabled() -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _ffprobe_cache_key(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{os.path.abspath(path)}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
    except Exception:
        return os.path.abspath(path) if path else ""


def _parse_ffprobe_fps(raw: str | None) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            num = float(num_s)
            den = float(den_s)
            if den <= 0.0:
                return None
            fps = num / den
        else:
            fps = float(text)
    except Exception:
        return None
    if not math.isfinite(fps) or fps <= 0.0:
        return None
    # Ignore nonsense metadata values.
    if fps > 1000.0:
        return None
    return float(fps)


def _ffprobe_video_stream_info(path: str) -> dict | None:
    cache_key = _ffprobe_cache_key(path)
    if cache_key:
        cached = _FFPROBE_STREAM_CACHE.get(cache_key)
        if cache_key in _FFPROBE_STREAM_CACHE:
            _FFPROBE_STREAM_CACHE.move_to_end(cache_key)
            return cached

    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path or not os.path.isfile(path):
        return None

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        path,
    ]

    info = None
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
        if streams:
            stream = streams[0]
            w = int(stream.get("width") or 0)
            h = int(stream.get("height") or 0)
            if w > 0 and h > 0:
                info = {"width": int(w), "height": int(h)}
                fps = _parse_ffprobe_fps(stream.get("avg_frame_rate"))
                if fps is None:
                    fps = _parse_ffprobe_fps(stream.get("r_frame_rate"))
                if fps is not None:
                    info["fps"] = float(fps)
    except Exception:
        info = None

    if cache_key:
        _FFPROBE_STREAM_CACHE[cache_key] = info
        _FFPROBE_STREAM_CACHE.move_to_end(cache_key)
        while len(_FFPROBE_STREAM_CACHE) > _STREAM_CACHE_MAX:
            _FFPROBE_STREAM_CACHE.popitem(last=False)
    return info


def _ffprobe_video_geometry(path: str) -> tuple[int, int] | None:
    info = _ffprobe_video_stream_info(path)
    if not isinstance(info, dict):
        return None
    w = int(info.get("width") or 0)
    h = int(info.get("height") or 0)
    if w <= 0 or h <= 0:
        return None
    return w, h


def _decode_rgb48_frame(cmd: list[str], width: int, height: int) -> np.ndarray | None:
    try:
        cp = subprocess.run(cmd, capture_output=True, check=True)
    except Exception:
        return None
    expected = int(width) * int(height) * 3 * 2
    data = cp.stdout or b""
    if len(data) < expected:
        return None
    try:
        frame = np.frombuffer(data[:expected], dtype=np.uint16).reshape((height, width, 3))
    except Exception:
        return None
    return np.ascontiguousarray(frame)


def _decode_rgb48_frame_with_pts(
    cmd: list[str],
    width: int,
    height: int,
) -> tuple[np.ndarray | None, float | None]:
    try:
        cp = subprocess.run(cmd, capture_output=True, check=True)
    except Exception:
        return None, None
    expected = int(width) * int(height) * 3 * 2
    data = cp.stdout or b""
    if len(data) < expected:
        return None, None
    try:
        frame = np.frombuffer(data[:expected], dtype=np.uint16).reshape((height, width, 3))
    except Exception:
        return None, None

    pts = None
    try:
        stderr_text = (cp.stderr or b"").decode("utf-8", errors="ignore")
        matches = re.findall(r"pts_time:([+-]?(?:\d+\.?\d*|\.\d+))", stderr_text)
        if matches:
            candidate = float(matches[-1])
            if math.isfinite(candidate):
                pts = float(candidate)
    except Exception:
        pts = None

    return np.ascontiguousarray(frame), pts


def _frame_idx_to_seek_timestamp(frame_idx: int, fps: float) -> str | None:
    if fps <= 0.0 or not math.isfinite(fps):
        return None
    # Seek to the center of the frame interval to reduce edge rounding drift.
    ts = (max(0, int(frame_idx)) + 0.5) / float(fps)
    if not math.isfinite(ts) or ts < 0.0:
        ts = 0.0
    return f"{ts:.9f}"


def read_hdr_video_frame_rgb16(
    path: str,
    frame_idx: int,
    prefer_fast_seek: bool | None = None,
    fast_seek_pts_guard: bool | None = None,
) -> np.ndarray | None:
    use_fast_seek = (
        _HDR_FAST_SEEK_ENABLED
        if prefer_fast_seek is None
        else bool(prefer_fast_seek)
    )
    use_fast_seek_pts_guard = (
        _HDR_FAST_SEEK_PTS_GUARD_ENABLED
        if fast_seek_pts_guard is None
        else bool(fast_seek_pts_guard)
    )
    cache_key: tuple[str, int, int, int] | None = None
    try:
        st = os.stat(path)
        cache_key = (
            f"{os.path.abspath(path)}|{int(st.st_mtime_ns)}|{int(st.st_size)}",
            max(0, int(frame_idx)),
            1 if use_fast_seek else 0,
            1 if use_fast_seek_pts_guard else 0,
        )
        if cache_key in _HDR_FRAME_CACHE:
            cached = _HDR_FRAME_CACHE[cache_key]
            _HDR_FRAME_CACHE.move_to_end(cache_key)
            return None if cached is None else np.ascontiguousarray(cached)
    except Exception:
        cache_key = None

    ffmpeg = shutil.which("ffmpeg")
    stream_info = _ffprobe_video_stream_info(path)
    if not ffmpeg or not isinstance(stream_info, dict):
        return None

    width = int(stream_info.get("width") or 0)
    height = int(stream_info.get("height") or 0)
    fps = float(stream_info.get("fps") or 0.0)
    if width <= 0 or height <= 0:
        return None

    try:
        idx = max(0, int(frame_idx))
    except Exception:
        idx = 0

    out = None

    # Fast path: timestamp seek. For CFR media this avoids scanning from frame 0
    # while preserving 16-bit linear decode (rgb48le).
    seek_ts = _frame_idx_to_seek_timestamp(idx, fps)
    if seek_ts is not None and use_fast_seek:
        if use_fast_seek_pts_guard and fps > 0.0:
            fast_cmd = [
                ffmpeg,
                "-v",
                "info",
                "-ss",
                seek_ts,
                "-i",
                path,
                "-map",
                "0:v:0",
                "-vf",
                "showinfo",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le",
                "-",
            ]
            out, pts_time = _decode_rgb48_frame_with_pts(fast_cmd, width, height)
            if out is not None:
                target_pts = float(idx) / float(fps)
                tol_s = max(
                    1e-3,
                    float(_HDR_FAST_SEEK_PTS_TOL_FRAMES) / float(fps),
                )
                if pts_time is None or abs(float(pts_time) - target_pts) > tol_s:
                    out = None
        else:
            fast_cmd = [
                ffmpeg,
                "-v",
                "error",
                "-ss",
                seek_ts,
                "-i",
                path,
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb48le",
                "-",
            ]
            out = _decode_rgb48_frame(fast_cmd, width, height)

    # Fallback path: exact frame index via select filter.
    if out is None:
        select_expr = f"select=eq(n\\,{idx})"
        exact_cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-vf",
            select_expr,
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-",
        ]
        out = _decode_rgb48_frame(exact_cmd, width, height)

    if out is None:
        if cache_key is not None:
            _HDR_FRAME_CACHE[cache_key] = None
            _HDR_FRAME_CACHE.move_to_end(cache_key)
            while len(_HDR_FRAME_CACHE) > _HDR_FRAME_CACHE_MAX:
                _HDR_FRAME_CACHE.popitem(last=False)
        return None

    if cache_key is not None:
        _HDR_FRAME_CACHE[cache_key] = out
        _HDR_FRAME_CACHE.move_to_end(cache_key)
        while len(_HDR_FRAME_CACHE) > _HDR_FRAME_CACHE_MAX:
            _HDR_FRAME_CACHE.popitem(last=False)
    return out


def _decode_rgb48_frames_batch(
    cmd: list[str],
    width: int,
    height: int,
    frame_indices: list[int],
    cancel_check=None,
) -> dict[int, np.ndarray | None]:
    out: dict[int, np.ndarray | None] = {int(v): None for v in frame_indices}
    frame_bytes = int(width) * int(height) * 3 * 2
    if frame_bytes <= 0 or not frame_indices:
        return out

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while True:
            try:
                data, _stderr = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if callable(cancel_check):
                    try:
                        if bool(cancel_check()):
                            proc.kill()
                            proc.communicate()
                            return out
                    except Exception:
                        pass
        if proc.returncode != 0:
            return out
    except Exception:
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        return out

    data = data or b""
    for pos, idx in enumerate(frame_indices):
        start = int(pos) * frame_bytes
        end = start + frame_bytes
        if end > len(data):
            break
        try:
            frame = np.frombuffer(data[start:end], dtype=np.uint16).reshape(
                (height, width, 3)
            )
            out[int(idx)] = np.ascontiguousarray(frame)
        except Exception:
            out[int(idx)] = None
    return out


def read_hdr_video_frames_rgb16_exact(
    path: str,
    frame_indices,
    cancel_check=None,
) -> dict[int, np.ndarray | None]:
    """Decode multiple exact HDR frames with one FFmpeg select pass per chunk.

    This intentionally uses the same `select=eq(n,idx)` + `rgb48le` path as
    read_hdr_video_frame_rgb16(..., prefer_fast_seek=False), but amortizes the
    decoder scan across sorted benchmark post-verify frames.
    """
    try:
        indices = sorted({max(0, int(v)) for v in frame_indices})
    except Exception:
        indices = []
    if not indices:
        return {}

    result: dict[int, np.ndarray | None] = {int(v): None for v in indices}
    ffmpeg = shutil.which("ffmpeg")
    stream_info = _ffprobe_video_stream_info(path)
    if not ffmpeg or not isinstance(stream_info, dict):
        return result

    width = int(stream_info.get("width") or 0)
    height = int(stream_info.get("height") or 0)
    if width <= 0 or height <= 0:
        return result

    max_chunk = int(_HDR_EXACT_BATCH_MAX_FRAMES)
    for start in range(0, len(indices), max_chunk):
        if callable(cancel_check):
            try:
                if bool(cancel_check()):
                    return result
            except Exception:
                pass
        chunk = indices[start:start + max_chunk]
        select_expr = "+".join(f"eq(n\\,{int(idx)})" for idx in chunk)
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            path,
            "-map",
            "0:v:0",
            "-vf",
            f"select={select_expr}",
            "-vsync",
            "0",
            "-frames:v",
            str(len(chunk)),
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-",
        ]
        result.update(
            _decode_rgb48_frames_batch(
                cmd,
                width,
                height,
                chunk,
                cancel_check=cancel_check,
            )
        )
    return result


def read_image_any(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    if str(path).lower().endswith(".exr"):
        _ensure_openexr_enabled()
    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        return None
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)


def write_hdr_exr(path: str, bgr_frame: np.ndarray) -> bool:
    if not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        return False
    arr = np.ascontiguousarray(bgr_frame)
    if arr.dtype == np.uint16:
        arr_u16 = arr
        arr_f32 = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr_u16 = arr.astype(np.uint16) * 257
        arr_f32 = arr.astype(np.float32) / 255.0
    else:
        arr_f32 = arr.astype(np.float32, copy=False)
        if float(np.max(arr_f32)) > 1.0:
            arr_f32 = np.clip(arr_f32 / 65535.0, 0.0, 1.0)
        arr_u16 = np.clip(arr_f32 * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h, w = arr_u16.shape[:2]
        rgb48 = np.ascontiguousarray(arr_u16[:, :, ::-1]).tobytes()
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "-s:v",
            f"{w}x{h}",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-vf",
            "format=gbrpf32le",
            path,
        ]
        try:
            subprocess.run(cmd, input=rgb48, check=True, capture_output=True)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return True
        except Exception:
            pass
    _ensure_openexr_enabled()
    try:
        ok = bool(cv2.imwrite(path, arr_f32))
        return ok and os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def write_hdr_tiff(path: str, bgr_frame: np.ndarray) -> bool:
    if not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        return False
    arr = np.ascontiguousarray(bgr_frame)
    if arr.dtype == np.uint16:
        out = arr
    elif arr.dtype == np.uint8:
        out = arr.astype(np.uint16) * 257
    else:
        arr_f32 = arr.astype(np.float32, copy=False)
        if float(np.max(arr_f32)) <= 1.0:
            arr_f32 = arr_f32 * 65535.0
        out = np.clip(arr_f32, 0.0, 65535.0).astype(np.uint16)
    try:
        ok = bool(cv2.imwrite(path, out))
        return ok and os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def tensor_to_rgb_u16(tensor) -> np.ndarray:
    with torch.inference_mode():
        prepared = tensor[0] if isinstance(tensor, (tuple, list)) else tensor
        rgb_u16 = (
            prepared.squeeze(0)
            .clamp(0.0, 1.0)
            .mul(65535.0)
            .add_(0.5)
            .to(dtype=torch.uint16)
            .permute(1, 2, 0)
            .contiguous()
        )
    return rgb_u16.cpu().numpy()


def tensor_to_bgr_u16(tensor) -> np.ndarray:
    rgb = tensor_to_rgb_u16(tensor)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def frame_to_rgb48_bytes(frame: np.ndarray) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected HxWx3 frame.")
    if frame.dtype == np.uint16:
        rgb16 = np.ascontiguousarray(frame[:, :, ::-1])
    elif frame.dtype == np.uint8:
        rgb16 = np.ascontiguousarray(frame[:, :, ::-1].astype(np.uint16) * 257)
    else:
        arr = frame.astype(np.float32)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 65535.0
        rgb16 = np.ascontiguousarray(
            np.clip(arr[:, :, ::-1], 0.0, 65535.0).astype(np.uint16)
        )
    return rgb16.tobytes()
