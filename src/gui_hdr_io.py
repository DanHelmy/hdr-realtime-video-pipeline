from __future__ import annotations

from collections import OrderedDict
import json
import os
import shutil
import subprocess

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

FFMPEG_WINDOWS_DOWNLOAD_URL = "https://ffmpeg.org/download.html#build-windows"
_FFPROBE_GEOMETRY_CACHE: OrderedDict[str, tuple[int, int] | None] = OrderedDict()
_HDR_FRAME_CACHE: OrderedDict[tuple[str, int], np.ndarray | None] = OrderedDict()
_HDR_FRAME_CACHE_MAX = 8
_GEOMETRY_CACHE_MAX = 16


def hdr_ffmpeg_ready() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _ensure_openexr_enabled() -> None:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _ffprobe_video_geometry(path: str) -> tuple[int, int] | None:
    cache_key = ""
    try:
        st = os.stat(path)
        cache_key = f"{os.path.abspath(path)}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
    except Exception:
        cache_key = os.path.abspath(path) if path else ""
    if cache_key:
        cached = _FFPROBE_GEOMETRY_CACHE.get(cache_key)
        if cache_key in _FFPROBE_GEOMETRY_CACHE:
            _FFPROBE_GEOMETRY_CACHE.move_to_end(cache_key)
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
        "stream=width,height",
        "-of",
        "json",
        path,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(cp.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0]
        w = int(stream.get("width") or 0)
        h = int(stream.get("height") or 0)
        if w > 0 and h > 0:
            if cache_key:
                _FFPROBE_GEOMETRY_CACHE[cache_key] = (w, h)
                _FFPROBE_GEOMETRY_CACHE.move_to_end(cache_key)
                while len(_FFPROBE_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAX:
                    _FFPROBE_GEOMETRY_CACHE.popitem(last=False)
            return w, h
    except Exception:
        return None
    if cache_key:
        _FFPROBE_GEOMETRY_CACHE[cache_key] = None
        _FFPROBE_GEOMETRY_CACHE.move_to_end(cache_key)
        while len(_FFPROBE_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAX:
            _FFPROBE_GEOMETRY_CACHE.popitem(last=False)
    return None


def read_hdr_video_frame_rgb16(path: str, frame_idx: int) -> np.ndarray | None:
    cache_key: tuple[str, int] | None = None
    try:
        st = os.stat(path)
        cache_key = (
            f"{os.path.abspath(path)}|{int(st.st_mtime_ns)}|{int(st.st_size)}",
            max(0, int(frame_idx)),
        )
        if cache_key in _HDR_FRAME_CACHE:
            cached = _HDR_FRAME_CACHE[cache_key]
            _HDR_FRAME_CACHE.move_to_end(cache_key)
            return None if cached is None else np.ascontiguousarray(cached)
    except Exception:
        cache_key = None
    ffmpeg = shutil.which("ffmpeg")
    geom = _ffprobe_video_geometry(path)
    if not ffmpeg or geom is None:
        return None
    width, height = geom
    if width <= 0 or height <= 0:
        return None
    try:
        idx = max(0, int(frame_idx))
    except Exception:
        idx = 0
    select_expr = f"select=eq(n\\,{idx})"
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        select_expr,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, check=True)
    except Exception:
        return None
    expected = width * height * 3 * 2
    data = cp.stdout or b""
    if len(data) < expected:
        return None
    try:
        frame = np.frombuffer(data[:expected], dtype=np.uint16).reshape((height, width, 3))
    except Exception:
        return None
    out = np.ascontiguousarray(frame)
    if cache_key is not None:
        _HDR_FRAME_CACHE[cache_key] = out
        _HDR_FRAME_CACHE.move_to_end(cache_key)
        while len(_HDR_FRAME_CACHE) > _HDR_FRAME_CACHE_MAX:
            _HDR_FRAME_CACHE.popitem(last=False)
    return out


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
