"""Scaling/shader helpers and constants for GUI video rendering."""

from __future__ import annotations

import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BEST_UPSCALE_MODE = "On (Best)"
BEST_MPV_SCALE = "ewa_lanczossharp"
BEST_CV2_INTERP = cv2.INTER_LANCZOS4
UPSCALE_SHARPEN_STRENGTH = 0.0
UPSCALE_SHARPEN_SIGMA = 1.0

FSR_SHADER_PATH = os.path.join(_ROOT, "assets", "shaders", "FSR.glsl")
FSR_SHADER_URL = (
    "https://gist.githubusercontent.com/agyild/"
    "82219c545228d70c5604f865ce0b0ce5/raw/"
    "2623d743b9c23f500ba086f05b385dcb1557e15d/FSR.glsl"
)
FILMGRAIN_SHADER_PATH = os.path.join(_ROOT, "assets", "shaders", "filmgrain.glsl")
FILMGRAIN_SHADER_URL = (
    "https://raw.githubusercontent.com/haasn/gentoo-conf/"
    "xor/home/nand/.mpv/shaders/filmgrain.glsl"
)
SSIM_SUPERRES_SHADER_PATH = os.path.join(_ROOT, "assets", "shaders", "SSimSuperRes.glsl")
SSIM_SUPERRES_SHADER_URL = (
    "https://gist.githubusercontent.com/igv/"
    "2364ffa6e81540f29cb7ab4c9bc05b6b/raw/"
    "15d93440d0a24fc4b8770070be6a9fa2af6f200b/SSimSuperRes.glsl"
)
UPSCALER_CHOICES = ["EWA LanczosSharp", "FSR", "SSimSuperRes"]
DEFAULT_UPSCALER = "EWA LanczosSharp"


def _fit_with_aspect(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Fit source resolution into a bounding box while preserving aspect ratio."""
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    out_w = max(2, int(round(src_w * scale)))
    out_h = max(2, int(round(src_h * scale)))
    # Keep even dimensions for encoder/decoder friendliness.
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


def _limited_playback_fps(src_fps: float) -> float:
    """Limit playback FPS by halving high-FPS sources (e.g., 50->25, 60->30)."""
    fps = float(src_fps) if src_fps and src_fps > 0 else 30.0
    while fps > 30.0:
        fps *= 0.5
    return max(1.0, fps)


def _select_hdr_scale_kernel(
    proc_w: int,
    proc_h: int,
    out_w: int,
    out_h: int,
    upscale_choice: str | None = None,
) -> str:
    """Pick mpv scale kernel for HDR output."""
    if proc_w == out_w and proc_h == out_h:
        return "bicubic"
    if upscale_choice:
        return _normalize_upscale_choice(upscale_choice)
    return BEST_MPV_SCALE


def _select_hdr_scale_antiring(
    proc_w: int,
    proc_h: int,
    out_w: int,
    out_h: int,
    scale_kernel: str | None = None,
) -> float:
    """Tune antiring strength by processing resolution and kernel."""
    if proc_w >= out_w and proc_h >= out_h:
        return 0.0
    k = str(scale_kernel or "").strip().lower()
    if k == "fsr":
        return 0.0
    if "ssim" in k:
        return 0.0
    if proc_h <= 540 or proc_w <= 960:
        base = 0.30
    elif proc_h <= 720 or proc_w <= 1280:
        base = 0.22
    else:
        base = 0.10
    if "lanczossharp" in k or k == "ewa_lanczos":
        return max(0.0, base - 0.05)
    return base


def _select_mpv_cas_strength(
    proc_w: int,
    proc_h: int,
    out_w: int,
    out_h: int,
    using_fsr: bool = False,
    scale_kernel: str | None = None,
) -> float:
    """Select CAS sharpening strength for HDR upscale."""
    if proc_w >= out_w and proc_h >= out_h:
        return 0.0
    if using_fsr:
        return 0.0
    k = str(scale_kernel or "").strip().lower()
    if "ssim" in k:
        return 0.0
    if proc_h <= 540 or proc_w <= 960:
        base = 0.22
    elif proc_h <= 720 or proc_w <= 1280:
        base = 0.20
    else:
        base = 0.16
    if "lanczossharp" in k or k == "ewa_lanczos":
        return base + 0.02
    return base


def _normalize_upscale_choice(choice: str) -> str:
    c = str(choice or "").strip().lower()
    if "fsr" in c:
        return "fsr"
    if "ssim" in c:
        return "ssim_superres"
    return BEST_MPV_SCALE


def _ensure_fsr_shader() -> bool:
    """Ensure the FSR shader exists on disk (download on demand)."""
    if os.path.isfile(FSR_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(FSR_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(FSR_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(FSR_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[fsr] download failed: {exc}")
    return os.path.isfile(FSR_SHADER_PATH)


def _ensure_filmgrain_shader() -> bool:
    """Ensure the film grain shader exists on disk (download on demand)."""
    if os.path.isfile(FILMGRAIN_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(FILMGRAIN_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(FILMGRAIN_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(FILMGRAIN_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[filmgrain] download failed: {exc}")
    return os.path.isfile(FILMGRAIN_SHADER_PATH)


def _ensure_ssim_superres_shader() -> bool:
    """Ensure the SSimSuperRes shader exists on disk (download on demand)."""
    if os.path.isfile(SSIM_SUPERRES_SHADER_PATH):
        return True
    try:
        os.makedirs(os.path.dirname(SSIM_SUPERRES_SHADER_PATH), exist_ok=True)
        import urllib.request
        with urllib.request.urlopen(SSIM_SUPERRES_SHADER_URL, timeout=10) as resp:
            data = resp.read()
        if data:
            with open(SSIM_SUPERRES_SHADER_PATH, "wb") as f:
                f.write(data)
            return True
    except Exception as exc:
        print(f"[ssim] download failed: {exc}")
    return os.path.isfile(SSIM_SUPERRES_SHADER_PATH)


def _normalize_shader_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if p]


def _letterbox_bgr(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize with preserved aspect ratio and black padding to exact output size."""
    h, w = frame.shape[:2]
    if w == out_w and h == out_h:
        return frame

    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    x = (out_w - new_w) // 2
    y = (out_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def _apply_upscale_sharpen(
    img: np.ndarray,
    strength: float = UPSCALE_SHARPEN_STRENGTH,
    sigma: float = UPSCALE_SHARPEN_SIGMA,
) -> np.ndarray:
    """Mild unsharp mask after upscaling."""
    if strength <= 0.0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)

