"""Static GUI configuration/constants (precision + resolution presets)."""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _weight(name: str) -> str:
    return os.path.join(_HERE, "models", "weights", name)


PRECISIONS = {
    "FP16": {
        "precision": "fp16",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "FP32": {
        "precision": "fp32",
        "model": _weight("Ensemble_AGCM_LE.pth"),
    },
    "INT8 Mixed (PTQ)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_nohg.pt"),
    },
    "INT8 Mixed (QAT)": {
        "precision": "int8-mixed",
        "model": _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_mixed_qat_nohg.pt"),
    },
    "INT8 Full (PTQ)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_nohg.pt"),
    },
    "INT8 Full (QAT)": {
        "precision": "int8-full",
        "model": _weight("Ensemble_AGCM_LE_int8_full_qat.pt"),
        "model_nohg": _weight("Ensemble_AGCM_LE_int8_full_qat_nohg.pt"),
    },
}

DEFAULT_PRECISION_KEY = "INT8 Mixed (QAT)"
DEFAULT_RESOLUTION_KEY = "720p"
DEFAULT_USE_HG = False


def _select_model_path(precision_key: str, use_hg: bool) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model", "")
    if cfg.get("precision", "").startswith("int8") and not use_hg:
        model_path = cfg.get("model_nohg") or model_path
    return model_path


def _precision_is_available(precision_key: str) -> bool:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model")
    if model_path and os.path.isfile(model_path):
        return True
    if cfg.get("precision", "").startswith("int8"):
        alt_path = cfg.get("model_nohg")
        if alt_path and os.path.isfile(alt_path):
            return True
    return False


def _available_precision_keys() -> list[str]:
    keys = [k for k in PRECISIONS.keys() if _precision_is_available(k)]
    return keys or list(PRECISIONS.keys())


MAX_W, MAX_H = 1920, 1080

SOURCE_MODE_VIDEO = "video"
SOURCE_MODE_WINDOW = "window_capture"

SOURCE_MODE_LABELS = {
    SOURCE_MODE_VIDEO: "Video Player",
    SOURCE_MODE_WINDOW: "Browser Window Capture (Experimental)",
}

def _env_live_fps(name: str, default: float, *, max_value: float = 120.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = float(default)
    if value <= 0.0:
        value = float(default)
    return max(1.0, min(float(max_value), float(value)))


def _env_live_int(name: str, default: int, *, min_value: int = 1, max_value: int = 12) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = int(default)
    return max(int(min_value), min(int(max_value), int(value)))


# Browser-window video has two separate rates:
# - observe FPS controls how often we check Chrome/DWM for fresh compositor frames
# - process FPS controls how often HDRTVNet++ runs on browser frames
# - display FPS controls the steady raw-video stream fed to mpv
# Keep display feed at or below process FPS. mpv/display vsync should own frame
# repeats; oversampling the pipe can create uneven duplicate cadence on 24 fps
# browser video.
LIVE_CAPTURE_PROCESS_FPS = _env_live_fps("HDRTVNET_LIVE_CAPTURE_PROCESS_FPS", 24.0)
LIVE_CAPTURE_OBSERVE_FPS = _env_live_fps(
    "HDRTVNET_LIVE_CAPTURE_OBSERVE_FPS",
    max(30.0, LIVE_CAPTURE_PROCESS_FPS),
)
LIVE_CAPTURE_MPV_BUFFER_FRAMES = _env_live_int(
    "HDRTVNET_LIVE_CAPTURE_MPV_BUFFER_FRAMES",
    3,
)
LIVE_CAPTURE_PRESENT_MAX_FPS = LIVE_CAPTURE_PROCESS_FPS
LIVE_CAPTURE_DISPLAY_FPS = min(
    LIVE_CAPTURE_PROCESS_FPS,
    _env_live_fps(
        "HDRTVNET_LIVE_CAPTURE_DISPLAY_FPS",
        LIVE_CAPTURE_PROCESS_FPS,
    ),
)


def _normalize_source_mode(mode: str | None) -> str:
    text = str(mode or "").strip().lower()
    text = text.replace("(experimental)", "").strip()
    # Keep legacy labels accepted so older prefs/CLI args still route into the
    # live browser-window capture backend.
    if text in {
        SOURCE_MODE_WINDOW,
        "window",
        "capture",
        "window capture",
        "browser window",
        "browser window capture",
        "live browser window",
        "live browser window capture",
        "live window",
        "browser tab",
        "browser tab capture",
        "tab capture",
    }:
        return SOURCE_MODE_WINDOW
    return SOURCE_MODE_VIDEO


def _source_mode_label(mode: str | None) -> str:
    key = _normalize_source_mode(mode)
    return SOURCE_MODE_LABELS.get(key, SOURCE_MODE_LABELS[SOURCE_MODE_VIDEO])


# Resolution-scale presets (process lower resolution).
RESOLUTION_SCALES = {
    "1080p": None,  # full output resolution path (no upscale stage)
    "720p": (1280, 720),
    "540p": (960, 540),
}

_PROCESSING_PRESET_ORDER = tuple(RESOLUTION_SCALES.keys())


def _processing_preset_dims(resolution_key: str) -> tuple[int, int]:
    dims = RESOLUTION_SCALES.get(str(resolution_key or ""))
    if dims is None:
        return MAX_W, MAX_H
    return int(dims[0]), int(dims[1])


def _max_processing_preset_for_source(src_w: int, src_h: int) -> str:
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    ascending = tuple(reversed(_PROCESSING_PRESET_ORDER))
    for key in ascending:
        pw, ph = _processing_preset_dims(key)
        if src_w <= pw and src_h <= ph:
            return key
    return "1080p"


def _processing_preset_options_for_source(src_w: int, src_h: int) -> list[str]:
    max_key = _max_processing_preset_for_source(src_w, src_h)
    options: list[str] = []
    include = False
    for key in _PROCESSING_PRESET_ORDER:
        if key == max_key:
            include = True
        if include:
            options.append(key)
    return options or list(_PROCESSING_PRESET_ORDER)


def _source_is_below_processing_preset(
    src_w: int,
    src_h: int,
    resolution_key: str,
) -> bool:
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    pw, ph = _processing_preset_dims(resolution_key)
    return src_w <= pw and src_h <= ph and (src_w < pw or src_h < ph)
