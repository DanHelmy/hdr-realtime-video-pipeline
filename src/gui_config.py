"""Static GUI configuration/constants (precision + resolution presets)."""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _weight(name: str) -> str:
    return os.path.join(_HERE, "models", "weights", name)


PRECISIONS = {
    "FP16": {
        "precision": "fp16",
        "engine_mode": "FP16",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/HR.pt"),
        "trt_model_nohg": _weight("original/HR.pt"),
        "trt_hg_weights": _weight("original/HG.pt"),
    },
    "FP32": {
        "precision": "fp32",
        "engine_mode": "FP32",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/HR.pt"),
        "trt_model_nohg": _weight("original/HR.pt"),
        "trt_hg_weights": _weight("original/HG.pt"),
    },
    "INT8 Mixed (PTQ)": {
        "precision": "int8-mixed",
        "engine_mode": "original_int8-mixed-ptq",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_ptq.pt"),
    },
    "INT8 Mixed (QAT)": {
        "precision": "int8-mixed",
        "engine_mode": "original_int8-mixed-qat",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_qat.pt"),
    },
    "INT8 Mixed (QAT) (Film)": {
        "precision": "int8-mixed",
        "engine_mode": "original_int8-mixed-qat-film",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat_film.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat_film.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_qat_film.pt"),
    },
    "INT8 Full (PTQ)": {
        "precision": "int8-full",
        "engine_mode": "original_int8-full-ptq",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_ptq.pt"),
    },
    "INT8 Full (QAT)": {
        "precision": "int8-full",
        "engine_mode": "original_int8-full-qat",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full_qat.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_qat.pt"),
    },
    "INT8 Full (QAT) (Film)": {
        "precision": "int8-full",
        "engine_mode": "original_int8-full-qat-film",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat_film.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full_qat_film.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_qat_film.pt"),
    },
}

DEFAULT_PRECISION_KEY = "INT8 Mixed (QAT)"
DEFAULT_RESOLUTION_KEY = "1080p"
DEFAULT_USE_HG = False
INT8_HG_WARNING = (
    "INT8 TensorRT speedups depend on the selected graph and GPU generation. "
    "Using HG can require generating both the local HG TensorRT source model "
    "checkpoint and the TensorRT engine before playback starts. Mixed QAT "
    "presets use protected TensorRT source checkpoints by default; Full INT8 "
    "remains a strict all-quantizer baseline for testing."
)


def _select_model_path(precision_key: str, use_hg: bool) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("model", "")
    if not use_hg:
        model_path = cfg.get("model_nohg") or model_path
    return model_path


def _select_tensorrt_model_path(precision_key: str, use_hg: bool) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    model_path = cfg.get("trt_model") or cfg.get("model", "")
    if not use_hg:
        model_path = (
            cfg.get("trt_model_nohg")
            or cfg.get("model_nohg")
            or model_path
        )
    return model_path


def _select_hg_weights_path(precision_key: str, *, tensorrt: bool = False) -> str:
    cfg = PRECISIONS.get(precision_key, {})
    if tensorrt:
        if "trt_hg_weights" in cfg:
            return cfg.get("trt_hg_weights") or ""
        return _weight("original/HG.pt")
    return cfg.get("hg_weights") or _weight("original/HG.pt")


def _precision_engine_mode_base(precision_key: str) -> str:
    cfg = PRECISIONS.get(str(precision_key), {})
    return str(
        cfg.get("engine_mode")
        or cfg.get("precision")
        or precision_key
        or "mode"
    ).strip()


def _precision_is_int8(precision_key: str) -> bool:
    return str(PRECISIONS.get(precision_key, {}).get("precision", "")).startswith("int8")


def _int8_precision_warning(precision_key: str, use_hg: bool) -> str:
    if not _precision_is_int8(precision_key):
        return ""
    if not use_hg:
        return (
            "No-HG INT8 uses the HR/ACGM/LE TensorRT source path. It is "
            "usually the fastest preset when HG reconstruction is not needed."
        )
    return INT8_HG_WARNING


def _precision_is_available(precision_key: str) -> bool:
    cfg = PRECISIONS.get(precision_key, {})
    model_paths = [
        cfg.get("model"),
        cfg.get("model_nohg"),
        cfg.get("trt_model"),
        cfg.get("trt_model_nohg"),
    ]
    for model_path in model_paths:
        if model_path and os.path.isfile(model_path):
            return True
    if cfg.get("precision", "").startswith("int8"):
        alt_path = cfg.get("model_nohg")
        if alt_path and os.path.isfile(alt_path):
            return True
    return False


def _available_precision_keys() -> list[str]:
    keys = [
        k
        for k, cfg in PRECISIONS.items()
        if not bool(cfg.get("hidden", False)) and _precision_is_available(k)
    ]
    return keys or [
        k for k, cfg in PRECISIONS.items() if not bool(cfg.get("hidden", False))
    ]


MAX_W, MAX_H = 3840, 2160

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


LIVE_CAPTURE_FPS_CHOICES = (24, 30, 60)


def live_capture_process_fps_from_value(value: object, default: float | None = None) -> float:
    """Clamp GUI live-capture FPS to the supported low-latency presets."""
    fallback = 24.0 if default is None else float(default)
    try:
        fps = int(round(float(value)))
    except Exception:
        fps = int(round(fallback))
    if fps in LIVE_CAPTURE_FPS_CHOICES:
        return float(fps)
    if int(round(fallback)) in LIVE_CAPTURE_FPS_CHOICES:
        return float(int(round(fallback)))
    return 24.0


def live_capture_observe_fps(process_fps: float) -> float:
    # Poll the compositor faster than the HDRTVNet++ budget so 24/30 fps
    # processing usually receives a recent browser frame without making the
    # mpv pipe itself run at high FPS.
    process_fps = max(1.0, float(process_fps or 24.0))
    return _env_live_fps(
        "HDRTVNET_LIVE_CAPTURE_OBSERVE_FPS",
        max(36.0, process_fps * 2.0),
    )


def live_capture_display_fps(process_fps: float) -> float:
    process_fps = max(1.0, float(process_fps or 24.0))
    return min(
        process_fps,
        _env_live_fps(
            "HDRTVNET_LIVE_CAPTURE_DISPLAY_FPS",
            process_fps,
        ),
    )


# Browser-window video has two separate rates:
# - observe FPS controls how often we check Chrome/DWM for fresh compositor frames
# - process FPS controls how often HDRTVNet++ runs on browser frames
# - display FPS controls the steady raw-video stream fed to mpv
# Keep display feed at or below process FPS. mpv/display vsync should own frame
# repeats; oversampling the pipe can create uneven duplicate cadence on 24 fps
# browser video.
LIVE_CAPTURE_PROCESS_FPS = _env_live_fps("HDRTVNET_LIVE_CAPTURE_PROCESS_FPS", 24.0)
DEFAULT_LIVE_CAPTURE_PROCESS_FPS = live_capture_process_fps_from_value(
    LIVE_CAPTURE_PROCESS_FPS,
    default=24.0,
)
LIVE_CAPTURE_OBSERVE_FPS = live_capture_observe_fps(LIVE_CAPTURE_PROCESS_FPS)
LIVE_CAPTURE_MPV_BUFFER_FRAMES = _env_live_int(
    "HDRTVNET_LIVE_CAPTURE_MPV_BUFFER_FRAMES",
    8,  # Extra buffer for smoother live playback
)
VIDEO_PLAYBACK_BUFFER_FRAMES = _env_live_int(
    "HDRTVNET_VIDEO_PLAYBACK_BUFFER_FRAMES",
    3,
    min_value=1,
    max_value=24,
)
VIDEO_PLAYBACK_PRESERVE_ORDER = (
    str(os.environ.get("HDRTVNET_VIDEO_PLAYBACK_PRESERVE_ORDER", "0"))
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
LIVE_CAPTURE_PRESENT_MAX_FPS = LIVE_CAPTURE_PROCESS_FPS
LIVE_CAPTURE_DISPLAY_FPS = live_capture_display_fps(LIVE_CAPTURE_PROCESS_FPS)


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
    "2160p": None,  # full processing/output preset; monitor upscale is separate
    "1440p": (2560, 1440),
    "1080p": (1920, 1080),
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
    return "2160p"


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
