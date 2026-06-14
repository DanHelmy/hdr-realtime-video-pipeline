from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings
from collections import deque
from datetime import datetime

if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from windows_runtime import (
    configure_cuda_environment,
    configure_msvc_build_environment,
    configure_rocm_sdk_environment,
    ensure_windows_supported,
    install_torch_windows_warning_filter,
    project_cache_root,
)

ensure_windows_supported("HDRTVNet++ CLI playback benchmark")
install_torch_windows_warning_filter()
configure_msvc_build_environment()
configure_cuda_environment()
configure_rocm_sdk_environment()

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_CACHE_ROOT = project_cache_root(__file__)
os.makedirs(_CACHE_ROOT, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_CACHE_ROOT, "torchinductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_ROOT, "triton"))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import cv2
import psutil
import torch

from cli_display import CliDisplaySink
from gui_compile_cache import _mark_compiled
from gui_config import (
    PRECISIONS,
    _precision_engine_mode_base,
    _precision_is_quantized_tensorrt,
    _select_hg_weights_path,
    _select_tensorrt_model_path,
)
from gui_scaling import _limited_playback_fps
from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    HDRTVNetTorch,
    _IS_NVIDIA,
    tensorrt_prebuilt_calibration_cache_path,
)
from timer import sleep_until
from video_source import VideoSource


def _weight(name: str) -> str:
    return str(_HERE / "models" / "weights" / name)


_RUN_PRESETS = {
    "fp16": {
        "precision": "fp16",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/HR.pt"),
        "trt_model_nohg": _weight("original/HR.pt"),
        "trt_hg_weights": _weight("original/HG.pt"),
        "label": "fp16",
        "gui_key": "FP16",
    },
    "fp32": {
        "precision": "fp32",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/HR.pt"),
        "trt_model_nohg": _weight("original/HR.pt"),
        "trt_hg_weights": _weight("original/HG.pt"),
        "label": "fp32",
        "gui_key": "FP32",
    },
    "int8-mixed-ptq": {
        "precision": "int8-mixed",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_ptq.pt"),
        "predequantize": "on",
        "label": "int8_mixed_ptq_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_mixed_ptq_trt_native",
        "gui_key": "INT8 Mixed (PTQ)",
    },
    "int8-full-ptq": {
        "precision": "int8-full",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_ptq.pt"),
        "predequantize": "on",
        "label": "int8_full_ptq_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_full_ptq_trt_native",
        "gui_key": "INT8 Full (PTQ)",
    },
    "int8-mixed-qat": {
        "precision": "int8-mixed",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_qat.pt"),
        "predequantize": "on",
        "label": "int8_mixed_qat_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_mixed_qat_trt_native",
        "gui_key": "INT8 Mixed (QAT)",
    },
    "int8-full-qat": {
        "precision": "int8-full",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full_qat.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_qat.pt"),
        "predequantize": "on",
        "label": "int8_full_qat_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_full_qat_trt_native",
        "gui_key": "INT8 Full (QAT)",
    },
    "int8-mixed-qat-film": {
        "precision": "int8-mixed",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat_film.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat_film.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_mixed_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_mixed_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_mixed_qat_film.pt"),
        "predequantize": "on",
        "label": "int8_mixed_qat_film_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_mixed_qat_film_trt_native",
        "gui_key": "INT8 Mixed (QAT) (Film)",
    },
    "int8-full-qat-film": {
        "precision": "int8-full",
        "model": _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat_film.pt"),
        "model_nohg": _weight("original/pytorch_int8/hr/HR_original_int8_full_qat_film.pt"),
        "trt_model": _weight("original/tensorrt/hr_hg/HR_HG_original_int8_full_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt/hr/HR_original_int8_full_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt/hg/HG_original_int8_full_qat_film.pt"),
        "predequantize": "on",
        "label": "int8_full_qat_film_predeq",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "int8_full_qat_film_trt_native",
        "gui_key": "INT8 Full (QAT) (Film)",
    },
    "fp8-mixed-ptq": {
        "precision": "fp8-mixed",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_mixed_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_mixed_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_mixed_ptq.pt"),
        "predequantize": "off",
        "label": "fp8_mixed_ptq_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_mixed_ptq_trt_native",
        "gui_key": "FP8 Mixed (PTQ)",
    },
    "fp8-mixed-qat": {
        "precision": "fp8-mixed",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_mixed_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_mixed_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_mixed_qat.pt"),
        "predequantize": "off",
        "label": "fp8_mixed_qat_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_mixed_qat_trt_native",
        "gui_key": "FP8 Mixed (QAT)",
    },
    "fp8-mixed-qat-film": {
        "precision": "fp8-mixed",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_mixed_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_mixed_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_mixed_qat_film.pt"),
        "predequantize": "off",
        "label": "fp8_mixed_qat_film_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_mixed_qat_film_trt_native",
        "gui_key": "FP8 Mixed (QAT) (Film)",
    },
    "fp8-full-ptq": {
        "precision": "fp8-full",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_full_ptq.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_full_ptq.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_full_ptq.pt"),
        "predequantize": "off",
        "label": "fp8_full_ptq_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_full_ptq_trt_native",
        "gui_key": "FP8 Full (PTQ)",
    },
    "fp8-full-qat": {
        "precision": "fp8-full",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_full_qat.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_full_qat.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_full_qat.pt"),
        "predequantize": "off",
        "label": "fp8_full_qat_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_full_qat_trt_native",
        "gui_key": "FP8 Full (QAT)",
    },
    "fp8-full-qat-film": {
        "precision": "fp8-full",
        "model": _weight("original/HR.pt"),
        "model_nohg": _weight("original/HR.pt"),
        "hg_weights": _weight("original/HG.pt"),
        "trt_model": _weight("original/tensorrt_fp8/hr_hg/HR_HG_original_fp8_full_qat_film.pt"),
        "trt_model_nohg": _weight("original/tensorrt_fp8/hr/HR_original_fp8_full_qat_film.pt"),
        "trt_hg_weights": _weight("original/tensorrt_fp8/hg/HG_original_fp8_full_qat_film.pt"),
        "predequantize": "off",
        "label": "fp8_full_qat_film_trt_native",
        "trt_predequantize": "off",
        "trt_qdq_fusion": "native",
        "trt_label": "fp8_full_qat_film_trt_native",
        "gui_key": "FP8 Full (QAT) (Film)",
    },
}
_DEFAULT_RUNS = [
    "int8-mixed-qat",
]
_REALTIME_CATCHUP_ENABLED = True
_REALTIME_SKIP_LAG_FRAMES = 1.1
_REALTIME_MAX_CATCHUP_SKIP = 6


_CSV_FIELDS = [
    "elapsed_s",
    "logged_at_local",
    "fps",
    "latency_ms",
    "model_latency_ms",
    "live_video_latency_ms",
    "frame",
    "cpu_mb",
    "gpu_mb",
    "model_mb",
    "model_size_label",
    "precision",
    "proc_res",
    "psnr_db",
    "sssim",
    "delta_e_itp",
    "hdr_vdp3",
    "objective_enabled",
    "objective_note",
    "hdr_vdp3_note",
    "is_live_capture",
    "decode_ms",
    "resize_ms",
    "infer_ms",
    "pre_ms",
    "run_ms",
    "post_ms",
    "render_ms",
    "fps_1p_low",
    "dropped_frames",
    "catchup_dropped_frames",
    "fps_limiter_dropped_frames",
    "source_loops",
    "playback_mode",
]


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{text}', expected WxH."
        ) from exc


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = value.strip("._-")
    return (value or "session")[:32]


def _stats(samples: list[dict], key: str) -> dict | None:
    vals = []
    for sample in samples:
        try:
            value = float(sample.get(key))
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    if not vals:
        return None
    return {
        "count": len(vals),
        "avg": sum(vals) / float(len(vals)),
        "min": min(vals),
        "max": max(vals),
        "last": vals[-1],
    }


def _fmt_stats(label: str, stats: dict | None, suffix: str = "") -> str:
    if not stats:
        return f"{label}: n/a"
    return (
        f"{label}: avg {stats['avg']:.3f}{suffix}, "
        f"min {stats['min']:.3f}{suffix}, "
        f"max {stats['max']:.3f}{suffix}, "
        f"last {stats['last']:.3f}{suffix}"
    )


def _tensorrt_device_memory_mb(processor) -> float | None:
    if not isinstance(processor, HDRTVNetTensorRT):
        return None

    def _read_size(obj) -> float | None:
        if obj is None:
            return None
        for attr in ("device_memory_size_v2", "device_memory_size"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    value = getattr(obj, attr, None)
                    size = value() if callable(value) else value
            except Exception:
                continue
            if isinstance(size, (int, float)) and size > 0:
                return float(size)
        if hasattr(obj, "get_device_memory_size"):
            try:
                size = obj.get_device_memory_size()
            except Exception:
                return None
            if isinstance(size, (int, float)) and size > 0:
                return float(size)
        return None

    size_bytes = _read_size(getattr(processor, "_trt_context", None))
    if size_bytes is None:
        size_bytes = _read_size(getattr(processor, "_trt_engine", None))
    if size_bytes is None:
        return None
    return size_bytes / (1024.0 * 1024.0)


def _resize_frame(frame, width: int, height: int):
    if frame.shape[1] == int(width) and frame.shape[0] == int(height):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _compiled_marker_predequantize_mode(precision: str, selected_mode: str, processor) -> str:
    if not str(precision or "").strip().lower().startswith("int8"):
        return "auto"
    mode = str(selected_mode or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        mode = "auto"
    if mode != "auto":
        return mode
    try:
        is_w8_model = bool(getattr(processor, "_is_w8_model"))
    except Exception:
        return "auto"
    return "off" if is_w8_model else "on"


def _mark_cache(processor, width: int, height: int, args, run: dict) -> None:
    if _IS_NVIDIA or not bool(getattr(processor, "_compiled", False)):
        return
    precision = str(run["precision"])
    try:
        _mark_compiled(
            int(width),
            int(height),
            precision,
            model_path=str(run["model"]),
            use_hg=bool(args.use_hg),
            predequantize_mode=_compiled_marker_predequantize_mode(
                precision,
                str(run.get("predequantize", "auto")),
                processor,
            ),
            compile_mode=str(
                getattr(processor, "_compile_mode", None)
                or ("max-autotune" if str(args.compile_mode) == "auto" else args.compile_mode)
            ),
            memory_format=str(getattr(processor, "_memory_format_name", None) or "contiguous"),
        )
    except Exception as exc:
        print(f"WARNING: could not write compile marker: {exc}", flush=True)


def _runtime_artifact_size_mb(processor, run: dict) -> float:
    if isinstance(processor, HDRTVNetTensorRT):
        engine_path = str(getattr(processor, "engine_path", "") or "")
        if engine_path and os.path.isfile(engine_path):
            return os.path.getsize(engine_path) / (1024 * 1024)
    return os.path.getsize(run["model"]) / (1024 * 1024)


def _selected_tensorrt_source_paths_for_cli(gui_key: str, use_hg: bool) -> list[str]:
    if not _IS_NVIDIA or not gui_key or not _precision_is_quantized_tensorrt(gui_key):
        return []
    paths = [_select_tensorrt_model_path(gui_key, use_hg)]
    if use_hg:
        paths.append(_select_hg_weights_path(gui_key, tensorrt=True))
    return [str(path) for path in paths if str(path or "").strip()]


def _ensure_tensorrt_sources_for_cli(gui_key: str, use_hg: bool) -> None:
    missing = [
        path
        for path in _selected_tensorrt_source_paths_for_cli(gui_key, use_hg)
        if not os.path.isfile(path)
    ]
    if not missing:
        return
    cfg = PRECISIONS.get(str(gui_key), {})
    precision = str(cfg.get("precision", "")).strip().lower()
    if precision.startswith("fp8"):
        script = _ROOT / "scripts" / "quantize" / "make_tensorrt_fp8_source_checkpoints.py"
        cmd = [sys.executable, "-u", str(script), "--missing-only", "--family", "hr"]
        if use_hg:
            cmd.extend(["--family", "hr_hg", "--family", "hg"])
    else:
        script = _ROOT / "scripts" / "quantize" / "split_tensorrt_sources.py"
        cmd = [sys.executable, "-u", str(script), "--only-original", "--missing-only"]
    if not script.is_file():
        raise FileNotFoundError(f"TensorRT source generator not found: {script}")
    print(
        "[bench] Preparing missing TensorRT source checkpoint(s): "
        f"{len(missing)}",
        flush=True,
    )
    subprocess.run(cmd, cwd=str(_ROOT), check=True)


def _make_processor(args, run: dict, width: int, height: int):
    predeq_text = str(run.get("predequantize", "auto"))
    predeq = {"auto": "auto", "on": True, "off": False}[predeq_text]
    gui_key = str(run.get("gui_key") or "").strip()
    manual_model = bool(run.get("custom_model", False) or run.get("manual_model", False))
    if _IS_NVIDIA and str(args.device).lower() != "cpu":
        if (
            manual_model
            and not bool(run.get("custom_model", False))
            and run.get("trt_model")
        ):
            run["model"] = str(
                run.get("trt_model_nohg")
                if (not bool(args.use_hg)) and run.get("trt_model_nohg")
                else run.get("trt_model")
            )
        if (not manual_model) and gui_key:
            run["model"] = _select_tensorrt_model_path(
                gui_key,
                bool(args.use_hg),
            )
        trt_hg_weights = None
        if bool(args.use_hg):
            if getattr(args, "hg_weights", None):
                trt_hg_weights = str(args.hg_weights).strip()
            elif run.get("trt_hg_weights"):
                trt_hg_weights = str(run.get("trt_hg_weights")).strip()
            elif run.get("hg_weights"):
                trt_hg_weights = str(run.get("hg_weights")).strip()
            elif gui_key:
                trt_hg_weights = _select_hg_weights_path(gui_key, tensorrt=True)
        if trt_hg_weights and not os.path.isfile(trt_hg_weights):
            trt_hg_weights = None
        mode_base = str(
            run.get("mode_name_base")
            or (_precision_engine_mode_base(gui_key) if gui_key else "")
            or run["precision"]
        )
        mode_name = f"{mode_base}_{'hg' if args.use_hg else 'nohg'}"
        calibration_cache = args.trt_calibration_cache
        if (
            not calibration_cache
            and not args.trt_calibration_dataset
            and str(args.trt_qdq_fusion) == "native"
        ):
            calibration_cache = tensorrt_prebuilt_calibration_cache_path(
                run["model"],
                int(width),
                int(height),
                run["precision"],
                mode_name,
                use_hg=bool(args.use_hg),
                predequantize=predeq,
                qdq_fusion=args.trt_qdq_fusion,
                require_exists=True,
            )
        calibration_video = (
            None
            if (
                args.trt_calibration_dataset
                or args.trt_calibration_cache
                or calibration_cache
            )
            else args.video
        )
        return HDRTVNetTensorRT(
            run["model"],
            device=args.device,
            precision=run["precision"],
            engine_width=int(width),
            engine_height=int(height),
            mode_name=mode_name,
            hg_weights=trt_hg_weights,
            use_hg=bool(args.use_hg),
            predequantize=predeq,
            qdq_fusion=args.trt_qdq_fusion,
            calibration_dataset=args.trt_calibration_dataset,
            calibration_video=calibration_video,
            calibration_frames=args.trt_calibration_frames,
            calibration_cache=calibration_cache,
        )
    hg_weights = None
    if bool(args.use_hg):
        if getattr(args, "hg_weights", None):
            hg_weights = str(args.hg_weights).strip()
        elif run.get("hg_weights"):
            hg_weights = str(run.get("hg_weights")).strip()
        elif gui_key:
            hg_weights = _select_hg_weights_path(gui_key)
    if hg_weights and not os.path.isfile(hg_weights):
        hg_weights = None
    return HDRTVNetTorch(
        run["model"],
        device=args.device,
        precision=run["precision"],
        compile_model=not args.no_compile,
        force_compile=args.force_compile,
        compile_mode=args.compile_mode,
        use_cuda_graphs=False,
        force_channels_last=False,
        predequantize=predeq,
        hg_weights=hg_weights,
        use_hg=bool(args.use_hg),
        warmup_passes=0,
    )


def _write_runtime_csv(path: pathlib.Path, samples: list[dict]) -> None:
    extras = []
    seen = set(_CSV_FIELDS)
    for sample in samples:
        for key in sample:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fields = _CSV_FIELDS + sorted(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample.get(key) for key in fields})


def _write_session_files(
    *,
    session_dir: pathlib.Path,
    reason: str,
    source_label: str,
    settings: dict,
    runtime_samples: list[dict],
    worker_summary: dict,
    started_at: str,
    ended_at: str,
) -> dict:
    runtime_summary = {
        key: _stats(runtime_samples, key)
        for key in (
            "fps",
            "latency_ms",
            "model_latency_ms",
            "live_video_latency_ms",
            "gpu_mb",
            "cpu_mb",
            "model_mb",
            "decode_ms",
            "resize_ms",
            "infer_ms",
            "pre_ms",
            "run_ms",
            "post_ms",
            "render_ms",
            "fps_1p_low",
            "dropped_frames",
            "catchup_dropped_frames",
            "fps_limiter_dropped_frames",
            "source_loops",
            "psnr_db",
            "sssim",
            "delta_e_itp",
            "hdr_vdp3",
        )
    }
    for sample in reversed(runtime_samples):
        label = str(sample.get("model_size_label") or "").strip()
        if label:
            runtime_summary["model_size_label"] = label
            break

    lines = [
        "HDRTVNet++ Playback Log",
        f"Reason: {reason}",
        f"Saved To: {session_dir}",
        f"Started: {started_at or 'n/a'}",
        f"Ended: {ended_at or 'n/a'}",
        f"Source: {source_label or 'n/a'}",
        f"Source Mode: {settings.get('source_mode') or 'cli_headless'}",
        "",
        "Settings:",
        f"  Precision: {settings.get('precision') or 'n/a'}",
        f"  Resolution: {settings.get('resolution') or 'n/a'}",
        f"  Upscale: {settings.get('upscale_mode') or 'n/a'}",
        f"  Use HG: {settings.get('use_hg')}",
        f"  Film Grain: {settings.get('film_grain')}",
        f"  Runtime Mode: {settings.get('runtime_execution_mode') or 'n/a'}",
        f"  Predequantize: {settings.get('predequantize_mode') or 'n/a'}",
        f"  HDR GT: {settings.get('hdr_ground_truth_path') or 'none'}",
        "",
        "Runtime Metrics:",
        f"  Samples Saved: {len(runtime_samples)}",
        "  " + _fmt_stats("FPS", runtime_summary.get("fps"), ""),
        "  " + _fmt_stats("1% Low FPS", runtime_summary.get("fps_1p_low"), ""),
        "  " + _fmt_stats("Latency", runtime_summary.get("latency_ms"), " ms"),
        "  "
        + _fmt_stats(
            "Inference Latency (sampled UI)",
            runtime_summary.get("model_latency_ms"),
            " ms",
        ),
        "  " + _fmt_stats("Render/Display", runtime_summary.get("render_ms"), " ms"),
        "  " + _fmt_stats("GPU Memory", runtime_summary.get("gpu_mb"), " MB"),
        "  " + _fmt_stats("CPU Memory", runtime_summary.get("cpu_mb"), " MB"),
    ]
    if runtime_summary.get("model_mb"):
        size_label = str(runtime_summary.get("model_size_label") or "Checkpoint").strip()
        lines.append(
            "  "
            + _fmt_stats(
                f"{size_label} Size",
                runtime_summary.get("model_mb"),
                " MB",
            )
        )
    exact_avg = worker_summary.get("avg_model_latency_ms")
    exact_count = int(worker_summary.get("model_latency_samples", 0) or 0)
    if exact_avg is None:
        lines.append("  Exact Inference Average: n/a")
    else:
        lines.append(
            "  Exact Inference Average: "
            f"{float(exact_avg):.3f} ms over {exact_count} frames"
        )
    lines.extend(["", "Compare Events: 0", "  None", ""])

    session_payload = {
        "reason": reason,
        "saved_at_local": ended_at,
        "started_at_local": started_at or None,
        "session_elapsed_s": worker_summary.get("session_elapsed_s", 0.0),
        "source_label": source_label or None,
        "settings": settings,
        "worker_summary": worker_summary,
        "runtime_metric_summary": runtime_summary,
        "runtime_metrics": runtime_samples,
        "compare_events": [],
        "files": {
            "summary_txt": "summary.txt",
            "session_json": "session.json",
            "runtime_metrics_csv": "runtime_metrics.csv",
            "compare_events_csv": None,
        },
    }

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    with (session_dir / "session.json").open("w", encoding="utf-8") as handle:
        json.dump(session_payload, handle, indent=2)
    _write_runtime_csv(session_dir / "runtime_metrics.csv", runtime_samples)
    return {
        "session_dir": str(session_dir),
        "summary_txt": str(session_dir / "summary.txt"),
        "session_json": str(session_dir / "session.json"),
        "runtime_metrics_csv": str(session_dir / "runtime_metrics.csv"),
        "avg_model_latency_ms": exact_avg,
        "avg_fps": (runtime_summary.get("fps") or {}).get("avg"),
        "avg_fps_1p_low": (runtime_summary.get("fps_1p_low") or {}).get("avg"),
        "avg_latency_ms": (runtime_summary.get("latency_ms") or {}).get("avg"),
        "avg_render_ms": (runtime_summary.get("render_ms") or {}).get("avg"),
        "max_gpu_mb": (runtime_summary.get("gpu_mb") or {}).get("max"),
        "max_cpu_mb": (runtime_summary.get("cpu_mb") or {}).get("max"),
        "model_mb": (runtime_summary.get("model_mb") or {}).get("last"),
        "settings": dict(settings),
        "worker_summary": dict(worker_summary),
    }


def _write_batch_summary(batch_dir: pathlib.Path, args, results: list[dict]) -> None:
    fields = [
        "precision",
        "resolution",
        "use_hg",
        "playback_mode",
        "target_fps",
        "avg_fps",
        "avg_fps_1p_low",
        "avg_latency_ms",
        "avg_model_latency_ms",
        "avg_render_ms",
        "max_gpu_mb",
        "max_cpu_mb",
        "model_mb",
        "dropped_frames",
        "catchup_dropped_frames",
        "fps_limiter_dropped_frames",
        "source_loops",
        "timed_frames",
        "processed_frames",
        "session_elapsed_s",
        "timed_elapsed_s",
        "session_dir",
        "summary_txt",
        "session_json",
        "runtime_metrics_csv",
    ]

    rows: list[dict] = []
    for result in results:
        settings = result.get("settings") if isinstance(result.get("settings"), dict) else {}
        worker = (
            result.get("worker_summary")
            if isinstance(result.get("worker_summary"), dict)
            else {}
        )
        rows.append(
            {
                "precision": settings.get("precision"),
                "resolution": settings.get("resolution"),
                "use_hg": settings.get("use_hg"),
                "playback_mode": settings.get("playback_mode"),
                "target_fps": settings.get("target_fps"),
                "avg_fps": result.get("avg_fps"),
                "avg_fps_1p_low": result.get("avg_fps_1p_low"),
                "avg_latency_ms": result.get("avg_latency_ms"),
                "avg_model_latency_ms": result.get("avg_model_latency_ms"),
                "avg_render_ms": result.get("avg_render_ms"),
                "max_gpu_mb": result.get("max_gpu_mb"),
                "max_cpu_mb": result.get("max_cpu_mb"),
                "model_mb": result.get("model_mb"),
                "dropped_frames": worker.get("dropped_frames"),
                "catchup_dropped_frames": worker.get("catchup_dropped_frames"),
                "fps_limiter_dropped_frames": worker.get("fps_limiter_dropped_frames"),
                "source_loops": worker.get("source_loops"),
                "timed_frames": worker.get("timed_frames"),
                "processed_frames": worker.get("processed_frames"),
                "session_elapsed_s": worker.get("session_elapsed_s"),
                "timed_elapsed_s": worker.get("timed_elapsed_s"),
                "session_dir": result.get("session_dir"),
                "summary_txt": result.get("summary_txt"),
                "session_json": result.get("session_json"),
                "runtime_metrics_csv": result.get("runtime_metrics_csv"),
            }
        )

    csv_paths = [
        batch_dir / "playback_benchmark_summary.csv",
        batch_dir / "batch_summary.csv",
    ]
    for csv_path in csv_paths:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    payload = {
        "benchmark": "playback_performance",
        "video": args.video,
        "display_backend": str(args.display_backend) if bool(args.display) else None,
        "wall_clock": bool(getattr(args, "wall_clock", False)),
        "playback_mode": str(getattr(args, "playback_mode", "throughput")),
        "loop_source": bool(getattr(args, "loop_source", False)),
        "duration_s": float(args.duration_s),
        "warmup_frames": int(args.warmup_frames),
        "sample_interval": int(args.sample_interval),
        "out_root": str(args.out_root),
        "batch_dir": str(batch_dir),
        "results": rows,
        "run_artifacts": results,
    }
    for json_path in (
        batch_dir / "playback_benchmark_summary.json",
        batch_dir / "batch_summary.json",
    ):
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    lines = [
        "HDRTVNet++ Playback Performance Benchmark",
        f"Video: {args.video}",
        f"Display: {str(args.display_backend) if bool(args.display) else 'headless'}",
        f"Wall Clock: {bool(getattr(args, 'wall_clock', False))}",
        f"Playback Mode: {str(getattr(args, 'playback_mode', 'throughput'))}",
        f"Duration: {float(args.duration_s):.1f} s",
        f"Batch Folder: {batch_dir}",
        "",
        (
            "Precision | Resolution | HG | FPS | 1% Low | Latency ms | "
            "Model ms | Render ms | VRAM MB | CPU MB | Artifact MB | Frames | Dropped"
        ),
        "-" * 120,
    ]
    for row in rows:
        lines.append(
            f"{row.get('precision') or '-'} | "
            f"{row.get('resolution') or '-'} | "
            f"{'on' if bool(row.get('use_hg')) else 'off'} | "
            f"{_fmt_num(row.get('avg_fps'), 2)} | "
            f"{_fmt_num(row.get('avg_fps_1p_low'), 2)} | "
            f"{_fmt_num(row.get('avg_latency_ms'), 2)} | "
            f"{_fmt_num(row.get('avg_model_latency_ms'), 2)} | "
            f"{_fmt_num(row.get('avg_render_ms'), 2)} | "
            f"{_fmt_num(row.get('max_gpu_mb'), 0)} | "
            f"{_fmt_num(row.get('max_cpu_mb'), 0)} | "
            f"{_fmt_num(row.get('model_mb'), 2)} | "
            f"{row.get('timed_frames') or '-'} | "
            f"{row.get('dropped_frames') or 0}"
        )
    (batch_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_num(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _run_one(args, run: dict, resolution: tuple[int, int], batch_dir: pathlib.Path) -> dict:
    width, height = resolution
    source = VideoSource(args.video, prefetch=args.prefetch)
    fps = float(source.fps or 30.0)
    playback_mode = str(getattr(args, "playback_mode", "throughput") or "throughput").lower()
    realtime_mode = playback_mode == "realtime"
    out_fps = _limited_playback_fps(fps)
    frame_stride = max(1, int(round(fps / out_fps))) if realtime_mode else 1
    frame_interval_s = 1.0 / max(1e-6, fps)
    next_frame_t = 0.0
    wall_clock = bool(getattr(args, "wall_clock", False))
    timed_frames_target = (
        None
        if wall_clock
        else max(1, int(round(float(args.duration_s) * fps)))
    )
    max_frames = (
        None
        if wall_clock
        else int(args.warmup_frames) + int(timed_frames_target)
    )
    source_label = pathlib.Path(args.video).name
    run_label = f"{width}x{height}_{run['label']}_{'hg' if args.use_hg else 'nohg'}"
    session_dir = batch_dir / run_label

    processor = _make_processor(args, run, width, height)
    if bool(getattr(processor, "_compiled", False)) and not args.skip_cache_warmup:
        processor.warmup_compile(int(width), int(height))
        _mark_cache(processor, width, height, args, run)
    next_frame_t = time.perf_counter()

    process = psutil.Process(os.getpid())
    use_cuda = torch.cuda.is_available() and str(args.device).lower() != "cpu"
    model_size_mb = _runtime_artifact_size_mb(processor, run)
    model_size_label = "Engine" if isinstance(processor, HDRTVNetTensorRT) else "Checkpoint"
    trt_device_mb = _tensorrt_device_memory_mb(processor)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_t = time.perf_counter()

    frame_idx = 0
    stats_frames = 0
    decode_ms = resize_ms = infer_ms = render_ms = pre_ms = run_ms = post_ms = 0.0
    frame_ms_sum = 0.0
    fps_samples = deque(maxlen=10000)
    model_latency_values: list[float] = []
    runtime_samples: list[dict] = []
    display = None
    timed_started_t: float | None = None
    timed_deadline_t: float | None = None
    last_presentation_stamp: float | None = None
    source_loops = 0
    dropped_frames = 0
    catchup_dropped_frames = 0
    fps_limiter_dropped_frames = 0

    def _read_source_frame():
        nonlocal source, source_loops
        while True:
            ok_read, frame_read = source.read()
            if ok_read:
                return True, frame_read
            if not bool(getattr(args, "loop_source", False)):
                return False, None
            try:
                source.release()
            except Exception:
                pass
            source = VideoSource(args.video, prefetch=args.prefetch)
            source_loops += 1

    try:
        if args.display:
            display = CliDisplaySink(
                enabled=True,
                backend=args.display_backend,
                width=width,
                height=height,
                fps=fps,
                window_name="HDRTVNet++ CLI Benchmark",
                # Match the GUI HDR mpv raw-video display path while leaving
                # the user's upscale shader out of the benchmark. The widget
                # still applies normal HDR metadata, dither/deband, and the
                # automatic downscaler path when mpv needs to downscale.
                scale_kernel="bicubic",
                scale_antiring=0.0,
                force_hdr_metadata=True,
                vsync_timed=False,
                embed_wid=str(getattr(args, "display_wid", "") or "").strip() or None,
            )
        while True:
            if max_frames is not None and frame_idx >= int(max_frames):
                break
            if (
                wall_clock
                and timed_deadline_t is not None
                and time.perf_counter() >= timed_deadline_t
            ):
                break
            if (
                wall_clock
                and frame_idx >= int(args.warmup_frames)
                and timed_started_t is None
            ):
                timed_started_t = time.perf_counter()
                timed_deadline_t = timed_started_t + max(0.001, float(args.duration_s))
                next_frame_t = timed_started_t
                last_presentation_stamp = None

            lag_s = 0.0
            if realtime_mode:
                now = time.perf_counter()
                if now < next_frame_t:
                    sleep_until(next_frame_t)
                    now = time.perf_counter()
                else:
                    lag_s = now - next_frame_t

            t0 = time.perf_counter()
            ok, frame = _read_source_frame()
            t1 = time.perf_counter()
            if not ok:
                break
            frame_idx += 1

            if (
                realtime_mode
                and _REALTIME_CATCHUP_ENABLED
                and lag_s > (frame_interval_s * _REALTIME_SKIP_LAG_FRAMES)
            ):
                skip_n = min(
                    _REALTIME_MAX_CATCHUP_SKIP,
                    max(0, int(lag_s / frame_interval_s)),
                )
                while skip_n > 0:
                    ok_skip, frame_skip = _read_source_frame()
                    if not ok_skip:
                        ok = False
                        break
                    frame = frame_skip
                    frame_idx += 1
                    dropped_frames += 1
                    catchup_dropped_frames += 1
                    next_frame_t += frame_interval_s
                    skip_n -= 1
                if not ok:
                    break

            if realtime_mode and frame_stride > 1 and (frame_idx % frame_stride) != 0:
                dropped_frames += 1
                fps_limiter_dropped_frames += 1
                next_frame_t += frame_interval_s
                continue

            present_t = (
                max(next_frame_t, time.perf_counter())
                if realtime_mode
                else None
            )
            frame = _resize_frame(frame, width, height)
            t2 = time.perf_counter()
            if bool(getattr(args, "model_stage_timing", False)):
                out, pre_t, run_t, post_t = processor.process_timed(frame)
            else:
                out = processor.process(frame)
                pre_t = 0.0
                run_t = 0.0
                post_t = 0.0
            t3 = time.perf_counter()
            if display is not None:
                if not display.show(out):
                    break
            t4 = time.perf_counter()
            if realtime_mode:
                presentation_stamp = max(t4, float(present_t or t4))
                if (
                    frame_idx > int(args.warmup_frames)
                    and last_presentation_stamp is not None
                    and presentation_stamp > last_presentation_stamp
                ):
                    fps_samples.append(1.0 / (presentation_stamp - last_presentation_stamp))
                last_presentation_stamp = presentation_stamp
                next_frame_t += frame_interval_s

            if frame_idx <= int(args.warmup_frames):
                continue

            stats_frames += 1
            decode_ms += (t1 - t0) * 1000.0
            resize_ms += (t2 - t1) * 1000.0
            infer_elapsed_ms = (t3 - t2) * 1000.0
            infer_ms += infer_elapsed_ms
            render_ms += (t4 - t3) * 1000.0
            pre_ms += float(pre_t)
            run_ms += (
                float(run_t)
                if bool(getattr(args, "model_stage_timing", False))
                else infer_elapsed_ms
            )
            post_ms += float(post_t)
            frame_ms = (
                ((t4 - t1) if realtime_mode else (t4 - t0)) * 1000.0
            )
            frame_ms_sum += frame_ms
            if frame_ms > 0 and not realtime_mode:
                fps_samples.append(1000.0 / frame_ms)
            model_latency = (
                float(run_t)
                if bool(getattr(args, "model_stage_timing", False))
                and float(run_t) > 0.0
                else infer_elapsed_ms
            )
            model_latency_values.append(model_latency)

            sample_due = stats_frames % int(args.sample_interval) == 0
            if wall_clock and timed_deadline_t is not None:
                sample_due = sample_due or time.perf_counter() >= timed_deadline_t
            elif timed_frames_target is not None:
                sample_due = sample_due or stats_frames == int(timed_frames_target)
            if sample_due:
                avg_frame_ms = frame_ms_sum / max(1, stats_frames)
                if realtime_mode and timed_started_t is not None:
                    timed_elapsed = max(1e-9, time.perf_counter() - timed_started_t)
                    avg_fps = stats_frames / timed_elapsed
                else:
                    avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
                sorted_fps = sorted(fps_samples)
                if sorted_fps:
                    k = max(1, int(len(sorted_fps) * 0.01))
                    one_percent_low = sum(sorted_fps[:k]) / k
                else:
                    one_percent_low = avg_fps
                gpu_mb = 0.0
                if use_cuda:
                    if trt_device_mb is not None:
                        gpu_mb = trt_device_mb
                    else:
                        try:
                            gpu_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                        except Exception:
                            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                sample = {
                    "elapsed_s": round(max(0.0, time.perf_counter() - started_t), 3),
                    "logged_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "fps": float(avg_fps),
                    "latency_ms": float(avg_frame_ms),
                    "model_latency_ms": float(sum(model_latency_values) / len(model_latency_values)),
                    "live_video_latency_ms": 0.0,
                    "is_live_capture": False,
                    "frame": int(frame_idx),
                    "cpu_mb": process.memory_info().rss / (1024 * 1024),
                    "gpu_mb": float(gpu_mb),
                    "model_mb": float(model_size_mb),
                    "model_size_label": model_size_label,
                    "precision": str(run["label"]),
                    "proc_res": f"{width}x{height}",
                    "psnr_db": None,
                    "sssim": None,
                    "delta_e_itp": None,
                    "hdr_vdp3": None,
                    "objective_enabled": False,
                    "objective_note": "",
                    "hdr_vdp3_note": "",
                    "decode_ms": decode_ms / max(1, stats_frames),
                    "resize_ms": resize_ms / max(1, stats_frames),
                    "infer_ms": infer_ms / max(1, stats_frames),
                    "pre_ms": pre_ms / max(1, stats_frames),
                    "run_ms": run_ms / max(1, stats_frames),
                    "post_ms": post_ms / max(1, stats_frames),
                    "render_ms": render_ms / max(1, stats_frames),
                    "fps_1p_low": float(one_percent_low),
                    "dropped_frames": int(dropped_frames),
                    "catchup_dropped_frames": int(catchup_dropped_frames),
                    "fps_limiter_dropped_frames": int(fps_limiter_dropped_frames),
                    "source_loops": int(source_loops),
                    "playback_mode": str(playback_mode),
                }
                runtime_samples.append(sample)
                print(
                    f"[bench] {run_label}: frames={stats_frames} "
                    f"fps={avg_fps:.2f} model={sample['model_latency_ms']:.3f}ms "
                    f"dropped={dropped_frames}",
                    flush=True,
                )
            if (
                wall_clock
                and timed_deadline_t is not None
                and time.perf_counter() >= timed_deadline_t
            ):
                break
            if timed_frames_target is not None and stats_frames >= int(timed_frames_target):
                break
    finally:
        try:
            source.release()
        except Exception:
            pass
        if display is not None:
            display.close()
        del processor
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed_s = round(max(0.0, time.perf_counter() - started_t), 3)
    avg_model_latency = (
        sum(model_latency_values) / len(model_latency_values)
        if model_latency_values
        else None
    )
    source_mode = (
        f"cli_display_{args.display_backend}"
        if bool(args.display)
        else "cli_headless"
    )
    settings = {
        "source_mode": source_mode,
        "precision": str(run["label"]),
        "resolution": f"{width}x{height}",
        "upscale_mode": "none",
        "use_hg": bool(args.use_hg),
        "film_grain": False,
        "runtime_execution_mode": "compiled" if not args.no_compile else "eager",
        "predequantize_mode": str(run.get("predequantize", "auto")),
        "objective_metrics_enabled": False,
        "hdr_ground_truth_path": None,
        "model_path": str(run["model"]),
        "duration_s": float(args.duration_s),
        "wall_clock": bool(wall_clock),
        "playback_mode": str(playback_mode),
        "target_fps": float(out_fps if realtime_mode else fps),
        "loop_source": bool(getattr(args, "loop_source", False)),
        "warmup_frames": int(args.warmup_frames),
        "display_backend": str(args.display_backend) if bool(args.display) else None,
        "model_stage_timing": bool(getattr(args, "model_stage_timing", False)),
        "full_int8_fp16_islands": str(
            os.environ.get("HDRTVNET_TRT_FULL_INT8_FP16_ISLANDS", "0")
        ).strip().lower()
        in {"1", "true", "yes", "on"},
    }
    worker_summary = {
        "avg_model_latency_ms": avg_model_latency,
        "model_latency_samples": len(model_latency_values),
        "session_elapsed_s": elapsed_s,
        "timed_elapsed_s": (
            round(max(0.0, time.perf_counter() - timed_started_t), 3)
            if timed_started_t is not None
            else None
        ),
        "processed_frames": int(frame_idx),
        "timed_frames": int(stats_frames),
        "source_loops": int(source_loops),
        "dropped_frames": int(dropped_frames),
        "catchup_dropped_frames": int(catchup_dropped_frames),
        "fps_limiter_dropped_frames": int(fps_limiter_dropped_frames),
    }
    return _write_session_files(
        session_dir=session_dir,
        reason="cli benchmark finished",
        source_label=source_label,
        settings=settings,
        runtime_samples=runtime_samples,
        worker_summary=worker_summary,
        started_at=started_at,
        ended_at=ended_at,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CLI playback benchmarks and write GUI-style playback logs."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video path to benchmark.",
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Optional model checkpoint override for all selected runs. "
            "Useful for benchmarking experimental TensorRT checkpoints "
            "without changing GUI defaults."
        ),
    )
    parser.add_argument(
        "--hg-weights",
        default="",
        help=(
            "Optional HG checkpoint override for all selected runs. Use this "
            "with --model when benchmarking the original HR/HG baseline."
        ),
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=_parse_resolution,
        default=[(1280, 720)],
        help="Resolutions to benchmark, e.g. 1280x720 1920x1080.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(_DEFAULT_RUNS),
        choices=tuple(_RUN_PRESETS.keys()),
        help="Precision/model presets to run.",
    )
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        help=(
            "Run timed measurement for true wall-clock duration after warmup "
            "instead of duration_s * source_fps frames."
        ),
    )
    parser.add_argument(
        "--loop-source",
        action="store_true",
        help="Loop the input video if EOF is reached before the benchmark finishes.",
    )
    parser.add_argument(
        "--playback-mode",
        default="throughput",
        choices=["throughput", "realtime"],
        help=(
            "throughput runs as fast as possible. realtime mirrors GUI video "
            "playback: source-FPS pacing plus catch-up frame skips when behind."
        ),
    )
    parser.add_argument("--warmup-frames", type=int, default=120)
    parser.add_argument("--sample-interval", type=int, default=120)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--use-hg", default="1", choices=["0", "1"])
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-cache-warmup", action="store_true")
    parser.add_argument(
        "--model-stage-timing",
        action="store_true",
        help=(
            "Use synchronized preprocess/run/post timing. This is diagnostic "
            "and slower; default uses the normal fast playback path."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show processed frames and include display cost in timing.",
    )
    parser.add_argument(
        "--display-backend",
        default="mpv",
        choices=["mpv", "opencv"],
        help="Display backend used with --display. Defaults to mpv, matching the GUI HDR path.",
    )
    parser.add_argument(
        "--display-wid",
        default=None,
        help=(
            "Advanced: native window handle for embedding the mpv display "
            "inside an existing GUI widget."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--trt-qdq-fusion",
        default="native",
        choices=["native", "auto", "none", "add", "add-mul", "elementwise"],
        help=(
            "TensorRT INT8 Q/DQ export mode. 'native' means explicit ModelOpt "
            "Q/DQ export with TensorRT's native Q/DQ fusion. Default: native."
        ),
    )
    parser.add_argument(
        "--full-int8-fp16-islands",
        default=None,
        choices=["on", "off"],
        help=(
            "Full INT8 TensorRT safety mode. Default/env is off: full INT8 "
            "engines disable FP16 tactics. Use 'on' to select the safe "
            "FP16-builder fallback for full INT8 presets."
        ),
    )
    parser.add_argument(
        "--trt-calibration-dataset",
        default=None,
        help=(
            "Legacy TensorRT native-implicit INT8 calibration input. Ignored by "
            "the current ModelOpt/QDQ runtime."
        ),
    )
    parser.add_argument(
        "--trt-calibration-cache",
        default=None,
        help="Legacy TensorRT native-implicit INT8 calibration cache path.",
    )
    parser.add_argument(
        "--trt-calibration-frames",
        type=int,
        default=64,
        help=(
            "Legacy native-implicit calibration frame/image count. Ignored by "
            "the current ModelOpt/QDQ runtime."
        ),
    )
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "logs" / "playback_sessions"),
        help="Root folder for GUI-style playback session logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.use_hg = str(args.use_hg).strip() != "0"
    if args.full_int8_fp16_islands is not None:
        os.environ["HDRTVNET_TRT_FULL_INT8_FP16_ISLANDS"] = (
            "1" if args.full_int8_fp16_islands == "on" else "0"
        )

    source_slug = _slug(pathlib.Path(args.video).stem)
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = pathlib.Path(args.out_root) / f"{batch_stamp}_{source_slug}_cli_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] Video: {args.video}", flush=True)
    print(f"[bench] Output: {batch_dir}", flush=True)
    if args.display:
        print(f"[bench] Display: {args.display_backend}", flush=True)
    results = []
    for resolution in args.resolutions:
        for run_key in args.runs:
            preset = dict(_RUN_PRESETS[run_key])
            if (not args.use_hg) and preset.get("model_nohg"):
                preset["model"] = preset["model_nohg"]
            manual_model = bool(preset.get("manual_model", False))
            if args.model:
                preset["model"] = os.path.abspath(args.model)
                preset["custom_model"] = True
                preset["mode_name_base"] = run_key
                preset["label"] = f"{preset.get('trt_label', preset.get('label', run_key))}_custom"
                preset["predequantize"] = "off" if _IS_NVIDIA else str(
                    preset.get("predequantize", "auto")
                )
                preset["qdq_fusion"] = str(args.trt_qdq_fusion)
            elif _IS_NVIDIA and str(args.device).lower() != "cpu":
                gui_key = str(preset.get("gui_key") or "").strip()
                if gui_key:
                    _ensure_tensorrt_sources_for_cli(gui_key, bool(args.use_hg))
                if (not manual_model) and gui_key:
                    preset["model"] = _select_tensorrt_model_path(
                        gui_key,
                        bool(args.use_hg),
                    )
                preset["predequantize"] = "off"
                preset["qdq_fusion"] = str(args.trt_qdq_fusion)
                preset["label"] = str(preset.get("trt_label", preset.get("label", run_key)))
                if str(args.trt_qdq_fusion) not in {"auto", "native"}:
                    fusion_label = str(args.trt_qdq_fusion).replace("-", "")
                    preset["label"] = f"{preset['label']}_{fusion_label}"
            if not os.path.isfile(preset["model"]):
                raise FileNotFoundError(f"Model not found: {preset['model']}")
            print(
                f"[bench] Starting {resolution[0]}x{resolution[1]} {preset['label']} "
                f"({'HG' if args.use_hg else 'no-HG'})",
                flush=True,
            )
            results.append(_run_one(args, preset, resolution, batch_dir))

    _write_batch_summary(batch_dir, args, results)
    print(f"[bench] Done: {batch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
