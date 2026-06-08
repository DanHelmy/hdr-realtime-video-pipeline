"""
Pre-build TensorRT engines for NVIDIA inference.

Playback/export/benchmark still build engines on demand when a cache entry is
missing. This script is the NVIDIA equivalent of compile_kernels.py for users
who want to populate the engine cache ahead of time.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from windows_runtime import ensure_windows_supported, install_torch_windows_warning_filter

ensure_windows_supported("HDRTVNet++ TensorRT engine builder")
install_torch_windows_warning_filter()

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    cleanup_tensorrt_onnx_after_engine,
    tensorrt_engine_path,
    tensorrt_engine_is_valid,
    tensorrt_engine_metadata_path,
    tensorrt_mode_name,
    tensorrt_onnx_path,
    tensorrt_prebuilt_calibration_cache_path,
)
from gui_config import (
    PRECISIONS,
    _precision_engine_mode_base,
    _select_hg_weights_path,
    _select_model_path,
    _select_tensorrt_model_path,
)


def _weight(name: str) -> str:
    return os.path.join(_HERE, "models", "weights", name)


_PRECISION_MAP = {
    "fp16": (
        "fp16",
        _weight("original/HR.pt"),
        _weight("original/HR.pt"),
    ),
    "fp32": (
        "fp32",
        _weight("original/HR.pt"),
        _weight("original/HR.pt"),
    ),
    "int8-mixed": (
        "int8-mixed",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat.pt"),
    ),
    "int8-mixed-ptq": (
        "int8-mixed",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_mixed.pt"),
    ),
    "int8-mixed-qat": (
        "int8-mixed",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat.pt"),
    ),
    "int8-mixed-qat-film": (
        "int8-mixed",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_mixed_qat_film.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_mixed_qat_film.pt"),
    ),
    "int8-full": (
        "int8-full",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_full.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_full.pt"),
    ),
    "int8-full-ptq": (
        "int8-full",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_full.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_full.pt"),
    ),
    "int8-full-qat": (
        "int8-full",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_full_qat.pt"),
    ),
    "int8-full-qat-film": (
        "int8-full",
        _weight("original/pytorch_int8/hg/HR_HG_original_int8_full_qat_film.pt"),
        _weight("original/pytorch_int8/hr/HR_original_int8_full_qat_film.pt"),
    ),
}


def _gui_precision_key_for_model(precision: str, model_path: str, use_hg: bool) -> str | None:
    target = os.path.abspath(os.path.normcase(model_path))
    for key, cfg in PRECISIONS.items():
        if str(cfg.get("precision", "")).strip().lower() != str(precision).strip().lower():
            continue
        candidate = _select_model_path(key, use_hg)
        if os.path.abspath(os.path.normcase(candidate)) == target:
            return key
    return None


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{text}', expected WxH (e.g. 1920x1080)"
        ) from exc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _ensure_default_tensorrt_sources(gui_precision_key: str | None, use_hg: bool) -> None:
    if not gui_precision_key:
        return
    cfg = PRECISIONS.get(gui_precision_key, {})
    if not str(cfg.get("precision", "")).strip().lower().startswith("int8"):
        return

    expected = [_select_tensorrt_model_path(gui_precision_key, use_hg)]
    if use_hg:
        expected.append(_select_hg_weights_path(gui_precision_key, tensorrt=True))
    missing = [
        path
        for path in expected
        if str(path or "").strip() and not os.path.isfile(path)
    ]
    if not missing:
        return

    script = os.path.abspath(
        os.path.join(
            os.path.dirname(_HERE),
            "scripts",
            "quantize",
            "split_tensorrt_sources.py",
        )
    )
    if not os.path.isfile(script):
        print(
            "[tensorrt] default TensorRT source files are missing, but "
            f"the split script was not found: {script}",
            file=sys.stderr,
        )
        return

    print(
        "[tensorrt] preparing missing default TensorRT source checkpoint(s): "
        f"{len(missing)}",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-u", script, "--missing-only"],
        cwd=os.path.dirname(_HERE),
        check=False,
    )


def _engine_tag_suffix() -> str:
    tag = str(os.environ.get("HDRTVNET_TRT_ENGINE_TAG", "")).strip()
    if not tag:
        return ""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_.-")
    return f"_{tag}" if tag else ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build cached TensorRT engines for selected resolutions."
    )
    parser.add_argument(
        "resolutions",
        nargs="+",
        type=_parse_resolution,
        metavar="WxH",
        help="One or more engine resolutions to build.",
    )
    parser.add_argument(
        "--precision",
        default="int8-mixed-qat",
        choices=tuple(_PRECISION_MAP.keys()),
        help="Model precision/mode to build. Default: int8-mixed-qat.",
    )
    parser.add_argument("--model", default=None, help="Override model path.")
    parser.add_argument(
        "--use-hg",
        default="1",
        choices=["1", "0"],
        help="Enable HG refinement (1/0). Default: 1",
    )
    parser.add_argument(
        "--hg-weights",
        default=None,
        help="Path to original/HG.pt or a TensorRT HG source checkpoint (overrides default path).",
    )
    parser.add_argument(
        "--predequantize",
        default="off",
        choices=["auto", "on", "off"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qdq-fusion",
        default="native",
        choices=["native", "auto", "none", "add", "add-mul", "elementwise"],
        help=(
            "TensorRT INT8 export mode. With ModelOpt enabled, 'native' means "
            "explicit Q/DQ export with TensorRT's native Q/DQ fusion. If "
            "ModelOpt is disabled, it falls back to legacy implicit/native "
            "TensorRT calibration. Default: native. "
            "'add' inserts Q/DQ on eligible Add inputs that already feed "
            "calibrated quantized paths; 'add-mul' also patches Mul inputs."
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
        "--calibration-dataset",
        default=None,
        help=(
            "Directory/image/manifest of SDR input frames for TensorRT native "
            "INT8 calibration. Takes priority over --calibration-video."
        ),
    )
    parser.add_argument(
        "--calibration-video",
        default=None,
        help="Video used for TensorRT native INT8 calibration.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=None,
        help=(
            "Number of frames/images for TensorRT native INT8 calibration. "
            "Default: 64. Use 0 for all dataset/video frames."
        ),
    )
    parser.add_argument(
        "--calibration-cache",
        default=None,
        help="TensorRT native INT8 calibration cache path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the .engine file already exists.",
    )
    parser.add_argument(
        "--force-onnx",
        action="store_true",
        help="Remove any stale .onnx before rebuilding.",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the exported ONNX beside the TensorRT engine for inspection.",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=None,
        choices=range(0, 6),
        metavar="0..5",
        help="TensorRT builder optimization level. Default: project setting/env, normally 5.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=None,
        help="TensorRT builder workspace cap in GiB. Default: no explicit cap.",
    )
    parser.add_argument(
        "--timing-cache",
        default=None,
        help="TensorRT timing cache path, or 'none' to disable.",
    )
    parser.add_argument(
        "--aux-streams",
        type=int,
        default=None,
        help="TensorRT max auxiliary streams during build. Omit for TensorRT default.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=0,
        help="Run N full preprocess+TensorRT+postprocess dummy frames after build/load.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        help="Warmup frames before --benchmark-runs timing. Default: 5.",
    )
    args = parser.parse_args()

    if args.opt_level is not None:
        os.environ["HDRTVNET_TRT_BUILDER_OPT_LEVEL"] = str(args.opt_level)
    if args.workspace_gb is not None:
        os.environ["HDRTVNET_TRT_WORKSPACE_GB"] = str(max(1.0, args.workspace_gb))
    if args.timing_cache is not None:
        os.environ["HDRTVNET_TRT_TIMING_CACHE"] = str(args.timing_cache)
    if args.aux_streams is not None:
        os.environ["HDRTVNET_TRT_AUX_STREAMS"] = str(max(0, args.aux_streams))
    if args.keep_onnx:
        os.environ["HDRTVNET_TRT_KEEP_ONNX"] = "1"
    if args.full_int8_fp16_islands is not None:
        os.environ["HDRTVNET_TRT_FULL_INT8_FP16_ISLANDS"] = (
            "1" if args.full_int8_fp16_islands == "on" else "0"
        )

    use_hg = str(args.use_hg).strip() != "0"
    precision, default_hg_model, default_nohg_model = _PRECISION_MAP[args.precision]
    default_model = default_hg_model if use_hg else default_nohg_model
    gui_precision_key = _gui_precision_key_for_model(precision, default_model, use_hg)
    selected_model = (
        _select_tensorrt_model_path(gui_precision_key, use_hg)
        if gui_precision_key
        else default_model
    )
    if not args.model:
        _ensure_default_tensorrt_sources(gui_precision_key, use_hg)
    model_path = os.path.abspath(args.model or selected_model)
    selected_hg_weights = args.hg_weights
    if use_hg and not selected_hg_weights and gui_precision_key:
        candidate_hg = _select_hg_weights_path(gui_precision_key, tensorrt=True)
        if candidate_hg and os.path.isfile(candidate_hg):
            selected_hg_weights = candidate_hg
    if str(args.predequantize).strip().lower() not in {"", "off"}:
        print(
            "TensorRT pre-dequantize is no longer supported; building the native "
            "TensorRT engine with predequantize=off.",
            flush=True,
        )
    predeq = False
    mode_base = (
        _precision_engine_mode_base(gui_precision_key)
        if gui_precision_key
        else args.precision
    )
    base_mode_name = f"{mode_base}_{'hg' if use_hg else 'nohg'}"
    tagged_base_mode_name = f"{base_mode_name}{_engine_tag_suffix()}"
    mode_name = tensorrt_mode_name(
        precision,
        tagged_base_mode_name,
        predequantize=predeq,
        qdq_fusion=args.qdq_fusion,
    )

    if not os.path.isfile(model_path):
        print(f"ERROR: model weights not found: {model_path}", file=sys.stderr)
        return 2

    for w, h in args.resolutions:
        calibration_cache = args.calibration_cache
        if (
            not calibration_cache
            and not args.calibration_dataset
            and not args.calibration_video
            and str(args.qdq_fusion) == "native"
            and not (
                str(precision).startswith("int8")
                and _env_bool("HDRTVNET_TRT_INT8_MODELOPT", True)
            )
        ):
            calibration_cache = tensorrt_prebuilt_calibration_cache_path(
                model_path,
                w,
                h,
                precision,
                base_mode_name,
                use_hg=use_hg,
                predequantize=predeq,
                qdq_fusion=args.qdq_fusion,
                require_exists=True,
            )
            if calibration_cache:
                print(
                    "[tensorrt] using prebuilt calibration cache: "
                    f"{calibration_cache}"
                )
        engine_path = tensorrt_engine_path(model_path, w, h, mode_name)
        onnx_path = tensorrt_onnx_path(model_path, w, h, mode_name)
        if (args.force or args.force_onnx) and os.path.isfile(engine_path):
            os.remove(engine_path)
            meta_path = tensorrt_engine_metadata_path(engine_path)
            if os.path.isfile(meta_path):
                os.remove(meta_path)
        if args.force_onnx and os.path.isfile(onnx_path):
            os.remove(onnx_path)
        engine_valid = tensorrt_engine_is_valid(
            engine_path,
            model_path=model_path,
            width=w,
            height=h,
            precision=precision,
            mode_name=tagged_base_mode_name,
            use_hg=use_hg,
            predequantize=predeq,
            qdq_fusion=args.qdq_fusion,
            hg_weights=selected_hg_weights,
            calibration_dataset=args.calibration_dataset,
            calibration_video=args.calibration_video,
            calibration_frames=args.calibration_frames,
            calibration_cache=calibration_cache,
            verbose=True,
        )
        if engine_valid:
            print(f"[tensorrt] cache hit: {engine_path}")
            cleanup_tensorrt_onnx_after_engine(onnx_path, engine_path)
            if args.benchmark_runs <= 0:
                continue
        print(
            f"[tensorrt] {'loading' if engine_valid else 'building'} "
            f"{args.precision} {'HG' if use_hg else 'no-HG'} engine for {w}x{h}"
        )
        processor = HDRTVNetTensorRT(
            model_path,
            device="auto",
            precision=precision,
            engine_width=w,
            engine_height=h,
            mode_name=tagged_base_mode_name,
            hg_weights=selected_hg_weights,
            use_hg=use_hg,
            predequantize=predeq,
            qdq_fusion=args.qdq_fusion,
            keep_onnx=args.keep_onnx,
            calibration_dataset=args.calibration_dataset,
            calibration_video=args.calibration_video,
            calibration_frames=args.calibration_frames,
            calibration_cache=calibration_cache,
        )
        if args.benchmark_runs > 0:
            import time
            import numpy as np
            import torch

            runs = max(1, int(args.benchmark_runs))
            warmup = max(0, int(args.benchmark_warmup))
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(warmup):
                processor.process(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                processor.process(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(1e-9, time.perf_counter() - t0)
            print(
                f"[tensorrt] benchmark {w}x{h}: "
                f"{(elapsed * 1000.0 / runs):.2f} ms/frame, {runs / elapsed:.2f} fps "
                f"(full pipeline, warmup={warmup}, runs={runs})"
            )
        del processor
        print(f"[tensorrt] ready: {engine_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
