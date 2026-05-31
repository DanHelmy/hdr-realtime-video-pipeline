"""
Build prepackaged TensorRT native-INT8 calibration caches for GUI presets.

Run this on an NVIDIA machine after setup. The generated .calib files are
written under src/models/tensorrt_calibration/ by default and are picked up
automatically by GUI playback/export/benchmark.
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys


_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from windows_runtime import ensure_windows_supported

ensure_windows_supported("HDRTVNet++ TensorRT calibration cache builder")

from gui_config import MAX_H, MAX_W, PRECISIONS, RESOLUTION_SCALES, _select_model_path
from models.hdrtvnet_torch import (
    HDRTVNetTensorRT,
    tensorrt_engine_metadata_path,
    tensorrt_engine_path,
    tensorrt_mode_name,
    tensorrt_onnx_path,
    tensorrt_prebuilt_calibration_cache_path,
)


def _resolution_dims(key: str) -> tuple[int, int]:
    dims = RESOLUTION_SCALES.get(key)
    if dims is None:
        return int(MAX_W), int(MAX_H)
    return int(dims[0]), int(dims[1])


def _remove_file(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError as exc:
        print(f"[calib] warning: could not remove {path}: {exc}", flush=True)


def _discard_temporary_engine(engine_path: str, onnx_path: str, keep_onnx: bool) -> None:
    _remove_file(engine_path)
    _remove_file(tensorrt_engine_metadata_path(engine_path))
    if not keep_onnx:
        _remove_file(onnx_path)
        _remove_file(f"{onnx_path}.data")


def _int8_precision_keys() -> list[str]:
    return [
        key for key, cfg in PRECISIONS.items()
        if str(cfg.get("precision") or "").startswith("int8")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TensorRT native-INT8 .calib files for GUI presets."
    )
    parser.add_argument(
        "--calibration-dataset",
        default=str(_ROOT / "dataset" / "train_sdr"),
        help="SDR image directory/manifest used to build calibration caches.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=256,
        help="Images sampled from the calibration dataset. Default: 256. Use 0 for all.",
    )
    parser.add_argument(
        "--precision-key",
        action="append",
        choices=_int8_precision_keys(),
        help="GUI precision key to build. Repeat to limit the matrix.",
    )
    parser.add_argument(
        "--resolution",
        action="append",
        choices=tuple(RESOLUTION_SCALES.keys()),
        help="GUI resolution preset to build. Repeat to limit the matrix.",
    )
    parser.add_argument(
        "--hg",
        default="all",
        choices=["all", "on", "off"],
        help="HG variants to build. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing .calib/.engine files before rebuilding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cache filenames that would be generated.",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep temporary ONNX files beside the engine cache for inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("HDRTVNET_TRT_WORKSPACE_GB"):
        os.environ["HDRTVNET_TRT_WORKSPACE_GB"] = "2"
        print("[calib] TensorRT calibration workspace default: 2 GiB", flush=True)
    else:
        print(
            "[calib] TensorRT calibration workspace: "
            f"{os.environ['HDRTVNET_TRT_WORKSPACE_GB']} GiB",
            flush=True,
        )
    dataset = os.path.abspath(os.path.expanduser(args.calibration_dataset))
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Calibration dataset not found: {dataset}")

    precision_keys = args.precision_key or _int8_precision_keys()
    resolutions = args.resolution or list(RESOLUTION_SCALES.keys())
    if args.hg == "all":
        hg_states = [False, True]
    else:
        hg_states = [args.hg == "on"]

    total = len(precision_keys) * len(resolutions) * len(hg_states)
    print(f"[calib] Dataset: {dataset}", flush=True)
    print(f"[calib] Matrix: {total} cache file(s)", flush=True)

    failures = 0
    built = 0
    skipped = 0
    for precision_key in precision_keys:
        cfg = PRECISIONS[precision_key]
        model_precision = str(cfg.get("precision") or "")
        for use_hg in hg_states:
            model_path = os.path.abspath(_select_model_path(precision_key, use_hg))
            if not os.path.isfile(model_path):
                print(f"[calib] missing model: {model_path}", flush=True)
                failures += 1
                continue
            mode_name = f"{precision_key}_{'hg' if use_hg else 'nohg'}"
            for res_key in resolutions:
                width, height = _resolution_dims(res_key)
                cache_path = tensorrt_prebuilt_calibration_cache_path(
                    model_path,
                    width,
                    height,
                    model_precision,
                    mode_name,
                    use_hg=use_hg,
                    predequantize=False,
                    qdq_fusion="native",
                    require_exists=False,
                )
                if not cache_path:
                    print(
                        f"[calib] skipped non-native INT8 preset: {precision_key} "
                        f"{res_key} {'HG' if use_hg else 'no-HG'}",
                        flush=True,
                    )
                    skipped += 1
                    continue

                print(
                    f"[calib] {precision_key} {res_key} "
                    f"{'HG' if use_hg else 'no-HG'} -> {cache_path}",
                    flush=True,
                )
                if args.dry_run:
                    continue
                if os.path.isfile(cache_path) and not args.force:
                    skipped += 1
                    continue

                engine_mode = tensorrt_mode_name(
                    model_precision,
                    mode_name,
                    predequantize=False,
                    qdq_fusion="native",
                )
                engine_path = tensorrt_engine_path(
                    model_path,
                    width,
                    height,
                    engine_mode,
                )
                onnx_path = tensorrt_onnx_path(
                    model_path,
                    width,
                    height,
                    engine_mode,
                )
                if args.force:
                    _remove_file(cache_path)
                    _remove_file(engine_path)
                    _remove_file(tensorrt_engine_metadata_path(engine_path))
                    _remove_file(onnx_path)
                    _remove_file(f"{onnx_path}.data")

                try:
                    processor = HDRTVNetTensorRT(
                        model_path,
                        device="auto",
                        precision=model_precision,
                        engine_width=width,
                        engine_height=height,
                        mode_name=mode_name,
                        use_hg=use_hg,
                        predequantize=False,
                        qdq_fusion="native",
                        keep_onnx=args.keep_onnx,
                        calibration_dataset=dataset,
                        calibration_frames=args.calibration_frames,
                        calibration_cache=cache_path,
                    )
                    del processor
                    gc.collect()
                except Exception as exc:
                    print(f"[calib] FAILED: {exc}", flush=True)
                    _discard_temporary_engine(engine_path, onnx_path, args.keep_onnx)
                    failures += 1
                    continue

                if os.path.isfile(cache_path):
                    _discard_temporary_engine(engine_path, onnx_path, args.keep_onnx)
                    print(
                        "[calib] discarded temporary calibration engine; "
                        "runtime/build tools will rebuild engines with their normal workspace",
                        flush=True,
                    )
                    built += 1
                else:
                    print(f"[calib] FAILED: cache was not written: {cache_path}", flush=True)
                    failures += 1

    print(
        f"[calib] done: built={built}, skipped={skipped}, failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
