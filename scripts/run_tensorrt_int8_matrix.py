"""Run the TensorRT INT8 readiness matrix on a NVIDIA machine.

This script builds TensorRT INT8 engines with predequantization off. The
default INT8 path is native TensorRT PTQ calibration, so ONNX artifacts do not
contain explicit Q/DQ nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_torch import (  # noqa: E402
    tensorrt_engine_path,
    tensorrt_mode_name,
    tensorrt_onnx_path,
    tensorrt_source_checkpoint_path,
)


_VARIANTS = {
    "fp32": {
        "builder_precision": "fp32",
        "runtime_precision": "fp32",
        "model": "distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt",
        "model_nohg": "distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt",
        "cli_run": "fp32",
        "gui_key": "FP32",
    },
    "fp16": {
        "builder_precision": "fp16",
        "runtime_precision": "fp16",
        "model": "distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt",
        "model_nohg": "distilled/hr/HR_qfriendly_spatialmixglobal_fp32.pt",
        "cli_run": "fp16",
        "gui_key": "FP16",
    },
    "int8-mixed-ptq": {
        "builder_precision": "int8-mixed-ptq",
        "runtime_precision": "int8-mixed",
        "model": "pytorch_int8/hg/HR_HG_int8_mixed.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_mixed.pt",
        "cli_run": "int8-mixed-ptq",
        "gui_key": "INT8 Mixed (PTQ)",
    },
    "int8-mixed-qat": {
        "builder_precision": "int8-mixed-qat",
        "runtime_precision": "int8-mixed",
        "model": "pytorch_int8/hg/HR_HG_int8_mixed_qat.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_mixed_qat.pt",
        "cli_run": "int8-mixed-qat",
        "gui_key": "INT8 Mixed (QAT)",
    },
    "int8-mixed-qat-film": {
        "builder_precision": "int8-mixed-qat-film",
        "runtime_precision": "int8-mixed",
        "model": "pytorch_int8/hg/HR_HG_int8_mixed_qat_film.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_mixed_qat_film.pt",
        "cli_run": "int8-mixed-qat-film",
        "gui_key": "INT8 Mixed (QAT) (Film)",
    },
    "int8-full-ptq": {
        "builder_precision": "int8-full-ptq",
        "runtime_precision": "int8-full",
        "model": "pytorch_int8/hg/HR_HG_int8_full.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_full.pt",
        "cli_run": "int8-full-ptq",
        "gui_key": "INT8 Full (PTQ)",
    },
    "int8-full-qat": {
        "builder_precision": "int8-full-qat",
        "runtime_precision": "int8-full",
        "model": "pytorch_int8/hg/HR_HG_int8_full_qat.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_full_qat.pt",
        "cli_run": "int8-full-qat",
        "gui_key": "INT8 Full (QAT)",
    },
    "int8-full-qat-film": {
        "builder_precision": "int8-full-qat-film",
        "runtime_precision": "int8-full",
        "model": "pytorch_int8/hg/HR_HG_int8_full_qat_film.pt",
        "model_nohg": "pytorch_int8/hr/HR_int8_full_qat_film.pt",
        "cli_run": "int8-full-qat-film",
        "gui_key": "INT8 Full (QAT) (Film)",
    },
}


def _configure_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{text}', expected WxH") from exc


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd))
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(f"[run] exit={proc.returncode} log={log_path}")
    return int(proc.returncode)


def _model_path(variant: str, use_hg: bool) -> Path:
    cfg = _VARIANTS[variant]
    name = cfg["model"] if use_hg else cfg["model_nohg"]
    return _SRC / "models" / "weights" / str(name)


def _paths_for(variant: str, use_hg: bool, resolution: tuple[int, int], qdq_fusion: str):
    cfg = _VARIANTS[variant]
    model_path = _model_path(variant, use_hg)
    runtime_precision = str(cfg["runtime_precision"])
    base_mode = f"{cfg['gui_key']}_{'hg' if use_hg else 'nohg'}"
    mode = tensorrt_mode_name(
        runtime_precision,
        base_mode,
        predequantize=False,
        qdq_fusion=qdq_fusion,
    )
    w, h = resolution
    return {
        "model_path": str(model_path),
        "tensorrt_source_path": str(tensorrt_source_checkpoint_path(str(model_path))),
        "engine_path": tensorrt_engine_path(str(model_path), w, h, mode),
        "onnx_path": tensorrt_onnx_path(str(model_path), w, h, mode),
        "mode": mode,
        "base_mode": base_mode,
        "runtime_precision": runtime_precision,
    }


def main() -> int:
    _configure_console()
    parser = argparse.ArgumentParser(description="Build, inspect, and benchmark TensorRT INT8 variants.")
    parser.add_argument("--resolutions", nargs="+", type=_parse_resolution, default=[(1280, 720)])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "fp32",
            "fp16",
            "int8-mixed-ptq",
            "int8-full-ptq",
            "int8-mixed-qat",
            "int8-full-qat",
            "int8-mixed-qat-film",
            "int8-full-qat-film",
        ],
        choices=tuple(_VARIANTS.keys()),
    )
    parser.add_argument("--use-hg", nargs="+", default=["0"], choices=["0", "1"])
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Optional video for CLI playback benchmark. Engine builds use a "
            "prebuilt .calib by default unless calibration inputs are provided."
        ),
    )
    parser.add_argument(
        "--calibration-dataset",
        default=None,
        help=(
            "Directory/image/manifest of SDR input frames for TensorRT native "
            "INT8 calibration. Takes priority over prebuilt .calib files."
        ),
    )
    parser.add_argument(
        "--calibration-cache",
        default=None,
        help="TensorRT native INT8 calibration cache path.",
    )
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--benchmark-runs", type=int, default=30, help="Dummy build-script benchmark runs.")
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--opt-level", type=int, default=5, choices=range(0, 6))
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=None,
        help="TensorRT builder workspace cap in GiB. Default: no explicit cap.",
    )
    parser.add_argument(
        "--qdq-fusion",
        default="native",
        choices=["native", "auto", "none", "add", "add-mul", "elementwise"],
        help=(
            "TensorRT INT8 mode. 'native' uses plain ONNX layers plus TensorRT "
            "PTQ calibration; other values use explicit Q/DQ. Default: native."
        ),
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=64,
        help="Frame count for TensorRT native INT8 calibration. Default: 64. Use 0 for all.",
    )
    parser.add_argument(
        "--calibrate-from-video",
        action="store_true",
        help=(
            "Use --video as the TensorRT calibration source when no dataset/cache "
            "is provided. By default, --video is only used for playback benchmark."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild engines.")
    parser.add_argument("--skip-source-validation", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--keep-onnx", action="store_true", default=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "logs" / "tensorrt_int8_matrix" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["HDRTVNET_TRT_BUILDER_OPT_LEVEL"] = str(args.opt_level)
    if args.workspace_gb is not None:
        env["HDRTVNET_TRT_WORKSPACE_GB"] = str(max(1.0, args.workspace_gb))

    rows: list[dict[str, object]] = []

    if not args.skip_source_validation and str(args.qdq_fusion) == "native":
        print("[matrix] source Q/DQ validation skipped for native TensorRT INT8")
    elif not args.skip_source_validation:
        code = _run(
            [
                sys.executable,
                "scripts/validate_tensorrt_sources.py",
                "--resolution",
                "256x256",
                "--qdq-fusion",
                str(args.qdq_fusion),
                "--output-dir",
                str(out_dir / "source_validation"),
            ],
            cwd=_ROOT,
            log_path=out_dir / "logs" / "source_validation.log",
            env=env,
        )
        if code != 0:
            return code

    for use_hg_text in args.use_hg:
        use_hg = str(use_hg_text).strip() != "0"
        for resolution in args.resolutions:
            for variant in args.variants:
                paths = _paths_for(variant, use_hg, resolution, args.qdq_fusion)
                row = {
                    "variant": variant,
                    "use_hg": use_hg,
                    "resolution": f"{resolution[0]}x{resolution[1]}",
                    **paths,
                }
                rows.append(row)
                if args.skip_build:
                    continue
                cmd = [
                    sys.executable,
                    "src/build_tensorrt_engines.py",
                    f"{resolution[0]}x{resolution[1]}",
                    "--precision",
                    str(_VARIANTS[variant]["builder_precision"]),
                    "--model",
                    paths["model_path"],
                    "--use-hg",
                    "1" if use_hg else "0",
                    "--predequantize",
                    "off",
                    "--qdq-fusion",
                    args.qdq_fusion,
                    "--opt-level",
                    str(args.opt_level),
                    "--benchmark-runs",
                    str(max(0, args.benchmark_runs)),
                    "--benchmark-warmup",
                    str(max(0, args.benchmark_warmup)),
                    "--keep-onnx",
                ]
                if args.workspace_gb is not None:
                    cmd += ["--workspace-gb", str(args.workspace_gb)]
                if str(args.qdq_fusion) == "native" and args.calibration_dataset:
                    cmd += [
                        "--calibration-dataset",
                        str(args.calibration_dataset),
                        "--calibration-frames",
                        str(args.calibration_frames),
                    ]
                elif str(args.qdq_fusion) == "native" and args.calibration_cache:
                    cmd += [
                        "--calibration-cache",
                        str(args.calibration_cache),
                        "--calibration-frames",
                        str(args.calibration_frames),
                    ]
                elif (
                    str(args.qdq_fusion) == "native"
                    and args.video
                    and args.calibrate_from_video
                ):
                    cmd += [
                        "--calibration-video",
                        str(args.video),
                        "--calibration-frames",
                        str(args.calibration_frames),
                    ]
                if args.force:
                    cmd.append("--force")
                    cmd.append("--force-onnx")
                log_name = f"build_{variant}_{'hg' if use_hg else 'nohg'}_{resolution[0]}x{resolution[1]}.log"
                row["build_returncode"] = _run(cmd, cwd=_ROOT, log_path=out_dir / "logs" / log_name, env=env)

                inspect_cmd = [
                    sys.executable,
                    "scripts/inspect_tensorrt_artifacts.py",
                    "--output",
                    str(out_dir / "inspect" / f"{variant}_{'hg' if use_hg else 'nohg'}_{resolution[0]}x{resolution[1]}.json"),
                ]
                if Path(paths["onnx_path"]).is_file():
                    inspect_cmd += ["--onnx", paths["onnx_path"]]
                if Path(paths["engine_path"]).is_file():
                    inspect_cmd += ["--engine", paths["engine_path"]]
                if len(inspect_cmd) > 5:
                    row["inspect_returncode"] = _run(
                        inspect_cmd,
                        cwd=_ROOT,
                        log_path=out_dir / "logs" / f"inspect_{variant}_{'hg' if use_hg else 'nohg'}_{resolution[0]}x{resolution[1]}.log",
                        env=env,
                    )

    if args.video and not args.skip_benchmark:
        for use_hg_text in args.use_hg:
            use_hg = str(use_hg_text).strip() != "0"
            for resolution in args.resolutions:
                cli_runs = [
                    str(_VARIANTS[variant]["cli_run"])
                    for variant in args.variants
                    if str(_VARIANTS[variant]["cli_run"]) in {
                        "fp32",
                        "fp16",
                        "int8-mixed-ptq",
                        "int8-mixed-qat",
                        "int8-mixed-qat-film",
                        "int8-full-ptq",
                        "int8-full-qat",
                        "int8-full-qat-film",
                    }
                ]
                if not cli_runs:
                    continue
                cmd = [
                    sys.executable,
                    "src/cli_playback_benchmark.py",
                    "--video",
                    str(args.video),
                    "--resolutions",
                    f"{resolution[0]}x{resolution[1]}",
                    "--runs",
                    *cli_runs,
                    "--use-hg",
                    "1" if use_hg else "0",
                    "--duration-s",
                    str(args.duration_s),
                    "--warmup-frames",
                    str(args.warmup_frames),
                    "--trt-qdq-fusion",
                    args.qdq_fusion,
                    "--trt-calibration-frames",
                    str(args.calibration_frames),
                    "--out-root",
                    str(out_dir / "playback"),
                ]
                if args.calibration_dataset:
                    cmd += ["--trt-calibration-dataset", str(args.calibration_dataset)]
                if args.calibration_cache:
                    cmd += ["--trt-calibration-cache", str(args.calibration_cache)]
                _run(
                    cmd,
                    cwd=_ROOT,
                    log_path=out_dir / "logs" / f"playback_{'hg' if use_hg else 'nohg'}_{resolution[0]}x{resolution[1]}.log",
                    env=env,
                )

    (out_dir / "matrix_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_fields = sorted({key for row in rows for key in row})
    with (out_dir / "matrix_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[matrix] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
