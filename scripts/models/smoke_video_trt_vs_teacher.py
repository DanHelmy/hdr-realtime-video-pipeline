"""Smoke-test a TensorRT video path against the original HR+HG teacher."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_torch import HDRTVNetTensorRT, HDRTVNetTorch  # noqa: E402


@contextmanager
def _env_override(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0
    mse = float(np.mean(diff * diff))
    return float(-10.0 * math.log10(mse + 1e-10))


def _mae_u8(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _diff_x8(student: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    diff = np.abs(student.astype(np.int16) - teacher.astype(np.int16))
    return np.clip(diff * 8, 0, 255).astype(np.uint8)


def _read_frame(cap: cv2.VideoCapture, frame_idx: int, width: int, height: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _parse_frames(text: str) -> list[int]:
    frames = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            frames.append(int(item))
    return frames


def _runtime_precision(preset: str) -> str:
    text = str(preset or "").strip().lower()
    if text.startswith("int8-mixed"):
        return "int8-mixed"
    if text.startswith("int8-full"):
        return "int8-full"
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", default="6606,23122,42940")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--teacher-base", default=str(_SRC / "models" / "weights" / "original" / "HR.pt"))
    parser.add_argument("--teacher-hg", default=str(_SRC / "models" / "weights" / "original" / "HG.pt"))
    parser.add_argument("--teacher-classifier", default="color_condition")
    parser.add_argument("--teacher-le-arch", default="sft")
    parser.add_argument("--teacher-hg-arch", default="pixelshuffle")
    parser.add_argument("--student-base", required=True)
    parser.add_argument("--student-hg", default="")
    parser.add_argument("--student-classifier", required=True)
    parser.add_argument("--student-le-arch", required=True)
    parser.add_argument("--student-hg-arch", required=True)
    parser.add_argument("--precision", default="int8-mixed-ptq")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"])
    parser.add_argument("--engine-tag", default="")
    parser.add_argument(
        "--base-mode-name",
        default="",
        help="Optional pre-tagged TensorRT base mode name, e.g. 'INT8 Mixed (PTQ)_hg_my_tag'.",
    )
    parser.add_argument("--include", default="")
    parser.add_argument(
        "--qat-composition",
        default=os.environ.get("HDRTVNET_TRT_INT8_QAT_COMPOSITION", "runtime"),
        choices=["runtime", "checkpoint"],
        help="TensorRT INT8 composition policy for the student engine.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--keep-onnx", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _parse_frames(args.frames)
    if not frames:
        raise RuntimeError("No frames requested")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    use_hg = str(args.use_hg).strip() != "0"

    with _env_override({
        "HDRTVNET_CLASSIFIER": args.teacher_classifier,
        "HDRTVNET_LE_ARCH": args.teacher_le_arch,
        "HDRTVNET_HG_ARCH": args.teacher_hg_arch,
    }):
        teacher = HDRTVNetTorch(
            args.teacher_base,
            precision="fp16",
            hg_weights=args.teacher_hg,
            use_hg=use_hg,
            compile_model=False,
            warmup_passes=0,
        )

    student_env = {
        "HDRTVNET_CLASSIFIER": args.student_classifier,
        "HDRTVNET_LE_ARCH": args.student_le_arch,
        "HDRTVNET_HG_ARCH": args.student_hg_arch,
        "HDRTVNET_TRT_INT8_SINGLE_INPUT": "1",
        "HDRTVNET_TRT_INT8_MODELOPT": "1",
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH": "1",
        "HDRTVNET_TRT_INT8_QAT_COMPOSITION": args.qat_composition,
        "HDRTVNET_TRT_INT8_MODELOPT_LAYER_POLICY": "checkpoint",
    }
    if args.engine_tag:
        student_env["HDRTVNET_TRT_ENGINE_TAG"] = args.engine_tag
    if args.include:
        student_env["HDRTVNET_TRT_INT8_MODELOPT_TORCH_INCLUDE"] = args.include
    if args.keep_onnx:
        student_env["HDRTVNET_TRT_KEEP_ONNX"] = "1"

    with _env_override(student_env):
        student = HDRTVNetTensorRT(
            args.student_base,
            precision=_runtime_precision(args.precision),
            mode_name=str(args.base_mode_name or args.precision),
            engine_width=args.width,
            engine_height=args.height,
            hg_weights=args.student_hg,
            use_hg=use_hg,
            keep_onnx=bool(args.keep_onnx),
        )

    rows = []
    try:
        for frame_idx in frames:
            source = _read_frame(cap, frame_idx, args.width, args.height)
            _save(out_dir / f"frame_{frame_idx:06d}_source_{args.width}x{args.height}.png", source)

            t0 = time.perf_counter()
            teacher_out = teacher.process(source)
            t1 = time.perf_counter()
            student_out, pre_ms, infer_ms, post_ms = student.process_timed(source)
            t2 = time.perf_counter()

            _save(out_dir / f"frame_{frame_idx:06d}_teacher.png", teacher_out)
            _save(out_dir / f"frame_{frame_idx:06d}_student_trt_mixed.png", student_out)
            _save(out_dir / f"frame_{frame_idx:06d}_diff_x8.png", _diff_x8(student_out, teacher_out))

            rows.append({
                "frame": int(frame_idx),
                "teacher_ms": float((t1 - t0) * 1000.0),
                "student_trt_mixed_ms": float((t2 - t1) * 1000.0),
                "student_pre_ms": float(pre_ms),
                "student_infer_ms": float(infer_ms),
                "student_post_ms": float(post_ms),
                "teacher_shape": list(teacher_out.shape),
                "student_shape": list(student_out.shape),
                "psnr_vs_teacher": _psnr_u8(student_out, teacher_out),
                "mae_vs_teacher": _mae_u8(student_out, teacher_out),
            })
            print(
                f"frame {frame_idx}: psnr={rows[-1]['psnr_vs_teacher']:.2f} dB, "
                f"mae={rows[-1]['mae_vs_teacher']:.5f}, "
                f"student={rows[-1]['student_trt_mixed_ms']:.2f} ms",
                flush=True,
            )
    finally:
        cap.release()

    summary = {
        "video": str(Path(args.video).resolve()),
        "total_frames": total_frames,
        "fps": fps,
        "width": int(args.width),
        "height": int(args.height),
        "student_base": str(args.student_base),
        "student_hg": str(args.student_hg),
        "student_classifier": str(args.student_classifier),
        "student_le_arch": str(args.student_le_arch),
        "student_hg_arch": str(args.student_hg_arch),
        "teacher_classifier": str(args.teacher_classifier),
        "teacher_le_arch": str(args.teacher_le_arch),
        "teacher_hg_arch": str(args.teacher_hg_arch),
        "precision": str(args.precision),
        "use_hg": bool(use_hg),
        "qat_composition": str(args.qat_composition),
        "engine_tag": str(args.engine_tag),
        "rows": rows,
    }
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
