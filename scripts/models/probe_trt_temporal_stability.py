"""Probe TensorRT INT8 temporal stability against the FP16 TensorRT path."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_torch import HDRTVNetTensorRT  # noqa: E402


@contextmanager
def _env_override(values: dict[str, str | None]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _repo_path(*parts: str) -> str:
    return str(_ROOT.joinpath(*parts))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0
    mse = float(np.mean(diff * diff))
    return float(-10.0 * math.log10(mse + 1e-10))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _highlight_mae(a: np.ndarray, b: np.ndarray) -> float:
    ref = b.astype(np.float32) / 255.0
    luma = ref[..., 2] * 0.2126 + ref[..., 1] * 0.7152 + ref[..., 0] * 0.0722
    mask = luma >= 0.65
    if not np.any(mask):
        return 0.0
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    return float(np.mean(diff[mask]))


def _temporal_delta_mae(candidate: np.ndarray, reference: np.ndarray) -> float:
    if len(candidate) < 2:
        return 0.0
    cand_delta = np.diff(candidate.astype(np.float32) / 255.0, axis=0)
    ref_delta = np.diff(reference.astype(np.float32) / 255.0, axis=0)
    return float(np.mean(np.abs(cand_delta - ref_delta)))


def _diff_x8(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    diff = np.abs(candidate.astype(np.int16) - reference.astype(np.int16))
    return np.clip(diff * 8, 0, 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 520), 38), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _contact_row(source: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    thumb_w = 640
    thumb_h = max(1, int(round(source.shape[0] * (thumb_w / source.shape[1]))))
    items = [
        _label(source, "source"),
        _label(reference, "fp16 trt"),
        _label(candidate, "candidate int8"),
        _label(_diff_x8(candidate, reference), "diff x8"),
    ]
    resized = [cv2.resize(item, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA) for item in items]
    return cv2.hconcat(resized)


def _read_frames(video: str, frame_indices: list[int], width: int, height: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frames_by_index: dict[int, np.ndarray] = {}
    try:
        requested = [int(frame_idx) for frame_idx in frame_indices]
        unique = sorted(set(requested))
        if not unique:
            return []
        span = unique[-1] - unique[0]
        if span <= 1000:
            wanted = set(unique)
            cap.set(cv2.CAP_PROP_POS_FRAMES, unique[0])
            current = unique[0]
            while current <= unique[-1]:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame {current}")
                if current in wanted:
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    frames_by_index[current] = frame
                current += 1
        else:
            for frame_idx in unique:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame {frame_idx}")
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frames_by_index[frame_idx] = frame
    finally:
        cap.release()
    return [frames_by_index[int(frame_idx)] for frame_idx in frame_indices]


def _parse_frames(text: str) -> list[int]:
    frames: list[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            frames.append(int(part))
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--reference-model",
        default=_repo_path("src", "models", "weights", "distilled", "hr", "HR_qfriendly_selectsft1235_fp32.pt"),
    )
    parser.add_argument(
        "--candidate-model",
        default=_repo_path("src", "models", "weights", "distilled", "hr", "HR_qfriendly_selectsft1235_fp32.pt"),
    )
    parser.add_argument("--precision", default="int8-mixed")
    parser.add_argument("--use-hg", default="0", choices=["0", "1"])
    parser.add_argument("--hg-weights", default="")
    parser.add_argument("--candidate-hg-weights", default="")
    parser.add_argument("--include", default="")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--outputs", default="")
    parser.add_argument("--output-policy", default="active")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--method", default="kl_div")
    parser.add_argument("--calib-steps", type=int, default=8)
    parser.add_argument("--score-steps", type=int, default=4)
    parser.add_argument("--effective-bits", type=float, default=8.25)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_indices = _parse_frames(args.frames)
    if not frame_indices:
        frame_indices = [args.start + i * args.stride for i in range(max(1, args.count))]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_hg = args.use_hg == "1"

    frames = _read_frames(args.video, frame_indices, args.width, args.height)

    if args.force:
        os.environ["HDRTVNET_TRT_FORCE_REBUILD"] = "1"

    reference = HDRTVNetTensorRT(
        args.reference_model,
        precision="fp16",
        engine_width=args.width,
        engine_height=args.height,
        mode_name=f"probe_fp16_{'hg' if use_hg else 'nohg'}",
        hg_weights=args.hg_weights or None,
        use_hg=use_hg,
    )

    candidate_env = {
        "HDRTVNET_TRT_ENGINE_TAG": args.tag,
        "HDRTVNET_TRT_INT8_MODELOPT": "1",
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH": "1",
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_MODE": args.mode,
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_METHOD": args.method,
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_CALIB_STEPS": str(args.calib_steps),
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_SCORE_STEPS": str(args.score_steps),
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_EFFECTIVE_BITS": str(args.effective_bits),
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_INCLUDE": args.include or None,
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_EXCLUDE": args.exclude or None,
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_OUTPUTS": args.outputs or None,
        "HDRTVNET_TRT_INT8_MODELOPT_TORCH_OUTPUT_POLICY": args.output_policy or None,
    }
    with _env_override(candidate_env):
        candidate = HDRTVNetTensorRT(
            args.candidate_model,
            precision=args.precision,
            engine_width=args.width,
            engine_height=args.height,
            mode_name=f"probe_{args.tag}_{'hg' if use_hg else 'nohg'}",
            hg_weights=args.candidate_hg_weights or args.hg_weights or None,
            use_hg=use_hg,
        )

    reference_out = []
    candidate_out = []
    rows = []
    for frame_idx, source in zip(frame_indices, frames):
        ref = reference.process(source)
        cand = candidate.process(source)
        reference_out.append(ref.copy())
        candidate_out.append(cand.copy())
        rows.append(
            {
                "frame": int(frame_idx),
                "psnr_vs_fp16": _psnr(cand, ref),
                "mae_vs_fp16": _mae(cand, ref),
                "highlight_mae_vs_fp16": _highlight_mae(cand, ref),
            }
        )
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_source.png"), source)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_fp16.png"), ref)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_candidate.png"), cand)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_diff_x8.png"), _diff_x8(cand, ref))

    ref_stack = np.stack(reference_out, axis=0)
    cand_stack = np.stack(candidate_out, axis=0)
    temporal_mae = _temporal_delta_mae(cand_stack, ref_stack)
    contact_rows = [_contact_row(src, ref, cand) for src, ref, cand in zip(frames, reference_out, candidate_out)]
    contact = cv2.vconcat(contact_rows)
    cv2.imwrite(str(out_dir / "contact.png"), contact)

    summary = {
        "video": str(Path(args.video).resolve()),
        "frames": frame_indices,
        "width": int(args.width),
        "height": int(args.height),
        "precision": str(args.precision),
        "tag": str(args.tag),
        "include": str(args.include),
        "exclude": str(args.exclude),
        "outputs": str(args.outputs),
        "output_policy": str(args.output_policy),
        "mode": str(args.mode),
        "method": str(args.method),
        "calib_steps": int(args.calib_steps),
        "score_steps": int(args.score_steps),
        "effective_bits": float(args.effective_bits),
        "mean_psnr_vs_fp16": float(np.mean([row["psnr_vs_fp16"] for row in rows])),
        "mean_mae_vs_fp16": float(np.mean([row["mae_vs_fp16"] for row in rows])),
        "max_mae_vs_fp16": float(np.max([row["mae_vs_fp16"] for row in rows])),
        "mean_highlight_mae_vs_fp16": float(np.mean([row["highlight_mae_vs_fp16"] for row in rows])),
        "temporal_delta_mae_vs_fp16": temporal_mae,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"contact -> {out_dir / 'contact.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
