"""Generate SDR/teacher-output PNG pairs from movie frames for Film QAT."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.quantize.quantize_int8_mixed_qat import (  # noqa: E402
    _build_fp32_model,
    prepare_model_input,
)


def _parse_frames(text: str) -> list[int]:
    frames = []
    for item in str(text or "").replace(";", ",").split(","):
        item = item.strip()
        if item:
            frames.append(int(item))
    return frames


def _even_frames(total_frames: int, count: int, start: int, end: int) -> list[int]:
    hi = total_frames - 1 if end <= 0 else min(total_frames - 1, end)
    lo = max(0, start)
    if count <= 1:
        return [lo]
    values = np.linspace(lo, hi, num=count, dtype=np.int64)
    return sorted({int(v) for v in values if 0 <= int(v) < total_frames})


def _tensor_to_bgr_u16(tensor: torch.Tensor) -> np.ndarray:
    arr = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    bgr = arr[:, :, ::-1]
    return np.ascontiguousarray(
        np.clip((bgr * 65535.0) + 0.5, 0.0, 65535.0).astype(np.uint16)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--teacher-base", required=True)
    parser.add_argument("--teacher-hg", default="")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"])
    parser.add_argument("--classifier", default="color_condition")
    parser.add_argument("--le-arch", default="selectsft1235")
    parser.add_argument("--hg-arch", default="directh16wide64x8")
    parser.add_argument("--frames", default="")
    parser.add_argument("--name-prefix", default="")
    parser.add_argument("--num-frames", type=int, default=160)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0)
    parser.add_argument("--max-long-edge", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["HDRTVNET_CLASSIFIER"] = args.classifier
    os.environ["HDRTVNET_LE_ARCH"] = args.le_arch
    os.environ["HDRTVNET_HG_ARCH"] = args.hg_arch

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    use_hg = str(args.use_hg).strip() != "0"
    teacher = _build_fp32_model(args.teacher_base, args.teacher_hg, use_hg)
    teacher = teacher.to(device=device, dtype=dtype).eval()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = _parse_frames(args.frames)
    if not frames:
        frames = _even_frames(
            total_frames,
            args.num_frames,
            args.start_frame,
            args.end_frame,
        )

    out_root = Path(args.output_root)
    sdr_dir = out_root / "sdr"
    hdr_dir = out_root / "hdr"
    sdr_dir.mkdir(parents=True, exist_ok=True)
    hdr_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with torch.inference_mode():
        for ordinal, frame_idx in enumerate(frames, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"skip frame {frame_idx}: decode failed", flush=True)
                continue
            if args.max_long_edge > 0:
                h, w = frame.shape[:2]
                longest = max(h, w)
                if longest > args.max_long_edge:
                    scale = float(args.max_long_edge) / float(longest)
                    new_w = max(8, int(round(w * scale / 8)) * 8)
                    new_h = max(8, int(round(h * scale / 8)) * 8)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
            inp = prepare_model_input(tensor, device, dtype)
            result = teacher(inp)
            output = result[0] if isinstance(result, (tuple, list)) else result

            prefix = str(args.name_prefix or "movie").strip() or "movie"
            safe_prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix)
            name = f"{safe_prefix}_{frame_idx:08d}.png"
            cv2.imwrite(str(sdr_dir / name), frame)
            cv2.imwrite(str(hdr_dir / name), _tensor_to_bgr_u16(output))
            written += 1
            if ordinal % 25 == 0 or ordinal == len(frames):
                print(f"wrote {written}/{len(frames)} pair(s)", flush=True)

    cap.release()
    print(f"done: {written} pairs -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
