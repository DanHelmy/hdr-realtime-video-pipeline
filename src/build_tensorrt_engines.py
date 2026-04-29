"""
Pre-build TensorRT engines for NVIDIA inference.

Playback/export/benchmark still build engines on demand when a cache entry is
missing. This script is the NVIDIA equivalent of compile_kernels.py for users
who want to populate the engine cache ahead of time.
"""

from __future__ import annotations

import argparse
import os
import sys

from windows_runtime import ensure_windows_supported

ensure_windows_supported("HDRTVNet++ TensorRT engine builder")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models.hdrtvnet_torch import HDRTVNetTensorRT, tensorrt_engine_path


def _weight(name: str) -> str:
    return os.path.join(_HERE, "models", "weights", name)


_PRECISION_MAP = {
    "fp16": ("fp16", _weight("Ensemble_AGCM_LE.pth")),
    "fp32": ("fp32", _weight("Ensemble_AGCM_LE.pth")),
    "int8-mixed": ("int8-mixed", _weight("Ensemble_AGCM_LE_int8_mixed_qat.pt")),
    "int8-full": ("int8-full", _weight("Ensemble_AGCM_LE_int8_full.pt")),
}


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = str(text).lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid resolution '{text}', expected WxH (e.g. 1920x1080)"
        ) from exc


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
        default="fp16",
        choices=tuple(_PRECISION_MAP.keys()),
        help="Model precision/mode to build.",
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
        help="Path to HG_weights.pth (overrides default path).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the .engine file already exists.",
    )
    args = parser.parse_args()

    precision, default_model = _PRECISION_MAP[args.precision]
    model_path = os.path.abspath(args.model or default_model)
    use_hg = str(args.use_hg).strip() != "0"
    mode_name = f"{args.precision}_{'hg' if use_hg else 'nohg'}"

    if not os.path.isfile(model_path):
        print(f"ERROR: model weights not found: {model_path}", file=sys.stderr)
        return 2

    for w, h in args.resolutions:
        engine_path = tensorrt_engine_path(model_path, w, h, mode_name)
        if args.force and os.path.isfile(engine_path):
            os.remove(engine_path)
        if os.path.isfile(engine_path):
            print(f"[tensorrt] cache hit: {engine_path}")
            continue
        print(
            f"[tensorrt] building {args.precision} "
            f"{'HG' if use_hg else 'no-HG'} engine for {w}x{h}"
        )
        processor = HDRTVNetTensorRT(
            model_path,
            device="auto",
            precision=precision,
            engine_width=w,
            engine_height=h,
            mode_name=mode_name,
            hg_weights=args.hg_weights,
            use_hg=use_hg,
        )
        del processor
        print(f"[tensorrt] ready: {engine_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
