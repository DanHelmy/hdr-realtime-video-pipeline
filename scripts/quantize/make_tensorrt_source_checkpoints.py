"""Generate TensorRT-targeted source checkpoints for every INT8 variant.

These outputs are not separately trained models. They are deterministic
TensorRT source checkpoints: native FP32 state plus the same INT8 masks and
activation qparams from the source checkpoint. TensorRT export lowers those
qparams to signed explicit Q/DQ for engine build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from make_portable_int8_checkpoint import (
    convert_checkpoint,
    default_tensorrt_source_path,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_WEIGHTS_DIR = _REPO_ROOT / "src" / "models" / "weights"
_DEFAULT_OUTPUT_DIR = _WEIGHTS_DIR / "tensorrt_sources"


def _default_inputs() -> list[Path]:
    return sorted(
        p
        for p in _WEIGHTS_DIR.glob("Ensemble_AGCM_LE_int8_*.pt")
        if p.is_file() and p.parent.name != "tensorrt_sources"
    )


def _expected_counts(checkpoint: dict) -> tuple[int, int, int, int]:
    arch = checkpoint.get("architecture", {})
    total = 149 if bool(arch.get("use_hg", True)) else 128
    quantization = str(checkpoint.get("quantization") or "")
    fp16 = len(checkpoint.get("fp16_layers") or [])
    if quantization == "w8a8_mixed":
        w8a8 = len(checkpoint.get("w8a8_layers") or [])
    elif quantization == "w8a8_full":
        w8a8 = total
    else:
        w8a8 = 0
    w8a16 = max(0, total - w8a8 - fp16)
    return w8a8, w8a16, fp16, total


def _summarize(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    w8a8, w8a16, fp16, total = _expected_counts(checkpoint)
    qparams = len(checkpoint.get("activation_qparams") or {})
    return (
        f"{path.name}: {checkpoint.get('quantization')} "
        f"W8A8={w8a8} W8A16={w8a16} FP16={fp16} total={total} "
        f"activation={checkpoint.get('activation_quant')} qparams={qparams} "
        f"target={checkpoint.get('target_backend')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate all TensorRT source checkpoints from runtime INT8 checkpoints."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        help="Optional checkpoint list. Defaults to every Ensemble_AGCM_LE_int8_*.pt under weights/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output folder. Default: src/models/weights/tensorrt_sources.",
    )
    parser.add_argument(
        "--activation-quant",
        default="source",
        choices=["symmetric", "source"],
        help=(
            "Activation qparams to store. Default 'source' preserves PyTorch "
            "checkpoint parity; TensorRT export still emits signed Q/DQ."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files.",
    )
    args = parser.parse_args()

    inputs = [p.resolve() for p in args.checkpoints] if args.checkpoints else _default_inputs()
    if not inputs:
        raise FileNotFoundError(f"No INT8 checkpoints found under {_WEIGHTS_DIR}")

    output_dir = args.output_dir.resolve()
    print(f"TensorRT source output: {output_dir}")
    for input_path in inputs:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = default_tensorrt_source_path(input_path, output_dir)
        if args.dry_run:
            print(f"[dry-run] {input_path} -> {output_path}")
            continue
        convert_checkpoint(
            input_path,
            output_path,
            activation_quant=args.activation_quant,
            target_backend="tensorrt",
        )
        print("  " + _summarize(output_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
