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
    default_tensorrt_source_path,
)
from split_distilled_tensorrt_sources import generate_tensorrt_source
from models.hdrtvnet_torch import tensorrt_source_checkpoint_validation_error


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_WEIGHTS_DIR = _REPO_ROOT / "src" / "models" / "weights"
_DEFAULT_OUTPUT_DIR = _WEIGHTS_DIR / "distilled"


def _default_inputs() -> list[Path]:
    return sorted(
        p
        for folder in (_WEIGHTS_DIR / "pytorch_int8" / "hr", _WEIGHTS_DIR / "pytorch_int8" / "hg")
        for p in folder.glob("*.pt")
        if p.is_file()
    )


def _quantizable_layers_from_state(checkpoint: dict) -> set[str]:
    layers: set[str] = set()
    for key, value in (checkpoint.get("state_dict") or {}).items():
        text = str(key)
        if text.endswith(".weight_int8"):
            layers.add(text[: -len(".weight_int8")])
            continue
        if text.endswith(".weight") and torch.is_tensor(value) and value.ndim in {2, 4}:
            layers.add(text[: -len(".weight")])
    return layers


def _expected_counts(checkpoint: dict) -> tuple[int, int, int, int]:
    quantization = str(checkpoint.get("quantization") or "")
    fp16 = len(checkpoint.get("fp16_layers") or [])
    qparams = checkpoint.get("activation_qparams") or {}
    if qparams:
        w8a8 = len(qparams)
    elif quantization == "w8a8_mixed":
        w8a8 = len(checkpoint.get("w8a8_layers") or [])
    else:
        w8a8 = len(
            {
                str(key)[: -len(".x_scale")]
                for key in (checkpoint.get("state_dict") or {})
                if str(key).endswith(".x_scale")
            }
        )
    total = len(_quantizable_layers_from_state(checkpoint))
    if quantization == "w8a8_full":
        total = max(total, w8a8)
    w8a16 = max(0, total - w8a8 - fp16)
    return w8a8, w8a16, fp16, max(total, w8a8 + w8a16 + fp16)


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
        help="Optional checkpoint list. Defaults to every organized .pt under weights/pytorch_int8/hr and weights/pytorch_int8/hg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output folder. Default: src/models/weights/distilled.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when the existing TensorRT source is current.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether TensorRT source checkpoints are current.",
    )
    args = parser.parse_args()

    inputs = [p.resolve() for p in args.checkpoints] if args.checkpoints else _default_inputs()
    if not inputs:
        raise FileNotFoundError(f"No INT8 checkpoints found under {_WEIGHTS_DIR}")

    output_dir = args.output_dir.resolve()
    print(f"TensorRT source output: {output_dir}")
    stale = []
    for input_path in inputs:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = default_tensorrt_source_path(input_path, output_dir)
        reason = tensorrt_source_checkpoint_validation_error(output_path, str(input_path))
        if reason is None and not args.force:
            print(f"[ok] {output_path.name} is current")
            continue
        if args.check:
            stale.append((input_path, output_path, reason or "forced"))
            print(f"[stale] {output_path.name}: {reason or 'forced'}")
            continue
        if args.dry_run:
            print(f"[dry-run] {input_path} -> {output_path} ({reason or 'forced'})")
            continue
        print(f"[generate] {input_path.name} -> {output_path.name} ({reason or 'forced'})")
        generate_tensorrt_source(
            input_path,
            output_path,
            activation_quant=args.activation_quant,
        )
        print("  " + _summarize(output_path))

    if args.check and stale:
        print(f"TensorRT source check failed: {len(stale)} stale/missing file(s)")
        return 1
    if args.check:
        print("TensorRT source check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
