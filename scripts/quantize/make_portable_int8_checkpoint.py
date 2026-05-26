"""
Convert runtime INT8 checkpoints into backend-neutral portable checkpoints.

The existing INT8 .pt files store W8/W8A8 wrapper state directly
(`weight_int8`, scales, activation ranges). That is usable by this PyTorch
runtime, but it couples the checkpoint to one runtime representation and makes
TensorRT export inherit asymmetric/full-graph choices.

This tool saves:
  - native FP32 model weights,
  - the mixed/full quantization recipe,
  - calibrated activation qparams for W8A8 layers.

At load time the runtime recreates PyTorch wrappers or TensorRT Q/DQ export
modules from the same neutral base.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_TENSORRT_SOURCE_DIR = (
    _REPO_ROOT / "src" / "models" / "weights" / "tensorrt_sources"
)
sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite
from models.hdrtvnet_torch import (
    W8A8Conv2d,
    W8A8Linear,
    _PORTABLE_INT8_CHECKPOINT_FORMAT,
    _PORTABLE_INT8_STATE_FORMAT,
    _TENSORRT_SOURCE_CHECKPOINT_SCHEMA,
    _predequantize_model,
    _quantize_model_mixed,
    _quantize_model_mixed_v2,
    _quantize_model_w8a8,
    _tensorrt_source_signature,
)


def _dtype_from_checkpoint(checkpoint: dict) -> torch.dtype:
    text = str(checkpoint.get("compute_dtype", "torch.float16"))
    return torch.float16 if "16" in text else torch.float32


def _build_model_from_arch(arch: dict) -> nn.Module:
    use_hg = bool(arch.get("use_hg", True))
    if use_hg:
        return HG_Composite(
            classifier=arch.get("classifier", "color_condition"),
            cond_c=arch.get("cond_c", 6),
            in_nc=arch.get("in_nc", 3),
            out_nc=arch.get("out_nc", 3),
            nf=arch.get("nf", 32),
            act_type=arch.get("act_type", "relu"),
            weighting_network=arch.get("weighting_network", False),
            hg_nf=arch.get("hg_nf", 64),
            mask_r=arch.get("mask_r", 0.75),
        )
    return Ensemble_AGCM_LE(
        classifier=arch.get("classifier", "color_condition"),
        cond_c=arch.get("cond_c", 6),
        in_nc=arch.get("in_nc", 3),
        out_nc=arch.get("out_nc", 3),
        nf=arch.get("nf", 32),
        act_type=arch.get("act_type", "relu"),
        weighting_network=arch.get("weighting_network", False),
    )


def _materialize_runtime_model(checkpoint: dict) -> nn.Module:
    arch = checkpoint.get("architecture", {})
    model = _build_model_from_arch(arch)
    compute_dtype = _dtype_from_checkpoint(checkpoint)
    quant_type = checkpoint.get("quantization", "")

    if quant_type == "w8a8_mixed":
        w8a8_layers = checkpoint.get("w8a8_layers", None)
        fp16_layers = checkpoint.get("fp16_layers", None)
        asymmetric = checkpoint.get("activation_quant", "symmetric") == "asymmetric"
        if w8a8_layers is not None:
            _quantize_model_mixed_v2(
                model,
                compute_dtype=compute_dtype,
                w8a8_layers=w8a8_layers,
                fp16_layers=fp16_layers,
                asymmetric=asymmetric,
            )
        else:
            _quantize_model_mixed(
                model,
                compute_dtype=compute_dtype,
                channel_threshold=checkpoint.get("channel_threshold", 32),
                fp16_layers=fp16_layers,
            )
    elif quant_type == "w8a8_full":
        asymmetric = checkpoint.get("activation_quant", "symmetric") == "asymmetric"
        _quantize_model_w8a8(model, compute_dtype=compute_dtype, asymmetric=asymmetric)
    else:
        raise ValueError(
            f"Unsupported quantization={quant_type!r}; expected w8a8_mixed or w8a8_full."
        )

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _scalar(tensor: torch.Tensor | float | int) -> float:
    if isinstance(tensor, torch.Tensor):
        return float(tensor.detach().float().reshape(-1)[0].item())
    return float(tensor)


def _collect_activation_qparams(
    model: nn.Module,
    target_activation_quant: str,
) -> dict[str, dict[str, float]]:
    qparams: dict[str, dict[str, float]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (W8A8Conv2d, W8A8Linear)):
            continue

        x_scale = max(_scalar(module.x_scale), 1e-8)
        if target_activation_quant == "source" and getattr(module, "is_asymmetric", False):
            qparams[name] = {
                "scale": x_scale,
                "zero": _scalar(getattr(module, "x_zero", torch.tensor(0.0))),
            }
            continue

        if getattr(module, "is_asymmetric", False):
            x_min = _scalar(getattr(module, "x_zero", torch.tensor(0.0)))
            x_max = x_min + x_scale * 255.0
            x_scale = max(abs(x_min), abs(x_max), 1e-8) / 127.0

        qparams[name] = {"scale": float(x_scale), "zero": 0.0}
    return qparams


def _native_fp32_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.is_floating_point():
                value = value.float()
        state[key] = value
    return state


def default_tensorrt_source_path(
    input_path: Path,
    output_dir: Path | None = None,
    suffix: str = "",
) -> Path:
    directory = output_dir if output_dir is not None else _DEFAULT_TENSORRT_SOURCE_DIR
    return directory / f"{input_path.stem}{suffix}{input_path.suffix}"


def _file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def convert_checkpoint(
    input_path: Path,
    output_path: Path,
    *,
    activation_quant: str,
    target_backend: str = "portable",
) -> None:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Not a checkpoint dictionary with state_dict: {input_path}")
    if checkpoint.get("checkpoint_format") == _PORTABLE_INT8_CHECKPOINT_FORMAT:
        raise ValueError(f"Already portable: {input_path}")

    source_activation_quant = checkpoint.get("activation_quant", "symmetric")
    target_activation_quant = (
        source_activation_quant if activation_quant == "source" else activation_quant
    )
    if target_activation_quant not in {"symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported target activation quantization: {target_activation_quant}")

    model = _materialize_runtime_model(checkpoint)
    qparams = _collect_activation_qparams(model, activation_quant)
    with contextlib.redirect_stdout(io.StringIO()):
        _predequantize_model(model, torch.float32)
    native_state = _native_fp32_state_dict(model)
    backend = str(target_backend or "portable").strip().lower()
    if backend not in {"portable", "tensorrt"}:
        raise ValueError(f"Unsupported target backend: {target_backend}")

    skip_keys = {
        "state_dict",
        "checkpoint_format",
        "state_format",
        "activation_qparams",
        "activation_scales",
        "activation_zero_points",
        "backend_neutral",
        "portable_source_checkpoint",
        "portable_recipe",
        "target_backend",
        "tensorrt_source_checkpoint",
        "tensorrt_source_schema",
        "tensorrt_source_signature",
        "tensorrt_source_runtime_checkpoint",
        "tensorrt_source_activation_quant_policy",
    }
    save_data = {k: v for k, v in checkpoint.items() if k not in skip_keys}
    save_data.update(
        {
            "state_dict": native_state,
            "checkpoint_format": _PORTABLE_INT8_CHECKPOINT_FORMAT,
            "state_format": _PORTABLE_INT8_STATE_FORMAT,
            "backend_neutral": True,
            "target_backend": backend,
            "tensorrt_source_checkpoint": backend == "tensorrt",
            "tensorrt_source_schema": (
                _TENSORRT_SOURCE_CHECKPOINT_SCHEMA if backend == "tensorrt" else None
            ),
            "tensorrt_source_signature": (
                _tensorrt_source_signature() if backend == "tensorrt" else None
            ),
            "tensorrt_source_runtime_checkpoint": (
                _file_fingerprint(input_path) if backend == "tensorrt" else None
            ),
            "tensorrt_source_activation_quant_policy": (
                activation_quant if backend == "tensorrt" else None
            ),
            "activation_quant": target_activation_quant,
            "activation_qparams": qparams,
            "source_activation_quant": source_activation_quant,
            "source_checkpoint_format": "runtime_int8_wrappers",
            "portable_source_checkpoint": str(input_path.resolve()),
            "portable_recipe": {
                "fp_state": "native_fp32",
                "w8a8_activation_qparams": target_activation_quant,
                "w8_weights": "recreated_from_native_state_at_load",
                "tensorrt": (
                    "explicit_signed_qdq_from_same_masks_and_qparams"
                    if backend == "tensorrt"
                    else "explicit_qdq_from_same_masks_and_qparams"
                ),
            },
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)
    int8_tensors = sum(
        1
        for value in native_state.values()
        if torch.is_tensor(value) and value.dtype == torch.int8
    )
    if backend == "tensorrt":
        print(f"Saved TensorRT source checkpoint: {output_path}")
    else:
        print(f"Saved portable checkpoint: {output_path}")
    print(f"  source activation quant : {source_activation_quant}")
    print(f"  target activation quant : {target_activation_quant}")
    print(f"  target backend          : {backend}")
    print("  materialized state      : native FP32 source weights")
    print(f"  activation qparams      : {len(qparams)}")
    print(f"  native state tensors    : {len(native_state)} (int8={int8_tensors})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert INT8 runtime-wrapper checkpoints to portable FP32+metadata .pt files."
    )
    parser.add_argument("checkpoints", nargs="+", help="Input INT8 .pt checkpoint(s).")
    parser.add_argument("--output", default=None, help="Output path for one input.")
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_TENSORRT_SOURCE_DIR),
        help="Directory for converted outputs.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix for generated outputs. Default keeps the source filename.",
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
        "--target-backend",
        default="portable",
        choices=["portable", "tensorrt"],
        help="Mark outputs as generic portable checkpoints or TensorRT source checkpoints.",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.checkpoints]
    if args.output and len(inputs) != 1:
        parser.error("--output can only be used with exactly one input checkpoint")

    output_dir = Path(args.output_dir) if args.output_dir else None
    for input_path in inputs:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = (
            Path(args.output)
            if args.output
            else default_tensorrt_source_path(input_path, output_dir, args.suffix)
        )
        convert_checkpoint(
            input_path,
            output_path,
            activation_quant=args.activation_quant,
            target_backend=args.target_backend,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
