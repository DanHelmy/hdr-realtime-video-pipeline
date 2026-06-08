"""Split portable TensorRT source checkpoints into HR and HG source files."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from make_portable_int8_checkpoint import convert_checkpoint


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_WEIGHTS = _REPO_ROOT / "src" / "models" / "weights"


def _load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expected checkpoint with state_dict: {path}")
    return checkpoint


def _filter_layer_list(checkpoint: dict, key: str, *, prefix: str | None) -> list[str]:
    values = []
    for layer in checkpoint.get(key) or []:
        text = str(layer)
        if prefix is None:
            if text.startswith("hg."):
                continue
            text = text[5:] if text.startswith("base.") else text
        else:
            if not text.startswith(prefix):
                continue
            text = text[len(prefix):]
        values.append(text)
    return values


def _filter_qparams(checkpoint: dict, *, prefix: str | None) -> dict:
    qparams = {}
    for layer, value in (checkpoint.get("activation_qparams") or {}).items():
        text = str(layer)
        if prefix is None:
            if text.startswith("hg."):
                continue
            text = text[5:] if text.startswith("base.") else text
        else:
            if not text.startswith(prefix):
                continue
            text = text[len(prefix):]
        qparams[text] = value
    return qparams


def _filter_weight_qparams(checkpoint: dict, *, prefix: str | None) -> dict:
    qparams = {}
    for layer, value in (checkpoint.get("weight_qparams") or {}).items():
        text = str(layer)
        if prefix is None:
            if text.startswith("hg."):
                continue
            text = text[5:] if text.startswith("base.") else text
        else:
            if not text.startswith(prefix):
                continue
            text = text[len(prefix):]
        qparams[text] = value
    return qparams


def _copy_split_quant_metadata(save_data: dict, checkpoint: dict, *, prefix: str | None) -> None:
    if checkpoint.get("activation_qparams") is not None:
        save_data["activation_qparams"] = _filter_qparams(checkpoint, prefix=prefix)
    if checkpoint.get("weight_qparams") is not None:
        save_data["weight_qparams"] = _filter_weight_qparams(checkpoint, prefix=prefix)
    for key in ("w8a8_layers", "fp16_layers"):
        if checkpoint.get(key) is not None:
            save_data[key] = _filter_layer_list(checkpoint, key, prefix=prefix)


def _split_hr_source_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = _load(input_path)
    arch = dict(checkpoint.get("architecture") or {})
    arch["use_hg"] = False
    state = {}
    for key, value in (checkpoint.get("state_dict") or {}).items():
        clean_key = key[5:] if str(key).startswith("base.") else str(key)
        if clean_key.startswith("hg."):
            continue
        state[clean_key] = value
    save_data = dict(checkpoint)
    save_data["state_dict"] = state
    save_data["architecture"] = arch
    _copy_split_quant_metadata(save_data, checkpoint, prefix=None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)
    print(f"Saved HR TensorRT source: {output_path} ({len(state)} tensors)")


def _split_hg_source_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = _load(input_path)
    arch = dict(checkpoint.get("architecture") or {})
    state = {}
    for key, value in (checkpoint.get("state_dict") or {}).items():
        if str(key).startswith("hg."):
            state[str(key)[3:]] = value
    if not state:
        raise ValueError(f"No HG tensors found in {input_path}")
    save_data = dict(checkpoint)
    save_data["state_dict"] = state
    save_data["architecture"] = {
        "hg_arch": arch.get("hg_arch"),
        "hg_nf": arch.get("hg_nf", 64),
        "in_nc": arch.get("in_nc", 3),
        "out_nc": arch.get("out_nc", 3),
        "mask_r": arch.get("mask_r", 0.75),
    }
    _copy_split_quant_metadata(save_data, checkpoint, prefix="hg.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)
    print(f"Saved HG TensorRT source: {output_path} ({len(state)} tensors)")


def generate_tensorrt_source(
    input_path: Path,
    output_path: Path,
    *,
    activation_quant: str = "source",
) -> None:
    stem = input_path.stem
    if stem.startswith("HR_HG_int8_") or stem.startswith("HR_HG_original_int8_"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_source = Path(tmp) / f"{stem}_composite_source{input_path.suffix}"
            convert_checkpoint(
                input_path,
                tmp_source,
                activation_quant=activation_quant,
                target_backend="tensorrt",
            )
            _split_hg_source_checkpoint(tmp_source, output_path)
        return

    convert_checkpoint(
        input_path,
        output_path,
        activation_quant=activation_quant,
        target_backend="tensorrt",
    )


def generate_composite_tensorrt_sources(
    input_path: Path,
    hr_output_path: Path,
    hg_output_path: Path,
    *,
    activation_quant: str = "source",
    write_hr: bool = True,
    write_hg: bool = True,
) -> None:
    if not write_hr and not write_hg:
        print(f"Skipped existing composite TensorRT sources: {input_path}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp_source = Path(tmp) / f"{input_path.stem}_composite_source{input_path.suffix}"
        convert_checkpoint(
            input_path,
            tmp_source,
            activation_quant=activation_quant,
            target_backend="tensorrt",
        )
        if write_hr:
            _split_hr_source_checkpoint(tmp_source, hr_output_path)
        else:
            print(f"Skipped existing HR TensorRT source: {hr_output_path}")
        if write_hg:
            _split_hg_source_checkpoint(tmp_source, hg_output_path)
        else:
            print(f"Skipped existing HG TensorRT source: {hg_output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate TensorRT source checkpoints from original INT8 checkpoints."
    )
    parser.add_argument("--weights-dir", type=Path, default=_WEIGHTS)
    parser.add_argument(
        "--only-original",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    args = parser.parse_args()

    weights = args.weights_dir
    mappings = {
        "mixed_ptq": "mixed_ptq",
        "mixed_qat": "mixed_qat",
        "mixed_qat_film": "mixed_qat_film",
        "full_ptq": "full_ptq",
        "full_qat": "full_qat",
        "full_qat_film": "full_qat_film",
    }

    original_int8 = weights / "original" / "pytorch_int8"
    original_trt = weights / "original" / "tensorrt"
    if not original_int8.is_dir():
        raise FileNotFoundError(f"Missing original PyTorch INT8 directory: {original_int8}")
    for tag, out_tag in mappings.items():
        runtime_tag = tag.replace("_ptq", "")
        hr_source = original_int8 / "hr" / f"HR_original_int8_{runtime_tag}.pt"
        hg_source = original_int8 / "hg" / f"HR_HG_original_int8_{runtime_tag}.pt"
        if hr_source.is_file():
            hr_output = original_trt / "hr" / f"HR_original_int8_{out_tag}.pt"
            if args.missing_only and hr_output.is_file():
                print(f"Skipped existing TensorRT source: {hr_output}")
            else:
                generate_tensorrt_source(hr_source, hr_output)
        if hg_source.is_file():
            hr_hg_output = original_trt / "hr_hg" / f"HR_HG_original_int8_{out_tag}.pt"
            hg_output = original_trt / "hg" / f"HG_original_int8_{out_tag}.pt"
            generate_composite_tensorrt_sources(
                hg_source,
                hr_hg_output,
                hg_output,
                write_hr=not (args.missing_only and hr_hg_output.is_file()),
                write_hg=not (args.missing_only and hg_output.is_file()),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
