"""Materialize quantization-friendly HDRTVNet++ experiment weights.

This does not train. It creates deterministic initialization checkpoints for
architecture experiments:
- HR selective-SFT bases: start from HR/Ensemble weights and drop only the
  SFT modules not present in that variant.
- HG fused-BN: folds eval BatchNorm into HG conv weights, preserving output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE  # noqa: E402
from models.hdrtvnet_modules.Hallucination_arch import (  # noqa: E402
    Hallucination_Generator,
    Hallucination_Generator_Direct,
    Hallucination_Generator_FusedBN,
)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a state_dict-like mapping")
    return {str(k).replace("module.", ""): v for k, v in payload.items()}


def _save_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def build_base_variant(
    base_path: Path,
    out_path: Path,
    le_arch: str,
    classifier: str,
) -> dict[str, object]:
    state = _load_state(base_path)
    model = Ensemble_AGCM_LE(classifier=classifier, le_arch=le_arch).eval()
    load_result = model.load_state_dict(state, strict=True)
    _save_state(model, out_path)

    original_keys = set(state.keys())
    variant_keys = set(model.state_dict().keys())
    dropped = sorted(k for k in original_keys if k not in variant_keys)
    return {
        "path": str(out_path),
        "source": str(base_path),
        "architecture": f"Ensemble_AGCM_LE(le_arch={le_arch})",
        "classifier": classifier,
        "le_arch": le_arch,
        "load_result": str(load_result),
        "dropped_sft_key_count": len(dropped),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
    }


def build_fusedbn_hg(hg_path: Path, out_path: Path) -> dict[str, object]:
    state = _load_state(hg_path)
    original = Hallucination_Generator().eval()
    fused = Hallucination_Generator_FusedBN().eval()
    original.load_state_dict(state, strict=True)
    load_result = fused.load_state_dict(state, strict=True)
    _save_state(fused, out_path)

    torch.manual_seed(123)
    img = torch.rand(1, 3, 64, 64)
    mask = (img.max(dim=1, keepdim=True).values > 0.75).float()
    with torch.no_grad():
        y0 = original((img, mask))
        y1 = fused((img, mask))
    return {
        "path": str(out_path),
        "source": str(hg_path),
        "architecture": "Hallucination_Generator_FusedBN",
        "load_result": str(load_result),
        "max_abs_diff_vs_original": float((y0 - y1).abs().max()),
        "mean_abs_diff_vs_original": float((y0 - y1).abs().mean()),
        "parameter_count": int(sum(p.numel() for p in fused.parameters())),
    }


def build_hg_variant(hg_path: Path, out_path: Path, hg_arch: str) -> dict[str, object]:
    hg_arch = str(hg_arch or "fusedbn").strip().lower()
    canonical = hg_arch.replace("-", "").replace("_", "")
    direct_match = re.fullmatch(
        r"directh(2|4|8|16|32)wide([0-9]+)x([0-9]+)",
        canonical,
    )
    if hg_arch in {"fusedbn", "fused-bn", "qfriendly", "quant-friendly"}:
        return build_fusedbn_hg(hg_path, out_path)
    if not direct_match:
        raise ValueError(f"Unsupported HG variant: {hg_arch}")

    bottleneck_scale = int(direct_match.group(1))
    wide_nf = int(direct_match.group(2))
    trunk_depth = int(direct_match.group(3))
    state = _load_state(hg_path)
    direct = Hallucination_Generator_Direct(
        trunk_depth=trunk_depth,
        wide_nf=wide_nf,
        bottleneck_scale=bottleneck_scale,
    ).eval()
    load_result = direct.load_state_dict(state, strict=True)
    _save_state(direct, out_path)

    torch.manual_seed(123)
    img = torch.rand(1, 3, 64, 64)
    mask = (img.max(dim=1, keepdim=True).values > 0.75).float()
    with torch.no_grad():
        y = direct((img, mask))
    return {
        "path": str(out_path),
        "source": str(hg_path),
        "architecture": (
            "Hallucination_Generator_Direct("
            f"bottleneck_scale={bottleneck_scale}, wide_nf={wide_nf}, "
            f"trunk_depth={trunk_depth})"
        ),
        "hg_arch": f"directh{bottleneck_scale}wide{wide_nf}x{trunk_depth}",
        "load_result": str(load_result),
        "identity_init_max_abs_diff": float((y - img).abs().max()),
        "identity_init_mean_abs_diff": float((y - img).abs().mean()),
        "parameter_count": int(sum(p.numel() for p in direct.parameters())),
    }


def parse_args() -> argparse.Namespace:
    weights_dir = _SRC / "models" / "weights"
    parser = argparse.ArgumentParser(
        description="Create quantization-friendly HDRTVNet++ experiment checkpoints."
    )
    parser.add_argument(
        "--base",
        default=str(weights_dir / "original" / "HR.pt"),
        help="HR/Ensemble_AGCM_LE base checkpoint.",
    )
    parser.add_argument(
        "--hg",
        default=str(weights_dir / "original" / "HG.pt"),
        help="Original HG checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(weights_dir / "experiments"),
        help="Output directory.",
    )
    parser.add_argument(
        "--base-variants",
        default="cleantrunk,cleantrunk_deep8,cleantrunk_deep12,cleantrunk_wide64x4,cleantrunk_wide64x8,cleantrunk_flat8,cleantrunk_flat16,cleantrunk_flatwide64x8,cleantrunk_flatall8,cleantrunk_flatallwide64x8,cleantrunk_flatallwide128x8,cleantrunk_flatallwide128x16,plainflatall8,plainflatallwide64x8,plainflatlinear8,plainflatlinear16,plainflatlinear32,plainflatlinearwide64x8,plainflatlinearwide64x16,plainflatlinearwide128x8,plainflatlinearwide128x16,plainbottleneckh8wide128x8,plainbottleneckh8wide128x16,plainbottleneckh8wide256x8,plainbottleneckh8wide256x16,plainbottleneckh16wide128x16,plainbottleneckh16wide256x16,plaindirecth8wide64x8,plaindirecth8wide128x8,plaindirecth8wide128x16,plaindirecth16wide128x16,plaindirecth16wide256x16,bottleneck_heavy,bottleneck_sft,lowres_sft,downpath_sft",
        help="Comma-separated LE architecture variants to materialize.",
    )
    parser.add_argument(
        "--classifier",
        default="color_condition",
        help="AGCM classifier variant for base checkpoints.",
    )
    parser.add_argument(
        "--hg-out",
        default="HG_weights_fusedbn.pth",
        help="Output filename for the fused-BN HG checkpoint.",
    )
    parser.add_argument(
        "--hg-variants",
        default="fusedbn",
        help=(
            "Comma-separated HG architecture variants to materialize. "
            "Examples: fusedbn,directh8wide64x8,directh16wide128x16"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = Path(args.base)
    hg_path = Path(args.hg)
    out_dir = Path(args.out_dir)

    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not hg_path.is_file():
        raise FileNotFoundError(hg_path)

    variant_names = [
        name.strip()
        for name in str(args.base_variants).replace(";", ",").split(",")
        if name.strip()
    ]
    base_variants = []
    classifier = str(args.classifier).strip() or "color_condition"
    classifier_suffix = "" if classifier == "color_condition" else f"_{classifier.replace('-', '_')}"
    for variant in variant_names:
        safe_name = variant.replace("-", "_")
        base_variants.append(
            build_base_variant(
                base_path,
                out_dir / f"HR_qfriendly_{safe_name}{classifier_suffix}_init.pt",
                variant,
                classifier,
            )
        )
    hg_variant_names = [
        name.strip()
        for name in str(args.hg_variants).replace(";", ",").split(",")
        if name.strip()
    ]
    hg_variants = []
    for index, variant in enumerate(hg_variant_names):
        if variant.strip().lower() in {"fusedbn", "fused-bn", "qfriendly", "quant-friendly"}:
            out_name = args.hg_out if index == 0 else "HG_weights_fusedbn.pth"
        else:
            out_name = f"HG_weights_{variant.replace('-', '_')}.pth"
        hg_variants.append(build_hg_variant(hg_path, out_dir / out_name, variant))
    hg_info = hg_variants[0] if hg_variants else build_fusedbn_hg(hg_path, out_dir / args.hg_out)
    manifest = {
        "schema": "hdrtvnet_quant_friendly_experiments_v1",
        "base_variants": base_variants,
        "hg": hg_info,
        "hg_variants": hg_variants,
        "env": {
            "HDRTVNET_LE_ARCH": "select one of base_variants[].le_arch",
            "HDRTVNET_HG_ARCH": "select one of hg_variants[].hg_arch or fusedbn",
        },
    }
    manifest_path = out_dir / f"quant_friendly_manifest{classifier_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
