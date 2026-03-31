"""
Full INT8 (W8A8) Quantization for HDRTVNet++.

Both weights AND activations are quantized to INT8:
  - Weights:      per-output-channel static scale (same as W8A16)
  - Activations:  per-tensor static scale (calibrated from real images)

At inference both are dequantized to FP16 and computed via standard
F.conv2d / F.linear.  torch.compile fuses quant + dequant + conv into
efficient GPU kernels.  Works on any GPU (NVIDIA / AMD).

Usage
-----
    python scripts/quantize/quantize_int8_full.py
    python scripts/quantize/quantize_int8_full.py --precision fp16
    python scripts/quantize/quantize_int8_full.py --num-calibrate 16
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite
from models.hdrtvnet_torch import (
    W8A8Conv2d, W8A8Linear,
    _quantize_model_w8a8, calibrate_w8a8,
)


# ===================================================================
# Helpers
# ===================================================================

def load_fp32_model(model_path: str, hg_weights: str, use_hg: bool) -> nn.Module:
    """Load the FP32 model (HG composite or base AGCM+LE)."""
    if use_hg:
        model = HG_Composite(
            classifier="color_condition",
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type="relu",
            weighting_network=False,
            hg_nf=64,
            mask_r=0.75,
        )
    else:
        model = Ensemble_AGCM_LE(
            classifier="color_condition",
            cond_c=6,
            in_nc=3,
            out_nc=3,
            nf=32,
            act_type="relu",
            weighting_network=False,
        )

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    if use_hg:
        model.base.load_state_dict(cleaned, strict=True)
    else:
        model.load_state_dict(cleaned, strict=True)

    if use_hg:
        hg_state = torch.load(hg_weights, map_location="cpu")
        if isinstance(hg_state, dict) and "state_dict" in hg_state:
            hg_state = hg_state["state_dict"]
        model.hg.load_state_dict(hg_state, strict=True)
    model.eval()
    return model


def load_calibration_images(calib_dir: str, max_images: int = 16,
                            max_long_edge: int = 960):
    """Load calibration images, resized to manageable dimensions."""
    paths = sorted(glob.glob(os.path.join(calib_dir, "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No images found in {calib_dir}")
    if max_images > 0:
        paths = paths[:max_images]
    tensors = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_long_edge:
            scale = max_long_edge / longest
            new_w = int(round(w * scale / 8)) * 8
            new_h = int(round(h * scale / 8)) * 8
        else:
            new_w = int(round(w / 8)) * 8
            new_h = int(round(h / 8)) * 8
        new_w, new_h = max(new_w, 8), max(new_h, 8)
        if new_w != w or new_h != h:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
        tensors.append(t)
    return tensors


def prepare_model_input(img_tensor, device, dtype):
    """Prepare (input, condition) tuple for the model."""
    img_dev = img_tensor.to(device=device, dtype=dtype)
    try:
        cond = F.interpolate(img_dev, scale_factor=0.25, mode="bicubic",
                             align_corners=False, antialias=True)
    except TypeError:
        cond = F.interpolate(img_dev, scale_factor=0.25, mode="bicubic",
                             align_corners=False)
    return (img_dev, cond)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full INT8 (W8A8) quantization for HDRTVNet++"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "Ensemble_AGCM_LE.pth"),
                        help="Path to FP32 .pth weights")
    parser.add_argument("--output",
                        default=os.path.join(
                            _REPO_ROOT,
                            "src",
                            "models",
                            "weights",
                            "Ensemble_AGCM_LE_int8_full.pt",
                        ),
                        help="Output path for quantized checkpoint")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision for runtime dequantization")
    parser.add_argument("--save-fp16", action="store_true",
                        help="Save checkpoint with FP16 compute buffers even "
                             "if calibration ran in FP32 (useful for CPU runs)")
    parser.add_argument("--activation-quant",
                        default="asymmetric",
                        choices=["symmetric", "asymmetric"],
                        help="Activation quantization mode for W8A8")
    parser.add_argument("--calibration-method",
                        default="percentile",
                        choices=["max", "percentile"],
                        help="Activation calibration method")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="Percentile for activation calibration (high)")
    parser.add_argument("--percentile-low", type=float, default=0.1,
                        help="Lower percentile for asymmetric calibration")
    parser.add_argument("--max-calib-samples", type=int, default=200000,
                        help="Max activation samples per layer for percentile")
    parser.add_argument(
        "--calibration-dir",
        default=os.path.join(_REPO_ROOT, "dataset", "train_sdr"),
                        help="Directory of SDR images for activation calibration")
    parser.add_argument("--hg-weights",
                        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "HG_weights.pth"),
                        help="Path to HG weights (HG_weights.pth)")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"],
                        help="Use HG refinement (1) or base AGCM+LE only (0)")
    parser.add_argument("--num-calibrate", type=int, default=16,
                        help="Number of images for activation scale calibration "
                             "(0 = all)")
    parser.add_argument("--num-validate", type=int, default=8,
                        help="Number of images for quality validation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (auto = GPU if available)")
    parser.add_argument("--calibration-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for activation calibration")
    args = parser.parse_args()

    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cpu":
        compute_dtype = torch.float32
    calib_device = device if args.calibration_device == "auto" else torch.device(
        args.calibration_device
    )
    calib_dtype = compute_dtype if calib_device.type != "cpu" else torch.float32

    use_hg = str(args.use_hg).strip() != "0"

    # ------------------------------------------------------------------
    # 1. Load FP32 model
    # ------------------------------------------------------------------
    print(f"Loading FP32 model from {args.model} ...")
    model = load_fp32_model(args.model, args.hg_weights, use_hg)

    # ------------------------------------------------------------------
    # 2. Quantize weights ' INT8 and prepare activation hooks
    # ------------------------------------------------------------------
    use_asym = args.activation_quant == "asymmetric"
    print("Quantizing weights to INT8 (W8A8) ...")
    _quantize_model_w8a8(model, compute_dtype, asymmetric=use_asym)

    # Cast remaining parameters (norms, etc.) to compute_dtype
    model = model.to(dtype=compute_dtype, device=device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Calibrate activation scales from real images
    # ------------------------------------------------------------------
    calib_desc = "all" if args.num_calibrate <= 0 else str(args.num_calibrate)
    print(f"\nCalibrating activation scales ({calib_desc} images) ...")
    calib_images = load_calibration_images(
        args.calibration_dir, args.num_calibrate
    )
    print(f"  Loaded {len(calib_images)} calibration images")
    if len(calib_images) == 0:
        raise RuntimeError("No valid calibration images loaded for calibration")

    # Build calibration inputs as (img, condition) tuples
    if calib_device != device or calib_dtype != compute_dtype:
        model = model.to(dtype=calib_dtype, device=calib_device)
        model.eval()

    calib_inputs = [prepare_model_input(img, calib_device, calib_dtype)
                    for img in calib_images]
    calibrate_w8a8(
        model, calib_inputs,
        method=args.calibration_method,
        percentile=args.percentile,
        percentile_low=args.percentile_low,
        max_samples=args.max_calib_samples,
    )

    if calib_device != device or calib_dtype != compute_dtype:
        model = model.to(dtype=compute_dtype, device=device)
        model.eval()

    # ------------------------------------------------------------------
    # 4. Save quantized + calibrated checkpoint
    # ------------------------------------------------------------------
    save_dtype = compute_dtype
    if args.save_fp16:
        save_dtype = torch.float16
        if compute_dtype != save_dtype:
            print("  [info] Saving checkpoint with FP16 compute buffers "
                  "after FP32 calibration.")
            model = model.to(dtype=save_dtype, device=device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = {
        "state_dict": model.state_dict(),
        "compute_dtype": str(save_dtype),
        "quantization": "w8a8_full",
        "activation_quant": args.activation_quant,
        "calibration_method": args.calibration_method,
        "calibration_percentile": args.percentile,
        "calibration_percentile_low": args.percentile_low,
        "architecture": {
            "classifier": "color_condition",
            "cond_c": 6,
            "in_nc": 3,
            "out_nc": 3,
            "nf": 32,
            "act_type": "relu",
            "weighting_network": False,
            "use_hg": use_hg,
            "hg_nf": 64,
            "mask_r": 0.75,
        },
    }
    torch.save(save_data, args.output)

    orig_bytes = os.path.getsize(args.model)
    orig_label = "base model"
    if use_hg:
        orig_bytes += os.path.getsize(args.hg_weights)
        orig_label = "base + HG models"
    orig_kb = orig_bytes / 1024
    quant_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved -> {args.output}")
    print(f"  Original ({orig_label}) : {orig_kb:,.1f} KB")
    print(f"  Quantized: {quant_kb:,.1f} KB")
    if quant_kb > 0:
        print(f"  Ratio    : {orig_kb / quant_kb:.2f}x")

    # ------------------------------------------------------------------
    # 5. Quality validation " FP16 reference vs W8A8
    # ------------------------------------------------------------------
    print(f"\nValidation ({device}, {args.precision}):")
    val_images = calib_images[:args.num_validate]
    print(f"  {len(val_images)} images")

    # FP32 reference (cast to compute_dtype for apples-to-apples)
    fp32_model = load_fp32_model(args.model, args.hg_weights, use_hg)
    fp32_model = fp32_model.to(dtype=compute_dtype, device=device).eval()

    psnrs = []
    with torch.inference_mode():
        for i, img in enumerate(val_images):
            inp = prepare_model_input(img, device, compute_dtype)
            ref, _ = fp32_model(inp)
            out, _ = model(inp)

            mse = ((ref.float() - out.float()) ** 2).mean().item()
            psnr = -10 * np.log10(mse + 1e-10)
            psnrs.append(psnr)
            print(f"  Image {i + 1:3d}: PSNR = {psnr:.2f} dB")

    print(f"  Average  : {np.mean(psnrs):.2f} dB")

    # ------------------------------------------------------------------
    # 6. Quick inference speed test
    # ------------------------------------------------------------------
    if device.type == "cuda":
        print(f"\nSpeed test ({device}):")
        test_inp = prepare_model_input(calib_images[0], device, compute_dtype)
        # Warmup
        for _ in range(5):
            with torch.inference_mode():
                model(test_inp)
        torch.cuda.synchronize()

        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model(test_inp)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  Eager : {avg:.1f} +/- {std:.1f} ms")

    print("\nDone!")


if __name__ == "__main__":
    main()



