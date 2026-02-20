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
    python quantize_int8_full.py
    python quantize_int8_full.py --precision fp16
    python quantize_int8_full.py --num-calibrate 16
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_torch import (
    W8A8Conv2d, W8A8Linear,
    _quantize_model_w8a8, calibrate_w8a8,
)


# ===================================================================
# Helpers
# ===================================================================

def load_fp32_model(model_path: str) -> nn.Module:
    """Load the FP32 Ensemble_AGCM_LE model."""
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
    model.load_state_dict(cleaned, strict=True)
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
    cond = F.interpolate(img_dev, scale_factor=0.25, mode="bilinear",
                         align_corners=False)
    return (img_dev, cond)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full INT8 (W8A8) quantization for HDRTVNet++"
    )
    parser.add_argument("--model", default="src/models/weights/Ensemble_AGCM_LE.pth",
                        help="Path to FP32 .pth weights")
    parser.add_argument("--output",
                        default="src/models/weights/Ensemble_AGCM_LE_int8_full.pt",
                        help="Output path for quantized checkpoint")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision for runtime dequantization")
    parser.add_argument("--calibration-dir", default="dataset/test_sdr",
                        help="Directory of SDR images for activation calibration")
    parser.add_argument("--num-calibrate", type=int, default=16,
                        help="Number of images for activation scale calibration")
    parser.add_argument("--num-validate", type=int, default=8,
                        help="Number of images for quality validation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (auto = GPU if available)")
    args = parser.parse_args()

    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ------------------------------------------------------------------
    # 1. Load FP32 model
    # ------------------------------------------------------------------
    print(f"Loading FP32 model from {args.model} ...")
    model = load_fp32_model(args.model)

    # ------------------------------------------------------------------
    # 2. Quantize weights → INT8 and prepare activation hooks
    # ------------------------------------------------------------------
    print("Quantizing weights to INT8 (W8A8) ...")
    _quantize_model_w8a8(model, compute_dtype)

    # Cast remaining parameters (norms, etc.) to compute_dtype
    model = model.to(dtype=compute_dtype, device=device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Calibrate activation scales from real images
    # ------------------------------------------------------------------
    print(f"\nCalibrating activation scales ({args.num_calibrate} images) ...")
    calib_images = load_calibration_images(
        args.calibration_dir, args.num_calibrate
    )
    print(f"  Loaded {len(calib_images)} calibration images")

    # Build calibration inputs as (img, condition) tuples
    calib_inputs = [prepare_model_input(img, device, compute_dtype)
                    for img in calib_images]
    calibrate_w8a8(model, calib_inputs)

    # ------------------------------------------------------------------
    # 4. Save quantized + calibrated checkpoint
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = {
        "state_dict": model.state_dict(),
        "compute_dtype": str(compute_dtype),
        "quantization": "w8a8_full",
        "architecture": {
            "classifier": "color_condition",
            "cond_c": 6,
            "in_nc": 3,
            "out_nc": 3,
            "nf": 32,
            "act_type": "relu",
            "weighting_network": False,
        },
    }
    torch.save(save_data, args.output)

    orig_kb = os.path.getsize(args.model) / 1024
    quant_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved → {args.output}")
    print(f"  Original : {orig_kb:,.1f} KB")
    print(f"  Quantized: {quant_kb:,.1f} KB")
    if quant_kb > 0:
        print(f"  Ratio    : {orig_kb / quant_kb:.2f}x")

    # ------------------------------------------------------------------
    # 5. Quality validation — FP16 reference vs W8A8
    # ------------------------------------------------------------------
    print(f"\nValidation ({device}, {args.precision}):")
    val_images = calib_images[:args.num_validate]
    print(f"  {len(val_images)} images")

    # FP32 reference (cast to compute_dtype for apples-to-apples)
    fp32_model = load_fp32_model(args.model)
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
