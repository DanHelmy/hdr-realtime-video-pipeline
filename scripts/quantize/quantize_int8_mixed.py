"""
Optimized Mixed INT8 Quantization for HDRTVNet++ (Sensitivity-Based).

Two key improvements over the original channel-threshold heuristic:

  1. **Per-layer sensitivity analysis** — quantize one layer at a time to
     W8A8, measure output MSE, rank by impact.  Layers causing less than
     ``--sensitivity-threshold`` MSE are assigned W8A8; the rest stay W8A16.

  2. **Asymmetric activation quantization** — W8A8 layers use unsigned
     [0, 255] with a zero-point instead of symmetric [-128, 127].
     This gives 2x precision for post-ReLU layers and ~1.8x for
     post-LeakyReLU, since the full 256 levels map the actual value range.

Usage
-----
    python scripts/quantize/quantize_int8_mixed.py
    python scripts/quantize/quantize_int8_mixed.py --sensitivity-threshold 1e-6
    python scripts/quantize/quantize_int8_mixed.py --legacy  # old channel-threshold mode
"""

import argparse
import glob
import math
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
    W8A8Conv2d, W8A8Linear, W8Conv2d, W8Linear,
    _quantize_model_mixed, _quantize_model_mixed_v2, calibrate_w8a8,
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
# Per-layer sensitivity analysis
# ===================================================================

def compute_layer_sensitivity(model, calibration_inputs, num_samples=8):
    """Measure per-layer quantization sensitivity (output MSE).

    For each Conv2d / Linear layer, inject W8A8 quantization noise
    (asymmetric activation + per-channel weight) into that layer alone
    and measure how much the final output changes vs the FP16 reference.

    Returns dict: {layer_name: mean_output_mse}.
    """
    model.eval()
    inputs = calibration_inputs[:num_samples]

    # Collect FP16 reference outputs
    ref_outputs = []
    with torch.inference_mode():
        for inp in inputs:
            out, _ = model(inp)
            ref_outputs.append(out.clone())

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))]
    print(f"  Sensitivity sweep: {len(layers)} layers x {len(inputs)} images")

    sensitivities = {}
    nonfinite_count = 0
    for idx, (name, module) in enumerate(layers):
        # Hook: inject W8A8 quant noise (asymmetric act + per-channel weight)
        def _make_hook(mod):
            def _hook(m, inp, out):
                x = inp[0]
                w = mod.weight

                # Per-output-channel weight quantization
                w_flat = w.reshape(w.shape[0], -1)
                w_sc = w_flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0

                # Asymmetric activation quantization [0, 255]
                x_min = x.detach().amin()
                x_max = x.detach().amax()
                x_range = (x_max - x_min).clamp(min=1e-8)
                x_sc = x_range / 255.0
                x_q = ((x - x_min) / x_sc).round().clamp(0, 255)
                x_dq = x_q * x_sc + x_min

                if isinstance(mod, nn.Conv2d):
                    w_q = (w / w_sc.view(-1, 1, 1, 1)).round().clamp(
                        -128, 127) * w_sc.view(-1, 1, 1, 1)
                    return F.conv2d(x_dq, w_q, mod.bias, mod.stride,
                                    mod.padding, mod.dilation, mod.groups)
                else:
                    w_q = (w / w_sc.view(-1, 1)).round().clamp(
                        -128, 127) * w_sc.view(-1, 1)
                    return F.linear(x_dq, w_q, mod.bias)
            return _hook

        h = module.register_forward_hook(_make_hook(module))

        mses = []
        with torch.inference_mode():
            for i, inp in enumerate(inputs):
                out, _ = model(inp)
                mse = ((out.float() - ref_outputs[i].float()) ** 2).mean().item()
                if not np.isfinite(mse):
                    mse = float("inf")
                    nonfinite_count += 1
                mses.append(mse)

        layer_mse = float(np.mean(mses)) if mses else float("inf")
        if not np.isfinite(layer_mse):
            layer_mse = float("inf")
        sensitivities[name] = layer_mse
        h.remove()

        if (idx + 1) % 32 == 0 or idx + 1 == len(layers):
            print(f"    {idx + 1}/{len(layers)} done...")

    if nonfinite_count > 0:
        print(f"  [warn] Replaced {nonfinite_count} non-finite sensitivity "
              f"measurements with +inf")
    return sensitivities


def select_w8a8_layers_threshold(sensitivities, threshold=1e-6):
    """Select layers for W8A8 using a fixed per-layer sensitivity threshold."""
    sorted_layers = sorted(
        sensitivities.items(),
        key=lambda x: x[1] if np.isfinite(x[1]) else float("inf"),
    )

    w8a8 = []
    w8a16 = []
    for name, mse in sorted_layers:
        if np.isfinite(mse) and mse <= threshold:
            w8a8.append(name)
        else:
            w8a16.append(name)

    return w8a8, w8a16


def select_w8a8_layers_auto(sensitivities, min_w8a16=0):
    """Auto-select W8A8/W8A16 split from the sensitivity curve.

    Uses a knee-point heuristic on log-sensitivity to choose where errors
    begin increasing sharply. This avoids hand-picking a fixed threshold.
    """
    sorted_layers = sorted(
        sensitivities.items(),
        key=lambda x: x[1] if np.isfinite(x[1]) else float("inf"),
    )
    finite_layers = [(name, float(mse)) for name, mse in sorted_layers
                     if np.isfinite(mse)]
    nonfinite_layers = [name for name, mse in sorted_layers
                        if not np.isfinite(mse)]
    n = len(finite_layers)
    total_n = len(sorted_layers)
    if total_n == 0:
        return [], [], {"knee_index": -1, "threshold": None,
                        "nonfinite_layers": 0}
    if n == 0:
        return [], [name for name, _ in sorted_layers], {
            "knee_index": -1,
            "threshold": None,
            "nonfinite_layers": len(nonfinite_layers),
        }

    mses = np.array([m for _, m in finite_layers], dtype=np.float64)
    y = np.log10(np.maximum(mses, 1e-20))
    x = np.arange(n, dtype=np.float64)

    if n < 3 or np.allclose(y, y[0]):
        split = n
        knee_index = n - 1
    else:
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        denom = math.hypot(x1 - x0, y1 - y0)
        if denom < 1e-12:
            split = n
            knee_index = n - 1
        else:
            # Distance of each point to line between first/last points.
            dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
            knee_index = int(np.argmax(dist))
            split = knee_index + 1

    max_w8a8 = n
    if min_w8a16 > 0:
        max_w8a8 = max(0, total_n - min_w8a16)
    if max_w8a8 <= 0:
        split = 0
    else:
        split = max(1, min(split, n, max_w8a8))

    w8a8 = [name for name, _ in finite_layers[:split]]
    w8a16 = [name for name, _ in finite_layers[split:]] + nonfinite_layers
    threshold = float(finite_layers[split - 1][1]) if split > 0 else None
    info = {
        "knee_index": split - 1 if split > 0 else -1,
        "threshold": threshold,
        "nonfinite_layers": len(nonfinite_layers),
    }
    return w8a8, w8a16, info


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized mixed INT8 quantization for HDRTVNet++ "
                    "(sensitivity-based with asymmetric activation quant)"
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
                            "Ensemble_AGCM_LE_int8_mixed.pt",
                        ),
                        help="Output path for quantized checkpoint")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision for runtime dequantization")
    parser.add_argument("--save-fp16", action="store_true",
                        help="Save checkpoint with FP16 compute buffers even "
                             "if calibration ran in FP32 (useful for CPU runs)")
    parser.add_argument("--calibration-method",
                        default="percentile",
                        choices=["max", "percentile"],
                        help="Activation calibration method for W8A8 layers")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="Percentile for activation calibration (high)")
    parser.add_argument("--percentile-low", type=float, default=0.1,
                        help="Lower percentile for asymmetric calibration")
    parser.add_argument("--max-calib-samples", type=int, default=200000,
                        help="Max activation samples per layer for percentile")
    parser.add_argument("--layer-selection", default="auto",
                        choices=["auto", "threshold"],
                        help="Layer assignment strategy for mixed quantization")
    parser.add_argument("--sensitivity-threshold", type=float, default=1e-6,
                        help="Per-layer MSE threshold for W8A8 assignment "
                             "(used when --layer-selection threshold)")
    parser.add_argument("--auto-min-w8a16", type=int, default=0,
                        help="Minimum number of layers to keep as W8A16 in "
                             "auto mode (0 = allow all-W8A8)")
    parser.add_argument("--num-sensitivity", type=int, default=8,
                        help="Images for sensitivity analysis")
    parser.add_argument(
        "--calibration-dir",
        default=os.path.join(_REPO_ROOT, "dataset", "train_sdr"),
                        help="Directory of SDR images for calibration")
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
    parser.add_argument("--full-cpu", action="store_true",
                        help="Run all phases on CPU")
    parser.add_argument("--sensitivity-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for sensitivity sweep")
    parser.add_argument("--calibration-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for activation calibration")
    # Legacy mode
    parser.add_argument("--legacy", action="store_true",
                        help="Use v1 channel-threshold heuristic (no sensitivity)")
    parser.add_argument("--channel-threshold", type=int, default=32,
                        help="(Legacy) Max channel count for W8A8 (1x1 convs)")
    args = parser.parse_args()

    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if args.full_cpu:
        device_str = "cpu"
        compute_dtype = torch.float32
    device = torch.device(device_str)
    if device.type == "cpu":
        compute_dtype = torch.float32

    sens_device = device if args.sensitivity_device == "auto" else torch.device(
        args.sensitivity_device
    )

    sens_dtype = compute_dtype if sens_device.type != "cpu" else torch.float32

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
    # 2. Quantization strategy
    # ------------------------------------------------------------------
    selection_mode = "legacy"
    selection_info = None

    if args.legacy:
        # v1: simple channel-threshold heuristic
        print(f"\nLegacy mixed INT8 (channel threshold = {args.channel_threshold}):")
        _quantize_model_mixed(model, compute_dtype,
                              channel_threshold=args.channel_threshold)
        w8a8_layers = None
        use_asymmetric = False
    else:
        selection_mode = args.layer_selection
        # v2: sensitivity analysis + asymmetric activation quant
        print(f"\n{'='*60}")
        print("Phase 1: Per-layer sensitivity analysis")
        print(f"{'='*60}")

        # Move to device for sensitivity sweep
        model = model.to(dtype=sens_dtype, device=sens_device)
        model.eval()

        calib_images = load_calibration_images(
            args.calibration_dir, max(args.num_calibrate, args.num_sensitivity)
        )
        print(f"  Loaded {len(calib_images)} images")
        if len(calib_images) == 0:
            raise RuntimeError(
                "No valid calibration images loaded for sensitivity analysis"
            )
        sens_inputs = [prepare_model_input(img, sens_device, sens_dtype)
                       for img in calib_images[:args.num_sensitivity]]

        t0 = time.perf_counter()
        sensitivities = compute_layer_sensitivity(
            model, sens_inputs, args.num_sensitivity
        )
        dt = time.perf_counter() - t0
        print(f"  Sweep took {dt:.1f}s")

        # Select layers
        if args.layer_selection == "auto":
            w8a8_layers, w8a16_layers, selection_info = select_w8a8_layers_auto(
                sensitivities, min_w8a16=args.auto_min_w8a16
            )
            thr = selection_info.get("threshold")
            if thr is None:
                print(f"  Auto split (knee @ index {selection_info['knee_index']}): "
                      "threshold unavailable")
            else:
                print(f"  Auto split (knee @ index {selection_info['knee_index']}): "
                      f"threshold ~= {thr:.2e}")
            bad = int(selection_info.get("nonfinite_layers", 0))
            if bad > 0:
                print(f"  [warn] {bad} layers had non-finite sensitivity "
                      "and were forced to W8A16")
        else:
            w8a8_layers, w8a16_layers = select_w8a8_layers_threshold(
                sensitivities, threshold=args.sensitivity_threshold
            )
            selection_info = {"threshold": float(args.sensitivity_threshold)}

        # Print sensitivity results
        print(f"\n{'='*60}")
        print("Phase 2: Layer assignment")
        print(f"{'='*60}")
        sorted_sens = sorted(
            sensitivities.items(),
            key=lambda x: x[1] if np.isfinite(x[1]) else float("inf"),
        )
        w8a8_set = set(w8a8_layers)
        print(f"\n  Per-layer sensitivity (MSE, ascending):")
        for name, mse in sorted_sens:
            tag = "W8A8" if name in w8a8_set else "W8A16"
            if np.isfinite(mse):
                psnr_str = (f"{-10 * np.log10(mse):.1f} dB"
                            if mse > 1e-10 else ">100 dB")
                mse_str = f"{mse:.2e}"
            else:
                psnr_str = "nan"
                mse_str = "nan"
            print(f"    [{tag:5s}] {name:55s}  MSE={mse_str:>8s}  PSNR={psnr_str}")

        print(f"\n  W8A8  (asymmetric): {len(w8a8_layers)} layers")
        print(f"  W8A16 (weight-only): {len(w8a16_layers)} layers")

        # Re-load FP32 model (clean weights for quantization)
        model = load_fp32_model(args.model, args.hg_weights, use_hg)
        use_asymmetric = True

        print(f"\n{'='*60}")
        print("Phase 3: Quantization + calibration")
        print(f"{'='*60}")
        _quantize_model_mixed_v2(model, compute_dtype,
                                  w8a8_layers=w8a8_layers,
                                  asymmetric=True)

    # List W8A8 layers
    print("\n  W8A8 layers:")
    w8a8_params = 0
    w8_params = 0
    for name, m in model.named_modules():
        if isinstance(m, (W8A8Conv2d, W8A8Linear)):
            asym_tag = " (asymmetric)" if m.is_asymmetric else ""
            if isinstance(m, W8A8Conv2d):
                n_params = m.weight_int8.numel()
                print(f"    {name}: Conv2d({m.in_channels}->{m.out_channels}, "
                      f"k={m.kernel_size}){asym_tag}  [{n_params} params]")
            else:
                n_params = m.weight_int8.numel()
                print(f"    {name}: Linear({m.in_features}->{m.out_features})"
                      f"{asym_tag}  [{n_params} params]")
            w8a8_params += n_params
        elif isinstance(m, (W8Conv2d, W8Linear)):
            if isinstance(m, W8Conv2d):
                w8_params += m.weight_int8.numel()
            else:
                w8_params += m.weight_int8.numel()
    total_params = w8a8_params + w8_params
    print(f"\n  Composition: W8A8 = {w8a8_params:,} params "
          f"({100*w8a8_params/max(total_params,1):.1f}%), "
          f"W8A16 = {w8_params:,} params "
          f"({100*w8_params/max(total_params,1):.1f}%)")

    # Cast remaining parameters to compute_dtype + move to device
    model = model.to(dtype=compute_dtype, device=device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Calibrate activation scales
    # ------------------------------------------------------------------
    calib_desc = "all" if args.num_calibrate <= 0 else str(args.num_calibrate)
    print(f"\nCalibrating activation scales ({calib_desc} images) ...")
    calib_images = load_calibration_images(
        args.calibration_dir, args.num_calibrate
    )
    print(f"  Loaded {len(calib_images)} calibration images")
    if len(calib_images) == 0:
        raise RuntimeError("No valid calibration images loaded for calibration")

    # Calibration can be run on a different device (CPU on ROCm)
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

    # Move back to main device for validation/speed
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
        "quantization": "w8a8_mixed",
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

    if args.legacy:
        save_data["channel_threshold"] = args.channel_threshold
    else:
        save_data["w8a8_layers"] = w8a8_layers
        save_data["activation_quant"] = "asymmetric"
        save_data["selection_mode"] = selection_mode
        if selection_mode == "threshold":
            save_data["sensitivity_threshold"] = args.sensitivity_threshold
        elif selection_info is not None:
            save_data["auto_selection"] = selection_info

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
    # 5. Quality validation -- FP16 reference vs Mixed INT8
    # ------------------------------------------------------------------
    print(f"\nValidation ({device}, {args.precision}):")
    val_images = calib_images[:args.num_validate]
    print(f"  {len(val_images)} images")

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
            t0_t = time.perf_counter()
            with torch.inference_mode():
                model(test_inp)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0_t) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        print(f"  Eager : {avg:.1f} +/- {std:.1f} ms")

    print("\nDone!")


if __name__ == "__main__":
    main()
