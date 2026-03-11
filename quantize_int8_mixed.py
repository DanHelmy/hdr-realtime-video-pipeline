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
    python quantize_int8_mixed.py
    python quantize_int8_mixed.py --sensitivity-threshold 1e-6
    python quantize_int8_mixed.py --legacy  # old channel-threshold mode
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
                mses.append(mse)

        sensitivities[name] = np.mean(mses)
        h.remove()

        if (idx + 1) % 32 == 0 or idx + 1 == len(layers):
            print(f"    {idx + 1}/{len(layers)} done...")

    return sensitivities


def select_w8a8_layers(sensitivities, threshold=1e-6):
    """Select layers for W8A8 based on sensitivity threshold.

    Layers whose individual W8A8 quantization causes output MSE < threshold
    are safe for W8A8.  The rest stay W8A16 (weight-only).

    Returns (w8a8_names, w8a16_names) sorted by sensitivity.
    """
    sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1])

    w8a8 = []
    w8a16 = []
    for name, mse in sorted_layers:
        if mse <= threshold:
            w8a8.append(name)
        else:
            w8a16.append(name)

    return w8a8, w8a16


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized mixed INT8 quantization for HDRTVNet++ "
                    "(sensitivity-based with asymmetric activation quant)"
    )
    parser.add_argument("--model", default="src/models/weights/Ensemble_AGCM_LE.pth",
                        help="Path to FP32 .pth weights")
    parser.add_argument("--output",
                        default="src/models/weights/Ensemble_AGCM_LE_int8_mixed.pt",
                        help="Output path for quantized checkpoint")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision for runtime dequantization")
    parser.add_argument("--save-fp16", action="store_true",
                        help="Save checkpoint with FP16 compute buffers even "
                             "if calibration ran in FP32 (useful for CPU runs)")
    parser.add_argument("--sensitivity-threshold", type=float, default=1e-6,
                        help="Max per-layer MSE for W8A8 assignment. "
                             "Lower = more conservative (fewer W8A8 layers)")
    parser.add_argument("--num-sensitivity", type=int, default=8,
                        help="Images for sensitivity analysis")
    parser.add_argument("--calibration-dir", default="dataset/test_sdr",
                        help="Directory of SDR images for calibration")
    parser.add_argument("--hg-weights",
                        default="Source Pipeline/pretrained_models/HG_weights.pth",
                        help="Path to HG weights (HG_weights.pth)")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"],
                        help="Use HG refinement (1) or base AGCM+LE only (0)")
    parser.add_argument("--num-calibrate", type=int, default=16,
                        help="Number of images for activation scale calibration")
    parser.add_argument("--num-validate", type=int, default=8,
                        help="Number of images for quality validation")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (auto = GPU if available)")
    parser.add_argument("--full-cpu", action="store_true",
                        help="Run all phases on CPU (avoids ROCm issues)")
    parser.add_argument("--sensitivity-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for sensitivity sweep (auto: "
                             "use CPU on ROCm to avoid MIOpen issues)")
    parser.add_argument("--calibration-device", default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for activation calibration (auto: "
                             "use CPU on ROCm to avoid MIOpen issues)")
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

    sens_device = device
    if args.sensitivity_device != "auto":
        sens_device = torch.device(args.sensitivity_device)
    else:
        if device.type == "cuda" and torch.version.hip is not None:
            sens_device = torch.device("cpu")
            print("  [info] ROCm detected: running sensitivity on CPU "
                  "to avoid MIOpen kernel issues.")

    sens_dtype = compute_dtype if sens_device.type != "cpu" else torch.float32

    calib_device = device
    if args.calibration_device != "auto":
        calib_device = torch.device(args.calibration_device)
    else:
        if device.type == "cuda" and torch.version.hip is not None:
            calib_device = torch.device("cpu")
            print("  [info] ROCm detected: running calibration on CPU "
                  "to avoid MIOpen kernel issues.")

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
    if args.legacy:
        # v1: simple channel-threshold heuristic
        print(f"\nLegacy mixed INT8 (channel threshold = {args.channel_threshold}):")
        _quantize_model_mixed(model, compute_dtype,
                              channel_threshold=args.channel_threshold)
        w8a8_layers = None
        use_asymmetric = False
    else:
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
        sens_inputs = [prepare_model_input(img, sens_device, sens_dtype)
                       for img in calib_images[:args.num_sensitivity]]

        t0 = time.perf_counter()
        try:
            sensitivities = compute_layer_sensitivity(
                model, sens_inputs, args.num_sensitivity
            )
        except RuntimeError as exc:
            if sens_device.type != "cpu":
                print("  [warn] Sensitivity sweep failed on GPU; "
                      "retrying on CPU.")
                sens_device = torch.device("cpu")
                sens_dtype = torch.float32
                model = model.to(dtype=sens_dtype, device=sens_device)
                sens_inputs = [prepare_model_input(img, sens_device, sens_dtype)
                               for img in calib_images[:args.num_sensitivity]]
                sensitivities = compute_layer_sensitivity(
                    model, sens_inputs, args.num_sensitivity
                )
            else:
                raise exc
        dt = time.perf_counter() - t0
        print(f"  Sweep took {dt:.1f}s")

        # Select layers
        w8a8_layers, w8a16_layers = select_w8a8_layers(
            sensitivities, threshold=args.sensitivity_threshold
        )

        # Print sensitivity results
        print(f"\n{'='*60}")
        print("Phase 2: Layer assignment")
        print(f"{'='*60}")
        sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1])
        print(f"\n  Per-layer sensitivity (MSE, ascending):")
        for name, mse in sorted_sens:
            tag = "W8A8" if name in set(w8a8_layers) else "W8A16"
            psnr_str = (f"{-10 * np.log10(mse):.1f} dB"
                        if mse > 1e-10 else ">100 dB")
            print(f"    [{tag:5s}] {name:55s}  MSE={mse:.2e}  PSNR={psnr_str}")

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
    print(f"\nCalibrating activation scales ({args.num_calibrate} images) ...")
    calib_images = load_calibration_images(
        args.calibration_dir, args.num_calibrate
    )
    print(f"  Loaded {len(calib_images)} calibration images")

    # Calibration can be run on a different device (CPU on ROCm)
    if calib_device != device or calib_dtype != compute_dtype:
        model = model.to(dtype=calib_dtype, device=calib_device)
        model.eval()

    calib_inputs = [prepare_model_input(img, calib_device, calib_dtype)
                    for img in calib_images]
    calibrate_w8a8(model, calib_inputs)

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
        save_data["sensitivity_threshold"] = args.sensitivity_threshold

    torch.save(save_data, args.output)

    orig_kb = os.path.getsize(args.model) / 1024
    quant_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved -> {args.output}")
    print(f"  Original : {orig_kb:,.1f} KB")
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
