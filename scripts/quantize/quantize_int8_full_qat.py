"""
Quantization-Aware Training (QAT) for Full INT8 (W8A8) HDRTVNet++.

Starts from a PTQ W8A8 checkpoint and fine-tunes with fake quantization
in the forward pass so the model learns to compensate for quantization
error. Uses Straight-Through Estimator (STE) for gradient flow through
the non-differentiable round/clamp operations.

The output checkpoint is fully compatible with the existing inference
loader " same format as quantize_int8_full.py produces.

Usage
-----
    # Default: fine-tune from PTQ checkpoint
    python scripts/quantize/quantize_int8_full_qat.py

    # Custom epochs / learning rate
    python scripts/quantize/quantize_int8_full_qat.py --epochs 10 --lr 1e-5

    # Start fresh (PTQ + QAT from FP32, no existing full checkpoint needed)
    python scripts/quantize/quantize_int8_full_qat.py --from-scratch
"""

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
# Fake-quantization with STE (Straight-Through Estimator)
# ===================================================================

class _FakeQuantizeSTE(torch.autograd.Function):
    """Round-to-nearest with STE: forward quantizes, backward passes through."""

    @staticmethod
    def forward(ctx, x, scale, bits=8):
        qmin, qmax = -128, 127
        x_q = (x / scale).round().clamp(qmin, qmax)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _FakeQuantizeAsymSTE(torch.autograd.Function):
    """Asymmetric fake quantization with STE: unsigned [0, 255] with zero-point."""

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        x_q = ((x - zero_point) / scale).round().clamp(0, 255)
        return x_q * scale + zero_point

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quantize(x, scale):
    return _FakeQuantizeSTE.apply(x, scale)


def fake_quantize_asymmetric(x, scale, zero_point):
    return _FakeQuantizeAsymSTE.apply(x, scale, zero_point)


# ===================================================================
# QAT wrapper layers " W8A8 with learnable scales
# ===================================================================

class QATConv2d(nn.Module):
    """QAT Conv2d: fake-quantized weights + activations (W8A8)."""

    def __init__(self, conv_or_w8, compute_dtype=torch.float16,
                 asymmetric=False):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.asymmetric = asymmetric

        if isinstance(conv_or_w8, W8A8Conv2d):
            mod = conv_or_w8
            self.in_channels = mod.in_channels
            self.out_channels = mod.out_channels
            self.kernel_size = mod.kernel_size
            self.stride = mod.stride
            self.padding = mod.padding
            self.dilation = mod.dilation
            self.groups = mod.groups

            w_scale = mod.w_scale.float()
            w_int8 = mod.weight_int8
            w_float = w_int8.float() * w_scale.view(-1, 1, 1, 1)
            self.weight = nn.Parameter(w_float)
            self.bias = nn.Parameter(mod.bias.float().clone()) if mod.bias is not None else None

            self.w_scale = nn.Parameter(w_scale.clone())
            self.x_scale = nn.Parameter(mod.x_scale.float().clone())
            if hasattr(mod, "x_zero"):
                self.x_zero = nn.Parameter(mod.x_zero.float().clone())
            else:
                self.x_zero = nn.Parameter(torch.tensor(0.0))

        elif isinstance(conv_or_w8, nn.Conv2d):
            conv = conv_or_w8
            self.in_channels = conv.in_channels
            self.out_channels = conv.out_channels
            self.kernel_size = conv.kernel_size
            self.stride = conv.stride
            self.padding = conv.padding
            self.dilation = conv.dilation
            self.groups = conv.groups

            self.weight = nn.Parameter(conv.weight.data.float())
            self.bias = nn.Parameter(conv.bias.data.float()) if conv.bias is not None else None

            w_flat = self.weight.data.reshape(self.weight.shape[0], -1)
            w_sc = w_flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            self.w_scale = nn.Parameter(w_sc)
            self.x_scale = nn.Parameter(torch.tensor(1.0))
            self.x_zero = nn.Parameter(torch.tensor(0.0))
        else:
            raise TypeError(f"Unsupported module type: {type(conv_or_w8)}")

    def forward(self, x):
        w_scale = self.w_scale.abs().clamp(min=1e-8)
        w_fq = fake_quantize(self.weight, w_scale.view(-1, 1, 1, 1))
        w_fq = w_fq.to(x.dtype)

        if self.asymmetric:
            x_scale = self.x_scale.abs().clamp(min=1e-8).to(x.dtype)
            x_zp = self.x_zero.to(x.dtype)
            x = fake_quantize_asymmetric(x, x_scale, x_zp)
        else:
            x_scale = self.x_scale.abs().clamp(min=1e-8).to(x.dtype)
            x = fake_quantize(x, x_scale)

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv2d(x, w_fq, bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QATLinear(nn.Module):
    """QAT Linear: fake-quantized weights + activations (W8A8)."""

    def __init__(self, linear_or_w8, compute_dtype=torch.float16,
                 asymmetric=False):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.asymmetric = asymmetric

        if isinstance(linear_or_w8, W8A8Linear):
            mod = linear_or_w8
            self.in_features = mod.in_features
            self.out_features = mod.out_features

            w_scale = mod.w_scale.float()
            w_float = mod.weight_int8.float() * w_scale.view(-1, 1)
            self.weight = nn.Parameter(w_float)
            self.bias = nn.Parameter(mod.bias.float().clone()) if mod.bias is not None else None

            self.w_scale = nn.Parameter(w_scale.clone())
            self.x_scale = nn.Parameter(mod.x_scale.float().clone())
            if hasattr(mod, "x_zero"):
                self.x_zero = nn.Parameter(mod.x_zero.float().clone())
            else:
                self.x_zero = nn.Parameter(torch.tensor(0.0))

        elif isinstance(linear_or_w8, nn.Linear):
            linear = linear_or_w8
            self.in_features = linear.in_features
            self.out_features = linear.out_features

            self.weight = nn.Parameter(linear.weight.data.float())
            self.bias = nn.Parameter(linear.bias.data.float()) if linear.bias is not None else None

            w_sc = self.weight.data.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            self.w_scale = nn.Parameter(w_sc)
            self.x_scale = nn.Parameter(torch.tensor(1.0))
            self.x_zero = nn.Parameter(torch.tensor(0.0))
        else:
            raise TypeError(f"Unsupported module type: {type(linear_or_w8)}")

    def forward(self, x):
        w_scale = self.w_scale.abs().clamp(min=1e-8)
        w_fq = fake_quantize(self.weight, w_scale.view(-1, 1))
        w_fq = w_fq.to(x.dtype)

        if self.asymmetric:
            x_scale = self.x_scale.abs().clamp(min=1e-8).to(x.dtype)
            x_zp = self.x_zero.to(x.dtype)
            x = fake_quantize_asymmetric(x, x_scale, x_zp)
        else:
            x_scale = self.x_scale.abs().clamp(min=1e-8).to(x.dtype)
            x = fake_quantize(x, x_scale)

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_fq, bias)


# ===================================================================
# Model conversion: PTQ ' QAT ' PTQ
# ===================================================================

def convert_to_qat(model):
    """Replace W8A8* layers with QAT layers (learnable fake-quant)."""
    converted = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if len(parts) == 0 or parts[0] == "":
            continue
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if isinstance(module, W8A8Conv2d):
            setattr(parent, parts[-1],
                    QATConv2d(module, asymmetric=module.is_asymmetric))
            converted += 1
        elif isinstance(module, W8A8Linear):
            setattr(parent, parts[-1],
                    QATLinear(module, asymmetric=module.is_asymmetric))
            converted += 1

    print(f"  Converted {converted} layers to QAT mode")
    return model


def convert_qat_to_ptq(model, compute_dtype=torch.float16):
    """Freeze QAT layers back to W8A8* for inference."""
    converted = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if len(parts) == 0 or parts[0] == "":
            continue
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if isinstance(module, QATConv2d):
            tmp_conv = nn.Conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups,
                bias=(module.bias is not None),
            )
            tmp_conv.weight.data = module.weight.data.float()
            if module.bias is not None:
                tmp_conv.bias.data = module.bias.data.float()

            new_mod = W8A8Conv2d(tmp_conv, compute_dtype,
                                 asymmetric=module.asymmetric)
            new_mod.x_scale.fill_(module.x_scale.abs().clamp(min=1e-8).item())
            if module.asymmetric:
                new_mod.x_zero.fill_(module.x_zero.item())

            setattr(parent, parts[-1], new_mod)
            converted += 1

        elif isinstance(module, QATLinear):
            tmp_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            tmp_linear.weight.data = module.weight.data.float()
            if module.bias is not None:
                tmp_linear.bias.data = module.bias.data.float()

            new_mod = W8A8Linear(tmp_linear, compute_dtype,
                                 asymmetric=module.asymmetric)
            new_mod.x_scale.fill_(module.x_scale.abs().clamp(min=1e-8).item())
            if module.asymmetric:
                new_mod.x_zero.fill_(module.x_zero.item())

            setattr(parent, parts[-1], new_mod)
            converted += 1

    print(f"  Converted {converted} QAT layers back to PTQ for inference")
    return model


# ===================================================================
# Dataset helpers
# ===================================================================

def load_image_pair(sdr_path, hdr_path, max_long_edge=960):
    """Load an SDR/HDR image pair, resize, return as float32 tensors."""
    sdr = cv2.imread(sdr_path, cv2.IMREAD_UNCHANGED)
    hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
    if sdr is None or hdr is None:
        return None, None

    h, w = sdr.shape[:2]
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
        sdr = cv2.resize(sdr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        hdr = cv2.resize(hdr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    sdr = cv2.cvtColor(sdr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sdr_t = torch.from_numpy(np.transpose(sdr, (2, 0, 1))).unsqueeze(0)

    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    hdr_t = torch.from_numpy(np.transpose(hdr, (2, 0, 1))).unsqueeze(0)

    return sdr_t, hdr_t


def random_crop_pair(sdr_t, hdr_t, crop_size=256):
    """Random crop from paired SDR/HDR tensors (both same spatial dims)."""
    _, _, h, w = sdr_t.shape
    if h <= crop_size and w <= crop_size:
        return sdr_t, hdr_t
    top = random.randint(0, max(0, h - crop_size))
    left = random.randint(0, max(0, w - crop_size))
    sdr_c = sdr_t[:, :, top:top + crop_size, left:left + crop_size]
    hdr_c = hdr_t[:, :, top:top + crop_size, left:left + crop_size]
    return sdr_c, hdr_c


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
# Model helpers
# ===================================================================

def _build_fp32_model(model_path: str, hg_weights: str, use_hg: bool) -> nn.Module:
    if use_hg:
        model = HG_Composite(
            classifier="color_condition", cond_c=6, in_nc=3, out_nc=3,
            nf=32, act_type="relu", weighting_network=False,
            hg_nf=64, mask_r=0.75,
        )
    else:
        model = Ensemble_AGCM_LE(
            classifier="color_condition", cond_c=6, in_nc=3, out_nc=3,
            nf=32, act_type="relu", weighting_network=False,
        )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    if use_hg:
        model.base.load_state_dict(cleaned, strict=True)
        hg_state = torch.load(hg_weights, map_location="cpu")
        if isinstance(hg_state, dict) and "state_dict" in hg_state:
            hg_state = hg_state["state_dict"]
        model.hg.load_state_dict(hg_state, strict=True)
    else:
        model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


# ===================================================================
# Training loop
# ===================================================================

def train_qat(model, pairs, device, compute_dtype, args):
    """Fine-tune with QAT fake-quantization."""
    model.train()
    # ROCm: BatchNorm training can trigger MIOpen HIPRTC compile failures.
    # Freeze BN to inference behavior for stability while keeping QAT active.
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    qat_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(qat_params, lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(pairs), eta_min=args.lr * 0.01
    )

    criterion = nn.L1Loss()
    monitor_count = min(max(args.num_validate, 0), len(pairs))
    monitor_pairs = pairs[:monitor_count]

    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}, "
          f"crop={args.crop_size}, {len(pairs)} pairs")
    if monitor_pairs:
        print(f"  Model selection: best monitor L1 on {len(monitor_pairs)} "
              f"fixed full-image pairs")
    else:
        print("  Model selection: best training loss")

    best_train_loss = float("inf")
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    monitor_enabled = bool(monitor_pairs)

    def compute_monitor_l1():
        nonlocal monitor_enabled
        if not monitor_pairs or not monitor_enabled:
            return None
        losses = []
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for sdr_t, hdr_t in monitor_pairs:
                    inp = prepare_model_input(sdr_t, device, compute_dtype)
                    target = hdr_t.to(device=device, dtype=compute_dtype)
                    out, _ = model(inp)
                    if not torch.isfinite(out).all():
                        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                    loss = criterion(out, target)
                    if torch.isfinite(loss):
                        losses.append(loss.item())
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "miopen" in msg or "hiprtc" in msg:
                print("  [warn] Monitor pass disabled due ROCm MIOpen error; "
                      "falling back to training-loss selection.")
                monitor_enabled = False
                return None
            raise
        finally:
            if was_training:
                model.train()
                if hasattr(torch.version, "hip") and torch.version.hip is not None:
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
        if not losses:
            return None
        return float(np.mean(losses))

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        epoch_loss = 0.0
        num_batches = 0

        for sdr_t, hdr_t in pairs:
            sdr_crop, hdr_crop = random_crop_pair(sdr_t, hdr_t, args.crop_size)

            inp = prepare_model_input(sdr_crop, device, compute_dtype)
            target = hdr_crop.to(device=device, dtype=compute_dtype)

            output, _ = model(inp)
            if not torch.isfinite(output).all():
                output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
            loss = criterion(output, target)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        best_train_loss = min(best_train_loss, avg_loss)
        monitor_l1 = compute_monitor_l1()
        score = monitor_l1 if monitor_l1 is not None else avg_loss

        if monitor_l1 is None:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.6f}, "
                  f"lr={lr_now:.2e}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.6f}, "
                  f"monitor_l1={monitor_l1:.6f}, lr={lr_now:.2e}")

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    print(f"  Best training loss: {best_train_loss:.6f}")
    if monitor_pairs and monitor_enabled:
        print(f"  Selected epoch   : {best_epoch} "
              f"(best monitor L1={best_score:.6f})")
    else:
        print(f"  Selected epoch   : {best_epoch} "
              f"(best training loss)")
    return model


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning for Full INT8 (W8A8) HDRTVNet++"
    )
    parser.add_argument("--ptq-checkpoint",
                        default=os.path.join(
                            _REPO_ROOT,
                            "src",
                            "models",
                            "weights",
                            "Ensemble_AGCM_LE_int8_full.pt",
                        ),
                        help="PTQ full checkpoint to start from")
    parser.add_argument("--fp32-model",
                        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "Ensemble_AGCM_LE.pth"),
                        help="FP32 model (for --from-scratch or validation)")
    parser.add_argument("--output",
                        default=os.path.join(
                            _REPO_ROOT,
                            "src",
                            "models",
                            "weights",
                            "Ensemble_AGCM_LE_int8_full_qat.pt",
                        ),
                        help="Output path for QAT-finetuned checkpoint")
    parser.add_argument(
        "--sdr-dir",
        default=os.path.join(_REPO_ROOT, "dataset", "train_sdr"),
                        help="SDR images directory")
    parser.add_argument(
        "--hdr-dir",
        default=os.path.join(_REPO_ROOT, "dataset", "train_hdr"),
                        help="HDR ground-truth directory")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Start from FP32 model (PTQ + calibrate + QAT)")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision")
    parser.add_argument("--hg-weights",
                        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "HG_weights.pth"),
                        help="Path to HG weights (HG_weights.pth)")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"],
                        help="Use HG refinement (1) or base AGCM+LE only (0)")
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

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Random crop size for training patches")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max training images (0 = all)")
    parser.add_argument("--max-long-edge", type=int, default=960,
                        help="Resize images so longest edge <= this")

    # Validation
    parser.add_argument("--num-validate", type=int, default=8,
                        help="Number of images for quality validation")
    parser.add_argument("--num-calibrate", type=int, default=0,
                        help="Calibration images when --from-scratch (0 = all)")

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Device: {device}, Precision: {args.precision}")

    use_hg = str(args.use_hg).strip() != "0"

    # ------------------------------------------------------------------
    # 1. Load model (either from PTQ checkpoint or from scratch)
    # ------------------------------------------------------------------
    if args.from_scratch:
        print(f"\nLoading FP32 model from {args.fp32_model} ...")
        model = _build_fp32_model(args.fp32_model, args.hg_weights, use_hg)

        use_asym = args.activation_quant == "asymmetric"
        print("Applying full W8A8 PTQ ...")
        _quantize_model_w8a8(model, compute_dtype, asymmetric=use_asym)
        model = model.to(dtype=compute_dtype, device=device)
        model.eval()

        sdr_paths = sorted(glob.glob(os.path.join(args.sdr_dir, "*.png")))
        calib_paths = sdr_paths if args.num_calibrate <= 0 else sdr_paths[:args.num_calibrate]
        print(f"Calibrating ({len(calib_paths)} images) ...")
        calib_tensors = []
        for p in calib_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            longest = max(h, w)
            if longest > args.max_long_edge:
                sc = args.max_long_edge / longest
                new_w = int(round(w * sc / 8)) * 8
                new_h = int(round(h * sc / 8)) * 8
            else:
                new_w = int(round(w / 8)) * 8
                new_h = int(round(h / 8)) * 8
            new_w, new_h = max(new_w, 8), max(new_h, 8)
            if new_w != w or new_h != h:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)
            calib_tensors.append(t)
        if len(calib_tensors) == 0:
            raise RuntimeError(
                "No valid calibration images loaded for --from-scratch"
            )

        calib_inputs = [prepare_model_input(t, device, compute_dtype)
                        for t in calib_tensors]
        calibrate_w8a8(
            model, calib_inputs,
            method=args.calibration_method,
            percentile=args.percentile,
            percentile_low=args.percentile_low,
            max_samples=args.max_calib_samples,
        )
        activation_quant = args.activation_quant
        calibration_method = args.calibration_method
        calibration_percentile = args.percentile
        calibration_percentile_low = args.percentile_low
    else:
        print(f"\nLoading PTQ checkpoint from {args.ptq_checkpoint} ...")
        checkpoint = torch.load(args.ptq_checkpoint, map_location="cpu",
                                weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"{args.ptq_checkpoint} is not a valid checkpoint")

        arch = checkpoint.get("architecture", {})
        use_hg = bool(arch.get("use_hg", True))
        compute_dtype_str = checkpoint.get("compute_dtype", "torch.float16")
        compute_dtype = torch.float16 if "16" in compute_dtype_str else torch.float32

        activation_quant = checkpoint.get("activation_quant", "symmetric")
        use_asym = activation_quant == "asymmetric"
        calibration_method = checkpoint.get("calibration_method", "max")
        calibration_percentile = checkpoint.get("calibration_percentile", 99.9)
        calibration_percentile_low = checkpoint.get("calibration_percentile_low", 0.1)

        if use_hg:
            model = HG_Composite(
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
        else:
            model = Ensemble_AGCM_LE(
                classifier=arch.get("classifier", "color_condition"),
                cond_c=arch.get("cond_c", 6),
                in_nc=arch.get("in_nc", 3),
                out_nc=arch.get("out_nc", 3),
                nf=arch.get("nf", 32),
                act_type=arch.get("act_type", "relu"),
                weighting_network=arch.get("weighting_network", False),
            )

        _quantize_model_w8a8(model, compute_dtype, asymmetric=use_asym)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(dtype=compute_dtype, device=device)
        model.eval()
        print(f"  Loaded ({checkpoint.get('quantization', '?')})")

    # ------------------------------------------------------------------
    # 2. Load SDR/HDR pairs for training
    # ------------------------------------------------------------------
    print(f"\nLoading SDR/HDR pairs from {args.sdr_dir} + {args.hdr_dir} ...")
    sdr_paths = sorted(glob.glob(os.path.join(args.sdr_dir, "*.png")))
    hdr_paths = sorted(glob.glob(os.path.join(args.hdr_dir, "*.png")))

    if not sdr_paths or not hdr_paths:
        raise FileNotFoundError("No PNG images found in SDR/HDR directories")

    sdr_map = {os.path.basename(p): p for p in sdr_paths}
    hdr_map = {os.path.basename(p): p for p in hdr_paths}
    common = sorted(set(sdr_map.keys()) & set(hdr_map.keys()))
    if not common:
        raise FileNotFoundError("No matching filenames between SDR and HDR dirs")

    if args.max_images > 0:
        common = common[:args.max_images]

    pairs = []
    for name in common:
        sdr_t, hdr_t = load_image_pair(
            sdr_map[name], hdr_map[name], args.max_long_edge
        )
        if sdr_t is not None:
            pairs.append((sdr_t, hdr_t))
    if len(pairs) == 0:
        raise RuntimeError(
            "Matched filenames found, but no valid SDR/HDR pairs could be loaded"
        )

    print(f"  {len(pairs)} pairs loaded")

    # ------------------------------------------------------------------
    # 3. Convert to QAT mode
    # ------------------------------------------------------------------
    print("\nConverting to QAT mode ...")
    model = convert_to_qat(model)
    model = model.float()
    model = model.to(device)

    # ------------------------------------------------------------------
    # 4. QAT fine-tuning
    # ------------------------------------------------------------------
    train_dtype = torch.float32
    t0 = time.perf_counter()
    model = train_qat(model, pairs, device, train_dtype, args)
    dt = time.perf_counter() - t0
    print(f"  Training took {dt:.1f}s ({dt / 60:.1f} min)")

    # ------------------------------------------------------------------
    # 5. Convert back to PTQ format for inference
    # ------------------------------------------------------------------
    print("\nConverting back to PTQ format ...")
    model.eval()
    model = convert_qat_to_ptq(model, compute_dtype)
    model = model.to(dtype=compute_dtype, device=device)
    model.eval()

    # ------------------------------------------------------------------
    # 6. Save checkpoint (same format as quantize_int8_full.py)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = {
        "state_dict": model.state_dict(),
        "compute_dtype": str(compute_dtype),
        "quantization": "w8a8_full",
        "activation_quant": activation_quant,
        "calibration_method": calibration_method,
        "calibration_percentile": calibration_percentile,
        "calibration_percentile_low": calibration_percentile_low,
        "qat_epochs": args.epochs,
        "qat_lr": args.lr,
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

    orig_kb = os.path.getsize(args.fp32_model) / 1024
    quant_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved -> {args.output}")
    print(f"  Original : {orig_kb:,.1f} KB")
    print(f"  Quantized: {quant_kb:,.1f} KB")
    if quant_kb > 0:
        print(f"  Ratio    : {orig_kb / quant_kb:.2f}x")

    # ------------------------------------------------------------------
    # 7. Quality validation " FP16 reference vs QAT INT8
    # ------------------------------------------------------------------
    print(f"\nValidation ({device}, {args.precision}):")

    ref_model = _build_fp32_model(args.fp32_model, args.hg_weights, use_hg)
    ref_model = ref_model.to(dtype=compute_dtype, device=device).eval()

    val_pairs = pairs[:args.num_validate]
    print(f"  {len(val_pairs)} images")

    psnrs = []
    with torch.inference_mode():
        for i, (sdr_t, _) in enumerate(val_pairs):
            inp = prepare_model_input(sdr_t, device, compute_dtype)
            ref_out, _ = ref_model(inp)
            qat_out, _ = model(inp)

            mse = ((ref_out.float() - qat_out.float()) ** 2).mean().item()
            psnr = -10 * np.log10(mse + 1e-10)
            psnrs.append(psnr)
            print(f"  Image {i + 1:3d}: PSNR = {psnr:.2f} dB")

    avg_psnr = np.mean(psnrs)
    print(f"  Average  : {avg_psnr:.2f} dB")

    # ------------------------------------------------------------------
    # 8. Quick speed test
    # ------------------------------------------------------------------
    if device.type == "cuda":
        print(f"\nSpeed test ({device}):")
        test_sdr = pairs[0][0]
        test_inp = prepare_model_input(test_sdr, device, compute_dtype)
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




