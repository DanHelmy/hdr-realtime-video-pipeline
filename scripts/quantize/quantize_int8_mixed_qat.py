"""
Quantization-Aware Training (QAT) for Mixed INT8 HDRTVNet++.

Starts from a PTQ mixed checkpoint and fine-tunes with fake quantization
in the forward pass so the model learns to compensate for quantization
error.  Uses Straight-Through Estimator (STE) for gradient flow through
the non-differentiable round/clamp operations.

The output checkpoint is fully compatible with the existing inference
loader " same format as quantize_int8_mixed.py produces.

Usage
-----
    # Default: fine-tune from PTQ checkpoint
    python scripts/quantize/quantize_int8_mixed_qat.py

    # Custom epochs / learning rate
    python scripts/quantize/quantize_int8_mixed_qat.py --epochs 10 --lr 1e-5

    # Start fresh (PTQ + QAT from FP32, no existing mixed checkpoint needed)
    python scripts/quantize/quantize_int8_mixed_qat.py --from-scratch
"""

import argparse
import copy
import glob
import math
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
    W8A8Conv2d, W8A8Linear, W8Conv2d, W8Linear,
    _predequantize_conv, _predequantize_linear,
    _quantize_model_mixed, _quantize_model_mixed_v2, calibrate_w8a8,
)
from gui_objective_metrics import (
    _crop_shared_black_borders,
    _delta_e_itp_absolute_rgb,
    _delta_e_itp_bgr,
    _grade_normalize_absolute_rgb_to_ref,
    _grade_normalize_pred_to_ref,
    _linear_bgr_to_absolute_rgb,
    _prepare_metric_pair,
    _psnr_bgr,
    _ssim_bgr,
)


def configure_reproducibility(args):
    """Seed Python/NumPy/PyTorch and prefer deterministic kernels."""
    seed = int(args.seed)
    deterministic = str(args.deterministic).strip() != "0"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
        except Exception:
            pass

    try:
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None and hasattr(cuda_backends, "matmul"):
            cuda_backends.matmul.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass

    return seed, deterministic


def _expand_base_optional_layer_names(*names):
    expanded = []
    for name in names:
        if name.startswith("base."):
            expanded.append(name)
            expanded.append(name[len("base."):])
        elif name.startswith("hg."):
            expanded.append(name)
        else:
            expanded.append(name)
            expanded.append(f"base.{name}")
    return tuple(dict.fromkeys(expanded))


AGCM_CONTROL_LAYER_NAMES = _expand_base_optional_layer_names(
    "AGCM.cond_scale_first",
    "AGCM.cond_shift_first",
    "AGCM.cond_scale_HR",
    "AGCM.cond_shift_HR",
    "AGCM.cond_scale_last",
    "AGCM.cond_shift_last",
)

FP16_SENSITIVE_LAYER_NAMES = _expand_base_optional_layer_names(
    *AGCM_CONTROL_LAYER_NAMES,
    "AGCM.classifier.model.0",
    "AGCM.classifier.model.4",
    "AGCM.classifier.model.8",
    "AGCM.classifier.model.12",
    "AGCM.classifier.model.16",
    "AGCM.classifier.model.20",
    "AGCM.conv_first",
    "AGCM.HRconv",
    "AGCM.conv_last",
    "LE.cond_first.0",
    "LE.cond_first.2",
    "LE.cond_first.4",
    "LE.HR_conv1",
    "LE.HR_conv2",
    "LE.conv_last",
    "hg.conv1.0",
    "hg.conv10",
    "hg.conv_last",
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
        # STE: gradient passes through unchanged
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
    """Apply symmetric fake quantization with STE gradient."""
    return _FakeQuantizeSTE.apply(x, scale)


def fake_quantize_asymmetric(x, scale, zero_point):
    """Apply asymmetric fake quantization with STE gradient."""
    return _FakeQuantizeAsymSTE.apply(x, scale, zero_point)


# ===================================================================
# QAT wrapper layers " same interface as W8A8/W8 but with learnable scales
# ===================================================================

class QATConv2d(nn.Module):
    """QAT Conv2d: fake-quantized weights + optional fake-quantized activations.

    If `quantize_activations=True`, acts like W8A8Conv2d (both weight + act quantized).
    If `quantize_activations=False`, acts like W8Conv2d (weight-only).
    If `asymmetric=True`, activations use unsigned [0, 255] with zero-point.

    Weight scale and activation scale are learnable parameters.
    The actual weights are stored in FP16/FP32 and quantized in the forward pass.
    """

    def __init__(self, conv_or_w8, compute_dtype=torch.float16,
                 quantize_activations=False, asymmetric=False):
        super().__init__()
        self.quantize_activations = quantize_activations
        self.compute_dtype = compute_dtype
        self.asymmetric = asymmetric

        # Extract original conv parameters from either nn.Conv2d or W8*/W8A8*
        if isinstance(conv_or_w8, (W8A8Conv2d, W8Conv2d)):
            # Reconstruct float weights from INT8 + scale
            mod = conv_or_w8
            self.in_channels = mod.in_channels
            self.out_channels = mod.out_channels
            self.kernel_size = mod.kernel_size
            self.stride = mod.stride
            self.padding = mod.padding
            self.dilation = mod.dilation
            self.groups = mod.groups

            # Get weight scale
            if isinstance(mod, W8A8Conv2d):
                w_scale = mod.w_scale.float()
                w_int8 = mod.weight_int8
            else:
                w_scale = mod.scale.float()
                w_int8 = mod.weight_int8

            # Reconstruct FP32 weights
            w_float = w_int8.float() * w_scale.view(-1, 1, 1, 1)
            self.weight = nn.Parameter(w_float)

            if mod.bias is not None:
                self.bias = nn.Parameter(mod.bias.float().clone())
            else:
                self.bias = None

            # Initial weight scale (learnable)
            self.w_scale = nn.Parameter(w_scale.clone())

            # Activation scale (learnable, only used if quantize_activations)
            if isinstance(mod, W8A8Conv2d):
                self.x_scale = nn.Parameter(mod.x_scale.float().clone())
                if hasattr(mod, 'x_zero'):
                    self.x_zero = nn.Parameter(mod.x_zero.float().clone())
                else:
                    self.x_zero = nn.Parameter(torch.tensor(0.0))
            else:
                self.x_scale = nn.Parameter(torch.tensor(1.0))
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
            if conv.bias is not None:
                self.bias = nn.Parameter(conv.bias.data.float())
            else:
                self.bias = None

            # Compute initial weight scale
            w_flat = self.weight.data.reshape(self.weight.shape[0], -1)
            w_sc = w_flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            self.w_scale = nn.Parameter(w_sc)
            self.x_scale = nn.Parameter(torch.tensor(1.0))
            self.x_zero = nn.Parameter(torch.tensor(0.0))
        else:
            raise TypeError(f"Unsupported module type: {type(conv_or_w8)}")

    def forward(self, x):
        # Fake-quantize weights
        w_scale = self.w_scale.abs().clamp(min=1e-8)
        w_fq = fake_quantize(self.weight, w_scale.view(-1, 1, 1, 1))
        w_fq = w_fq.to(x.dtype)

        # Optionally fake-quantize activations
        if self.quantize_activations:
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
    """QAT Linear: fake-quantized weights + optional fake-quantized activations.

    If `quantize_activations=True`, acts like W8A8Linear (both weight + act).
    If `quantize_activations=False`, acts like W8Linear (weight-only).
    If `asymmetric=True`, activations use unsigned [0, 255] with zero-point.
    """

    def __init__(self, linear_or_w8, compute_dtype=torch.float16,
                 quantize_activations=False, asymmetric=False):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.quantize_activations = quantize_activations
        self.asymmetric = asymmetric

        if isinstance(linear_or_w8, (W8A8Linear, W8Linear)):
            mod = linear_or_w8
            self.in_features = mod.in_features
            self.out_features = mod.out_features

            if isinstance(mod, W8A8Linear):
                w_scale = mod.w_scale.float()
            else:
                w_scale = mod.scale.float()

            w_float = mod.weight_int8.float() * w_scale.view(-1, 1)
            self.weight = nn.Parameter(w_float)

            if mod.bias is not None:
                self.bias = nn.Parameter(mod.bias.float().clone())
            else:
                self.bias = None

            self.w_scale = nn.Parameter(w_scale.clone())

            # Activation scale (learnable, only used if quantize_activations)
            if isinstance(mod, W8A8Linear):
                self.x_scale = nn.Parameter(mod.x_scale.float().clone())
                if hasattr(mod, 'x_zero'):
                    self.x_zero = nn.Parameter(mod.x_zero.float().clone())
                else:
                    self.x_zero = nn.Parameter(torch.tensor(0.0))
            else:
                self.x_scale = nn.Parameter(torch.tensor(1.0))
                self.x_zero = nn.Parameter(torch.tensor(0.0))

        elif isinstance(linear_or_w8, nn.Linear):
            linear = linear_or_w8
            self.in_features = linear.in_features
            self.out_features = linear.out_features

            self.weight = nn.Parameter(linear.weight.data.float())
            if linear.bias is not None:
                self.bias = nn.Parameter(linear.bias.data.float())
            else:
                self.bias = None

            w_sc = self.weight.data.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            self.w_scale = nn.Parameter(w_sc)
            self.x_scale = nn.Parameter(torch.tensor(1.0))
            self.x_zero = nn.Parameter(torch.tensor(0.0))
        else:
            raise TypeError(f"Unsupported module type: {type(linear_or_w8)}")

    def forward(self, x):
        # Fake-quantize weights
        w_scale = self.w_scale.abs().clamp(min=1e-8)
        w_fq = fake_quantize(self.weight, w_scale.view(-1, 1))
        w_fq = w_fq.to(x.dtype)

        # Optionally fake-quantize activations
        if self.quantize_activations:
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
    """Replace W8A8*/W8* layers with QAT layers (learnable fake-quant)."""
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
                    QATConv2d(module, quantize_activations=True,
                              asymmetric=module.is_asymmetric))
            converted += 1
        elif isinstance(module, W8Conv2d):
            setattr(parent, parts[-1],
                    QATConv2d(module, quantize_activations=False))
            converted += 1
        elif isinstance(module, W8A8Linear):
            setattr(parent, parts[-1],
                    QATLinear(module, quantize_activations=True,
                              asymmetric=module.is_asymmetric))
            converted += 1
        elif isinstance(module, W8Linear):
            setattr(parent, parts[-1], QATLinear(module))
            converted += 1

    print(f"  Converted {converted} layers to QAT mode")
    return model


def convert_qat_to_ptq(model, compute_dtype=torch.float16):
    """Freeze QAT layers back to W8A8*/W8* for inference."""
    converted = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        if len(parts) == 0 or parts[0] == "":
            continue
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if isinstance(module, QATConv2d):
            # Create a temporary nn.Conv2d to pass to W8A8Conv2d/W8Conv2d
            tmp_conv = nn.Conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups,
                bias=(module.bias is not None),
            )
            # Use the QAT-tuned weights
            tmp_conv.weight.data = module.weight.data.float()
            if module.bias is not None:
                tmp_conv.bias.data = module.bias.data.float()

            if module.quantize_activations:
                new_mod = W8A8Conv2d(tmp_conv, compute_dtype,
                                     asymmetric=module.asymmetric)
                # Transfer the learned activation scale
                new_mod.x_scale.fill_(module.x_scale.abs().clamp(min=1e-8).item())
                if module.asymmetric:
                    new_mod.x_zero.fill_(module.x_zero.item())
            else:
                new_mod = W8Conv2d(tmp_conv, compute_dtype)

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

            if module.quantize_activations:
                new_mod = W8A8Linear(tmp_linear, compute_dtype,
                                     asymmetric=module.asymmetric)
                new_mod.x_scale.fill_(
                    module.x_scale.abs().clamp(min=1e-8).item())
                if module.asymmetric:
                    new_mod.x_zero.fill_(module.x_zero.item())
            else:
                new_mod = W8Linear(tmp_linear, compute_dtype)

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

    # SDR: uint8 [0,255] ' float [0,1]
    sdr = cv2.cvtColor(sdr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sdr_t = torch.from_numpy(np.transpose(sdr, (2, 0, 1))).unsqueeze(0)

    # HDR: uint16 [0,65535] ' float [0,1]
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
    target_dtype = dtype
    if torch.device(device).type == "cpu" and dtype == torch.float16:
        target_dtype = torch.float32
    img_dev = img_tensor.to(device=device, dtype=target_dtype)
    try:
        cond = F.interpolate(img_dev, scale_factor=0.25, mode="bicubic",
                             align_corners=False, antialias=True)
    except TypeError:
        cond = F.interpolate(img_dev, scale_factor=0.25, mode="bicubic",
                             align_corners=False)
    return (img_dev, cond)


def masked_mean_abs(diff, mask):
    """Masked mean absolute value with safe zero-mask handling."""
    if mask.shape[1] == 1 and diff.shape[1] != 1:
        mask = mask.expand(-1, diff.shape[1], -1, -1)
    weight = mask.to(dtype=diff.dtype)
    denom = weight.sum().clamp(min=1.0)
    return (diff.abs() * weight).sum() / denom


def build_highlight_neutral_mask(target, highlight_threshold, neutral_threshold):
    """Weight bright, near-neutral highlights more strongly."""
    highlight_threshold = float(np.clip(highlight_threshold, 0.0, 0.999999))
    neutral_threshold = max(float(neutral_threshold), 1e-6)
    peak = target.amax(dim=1, keepdim=True)
    floor = target.amin(dim=1, keepdim=True)
    chroma = peak - floor
    highlight_weight = ((peak - highlight_threshold) /
                        max(1.0 - highlight_threshold, 1e-6)).clamp_(0.0, 1.0)
    neutral_weight = ((neutral_threshold - chroma) /
                      neutral_threshold).clamp_(0.0, 1.0)
    return highlight_weight * neutral_weight


def build_rgb_luma(tensor):
    """BT.709 luma used for shadow-sensitive QAT masks and losses."""
    weights = tensor.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (tensor * weights).sum(dim=1, keepdim=True)


def build_dark_area_mask(target, dark_threshold, dark_floor):
    """Weight shadow detail while avoiding pure black borders/letterbox pixels."""
    dark_threshold = float(np.clip(dark_threshold, 1e-6, 1.0))
    dark_floor = float(np.clip(dark_floor, 0.0, max(dark_threshold - 1e-6, 0.0)))
    luma = build_rgb_luma(target)
    dark_weight = ((dark_threshold - luma) /
                   max(dark_threshold - dark_floor, 1e-6)).clamp(0.0, 1.0)
    if dark_floor > 0.0:
        signal_weight = (luma / max(dark_floor, 1e-6)).clamp(0.0, 1.0)
        dark_weight = dark_weight * signal_weight
    return dark_weight


def build_source_chroma_mask(source, args):
    """Select colorful SDR pixels where preserving hue/saturation is meaningful."""
    source_f = source.float().clamp(0.0, 1.0)
    peak = source_f.amax(dim=1, keepdim=True)
    floor = source_f.amin(dim=1, keepdim=True)
    chroma = peak - floor
    sat_thr = max(float(args.source_chroma_saturation_threshold), 1e-6)
    chroma_weight = ((chroma - sat_thr) / sat_thr).clamp(0.0, 1.0)
    luma_floor = max(float(args.source_chroma_luma_floor), 0.0)
    if luma_floor > 0.0:
        luma_weight = (build_rgb_luma(source_f) / luma_floor).clamp(0.0, 1.0)
        chroma_weight = chroma_weight * luma_weight
    return chroma_weight


def build_luma_normalized_chroma(tensor, args):
    """RGB ratios relative to luma preserve hue/chroma without constraining tone."""
    tensor_f = tensor.float().clamp(0.0, 1.0)
    eps = max(float(args.source_chroma_luma_floor), 1e-4)
    ratio_clip = max(float(args.source_chroma_ratio_clip), 1.0)
    luma = build_rgb_luma(tensor_f).clamp_min(eps)
    return (tensor_f / luma).clamp(0.0, ratio_clip)


def compute_tone_protection_coverages(target, args, source=None):
    highlight_mask = build_highlight_neutral_mask(
        target, args.highlight_threshold, args.neutral_threshold
    )
    dark_mask = build_dark_area_mask(
        target, args.dark_threshold, args.dark_floor
    )
    highlight_cov = float(highlight_mask.mean().item())
    dark_cov = float(dark_mask.mean().item())
    source_chroma_cov = 0.0
    if source is not None:
        source_chroma_cov = float(build_source_chroma_mask(source, args).mean().item())
    tone_score = (
        highlight_cov
        + float(args.dark_monitor_weight) * dark_cov
        + float(args.source_chroma_monitor_weight) * source_chroma_cov
    )
    return highlight_cov, dark_cov, source_chroma_cov, tone_score


def build_channel_balance_deltas(tensor):
    """Per-pixel RGB channel deltas; zero means perfectly neutral."""
    red = tensor[:, 0:1]
    green = tensor[:, 1:2]
    blue = tensor[:, 2:3]
    return torch.cat((red - green, green - blue, blue - red), dim=1)


def sample_training_crop_pair(sdr_t, hdr_t, args):
    """Pick a crop, biased toward highlight and shadow-protection coverage."""
    crop_size = int(args.crop_size)
    _, _, h, w = sdr_t.shape
    if h <= crop_size and w <= crop_size:
        return sdr_t, hdr_t

    attempts = max(1, int(getattr(args, "highlight_crop_attempts", 1)))
    best_score = -1.0
    best_crop = None

    for _ in range(attempts):
        top = random.randint(0, max(0, h - crop_size))
        left = random.randint(0, max(0, w - crop_size))
        sdr_c = sdr_t[:, :, top:top + crop_size, left:left + crop_size]
        hdr_c = hdr_t[:, :, top:top + crop_size, left:left + crop_size]
        highlight_cov, dark_cov, source_chroma_cov, _ = compute_tone_protection_coverages(
            hdr_c.float(), args, source=sdr_c.float()
        )
        score = (
            highlight_cov
            + float(args.dark_crop_weight) * dark_cov
            + float(args.source_chroma_crop_weight) * source_chroma_cov
        )
        if score > best_score:
            best_score = score
            best_crop = (sdr_c, hdr_c)

    return best_crop if best_crop is not None else random_crop_pair(
        sdr_t, hdr_t, crop_size
    )


def select_tone_protected_monitor_pairs(pairs, num_pairs, args):
    """Pick monitor pairs with strong highlight or shadow-detail coverage."""
    if not pairs:
        return [], {
            "positive_candidates": 0,
            "highlight_positive_candidates": 0,
            "dark_positive_candidates": 0,
            "source_chroma_positive_candidates": 0,
            "avg_tone_cov": 0.0,
            "min_tone_cov": 0.0,
            "max_tone_cov": 0.0,
            "avg_highlight_cov": 0.0,
            "min_highlight_cov": 0.0,
            "max_highlight_cov": 0.0,
            "avg_dark_cov": 0.0,
            "min_dark_cov": 0.0,
            "max_dark_cov": 0.0,
            "avg_source_chroma_cov": 0.0,
            "min_source_chroma_cov": 0.0,
            "max_source_chroma_cov": 0.0,
        }

    if num_pairs <= 0:
        num_pairs = len(pairs)

    scored = []
    for idx, pair in enumerate(pairs):
        sdr_t, hdr_t = pair
        highlight_cov, dark_cov, source_chroma_cov, tone_cov = compute_tone_protection_coverages(
            hdr_t.float(), args, source=sdr_t.float()
        )
        scored.append((tone_cov, highlight_cov, dark_cov, source_chroma_cov, idx, pair))

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
    selected = scored[:min(num_pairs, len(scored))]
    tone_coverages = [tone_cov for tone_cov, _, _, _, _, _ in selected]
    highlight_coverages = [hi_cov for _, hi_cov, _, _, _, _ in selected]
    dark_coverages = [dark_cov for _, _, dark_cov, _, _, _ in selected]
    source_chroma_coverages = [src_cov for _, _, _, src_cov, _, _ in selected]
    stats = {
        "positive_candidates": sum(tone_cov > 0.0 for tone_cov, _, _, _, _, _ in scored),
        "highlight_positive_candidates": sum(hi_cov > 0.0 for _, hi_cov, _, _, _, _ in scored),
        "dark_positive_candidates": sum(dark_cov > 0.0 for _, _, dark_cov, _, _, _ in scored),
        "source_chroma_positive_candidates": sum(src_cov > 0.0 for _, _, _, src_cov, _, _ in scored),
        "avg_tone_cov": float(np.mean(tone_coverages)) if tone_coverages else 0.0,
        "min_tone_cov": float(np.min(tone_coverages)) if tone_coverages else 0.0,
        "max_tone_cov": float(np.max(tone_coverages)) if tone_coverages else 0.0,
        "avg_highlight_cov": float(np.mean(highlight_coverages)) if highlight_coverages else 0.0,
        "min_highlight_cov": float(np.min(highlight_coverages)) if highlight_coverages else 0.0,
        "max_highlight_cov": float(np.max(highlight_coverages)) if highlight_coverages else 0.0,
        "avg_dark_cov": float(np.mean(dark_coverages)) if dark_coverages else 0.0,
        "min_dark_cov": float(np.min(dark_coverages)) if dark_coverages else 0.0,
        "max_dark_cov": float(np.max(dark_coverages)) if dark_coverages else 0.0,
        "avg_source_chroma_cov": float(np.mean(source_chroma_coverages)) if source_chroma_coverages else 0.0,
        "min_source_chroma_cov": float(np.min(source_chroma_coverages)) if source_chroma_coverages else 0.0,
        "max_source_chroma_cov": float(np.max(source_chroma_coverages)) if source_chroma_coverages else 0.0,
    }
    return [pair for _, _, _, _, _, pair in selected], stats


def load_image_pairs_from_dirs(sdr_dir, hdr_dir, max_long_edge, max_images=0):
    sdr_paths = sorted(glob.glob(os.path.join(sdr_dir, "*.png")))
    hdr_paths = sorted(glob.glob(os.path.join(hdr_dir, "*.png")))

    if not sdr_paths or not hdr_paths:
        raise FileNotFoundError("No PNG images found in SDR/HDR directories")

    sdr_map = {os.path.basename(p): p for p in sdr_paths}
    hdr_map = {os.path.basename(p): p for p in hdr_paths}
    common = sorted(set(sdr_map.keys()) & set(hdr_map.keys()))
    if not common:
        raise FileNotFoundError("No matching filenames between SDR and HDR dirs")

    if max_images > 0:
        common = common[:max_images]

    pairs = []
    for name in common:
        sdr_t, hdr_t = load_image_pair(
            sdr_map[name], hdr_map[name], max_long_edge
        )
        if sdr_t is not None:
            pairs.append((sdr_t, hdr_t))
    if len(pairs) == 0:
        raise RuntimeError(
            "Matched filenames found, but no valid SDR/HDR pairs could be loaded"
        )
    return pairs


def compute_loss_terms(output, target, args, teacher_output=None, source=None):
    """Composite QAT loss tuned to preserve tone plus SDR source chroma."""
    base_l1 = F.l1_loss(output, target)
    total = base_l1
    metrics = {
        "total": float(base_l1.detach().item()),
        "l1": float(base_l1.detach().item()),
    }

    teacher_ref = None
    if teacher_output is not None:
        teacher_ref = teacher_output.to(dtype=output.dtype)

    if teacher_ref is not None and args.teacher_loss_weight > 0:
        teacher_l1 = F.l1_loss(output, teacher_ref)
        total = total + args.teacher_loss_weight * teacher_l1
        metrics["teacher_l1"] = float(teacher_l1.detach().item())

    highlight_mask = build_highlight_neutral_mask(
        target, args.highlight_threshold, args.neutral_threshold
    )
    dark_mask = build_dark_area_mask(target, args.dark_threshold, args.dark_floor)
    metrics["highlight_cov"] = float(highlight_mask.mean().detach().item())
    metrics["dark_cov"] = float(dark_mask.mean().detach().item())

    source_ref = source.to(dtype=output.dtype) if source is not None else None
    if source_ref is not None and args.source_chroma_weight > 0:
        source_chroma_mask = build_source_chroma_mask(source_ref, args)
        output_source_chroma = build_luma_normalized_chroma(output, args)
        source_chroma = build_luma_normalized_chroma(source_ref, args)
        source_chroma_loss = masked_mean_abs(
            output_source_chroma - source_chroma,
            source_chroma_mask,
        )
        total = total + args.source_chroma_weight * source_chroma_loss
        metrics["source_chroma"] = float(source_chroma_loss.detach().item())
        metrics["source_chroma_cov"] = float(
            source_chroma_mask.mean().detach().item()
        )

    if args.highlight_loss_weight > 0:
        highlight_l1 = masked_mean_abs(output - target, highlight_mask)
        total = total + args.highlight_loss_weight * highlight_l1
        metrics["highlight_l1"] = float(highlight_l1.detach().item())

    if teacher_ref is not None and args.highlight_teacher_weight > 0:
        highlight_teacher = masked_mean_abs(output - teacher_ref, highlight_mask)
        total = total + args.highlight_teacher_weight * highlight_teacher
        metrics["highlight_teacher"] = float(highlight_teacher.detach().item())

    output_chroma = None
    target_chroma = None
    if args.highlight_chroma_weight > 0:
        output_chroma = output - output.mean(dim=1, keepdim=True)
        target_chroma = target - target.mean(dim=1, keepdim=True)
        highlight_chroma = masked_mean_abs(
            output_chroma - target_chroma, highlight_mask
        )
        total = total + args.highlight_chroma_weight * highlight_chroma
        metrics["highlight_chroma"] = float(highlight_chroma.detach().item())

    if args.highlight_balance_weight > 0:
        output_balance = build_channel_balance_deltas(output)
        target_balance = build_channel_balance_deltas(target)
        highlight_balance = masked_mean_abs(
            output_balance - target_balance, highlight_mask
        )
        total = total + args.highlight_balance_weight * highlight_balance
        metrics["highlight_balance"] = float(highlight_balance.detach().item())

    if args.dark_loss_weight > 0:
        dark_l1 = masked_mean_abs(output - target, dark_mask)
        total = total + args.dark_loss_weight * dark_l1
        metrics["dark_l1"] = float(dark_l1.detach().item())

    if teacher_ref is not None and args.dark_teacher_weight > 0:
        dark_teacher = masked_mean_abs(output - teacher_ref, dark_mask)
        total = total + args.dark_teacher_weight * dark_teacher
        metrics["dark_teacher"] = float(dark_teacher.detach().item())

    if args.dark_luma_weight > 0:
        dark_luma = masked_mean_abs(
            build_rgb_luma(output) - build_rgb_luma(target), dark_mask
        )
        total = total + args.dark_luma_weight * dark_luma
        metrics["dark_luma"] = float(dark_luma.detach().item())

    if args.dark_chroma_weight > 0:
        if output_chroma is None or target_chroma is None:
            output_chroma = output - output.mean(dim=1, keepdim=True)
            target_chroma = target - target.mean(dim=1, keepdim=True)
        dark_chroma = masked_mean_abs(output_chroma - target_chroma, dark_mask)
        total = total + args.dark_chroma_weight * dark_chroma
        metrics["dark_chroma"] = float(dark_chroma.detach().item())

    metrics["total"] = float(total.detach().item())
    return total, metrics


def accumulate_metrics(metric_sums, metrics):
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)


def average_metrics(metric_sums, count):
    if count <= 0:
        return {}
    return {key: value / count for key, value in metric_sums.items()}


def tensor_to_bgr_u16(tensor):
    arr = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    bgr = arr[:, :, ::-1]
    return np.ascontiguousarray(
        np.clip((bgr * 65535.0) + 0.5, 0.0, 65535.0).astype(np.uint16)
    )


def compute_objective_monitor_metrics(output, target, args):
    pred_bgr = tensor_to_bgr_u16(output)
    ref_bgr = tensor_to_bgr_u16(target)
    pred_bgr, ref_bgr, _ = _crop_shared_black_borders(pred_bgr, ref_bgr)
    pred_bgr, ref_bgr = _prepare_metric_pair(
        pred_bgr, ref_bgr,
        max_side=max(64, int(args.monitor_metric_max_side)),
    )

    metrics = {
        "objective_psnr_db": float(_psnr_bgr(pred_bgr, ref_bgr)),
        "objective_sssim": float(_ssim_bgr(pred_bgr, ref_bgr)),
        "objective_delta_e_itp": float(_delta_e_itp_bgr(pred_bgr, ref_bgr)),
    }

    norm_pred, norm_ref = _grade_normalize_pred_to_ref(pred_bgr, ref_bgr)
    metrics["objective_psnr_norm_db"] = float(_psnr_bgr(norm_pred, norm_ref))
    metrics["objective_sssim_norm"] = float(_ssim_bgr(norm_pred, norm_ref))
    norm_pred_abs, norm_ref_abs = _grade_normalize_absolute_rgb_to_ref(
        _linear_bgr_to_absolute_rgb(pred_bgr),
        _linear_bgr_to_absolute_rgb(ref_bgr),
    )
    metrics["objective_delta_e_itp_norm"] = float(
        _delta_e_itp_absolute_rgb(norm_pred_abs, norm_ref_abs)
    )
    return metrics


def monitor_score_from_metrics(metrics, args):
    mode = str(args.monitor_score).strip().lower()
    if mode == "loss":
        return float(metrics.get("total", float("inf")))

    required = (
        "objective_delta_e_itp_norm",
        "objective_delta_e_itp",
        "objective_sssim_norm",
        "objective_sssim",
        "objective_psnr_norm_db",
    )
    if any(key not in metrics for key in required):
        return float(metrics.get("total", float("inf")))

    de_norm = float(metrics["objective_delta_e_itp_norm"])
    de_raw = float(metrics["objective_delta_e_itp"])
    ssim_norm = float(metrics["objective_sssim_norm"])
    ssim_raw = float(metrics["objective_sssim"])
    psnr_norm = float(metrics["objective_psnr_norm_db"])

    score = (
        0.45 * (de_norm / 10.0)
        + 0.20 * (de_raw / 25.0)
        + 0.20 * max(0.0, (1.0 - ssim_norm) * 100.0)
        + 0.10 * max(0.0, (1.0 - ssim_raw) * 100.0)
        + 0.05 * (10.0 ** (-psnr_norm / 20.0))
    )
    if mode == "hybrid":
        score += 0.25 * float(metrics.get("total", 0.0))
    return float(score)


def format_metrics(metrics):
    parts = [
        f"total={metrics.get('total', 0.0):.6f}",
        f"l1={metrics.get('l1', 0.0):.6f}",
    ]
    if "teacher_l1" in metrics:
        parts.append(f"teacher={metrics['teacher_l1']:.6f}")
    if "highlight_l1" in metrics:
        parts.append(f"hi={metrics['highlight_l1']:.6f}")
    if "highlight_teacher" in metrics:
        parts.append(f"hi_teacher={metrics['highlight_teacher']:.6f}")
    if "highlight_chroma" in metrics:
        parts.append(f"hi_chroma={metrics['highlight_chroma']:.6f}")
    if "highlight_balance" in metrics:
        parts.append(f"hi_balance={metrics['highlight_balance']:.6f}")
    if "highlight_cov" in metrics:
        parts.append(f"hi_cov={metrics['highlight_cov']:.3f}")
    if "dark_l1" in metrics:
        parts.append(f"dark={metrics['dark_l1']:.6f}")
    if "dark_teacher" in metrics:
        parts.append(f"dark_teacher={metrics['dark_teacher']:.6f}")
    if "dark_luma" in metrics:
        parts.append(f"dark_luma={metrics['dark_luma']:.6f}")
    if "dark_chroma" in metrics:
        parts.append(f"dark_chroma={metrics['dark_chroma']:.6f}")
    if "dark_cov" in metrics:
        parts.append(f"dark_cov={metrics['dark_cov']:.3f}")
    if "source_chroma" in metrics:
        parts.append(f"src_chroma={metrics['source_chroma']:.6f}")
    if "source_chroma_cov" in metrics:
        parts.append(f"src_cov={metrics['source_chroma_cov']:.3f}")
    if "objective_psnr_norm_db" in metrics:
        parts.append(f"psnrN={metrics['objective_psnr_norm_db']:.2f}")
    if "objective_sssim_norm" in metrics:
        parts.append(f"ssimN={metrics['objective_sssim_norm']:.4f}")
    if "objective_delta_e_itp_norm" in metrics:
        parts.append(f"deN={metrics['objective_delta_e_itp_norm']:.3f}")
    if "objective_delta_e_itp" in metrics:
        parts.append(f"de={metrics['objective_delta_e_itp']:.3f}")
    return ", ".join(parts)


def list_w8a8_layer_names(model):
    return [
        name for name, module in model.named_modules()
        if isinstance(module, (W8A8Conv2d, W8A8Linear))
    ]


def resolve_protected_w8a8_layers(w8a8_layer_names, args):
    protected = set()
    if str(args.protect_agcm_controls).strip() != "0":
        protected.update(
            name for name in AGCM_CONTROL_LAYER_NAMES if name in w8a8_layer_names
        )
    if str(args.protect_sft_controls).strip() != "0":
        protected.update(
            name for name in w8a8_layer_names
            if "SFT_scale" in name or "SFT_shift" in name
        )
    return sorted(protected)


def demote_w8a8_layers_to_w8(model, layer_names, compute_dtype):
    """Convert selected W8A8 layers back to W8 to protect control paths."""
    converted = []
    target_names = set(layer_names)
    for name, module in list(model.named_modules()):
        if name not in target_names:
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        if isinstance(module, W8A8Conv2d):
            tmp_conv = nn.Conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                module.dilation, module.groups,
                bias=(module.bias is not None),
            )
            tmp_conv.weight.data = (
                module.weight_int8.float() *
                module.w_scale.float().view(-1, 1, 1, 1)
            )
            if module.bias is not None:
                tmp_conv.bias.data = module.bias.float().clone()
            setattr(parent, parts[-1], W8Conv2d(tmp_conv, compute_dtype))
            converted.append(name)
        elif isinstance(module, W8A8Linear):
            tmp_linear = nn.Linear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            tmp_linear.weight.data = (
                module.weight_int8.float() *
                module.w_scale.float().view(-1, 1)
            )
            if module.bias is not None:
                tmp_linear.bias.data = module.bias.float().clone()
            setattr(parent, parts[-1], W8Linear(tmp_linear, compute_dtype))
            converted.append(name)

    return sorted(converted)


def protect_mixed_control_layers(model, compute_dtype, args):
    """Keep high-leverage control layers in weight-only mode."""
    current_w8a8_layers = list_w8a8_layer_names(model)
    protected_layers = resolve_protected_w8a8_layers(current_w8a8_layers, args)
    if not protected_layers:
        return current_w8a8_layers, []
    removed = demote_w8a8_layers_to_w8(model, protected_layers, compute_dtype)
    return list_w8a8_layer_names(model), removed


def parse_layer_list(layer_spec: str):
    if not layer_spec:
        return []
    return sorted({name.strip() for name in str(layer_spec).split(",") if name.strip()})


def resolve_fp16_layer_prefs(candidate_layers, args):
    candidate_set = set(candidate_layers)
    fp16_layers = set()
    if str(args.fp16_sensitive_layers).strip() != "0":
        fp16_layers.update(
            name for name in FP16_SENSITIVE_LAYER_NAMES if name in candidate_set
        )
    fp16_layers.update(
        name for name in parse_layer_list(args.fp16_layers)
        if name in candidate_set
    )
    return sorted(fp16_layers)


def list_quantizable_layer_names(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear,
                               W8A8Conv2d, W8A8Linear, W8Conv2d, W8Linear)):
            names.append(name)
    return names


def promote_selected_layers_to_fp16(model, layer_names, compute_dtype):
    """Convert selected quantized layers back to native FP modules."""
    converted = []
    target_names = set(layer_names)
    for name, module in list(model.named_modules()):
        if name not in target_names:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if isinstance(module, (W8A8Conv2d, W8Conv2d)):
            setattr(parent, parts[-1], _predequantize_conv(module, compute_dtype))
            converted.append(name)
        elif isinstance(module, (W8A8Linear, W8Linear)):
            setattr(parent, parts[-1], _predequantize_linear(module, compute_dtype))
            converted.append(name)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            converted.append(name)
    return sorted(set(converted))


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

def train_qat(model, pairs, device, compute_dtype, args,
              teacher_model=None, teacher_dtype=None, monitor_pairs=None):
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

    # All parameters are trainable " QAT layers have learnable scales
    # and weights; non-QAT layers (e.g. PReLU) also benefit from fine-tuning.
    qat_params = [p for p in model.parameters() if p.requires_grad]

    batch_size = max(int(args.batch_size), 1)
    steps_per_epoch = max((len(pairs) + batch_size - 1) // batch_size, 1)

    optimizer = optim.Adam(qat_params, lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
    )

    if monitor_pairs is None:
        monitor_count = min(max(args.num_validate, 0), len(pairs))
        monitor_pairs = pairs[:monitor_count]
    if str(args.monitor_score).strip().lower() == "loss":
        score_name = "monitor total" if (
            args.teacher_loss_weight > 0
            or args.highlight_loss_weight > 0
            or args.highlight_teacher_weight > 0
            or args.highlight_chroma_weight > 0
            or args.highlight_balance_weight > 0
            or args.dark_loss_weight > 0
            or args.dark_teacher_weight > 0
            or args.dark_luma_weight > 0
            or args.dark_chroma_weight > 0
            or args.source_chroma_weight > 0
        ) else "monitor L1"
    else:
        score_name = f"monitor {args.monitor_score}"

    monitor_teacher_outputs = None
    if monitor_pairs and teacher_model is not None:
        monitor_teacher_outputs = []
        with torch.inference_mode():
            for sdr_t, _ in monitor_pairs:
                teacher_inp = prepare_model_input(sdr_t, device, teacher_dtype)
                teacher_output, _ = teacher_model(teacher_inp)
                if not torch.isfinite(teacher_output).all():
                    teacher_output = torch.nan_to_num(
                        teacher_output, nan=0.0, posinf=1.0, neginf=0.0
                    )
                monitor_teacher_outputs.append(teacher_output.detach().cpu())

    print(f"\n  Training: {args.epochs} epochs, lr={args.lr}, "
          f"crop={args.crop_size}, batch={batch_size}, {len(pairs)} pairs")
    print("  Strategy: adaptive mixed QAT "
          "(W8A8 for speed, W8A16/FP16 for tone-sensitive paths)")
    print("  Loss recipe: "
          f"L1 + {args.teacher_loss_weight:.3f}*teacher "
          f"+ {args.highlight_loss_weight:.3f}*highlight "
          f"+ {args.highlight_teacher_weight:.3f}*highlight_teacher "
          f"+ {args.highlight_chroma_weight:.3f}*highlight_chroma "
          f"+ {args.highlight_balance_weight:.3f}*highlight_balance "
          f"+ {args.dark_loss_weight:.3f}*dark "
          f"+ {args.dark_teacher_weight:.3f}*dark_teacher "
          f"+ {args.dark_luma_weight:.3f}*dark_luma "
          f"+ {args.dark_chroma_weight:.3f}*dark_chroma "
          f"+ {args.source_chroma_weight:.3f}*source_chroma")
    print("  Crop sampling: "
          f"best-of-{max(1, int(args.highlight_crop_attempts))} random crops "
          "biased toward bright neutral highlights and shadow detail")
    if monitor_pairs:
        print(f"  Model selection: best {score_name} on {len(monitor_pairs)} "
              f"fixed full-image pairs")
        if str(args.monitor_score).strip().lower() != "loss":
            print("  Objective monitor: PSNR/SSIM/DeltaEITP without HDR-VDP3 "
                  f"(max_side={max(64, int(args.monitor_metric_max_side))})")
    else:
        print("  Model selection: best training total")

    best_train_loss = float("inf")
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    monitor_enabled = bool(monitor_pairs)
    epochs_without_improve = 0

    def compute_monitor_metrics():
        nonlocal monitor_enabled
        if not monitor_pairs or not monitor_enabled:
            return None
        metric_sums = {}
        num_items = 0
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for idx, (sdr_t, hdr_t) in enumerate(monitor_pairs):
                    inp = prepare_model_input(sdr_t, device, compute_dtype)
                    target = hdr_t.to(device=device, dtype=compute_dtype)
                    teacher_output = None
                    if teacher_model is not None:
                        if monitor_teacher_outputs is not None:
                            teacher_output = monitor_teacher_outputs[idx].to(
                                device=device, dtype=compute_dtype
                            )
                        else:
                            teacher_inp = prepare_model_input(
                                sdr_t, device, teacher_dtype
                            )
                            teacher_output, _ = teacher_model(teacher_inp)
                            if not torch.isfinite(teacher_output).all():
                                teacher_output = torch.nan_to_num(
                                    teacher_output, nan=0.0, posinf=1.0, neginf=0.0
                                )
                    output, _ = model(inp)
                    if not torch.isfinite(output).all():
                        output = torch.nan_to_num(
                            output, nan=0.0, posinf=1.0, neginf=0.0
                        )
                    loss, metrics = compute_loss_terms(
                        output, target, args,
                        teacher_output=teacher_output,
                        source=inp[0],
                    )
                    if torch.isfinite(loss):
                        if str(args.monitor_score).strip().lower() != "loss":
                            try:
                                metrics.update(
                                    compute_objective_monitor_metrics(
                                        output, target, args
                                    )
                                )
                            except Exception as exc:
                                print(f"  [warn] Objective monitor failed: {exc}")
                                monitor_enabled = False
                                return None
                        accumulate_metrics(metric_sums, metrics)
                        num_items += 1
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "miopen" in msg or "hiprtc" in msg:
                print("  [warn] Monitor pass disabled due ROCm MIOpen error; "
                      "falling back to training-total selection.")
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
        if num_items == 0:
            return None
        return average_metrics(metric_sums, num_items)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        epoch_metrics = {}
        num_samples = 0

        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start + batch_size]
            if not batch_pairs:
                continue

            cropped_groups = {}
            for sdr_t, hdr_t in batch_pairs:
                sdr_crop, hdr_crop = sample_training_crop_pair(sdr_t, hdr_t, args)
                shape = tuple(sdr_crop.shape[-2:])
                cropped_groups.setdefault(shape, []).append((sdr_crop, hdr_crop))

            batch_total = sum(len(group) for group in cropped_groups.values())
            if batch_total <= 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            batch_valid = 0

            for group_pairs in cropped_groups.values():
                sdr_batch = torch.cat([sdr_crop for sdr_crop, _ in group_pairs], dim=0)
                hdr_batch = torch.cat([hdr_crop for _, hdr_crop in group_pairs], dim=0)

                inp = prepare_model_input(sdr_batch, device, compute_dtype)
                target = hdr_batch.to(device=device, dtype=compute_dtype)
                teacher_output = None
                if teacher_model is not None:
                    with torch.inference_mode():
                        teacher_inp = prepare_model_input(
                            sdr_batch, device, teacher_dtype
                        )
                        teacher_output, _ = teacher_model(teacher_inp)
                        if not torch.isfinite(teacher_output).all():
                            teacher_output = torch.nan_to_num(
                                teacher_output, nan=0.0, posinf=1.0, neginf=0.0
                            )

                output, _ = model(inp)
                if not torch.isfinite(output).all():
                    output = torch.nan_to_num(
                        output, nan=0.0, posinf=1.0, neginf=0.0
                    )

                loss, loss_metrics = compute_loss_terms(
                    output, target, args,
                    teacher_output=teacher_output,
                    source=inp[0],
                )
                if not torch.isfinite(loss):
                    continue

                group_size = sdr_batch.shape[0]
                scaled_loss = loss * (group_size / batch_total)
                scaled_loss.backward()
                accumulate_metrics(
                    epoch_metrics,
                    {key: value * group_size for key, value in loss_metrics.items()},
                )
                num_samples += group_size
                batch_valid += group_size

            if batch_valid == 0:
                continue

            torch.nn.utils.clip_grad_norm_(qat_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_metrics = average_metrics(epoch_metrics, max(num_samples, 1))
        lr_now = scheduler.get_last_lr()[0]
        avg_total = avg_metrics.get("total", float("inf"))
        best_train_loss = min(best_train_loss, avg_total)
        monitor_metrics = compute_monitor_metrics()
        score = (
            monitor_score_from_metrics(monitor_metrics, args)
            if monitor_metrics is not None else avg_total
        )

        if monitor_metrics is None:
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"{format_metrics(avg_metrics)}, lr={lr_now:.2e}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"train[{format_metrics(avg_metrics)}], "
                  f"monitor[{format_metrics(monitor_metrics)}], "
                  f"lr={lr_now:.2e}")

        improved = score < (best_score - args.early_stop_min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if (args.early_stop_patience > 0 and
                    epochs_without_improve >= args.early_stop_patience):
                print("  Early stopping: "
                      f"no {score_name} improvement > {args.early_stop_min_delta:.1e} "
                      f"for {epochs_without_improve} epoch(s)")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    print(f"  Best training loss: {best_train_loss:.6f}")
    if monitor_pairs and monitor_enabled:
        print(f"  Selected epoch   : {best_epoch} "
              f"(best {score_name}={best_score:.6f})")
    else:
        print(f"  Selected epoch   : {best_epoch} "
              f"(best training total)")
    return model


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning for Mixed INT8 HDRTVNet++"
    )
    parser.add_argument("--ptq-checkpoint",
                        default=os.path.join(
                            _REPO_ROOT,
                            "src",
                            "models",
                            "weights",
                            "Ensemble_AGCM_LE_int8_mixed.pt",
                        ),
                        help="PTQ mixed checkpoint to start from")
    parser.add_argument("--fp32-model",
                        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "Ensemble_AGCM_LE.pth"),
                        help="FP32 model (for --from-scratch or validation)")
    parser.add_argument("--output",
                        default=os.path.join(
                            _REPO_ROOT,
                            "src",
                            "models",
                            "weights",
                            "Ensemble_AGCM_LE_int8_mixed_qat.pt",
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
    parser.add_argument("--val-sdr-dir", default="",
                        help="Optional held-out SDR validation directory for QAT model selection")
    parser.add_argument("--val-hdr-dir", default="",
                        help="Optional held-out HDR validation directory for QAT model selection")
    parser.add_argument("--max-val-images", type=int, default=0,
                        help="Max held-out validation images (0 = all)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Start from FP32 model (PTQ + calibrate + QAT)")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                        help="Compute precision")
    parser.add_argument("--channel-threshold", type=int, default=32,
                        help="Max channel count for W8A8 layers")
    parser.add_argument("--hg-weights",
                        default=os.path.join(_REPO_ROOT, "src", "models", "weights", "HG_weights.pth"),
                        help="Path to HG weights (HG_weights.pth)")
    parser.add_argument("--use-hg", default="1", choices=["1", "0"],
                        help="Use HG refinement (1) or base AGCM+LE only (0)")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Random crop size for training patches")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Training batch size for cropped patches")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max training images (0 = all)")
    parser.add_argument("--max-long-edge", type=int, default=960,
                        help="Resize images so longest edge  this")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.05,
                        help="Weight for staying close to the starting PTQ output")
    parser.add_argument("--highlight-loss-weight", type=float, default=0.10,
                        help="Extra L1 weight on bright near-neutral highlights")
    parser.add_argument("--highlight-teacher-weight", type=float, default=0.10,
                        help="Extra teacher-anchor weight on bright near-neutral highlights")
    parser.add_argument("--highlight-chroma-weight", type=float, default=0.05,
                        help="Extra chroma-preservation weight on bright neutral highlights")
    parser.add_argument("--highlight-balance-weight", type=float, default=0.10,
                        help="Extra RGB balance weight to keep bright neutral highlights from drifting warm/cool")
    parser.add_argument("--highlight-threshold", type=float, default=0.75,
                        help="Target intensity threshold for highlight-focused losses")
    parser.add_argument("--neutral-threshold", type=float, default=0.08,
                        help="Max target channel spread treated as near-neutral")
    parser.add_argument("--dark-loss-weight", type=float, default=0.08,
                        help="Extra L1 weight on non-black shadow/detail pixels")
    parser.add_argument("--dark-teacher-weight", type=float, default=0.10,
                        help="Extra teacher-anchor weight on dark shadow/detail pixels")
    parser.add_argument("--dark-luma-weight", type=float, default=0.06,
                        help="Extra luma-preservation weight in dark areas")
    parser.add_argument("--dark-chroma-weight", type=float, default=0.04,
                        help="Extra chroma-preservation weight in dark areas")
    parser.add_argument("--dark-threshold", type=float, default=0.16,
                        help="Target luma threshold for dark-area losses")
    parser.add_argument("--dark-floor", type=float, default=0.01,
                        help="Luma floor below which pure black pixels are down-weighted")
    parser.add_argument("--dark-crop-weight", type=float, default=1.0,
                        help="Relative crop-selection weight for dark-area coverage")
    parser.add_argument("--dark-monitor-weight", type=float, default=1.0,
                        help="Relative monitor-selection weight for dark-area coverage")
    parser.add_argument("--source-chroma-weight", type=float, default=0.0,
                        help="Preserve SDR source RGB ratios without constraining HDR luma")
    parser.add_argument("--source-chroma-saturation-threshold", type=float, default=0.05,
                        help="Minimum SDR channel spread before source chroma anchoring applies")
    parser.add_argument("--source-chroma-luma-floor", type=float, default=0.02,
                        help="SDR luma floor for stable source chroma-ratio anchoring")
    parser.add_argument("--source-chroma-ratio-clip", type=float, default=6.0,
                        help="Clamp RGB/luma ratios used by source chroma anchoring")
    parser.add_argument("--source-chroma-crop-weight", type=float, default=0.5,
                        help="Relative crop-selection weight for colorful SDR source coverage")
    parser.add_argument("--source-chroma-monitor-weight", type=float, default=0.5,
                        help="Relative monitor-selection weight for colorful SDR source coverage")
    parser.add_argument("--highlight-crop-attempts", type=int, default=4,
                        help="Random crop attempts per image; keep the crop with the strongest tone-protection coverage")
    parser.add_argument("--protect-agcm-controls", default="1", choices=["1", "0"],
                        help="Keep AGCM cond_scale/cond_shift layers in W8A16")
    parser.add_argument("--protect-sft-controls", default="1", choices=["1", "0"],
                        help="Also keep SFT scale/shift layers in W8A16")
    parser.add_argument("--fp16-sensitive-layers", default="1", choices=["1", "0"],
                        help="Keep a curated set of the most color-sensitive layers in FP16")
    parser.add_argument("--fp16-layers", default="",
                        help="Comma-separated extra layer names to keep in FP16")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                        help="Stop after this many non-improving epochs (0 disables)")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5,
                        help="Minimum monitor improvement required to reset early stopping")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for repeatable QAT runs")
    parser.add_argument("--deterministic", default="1", choices=["1", "0"],
                        help="Enable deterministic PyTorch behavior where available")

    # Validation
    parser.add_argument("--num-validate", type=int, default=8,
                        help="Number of images for quality validation (0 = all monitor pairs)")
    parser.add_argument("--num-calibrate", type=int, default=0,
                        help="Calibration images when --from-scratch (0 = all)")
    parser.add_argument("--monitor-score", default="loss",
                        choices=["loss", "objective", "hybrid"],
                        help="Epoch selection score for monitor pairs")
    parser.add_argument("--monitor-metric-max-side", type=int, default=720,
                        help="Max side for objective monitor metrics")

    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    compute_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cpu" and compute_dtype == torch.float16:
        compute_dtype = torch.float32

    seed, deterministic = configure_reproducibility(args)

    print(f"Device: {device}, Precision: {args.precision}")
    if device.type == "cpu" and args.precision == "fp16":
        print("Note: fp16 requested on CPU, using fp32 compute fallback")
    print(f"Reproducibility: seed={seed}, deterministic={deterministic}")

    use_hg = str(args.use_hg).strip() != "0"
    selection_mode = "legacy"
    selection_info = None
    sensitivity_threshold = None
    fp16_layers = []

    # ------------------------------------------------------------------
    # 1. Load model (either from PTQ checkpoint or from scratch)
    # ------------------------------------------------------------------
    if args.from_scratch:
        print(f"\nLoading FP32 model from {args.fp32_model} ...")
        model = _build_fp32_model(args.fp32_model, args.hg_weights, use_hg)

        print(f"Applying mixed PTQ (channel_threshold={args.channel_threshold}) ...")
        candidate_layers = [
            name for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        fp16_layers = resolve_fp16_layer_prefs(candidate_layers, args)
        _quantize_model_mixed(
            model, compute_dtype, args.channel_threshold, fp16_layers=fp16_layers
        )
        model = model.to(dtype=compute_dtype, device=device)
        model.eval()

        # Calibrate activation scales
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
        calibrate_w8a8(model, calib_inputs)
        channel_threshold = args.channel_threshold
        w8a8_layers = None  # v1 from-scratch uses channel heuristic
        use_asym = False
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
        if device.type == "cpu" and compute_dtype == torch.float16:
            compute_dtype = torch.float32
        channel_threshold = checkpoint.get("channel_threshold", 32)
        selection_mode = checkpoint.get(
            "selection_mode",
            "threshold" if checkpoint.get("w8a8_layers", None) is not None else "legacy",
        )
        selection_info = checkpoint.get("auto_selection", None)
        sensitivity_threshold = checkpoint.get("sensitivity_threshold", None)
        fp16_layers = checkpoint.get("fp16_layers", None) or []

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

        # Support both v1 (channel-threshold) and v2 (sensitivity-based) checkpoints
        w8a8_layers = checkpoint.get("w8a8_layers", None)
        use_asym = checkpoint.get("activation_quant", "symmetric") == "asymmetric"
        if w8a8_layers is not None:
            _quantize_model_mixed_v2(model, compute_dtype,
                                      w8a8_layers=w8a8_layers,
                                      fp16_layers=fp16_layers,
                                      asymmetric=use_asym)
        else:
            _quantize_model_mixed(
                model, compute_dtype, channel_threshold, fp16_layers=fp16_layers
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(dtype=compute_dtype, device=device)
        model.eval()
        if w8a8_layers is not None:
            extra = f", mode={selection_mode}, W8A8={len(w8a8_layers)}"
            if fp16_layers:
                extra += f", FP16={len(fp16_layers)}"
            if sensitivity_threshold is not None:
                extra += f", threshold={sensitivity_threshold:.2e}"
            print(f"  Loaded ({checkpoint.get('quantization', '?')}{extra})")
        else:
            print(f"  Loaded ({checkpoint.get('quantization', '?')}, "
                  f"threshold={channel_threshold})")

    protected_layers_removed = []
    if (str(args.protect_agcm_controls).strip() != "0" or
            str(args.protect_sft_controls).strip() != "0"):
        print("\nProtecting mixed control layers ...")
        protected_w8a8_layers, protected_layers_removed = (
            protect_mixed_control_layers(model, compute_dtype, args)
        )
        if protected_layers_removed:
            w8a8_layers = protected_w8a8_layers
            selection_mode = (
                f"{selection_mode}+protected"
                if "protected" not in selection_mode else selection_mode
            )
            print(f"  Demoted {len(protected_layers_removed)} control layers to W8A16")
            for name in protected_layers_removed:
                print(f"    - {name}")
        else:
            print("  No matching W8A8 control layers needed protection")

    promoted_fp16_layers = []
    requested_fp16_layers = resolve_fp16_layer_prefs(
        list_quantizable_layer_names(model), args
    )
    if requested_fp16_layers:
        print("\nPromoting selected layers to FP16 ...")
        promoted_fp16_layers = promote_selected_layers_to_fp16(
            model, requested_fp16_layers, compute_dtype
        )
        if promoted_fp16_layers:
            fp16_layers = sorted(set(fp16_layers) | set(promoted_fp16_layers))
            if w8a8_layers is not None:
                w8a8_layers = [name for name in w8a8_layers if name not in set(fp16_layers)]
            selection_mode = (
                f"{selection_mode}+fp16"
                if "fp16" not in selection_mode else selection_mode
            )
            print(f"  Promoted {len(promoted_fp16_layers)} layers to FP16")
            for name in promoted_fp16_layers:
                print(f"    - {name}")
        else:
            print("  No matching layers were promoted to FP16")

    # ------------------------------------------------------------------
    # 2. Load SDR/HDR pairs for training
    # ------------------------------------------------------------------
    print(f"\nLoading SDR/HDR pairs from {args.sdr_dir} + {args.hdr_dir} ...")
    pairs = load_image_pairs_from_dirs(
        args.sdr_dir, args.hdr_dir, args.max_long_edge, args.max_images
    )

    print(f"  {len(pairs)} pairs loaded")

    monitor_source = "train"
    monitor_candidate_pairs = pairs
    if args.val_sdr_dir and args.val_hdr_dir:
        print(f"\nLoading held-out monitor pairs from {args.val_sdr_dir} + {args.val_hdr_dir} ...")
        monitor_candidate_pairs = load_image_pairs_from_dirs(
            args.val_sdr_dir,
            args.val_hdr_dir,
            args.max_long_edge,
            args.max_val_images,
        )
        monitor_source = "validation"
        print(f"  {len(monitor_candidate_pairs)} held-out pairs loaded")
    elif args.val_sdr_dir or args.val_hdr_dir:
        raise ValueError("Provide both --val-sdr-dir and --val-hdr-dir, or neither")

    monitor_pairs, monitor_stats = select_tone_protected_monitor_pairs(
        monitor_candidate_pairs, args.num_validate, args
    )
    if monitor_pairs:
        print("  Tone-aware monitor selection:")
        print(f"    using {len(monitor_pairs)} full-image pairs "
              f"from {monitor_source} "
              f"(avg hi_cov={monitor_stats['avg_highlight_cov']:.4f}, "
              f"max={monitor_stats['max_highlight_cov']:.4f}, "
              f"avg dark_cov={monitor_stats['avg_dark_cov']:.4f}, "
              f"max={monitor_stats['max_dark_cov']:.4f}, "
              f"avg src_chroma_cov={monitor_stats['avg_source_chroma_cov']:.4f}, "
              f"max={monitor_stats['max_source_chroma_cov']:.4f}, "
              f"positive candidates={monitor_stats['positive_candidates']})")
        if monitor_stats["max_highlight_cov"] <= 0.0:
            print("    [warn] No monitor image contained qualifying highlight pixels")
        if monitor_stats["max_dark_cov"] <= 0.0:
            print("    [warn] No monitor image contained qualifying dark pixels")
        if args.source_chroma_weight > 0 and monitor_stats["max_source_chroma_cov"] <= 0.0:
            print("    [warn] No monitor image contained qualifying SDR chroma pixels")

    teacher_model = None
    teacher_dtype = compute_dtype
    if (
        args.teacher_loss_weight > 0
        or args.highlight_teacher_weight > 0
        or args.dark_teacher_weight > 0
    ):
        print("\nPreparing PTQ teacher model ...")
        teacher_model = copy.deepcopy(model).to(dtype=teacher_dtype, device=device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # 3. Convert to QAT mode
    # ------------------------------------------------------------------
    print("\nConverting to QAT mode ...")
    model = convert_to_qat(model)
    model = model.float()  # QAT trains in FP32 for gradient stability
    model = model.to(device)

    # ------------------------------------------------------------------
    # 4. QAT fine-tuning
    # ------------------------------------------------------------------
    train_dtype = torch.float32  # train in FP32
    t0 = time.perf_counter()
    model = train_qat(
        model, pairs, device, train_dtype, args,
        teacher_model=teacher_model, teacher_dtype=teacher_dtype,
        monitor_pairs=monitor_pairs,
    )
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
    # 6. Save checkpoint (same format as quantize_int8_mixed.py)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_data = {
        "state_dict": model.state_dict(),
        "compute_dtype": str(compute_dtype),
        "quantization": "w8a8_mixed",
        "qat_strategy": "adaptive_mixed_precision",
        "channel_threshold": channel_threshold,
        "qat_epochs": args.epochs,
        "qat_lr": args.lr,
        "qat_recipe": {
            "teacher_loss_weight": args.teacher_loss_weight,
            "highlight_loss_weight": args.highlight_loss_weight,
            "highlight_teacher_weight": args.highlight_teacher_weight,
            "highlight_chroma_weight": args.highlight_chroma_weight,
            "highlight_balance_weight": args.highlight_balance_weight,
            "highlight_threshold": args.highlight_threshold,
            "neutral_threshold": args.neutral_threshold,
            "dark_loss_weight": args.dark_loss_weight,
            "dark_teacher_weight": args.dark_teacher_weight,
            "dark_luma_weight": args.dark_luma_weight,
            "dark_chroma_weight": args.dark_chroma_weight,
            "dark_threshold": args.dark_threshold,
            "dark_floor": args.dark_floor,
            "dark_crop_weight": args.dark_crop_weight,
            "dark_monitor_weight": args.dark_monitor_weight,
            "source_chroma_weight": args.source_chroma_weight,
            "source_chroma_saturation_threshold": args.source_chroma_saturation_threshold,
            "source_chroma_luma_floor": args.source_chroma_luma_floor,
            "source_chroma_ratio_clip": args.source_chroma_ratio_clip,
            "source_chroma_crop_weight": args.source_chroma_crop_weight,
            "source_chroma_monitor_weight": args.source_chroma_monitor_weight,
            "protect_agcm_controls": str(args.protect_agcm_controls).strip() != "0",
            "protect_sft_controls": str(args.protect_sft_controls).strip() != "0",
            "fp16_sensitive_layers": str(args.fp16_sensitive_layers).strip() != "0",
            "fp16_layers": parse_layer_list(args.fp16_layers),
            "batch_size": args.batch_size,
            "highlight_crop_attempts": args.highlight_crop_attempts,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "monitor_selection": "tone_protection_coverage_topk",
            "monitor_source": monitor_source,
            "monitor_score": args.monitor_score,
            "monitor_metric_max_side": args.monitor_metric_max_side,
            "val_sdr_dir": args.val_sdr_dir,
            "val_hdr_dir": args.val_hdr_dir,
            "monitor_avg_tone_cov": monitor_stats["avg_tone_cov"],
            "monitor_max_tone_cov": monitor_stats["max_tone_cov"],
            "monitor_avg_highlight_cov": monitor_stats["avg_highlight_cov"],
            "monitor_max_highlight_cov": monitor_stats["max_highlight_cov"],
            "monitor_avg_dark_cov": monitor_stats["avg_dark_cov"],
            "monitor_max_dark_cov": monitor_stats["max_dark_cov"],
            "monitor_avg_source_chroma_cov": monitor_stats["avg_source_chroma_cov"],
            "monitor_max_source_chroma_cov": monitor_stats["max_source_chroma_cov"],
            "monitor_positive_candidates": monitor_stats["positive_candidates"],
            "monitor_highlight_positive_candidates": monitor_stats["highlight_positive_candidates"],
            "monitor_dark_positive_candidates": monitor_stats["dark_positive_candidates"],
            "monitor_source_chroma_positive_candidates": monitor_stats["source_chroma_positive_candidates"],
            "seed": seed,
            "deterministic": deterministic,
            "teacher_source": "ptq_checkpoint",
        },
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
    # Propagate v2 metadata from source checkpoint
    if w8a8_layers is not None:
        save_data["w8a8_layers"] = w8a8_layers
        save_data["selection_mode"] = selection_mode
        if selection_info is not None:
            save_data["auto_selection"] = selection_info
        if sensitivity_threshold is not None:
            save_data["sensitivity_threshold"] = sensitivity_threshold
    if fp16_layers:
        save_data["fp16_layers"] = fp16_layers
    if protected_layers_removed:
        save_data["protected_layers_removed"] = protected_layers_removed
    if use_asym:
        save_data["activation_quant"] = "asymmetric"
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

    # Load FP16 reference model
    ref_model = _build_fp32_model(args.fp32_model, args.hg_weights, use_hg)
    ref_model = ref_model.to(dtype=compute_dtype, device=device).eval()

    val_pairs = monitor_pairs if monitor_pairs else pairs[:args.num_validate]
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

