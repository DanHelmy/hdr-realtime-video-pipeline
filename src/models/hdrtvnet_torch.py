import math
import os
import pathlib
import re
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hip_sdk_detection import detect_hip_sdk_windows
from models.hdrtvnet_modules.Ensemble_AGCM_LE_arch import Ensemble_AGCM_LE
from models.hdrtvnet_modules.HG_Composite_arch import HG_Composite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DEFAULT_HG_WEIGHTS = os.path.join(
    _ROOT, "src", "models", "weights", "HG_weights.pth"
)

_HAS_COMPILE = hasattr(torch, "compile")          # PyTorch >= 2.0
_HAS_CUDA_GRAPHS = hasattr(torch.cuda, "CUDAGraph")  # PyTorch >= 1.10
_IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
_IS_NVIDIA = torch.cuda.is_available() and not _IS_ROCM

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if os.name == "nt" and _IS_ROCM:
    _HAS_HIP_SDK, _HIP_SDK_ROOT = detect_hip_sdk_windows()
else:
    _HAS_HIP_SDK, _HIP_SDK_ROOT = False, None

# Enable TF32 on Ampere+ GPUs " harmless no-op on AMD
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ===================================================================
# Weight-only INT8 replacement layers (gpt-fast / torchao style)
# ===================================================================

class W8Conv2d(nn.Module):
    """Conv2d with INT8 weights, dequantized to compute_dtype at runtime."""

    def __init__(self, conv: nn.Conv2d, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.compute_dtype = compute_dtype

        w = conv.weight.data.float()
        w_flat = w.reshape(w.shape[0], -1)
        scale = w_flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w / scale.view(-1, 1, 1, 1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale.to(compute_dtype))
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.to(compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(self.compute_dtype) * self.scale.view(-1, 1, 1, 1)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class W8Linear(nn.Module):
    """Linear with INT8 weights, dequantized to compute_dtype at runtime."""

    def __init__(self, linear: nn.Linear, compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype

        w = linear.weight.data.float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w / scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale.to(compute_dtype))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_int8.to(self.compute_dtype) * self.scale.view(-1, 1)
        return F.linear(x, w, self.bias)




# ===================================================================
# Full INT8 (W8A8) replacement layers " weights AND activations INT8
# ===================================================================

class W8A8Conv2d(nn.Module):
    """Conv2d with INT8 weights and INT8 activations (static quantization).

    Weights use per-output-channel scale.  Activations use per-tensor scale
    determined during a calibration pass.  At inference both are dequantized
    to ``compute_dtype`` and passed to F.conv2d.  torch.compile fuses the
    dequant + conv into a single efficient kernel.

    When ``asymmetric=True``, activations are quantized to unsigned [0, 255]
    with a zero-point, giving 2- precision for non-negative (post-ReLU)
    distributions and ~1.8- for post-LeakyReLU distributions.
    """

    def __init__(self, conv: nn.Conv2d, compute_dtype: torch.dtype = torch.float16,
                 asymmetric: bool = False):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.compute_dtype = compute_dtype
        self.is_asymmetric = asymmetric

        # Quantize weights (identical to W8Conv2d)
        w = conv.weight.data.float()
        w_flat = w.reshape(w.shape[0], -1)
        w_scale = w_flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w / w_scale.view(-1, 1, 1, 1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("w_scale", w_scale.to(compute_dtype))
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.to(compute_dtype))
        else:
            self.bias = None
        # Activation scale " set during calibration via calibrate_w8a8()
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=compute_dtype))
        if asymmetric:
            self.register_buffer("x_zero", torch.tensor(0.0, dtype=compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_asymmetric:
            # Asymmetric: map [x_zero, x_zero + x_scale*255] ' [0, 255]
            x_q = ((x - self.x_zero) / self.x_scale).round().clamp(0, 255)
            x_deq = x_q * self.x_scale + self.x_zero
        else:
            # Symmetric: map [-x_scale*127, x_scale*127] ' [-128, 127]
            x_int8 = (x / self.x_scale).round().clamp(-128, 127).to(torch.int8)
            x_deq = x_int8.to(self.compute_dtype) * self.x_scale
        # Dequantize weights
        w = self.weight_int8.to(self.compute_dtype) * self.w_scale.view(-1, 1, 1, 1)
        return F.conv2d(x_deq, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class W8A8Linear(nn.Module):
    """Linear with INT8 weights and INT8 activations (static quantization)."""

    def __init__(self, linear: nn.Linear, compute_dtype: torch.dtype = torch.float16,
                 asymmetric: bool = False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.compute_dtype = compute_dtype
        self.is_asymmetric = asymmetric

        w = linear.weight.data.float()
        w_scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        w_int8 = (w / w_scale.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("w_scale", w_scale.to(compute_dtype))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(compute_dtype))
        else:
            self.bias = None
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=compute_dtype))
        if asymmetric:
            self.register_buffer("x_zero", torch.tensor(0.0, dtype=compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_asymmetric:
            x_q = ((x - self.x_zero) / self.x_scale).round().clamp(0, 255)
            x_deq = x_q * self.x_scale + self.x_zero
        else:
            x_int8 = (x / self.x_scale).round().clamp(-128, 127).to(torch.int8)
            x_deq = x_int8.to(self.compute_dtype) * self.x_scale
        w = self.weight_int8.to(self.compute_dtype) * self.w_scale.view(-1, 1)
        return F.linear(x_deq, w, self.bias)


def _quantize_model_w8a8(model: nn.Module,
                         compute_dtype: torch.dtype = torch.float16,
                         asymmetric: bool = False) -> nn.Module:
    """Replace all Conv2d and Linear with W8A8 equivalents (in-place)."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            if isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1],
                        W8A8Conv2d(module, compute_dtype,
                                   asymmetric=asymmetric))
            else:
                setattr(parent, parts[-1],
                        W8A8Linear(module, compute_dtype,
                                   asymmetric=asymmetric))
            replaced += 1
    print(f"  Quantized {replaced} layers to W8A8 (INT8 weights + activations, "
          f"{compute_dtype} compute)")
    return model


# ===================================================================
# Pre-dequantization " convert W8* layers back to native FP16 layers
# for GPUs without INT8 tensor cores (removes per-inference dequant
# overhead while keeping the compressed checkpoint on disk).
# ===================================================================

def _predequantize_conv(w8_mod, compute_dtype):
    """Convert a W8Conv2d or W8A8Conv2d back to a standard nn.Conv2d."""
    conv = nn.Conv2d(
        w8_mod.in_channels, w8_mod.out_channels,
        w8_mod.kernel_size, stride=w8_mod.stride,
        padding=w8_mod.padding, dilation=w8_mod.dilation,
        groups=w8_mod.groups,
        bias=w8_mod.bias is not None,
    )
    # Reconstruct FP weights: w_fp = w_int8 * scale
    if hasattr(w8_mod, "w_scale"):
        scale = w8_mod.w_scale  # W8A8Conv2d
    else:
        scale = w8_mod.scale   # W8Conv2d
    w_fp = w8_mod.weight_int8.to(compute_dtype) * scale.view(-1, 1, 1, 1)
    conv.weight = nn.Parameter(w_fp, requires_grad=False)
    if w8_mod.bias is not None:
        conv.bias = nn.Parameter(w8_mod.bias.data.clone(), requires_grad=False)
    return conv


def _predequantize_linear(w8_mod, compute_dtype):
    """Convert a W8Linear or W8A8Linear back to a standard nn.Linear."""
    linear = nn.Linear(
        w8_mod.in_features, w8_mod.out_features,
        bias=w8_mod.bias is not None,
    )
    if hasattr(w8_mod, "w_scale"):
        scale = w8_mod.w_scale  # W8A8Linear
    else:
        scale = w8_mod.scale   # W8Linear
    w_fp = w8_mod.weight_int8.to(compute_dtype) * scale.view(-1, 1)
    linear.weight = nn.Parameter(w_fp, requires_grad=False)
    if w8_mod.bias is not None:
        linear.bias = nn.Parameter(w8_mod.bias.data.clone(), requires_grad=False)
    return linear


def _predequantize_model(model: nn.Module,
                         compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    """Convert all W8*/W8A8* layers back to native FP Conv2d/Linear in-place.

    This is used on GPUs without INT8 tensor cores (e.g. AMD RDNA3) to
    eliminate the per-inference dequant'FP16 overhead.  The model still
    loads from a 2.94- compressed INT8 checkpoint " we just decompress
    the weights once at load time instead of every forward pass.

    Returns the model with all quantized layers replaced by native FP16
    layers (same inference speed as the original FP16 model).
    """
    converted = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (W8Conv2d, W8A8Conv2d, W8Linear, W8A8Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            if isinstance(module, (W8Conv2d, W8A8Conv2d)):
                setattr(parent, parts[-1],
                        _predequantize_conv(module, compute_dtype))
            else:
                setattr(parent, parts[-1],
                        _predequantize_linear(module, compute_dtype))
            converted += 1
    print(f"  Pre-dequantized {converted} INT8 layers -> native FP16 "
          f"(no runtime dequant overhead)")
    return model


class TRTQDQConv2d(nn.Module):
    """Conv2d export module that emits ONNX Q/DQ nodes for TensorRT.

    It reconstructs the calibrated quantized weights from W8/W8A8 runtime
    modules, then uses PyTorch fake-quant ops because the ONNX exporter lowers
    them to QuantizeLinear/DequantizeLinear.
    """

    def __init__(self, w8_mod, quantize_activation: bool):
        super().__init__()
        self.stride = w8_mod.stride
        self.padding = w8_mod.padding
        self.dilation = w8_mod.dilation
        self.groups = w8_mod.groups
        self.quantize_activation = bool(quantize_activation)
        self.activation_asymmetric = bool(getattr(w8_mod, "is_asymmetric", False))

        if hasattr(w8_mod, "w_scale"):
            w_scale = w8_mod.w_scale.detach().float()
        else:
            w_scale = w8_mod.scale.detach().float()
        weight = w8_mod.weight_int8.detach().float() * w_scale.view(-1, 1, 1, 1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("w_scale", w_scale.clamp(min=1e-8))
        self.register_buffer(
            "w_zero",
            torch.zeros_like(w_scale, dtype=torch.int32),
        )
        if w8_mod.bias is not None:
            self.bias = nn.Parameter(w8_mod.bias.detach().float().clone(),
                                     requires_grad=False)
        else:
            self.bias = None

        x_scale = getattr(w8_mod, "x_scale", None)
        if x_scale is None:
            x_scale = torch.tensor(1.0)
        self.register_buffer(
            "x_scale",
            x_scale.detach().float().reshape(()).clamp(min=1e-8),
        )
        x_zero = getattr(w8_mod, "x_zero", None)
        if x_zero is None:
            x_zero = torch.tensor(0.0)
        # Convert the existing asymmetric min/scale form into ONNX affine
        # zero-point form. Clamping keeps the graph valid for uint8 Q/DQ.
        zp = torch.round(-x_zero.detach().float() / self.x_scale).clamp(0, 255)
        self.register_buffer("x_zero_point", zp.to(torch.int32).reshape(()))

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activation:
            return x
        scale = self.x_scale.to(device=x.device)
        if self.activation_asymmetric:
            zp = self.x_zero_point.to(device=x.device)
            return torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)
        zp = torch.tensor(0, dtype=torch.int32, device=x.device)
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)

    def _quantize_weight(self) -> torch.Tensor:
        scale = self.w_scale.to(device=self.weight.device)
        zero = self.w_zero.to(device=self.weight.device)
        return torch.fake_quantize_per_channel_affine(
            self.weight,
            scale,
            zero,
            0,
            -128,
            127,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._quantize_input(x)
        w = self._quantize_weight().to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.conv2d(x, w, bias, self.stride, self.padding,
                        self.dilation, self.groups)


class TRTQDQLinear(nn.Module):
    """Linear export module that emits ONNX Q/DQ nodes for TensorRT."""

    def __init__(self, w8_mod, quantize_activation: bool):
        super().__init__()
        self.quantize_activation = bool(quantize_activation)
        self.activation_asymmetric = bool(getattr(w8_mod, "is_asymmetric", False))

        if hasattr(w8_mod, "w_scale"):
            w_scale = w8_mod.w_scale.detach().float()
        else:
            w_scale = w8_mod.scale.detach().float()
        weight = w8_mod.weight_int8.detach().float() * w_scale.view(-1, 1)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("w_scale", w_scale.clamp(min=1e-8))
        self.register_buffer(
            "w_zero",
            torch.zeros_like(w_scale, dtype=torch.int32),
        )
        if w8_mod.bias is not None:
            self.bias = nn.Parameter(w8_mod.bias.detach().float().clone(),
                                     requires_grad=False)
        else:
            self.bias = None

        x_scale = getattr(w8_mod, "x_scale", None)
        if x_scale is None:
            x_scale = torch.tensor(1.0)
        self.register_buffer(
            "x_scale",
            x_scale.detach().float().reshape(()).clamp(min=1e-8),
        )
        x_zero = getattr(w8_mod, "x_zero", None)
        if x_zero is None:
            x_zero = torch.tensor(0.0)
        zp = torch.round(-x_zero.detach().float() / self.x_scale).clamp(0, 255)
        self.register_buffer("x_zero_point", zp.to(torch.int32).reshape(()))

    def _quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activation:
            return x
        scale = self.x_scale.to(device=x.device)
        if self.activation_asymmetric:
            zp = self.x_zero_point.to(device=x.device)
            return torch.fake_quantize_per_tensor_affine(x, scale, zp, 0, 255)
        zp = torch.tensor(0, dtype=torch.int32, device=x.device)
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)

    def _quantize_weight(self) -> torch.Tensor:
        scale = self.w_scale.to(device=self.weight.device)
        zero = self.w_zero.to(device=self.weight.device)
        return torch.fake_quantize_per_channel_affine(
            self.weight,
            scale,
            zero,
            0,
            -128,
            127,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._quantize_input(x)
        w = self._quantize_weight().to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def _convert_model_to_tensorrt_qdq(model: nn.Module) -> nn.Module:
    """Replace W8/W8A8 runtime wrappers with TensorRT-friendly Q/DQ modules."""
    converted_w8a8 = 0
    converted_w8 = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, (W8Conv2d, W8A8Conv2d, W8Linear, W8A8Linear)):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        quantize_activation = isinstance(module, (W8A8Conv2d, W8A8Linear))
        if isinstance(module, (W8Conv2d, W8A8Conv2d)):
            replacement = TRTQDQConv2d(module, quantize_activation)
        else:
            replacement = TRTQDQLinear(module, quantize_activation)
        setattr(parent, parts[-1], replacement)
        if quantize_activation:
            converted_w8a8 += 1
        else:
            converted_w8 += 1
    if converted_w8a8 or converted_w8:
        print(
            "TensorRT Q/DQ export: "
            f"{converted_w8a8} W8A8 layers, {converted_w8} W8A16 layers"
        )
    return model


def _is_memory_bound_1x1(module: nn.Conv2d, channel_threshold: int = 32) -> bool:
    """Decide if a Conv2d is a memory-bound 1-1 that benefits from INT8 IO.

    Our Triton benchmark showed 1.74-2.28- speedup for 1-1 convs with
    max(C_in, C_out)  32 (SFT layers) but 0.54-0.91- for larger channels
    (AGCM / condition network).  Following the FSR4 approach of using INT8
    as *memory compression* rather than compute format, we apply W8A8 only
    to these bandwidth-sensitive layers.
    """
    if module.kernel_size != (1, 1):
        return False
    return max(module.in_channels, module.out_channels) <= channel_threshold


def _quantize_model_mixed(model: nn.Module,
                          compute_dtype: torch.dtype = torch.float16,
                          channel_threshold: int = 32,
                          fp16_layers: list = None) -> nn.Module:
    """Selective mixed INT8: W8A8 for memory-bound 1-1 convs, W8A16 otherwise.

    Strategy inspired by FSR4's INT8 path (DP4A-based memory compression):
      * SFT 1-1 convs (32 ch) ' W8A8  (INT8 weights + activations)
      * 3-3 convs / large 1-1s  ' W8A16 (INT8 weights only)
      * Linear layers            ' W8A16 (INT8 weights only)
      * Explicitly exempted layers ' FP16 (leave native Conv2d / Linear)
    """
    fp16_set = set(fp16_layers or [])
    w8a8_count = 0
    w8_count = 0
    fp16_count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            if name in fp16_set:
                fp16_count += 1
            elif isinstance(module, nn.Conv2d) and _is_memory_bound_1x1(
                    module, channel_threshold):
                setattr(parent, parts[-1], W8A8Conv2d(module, compute_dtype))
                w8a8_count += 1
            elif isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1], W8Conv2d(module, compute_dtype))
                w8_count += 1
            else:
                setattr(parent, parts[-1], W8Linear(module, compute_dtype))
                w8_count += 1
    total = w8a8_count + w8_count + fp16_count
    print(f"  Mixed INT8: {w8a8_count} layers -> W8A8, {w8_count} layers -> W8A16, "
          f"{fp16_count} layers -> FP16 ({total} total, {compute_dtype} compute)")
    return model


def _quantize_model_mixed_v2(model: nn.Module,
                              compute_dtype: torch.dtype = torch.float16,
                              w8a8_layers: list = None,
                              fp16_layers: list = None,
                              asymmetric: bool = True) -> nn.Module:
    """Sensitivity-based mixed INT8: W8A8 for specified layers, W8A16 otherwise.

    Unlike v1 which uses a simple channel-count heuristic, this version
    accepts an explicit list of layer names determined by per-layer
    sensitivity analysis.  All W8A8 layers use asymmetric activation
    quantization [0, 255] with zero-point by default.

    Args:
        model: FP32 model to quantize (modified in-place).
        compute_dtype: Runtime precision for dequantized values.
        w8a8_layers: List of layer names to assign W8A8.
        fp16_layers: List of layer names to keep as native FP16/FP32 layers.
        asymmetric: Use asymmetric activation quantization for W8A8 layers.
    """
    w8a8_set = set(w8a8_layers or [])
    fp16_set = set(fp16_layers or [])
    w8a8_count = 0
    w8_count = 0
    fp16_count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)

            if name in fp16_set:
                fp16_count += 1
            elif name in w8a8_set:
                if isinstance(module, nn.Conv2d):
                    setattr(parent, parts[-1],
                            W8A8Conv2d(module, compute_dtype,
                                       asymmetric=asymmetric))
                else:
                    setattr(parent, parts[-1],
                            W8A8Linear(module, compute_dtype,
                                       asymmetric=asymmetric))
                w8a8_count += 1
            elif isinstance(module, nn.Conv2d):
                setattr(parent, parts[-1], W8Conv2d(module, compute_dtype))
                w8_count += 1
            else:
                setattr(parent, parts[-1], W8Linear(module, compute_dtype))
                w8_count += 1

    total = w8a8_count + w8_count + fp16_count
    mode = "asymmetric" if asymmetric else "symmetric"
    print(f"  Mixed INT8 v2: {w8a8_count} layers -> W8A8 ({mode}), "
          f"{w8_count} layers -> W8A16, {fp16_count} layers -> FP16 "
          f"({total} total, {compute_dtype} compute)")
    return model


def calibrate_w8a8(model: nn.Module, calibration_inputs: list,
                   method: str = "max", percentile: float = 99.9,
                   percentile_low: float = 0.1,
                   max_samples: int = 200000) -> None:
    """Run calibration data through the model to determine activation scales.

    method:
      - "max": max-abs (symmetric) or min/max (asymmetric)
      - "percentile": percentile clipping for activations

    For symmetric layers:
      - max: x_scale = max_abs / 127
      - percentile: x_scale = perc(|x|) / 127
    For asymmetric layers:
      - max: x_zero = min, x_scale = (max - min) / 255
      - percentile: x_zero = perc_low(x), x_scale = (perc_high - perc_low) / 255
    """
    # Attach hooks to record activation ranges
    hooks = []
    stats = {}  # name -> dict of running stats

    def _sample_flat(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        if flat.numel() <= max_samples:
            return flat
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        return flat[idx]

    def _make_hook(name, is_asym):
        def _hook(module, inp, out):
            x = inp[0].detach()
            s = stats.get(name)
            if s is None:
                s = {"max_abs": 0.0, "min": float("inf"), "max": float("-inf"),
                     "p_abs": 0.0, "p_min": float("inf"), "p_max": float("-inf")}
                stats[name] = s

            if method == "percentile":
                x_s = _sample_flat(x.float())
                if is_asym:
                    p_low = torch.quantile(x_s, percentile_low / 100.0).item()
                    p_high = torch.quantile(x_s, percentile / 100.0).item()
                    s["p_min"] = min(s["p_min"], p_low)
                    s["p_max"] = max(s["p_max"], p_high)
                else:
                    p_abs = torch.quantile(x_s.abs(), percentile / 100.0).item()
                    s["p_abs"] = max(s["p_abs"], p_abs)

            abs_max = x.abs().amax().item()
            x_min = x.amin().item()
            x_max = x.amax().item()
            s["max_abs"] = max(s["max_abs"], abs_max)
            s["min"] = min(s["min"], x_min)
            s["max"] = max(s["max"], x_max)
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, (W8A8Conv2d, W8A8Linear)):
            hooks.append(module.register_forward_hook(
                _make_hook(name, module.is_asymmetric)
            ))

    # Run calibration
    model.eval()
    with torch.inference_mode():
        for inp in calibration_inputs:
            model(inp)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Set activation scales
    calibrated = 0
    asym_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (W8A8Conv2d, W8A8Linear)) and name in stats:
            s = stats[name]
            if module.is_asymmetric:
                if method == "percentile":
                    x_min = s["p_min"] if not math.isinf(s["p_min"]) else s["min"]
                    x_max = s["p_max"] if not math.isinf(s["p_max"]) else s["max"]
                else:
                    x_min, x_max = s["min"], s["max"]
                x_range = max(x_max - x_min, 1e-8)
                module.x_scale.fill_(x_range / 255.0)
                module.x_zero.fill_(x_min)
                asym_count += 1
            else:
                if method == "percentile":
                    val = s["p_abs"] if s["p_abs"] != 0.0 else s["max_abs"]
                else:
                    val = s["max_abs"]
                module.x_scale.fill_(max(val, 1e-8) / 127.0)
            calibrated += 1

    sym_count = calibrated - asym_count
    print(f"  Calibrated {calibrated} layers "
          f"({asym_count} asymmetric, {sym_count} symmetric)")

def _get_gpu_info() -> str:
    """Return GPU name string, or empty if unavailable."""
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return ""


class HDRTVNetTorch:
    """PyTorch inference wrapper for HDRTVNet with platform-aware optimizations.

    Optimizations applied automatically per platform:
      * torch.inference_mode() (lower overhead than no_grad).
      * Pre-allocated GPU tensors to avoid per-frame allocation.
      * AMD/CUDA PyTorch path: cudnn.benchmark + channels_last.
      * AMD PyTorch path: torch.compile with max-autotune when available.
      * Optional CUDA-graph replay for static-shape inputs.
      * Pre-dequantize INT8 weights on PyTorch backends without tensor cores (auto).
    """

    def __init__(self, model_path, device="auto", precision="auto",
                 compile_model=True, force_compile=False, compile_mode="auto",
                 use_cuda_graphs=False, force_channels_last=False,
                 predequantize="auto", hg_weights=None, use_hg=True):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.precision = self._resolve_precision(precision, self.device)
        self._use_cuda = self.device.type == "cuda"
        self._dtype = {"fp16": torch.float16}.get(
            self.precision, torch.float32
        )
        self._hg_weights_explicit = hg_weights is not None
        self._hg_weights = hg_weights or _DEFAULT_HG_WEIGHTS
        self._use_hg = bool(use_hg)
        self._is_flat_model = False  # True for old-style FX quantized models
        self._is_w8_model = False    # True for GPU W8 weight-only INT8
        self._predequantize = predequantize  # "auto", True, or False

        # ---- Print platform info ------------------------------------------
        if self._use_cuda:
            gpu_name = _get_gpu_info()
            backend = "ROCm" if _IS_ROCM else "CUDA"
            print(f"GPU: {gpu_name} ({backend})")

        self.model = self._load_model(model_path)

        # ---- Platform-specific: channels_last + cudnn.benchmark -----------
        # AMD continues to use the PyTorch runtime, with channels_last applied
        # by default as the AMD fast path requested by the GUI/runtime layer.
        if self._use_cuda and (_IS_ROCM or _IS_NVIDIA or force_channels_last):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            self.model = self.model.to(memory_format=torch.channels_last)
            self._use_channels_last = True
            if _IS_ROCM:
                print("ROCm: cudnn.benchmark + channels_last enabled")
            else:
                print("NVIDIA: cudnn.benchmark + channels_last enabled")
        else:
            self._use_channels_last = False

        # ---- torch.compile (PyTorch 2.x) ----------------------------------
        # NVIDIA / ROCm-Linux: auto-enabled when Triton is available.
        # ROCm-Windows: auto-enabled when HIP SDK is detected; use
        #   --force-compile to override if auto-detection misses it.
        self._compiled = False
        _rocm_win_no_sdk = (
            _IS_ROCM and os.name == "nt"
            and not force_compile and not _HAS_HIP_SDK
        )
        if compile_model and _HAS_COMPILE and self._use_cuda and not _IS_NVIDIA:
            if _rocm_win_no_sdk:
                print("torch.compile skipped on ROCm-Windows \u2014 HIP SDK "
                      "not detected.\n"
                      "  Install AMD HIP SDK or use --force-compile")
            elif not _HAS_TRITON:
                print("torch.compile skipped \u2014 Triton not installed.\n"
                      "  pip install triton")
            else:
                if _IS_ROCM and os.name == "nt":
                    src = "--force-compile" if force_compile else "HIP SDK auto-detected"
                    print(f"ROCm-Windows: {src}, enabling torch.compile...")
                try:
                    if compile_mode == "auto":
                        compile_mode = "max-autotune"
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False,
                    )
                    self._compiled = True
                    print(f"torch.compile enabled (mode={compile_mode})")
                except Exception as exc:
                    print(f"torch.compile setup failed: {exc}")

        self.expected_hw = None
        self.is_static_input_model = False

        # ---- Pre-allocated buffer state ------------------------------------
        self._buf_hw = None
        self._gpu_input = None       # persistent GPU tensor (1,3,H,W)
        self._gpu_cond = None        # persistent GPU tensor (1,3,H//4,W//4)
        self._pin_input = None       # pinned host staging tensor (H,W,3) uint8
        self._pin_output = None      # pinned host tensor for D2H (H,W,3) uint8
        self._d2h_stream = (
            torch.cuda.Stream() if self._use_cuda else None
        )

        # ---- CUDA graph state (optional) -----------------------------------
        self._use_cuda_graphs = (
            use_cuda_graphs and self._use_cuda and _HAS_CUDA_GRAPHS
        )
        self._graph = None
        self._graph_input = None
        self._graph_cond = None
        self._graph_output = None
        self._graph_hw = None

        print(f"PyTorch device : {self.device}")
        print(f"PyTorch precision: {self.precision}")

    # -----------------------------------------------------------------------
    # Device / precision helpers
    # -----------------------------------------------------------------------
    def _resolve_device(self, device):
        mode = device.lower()
        if mode == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm device not available for PyTorch.")
            return torch.device("cuda:0")
        if mode == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be one of: auto, cuda, cpu")

    def _resolve_precision(self, precision, device):
        p = precision.lower()
        if p not in {"auto", "fp16", "fp32", "int8-full", "int8-mixed"}:
            raise ValueError("precision must be one of: auto, fp16, fp32, int8-full, int8-mixed")
        if p in ("int8-full", "int8-mixed"):
            return p  # runs on GPU via W8A8 dequant
        if p == "auto":
            return "fp16" if device.type == "cuda" else "fp32"
        if device.type != "cuda" and p == "fp16":
            return "fp32"
        return p

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    def _load_int8_model(self, model_path):
        """Load a GPU INT8 quantized model (W8A8 full or mixed).

        The checkpoint contains a state_dict where Conv2d and Linear layers
        have been replaced with W8A8* equivalents.  We rebuild the
        architecture with the correct replacement layers.
        """
        checkpoint = torch.load(model_path, map_location="cpu",
                                weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                f"{model_path} is not an INT8 checkpoint.\n"
                "  Re-run: python scripts/quantize/quantize_int8_full.py or "
                "python scripts/quantize/quantize_int8_mixed.py"
            )
        arch = checkpoint.get("architecture", {})
        compute_dtype_str = checkpoint.get("compute_dtype", "torch.float16")
        compute_dtype = torch.float16 if "16" in compute_dtype_str else torch.float32
        if self.device.type != "cuda" and compute_dtype == torch.float16:
            # CPU kernels do not reliably support FP16 for all ops used here
            # (e.g., bicubic antialias in preprocess). Force FP32 on CPU.
            print(
                "WARNING: INT8 checkpoint requests FP16 compute on CPU; "
                "falling back to FP32 compute for compatibility."
            )
            compute_dtype = torch.float32
        self._dtype = compute_dtype
        quant_type = checkpoint.get("quantization", "w8_weight_only")
        state_dict = checkpoint.get("state_dict", {})
        try:
            sd_numel = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
        except Exception:
            sd_numel = 0

        if compute_dtype == torch.float32 and self.device.type == "cuda":
            print(
                "WARNING: INT8 checkpoint uses FP32 compute_dtype; this is slower"
                " on GPU. Consider re-quantizing with FP16 compute."
            )

        # Build base architecture (HG composite by default)
        use_hg = bool(arch.get("use_hg", True))
        if not self._use_hg and use_hg:
            raise ValueError(
                "INT8 HG checkpoint provided but HG is disabled. "
                "Enable HG or use non-HG INT8 checkpoints."
            )
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
        # Sanity-check no-HG INT8 checkpoints against expected parameter count.
        if (not use_hg) and sd_numel:
            try:
                expected_numel = sum(
                    v.numel() for v in model.state_dict().values() if hasattr(v, "numel")
                )
            except Exception:
                expected_numel = 0
            if expected_numel and sd_numel < (0.5 * expected_numel):
                print(
                    "WARNING: INT8 no-HG checkpoint is much smaller than expected; "
                    "this may be an incomplete/stub model. "
                    "Regenerate a full-size no-HG INT8 checkpoint for fair speed."
                )

        # Replace Conv2d / Linear with correct quantized equivalents
        if quant_type == "w8a8_mixed":
            w8a8_layers = checkpoint.get("w8a8_layers", None)
            fp16_layers = checkpoint.get("fp16_layers", None)
            use_asym = checkpoint.get(
                "activation_quant", "symmetric") == "asymmetric"
            if w8a8_layers is not None:
                # v2: explicit layer list from sensitivity analysis
                _quantize_model_mixed_v2(model, compute_dtype,
                                         w8a8_layers=w8a8_layers,
                                         fp16_layers=fp16_layers,
                                         asymmetric=use_asym)
            else:
                # v1: channel-threshold heuristic (legacy checkpoints)
                _quantize_model_mixed(model, compute_dtype,
                                      channel_threshold=checkpoint.get(
                                          "channel_threshold", 32),
                                      fp16_layers=fp16_layers)
        elif quant_type == "w8a8_full":
            use_asym = checkpoint.get(
                "activation_quant", "symmetric") == "asymmetric"
            _quantize_model_w8a8(model, compute_dtype, asymmetric=use_asym)
        else:
            raise ValueError(
                f"Unknown quantization type '{quant_type}' in checkpoint.\n"
                "  Supported: w8a8_full, w8a8_mixed")

        # Load the quantized state_dict
        model.load_state_dict(state_dict, strict=True)
        # Cast ALL remaining layers (InstanceNorm, etc.) to compute_dtype.
        # On ROCm, InstanceNorm in FP32 triggers broken MIOpen JIT compilation.
        # On CUDA, this is a harmless optimization (avoids unnecessary casts).
        model = model.to(dtype=compute_dtype, device=self.device)
        model.eval()

        # ---- Pre-dequantize on GPUs without INT8 tensor cores -------------
        # Auto: AMD RDNA3 (no native INT8 conv) ' pre-dequantize
        #       NVIDIA Turing+ (sm >= 75) ' keep INT8 for tensor core speedup
        should_predequantize = self._predequantize
        if should_predequantize == "auto":
            if _IS_ROCM:
                # AMD has no native INT8 conv kernels in MIOpen
                should_predequantize = True
                print("Auto-detected AMD GPU: pre-dequantizing INT8 ' FP16 "
                      "(no native INT8 conv on RDNA3)")
            elif _IS_NVIDIA:
                props = torch.cuda.get_device_properties(0)
                has_int8_tc = (props.major > 7 or
                               (props.major == 7 and props.minor >= 5))
                should_predequantize = not has_int8_tc
                if should_predequantize:
                    print(f"NVIDIA sm_{props.major}{props.minor}: no INT8 "
                          f"tensor cores, pre-dequantizing ' FP16")
            else:
                should_predequantize = False

        if should_predequantize and should_predequantize != "auto":
            _predequantize_model(model, compute_dtype)
            self._is_w8_model = False  # now a regular FP16 model
            label = {"w8a8_full": "W8A8", "w8a8_mixed": "Mixed W8A8/W8A16"}.get(
                quant_type, quant_type)
            if quant_type == "w8a8_mixed" and checkpoint.get("fp16_layers"):
                label = "Mixed W8A8/W8A16/FP16"
            print(f"Loaded {label} INT8 checkpoint ' pre-dequantized to FP16 "
                  f"(compressed storage, native FP16 speed)")
        else:
            self._is_w8_model = True
            label = {"w8a8_full": "W8A8", "w8a8_mixed": "Mixed W8A8/W8A16"}.get(
                quant_type, quant_type)
            if quant_type == "w8a8_mixed" and checkpoint.get("fp16_layers"):
                label = "Mixed W8A8/W8A16/FP16"
            print(f"Loaded {label} INT8 model (compute_dtype={compute_dtype})")
        return model

    def _resolve_hg_weights(self, model_path):
        """Find HG_weights.pth from common locations."""
        candidates = []
        seen = set()

        def _add(path):
            if not path:
                return
            abs_path = os.path.abspath(os.path.expanduser(str(path)))
            if abs_path in seen:
                return
            seen.add(abs_path)
            candidates.append(abs_path)

        # User override (if provided) first.
        _add(self._hg_weights)
        # Adjacent to the selected model checkpoint.
        _add(os.path.join(os.path.dirname(os.path.abspath(model_path)), "HG_weights.pth"))
        # Repo-default runtime path.
        _add(_DEFAULT_HG_WEIGHTS)
        # Relative path from current working directory (when launched outside repo root).
        _add(os.path.join(os.getcwd(), "src", "models", "weights", "HG_weights.pth"))

        for path in candidates:
            if os.path.isfile(path):
                return path, candidates
        return None, candidates

    def _load_model(self, model_path):
        # INT8 quantized models use a different loading path
        if self.precision in ("int8-full", "int8-mixed"):
            return self._load_int8_model(model_path)

        ext = os.path.splitext(model_path)[1].lower()
        if ext in {".pt", ".ts"}:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
        else:
            use_hg = bool(self._use_hg)
            hg_weights_path = None
            searched_hg_paths = []
            if use_hg:
                hg_weights_path, searched_hg_paths = self._resolve_hg_weights(model_path)
                if hg_weights_path:
                    self._hg_weights = hg_weights_path
                elif self._hg_weights_explicit:
                    searched = "\n".join(f"  - {p}" for p in searched_hg_paths)
                    raise FileNotFoundError(
                        f"HG weights not found: {self._hg_weights}\n"
                        "  Searched paths:\n"
                        f"{searched}\n"
                        "  Check --hg-weights or disable HG with --use-hg 0."
                    )
                else:
                    searched = "\n".join(f"  - {p}" for p in searched_hg_paths)
                    print(
                        "WARNING: HG weights not found; continuing with no-HG model.\n"
                        "  Searched paths:\n"
                        f"{searched}\n"
                        "  To enable HG, place HG_weights.pth under "
                        "src/models/weights/ or pass --hg-weights."
                    )
                    use_hg = False

            self._use_hg = use_hg

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
                ).to(self.device)

                base_state = torch.load(model_path, map_location=self.device,
                                        weights_only=True)
                cleaned = {}
                for key, value in base_state.items():
                    cleaned[key[7:] if key.startswith("module.") else key] = value
                model.base.load_state_dict(cleaned, strict=True)

                hg_state = torch.load(hg_weights_path, map_location=self.device)
                if isinstance(hg_state, dict) and "state_dict" in hg_state:
                    hg_state = hg_state["state_dict"]
                model.hg.load_state_dict(hg_state, strict=True)
                model.eval()
            else:
                model = Ensemble_AGCM_LE(
                    classifier="color_condition",
                    cond_c=6,
                    in_nc=3,
                    out_nc=3,
                    nf=32,
                    act_type="relu",
                    weighting_network=False,
                ).to(self.device)

                state_dict = torch.load(model_path, map_location=self.device,
                                        weights_only=True)
                cleaned = {}
                for key, value in state_dict.items():
                    cleaned[key[7:] if key.startswith("module.") else key] = value
                model.load_state_dict(cleaned, strict=True)
                model.eval()

        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half()
        else:
            model = model.float()

        return model

    # -----------------------------------------------------------------------
    # Buffer management " allocate once, reuse every frame
    # -----------------------------------------------------------------------
    def _ensure_buffers(self, h, w):
        """Allocate / reallocate persistent GPU tensors when resolution changes.
        For the common case (fixed resolution) this is a no-op after the first
        frame."""
        if self._buf_hw == (h, w):
            return
        self._buf_hw = (h, w)
        cond_h, cond_w = max(1, h // 4), max(1, w // 4)

        # Persistent GPU tensors " avoids torch.empty() + .to(device) per frame
        mem_fmt = (torch.channels_last if self._use_channels_last
                   else torch.contiguous_format)
        self._gpu_input = torch.empty(
            (1, 3, h, w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)
        self._gpu_cond = torch.empty(
            (1, 3, cond_h, cond_w), dtype=self._dtype, device=self.device,
        ).to(memory_format=mem_fmt)

        # Pinned host buffers " page-locked memory makes non_blocking=True
        # actually overlap H2D/D2H with GPU compute
        if self._use_cuda:
            self._pin_input = torch.empty(
                (h, w, 3), dtype=torch.uint8, pin_memory=True
            )
            self._pin_output = torch.empty(
                (h, w, 3), dtype=torch.uint8, pin_memory=True
            )

        # Invalidate any cached CUDA graph on resolution change
        self._graph = None
        self._graph_hw = None

    # -----------------------------------------------------------------------
    # Preprocess " keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._ensure_buffers(h, w)

        # Copy frame into pinned staging buffer, then async H2D
        if self._use_cuda and self._pin_input is not None:
            self._pin_input.copy_(torch.from_numpy(frame_bgr))  # CPU'pinned (memcpy)
            raw = self._pin_input.to(
                device=self.device, non_blocking=True            # pinned'GPU (async DMA)
            )
        else:
            raw = torch.from_numpy(frame_bgr).to(
                device=self.device, non_blocking=self._use_cuda
            )
        # (H,W,3) uint8 ' BGR'RGB via channel flip ' CHW ' add batch ' fp ' /255
        raw = raw.flip(2)                                  # BGR ' RGB
        raw = raw.permute(2, 0, 1).unsqueeze(0)           # (1,3,H,W)
        self._gpu_input.copy_(
            raw.to(dtype=self._dtype).mul_(1.0 / 255.0),
            non_blocking=self._use_cuda,
        )

        # Condition tensor (0.25- spatial)
        try:
            cond = F.interpolate(
                self._gpu_input,
                scale_factor=0.25,
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
                antialias=True,
            )
        except TypeError:
            cond = F.interpolate(
                self._gpu_input,
                scale_factor=0.25,
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
        self._gpu_cond.copy_(cond, non_blocking=self._use_cuda)

        return self._gpu_input, self._gpu_cond

    # -----------------------------------------------------------------------
    # Inference " CUDA graph replay when possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def infer(self, input_cond):
        tensor, cond = input_cond

        # --- CUDA graph path (static shapes only) -------------------------
        if self._use_cuda_graphs and self._use_cuda:
            h, w = tensor.shape[2], tensor.shape[3]
            if self._graph is not None and self._graph_hw == (h, w):
                self._graph_input.copy_(tensor)
                self._graph_cond.copy_(cond)
                self._graph.replay()
                return self._graph_output
            else:
                self._graph_hw = (h, w)
                self._graph_input = tensor.clone()
                self._graph_cond = cond.clone()

                # Warmup (required before recording)
                for _ in range(3):
                    _ = self.model((self._graph_input, self._graph_cond))
                torch.cuda.synchronize()

                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._graph_output = self.model(
                        (self._graph_input, self._graph_cond)
                    )
                self._graph_input.copy_(tensor)
                self._graph_cond.copy_(cond)
                self._graph.replay()
                return self._graph_output

        # --- Flat model path (old FX quantized) ----------------------------
        if self._is_flat_model:
            return self.model(tensor, cond)

        # --- Eager / compiled path -----------------------------------------
        try:
            return self.model((tensor, cond))
        except Exception as exc:
            if self._compiled:
                print(f"torch.compile failed at runtime, reverting to eager: {exc}")
                self.model = self.model._orig_mod
                self._compiled = False
                return self.model((tensor, cond))
            raise

    # -----------------------------------------------------------------------
    # Postprocess " keep as much work on GPU as possible
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def postprocess(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        # All math on GPU: clamp, scale, quantize, channel-flip (RGB'BGR)
        t = output.squeeze(0)                      # (3,H,W)  fp16/fp32
        t = t.clamp_(0.0, 1.0).mul_(255.0).add_(0.5)  # round-to-nearest
        t = t.to(dtype=torch.uint8)                # quantize on GPU
        t = t.flip(0)                              # RGB'BGR via channel flip
        t = t.permute(1, 2, 0).contiguous()        # CHW ' HWC, contiguous for cv2

        # Async D2H into pinned host buffer (avoids implicit sync)
        if self._use_cuda and self._pin_output is not None:
            self._pin_output.copy_(t, non_blocking=True)   # GPU'pinned (async DMA)
            torch.cuda.current_stream().synchronize()       # ensure D2H completes
            return self._pin_output.numpy()                 # zero-copy view
        return t.cpu().numpy()                              # fallback: CPU path

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def process(self, frame_bgr):
        tensor, cond = self.preprocess(frame_bgr)
        out = self.infer((tensor, cond))
        return self.postprocess(out)

    @torch.inference_mode()
    def process_timed(self, frame_bgr):
        t0 = time.perf_counter()
        tensor, cond = self.preprocess(frame_bgr)
        if self._use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = self.infer((tensor, cond))
        if self._use_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        output = self.postprocess(out)
        t3 = time.perf_counter()

        return output, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0

    # -----------------------------------------------------------------------
    # Compile cache warmup - pre-compile Triton kernels for a given resolution
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def warmup_compile(self, width=1920, height=1080):
        """Run a dummy frame through the compiled model to trigger and cache
        Triton kernel compilation for the given resolution.

        After this call, any video at the same resolution will start
        inference immediately without recompilation.  Triton persists
        compiled kernels to its on-disk cache, so subsequent *process
        launches* also skip compilation.

        Only meaningful when torch.compile is active; returns immediately
        otherwise.
        """
        if not self._compiled:
            return

        print(
            f"Warming up torch.compile cache for {width}x{height} - "
            "this may take several minutes on first run...",
            flush=True,
        )
        t0 = time.perf_counter()
        heartbeat_stop = threading.Event()

        def _heartbeat():
            # Keep the compile dialog alive with periodic progress notes.
            while not heartbeat_stop.wait(10.0):
                elapsed = time.perf_counter() - t0
                print(
                    f"[compile] still compiling {width}x{height} "
                    f"({elapsed:.0f}s elapsed) ...",
                    flush=True,
                )

        heartbeat = threading.Thread(
            target=_heartbeat,
            name="hdrtvnet-warmup-heartbeat",
            daemon=True,
        )
        heartbeat.start()

        try:
            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            self.process(dummy)  # triggers full compile -> Triton disk cache
            if self._use_cuda:
                torch.cuda.synchronize()
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=0.5)

        elapsed = time.perf_counter() - t0
        print(
            f"Compile cache warm - {width}x{height} ready ({elapsed:.1f}s)",
            flush=True,
        )

    def end_profiling(self):
        return None


def _sanitize_engine_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text or "").strip())
    return token.strip(".-") or "model"


def _engine_cache_dir() -> str:
    root = os.environ.get(
        "HDRTVNET_TENSORRT_CACHE",
        os.path.join(_ROOT, "src", "models", "engines"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def tensorrt_engine_path(
    model_path: str,
    width: int,
    height: int,
    mode: str,
) -> str:
    model_name = _sanitize_engine_token(pathlib.Path(model_path).stem)
    resolution = f"{int(width)}x{int(height)}"
    mode_name = _sanitize_engine_token(mode)
    return os.path.join(_engine_cache_dir(), f"{model_name}_{resolution}_{mode_name}.engine")


def tensorrt_onnx_path(
    model_path: str,
    width: int,
    height: int,
    mode: str,
) -> str:
    return os.path.splitext(tensorrt_engine_path(model_path, width, height, mode))[0] + ".onnx"


def tensorrt_mode_name(precision: str, mode: str) -> str:
    mode_name = str(mode or precision or "mode")
    if str(precision).startswith("int8") and "qdq" not in mode_name.lower():
        mode_name = f"{mode_name}_qdqv1"
    return mode_name


class _ONNXExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, flat_model: bool = False):
        super().__init__()
        self.model = model
        self.flat_model = bool(flat_model)

    def forward(self, tensor: torch.Tensor, cond: torch.Tensor):
        if self.flat_model:
            out = self.model(tensor, cond)
        else:
            out = self.model((tensor, cond))
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def _export_tensorrt_onnx_from_model(
    *,
    model: nn.Module,
    onnx_path: str,
    width: int,
    height: int,
    dtype: torch.dtype,
    device: torch.device,
    precision: str,
    flat_model: bool,
    force: bool = False,
) -> str:
    if os.path.isfile(onnx_path) and not force:
        print(f"TensorRT ONNX cache hit: {onnx_path}")
        return onnx_path

    export_model = getattr(model, "_orig_mod", model).eval()
    if str(precision).startswith("int8"):
        export_model = _convert_model_to_tensorrt_qdq(export_model).eval()
    wrapper = _ONNXExportWrapper(export_model, flat_model=flat_model).eval()
    h, w = int(height), int(width)
    cond_h, cond_w = max(1, h // 4), max(1, w // 4)
    tensor = torch.zeros((1, 3, h, w), dtype=dtype, device=device)
    cond = torch.zeros((1, 3, cond_h, cond_w), dtype=dtype, device=device)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"TensorRT ONNX export: {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (tensor, cond),
            onnx_path,
            input_names=["input", "cond"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
        )
    return onnx_path


class HDRTVNetTensorRT(HDRTVNetTorch):
    """TensorRT runtime with PyTorch preprocessing/postprocessing parity.

    Engines are built lazily for the selected model/resolution/mode and then
    reused from disk. INT8 checkpoints are loaded from the existing quantized
    files, then their W8/W8A8 wrappers are converted to ONNX Q/DQ export
    modules so TensorRT receives an explicit-quantization graph. No TensorRT
    runtime calibration is performed.
    """

    def __init__(
        self,
        model_path,
        device="auto",
        precision="auto",
        engine_width=1920,
        engine_height=1080,
        mode_name="fp16",
        hg_weights=None,
        use_hg=True,
    ):
        if not _IS_NVIDIA:
            raise RuntimeError("TensorRT backend is only enabled for NVIDIA CUDA devices.")

        self._engine_width = int(engine_width)
        self._engine_height = int(engine_height)
        self._engine_mode_name = tensorrt_mode_name(precision, mode_name)
        self.engine_path = tensorrt_engine_path(
            model_path,
            self._engine_width,
            self._engine_height,
            self._engine_mode_name,
        )
        self.onnx_path = tensorrt_onnx_path(
            model_path,
            self._engine_width,
            self._engine_height,
            self._engine_mode_name,
        )
        self._trt_runtime = None
        self._trt_engine = None
        self._trt_context = None
        self._trt_input_names = []
        self._trt_output_names = []
        self._trt_output = None
        self._trt_output_shape = None

        super().__init__(
            model_path,
            device=device,
            precision=precision,
            compile_model=False,
            force_compile=False,
            compile_mode="default",
            use_cuda_graphs=False,
            force_channels_last=False,
            predequantize=False,
            hg_weights=hg_weights,
            use_hg=use_hg,
        )

        # TensorRT bindings use dense NCHW buffers; keep preprocessing and
        # postprocessing identical, but make the staging tensors contiguous.
        self._use_channels_last = False

        if not os.path.isfile(self.engine_path):
            print(f"TensorRT engine cache miss: {self.engine_path}")
            self._build_engine_from_loaded_model()
        else:
            print(f"TensorRT engine cache hit: {self.engine_path}")

        # Runtime must not retain the PyTorch model after the engine exists.
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._load_engine()
        self._compiled = False
        print(f"TensorRT engine : {self.engine_path}")

    def _build_engine_from_loaded_model(self) -> None:
        try:
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError(
                "TensorRT is not installed. Install NVIDIA TensorRT Python "
                "bindings to build cached engines."
            ) from exc

        if self.model is None:
            raise RuntimeError("Cannot build TensorRT engine without a loaded PyTorch model.")

        onnx_path = _export_tensorrt_onnx_from_model(
            model=self.model,
            onnx_path=self.onnx_path,
            width=self._engine_width,
            height=self._engine_height,
            dtype=self._dtype,
            device=self.device,
            precision=self.precision,
            flat_model=getattr(self, "_is_flat_model", False),
            force=False,
        )
        self._build_engine_from_onnx(onnx_path, trt)

    def _build_engine_from_onnx(self, onnx_path: str, trt) -> None:
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        workspace_gb = float(os.environ.get("HDRTVNET_TRT_WORKSPACE_GB", "4"))
        workspace_bytes = int(max(1.0, workspace_gb) * (1024 ** 3))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes

        if self.precision != "fp32" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if str(self.precision).startswith("int8") and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        os.makedirs(os.path.dirname(self.engine_path), exist_ok=True)
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("TensorRT engine build returned no serialized engine.")
            pathlib.Path(self.engine_path).write_bytes(bytes(serialized))
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("TensorRT engine build failed.")
            pathlib.Path(self.engine_path).write_bytes(bytes(engine.serialize()))

    def _load_engine(self) -> None:
        try:
            import tensorrt as trt
        except Exception as exc:
            raise RuntimeError(
                "TensorRT is not installed. Install NVIDIA TensorRT Python bindings."
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        self._trt_runtime = trt.Runtime(logger)
        data = pathlib.Path(self.engine_path).read_bytes()
        self._trt_engine = self._trt_runtime.deserialize_cuda_engine(data)
        if self._trt_engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
        self._trt_context = self._trt_engine.create_execution_context()
        if self._trt_context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self._discover_bindings()

    def _discover_bindings(self) -> None:
        engine = self._trt_engine
        try:
            import tensorrt as trt
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._trt_input_names.append(name)
                else:
                    self._trt_output_names.append(name)
        except AttributeError:
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                if engine.binding_is_input(i):
                    self._trt_input_names.append(name)
                else:
                    self._trt_output_names.append(name)
        if len(self._trt_input_names) < 2 or not self._trt_output_names:
            raise RuntimeError("Unexpected TensorRT bindings; expected input, cond, output.")

    def _tensor_dtype_for_output(self, output_name: str):
        try:
            import tensorrt as trt
            dtype = self._trt_engine.get_tensor_dtype(output_name)
            if dtype == trt.float16:
                return torch.float16
            if dtype == trt.float32:
                return torch.float32
            if dtype == trt.int8:
                return torch.int8
        except Exception:
            pass
        return self._dtype

    def infer(self, input_cond):
        tensor, cond = input_cond
        tensor = tensor.contiguous()
        cond = cond.contiguous()
        ctx = self._trt_context
        engine = self._trt_engine
        stream = torch.cuda.current_stream().cuda_stream

        input_name = self._trt_input_names[0]
        cond_name = self._trt_input_names[1]
        output_name = self._trt_output_names[0]

        if hasattr(ctx, "set_input_shape"):
            ctx.set_input_shape(input_name, tuple(tensor.shape))
            ctx.set_input_shape(cond_name, tuple(cond.shape))
            out_shape = tuple(int(v) for v in ctx.get_tensor_shape(output_name))
            for name, ptr in (
                (input_name, tensor.data_ptr()),
                (cond_name, cond.data_ptr()),
            ):
                ctx.set_tensor_address(name, int(ptr))
            if self._trt_output is None or self._trt_output_shape != out_shape:
                self._trt_output = torch.empty(
                    out_shape,
                    dtype=self._tensor_dtype_for_output(output_name),
                    device=self.device,
                )
                self._trt_output_shape = out_shape
            ctx.set_tensor_address(output_name, int(self._trt_output.data_ptr()))
            ok = ctx.execute_async_v3(stream_handle=stream)
        else:
            bindings = [0] * engine.num_bindings
            for name, value in ((input_name, tensor), (cond_name, cond)):
                idx = engine.get_binding_index(name)
                try:
                    ctx.set_binding_shape(idx, tuple(value.shape))
                except Exception:
                    pass
                bindings[idx] = int(value.data_ptr())
            out_idx = engine.get_binding_index(output_name)
            out_shape = tuple(int(v) for v in ctx.get_binding_shape(out_idx))
            if self._trt_output is None or self._trt_output_shape != out_shape:
                self._trt_output = torch.empty(
                    out_shape,
                    dtype=self._tensor_dtype_for_output(output_name),
                    device=self.device,
                )
                self._trt_output_shape = out_shape
            bindings[out_idx] = int(self._trt_output.data_ptr())
            ok = ctx.execute_async_v2(bindings=bindings, stream_handle=stream)

        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        return self._trt_output
